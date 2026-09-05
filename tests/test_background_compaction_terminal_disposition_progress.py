"""Terminal-range retry and forward-progress contracts.

These cases keep time deterministic so durable suppression is tested without
sleeping.  A terminal provider failure is a cooldown, not a permanent verdict
on the source.  A genuinely new row may extend the range immediately, while a
large pre-existing backlog falls back to one exact lossless prefix so the
frontier can advance without an oversized provider request.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


class Clock:
    def __init__(self, value: float = 1_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _config(db_path, **overrides):
    values = {
        "database_path": str(db_path),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "background_compaction_global_limit": 2,
        "background_compaction_max_passes": 4,
        "background_compaction_job_timeout_seconds": 10,
        "background_compaction_lease_seconds": 30,
        "background_compaction_backlog_tokens": 1,
        "background_compaction_prepare_ratio": 0.55,
        "background_compaction_min_provider_source_tokens": 1,
        "fresh_tail_count": 0,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 100,
        "cache_friendly_condensation_enabled": False,
        "summary_circuit_breaker_cooldown_seconds": 300,
    }
    values.update(overrides)
    return LCMConfig(**values)


def _seed(db_path, count: int) -> list[int]:
    messages = MessageStore(str(db_path))
    try:
        ids = messages.append_batch(
            "s",
            [
                {"role": "user", "content": f"durable source row {index}"}
                for index in range(count)
            ],
            token_estimates=[100] * count,
            conversation_id="c",
        )
    finally:
        messages.close()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE messages SET timestamp=900, ingested_at=900")
    SummaryDAG(str(db_path)).close()
    lifecycle = LifecycleStateStore(str(db_path))
    try:
        lifecycle.bind_session("s", conversation_id="c")
    finally:
        lifecycle.close()
    return ids


def _append_new(db_path, *, ingested_at: float) -> int:
    messages = MessageStore(str(db_path))
    try:
        store_id = messages.append_batch(
            "s",
            [{"role": "assistant", "content": "genuinely new eligible row"}],
            token_estimates=[100],
            conversation_id="c",
        )[0]
    finally:
        messages.close()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE messages SET timestamp=?, ingested_at=? WHERE store_id=?",
            (100.0, ingested_at, store_id),
        )
    return store_id


def _capture(background, config):
    return background.capture_and_enqueue(
        conversation_id="c",
        session_id="s",
        expected_frontier_store_id=0,
        active_tokens=900_000,
        effective_context_window=1_000_000,
        policy_fingerprint=background_policy_fingerprint(config),
    )


def _exhaust_transient_job(background, clock, job, reason: str):
    for attempt in range(5):
        claimed = background.claim_next_job(owner=f"worker-{attempt}")
        assert claimed is not None and claimed.job_id == job.job_id
        failed = background.mark_retry(
            job.job_id,
            owner=f"worker-{attempt}",
            generation=claimed.lease_generation,
            error=TimeoutError("redacted timeout")
            if reason == "provider_timeout"
            else RuntimeError("redacted provider failure"),
            completion_reason=reason,
        )
        assert failed is not False
        if attempt < 4:
            assert failed.state == "retry"
            clock.advance(300)
    assert failed.state == "failed"
    return job, failed


@pytest.mark.parametrize("reason", ["provider_timeout", "provider_error"])
def test_max_attempt_transient_disposition_new_row_bypasses_cooldown(
    tmp_path,
    reason,
):
    db_path = tmp_path / f"{reason}-new-material.db"
    source_ids = _seed(db_path, 4)
    clock = Clock()
    config = _config(db_path)
    background = BackgroundCompactionStore(str(db_path), config=config, now=clock)
    try:
        initial = _capture(background, config)
        assert initial is not None
        assert initial.source_store_ids == tuple(source_ids)
        _job, failed = _exhaust_transient_job(background, clock, initial, reason)
        disposition = background.connection.execute(
            """
            SELECT reason_code, first_seen_at, resolved_at
            FROM lcm_compaction_range_dispositions
            WHERE source_identity_hash=?
            """,
            (failed.source_identity_hash,),
        ).fetchone()
        assert tuple(disposition) == (reason, clock(), None)

        job_count = background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs"
        ).fetchone()[0]
        assert _capture(background, config) is None
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs"
        ).fetchone()[0] == job_count

        # Arrival is new even though the source's historical event timestamp is
        # older than both the disposition and its prefix.
        new_id = _append_new(db_path, ingested_at=clock() + 1)
        replacement = _capture(background, config)
        assert replacement is not None
        assert replacement.job_id != failed.job_id
        assert replacement.source_store_ids == tuple([*source_ids, new_id])
        chunks = background.source_chunks(replacement.job_id)
        assert 1 <= len(chunks) <= config.background_compaction_max_passes
        assert tuple(
            store_id for chunk in chunks for store_id in chunk.source_store_ids
        ) == replacement.source_store_ids

        prepared = background.run_next_job(
            owner="replacement-worker",
            summarizer=lambda **_kwargs: ("bounded replacement summary", 1),
        )
        assert prepared is not None and prepared.state == "prepared"
        assert len(background.get_prepared_nodes(prepared.job_id)) <= 4
    finally:
        background.close()


@pytest.mark.parametrize("reason", ["provider_timeout", "provider_error"])
def test_unchanged_transient_disposition_retries_only_after_durable_cooldown(
    tmp_path,
    reason,
):
    db_path = tmp_path / f"{reason}-cooldown.db"
    source_ids = _seed(db_path, 4)
    clock = Clock()
    config = _config(db_path)
    background = BackgroundCompactionStore(str(db_path), config=config, now=clock)
    try:
        initial = _capture(background, config)
        assert initial is not None
        _job, failed = _exhaust_transient_job(
            background,
            clock,
            initial,
            reason,
        )
        # The retry gate is a database contract, not process-local memory.
        background.close()
        background = BackgroundCompactionStore(
            str(db_path),
            config=config,
            now=clock,
        )
        job_count = background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs"
        ).fetchone()[0]

        clock.advance(config.summary_circuit_breaker_cooldown_seconds - 1)
        assert _capture(background, config) is None
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs"
        ).fetchone()[0] == job_count

        clock.advance(1)
        retry = _capture(background, config)
        assert retry is not None
        assert retry.job_id != failed.job_id
        assert retry.source_store_ids == tuple(source_ids)
        disposition = background.connection.execute(
            """
            SELECT resolved_at FROM lcm_compaction_range_dispositions
            WHERE source_identity_hash=?
            """,
            (failed.source_identity_hash,),
        ).fetchone()
        assert disposition[0] == clock()
    finally:
        background.close()


def test_far_new_row_projects_exact_terminal_prefix_then_publishes_without_gaps(
    tmp_path,
):
    db_path = tmp_path / "far-new-row.db"
    # Thirteen full chunks put the future row beyond 2 * max_passes while still
    # remaining inside the bounded foreground capture window.
    source_ids = _seed(db_path, 13)
    clock = Clock()
    config = _config(db_path)
    background = BackgroundCompactionStore(str(db_path), config=config, now=clock)
    try:
        initial = _capture(background, config)
        assert initial is not None
        exact_prefix = tuple(source_ids[: config.background_compaction_max_passes])
        assert initial.source_store_ids == exact_prefix
        failed = background.run_next_job(
            owner="terminal-worker",
            summarizer=lambda **_kwargs: ("still not compressing", 3),
        )
        assert failed is not None and failed.state == "failed"
        assert failed.completion_reason == "non_compressing_output"

        # Rows that predate the disposition do not by themselves erase exact
        # terminal suppression.
        assert _capture(background, config) is None

        new_id = _append_new(db_path, ingested_at=clock() + 1)
        progressed = _capture(background, config)
        assert progressed is not None
        assert progressed.job_id != failed.job_id
        assert progressed.job_kind == "lossless_raw_projection"
        assert progressed.state == "prepared"
        assert progressed.source_store_ids == exact_prefix
        prepared_nodes = background.get_prepared_nodes(progressed.job_id)
        assert len(prepared_nodes) == 1
        assert prepared_nodes[0].source_store_ids == exact_prefix
        assert prepared_nodes[0].creation_provider == "local"

        lease = background.acquire_publication_lease(
            progressed.job_id,
            owner="publisher",
        )
        assert lease is not None
        published = background.publish_prepared(
            progressed.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
            observed_active_tokens=900_000,
            effective_context_window=1_000_000,
            publication_trigger_tokens=900_000,
            emergency_trigger_tokens=950_000,
        )
        assert published.promoted is True
        assert published.captured_watermark_store_id == exact_prefix[-1]

        covered = background.connection.execute(
            """
            SELECT store_id FROM lcm_compaction_node_sources
            WHERE session_id='s' ORDER BY store_id
            """
        ).fetchall()
        assert [row[0] for row in covered] == list(exact_prefix)
        assert background.connection.execute(
            "SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id='c'"
        ).fetchone()[0] == exact_prefix[-1]
        raw = background.connection.execute(
            "SELECT store_id FROM messages WHERE session_id='s' ORDER BY store_id"
        ).fetchall()
        assert [row[0] for row in raw] == [*source_ids, new_id]
    finally:
        background.close()


def test_row_appended_after_capture_before_terminal_failure_extends_range(
    tmp_path,
):
    db_path = tmp_path / "append-between-capture-and-failure.db"
    source_ids = _seed(db_path, 4)
    clock = Clock()
    config = _config(db_path)
    background = BackgroundCompactionStore(str(db_path), config=config, now=clock)
    try:
        initial = _capture(background, config)
        assert initial is not None
        assert initial.source_store_ids == tuple(source_ids)

        # This row was not in the durable capture, but its arrival predates the
        # provider's terminal result. Failure time must not become the capture
        # boundary or the row will remain hidden behind the exact disposition.
        new_id = _append_new(db_path, ingested_at=clock() + 1)
        clock.advance(2)
        failed = background.run_next_job(
            owner="terminal-worker",
            summarizer=lambda **_kwargs: ("not compressing", 3),
        )
        assert failed is not None and failed.state == "failed"
        assert failed.completion_reason == "non_compressing_output"
        disposition = background.connection.execute(
            """
            SELECT capture_started_at, first_seen_at
            FROM lcm_compaction_range_dispositions
            WHERE source_identity_hash=?
            """,
            (failed.source_identity_hash,),
        ).fetchone()
        assert tuple(disposition) == (initial.created_at, clock())

        replacement = _capture(background, config)
        assert replacement is not None
        assert replacement.job_id != failed.job_id
        assert replacement.source_store_ids == tuple([*source_ids, new_id])
        chunks = background.source_chunks(replacement.job_id)
        assert 1 <= len(chunks) <= config.background_compaction_max_passes
        assert tuple(
            store_id for chunk in chunks for store_id in chunk.source_store_ids
        ) == replacement.source_store_ids
    finally:
        background.close()
