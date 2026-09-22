"""Strict contracts for queue/frontier races before provider execution.

These tests use independent SQLite connections plus explicit events.  They do
not rely on scheduling sleeps, so a failure identifies a missing state
revalidation rather than a timing accident.
"""

from __future__ import annotations

import sqlite3
import threading

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


def _config(db_path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=1,
        background_compaction_global_limit=2,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=120,
        background_compaction_lease_seconds=300,
        background_compaction_backlog_tokens=1,
        background_compaction_min_provider_source_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=2,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=1_000,
        context_reserve_tokens=100_000,
    )


def _initialize(db_path, *, count: int = 8) -> list[int]:
    store = MessageStore(db_path)
    ids = store.append_batch(
        "s",
        [{"role": "user", "content": f"raw-{index}"} for index in range(count)],
        token_estimates=[10] * count,
        conversation_id="c",
    )
    store.close()
    dag = SummaryDAG(db_path)
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session("s", conversation_id="c")
    lifecycle.close()
    return ids


def _append(db_path, *, start: int, count: int) -> list[int]:
    """Append through an independent foreground-store connection."""
    store = MessageStore(db_path)
    try:
        return store.append_batch(
            "s",
            [
                {"role": "user", "content": f"newer-{index}"}
                for index in range(start, start + count)
            ],
            token_estimates=[10] * count,
            conversation_id="c",
        )
    finally:
        store.close()


def test_capture_coalesces_new_rows_into_retry_without_clearing_backoff(tmp_path):
    """A same-policy retry is extended, not returned stale or made runnable."""
    db_path = tmp_path / "retry-coalesce.db"
    clock = Clock()
    initial_ids = _initialize(db_path)
    config = _config(db_path)
    policy = background_policy_fingerprint(config)
    queue = BackgroundCompactionStore(db_path, config=config, now=clock)
    try:
        first = queue.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )
        assert first is not None
        assert first.source_store_ids == tuple(initial_ids[:-2])

        claimed = queue.claim_next_job(owner="worker-a", policy_fingerprint=policy)
        assert claimed is not None
        retry = queue.mark_retry(
            claimed.job_id,
            owner="worker-a",
            generation=claimed.lease_generation,
            error=RuntimeError("temporary provider failure"),
        )
        assert retry
        retry_before = queue.get_job(claimed.job_id)
        assert retry_before is not None
        assert retry_before.state == "retry"
        assert retry_before.next_retry_at is not None

        newer_ids = _append(db_path, start=len(initial_ids), count=4)
        coalesced = queue.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )

        assert coalesced is not None
        assert coalesced.job_id == retry_before.job_id
        assert coalesced.source_store_ids == tuple(initial_ids + newer_ids[:-2])
        assert coalesced.captured_watermark_store_id == newer_ids[-3]
        assert coalesced.state == "retry"
        assert coalesced.next_retry_at == retry_before.next_retry_at
        assert coalesced.last_error == retry_before.last_error
        assert coalesced.attempt_count == retry_before.attempt_count
        assert queue.claim_next_job(owner="worker-too-early", now=clock.value) is None
    finally:
        queue.close()


def test_frontier_advance_between_capture_and_enqueue_returns_no_stale_job(
    tmp_path, monkeypatch
):
    """A foreground frontier CAS race is a no-op, not a foreground exception."""
    db_path = tmp_path / "capture-frontier-race.db"
    ids = _initialize(db_path)
    config = _config(db_path)
    policy = background_policy_fingerprint(config)
    queue = BackgroundCompactionStore(db_path, config=config)
    original_enqueue = queue.enqueue_or_coalesce
    advanced = threading.Event()

    def advance_then_enqueue(**kwargs):
        # This independent writer runs after capture's source read has
        # completed but before enqueue_or_coalesce begins its write transaction.
        lifecycle = LifecycleStateStore(db_path)
        try:
            lifecycle.advance_frontier("c", "s", ids[0])
        finally:
            lifecycle.close()
        advanced.set()
        return original_enqueue(**kwargs)

    monkeypatch.setattr(queue, "enqueue_or_coalesce", advance_then_enqueue)
    try:
        result = queue.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )
        assert advanced.is_set()
        assert result is None

        check = sqlite3.connect(str(db_path))
        try:
            active_jobs = check.execute(
                """
                SELECT COUNT(*) FROM lcm_compaction_jobs
                WHERE conversation_id='c'
                  AND state IN ('queued','running','prepared','retry')
                """
            ).fetchone()[0]
            frontier = check.execute(
                """
                SELECT current_frontier_store_id FROM lcm_lifecycle_state
                WHERE conversation_id='c'
                """
            ).fetchone()[0]
        finally:
            check.close()
        assert active_jobs == 0
        assert frontier == ids[0]
    finally:
        queue.close()


def test_superseded_claim_is_revalidated_before_any_provider_call(
    tmp_path, monkeypatch
):
    """Losing the claimed generation before I/O fences the provider entirely."""
    db_path = tmp_path / "supersede-before-provider.db"
    _initialize(db_path)
    config = _config(db_path)
    policy = background_policy_fingerprint(config)
    queue = BackgroundCompactionStore(db_path, config=config)
    job = queue.capture_and_enqueue(
        conversation_id="c",
        session_id="s",
        expected_frontier_store_id=0,
        active_tokens=1,
        effective_context_window=372_000,
        policy_fingerprint=policy,
    )
    assert job is not None

    claim_reached = threading.Event()
    allow_worker_to_continue = threading.Event()
    provider_calls: list[dict] = []
    worker_result: list[object] = []
    worker_errors: list[BaseException] = []
    original_source_chunks = queue.source_chunks

    def gated_source_chunks(job_id):
        chunks = original_source_chunks(job_id)
        claim_reached.set()
        assert allow_worker_to_continue.wait(timeout=5)
        return chunks

    def summarizer(**kwargs):
        provider_calls.append(kwargs)
        return "must never be started", 0

    monkeypatch.setattr(queue, "source_chunks", gated_source_chunks)

    def run_worker():
        try:
            worker_result.append(
                queue.run_next_job(owner="worker-a", summarizer=summarizer)
            )
        except BaseException as exc:  # Preserve thread failures for the assertion thread.
            worker_errors.append(exc)

    worker = threading.Thread(target=run_worker, name="queue-race-worker")
    worker.start()
    try:
        assert claim_reached.wait(timeout=5)

        # Simulate an independent foreground publication superseding the
        # claimed generation while the worker is between claim and provider.
        foreground = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            foreground.execute("BEGIN IMMEDIATE")
            running = foreground.execute(
                """
                SELECT state, lease_generation FROM lcm_compaction_jobs
                WHERE job_id=?
                """,
                (job.job_id,),
            ).fetchone()
            assert running is not None and running[0] == "running"
            foreground.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='superseded', completion_reason='foreground_race',
                    lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    lease_generation=lease_generation+1
                WHERE job_id=? AND state='running'
                """,
                (job.job_id,),
            )
            foreground.execute(
                "DELETE FROM lcm_compaction_provider_leases WHERE job_id=?",
                (job.job_id,),
            )
            foreground.execute("COMMIT")
        finally:
            foreground.close()
        allow_worker_to_continue.set()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert worker_errors == []
        assert provider_calls == []
        assert len(worker_result) == 1
        result = worker_result[0]
        assert result is not None
        assert result.state == "superseded"
        assert queue.get_prepared_nodes(job.job_id) == []
    finally:
        allow_worker_to_continue.set()
        worker.join(timeout=5)
        queue.close()
