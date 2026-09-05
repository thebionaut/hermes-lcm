"""Regression coverage for the September 2026 background-performance sprint."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


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
        "fresh_tail_count": 0,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 120_000,
        "summary_model": "primary/model-a",
        "summary_fallback_models": ["fallback/model-b"],
        "context_reserve_tokens": 100_000,
    }
    values.update(overrides)
    return LCMConfig(**values)


def _initialize(db_path, messages, token_estimates, *, session="s", conversation="c"):
    store = MessageStore(str(db_path))
    try:
        source_ids = store.append_batch(
            session,
            messages,
            token_estimates=token_estimates,
            conversation_id=conversation,
        )
    finally:
        store.close()
    dag = SummaryDAG(str(db_path))
    dag.close()
    lifecycle = LifecycleStateStore(str(db_path))
    try:
        lifecycle.bind_session(session, conversation_id=conversation)
    finally:
        lifecycle.close()
    return source_ids


def _capture(background, config, *, session="s", conversation="c", frontier=0):
    return background.capture_and_enqueue(
        conversation_id=conversation,
        session_id=session,
        expected_frontier_store_id=frontier,
        active_tokens=900_000,
        effective_context_window=1_000_000,
        policy_fingerprint=background_policy_fingerprint(config),
    )


def test_two_hundred_same_session_finalize_rebinds_preserve_job_and_fence(tmp_path):
    db_path = tmp_path / "stable-fence.db"
    config = _config(db_path)
    source_ids = _initialize(
        db_path,
        [{"role": "user", "content": "durable backlog"}],
        [32_000],
    )
    lifecycle = LifecycleStateStore(str(db_path))
    background = BackgroundCompactionStore(str(db_path), config=config)
    try:
        initial = lifecycle.get_by_conversation("c")
        assert initial is not None
        generation = background.activate_session(
            conversation_id="c",
            session_id="s",
            expected_lifecycle_bound_at=initial.current_bound_at,
            expected_lifecycle_reset_at=initial.last_reset_at,
        )
        job = background.enqueue_or_coalesce(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            captured_watermark_store_id=source_ids[-1],
            source_store_ids=source_ids,
            source_token_count=32_000,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert job is not None

        for _ in range(200):
            finalized = lifecycle.finalize_session("c", "s", frontier_store_id=0)
            assert finalized is not None and finalized.current_session_id is None
            rebound = lifecycle.bind_session("s", conversation_id="c")
            assert rebound.current_frontier_store_id == 0
            assert background.activate_session(
                conversation_id="c",
                session_id="s",
                expected_lifecycle_bound_at=rebound.current_bound_at,
                expected_lifecycle_reset_at=rebound.last_reset_at,
            ) == generation

        persisted = background.get_job(job.job_id)
        assert persisted is not None and persisted.state == "queued"
        assert persisted.session_fence_generation == generation
        with sqlite3.connect(db_path) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM lcm_compaction_jobs WHERE conversation_id='c'"
            ).fetchone()[0] == 1
    finally:
        background.close()
        lifecycle.close()


def test_stale_finalize_cannot_rewind_a_frontier_advanced_by_another_process(
    tmp_path,
):
    db_path = tmp_path / "finalize-frontier-race.db"
    source_ids = _initialize(
        db_path,
        [{"role": "user", "content": "already published source"}],
        [1_000],
    )
    lifecycle = LifecycleStateStore(str(db_path))
    try:
        lifecycle.advance_frontier("c", "s", source_ids[0])

        # A stale process still holding frontier zero finalizes the ordinary
        # turn after its sibling committed the publication frontier.
        finalized = lifecycle.finalize_session(
            "c",
            "s",
            frontier_store_id=0,
        )
        assert finalized is not None
        assert finalized.last_finalized_frontier_store_id == source_ids[0]

        rebound = lifecycle.bind_session("s", conversation_id="c")
        assert rebound.current_frontier_store_id == source_ids[0]
    finally:
        lifecycle.close()


def test_finalize_captures_a_sibling_frontier_advance_after_its_read(
    tmp_path,
    monkeypatch,
):
    """Exercise the real two-connection read/update interleaving."""
    db_path = tmp_path / "finalize-frontier-interleaving.db"
    finalizer = LifecycleStateStore(str(db_path))
    publisher = LifecycleStateStore(str(db_path))
    finalizer.bind_session("s", conversation_id="c")
    finalizer.advance_frontier("c", "s", 7)

    state_read = threading.Event()
    allow_finalize = threading.Event()
    original_get = finalizer.get_by_conversation

    def pause_after_first_read(conversation_id):
        state = original_get(conversation_id)
        if not state_read.is_set():
            state_read.set()
            assert allow_finalize.wait(timeout=2)
        return state

    monkeypatch.setattr(finalizer, "get_by_conversation", pause_after_first_read)
    errors: list[BaseException] = []

    def finalize() -> None:
        try:
            finalizer.finalize_session("c", "s", frontier_store_id=0)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread = threading.Thread(target=finalize)
    try:
        thread.start()
        assert state_read.wait(timeout=2)
        advanced = publisher.advance_frontier("c", "s", 11)
        assert advanced is not None
        assert advanced.current_frontier_store_id == 11
        allow_finalize.set()
        thread.join(timeout=2)

        assert not thread.is_alive()
        assert errors == []
        persisted = publisher.get_by_conversation("c")
        assert persisted is not None
        assert persisted.current_session_id is None
        assert persisted.last_finalized_frontier_store_id == 11
    finally:
        allow_finalize.set()
        thread.join(timeout=2)
        publisher.close()
        finalizer.close()


@pytest.mark.parametrize(
    ("content", "source_tokens"),
    [
        (
            "[Externalized LCM ingest payload: kind=media; chars=3800000; "
            "bytes=3800000; ref=fixture://media]",
            44,
        ),
        ("isolated user correction between canonical ranges", 78),
    ],
)
def test_tiny_gap_is_prepared_losslessly_without_provider_and_publishes_once(
    tmp_path,
    content,
    source_tokens,
):
    db_path = tmp_path / f"tiny-gap-{source_tokens}.db"
    config = _config(db_path)
    source_ids = _initialize(
        db_path,
        [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "already canonical island"},
        ],
        [source_tokens, 50],
    )
    dag = SummaryDAG(str(db_path))
    dag.add_node(
        SummaryNode(
            session_id="s",
            depth=0,
            summary="existing canonical island",
            source_ids=[source_ids[1]],
            source_type="messages",
            source_token_count=50,
            token_count=4,
        )
    )
    dag.close()

    background = BackgroundCompactionStore(str(db_path), config=config)
    provider_calls = []
    try:
        job = _capture(background, config)
        assert job is not None
        assert job.source_store_ids == (source_ids[0],)
        assert job.state == "prepared"
        assert job.completion_reason == "insufficient_compressible_source"

        assert background.run_next_job(
            owner="must-not-run",
            summarizer=lambda **kwargs: provider_calls.append(kwargs) or ("bad", 1),
        ) is None
        assert provider_calls == []

        prepared = background.get_prepared_nodes(job.job_id)
        assert len(prepared) == 1
        assert prepared[0].source_store_ids == (source_ids[0],)
        assert prepared[0].creation_provider == "local"
        assert prepared[0].creation_model == "deterministic/lossless-raw-v1"
        assert content in prepared[0].summary

        lease = background.acquire_publication_lease(job.job_id, owner="publisher")
        assert lease is not None
        published = background.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
            observed_active_tokens=900_000,
            effective_context_window=1_000_000,
            publication_trigger_tokens=900_000,
            emergency_trigger_tokens=950_000,
        )
        assert published.promoted is True

        dag = SummaryDAG(str(db_path))
        try:
            assert dag.source_message_ids(published.node_ids[0], limit=10) == [
                source_ids[0]
            ]
        finally:
            dag.close()
        with sqlite3.connect(db_path) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM lcm_compaction_source_coverage "
                "WHERE session_id='s' AND store_id=?",
                (source_ids[0],),
            ).fetchone()[0] == 1
    finally:
        background.close()


def test_lossless_projection_fails_closed_when_source_changes_before_prepare(
    tmp_path,
    monkeypatch,
):
    """A stale exact projection cannot be paired with a newer raw identity."""
    db_path = tmp_path / "lossless-projection-source-race.db"
    config = _config(db_path)
    source_ids = _initialize(
        db_path,
        [{"role": "user", "content": "original isolated row"}],
        [44],
    )
    background = BackgroundCompactionStore(str(db_path), config=config)
    original_prepare = background._prepare_lossless_projection

    def mutate_then_prepare(**kwargs):
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE messages SET content=? WHERE store_id=?",
                ("concurrently changed row", source_ids[0]),
            )
        return original_prepare(**kwargs)

    monkeypatch.setattr(
        background,
        "_prepare_lossless_projection",
        mutate_then_prepare,
    )
    try:
        assert _capture(background, config) is None
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_prepared_nodes"
        ).fetchone()[0] == 0
    finally:
        background.close()


def test_noncompressing_range_is_terminally_suppressed_across_restart(tmp_path):
    db_path = tmp_path / "durable-disposition.db"
    config = _config(
        db_path,
        background_compaction_max_passes=4,
        background_compaction_min_provider_source_tokens=1,
        dynamic_leaf_chunk_enabled=False,
        cache_friendly_condensation_enabled=False,
        leaf_chunk_tokens=100,
    )
    source_ids = _initialize(
        db_path,
        [
            {"role": "user", "content": f"large but incompressible fixture {index}"}
            for index in range(4)
        ],
        [100] * 4,
    )
    first = BackgroundCompactionStore(str(db_path), config=config)
    try:
        job = _capture(first, config)
        assert job is not None and job.source_store_ids == tuple(source_ids)
        assert job.source_token_count == (
            config.background_compaction_max_passes * config.leaf_chunk_tokens
        )
        rejected = first.run_next_job(
            owner="worker-a",
            summarizer=lambda **_kwargs: ("deterministic fallback", 3),
        )
        assert rejected is not None and rejected.state == "failed"
        assert rejected.completion_reason == "non_compressing_output"
    finally:
        first.close()

    second = BackgroundCompactionStore(str(db_path), config=config)
    try:
        assert _capture(second, config) is None
        with sqlite3.connect(db_path) as conn:
            disposition = conn.execute(
                "SELECT reason_code, source_identity_hash "
                "FROM lcm_compaction_range_dispositions"
            ).fetchone()
            assert disposition is not None
            assert disposition[0] == "non_compressing_output"
            assert disposition[1]

        store = MessageStore(str(db_path))
        try:
            appended_ids = store.append_batch(
                "s",
                [{"role": "assistant", "content": "new compactable material"}],
                token_estimates=[100],
                conversation_id="c",
            )
        finally:
            store.close()
        replacement = _capture(second, config)
        assert replacement is not None
        assert replacement.job_id != job.job_id
        assert replacement.captured_watermark_store_id > job.captured_watermark_store_id
        expected_ids = tuple([*source_ids, *appended_ids])
        assert replacement.source_store_ids == expected_ids
        chunks = second.source_chunks(replacement.job_id)
        assert len(chunks) <= config.background_compaction_max_passes
        assert tuple(
            store_id for chunk in chunks for store_id in chunk.source_store_ids
        ) == expected_ids
        prepared = second.run_next_job(
            owner="replacement-worker",
            summarizer=lambda **_kwargs: ("prepared changed-source summary", 1),
        )
        assert prepared is not None
        assert prepared.job_id == replacement.job_id
        assert prepared.state == "prepared"
        assert prepared.completion_reason == "prepared"
        prepared_nodes = second.get_prepared_nodes(replacement.job_id)
        assert len(prepared_nodes) <= config.background_compaction_max_passes
        assert tuple(
            store_id
            for node in prepared_nodes
            for store_id in node.source_store_ids
        ) == expected_ids
        assert second.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_node_sources "
            f"WHERE session_id='s' AND store_id IN ({','.join('?' for _ in expected_ids)})",
            expected_ids,
        ).fetchone()[0] == 0
    finally:
        second.close()


def test_fallback_route_attribution_and_publication_trigger_are_durable(tmp_path, monkeypatch):
    db_path = tmp_path / "route-telemetry.db"
    config = _config(db_path)
    _initialize(
        db_path,
        [{"role": "user", "content": "representative source " * 200}],
        [32_000],
    )

    def fake_llm(_prompt, _max_tokens, model="", timeout=None, route_info=None):
        del timeout
        if route_info is not None:
            provider, _, model_name = model.partition("/")
            route_info.update({"provider": provider, "model": model_name or model})
        if model == "primary/model-a":
            return None
        return "short fallback summary"

    monkeypatch.setattr("hermes_lcm.escalation._call_llm_for_summary", fake_llm)
    background = BackgroundCompactionStore(str(db_path), config=config)
    try:
        job = _capture(background, config)
        assert job is not None
        prepared_job = background.run_next_job(owner="route-worker")
        assert prepared_job is not None and prepared_job.state == "prepared"
        prepared = background.get_prepared_nodes(job.job_id)[0]
        assert prepared.creation_provider == "fallback"
        assert prepared.creation_model == "model-b"
        assert prepared.route_latency_ms >= 0
        assert prepared.attempt_number == 1

        lease = background.acquire_publication_lease(job.job_id, owner="publisher")
        assert lease is not None
        result = background.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
            observed_active_tokens=900_321,
            effective_context_window=1_000_000,
            publication_trigger_tokens=900_000,
            emergency_trigger_tokens=950_000,
        )
        assert result.promoted is True

        status = background.status(
            conversation_id="c",
            publication_trigger_tokens=900_000,
            emergency_trigger_tokens=950_000,
        )
        telemetry = status["last_publication_telemetry"]
        assert telemetry["job_id"] == job.job_id
        assert telemetry["observed_active_tokens"] == 900_321
        assert telemetry["effective_context_window"] == 1_000_000
        assert telemetry["publication_trigger_tokens"] == 900_000
        assert telemetry["emergency_trigger_tokens"] == 950_000
        assert telemetry["publication_fence_generation"] == job.session_fence_generation
        assert telemetry["prepared_to_published_seconds"] >= 0
        assert telemetry["range_identity"] == job.source_identity_hash
        assert telemetry["provider"] == "fallback"
        assert telemetry["model"] == "model-b"
    finally:
        background.close()


def test_engine_calculates_and_persists_fixed_reserve_publication_telemetry(
    tmp_path,
):
    db_path = tmp_path / "engine-publication-telemetry.db"
    config = _config(
        db_path,
        background_compaction_min_provider_source_tokens=1,
        leaf_chunk_tokens=120_000,
    )
    engine = LCMEngine(config)
    messages = [
        {"role": "user", "content": "telemetry source " * 1_000},
        {"role": "assistant", "content": "durable answer " * 1_000},
    ]
    try:
        engine.update_model(
            model="glm-test",
            provider="zai",
            context_length=1_000_000,
        )
        engine.on_session_start(
            "s",
            conversation_id="c",
            context_length=1_000_000,
        )
        engine.ingest(messages)
        prepared = engine.run_background_compaction_once(
            summarizer=lambda **_kwargs: ("short durable summary", 1)
        )
        assert prepared is not None and prepared.state == "prepared"

        published = engine.promote_prepared_compaction(
            prepared.job_id,
            messages,
            current_tokens=900_000,
        )
        assert published.promoted is True
        telemetry = engine.get_background_compaction_status()[
            "last_publication_telemetry"
        ]
        assert telemetry["observed_active_tokens"] == 900_000
        assert telemetry["effective_context_window"] == 1_000_000
        assert telemetry["publication_trigger_tokens"] == 900_000
        assert telemetry["trigger_policy"] == "fixed_reserve"
    finally:
        engine.shutdown()


def test_reconcile_lock_wait_uses_same_foreground_deadline(tmp_path, monkeypatch):
    config = _config(tmp_path / "deadline.db", reconcile_budget_seconds=45.0)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    engine._ingest_cursor_needs_reconcile = True
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_reconcile_lock():
        with engine._reconcile_lock:
            lock_held.set()
            release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_reconcile_lock)
    holder.start()
    assert lock_held.wait(timeout=1)
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    result_box = {}

    def run_compression():
        started = time.monotonic()
        try:
            result_box["messages"] = engine.compress(
                [{"role": "user", "content": "preserve me"}],
                current_tokens=360_000,
            )
        except BaseException as exc:  # assertion inspects public fail-safe below
            result_box["error"] = exc
        result_box["elapsed"] = time.monotonic() - started

    worker = threading.Thread(target=run_compression)
    worker.start()
    try:
        worker.join(timeout=0.25)
        assert not worker.is_alive(), "foreground compression ignored its wall deadline"
        assert "error" not in result_box
        assert result_box["messages"] == [
            {"role": "user", "content": "preserve me"}
        ]
        assert result_box["elapsed"] < 0.20
        assert engine._last_compression_noop_reason == "foreground deadline exhausted during reconciliation"
    finally:
        release_lock.set()
        holder.join(timeout=1)
        worker.join(timeout=1)
        engine.shutdown()
