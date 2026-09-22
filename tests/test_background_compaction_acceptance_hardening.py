"""Focused acceptance hardening for the September compaction sprint.

These regressions close narrow assertion gaps left by the broader sprint
suite.  They intentionally avoid scheduler sleeps and keep provider outcomes
deterministic.
"""

from __future__ import annotations

import time

import pytest

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.compaction import _AUTOMATIC_FOREGROUND_MAX_SECONDS
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def _config(db_path, **overrides):
    values = {
        "database_path": str(db_path),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "background_compaction_global_limit": 2,
        "background_compaction_max_passes": 4,
        "background_compaction_job_timeout_seconds": 5,
        "background_compaction_lease_seconds": 30,
        "background_compaction_backlog_tokens": 1,
        "background_compaction_prepare_ratio": 0.55,
        "background_compaction_min_provider_source_tokens": 1,
        "fresh_tail_count": 0,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 120_000,
        "summary_model": "primary/model-a",
        "summary_fallback_models": ["fallback/model-b"],
        "context_reserve_tokens": 100_000,
    }
    values.update(overrides)
    return LCMConfig(**values)


def _activate_fence(background, lifecycle, *, session_id="session-a"):
    state = lifecycle.bind_session(
        session_id,
        conversation_id="conversation-a",
    )
    generation = background.activate_session(
        conversation_id="conversation-a",
        session_id=session_id,
        expected_lifecycle_bound_at=state.current_bound_at,
        expected_lifecycle_reset_at=state.last_reset_at,
    )
    assert generation is not None
    return state, generation


@pytest.mark.parametrize("invalidation", ["reset", "rollover"])
def test_real_reset_and_rollover_advance_publication_fence_generation(
    tmp_path,
    invalidation,
):
    db_path = tmp_path / f"fence-generation-{invalidation}.db"
    lifecycle = LifecycleStateStore(str(db_path))
    background = BackgroundCompactionStore(str(db_path), config=_config(db_path))
    try:
        _state, original_generation = _activate_fence(background, lifecycle)

        if invalidation == "reset":
            invalidated_generation = background.begin_session_reset(
                conversation_id="conversation-a",
                session_id="session-a",
            )
            expected_session_id = "session-a"
            expected_publishable = 0
        else:
            rolled = lifecycle.record_rollover(
                "conversation-a",
                old_session_id="session-a",
                new_session_id="session-b",
            )
            invalidated_generation = background.activate_session(
                conversation_id="conversation-a",
                session_id="session-b",
                expected_lifecycle_bound_at=rolled.current_bound_at,
                expected_lifecycle_reset_at=rolled.last_reset_at,
            )
            expected_session_id = "session-b"
            expected_publishable = 1

        assert invalidated_generation == original_generation + 1
        fence = background.connection.execute(
            """
            SELECT session_id, generation, publishable
            FROM lcm_compaction_session_fences
            WHERE conversation_id='conversation-a'
            """
        ).fetchone()
        assert fence is not None
        assert tuple(fence) == (
            expected_session_id,
            invalidated_generation,
            expected_publishable,
        )
    finally:
        background.close()
        lifecycle.close()


def test_reset_before_first_fence_stays_closed_across_store_restart(tmp_path):
    db_path = tmp_path / "reset-before-first-fence.db"
    lifecycle = LifecycleStateStore(str(db_path))
    background = BackgroundCompactionStore(str(db_path), config=_config(db_path))
    rebound_background = None
    try:
        state = lifecycle.bind_session(
            "session-a",
            conversation_id="conversation-a",
        )
        reset_generation = background.begin_session_reset(
            conversation_id="conversation-a",
            session_id="session-a",
        )
        assert reset_generation == 1
        background.close()

        rebound_background = BackgroundCompactionStore(
            str(db_path),
            config=_config(db_path),
        )
        reactivated = rebound_background.activate_session(
            conversation_id="conversation-a",
            session_id="session-a",
            expected_lifecycle_bound_at=state.current_bound_at,
            expected_lifecycle_reset_at=state.last_reset_at,
        )

        assert reactivated is None
        fence = rebound_background.connection.execute(
            """
            SELECT generation, publishable, lifecycle_bound_at,
                   lifecycle_reset_at
            FROM lcm_compaction_session_fences
            WHERE conversation_id='conversation-a'
            """
        ).fetchone()
        assert fence is not None
        assert tuple(fence) == (
            reset_generation,
            0,
            state.current_bound_at,
            float(state.last_reset_at or 0.0),
        )
    finally:
        if rebound_background is not None:
            rebound_background.close()
        else:
            background.close()
        lifecycle.close()


def test_automatic_foreground_wall_is_exactly_fifteen_seconds():
    assert _AUTOMATIC_FOREGROUND_MAX_SECONDS == 15.0


def test_repeated_no_progress_calls_stay_fast_without_admission_block(
    tmp_path,
    monkeypatch,
):
    from hermes_lcm.engine import LCMEngine

    config = _config(tmp_path / "no-progress-admission.db")
    engine = LCMEngine(config)
    messages = [
        {"role": "user", "content": "durable backlog " * 200},
        {"role": "assistant", "content": "background work remains pending"},
    ]

    def fail_if_provider_wait_starts(*_args, **_kwargs):
        raise AssertionError("a no-progress retry entered foreground provider work")

    monkeypatch.setattr(
        engine,
        "_call_with_hard_foreground_deadline",
        fail_if_provider_wait_starts,
    )
    try:
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        # Match the host's real call boundary: prompt usage from the completed
        # turn supplies the 810K pressure signal, then preflight ingests and
        # opportunistically enqueues without asking the host to enter its
        # blocking compression path.
        engine.last_prompt_tokens = 810_000
        assert engine.should_compress(prompt_tokens=810_000) is False

        started = time.perf_counter()
        for _ in range(20):
            assert engine.should_compress_preflight(messages) is False
            assert engine.should_compress(prompt_tokens=810_000) is False
        repeated_elapsed = time.perf_counter() - started

        # Repeated host admission remains a fast local ingest/enqueue check; it
        # never enters the foreground provider path or amplifies durable rows.
        assert repeated_elapsed < 2.0
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "background preparation queued"
        assert engine._store.get_session_count("session-a") == len(messages)
        assert engine._background_compaction.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_jobs
            WHERE conversation_id='conversation-a'
            """
        ).fetchone()[0] == 1
    finally:
        engine.shutdown()


def _seed_claimable_job(db_path):
    config = _config(db_path)
    messages = MessageStore(str(db_path))
    try:
        source_ids = messages.append_batch(
            "session-a",
            [{"role": "user", "content": "provider failure source"}],
            token_estimates=[1_000],
            conversation_id="conversation-a",
        )
    finally:
        messages.close()
    dag = SummaryDAG(str(db_path))
    dag.close()
    lifecycle = LifecycleStateStore(str(db_path))
    background = BackgroundCompactionStore(str(db_path), config=config)
    try:
        _state, _generation = _activate_fence(background, lifecycle)
        job = background.enqueue_or_coalesce(
            conversation_id="conversation-a",
            session_id="session-a",
            expected_frontier_store_id=0,
            captured_watermark_store_id=source_ids[-1],
            source_store_ids=source_ids,
            source_token_count=1_000,
            policy_fingerprint=background_policy_fingerprint(config),
            expected_node_count=1,
        )
        assert job is not None
        return background, lifecycle, job
    except BaseException:
        background.close()
        lifecycle.close()
        raise


@pytest.mark.parametrize(
    ("outcome", "expected_state", "expected_reason"),
    [
        ("timeout", "retry", "provider_timeout"),
        ("non_compressing", "failed", "non_compressing_output"),
    ],
)
def test_ordinary_provider_failures_clear_job_and_provider_leases(
    tmp_path,
    outcome,
    expected_state,
    expected_reason,
):
    background, lifecycle, job = _seed_claimable_job(
        tmp_path / f"ordinary-{outcome}-lease-cleanup.db"
    )

    def summarizer(**_kwargs):
        if outcome == "timeout":
            raise TimeoutError("deterministic provider timeout")
        return "not compressing", 3

    try:
        result = background.run_next_job(
            owner=f"{outcome}-worker",
            summarizer=summarizer,
        )
        assert result is not None
        assert result.job_id == job.job_id
        assert result.state == expected_state
        assert result.completion_reason == expected_reason

        lease_row = background.connection.execute(
            """
            SELECT lease_owner, lease_expires_at, heartbeat_at
            FROM lcm_compaction_jobs
            WHERE job_id=?
            """,
            (job.job_id,),
        ).fetchone()
        assert tuple(lease_row) == (None, None, None)
        assert background.connection.execute(
            """
            SELECT COUNT(*)
            FROM lcm_compaction_provider_leases
            WHERE job_id=?
            """,
            (job.job_id,),
        ).fetchone()[0] == 0
    finally:
        background.close()
        lifecycle.close()
