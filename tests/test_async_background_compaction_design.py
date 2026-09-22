"""Executable contract tests for durable asynchronous background compaction.

The cases are intentionally deterministic: clocks are injected, SQLite
connections are independent, and concurrency uses barriers/events rather than
sleep-based races.  Together with the existing LCM suites these map the 27
required behaviours in the deployment specification.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

import hermes_lcm.background_compaction as background_module
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    BackgroundRuntimeHandle,
    _BackgroundRuntime,
    _RuntimeEntry,
    _background_runtime_available,
    _canonical_runtime_key,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.engine import LCMEngine
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
    values = dict(
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
        leaf_chunk_tokens=25,
        context_reserve_tokens=100_000,
    )
    values.update(overrides)
    return LCMConfig(**values)


def _seed(db_path, *, session="s", conversation="c", messages=None, tokens=None):
    messages = messages or [
        {"role": "user", "content": f"raw-{idx}"} for idx in range(8)
    ]
    tokens = tokens or [10] * len(messages)
    store = MessageStore(db_path)
    ids = store.append_batch(
        session,
        messages,
        token_estimates=tokens,
        conversation_id=conversation,
    )
    store.close()
    dag = SummaryDAG(db_path)
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session(session, conversation_id=conversation)
    lifecycle.close()
    return ids


def _explicit_job(bg, ids, *, conversation="c", session="s", frontier=0, policy="p1"):
    return bg.enqueue_or_coalesce(
        conversation_id=conversation,
        session_id=session,
        expected_frontier_store_id=frontier,
        captured_watermark_store_id=ids[-1],
        source_store_ids=ids,
        source_token_count=len(ids) * 10,
        policy_fingerprint=policy,
        expected_node_count=1,
    )


def _prepare(bg, job, *, owner="worker-a", summary="prepared summary"):
    claimed = bg.claim_next_job(owner=owner)
    assert claimed and claimed.job_id == job.job_id
    return bg.mark_prepared(
        job.job_id,
        owner=owner,
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": summary,
                "source_store_ids": list(job.source_store_ids),
                "source_token_count": job.source_token_count,
                "summary_token_count": 3,
                "creation_model": "test-model",
                "creation_provider": "test-provider",
            }
        ],
    )


def _publication_lease(bg, job_id, *, owner="publisher"):
    leased = bg.acquire_publication_lease(job_id, owner=owner)
    assert leased is not None
    return leased


def test_disabled_by_default_creates_no_background_tables(tmp_path):
    """Coverage 1: disabled is inert and does not migrate the optional schema."""
    db_path = tmp_path / "disabled.db"
    engine = LCMEngine(LCMConfig(database_path=str(db_path)))
    try:
        status = engine.get_background_compaction_status()
        assert status["enabled"] is False
        tables = {
            row[0]
            for row in engine._store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "lcm_compaction_jobs" not in tables
    finally:
        engine.shutdown()


def test_foreground_ingest_enqueues_without_summary_or_turn_mutation(tmp_path, monkeypatch):
    """Coverage 2, 5: foreground only ingests/enqueues; prepared work stays invisible."""
    db_path = tmp_path / "foreground.db"
    engine = LCMEngine(
        _config(db_path, fresh_tail_count=1, background_compaction_workers=0)
    )
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    calls = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: calls.append(kwargs) or ("should not run", 1),
    )
    messages = [{"role": "user", "content": "x " * 40} for _ in range(5)]
    before = json.loads(json.dumps(messages))
    try:
        engine.ingest(messages)
        assert messages == before
        assert calls == []
        status = engine.get_background_compaction_status()
        assert status["queued_jobs"] == 1
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        engine.shutdown()


def test_post_llm_ingest_uses_full_prompt_pressure_for_prepare_ratio(tmp_path):
    """The 55% trigger includes prompt overhead absent from message estimates."""
    db_path = tmp_path / "prompt-pressure.db"
    engine = LCMEngine(
        _config(
            db_path,
            background_compaction_workers=0,
            background_compaction_backlog_tokens=1_000_000,
            fresh_tail_count=1,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"small-{index}"}
        for index in range(5)
    ]
    # 210k is above 55% of 372k while these message bodies are only a few
    # tokens.  In production the difference is system/tool-schema/cache
    # overhead reported by the host's post-LLM usage event.
    engine.last_prompt_tokens = 210_000
    try:
        engine.ingest(messages)
        assert engine.get_background_compaction_status()["queued_jobs"] == 1
    finally:
        engine.shutdown()


def test_capture_uses_immutable_watermark_and_preserves_fresh_tail(tmp_path):
    """Coverage 3, 21: selection is durable and never captures the protected suffix."""
    db_path = tmp_path / "watermark.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        job = bg.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint="p1",
        )
        assert job is not None
        assert job.captured_watermark_store_id == ids[-3]
        assert list(job.source_store_ids) == ids[:-2]
        _seed(
            db_path,
            messages=[{"role": "user", "content": "newer"}],
            tokens=[10],
        )
        assert bg.get_job(job.job_id).captured_watermark_store_id == ids[-3]
    finally:
        bg.close()


def test_background_capture_honors_24k_fresh_tail_token_cap(
    tmp_path,
    monkeypatch,
):
    """Coverage 21: the production token cap can narrow the protected suffix."""
    db_path = tmp_path / "fresh-tail-token-cap.db"
    ids = _seed(
        db_path,
        messages=[{"role": "user", "content": f"message-{index}"} for index in range(4)],
        tokens=[10_000, 10_000, 10_000, 10_000],
    )
    monkeypatch.setattr(
        "hermes_lcm.fresh_tail.count_message_tokens",
        lambda message: int(message.get("token_estimate") or 0),
    )
    bg = BackgroundCompactionStore(
        db_path,
        config=_config(
            db_path,
            fresh_tail_count=4,
            fresh_tail_max_tokens=24_000,
        ),
    )
    try:
        job = bg.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint="p1",
        )
        assert job is not None
        assert job.source_store_ids == tuple(ids[:2])
        assert job.captured_watermark_store_id == ids[1]
    finally:
        bg.close()


def test_tool_call_group_is_not_split_at_chunk_or_watermark(tmp_path):
    """Coverage 20: assistant tool call and contiguous results are indivisible."""
    db_path = tmp_path / "tools.db"
    messages = [
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "user", "content": "tail-a"},
        {"role": "assistant", "content": "tail-b"},
    ]
    ids = _seed(db_path, messages=messages, tokens=[10, 20, 20, 10, 10])
    bg = BackgroundCompactionStore(
        db_path,
        config=_config(db_path, leaf_chunk_tokens=15, fresh_tail_count=2),
    )
    try:
        job = bg.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint="p1",
        )
        assert list(job.source_store_ids) == ids[:3]
        chunks = bg.source_chunks(job.job_id)
        assert any(chunk.source_store_ids == tuple(ids[1:3]) for chunk in chunks)
    finally:
        bg.close()


def test_one_active_job_and_queue_coalescing_never_skips_frontier(tmp_path):
    """Coverage 8, 23: partial unique ownership plus monotonic queued coalescing."""
    db_path = tmp_path / "coalesce.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        first = _explicit_job(bg, ids[:3])
        second = _explicit_job(bg, ids[:5])
        assert second.job_id == first.job_id
        assert list(second.source_store_ids) == ids[:5]
        assert second.expected_frontier_store_id == 0
        assert bg.status(conversation_id="c")["queued_jobs"] == 1
    finally:
        bg.close()


def test_session_or_policy_rollover_supersedes_stale_active_job(tmp_path):
    """A stale one-active row cannot block a new session or policy generation."""
    db_path = tmp_path / "rollover.db"
    first_ids = _seed(db_path, session="s1", conversation="c")
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        old = _explicit_job(bg, first_ids[:3], session="s1", policy="old-policy")
        second_ids = _seed(
            db_path,
            session="s2",
            conversation="c",
            messages=[{"role": "user", "content": "new-session"}],
            tokens=[10],
        )
        lifecycle = LifecycleStateStore(db_path)
        try:
            second_state = lifecycle.get_by_conversation("c")
        finally:
            lifecycle.close()
        assert second_state is not None
        assert second_state.current_bound_at is not None
        assert bg.activate_session(
            conversation_id="c",
            session_id="s2",
            expected_lifecycle_bound_at=second_state.current_bound_at,
            expected_lifecycle_reset_at=second_state.last_reset_at,
        ) is not None
        replacement = _explicit_job(
            bg,
            second_ids,
            session="s2",
            policy="new-policy",
        )
        assert replacement.job_id != old.job_id
        assert replacement.session_id == "s2"
        assert bg.get_job(old.job_id).state == "superseded"
    finally:
        bg.close()


def test_durable_queued_work_has_cross_process_wakeup_backstop(tmp_path):
    """A missed in-process notify cannot strand a committed queued job forever."""
    db_path = tmp_path / "queued-wakeup.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        _explicit_job(bg, ids)
        assert 0 < bg._next_maintenance_delay() <= 30
    finally:
        bg.close()


def test_atomic_claim_and_global_limit_across_independent_connections(tmp_path):
    """Coverage 9, 10: BEGIN IMMEDIATE claim enforces the durable global limit."""
    db_path = tmp_path / "claims.db"
    ids_a = _seed(db_path, session="s1", conversation="c1")
    ids_b = _seed(
        db_path,
        session="s2",
        conversation="c2",
        messages=[{"role": "user", "content": f"second-{idx}"} for idx in range(8)],
        tokens=[10] * 8,
    )
    config = _config(db_path, background_compaction_global_limit=1)
    first = BackgroundCompactionStore(db_path, config=config)
    second = BackgroundCompactionStore(db_path, config=config)
    _explicit_job(first, ids_a, conversation="c1", session="s1")
    _explicit_job(first, ids_b, conversation="c2", session="s2")
    barrier = threading.Barrier(3)
    claims = []

    def claim(store, owner):
        barrier.wait()
        claims.append(store.claim_next_job(owner=owner))

    threads = [
        threading.Thread(target=claim, args=(first, "a")),
        threading.Thread(target=claim, args=(second, "b")),
    ]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        assert sum(job is not None for job in claims) == 1
        assert first.status()["running_jobs"] == 1
    finally:
        first.close()
        second.close()


def test_global_provider_limit_configuration_is_clamped_to_two(tmp_path):
    config = _config(
        tmp_path / "configured-global-cap.db",
        background_compaction_global_limit=99,
    )

    assert config.background_compaction_global_limit == 2


@pytest.mark.parametrize("admission_path", ["job_claim", "emergency"])
def test_untrusted_provider_limit_cannot_admit_more_than_two_cross_connection(
    tmp_path,
    admission_path,
):
    db_path = tmp_path / f"hard-global-cap-{admission_path}.db"
    config = _config(db_path)
    # Exercise defense in depth after config construction, as a stale embedder
    # can mutate or pass an unvalidated value directly to the store API.
    config.background_compaction_global_limit = 99
    stores: list[BackgroundCompactionStore] = []
    try:
        if admission_path == "job_claim":
            source_ranges = [
                _seed(
                    db_path,
                    session=f"session-{index}",
                    conversation=f"conversation-{index}",
                )
                for index in range(3)
            ]
        stores = [
            BackgroundCompactionStore(db_path, config=config)
            for _ in range(3)
        ]
        if admission_path == "job_claim":
            for index, source_ids in enumerate(source_ranges):
                _explicit_job(
                    stores[0],
                    source_ids,
                    session=f"session-{index}",
                    conversation=f"conversation-{index}",
                )
            admitted = [
                store.claim_next_job(
                    owner=f"worker-{index}",
                    global_limit=99,
                )
                for index, store in enumerate(stores)
            ]
        else:
            admitted = [
                store.acquire_emergency_provider_permit(
                    conversation_id=f"conversation-{index}",
                    owner=f"foreground-{index}",
                    global_limit=99,
                )
                for index, store in enumerate(stores)
            ]

        assert sum(item is not None for item in admitted) == 2
        assert stores[0].status()["active_provider_permits"] == 2
        assert stores[0].status()["global_limit"] == 2
    finally:
        for store in stores:
            store.close()


def test_heartbeat_expiry_reclaim_and_late_worker_fence(tmp_path):
    """Coverage 11-13: generation-fenced leases extend, expire, and reject late owners."""
    clock = Clock()
    db_path = tmp_path / "leases.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(
        db_path,
        config=_config(db_path, background_compaction_lease_seconds=10),
        now=clock,
    )
    try:
        queued = _explicit_job(bg, ids)
        old = bg.claim_next_job(owner="old")
        old_expiry = old.lease_expires_at
        clock.advance(2)
        assert bg.heartbeat(old.job_id, owner="old", generation=old.lease_generation)
        assert bg.get_job(old.job_id).lease_expires_at > old_expiry
        clock.advance(11)
        assert bg.recover_expired_leases() == 1
        replacement = bg.claim_next_job(owner="new")
        assert replacement.job_id == queued.job_id
        assert replacement.lease_generation > old.lease_generation
        assert not bg.mark_prepared(
            old.job_id,
            owner="old",
            generation=old.lease_generation,
            prepared_nodes=[],
        )
    finally:
        bg.close()


def test_provider_timeout_retries_and_restart_recovers_jobs_without_raw_loss(tmp_path):
    """Coverage 17, 18: timeout/backoff and expired-running recovery are durable."""
    clock = Clock()
    db_path = tmp_path / "restart.db"
    ids = _seed(db_path)
    before = sqlite3.connect(db_path).execute(
        "SELECT store_id, content FROM messages ORDER BY store_id"
    ).fetchall()
    first = BackgroundCompactionStore(db_path, config=_config(db_path), now=clock)
    job = _explicit_job(first, ids)
    claimed = first.claim_next_job(owner="dead")
    first.mark_retry(
        job.job_id,
        owner="dead",
        generation=claimed.lease_generation,
        error=TimeoutError("provider timeout token=secret-value"),
    )
    retry = first.get_job(job.job_id)
    assert retry.state == "retry"
    assert "secret-value" not in retry.last_error
    first.close()

    clock.advance(1_000)
    restarted = BackgroundCompactionStore(db_path, config=_config(db_path), now=clock)
    try:
        restarted.recover_expired_leases()
        assert restarted.claim_next_job(owner="restart") is not None
        after = sqlite3.connect(db_path).execute(
            "SELECT store_id, content FROM messages ORDER BY store_id"
        ).fetchall()
        assert after == before
    finally:
        restarted.close()


def test_plain_queued_job_survives_process_restart_and_is_claimable(tmp_path):
    """Coverage 18: an untouched queued job is durable across store reopen."""
    db_path = tmp_path / "queued-restart.db"
    ids = _seed(db_path)
    first = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        queued = _explicit_job(first, ids)
        assert queued.state == "queued"
    finally:
        first.close()

    restarted = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        recovered = restarted.get_job(queued.job_id)
        assert recovered is not None
        assert recovered.state == "queued"
        claimed = restarted.claim_next_job(owner="restart-worker")
        assert claimed is not None
        assert claimed.job_id == queued.job_id
        assert claimed.state == "running"
    finally:
        restarted.close()


def test_timeout_ignoring_provider_keeps_global_slot_fenced_until_return(tmp_path):
    """A timed-out helper cannot create hidden provider concurrency or spend."""
    db_path = tmp_path / "timeout-fence.db"
    ids_a = _seed(db_path, session="s1", conversation="c1")
    ids_b = _seed(
        db_path,
        session="s2",
        conversation="c2",
        messages=[{"role": "user", "content": f"second-{idx}"} for idx in range(8)],
        tokens=[10] * 8,
    )
    config = _config(
        db_path,
        background_compaction_global_limit=1,
        background_compaction_job_timeout_seconds=1,
        background_compaction_lease_seconds=10,
    )
    first = BackgroundCompactionStore(db_path, config=config)
    second = BackgroundCompactionStore(db_path, config=config)
    policy = background_policy_fingerprint(config)
    _explicit_job(
        first,
        ids_a[:4],
        conversation="c1",
        session="s1",
        policy=policy,
    )
    _explicit_job(
        first,
        ids_b[:4],
        conversation="c2",
        session="s2",
        policy=policy,
    )
    provider_release = threading.Event()
    provider_returned = threading.Event()

    def ignores_timeout(**_kwargs):
        try:
            assert provider_release.wait(timeout=5)
            return "late summary", 1
        finally:
            provider_returned.set()

    try:
        started = time.monotonic()
        timed_out = first.run_next_job(owner="worker-a", summarizer=ignores_timeout)
        assert time.monotonic() - started < 2.0
        assert timed_out.state == "retry"
        assert second.claim_next_job(owner="worker-b") is None

        provider_release.set()
        assert provider_returned.wait(timeout=2)
        deadline = time.monotonic() + 2
        replacement = None
        while replacement is None and time.monotonic() < deadline:
            replacement = second.claim_next_job(owner="worker-b")
            if replacement is None:
                threading.Event().wait(0.01)
        assert replacement is not None
        assert replacement.conversation_id == "c2"
    finally:
        provider_release.set()
        first.close()
        second.close()


def test_partial_preparation_is_durable_and_retry_resumes_remaining_chunks(tmp_path):
    """Successful chunks survive a later provider error and are not regenerated."""
    clock = Clock()
    db_path = tmp_path / "partial-progress.db"
    ids = _seed(db_path, messages=[
        {"role": "user", "content": f"raw-{idx}"} for idx in range(4)
    ], tokens=[10, 10, 10, 10])
    bg = BackgroundCompactionStore(
        db_path,
        config=_config(db_path, leaf_chunk_tokens=20, fresh_tail_count=0),
        now=clock,
    )
    assert bg.enqueue_or_coalesce(
        conversation_id="c",
        session_id="s",
        expected_frontier_store_id=0,
        captured_watermark_store_id=ids[-1],
        source_store_ids=ids,
        source_token_count=40,
        policy_fingerprint=background_policy_fingerprint(bg.config),
        expected_node_count=2,
    ) is not None
    first_calls = []

    def fail_second(**kwargs):
        first_calls.append(kwargs["text"])
        if len(first_calls) == 2:
            raise RuntimeError("second chunk failed")
        return "first chunk summary", 1

    try:
        retried = bg.run_next_job(owner="worker-a", summarizer=fail_second)
        assert retried.state == "retry"
        assert retried.prepared_node_count == 1
        clock.advance(1_000)
        resumed_calls = []
        prepared = bg.run_next_job(
            owner="worker-b",
            summarizer=lambda **kwargs: resumed_calls.append(kwargs["text"])
            or ("second chunk summary", 1),
        )
        assert prepared.state == "prepared"
        assert prepared.prepared_node_count == 2
        assert len(resumed_calls) == 1
    finally:
        bg.close()


def test_background_provider_input_uses_canonical_scrub_and_ignore_policy(tmp_path, monkeypatch):
    """Historical raw rows are re-scrubbed and ignored dependents never reach a model."""
    db_path = tmp_path / "provider-scrub.db"
    ids = _seed(
        db_path,
        messages=[
            {"role": "system", "content": "durable system anchor"},
            {"role": "user", "content": "IGNORE: private instruction"},
            {"role": "assistant", "content": "dependent-secret"},
            {"role": "user", "content": "keep api_key=supersecretvalue"},
        ],
        tokens=[10, 10, 10, 10],
    )
    config = _config(
        db_path,
        fresh_tail_count=0,
        leaf_chunk_tokens=100,
        ignore_message_patterns=[r"^IGNORE:"],
        sensitive_patterns_enabled=True,
    )

    class SafePattern:
        pattern = r"^IGNORE:"

        def search(self, text, timeout=None):
            del timeout
            return text.startswith("IGNORE:")

    monkeypatch.setattr(
        "hermes_lcm.summary_serialization.compile_message_patterns",
        lambda _patterns: [SafePattern()],
    )
    bg = BackgroundCompactionStore(db_path, config=config)
    job = bg.enqueue_or_coalesce(
        conversation_id="c",
        session_id="s",
        expected_frontier_store_id=0,
        captured_watermark_store_id=ids[-1],
        source_store_ids=ids,
        source_token_count=40,
        policy_fingerprint=background_policy_fingerprint(config),
        expected_node_count=1,
    )
    provider_inputs = []
    try:
        prepared = bg.run_next_job(
            owner="worker",
            summarizer=lambda **kwargs: provider_inputs.append(kwargs["text"])
            or ("safe summary", 1),
        )
        assert prepared.state == "prepared", prepared.last_error
        assert len(provider_inputs) == 1
        assert "private instruction" not in provider_inputs[0]
        assert "dependent-secret" not in provider_inputs[0]
        assert "durable system anchor" not in provider_inputs[0]
        assert "supersecretvalue" not in provider_inputs[0]
        assert "keep" in provider_inputs[0]
        nodes = bg.get_prepared_nodes(job.job_id)
        assert nodes[0].source_store_ids == tuple(ids)
        assert job.source_store_ids == tuple(ids)

        leased = _publication_lease(bg, job.job_id)
        result = bg.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=leased.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert result.promoted
        dag = SummaryDAG(db_path)
        try:
            assert dag.source_message_ids(result.node_ids[0], limit=100) == ids
        finally:
            dag.close()
    finally:
        bg.close()


def test_cache_friendly_mode_is_part_of_the_background_policy_fingerprint(tmp_path):
    """Prepared node shape cannot cross a cache-condensation policy change."""
    disabled = _config(tmp_path / "disabled.db", cache_friendly_condensation_enabled=False)
    enabled = _config(tmp_path / "enabled.db", cache_friendly_condensation_enabled=True)
    assert background_policy_fingerprint(disabled) != background_policy_fingerprint(enabled)


def test_retry_budget_eventually_fails_job(tmp_path):
    """Persistent provider failure has a bounded terminal state."""
    clock = Clock()
    db_path = tmp_path / "retry-budget.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path), now=clock)
    job = _explicit_job(bg, ids)
    try:
        for attempt in range(5):
            claimed = bg.claim_next_job(owner=f"worker-{attempt}")
            assert claimed is not None
            result = bg.mark_retry(
                job.job_id,
                owner=f"worker-{attempt}",
                generation=claimed.lease_generation,
                error=RuntimeError("persistent provider failure"),
            )
            clock.advance(1_000)
        assert result.state == "failed"
        assert bg.claim_next_job(owner="extra-worker") is None
    finally:
        bg.close()


def test_source_identity_change_supersedes_prepared_publication(tmp_path):
    """Content mutation with unchanged row ids cannot publish stale derived text."""
    db_path = tmp_path / "source-identity.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        job = _explicit_job(bg, ids[:4])
        _prepare(bg, job)
        sqlite3.connect(db_path).execute(
            "UPDATE messages SET content='mutated' WHERE store_id=?", (ids[1],)
        ).connection.commit()
        leased = _publication_lease(bg, job.job_id)
        result = bg.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=leased.lease_generation,
            policy_fingerprint="p1",
        )
        assert result.promoted is False
        assert result.reason == "source_identity_mismatch"
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
        assert conn.execute(
            "SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id='c'"
        ).fetchone()[0] == 0
    finally:
        bg.close()


def test_prepared_is_invisible_then_atomic_publish_keeps_newer_raw_and_expands(tmp_path):
    """Coverage 4, 5, 22: publish exact lineage only to watermark; raw rows survive."""
    db_path = tmp_path / "publish.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    dag = SummaryDAG(db_path)
    try:
        job = _explicit_job(bg, ids[:5])
        prepared = _prepare(bg, job)
        assert prepared.state == "prepared"
        assert dag.get_session_node_count("s") == 0
        newer = _seed(
            db_path,
            messages=[{"role": "user", "content": "above-watermark"}],
            tokens=[10],
        )[0]
        leased = _publication_lease(bg, job.job_id)
        result = bg.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=leased.lease_generation,
            policy_fingerprint="p1",
        )
        assert result.promoted is True
        nodes = dag.get_session_nodes("s")
        assert len(nodes) == 1
        assert nodes[0].source_ids == ids[:5]
        assert newer > job.captured_watermark_store_id
        assert MessageStore(db_path).get_range("s", newer, newer)[0]["content"] == "above-watermark"
    finally:
        dag.close()
        bg.close()


@pytest.mark.parametrize("failure_kind", ["frontier", "after_insert"])
def test_publication_cas_or_commit_failure_is_all_or_nothing(tmp_path, failure_kind):
    """Coverage 14, 19: stale frontier and injected commit failure never half-publish."""
    db_path = tmp_path / f"atomic-{failure_kind}.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        job = _explicit_job(bg, ids[:4])
        _prepare(bg, job)
        leased = _publication_lease(bg, job.job_id)
        if failure_kind == "frontier":
            LifecycleStateStore(db_path).advance_frontier("c", "s", ids[0])
            result = bg.publish_prepared(
                job.job_id,
                owner="publisher",
                generation=leased.lease_generation,
                policy_fingerprint="p1",
            )
            assert result.promoted is False
            assert result.reason == "frontier_mismatch"
        else:
            with pytest.raises(RuntimeError, match="injected background publication failure"):
                bg.publish_prepared(
                    job.job_id,
                    owner="publisher",
                    generation=leased.lease_generation,
                    policy_fingerprint="p1",
                    failure_hook="after_canonical_insert",
                )
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
        frontier = conn.execute(
            "SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id='c'"
        ).fetchone()[0]
        assert frontier in {0, ids[0]}
    finally:
        bg.close()


def test_two_publishers_cannot_publish_same_range(tmp_path):
    """Coverage 16: publication lease + job/frontier CAS permits exactly one winner."""
    db_path = tmp_path / "double-publish.db"
    ids = _seed(db_path)
    a = BackgroundCompactionStore(db_path, config=_config(db_path))
    b = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        job = _explicit_job(a, ids[:4])
        _prepare(a, job)
        leased = _publication_lease(a, job.job_id, owner="pub-a")
        first = a.publish_prepared(
            job.job_id,
            owner="pub-a",
            generation=leased.lease_generation,
            policy_fingerprint="p1",
        )
        second = b.publish_prepared(
            job.job_id,
            owner="pub-b",
            generation=leased.lease_generation,
            policy_fingerprint="p1",
        )
        assert [first.promoted, second.promoted].count(True) == 1
        assert sqlite3.connect(db_path).execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == 1
    finally:
        a.close()
        b.close()


def test_foreground_emergency_supersedes_overlap_and_is_one_pass_bounded(tmp_path, monkeypatch):
    """Coverage 15, 26: emergency is one leaf/15s and fences prepared overlap."""
    db_path = tmp_path / "emergency.db"
    engine = LCMEngine(
        _config(
            db_path,
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            background_compaction_workers=0,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [{"role": "user", "content": f"m{idx} " + "x " * 20} for idx in range(6)]
    calls = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: calls.append(kwargs.get("timeout")) or ("emergency summary", 1),
    )
    try:
        engine.ingest(messages)
        job = engine.run_background_compaction_once(
            summarizer=lambda **kwargs: ("prepared", 1)
        )
        assert job.state == "prepared"
        # Hold the short publication lease in another owner so the emergency
        # path must fail closed on prepared publication and compact raw once.
        blocked = engine._background_compaction.acquire_publication_lease(
            job.job_id,
            owner="blocked-publisher",
        )
        assert blocked is not None
        engine.compress(messages, current_tokens=340_000)
        assert len(calls) <= 1
        assert not calls or calls[0] <= 15
        assert engine.get_background_compaction_status()["superseded_jobs"] >= 1
    finally:
        engine.shutdown()


def test_explicit_overlap_supersession_records_each_affected_job(tmp_path):
    """Overlap cleanup is atomic and emits a durable event per affected job."""
    db_path = tmp_path / "supersede-overlap.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(
        db_path,
        config=_config(db_path, background_compaction_workers=0),
    )
    try:
        job = _explicit_job(bg, ids[:3])
        assert job.state == "queued"

        changed = bg.supersede_overlapping(
            conversation_id="c",
            new_frontier_store_id=ids[1],
            reason="foreground_emergency_compaction",
        )

        assert changed == 1
        assert bg.get_job(job.job_id).state == "superseded"
        status = bg.status(conversation_id="c")
        assert status["durable_event_counts"]["prepared_summary_superseded"] == 1
    finally:
        bg.close()


@pytest.mark.parametrize(
    ("context_length", "publication_trigger"),
    [(372_000, 272_000), (900_000, 800_000), (1_000_000, 900_000)],
)
def test_fixed_reserve_publication_waits_but_manual_sync_still_works(
    tmp_path,
    monkeypatch,
    context_length,
    publication_trigger,
):
    """Coverage 6, 7, 25: publication waits for E-100k; force keeps manual sync."""
    db_path = tmp_path / "threshold.db"
    engine = LCMEngine(
        _config(
            db_path,
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            background_compaction_workers=0,
        )
    )
    engine.on_session_start(
        "s",
        conversation_id="c",
        context_length=context_length,
    )
    messages = [{"role": "user", "content": f"m{idx} " + "x " * 20} for idx in range(6)]
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("manual summary", 1),
    )
    try:
        engine.ingest(messages)
        prepared = engine.run_background_compaction_once(
            summarizer=lambda **kwargs: ("prepared summary", 1)
        )
        before = engine._dag.get_session_node_count("s")
        assert prepared.state == "prepared"
        assert engine.should_compress(prompt_tokens=publication_trigger - 1) is False
        assert engine._dag.get_session_node_count("s") == before
        below = engine.promote_prepared_compaction(
            prepared.job_id,
            messages,
            current_tokens=publication_trigger - 1,
        )
        assert below.promoted is False
        assert below.reason == "publication_trigger_not_reached"
        assert engine.should_compress(prompt_tokens=publication_trigger) is True
        promoted = engine.promote_prepared_compaction(
            prepared.job_id,
            messages,
            current_tokens=publication_trigger,
        )
        assert promoted.promoted is True
        assert promoted.messages is not messages

        manual = LCMEngine(_config(tmp_path / "manual.db", background_compaction_enabled=False, fresh_tail_count=1, leaf_chunk_tokens=1))
        manual.on_session_start("manual", conversation_id="manual", context_length=372_000)
        try:
            output = manual.compress(messages, current_tokens=1, force=True)
            assert output != messages
            assert manual.last_compression_status == "compacted"
        finally:
            manual.shutdown()
    finally:
        engine.shutdown()


def test_worker_shutdown_waits_for_transaction_boundary(tmp_path):
    """Coverage 24: shutdown does not close a connection inside an active write."""
    db_path = tmp_path / "shutdown.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    entered = threading.Event()
    release = threading.Event()

    def transaction_hook():
        entered.set()
        assert release.wait(timeout=5)

    thread = threading.Thread(
        target=lambda: bg.enqueue_or_coalesce(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            captured_watermark_store_id=ids[-1],
            source_store_ids=ids,
            source_token_count=80,
            policy_fingerprint="p1",
            expected_node_count=1,
            transaction_hook=transaction_hook,
        )
    )
    thread.start()
    assert entered.wait(timeout=5)
    close_done = threading.Event()
    closer = threading.Thread(target=lambda: (bg.close(), close_done.set()))
    closer.start()
    assert not close_done.is_set()
    release.set()
    thread.join(timeout=5)
    closer.join(timeout=5)
    assert close_done.is_set()


def test_engine_shutdown_releases_process_global_worker(tmp_path):
    """Normal shutdown cannot leak or duplicate the process-global worker."""
    db_path = tmp_path / "engine-shutdown.db"
    engine = LCMEngine(_config(db_path, background_compaction_workers=1))
    handle = engine._background_runtime_handle
    assert _background_runtime_available(db_path)
    engine.shutdown()
    assert handle._released is True
    assert _background_runtime_available(db_path) is False


def test_slow_runtime_retirement_rebinds_immediate_acquire_without_overlap(
    tmp_path,
    monkeypatch,
):
    """A timed-out stop yields one deferred, eventually live successor handle."""

    class ControlledThread:
        def __init__(self):
            self.exited = threading.Event()

        def is_alive(self):
            return not self.exited.is_set()

        def join(self, timeout=None):
            self.exited.wait(timeout=timeout)

    class ControlledRuntime:
        def __init__(self, key, db_path, config, *, auto_stop=False):
            self.key = key
            self.db_path = str(db_path)
            self.config = config
            self._thread = ControlledThread()
            self._stopping = False
            self._auto_stop = auto_stop
            self.stop_requested = threading.Event()
            self.notifications = 0

        @property
        def alive(self):
            return self._thread.is_alive() and not self._stopping

        def notify(self):
            self.notifications += 1

        def request_stop(self):
            self._stopping = True
            self.stop_requested.set()
            if self._auto_stop:
                self._thread.exited.set()

        def wait_stopped(self, timeout=None):
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()

    db_path = tmp_path / "slow-runtime-rebind.db"
    config = _config(db_path)
    key = _canonical_runtime_key(db_path)
    original = ControlledRuntime(key, db_path, config)
    entry = _RuntimeEntry(runtime=original, references=1)
    first_handle = BackgroundRuntimeHandle(key, original, entry=entry)
    replacements = []
    replacement_started = threading.Event()

    def replacement_factory(replacement_key, replacement_db_path, replacement_config):
        replacement = ControlledRuntime(
            replacement_key,
            replacement_db_path,
            replacement_config,
            auto_stop=True,
        )
        replacements.append(replacement)
        replacement_started.set()
        return replacement

    monkeypatch.setattr(background_module, "_BackgroundRuntime", replacement_factory)
    monkeypatch.setattr(background_module, "_RUNTIME_STOP_TIMEOUT_SECONDS", 0.01)
    with background_module._RUNTIME_LOCK:
        background_module._RUNTIME_REGISTRY[key] = entry

    replacement_handle = None
    releaser = threading.Thread(target=first_handle.release)
    try:
        releaser.start()
        assert original.stop_requested.wait(timeout=1)

        # The old runtime is stopping but still alive.  This acquire is bound to
        # the registry entry rather than being returned as a permanent no-op.
        replacement_handle = background_module.acquire_background_runtime(
            db_path,
            config=config,
        )
        assert replacement_handle._key == key
        assert replacement_handle._released is False
        assert replacements == []

        releaser.join(timeout=1)
        assert not releaser.is_alive()
        assert original._thread.is_alive()

        original._thread.exited.set()
        assert replacement_started.wait(timeout=1)
        with background_module._RUNTIME_LOCK:
            assert background_module._RUNTIME_REGISTRY[key] is entry
            assert entry.runtime is replacements[0]
        assert len(replacements) == 1
        assert _background_runtime_available(db_path)

        replacement_handle.notify()
        assert replacements[0].notifications == 1
    finally:
        original._thread.exited.set()
        if replacement_handle is not None:
            replacement_handle.release()
        releaser.join(timeout=1)
        with background_module._RUNTIME_LOCK:
            background_module._RUNTIME_REGISTRY.pop(key, None)


def test_write_transaction_preserves_primary_error_when_cleanup_also_fails(caplog):
    """Rollback and busy-timeout cleanup failures cannot replace the root cause."""

    class Cursor:
        def fetchone(self):
            return (30_000,)

    class FailingCleanupConnection:
        def __init__(self):
            self.in_transaction = False
            self.commands = []

        def execute(self, command):
            self.commands.append(command)
            if command == "PRAGMA busy_timeout":
                return Cursor()
            if command == "BEGIN IMMEDIATE":
                self.in_transaction = True
                return Cursor()
            if command == "ROLLBACK":
                raise RuntimeError("rollback cleanup failed")
            if command == "PRAGMA busy_timeout=30000":
                raise RuntimeError("timeout restoration failed")
            return Cursor()

    store = object.__new__(BackgroundCompactionStore)
    store._lock = threading.RLock()
    store._conn = FailingCleanupConnection()
    primary = ValueError("provider preparation failed")

    with pytest.raises(ValueError) as raised:
        with store._write_transaction(busy_timeout_ms=10):
            raise primary

    assert raised.value is primary
    assert "ROLLBACK" in store._conn.commands
    assert "PRAGMA busy_timeout=30000" in store._conn.commands
    assert "could not roll back" in caplog.text
    assert "could not restore SQLite busy_timeout" in caplog.text


def test_background_runtime_installs_summary_safety_guards(tmp_path):
    """Automatic workers enforce the same circuit breaker and spend guard."""
    db_path = tmp_path / "runtime-guards.db"
    _seed(db_path)
    runtime = _BackgroundRuntime("runtime-guards", db_path, _config(db_path))
    try:
        assert runtime._circuit_breaker is not None
        assert runtime._spend_guard is not None
    finally:
        runtime.stop()


def test_schema_startup_is_idempotent_with_two_connections(tmp_path):
    """Coverage 18, 24: simultaneous lazy migration is complete and idempotent."""
    db_path = tmp_path / "schema.db"
    barrier = threading.Barrier(3)
    stores = []
    errors = []

    def open_store():
        barrier.wait()
        try:
            stores.append(BackgroundCompactionStore(db_path, config=_config(db_path)))
        except Exception as exc:  # pragma: no cover - assertion reports detail
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    try:
        assert errors == []
        conn = sqlite3.connect(db_path)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"lcm_compaction_jobs", "lcm_compaction_prepared_nodes"} <= tables
        assert conn.execute(
            "SELECT COUNT(*) FROM lcm_migration_state WHERE step_name='background_compaction_v1'"
        ).fetchone()[0] == 1
    finally:
        for store in stores:
            store.close()


def test_same_process_publication_preserves_exact_in_memory_tail(tmp_path):
    """Publishing cannot reconstruct the current turn's post-watermark suffix."""
    db_path = tmp_path / "same-process-live-tail.db"
    engine = LCMEngine(
        _config(
            db_path,
            background_compaction_workers=0,
            fresh_tail_count=1,
            leaf_chunk_tokens=10_000,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    live_tail = {
        "role": "user",
        "content": [
            {"type": "text", "text": "exact live multimodal tail"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.invalid/live-tail.png",
                    "detail": "low",
                },
            },
        ],
        "provider_metadata": {
            "anthropic": {"cache_control": {"type": "ephemeral"}}
        },
        "client_turn_metadata": {"nonce": "same-process-tail", "ordinal": 7},
    }
    expected_tail = json.loads(json.dumps(live_tail))
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old-1"},
        {"role": "assistant", "content": "old-2"},
        {"role": "user", "content": "old-3"},
        live_tail,
    ]
    try:
        engine.ingest(messages)
        durable_ids = [
            int(row[0])
            for row in engine._store.connection.execute(
                "SELECT store_id FROM messages WHERE session_id='s' ORDER BY store_id"
            ).fetchall()
        ]
        job = _explicit_job(
            engine._background_compaction,
            durable_ids[:-1],
            policy=background_policy_fingerprint(engine._config),
        )
        prepared = _prepare(engine._background_compaction, job)
        assert prepared.state == "prepared"
        durable_max = durable_ids[-1]
        assert prepared.captured_watermark_store_id < durable_max

        promoted = engine.promote_prepared_compaction(
            prepared.job_id,
            messages,
            current_tokens=272_000,
        )
        assert promoted.promoted is True
        exact_matches = [
            message
            for message in promoted.messages
            if message.get("client_turn_metadata", {}).get("nonce")
            == "same-process-tail"
        ]
        assert exact_matches == [expected_tail]
    finally:
        engine.shutdown()


def test_sibling_frontier_adoption_merges_durable_only_row_into_exact_live_suffix(tmp_path):
    """Sibling adoption keeps live objects and adds each concurrent durable row once."""
    db_path = tmp_path / "sibling-live-tail.db"
    live_tail = {
        "role": "user",
        "content": [
            {"type": "text", "text": "sibling live multimodal tail"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/sibling-tail.png"},
            },
        ],
        "provider_metadata": {"openai": {"retention": "in-memory-only"}},
        "client_turn_metadata": {"nonce": "sibling-tail", "ordinal": 11},
    }
    expected_tail = json.loads(json.dumps(live_tail))
    seeded_messages = [
        {"role": "user", "content": "old-a"},
        {"role": "assistant", "content": "old-b"},
        {"role": "user", "content": "old-c"},
        live_tail,
    ]
    ids = _seed(
        db_path,
        messages=seeded_messages,
        tokens=[10] * len(seeded_messages),
    )
    config = _config(
        db_path,
        background_compaction_workers=0,
        fresh_tail_count=1,
        leaf_chunk_tokens=10_000,
    )
    publisher = LCMEngine(config)
    sibling = LCMEngine(config)
    publisher.on_session_start("s", conversation_id="c", context_length=372_000)
    sibling.on_session_start("s", conversation_id="c", context_length=372_000)
    try:
        policy = background_policy_fingerprint(config)
        job = _explicit_job(publisher._background_compaction, ids[:-1], policy=policy)
        _prepare(publisher._background_compaction, job)
        lease = _publication_lease(publisher._background_compaction, job.job_id)
        published = publisher._background_compaction.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=policy,
        )
        assert published.promoted is True

        concurrent_store = MessageStore(db_path)
        try:
            concurrent_store.append(
                "s",
                {"role": "assistant", "content": "durable-only-concurrent"},
                token_estimate=10,
                conversation_id="c",
            )
        finally:
            concurrent_store.close()

        in_memory_messages = [
            {"role": "system", "content": "system"},
            expected_tail,
        ]
        adopted = sibling._adopt_committed_background_frontier(in_memory_messages)
        assert adopted is not None
        assert sum(
            message.get("content") == "durable-only-concurrent"
            for message in adopted
        ) == 1
        exact_matches = [
            message
            for message in adopted
            if message.get("client_turn_metadata", {}).get("nonce")
            == "sibling-tail"
        ]
        assert exact_matches == [expected_tail]
    finally:
        publisher.shutdown()
        sibling.shutdown()


def test_complete_durable_suffix_paginates_past_ten_thousand_rows(tmp_path):
    """Every durable suffix row is returned even beyond MessageStore's 10k default."""
    db_path = tmp_path / "large-durable-suffix.db"
    engine = LCMEngine(
        _config(db_path, background_compaction_workers=0, fresh_tail_count=0)
    )
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    row_count = 10_007
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            """
            INSERT INTO messages(
                session_id, source, conversation_id, role, content,
                timestamp, token_estimate, pinned
            ) VALUES('s', 'test', 'c', 'user', ?, ?, 1, 0)
            """,
            ((f"suffix-{index}", float(index + 1)) for index in range(row_count)),
        )
        conn.commit()
    finally:
        conn.close()
    try:
        suffix = engine._all_durable_messages_after(0)
        assert len(suffix) == row_count
        assert [int(row["store_id"]) for row in suffix] == sorted(
            int(row["store_id"]) for row in suffix
        )
        assert len({int(row["store_id"]) for row in suffix}) == row_count
        assert suffix[-1]["content"] == f"suffix-{row_count - 1}"
    finally:
        engine.shutdown()


def test_durable_diagnostics_are_conversation_scoped_and_show_policy_debt_and_coverage(
    tmp_path,
):
    """Durable events and last results never bleed across conversations."""
    db_path = tmp_path / "durable-diagnostics.db"
    config = _config(
        db_path,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=32_123,
        background_compaction_prepare_ratio=0.61,
    )
    a_ids = _seed(db_path, session="sa", conversation="ca")
    b_ids = _seed(db_path, session="sb", conversation="cb")
    bg = BackgroundCompactionStore(db_path, config=config)
    try:
        a_job = _explicit_job(bg, a_ids[:4], conversation="ca", session="sa")
        assert _prepare(bg, a_job, owner="worker-a").state == "prepared"
        a_lease = _publication_lease(bg, a_job.job_id, owner="publisher-a")
        assert bg.publish_prepared(
            a_job.job_id,
            owner="publisher-a",
            generation=a_lease.lease_generation,
            policy_fingerprint="p1",
        ).promoted

        b_job = _explicit_job(bg, b_ids[:4], conversation="cb", session="sb")
        assert _prepare(bg, b_job, owner="worker-b").state == "prepared"
        b_lease = _publication_lease(bg, b_job.job_id, owner="publisher-b")
        assert bg.publish_prepared(
            b_job.job_id,
            owner="publisher-b",
            generation=b_lease.lease_generation,
            policy_fingerprint="p1",
        ).promoted

        status = bg.status(conversation_id="ca")
        assert status["last_preparation_result"]["job_id"] == a_job.job_id
        assert status["last_publication_result"]["job_id"] == a_job.job_id
        assert status["durable_event_counts"]["background_summary_prepared"] == 1
        assert status["durable_event_counts"]["prepared_summary_published"] == 1
        assert "prepared_summary_superseded" not in status["durable_event_reasons"]
        assert status["backlog_trigger_tokens"] == 32_123
        assert status["prepare_ratio"] == pytest.approx(0.61)
        assert status["active_coverage"] == "durable_published"

        debt_ids = _seed(db_path, session="sd", conversation="cd")
        debt_job = _explicit_job(
            bg,
            debt_ids[:3],
            conversation="cd",
            session="sd",
        )
        assert debt_job.source_token_count > 0
        debt_status = bg.status(conversation_id="cd")
        assert debt_status["current_debt_tokens"] == debt_job.source_token_count
        assert "worker_unavailable" in debt_status["durable_event_reasons"]
    finally:
        bg.close()


def test_provider_permit_diagnostics_recovery_and_reason_events_are_durable(tmp_path):
    """Provider permits, timeout/recovery, and bounded event text are auditable."""
    clock = Clock()
    db_path = tmp_path / "permit-diagnostics.db"
    config = _config(
        db_path,
        background_compaction_workers=0,
        background_compaction_lease_seconds=10,
    )
    a_ids = _seed(db_path, session="sa", conversation="ca")
    b_ids = _seed(db_path, session="sb", conversation="cb")
    bg = BackgroundCompactionStore(db_path, config=config, now=clock)
    try:
        a_job = _explicit_job(bg, a_ids, conversation="ca", session="sa")
        b_job = _explicit_job(bg, b_ids, conversation="cb", session="sb")
        a_claim = bg.claim_next_job(owner="active-worker")
        b_claim = bg.claim_next_job(owner="timeout-worker")
        assert {a_claim.job_id, b_claim.job_id} == {a_job.job_id, b_job.job_id}
        timeout_claim = a_claim if a_claim.job_id == b_job.job_id else b_claim
        active_claim = b_claim if timeout_claim is a_claim else a_claim
        assert bg.mark_retry(
            timeout_claim.job_id,
            owner="timeout-worker" if timeout_claim is b_claim else "active-worker",
            generation=timeout_claim.lease_generation,
            error=TimeoutError("provider timeout password=supersecretvalue"),
            keep_provider_permit=True,
            completion_reason="provider_timeout",
        )

        overall = bg.status()
        assert overall["active_provider_permits"] == 2
        assert overall["timed_out_provider_permits"] == 1
        assert 0 <= overall["provider_permit_oldest_age_seconds"] <= 10
        assert 0 <= overall["provider_permit_min_lease_remaining_seconds"] <= 10
        timed_out = bg.status(conversation_id=timeout_claim.conversation_id)
        assert timed_out["active_provider_permits"] == 1
        assert timed_out["timed_out_provider_permits"] == 1
        assert "provider_timeout" in timed_out["durable_event_reasons"]

        bg.record_diagnostic_event(
            "sqlite_lock_contention",
            conversation_id=timeout_claim.conversation_id,
            session_id=timeout_claim.session_id,
            detail="authorization=Bearer-secret " + "x" * 2_000,
        )
        event_row = sqlite3.connect(db_path).execute(
            """
            SELECT detail FROM lcm_compaction_events
            WHERE conversation_id=? AND reason_code='sqlite_lock_contention'
            ORDER BY event_id DESC LIMIT 1
            """,
            (timeout_claim.conversation_id,),
        ).fetchone()
        assert event_row is not None
        assert "Bearer-secret" not in event_row[0]
        assert len(event_row[0]) <= 512

        clock.advance(11)
        assert bg.recover_expired_leases() >= 1
        recovered = bg.status(conversation_id=active_claim.conversation_id)
        assert "lease_expiration" in recovered["durable_event_reasons"]
    finally:
        bg.close()


def test_no_eligible_range_and_emergency_publish_emit_exact_durable_reasons(tmp_path):
    """Source-range failure and emergency publication use stable reason codes."""
    db_path = tmp_path / "reason-events.db"
    config = _config(db_path, background_compaction_workers=0)
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=config)
    try:
        policy = background_policy_fingerprint(config)
        _explicit_job(bg, ids[:4], policy=policy)
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM messages WHERE store_id=?", (ids[1],))
        conn.commit()
        conn.close()
        retry = bg.run_next_job(
            owner="worker",
            summarizer=lambda **_kwargs: ("must not run", 1),
        )
        assert retry.state == "retry"
        assert "no_eligible_raw_range" in bg.status(
            conversation_id="c"
        )["durable_event_reasons"]

        emergency_ids = _seed(db_path, session="se", conversation="ce")
        lifecycle = LifecycleStateStore(db_path)
        try:
            emergency_state = lifecycle.get_by_conversation("ce")
        finally:
            lifecycle.close()
        assert emergency_state is not None
        assert emergency_state.current_bound_at is not None
        emergency_generation = bg.activate_session(
            conversation_id="ce",
            session_id="se",
            expected_lifecycle_bound_at=emergency_state.current_bound_at,
        )
        assert emergency_generation is not None
        emergency_snapshot = bg.capture_foreground_source_snapshot(
            conversation_id="ce",
            session_id="se",
            expected_frontier_store_id=0,
            captured_watermark_store_id=emergency_ids[1],
            expected_source_store_ids=emergency_ids[:2],
        )
        assert emergency_snapshot is not None
        emergency_identity = bg.foreground_source_identity_hash(
            emergency_snapshot,
            emergency_ids[:2],
        )
        assert emergency_identity
        result = bg.publish_foreground_node(
            {
                "summary": "emergency summary",
                "token_count": 3,
                "source_token_count": 20,
                "source_ids": emergency_ids[:2],
            },
            conversation_id="ce",
            session_id="se",
            expected_session_fence_generation=emergency_generation,
            expected_frontier_store_id=0,
            captured_watermark_store_id=emergency_ids[1],
            expected_source_identity_hash=emergency_identity,
            debt_tokens=20,
        )
        assert result.promoted
        assert "foreground_emergency_compaction" in bg.status(
            conversation_id="ce"
        )["durable_event_reasons"]

        stale_ids = _seed(db_path, session="ss", conversation="cs")
        stale_job = _explicit_job(
            bg,
            stale_ids[:3],
            conversation="cs",
            session="ss",
        )
        assert _prepare(bg, stale_job, owner="stale-worker").state == "prepared"
        lifecycle = LifecycleStateStore(db_path)
        try:
            lifecycle.advance_frontier("cs", "ss", stale_ids[0])
        finally:
            lifecycle.close()
        stale_lease = _publication_lease(
            bg,
            stale_job.job_id,
            owner="stale-publisher",
        )
        stale_result = bg.publish_prepared(
            stale_job.job_id,
            owner="stale-publisher",
            generation=stale_lease.lease_generation,
            policy_fingerprint="p1",
        )
        assert stale_result.promoted is False
        assert "prepared_summary_superseded" in bg.status(
            conversation_id="cs"
        )["durable_event_reasons"]
    finally:
        bg.close()


def test_publication_fault_releases_lease_and_another_publisher_can_retry(tmp_path):
    """A rolled-back publication never strands its pre-existing publication lease."""
    db_path = tmp_path / "publication-lease-retry.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        job = _explicit_job(bg, ids[:4])
        assert _prepare(bg, job).state == "prepared"
        first = _publication_lease(bg, job.job_id, owner="publisher-a")
        with pytest.raises(RuntimeError, match="injected background publication failure"):
            bg.publish_prepared(
                job.job_id,
                owner="publisher-a",
                generation=first.lease_generation,
                policy_fingerprint="p1",
                failure_hook="after_canonical_insert",
            )

        second = bg.acquire_publication_lease(job.job_id, owner="publisher-b")
        assert second is not None
        assert second.lease_owner == "publisher-b"
        published = bg.publish_prepared(
            job.job_id,
            owner="publisher-b",
            generation=second.lease_generation,
            policy_fingerprint="p1",
        )
        assert published.promoted
        assert sqlite3.connect(db_path).execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == 1
    finally:
        bg.close()


def test_enqueue_and_coalesce_persist_exact_lifecycle_debt_transactionally(tmp_path):
    """Preparation-only pressure records debt before any foreground compression."""
    db_path = tmp_path / "enqueue-debt.db"
    ids = _seed(db_path)
    bg = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        first = _explicit_job(bg, ids[:3])
        conn = sqlite3.connect(db_path)
        debt = conn.execute(
            """
            SELECT debt_kind, debt_size_estimate
            FROM lcm_lifecycle_state WHERE conversation_id='c'
            """
        ).fetchone()
        assert debt == ("raw_backlog", first.source_token_count)

        coalesced = bg.enqueue_or_coalesce(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            captured_watermark_store_id=ids[4],
            source_store_ids=ids[:5],
            source_token_count=50,
            policy_fingerprint="p1",
            expected_node_count=1,
        )
        debt = conn.execute(
            """
            SELECT debt_kind, debt_size_estimate
            FROM lcm_lifecycle_state WHERE conversation_id='c'
            """
        ).fetchone()
        assert coalesced.job_id == first.job_id
        assert debt == ("raw_backlog", coalesced.source_token_count)
        assert bg.status(conversation_id="c")["current_debt_tokens"] == (
            coalesced.source_token_count
        )
        conn.close()
    finally:
        bg.close()
