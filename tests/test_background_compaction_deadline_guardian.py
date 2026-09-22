"""Deadline, terminal-attempt, and permit-guardian regression contracts."""

from __future__ import annotations

import logging
import threading

import pytest

import hermes_lcm.background_compaction as background_compaction_module
from hermes_lcm import escalation
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


def _config(db_path, **overrides) -> LCMConfig:
    values = {
        "database_path": str(db_path),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "background_compaction_global_limit": 1,
        "background_compaction_max_passes": 4,
        "background_compaction_job_timeout_seconds": 1,
        "background_compaction_lease_seconds": 30,
        "background_compaction_backlog_tokens": 1,
        "background_compaction_min_provider_source_tokens": 1,
        "background_compaction_prepare_ratio": 0.55,
        "fresh_tail_count": 0,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 120_000,
        "summary_model": "primary/model-a",
        "summary_fallback_models": ["fallback/model-b"],
    }
    values.update(overrides)
    return LCMConfig(**values)


def _seed_and_enqueue(background: BackgroundCompactionStore):
    db_path = str(background.db_path)
    store = MessageStore(db_path)
    try:
        source_ids = store.append_batch(
            "session",
            [
                {"role": "user", "content": f"source-{index} " + "x " * 20}
                for index in range(4)
            ],
            token_estimates=[100] * 4,
            conversation_id="conversation",
        )
    finally:
        store.close()
    SummaryDAG(db_path).close()
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session("session", conversation_id="conversation")
    finally:
        lifecycle.close()
    job = background.enqueue_or_coalesce(
        conversation_id="conversation",
        session_id="session",
        expected_frontier_store_id=0,
        captured_watermark_store_id=source_ids[-1],
        source_store_ids=source_ids,
        source_token_count=400,
        policy_fingerprint=background_policy_fingerprint(background.config),
        expected_node_count=1,
    )
    assert job is not None
    return job


def _observe_provider_permit_release(monkeypatch) -> threading.Event:
    """Return an event set after a durable permit-release attempt completes."""

    released = threading.Event()
    original_release = BackgroundCompactionStore.release_provider_permit

    def observed_release(self, *args, **kwargs):
        result = original_release(self, *args, **kwargs)
        released.set()
        return result

    monkeypatch.setattr(
        BackgroundCompactionStore,
        "release_provider_permit",
        observed_release,
    )
    return released


def test_run_next_job_passes_absolute_deadline_and_cancel_event_to_escalation(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "background-deadline-forwarding.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    _seed_and_enqueue(background)
    clock = _Clock()
    external_cancel = threading.Event()
    seen: dict[str, object] = {}
    provider_calls: list[str] = []

    def late_primary(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
    ):
        del timeout, route_info
        provider_calls.append(model)
        clock.advance(2.0)
        return "late provider summary"

    def compatible_summarizer(*, deadline, cancel_event, **kwargs):
        seen.update(deadline=deadline, cancel_event=cancel_event)
        return escalation.summarize_with_escalation(
            deadline=deadline,
            cancel_event=cancel_event,
            **kwargs,
        )

    monkeypatch.setattr(background_compaction_module.time, "monotonic", clock)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", late_primary)
    try:
        result = background.run_next_job(
            owner="deadline-worker",
            summarizer=compatible_summarizer,
            cancel_event=external_cancel,
        )
        assert result is not None
        assert result.state == "retry"
        assert result.completion_reason == "provider_timeout"
        assert seen["deadline"] == pytest.approx(101.0)
        assert seen["cancel_event"] is not external_cancel
        assert seen["cancel_event"].is_set()
        assert not external_cancel.is_set()
        assert provider_calls == ["primary/model-a"]
    finally:
        background.close()


def test_lease_loss_cancels_escalation_before_fallback_or_l2(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "background-lease-loss-cancellation.db"
    config = _config(db_path, background_compaction_lease_seconds=1)
    background = BackgroundCompactionStore(db_path, config=config)
    _seed_and_enqueue(background)
    permit_released = _observe_provider_permit_release(monkeypatch)
    external_cancel = threading.Event()
    provider_returned = threading.Event()
    provider_calls: list[str] = []
    seen_cancel: dict[str, threading.Event] = {}
    heartbeat_calls = 0
    original_heartbeat = background.heartbeat

    def lose_lease_after_provider_start(*args, **kwargs):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls <= 2:
            return original_heartbeat(*args, **kwargs)
        return False

    def primary_waiting_for_cancel(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
    ):
        del timeout, route_info
        provider_calls.append(model)
        assert seen_cancel["event"].wait(timeout=3)
        provider_returned.set()
        return "lease-late provider summary"

    def compatible_summarizer(*, deadline, cancel_event, **kwargs):
        assert cancel_event is not external_cancel
        seen_cancel["event"] = cancel_event
        return escalation.summarize_with_escalation(
            deadline=deadline,
            cancel_event=cancel_event,
            **kwargs,
        )

    monkeypatch.setattr(background, "heartbeat", lose_lease_after_provider_start)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", primary_waiting_for_cancel)
    try:
        result = background.run_next_job(
            owner="lease-worker",
            summarizer=compatible_summarizer,
            cancel_event=external_cancel,
        )
        assert result is not None
        assert provider_returned.wait(timeout=3)
        assert seen_cancel["event"].is_set()
        assert not external_cancel.is_set()
        assert provider_calls == ["primary/model-a"]
        assert permit_released.wait(timeout=3)
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases"
        ).fetchone()[0] == 0
    finally:
        if "event" in seen_cancel:
            seen_cancel["event"].set()
        external_cancel.set()
        background.close()


def test_terminal_fallback_reason_overrides_late_or_partial_trace(tmp_path):
    db_path = tmp_path / "terminal-attempt-reason.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    job = _seed_and_enqueue(background)
    claimed = background.claim_next_job(owner="attempt-worker")
    assert claimed is not None and claimed.job_id == job.job_id
    try:
        background._record_compaction_attempts(
            job=claimed,
            chunk_ordinal=0,
            source_token_count=400,
            trace={
                "attempts": [
                    {
                        "escalation_level": 1,
                        "logical_route_ordinal": 0,
                        "outcome": "success",
                        "failure_reason": "",
                        "summary_token_count": 20,
                    }
                ]
            },
            fallback_reason="provider_timeout",
            fallback_latency_ms=123,
        )
        background._record_compaction_attempts(
            job=claimed,
            chunk_ordinal=1,
            source_token_count=400,
            trace={"attempts": [{"actual_provider": "provider-only"}]},
            fallback_reason="stale_lease",
            fallback_latency_ms=456,
        )

        rows = background.connection.execute(
            """
            SELECT chunk_ordinal, outcome, failure_reason, summary_token_count
            FROM lcm_compaction_attempts
            WHERE job_id=?
            ORDER BY chunk_ordinal
            """,
            (job.job_id,),
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (0, "failure", "provider_timeout", 0),
            (1, "failure", "stale_lease", 0),
        ]
        latest = background.status()["last_provider_attempt"]
        assert latest["outcome"] == "failure"
        assert latest["failure_reason"] == "stale_lease"
    finally:
        background.close()


def test_guardian_retries_with_bounded_backoff_and_sanitized_rate_limited_logs(
    tmp_path,
    monkeypatch,
    caplog,
):
    db_path = tmp_path / "guardian-backoff.db"
    owner = BackgroundCompactionStore(db_path, config=_config(db_path))
    completed = threading.Event()
    completed.set()
    notified = threading.Event()
    delays: list[float] = []
    opens = 0

    class FlakyGuardianStore:
        def __init__(self, *_args, **_kwargs):
            nonlocal opens
            opens += 1
            if opens <= 5:
                raise RuntimeError("authorization=super-secret-provider-token")

        def release_provider_permit(self, *_args, **_kwargs):
            return True

        def close(self):
            return None

    monkeypatch.setattr(
        background_compaction_module,
        "BackgroundCompactionStore",
        FlakyGuardianStore,
    )
    monkeypatch.setattr(background_compaction_module.time, "sleep", delays.append)
    monkeypatch.setattr(
        background_compaction_module,
        "_notify_background_runtime",
        lambda _db_path: notified.set(),
    )
    caplog.set_level(logging.WARNING, logger=background_compaction_module.__name__)
    try:
        assert owner._start_provider_permit_guardian(
            completed,
            job_id="guardian-job",
            owner="guardian-owner",
            generation=1,
        ) is True
        assert notified.wait(timeout=3)
        assert delays[:5] == pytest.approx([0.25, 0.5, 1.0, 2.0, 4.0])
        retry_logs = [
            record
            for record in caplog.records
            if "guardian open failed" in record.getMessage()
        ]
        assert len(retry_logs) == 1
        assert retry_logs[0].exc_info is None
        assert "super-secret-provider-token" not in retry_logs[0].getMessage()
    finally:
        owner.close()


def test_guardian_start_failure_transfers_release_to_provider_thread(
    tmp_path,
    monkeypatch,
    caplog,
):
    db_path = tmp_path / "guardian-start-failure.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    job = _seed_and_enqueue(background)
    provider_entered = threading.Event()
    provider_release = threading.Event()
    provider_returned = threading.Event()
    permit_released = _observe_provider_permit_release(monkeypatch)
    original_start = threading.Thread.start

    def selective_start(thread):
        if thread.name.startswith("lcm-bg-permit-guardian-"):
            raise RuntimeError("authorization=guardian-start-secret")
        return original_start(thread)

    def ignores_timeout(**_kwargs):
        provider_entered.set()
        try:
            assert provider_release.wait(timeout=5)
            return "late physical provider result", 1
        finally:
            provider_returned.set()

    monkeypatch.setattr(threading.Thread, "start", selective_start)
    caplog.set_level(logging.WARNING, logger=background_compaction_module.__name__)
    result = None
    caught: BaseException | None = None
    try:
        try:
            result = background.run_next_job(
                owner="thread-start-worker",
                summarizer=ignores_timeout,
            )
        except BaseException as exc:  # RED captures the pre-fix escape.
            caught = exc
        assert provider_entered.wait(timeout=2)
        provider_release.set()
        assert provider_returned.wait(timeout=2)
        assert permit_released.wait(timeout=2)
        lease_row = background.connection.execute(
            """
            SELECT lease_owner, lease_expires_at, heartbeat_at
            FROM lcm_compaction_jobs
            WHERE job_id=?
            """,
            (job.job_id,),
        ).fetchone()
        assert lease_row is not None
        assert tuple(lease_row) == (None, None, None)
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases"
        ).fetchone()[0] == 0
        assert caught is None
        assert result is not None and result.state == "retry"
        start_logs = [
            record
            for record in caplog.records
            if "guardian thread start failed" in record.getMessage()
        ]
        assert len(start_logs) == 1
        assert start_logs[0].exc_info is None
        assert "guardian-start-secret" not in start_logs[0].getMessage()
    finally:
        provider_release.set()
        background.close()


def test_guardian_start_failure_never_runs_retry_loop_on_provider_thread(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "guardian-start-off-provider-thread.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    completed = threading.Event()
    completed.set()
    guardian_opened = threading.Event()
    allow_guardian_open = threading.Event()
    guardian_finished = threading.Event()
    provider_finished = threading.Event()
    guardian_thread_ids: list[int] = []
    release_calls: list[int] = []
    result_box: dict[str, object] = {}
    original_start = threading.Thread.start

    class BlockingGuardianStore:
        def __init__(self, *_args, **_kwargs):
            guardian_thread_ids.append(threading.get_ident())
            guardian_opened.set()
            assert allow_guardian_open.wait(timeout=3)

        def release_provider_permit(self, *_args, **_kwargs):
            release_calls.append(threading.get_ident())
            return True

        def close(self):
            return None

    def selective_start(thread):
        if thread.name.startswith("lcm-bg-permit-guardian-"):
            raise RuntimeError("guardian thread unavailable")
        return original_start(thread)

    def invoke_from_provider_thread() -> None:
        provider_thread_id = threading.get_ident()
        result_box["provider_thread_id"] = provider_thread_id
        result_box["started"] = background._start_provider_permit_guardian(
            completed,
            job_id="guardian-job",
            owner="guardian-owner",
            generation=1,
            on_start_failure=lambda finalizer: finalizer(),
        )
        provider_finished.set()

    monkeypatch.setattr(
        background_compaction_module,
        "BackgroundCompactionStore",
        BlockingGuardianStore,
    )
    monkeypatch.setattr(threading.Thread, "start", selective_start)
    monkeypatch.setattr(
        background_compaction_module,
        "_notify_background_runtime",
        lambda _db_path: guardian_finished.set(),
    )
    provider = threading.Thread(
        target=invoke_from_provider_thread,
        name="simulated-foreground-provider",
    )
    try:
        provider.start()
        assert guardian_opened.wait(timeout=2)

        # Even while durable cleanup is blocked, the physical provider thread
        # must be free to return.  The infinite-retry guardian owns cleanup on
        # a distinct fallback thread.
        assert provider_finished.wait(timeout=1)
        assert guardian_thread_ids
        assert guardian_thread_ids[0] != result_box["provider_thread_id"]
    finally:
        allow_guardian_open.set()
        provider.join(timeout=2)
        assert guardian_finished.wait(timeout=2)
        assert release_calls == guardian_thread_ids
        background.close()
