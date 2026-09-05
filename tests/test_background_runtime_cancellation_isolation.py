"""Cancellation boundaries for the persistent background runtime."""

from __future__ import annotations

import threading
import time

import hermes_lcm.escalation as escalation
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    CompactionJob,
    _BackgroundRuntime,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def _config(db_path, **overrides) -> LCMConfig:
    values = dict(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=1,
        background_compaction_global_limit=1,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=1,
        background_compaction_lease_seconds=10,
        background_compaction_backlog_tokens=1,
        background_compaction_min_provider_source_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=0,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=10_000,
        context_reserve_tokens=100_000,
    )
    values.update(overrides)
    return LCMConfig(**values)


def _seed_conversation(db_path, *, conversation: str, session: str, content: str) -> list[int]:
    messages = MessageStore(db_path)
    try:
        ids = messages.append_batch(
            session,
            [{"role": "user", "content": content}],
            token_estimates=[1_000],
            conversation_id=conversation,
        )
    finally:
        messages.close()
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session(session, conversation_id=conversation)
    finally:
        lifecycle.close()
    return ids


def _enqueue(
    store: BackgroundCompactionStore,
    ids: list[int],
    *,
    conversation: str,
    session: str,
    priority: int = 0,
) -> CompactionJob:
    job = store.enqueue_or_coalesce(
        conversation_id=conversation,
        session_id=session,
        expected_frontier_store_id=0,
        captured_watermark_store_id=ids[-1],
        source_store_ids=ids,
        source_token_count=1_000,
        policy_fingerprint=background_policy_fingerprint(store.config),
        expected_node_count=1,
        priority=priority,
    )
    assert job is not None
    return job


def test_job_timeout_does_not_poison_persistent_runtime_for_next_job(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "runtime-job-cancel-isolation.db"
    SummaryDAG(db_path).close()
    first_ids = _seed_conversation(
        db_path,
        conversation="first-conversation",
        session="first-session",
        content="first-runtime-job",
    )
    second_ids = _seed_conversation(
        db_path,
        conversation="second-conversation",
        session="second-session",
        content="second-runtime-job",
    )
    config = _config(db_path)
    writer = BackgroundCompactionStore(db_path, config=config)
    first_job = _enqueue(
        writer,
        first_ids,
        conversation="first-conversation",
        session="first-session",
        priority=20,
    )
    second_job = _enqueue(
        writer,
        second_ids,
        conversation="second-conversation",
        session="second-session",
        priority=10,
    )
    first_entered = threading.Event()
    first_cancelled = threading.Event()
    allow_first_provider_return = threading.Event()
    first_retry_persisted = threading.Event()
    first_permit_released = threading.Event()
    second_entered = threading.Event()
    second_prepared = threading.Event()
    seen_cancel_events: dict[str, threading.Event] = {}
    original_mark_prepared = BackgroundCompactionStore.mark_prepared
    original_mark_retry = BackgroundCompactionStore.mark_retry
    original_release_provider_permit = (
        BackgroundCompactionStore.release_provider_permit
    )

    def summarize(*, text, cancel_event, **_kwargs):
        if "first-runtime-job" in text:
            seen_cancel_events["first"] = cancel_event
            first_entered.set()
            assert cancel_event.wait(timeout=3)
            first_cancelled.set()
            assert allow_first_provider_return.wait(timeout=3)
            return "late first summary", 1
        assert "second-runtime-job" in text
        seen_cancel_events["second"] = cancel_event
        second_entered.set()
        return "second summary", 1

    def observed_mark_prepared(store, *args, **kwargs):
        result = original_mark_prepared(store, *args, **kwargs)
        if isinstance(result, CompactionJob) and result.job_id == second_job.job_id:
            second_prepared.set()
        return result

    def observed_mark_retry(store, *args, **kwargs):
        result = original_mark_retry(store, *args, **kwargs)
        if isinstance(result, CompactionJob) and result.job_id == first_job.job_id:
            first_retry_persisted.set()
        return result

    def observed_release_provider_permit(store, job_id, *args, **kwargs):
        result = original_release_provider_permit(
            store,
            job_id,
            *args,
            **kwargs,
        )
        if str(job_id) == first_job.job_id:
            first_permit_released.set()
        return result

    monkeypatch.setattr(escalation, "summarize_with_escalation", summarize)
    monkeypatch.setattr(
        BackgroundCompactionStore,
        "mark_prepared",
        observed_mark_prepared,
    )
    monkeypatch.setattr(
        BackgroundCompactionStore,
        "mark_retry",
        observed_mark_retry,
    )
    monkeypatch.setattr(
        BackgroundCompactionStore,
        "release_provider_permit",
        observed_release_provider_permit,
    )
    runtime = _BackgroundRuntime("runtime-job-cancel-isolation", db_path, config)
    try:
        assert first_entered.wait(timeout=2)
        assert first_cancelled.wait(timeout=3)
        assert first_retry_persisted.wait(timeout=2)
        allow_first_provider_return.set()
        assert first_permit_released.wait(timeout=2)
        runtime.notify()
        assert second_entered.wait(timeout=2)
        assert second_prepared.wait(timeout=2)

        persisted_first = writer.get_job(first_job.job_id)
        persisted_second = writer.get_job(second_job.job_id)
        assert persisted_first is not None
        assert persisted_first.state == "retry"
        assert persisted_first.completion_reason == "provider_timeout"
        assert persisted_second is not None
        assert persisted_second.state == "prepared"
        assert not runtime._stop_event.is_set()
        assert seen_cancel_events["first"].is_set()
        assert not seen_cancel_events["second"].is_set()
        assert seen_cancel_events["first"] is not seen_cancel_events["second"]
        assert seen_cancel_events["first"] is not runtime._stop_event
        assert seen_cancel_events["second"] is not runtime._stop_event
    finally:
        allow_first_provider_return.set()
        runtime.stop()
        writer.close()


def test_runtime_stop_signal_cancels_active_job_and_stops_promptly(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "runtime-stop-cancellation.db"
    SummaryDAG(db_path).close()
    ids = _seed_conversation(
        db_path,
        conversation="stop-conversation",
        session="stop-session",
        content="runtime-stop-job",
    )
    config = _config(db_path, background_compaction_job_timeout_seconds=30)
    writer = BackgroundCompactionStore(db_path, config=config)
    job = _enqueue(
        writer,
        ids,
        conversation="stop-conversation",
        session="stop-session",
    )
    provider_entered = threading.Event()
    provider_cancelled = threading.Event()
    provider_returned = threading.Event()
    permit_released = threading.Event()
    seen_cancel_events: list[threading.Event] = []
    original_release_provider_permit = (
        BackgroundCompactionStore.release_provider_permit
    )

    def summarize(*, cancel_event, **_kwargs):
        seen_cancel_events.append(cancel_event)
        provider_entered.set()
        try:
            assert cancel_event.wait(timeout=3)
            provider_cancelled.set()
            return "cancelled provider result", 1
        finally:
            provider_returned.set()

    def observed_release_provider_permit(store, *args, **kwargs):
        result = original_release_provider_permit(store, *args, **kwargs)
        permit_released.set()
        return result

    monkeypatch.setattr(escalation, "summarize_with_escalation", summarize)
    monkeypatch.setattr(
        BackgroundCompactionStore,
        "release_provider_permit",
        observed_release_provider_permit,
    )
    runtime = _BackgroundRuntime("runtime-stop-cancellation", db_path, config)
    try:
        assert provider_entered.wait(timeout=2)
        started = time.monotonic()
        runtime.request_stop()
        assert provider_cancelled.wait(timeout=2)
        assert provider_returned.wait(timeout=2)
        assert runtime.wait_stopped(timeout=2)
        assert permit_released.wait(timeout=2)
        assert time.monotonic() - started < 2
        assert runtime._stop_event.is_set()
        assert seen_cancel_events[0].is_set()
        assert seen_cancel_events[0] is not runtime._stop_event

        persisted = writer.get_job(job.job_id)
        assert persisted is not None
        assert persisted.state == "retry"
        assert persisted.completion_reason == "provider_timeout"
        assert persisted.lease_owner is None
        assert persisted.lease_expires_at is None
        assert writer.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases"
        ).fetchone()[0] == 0
    finally:
        runtime.stop()
        writer.close()
