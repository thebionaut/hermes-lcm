"""Deadline contracts for automatic entry setup and permit cleanup.

The synchronization events below are barriers, not timing polls.  Every
foreground assertion is made while the contended operation is still held.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import hermes_lcm.background_compaction as background_compaction_module
from hermes_lcm.background_compaction import ForegroundProviderPermitUnavailable
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import ForegroundDeadlineExceeded


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> int:
        self.value += seconds
        return 0


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "entry-cleanup-deadline.db"),
            background_compaction_enabled=True,
            background_compaction_workers=0,
            background_compaction_global_limit=1,
            context_reserve_tokens=100_000,
            fresh_tail_count=1,
            fresh_tail_max_tokens=0,
        )
    )
    engine.on_session_start("session", conversation_id="conversation", context_length=372_000)
    return engine


@contextmanager
def _held_lock(lock):
    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with lock:
            entered.set()
            assert release.wait(timeout=3)

    holder = threading.Thread(target=hold)
    holder.start()
    assert entered.wait(timeout=2)
    try:
        yield
    finally:
        release.set()
        holder.join(timeout=2)
        assert not holder.is_alive()


def _invoke(operation):
    finished = threading.Event()
    result: dict[str, object] = {}

    def run() -> None:
        try:
            result["value"] = operation()
        except BaseException as exc:  # pragma: no cover - asserted by caller
            result["error"] = exc
        finally:
            finished.set()

    caller = threading.Thread(target=run)
    caller.start()
    return caller, finished, result


def test_automatic_entry_never_waits_past_wall_for_auxiliary_lock(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "raw input remains authoritative"}]
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        with _held_lock(engine._auxiliary_session_lock):
            caller, finished, result = _invoke(
                lambda: engine.compress(messages, current_tokens=100_000)
            )
            assert finished.wait(timeout=0.20)
            assert result.get("error") is None
            assert result.get("value") is messages
            assert engine._store.get_session_count("session") == 0
        caller.join(timeout=2)
        assert not caller.is_alive()
    finally:
        engine.shutdown()


def test_late_auxiliary_recheck_never_waits_past_wall_for_store_count(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "store-count input remains raw"}]
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        with _held_lock(engine._store._write_lock):
            caller, finished, result = _invoke(
                lambda: engine.compress(messages, current_tokens=100_000)
            )
            assert finished.wait(timeout=0.20)
            assert result.get("error") is None
            assert result.get("value") is messages
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert engine._store.get_session_count("session") == 0
    finally:
        engine.shutdown()


def test_background_enqueue_bypass_recheck_uses_shared_deadline(tmp_path):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "enqueue classification stays bounded"}]
    try:
        with _held_lock(engine._auxiliary_session_lock):
            caller, finished, result = _invoke(
                lambda: engine._maybe_enqueue_background_compaction(
                    messages,
                    observed_tokens=100_000,
                    deadline=time.monotonic() + 0.05,
                )
            )
            assert finished.wait(timeout=0.20)
            assert isinstance(result.get("error"), TimeoutError)
        caller.join(timeout=2)
        assert not caller.is_alive()
    finally:
        engine.shutdown()


def test_automatic_context_sanitizer_is_detached_from_foreground_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    messages = [{"role": "user", "content": "sanitizer input remains authoritative"}]
    entered = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}

    def blocked_sanitizer(values, **kwargs):
        observed.update(kwargs)
        entered.set()
        assert release.wait(timeout=3)
        return list(values)

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(engine, "_sanitize_active_context_messages", blocked_sanitizer)
    caller, finished, result = _invoke(
        lambda: engine.compress(messages, current_tokens=10_000)
    )
    try:
        assert entered.wait(timeout=2)
        assert finished.wait(timeout=0.20)
        assert result.get("error") is None
        assert result.get("value") is messages
        assert observed.get("cancel_event") is not None
        assert observed["cancel_event"].is_set()
    finally:
        release.set()
        caller.join(timeout=2)
        engine.shutdown()


def test_bypass_host_constructor_is_detached_from_automatic_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    engine._session_stateless = True
    engine.threshold_tokens = 1
    messages = [{"role": "user", "content": "preserve while host setup is blocked"}]
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    constructor_returned = threading.Event()

    class BlockingContextCompressor:
        compression_count = 0

        def __init__(self, *_args, **_kwargs):
            constructor_entered.set()
            assert release_constructor.wait(timeout=3)
            constructor_returned.set()

        def update_model(self, **_kwargs):
            return None

        def on_session_start(self, *_args, **_kwargs):
            return None

        def compress(self, values, **_kwargs):
            return list(values)

    monkeypatch.setitem(
        sys.modules,
        "agent.context_compressor",
        SimpleNamespace(ContextCompressor=BlockingContextCompressor),
    )
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    caller, finished, result = _invoke(
        lambda: engine.compress(messages, current_tokens=100_000)
    )
    try:
        assert constructor_entered.wait(timeout=2)
        assert finished.wait(timeout=0.20)
        assert result.get("error") is None
        assert result.get("value") is messages
        assert engine._host_fallback_compressor is None
        release_constructor.set()
        assert constructor_returned.wait(timeout=2)
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert engine._host_fallback_compressor is None
    finally:
        release_constructor.set()
        caller.join(timeout=2)
        engine.shutdown()


def test_preflight_and_compress_share_one_automatic_wall(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    clock = _Clock()
    engine._foreground_monotonic = clock
    messages = [{"role": "user", "content": "one admission, one wall"}]
    captured: dict[str, object] = {}

    def enqueue(_messages, _observed_tokens, **_kwargs):
        clock.advance(10.0)
        return None

    def compress_impl(values, **kwargs):
        captured.update(kwargs)
        return values

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        15.0,
    )
    monkeypatch.setattr(engine, "_maybe_enqueue_background_compaction", enqueue)
    monkeypatch.setattr(engine, "_background_preflight_requested", lambda *_args: True)
    monkeypatch.setattr(engine, "_compress_impl", compress_impl)
    try:
        assert engine.should_compress_preflight(messages) is True
        assert engine.compress(messages) is messages
        assert captured["foreground_deadline"] == pytest.approx(115.0)
    finally:
        engine.shutdown()


def test_expired_deadline_records_status_but_starts_no_durable_diagnostic(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _Clock()
    engine._foreground_monotonic = clock
    background = engine._background_compaction
    assert background is not None
    durable_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        background,
        "record_diagnostic_event",
        lambda *_args, **kwargs: durable_calls.append(kwargs),
    )
    messages = [{"role": "user", "content": "preserved"}]
    try:
        assert engine._record_foreground_deadline_exhaustion(
            messages,
            ForegroundDeadlineExceeded("setup", "expired"),
            started_at=99.0,
            deadline=99.5,
        ) is messages
        assert engine.last_compression_status == "noop"
        assert durable_calls == []
    finally:
        engine.shutdown()


def test_emergency_permit_acquire_rolls_back_when_commit_crosses_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    helper = engine._background_compaction
    assert helper is not None
    fence_generation = engine._reconcile_background_publication_fence()
    assert fence_generation is not None
    clock = _Clock()
    engine._foreground_monotonic = clock
    monkeypatch.setattr(
        background_compaction_module,
        "_current_linux_process_identity",
        lambda: (4321, "123", "test-boot"),
    )
    connection = helper.connection
    assert connection is not None
    connection.create_function(
        "expire_permit_acquire_deadline",
        0,
        lambda: clock.advance(1.0),
    )
    connection.execute(
        """
        CREATE TEMP TRIGGER expire_permit_acquire_after_insert
        AFTER INSERT ON lcm_compaction_provider_leases
        BEGIN
          SELECT expire_permit_acquire_deadline();
        END
        """
    )
    try:
        with pytest.raises(ForegroundProviderPermitUnavailable):
            engine._acquire_foreground_emergency_provider_permit(
                deadline=100.5,
                expected_session_fence_generation=fence_generation,
            )
        assert connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases"
        ).fetchone()[0] == 0
        assert not hasattr(
            engine._thread_context,
            "foreground_emergency_provider_permit",
        )
    finally:
        engine.shutdown()


def test_provider_permit_release_rolls_back_when_commit_crosses_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    helper = engine._background_compaction
    assert helper is not None
    clock = _Clock()
    monkeypatch.setattr(
        background_compaction_module,
        "_current_linux_process_identity",
        lambda: (4321, "123", "test-boot"),
    )
    permit = helper.acquire_emergency_provider_permit(
        conversation_id="conversation",
        owner="foreground-owner",
        global_limit=1,
        now=100.0,
    )
    assert permit is not None
    connection = helper.connection
    assert connection is not None
    connection.create_function(
        "expire_permit_release_deadline",
        0,
        lambda: clock.advance(1.0),
    )
    connection.execute(
        """
        CREATE TEMP TRIGGER expire_permit_release_after_delete
        AFTER DELETE ON lcm_compaction_provider_leases
        BEGIN
          SELECT expire_permit_release_deadline();
        END
        """
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            helper.release_provider_permit(
                permit.job_id,
                owner=permit.lease_owner,
                generation=permit.lease_generation,
                busy_timeout_ms=0,
                lock_timeout_seconds=0.0,
                write_deadline=100.5,
                deadline_clock=clock,
            )
        assert connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases WHERE job_id=?",
            (permit.job_id,),
        ).fetchone()[0] == 1
    finally:
        engine.shutdown()


def test_expired_compress_finally_transfers_cleanup_without_waiting(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _Clock()
    engine._foreground_monotonic = clock
    messages = [{"role": "user", "content": "caller context"}]
    release_entered = threading.Event()
    unblock_release = threading.Event()
    guardian_started = threading.Event()
    release_kwargs: list[dict[str, object]] = []

    class PermitStore:
        def release_provider_permit(self, *_args, **kwargs):
            release_kwargs.append(kwargs)
            release_entered.set()
            assert unblock_release.wait(timeout=3)
            return True

        def _start_provider_permit_guardian(self, *_args, **_kwargs):
            guardian_started.set()
            return True

    permit_store = PermitStore()
    permit = SimpleNamespace(
        job_id="permit-job",
        lease_owner="permit-owner",
        lease_generation=7,
    )

    def expire_during_work(*_args, **_kwargs):
        engine._thread_context.foreground_emergency_provider_permit = {
            "permit": permit,
            "permit_store": permit_store,
            "guardian_owned": False,
        }
        clock.advance(1.0)
        raise ForegroundDeadlineExceeded("summarization", "expired")

    monkeypatch.setattr(engine, "_compress_impl", expire_during_work)
    engine._background_compaction = None
    caller, finished, result = _invoke(lambda: engine.compress(messages, deadline=100.5))
    try:
        assert finished.wait(timeout=0.20)
        assert result.get("error") is None
        assert result.get("value") is messages
        assert not release_entered.is_set()
        assert release_kwargs == []
        assert guardian_started.is_set()
        assert not hasattr(
            engine._thread_context,
            "foreground_emergency_provider_permit",
        )
    finally:
        unblock_release.set()
        caller.join(timeout=2)
        engine.shutdown()


def test_manual_cleanup_without_deadline_still_releases_inline(tmp_path):
    engine = _engine(tmp_path)
    releases: list[tuple[str, str, int]] = []
    guardians: list[str] = []

    class PermitStore:
        def release_provider_permit(self, job_id, *, owner, generation, **_kwargs):
            releases.append((job_id, owner, generation))
            return True

        def _start_provider_permit_guardian(self, *_args, **_kwargs):
            guardians.append("guardian")
            return True

    engine._thread_context.foreground_emergency_provider_permit = {
        "permit": SimpleNamespace(
            job_id="manual-permit",
            lease_owner="manual-owner",
            lease_generation=3,
        ),
        "permit_store": PermitStore(),
        "guardian_owned": False,
    }
    try:
        engine._release_foreground_emergency_provider_permit(deadline=None)
        assert releases == [("manual-permit", "manual-owner", 3)]
        assert guardians == []
    finally:
        engine.shutdown()
