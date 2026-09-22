"""Whole-path deadline and lossless fail-safe contracts for foreground LCM."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    ForegroundProviderPermitUnavailable,
)
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm import escalation


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _config(db_path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1_000_000,
        context_reserve_tokens=100_000,
        reconcile_budget_seconds=45.0,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
    )


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(_config(tmp_path / "deadline.db"))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    return engine


def test_deadline_starts_before_initial_token_count_and_preserves_input(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    messages = [{"role": "user", "content": "preserve exactly"}]
    before = [dict(message) for message in messages]

    def consume_budget(_messages):
        clock.advance(0.051)
        return 1

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "hermes_lcm.compaction.count_messages_tokens",
        consume_budget,
    )
    try:
        result = engine.compress(messages, current_tokens=100_000)
        assert result is messages
        assert messages == before
        assert engine._store.get_session_count("s") == 0
        assert engine.last_compression_status == "noop"
        assert engine.last_compression_noop_reason == (
            "foreground deadline exhausted during token counting"
        )
        assert engine._last_foreground_deadline["stage"] == "token counting"
        # Once the wall is exhausted, even optional diagnostic writes are
        # deferred/skipped rather than extending the foreground request.
        assert (
            engine.get_background_compaction_status()["durable_event_counts"].get(
                "foreground_deadline_exhausted",
                0,
            )
            == 0
        )
    finally:
        engine.shutdown()


def test_reconcile_lock_uses_caller_wall_not_configured_45_seconds(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    engine._ingest_cursor_needs_reconcile = True
    observed: dict[str, float] = {}

    class _ExpiringLock:
        def acquire(self, *, timeout=None):
            observed["timeout"] = float(timeout)
            clock.advance(float(timeout))
            return False

        def release(self):  # pragma: no cover - failed acquisition never releases
            raise AssertionError("unacquired lock released")

    engine._reconcile_lock = _ExpiringLock()
    messages = [{"role": "user", "content": "raw remains authoritative"}]
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        result = engine.compress(messages, current_tokens=100_000)
        assert result is messages
        assert 0 < observed["timeout"] <= 0.05
        assert engine._ingest_cursor == 0
        assert engine._ingest_cursor_needs_reconcile is True
        assert engine._store.get_session_count("s") == 0
        assert engine.last_compression_noop_reason == (
            "foreground deadline exhausted during reconciliation"
        )
    finally:
        engine._reconcile_lock = threading.Lock()
        engine.shutdown()


def test_sqlite_write_wait_is_bounded_and_append_remains_atomic(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    blocker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    messages = [
        {"role": "user", "content": "row one"},
        {"role": "assistant", "content": "row two"},
    ]
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        started = time.monotonic()
        result = engine.compress(messages, current_tokens=100_000)
        elapsed = time.monotonic() - started
        assert result is messages
        assert elapsed < 0.20
        assert engine._ingest_cursor == 0
        assert engine._store.get_session_count("s") == 0
        assert engine.last_compression_noop_reason == (
            "foreground deadline exhausted during ingest"
        )
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
        engine.shutdown()


def test_deadline_after_atomic_ingest_keeps_raw_rows_and_original_context(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    messages = [
        {"role": "user", "content": "durable one"},
        {"role": "assistant", "content": "durable two"},
    ]

    def consume_remaining_budget(*_args, **_kwargs):
        clock.advance(0.051)

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        engine,
        "_maybe_enqueue_background_compaction",
        consume_remaining_budget,
    )
    try:
        result = engine.compress(messages, current_tokens=100_000)
        assert result is messages
        assert engine._store.get_session_count("s") == len(messages)
        assert engine._ingest_cursor == len(messages)
        rows = engine._store.get_session_messages("s")
        assert [row["content"] for row in rows] == [
            "durable one",
            "durable two",
        ]
        assert engine.last_compression_noop_reason == (
            "foreground deadline exhausted during background enqueue"
        )
    finally:
        engine.shutdown()


def test_slow_context_reconstruction_uses_entry_deadline_and_preserves_context(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    messages = [{"role": "user", "content": "durable source remains intact"}]
    before = [dict(message) for message in messages]
    observed: dict[str, float | None] = {}
    original_assemble = engine._assemble_context

    def deliberately_slow_assemble(*args, deadline=None, **kwargs):
        observed["deadline"] = deadline
        clock.advance(0.051)
        if deadline is None:
            return original_assemble(*args, **kwargs)
        return original_assemble(*args, deadline=deadline, **kwargs)

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        engine,
        "_should_force_overflow_recovery",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        engine,
        "_maybe_enqueue_background_compaction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_assemble_context", deliberately_slow_assemble)
    try:
        result = engine.compress(messages, current_tokens=100_000)

        assert result is messages
        assert messages == before
        assert observed["deadline"] == pytest.approx(100.05)
        assert engine._store.get_session_count("s") == 1
        assert engine._store.get_session_messages("s")[0]["content"] == before[0]["content"]
        assert engine.last_compression_noop_reason == (
            "foreground deadline exhausted during context reconstruction"
        )
    finally:
        engine.shutdown()


def test_bypassed_host_compressor_cannot_escape_the_foreground_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    call_finished = threading.Event()
    result_box: dict[str, object] = {}

    class BlockingHostCompressor:
        compression_count = 0
        _last_compress_aborted = False
        _last_summary_error = None
        _last_summary_auth_failure = False
        _last_summary_network_failure = False
        _last_summary_dropped_count = 0
        _last_summary_fallback_used = False
        _last_aux_model_failure_error = None
        _last_aux_model_failure_model = None

        def compress(self, messages, **_kwargs):
            provider_entered.set()
            assert release_provider.wait(timeout=2)
            return list(messages[:1])

    messages = [{"role": "user", "content": "preserve this caller context"}]
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        engine,
        "_bypasses_lcm_context_management",
        lambda: True,
    )
    monkeypatch.setattr(
        engine,
        "_get_host_fallback_compressor",
        lambda: BlockingHostCompressor(),
    )
    engine.threshold_tokens = 1

    def compress() -> None:
        try:
            result_box["value"] = engine.compress(messages, current_tokens=100_000)
        except BaseException as exc:  # pragma: no cover - reported below
            result_box["error"] = exc
        finally:
            call_finished.set()

    thread = threading.Thread(target=compress)
    try:
        thread.start()
        assert provider_entered.wait(timeout=2)
        completed_within_wall = call_finished.wait(timeout=0.15)
        release_provider.set()
        thread.join(timeout=2)

        assert completed_within_wall
        assert result_box.get("error") is None
        assert result_box.get("value") is messages
        assert engine.last_compression_noop_reason == (
            "foreground deadline exhausted during bypass compression"
        )
    finally:
        release_provider.set()
        thread.join(timeout=2)
        engine.shutdown()


def test_bypassed_host_compressor_receives_cancel_and_is_detached_after_timeout(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    provider_entered = threading.Event()
    provider_returned = threading.Event()
    seen: dict[str, object] = {}

    class BlockingHostCompressor:
        compression_count = 0
        _last_compress_aborted = False
        _last_summary_error = None
        _last_summary_auth_failure = False
        _last_summary_network_failure = False
        _last_summary_dropped_count = 0
        _last_summary_fallback_used = False
        _last_aux_model_failure_error = None
        _last_aux_model_failure_model = None

        def compress(
            self,
            messages,
            *,
            current_tokens=None,
            focus_topic=None,
            force=False,
            deadline=None,
            cancel_event=None,
        ):
            del current_tokens, focus_topic, force
            seen.update(deadline=deadline, cancel_event=cancel_event)
            provider_entered.set()
            assert cancel_event is not None
            assert cancel_event.wait(timeout=2)
            provider_returned.set()
            return list(messages[:1])

    compressor = BlockingHostCompressor()
    messages = [{"role": "user", "content": "preserve this caller context"}]
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(engine, "_bypasses_lcm_context_management", lambda: True)
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: compressor)
    engine._host_fallback_compressor = compressor
    engine._host_fallback_session_id = "s"
    engine.threshold_tokens = 1
    try:
        result = engine.compress(messages, current_tokens=100_000)
        assert result is messages
        assert provider_entered.is_set()
        assert provider_returned.wait(timeout=2)
        assert isinstance(seen.get("cancel_event"), threading.Event)
        assert engine._host_fallback_compressor is None
        assert engine._host_fallback_session_id == ""
    finally:
        cancel_event = seen.get("cancel_event")
        if isinstance(cancel_event, threading.Event):
            cancel_event.set()
        engine.shutdown()


def test_condensation_provider_cannot_escape_the_foreground_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    call_finished = threading.Event()
    result_box: dict[str, object] = {}
    nodes = [
        SummaryNode(
            session_id="s",
            depth=0,
            summary=f"source summary {ordinal}",
            token_count=10,
            source_token_count=100,
            source_ids=[],
            source_type="messages",
        )
        for ordinal in range(2)
    ]
    for node in nodes:
        engine._dag.add_node(node)

    def blocking_summary(**_kwargs):
        provider_entered.set()
        assert release_provider.wait(timeout=2)
        return "late condensation", 1

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        blocking_summary,
    )

    def condense() -> None:
        try:
            result_box["value"] = engine._condense_summary_nodes(
                nodes,
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            call_finished.set()

    thread = threading.Thread(target=condense)
    try:
        thread.start()
        assert provider_entered.wait(timeout=2)
        completed_within_wall = call_finished.wait(timeout=0.15)
        release_provider.set()
        thread.join(timeout=2)

        assert completed_within_wall
        assert isinstance(result_box.get("error"), TimeoutError)
        assert engine._dag.get_session_node_count("s") == 2
    finally:
        release_provider.set()
        thread.join(timeout=2)
        engine.shutdown()


def test_condensation_rejects_a_result_completed_after_the_absolute_deadline(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    nodes = [
        SummaryNode(
            session_id="s",
            depth=0,
            summary=f"source summary {ordinal}",
            token_count=10,
            source_token_count=100,
            source_ids=[],
            source_type="messages",
        )
        for ordinal in range(2)
    ]
    for node in nodes:
        engine._dag.add_node(node)

    def late_summary(**_kwargs):
        clock.advance(1.0)
        return "late condensation", 1

    monkeypatch.setattr("hermes_lcm.engine.time.monotonic", clock)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        late_summary,
    )
    try:
        with pytest.raises(TimeoutError):
            engine._condense_summary_nodes(nodes, deadline=100.5)
        assert engine._dag.get_session_node_count("s") == 2
    finally:
        engine.shutdown()


def test_optional_extraction_cannot_escape_its_foreground_budget(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    provider_entered = threading.Event()
    release_provider = threading.Event()
    call_finished = threading.Event()
    errors: list[BaseException] = []

    def blocking_extraction(**_kwargs):
        provider_entered.set()
        assert release_provider.wait(timeout=2)
        return True

    monkeypatch.setattr(
        "hermes_lcm.engine.extract_before_compaction",
        blocking_extraction,
    )

    def extract() -> None:
        try:
            engine._run_pre_compaction_extraction(
                [{"role": "user", "content": "extract without blocking"}],
                timeout_seconds=0.05,
            )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            call_finished.set()

    thread = threading.Thread(target=extract)
    try:
        thread.start()
        assert provider_entered.wait(timeout=2)
        completed_within_budget = call_finished.wait(timeout=0.15)
        release_provider.set()
        thread.join(timeout=2)

        assert completed_within_budget
        assert errors == []
    finally:
        release_provider.set()
        thread.join(timeout=2)
        engine.shutdown()


def test_extraction_uses_the_original_deadline_before_serialization(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    provider_entered = threading.Event()

    def slow_serialization(_messages):
        clock.advance(0.6)
        return "serialized"

    def extraction_provider(**_kwargs):
        provider_entered.set()
        return True

    monkeypatch.setattr(engine, "_serialize_messages", slow_serialization)
    monkeypatch.setattr(
        "hermes_lcm.engine.extract_before_compaction",
        extraction_provider,
    )
    try:
        engine._run_pre_compaction_extraction(
            [{"role": "user", "content": "deadline is already running"}],
            deadline=100.5,
        )
        assert not provider_entered.is_set()
    finally:
        engine.shutdown()


def test_late_extraction_result_cannot_persist_after_the_deadline(
    tmp_path,
    monkeypatch,
):
    import hermes_lcm.extraction as extraction_module

    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    output_dir = tmp_path / "late-extractions"
    engine._config.extraction_output_path = str(output_dir)

    def late_extraction(_prompt, model="", timeout=None):
        del model, timeout
        clock.advance(1.0)
        return "- This late result must never reach disk"

    monkeypatch.setattr(extraction_module, "_call_extraction_llm", late_extraction)
    try:
        engine._run_pre_compaction_extraction(
            [{"role": "user", "content": "do not persist late output"}],
            deadline=100.5,
        )
        assert not list(output_dir.glob("*.md"))
    finally:
        engine.shutdown()


def test_extraction_cancelled_after_directory_creation_never_appends(
    tmp_path,
    monkeypatch,
):
    import hermes_lcm.extraction as extraction_module

    mkdir_entered = threading.Event()
    release_mkdir = threading.Event()
    worker_finished = threading.Event()
    cancel_event = threading.Event()
    result_box: dict[str, object] = {}
    output_dir = tmp_path / "cancelled-extractions"
    original_mkdir = extraction_module.Path.mkdir

    def barrier_mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        mkdir_entered.set()
        assert release_mkdir.wait(timeout=2)
        return result

    monkeypatch.setattr(extraction_module.Path, "mkdir", barrier_mkdir)
    monkeypatch.setattr(
        extraction_module,
        "_call_extraction_llm",
        lambda *_args, **_kwargs: "- Never append this cancelled result",
    )

    def extract() -> None:
        try:
            result_box["value"] = extraction_module.extract_before_compaction(
                "serialized",
                output_path=str(output_dir),
                cancel_event=cancel_event,
            )
        finally:
            worker_finished.set()

    worker = threading.Thread(target=extract)
    try:
        worker.start()
        assert mkdir_entered.wait(timeout=2)
        cancel_event.set()
        release_mkdir.set()
        assert worker_finished.wait(timeout=2)
        assert result_box.get("value") is False
        assert not list(output_dir.glob("*.md"))
    finally:
        cancel_event.set()
        release_mkdir.set()
        worker.join(timeout=2)


def test_engine_extraction_persistence_is_async_and_drops_results_late_after_mkdir(
    tmp_path,
    monkeypatch,
):
    import hermes_lcm.extraction as extraction_module

    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    output_dir = tmp_path / "bounded-extractions"
    engine._config.extraction_output_path = str(output_dir)
    mkdir_entered = threading.Event()
    release_mkdir = threading.Event()
    call_finished = threading.Event()
    original_mkdir = extraction_module.Path.mkdir

    def barrier_mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        mkdir_entered.set()
        assert release_mkdir.wait(timeout=2)
        return result

    monkeypatch.setattr(extraction_module.Path, "mkdir", barrier_mkdir)
    monkeypatch.setattr(
        extraction_module,
        "_call_extraction_llm",
        lambda *_args, **_kwargs: "- Timely provider result",
    )

    def extract() -> None:
        try:
            engine._run_pre_compaction_extraction(
                [{"role": "user", "content": "bounded file persistence"}],
                deadline=100.5,
            )
        finally:
            call_finished.set()

    caller = threading.Thread(target=extract)
    try:
        caller.start()
        assert mkdir_entered.wait(timeout=2)
        assert call_finished.wait(timeout=0.2)
        clock.advance(1.0)
        release_mkdir.set()
        assert engine._extraction_persistence_idle.wait(timeout=2)
        assert not list(output_dir.glob("*.md"))
    finally:
        clock.advance(1.0)
        release_mkdir.set()
        caller.join(timeout=2)
        engine.shutdown()


def test_hard_deadline_calls_are_single_flight_after_a_stuck_timeout(
    tmp_path,
    monkeypatch,
):
    import hermes_lcm.compaction as compaction_module

    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    first_entered = threading.Event()
    release_first = threading.Event()
    first_exited = threading.Event()
    later_calls: list[str] = []
    real_thread = threading.Thread

    class EntryHandshakingWorker:
        def __init__(self, *, target, name, daemon):
            self._thread = real_thread(target=target, name=name, daemon=daemon)

        def start(self):
            self._thread.start()
            assert first_entered.wait(timeout=2)
            # Expire the absolute deadline only after the operation has entered,
            # so suite load cannot turn this into a worker-scheduling test.
            clock.advance(2.0)

    def stuck_call():
        first_entered.set()
        try:
            assert release_first.wait(timeout=3)
            return "late"
        finally:
            first_exited.set()

    def forbidden_later_call():
        later_calls.append("launched")
        return "forbidden"

    monkeypatch.setattr(
        compaction_module.threading,
        "Thread",
        EntryHandshakingWorker,
    )
    try:
        with pytest.raises(TimeoutError):
            engine._call_with_hard_foreground_deadline(
                stuck_call,
                deadline=clock() + 1.0,
                timeout_error=TimeoutError("first deadline"),
            )
        assert first_entered.is_set()

        for _ in range(3):
            with pytest.raises(TimeoutError):
                engine._call_with_hard_foreground_deadline(
                    forbidden_later_call,
                    deadline=clock() + 1.0,
                    timeout_error=TimeoutError("single-flight occupied"),
                )
        assert later_calls == []
    finally:
        release_first.set()
        assert first_exited.wait(timeout=2)
        monkeypatch.setattr(compaction_module.threading, "Thread", real_thread)
        engine.shutdown()


def test_hard_deadline_worker_cancelled_before_scheduling_never_enters_operation(
    tmp_path,
    monkeypatch,
):
    import hermes_lcm.compaction as compaction_module

    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    release_worker = threading.Event()
    worker_exited = threading.Event()
    operation_entered = threading.Event()
    cancel_event = threading.Event()
    real_thread = threading.Thread

    class DelayedWorker:
        def __init__(self, *, target, name, daemon):
            del name

            def wait_then_run():
                try:
                    assert release_worker.wait(timeout=2)
                    target()
                finally:
                    worker_exited.set()

            self._thread = real_thread(target=wait_then_run, daemon=daemon)

        def start(self):
            self._thread.start()
            # Deterministically expire the caller's absolute deadline while the
            # newly-created worker is still held behind the scheduling barrier.
            clock.advance(2.0)

    monkeypatch.setattr(compaction_module.threading, "Thread", DelayedWorker)
    try:
        with pytest.raises(TimeoutError, match="delayed worker deadline"):
            engine._call_with_hard_foreground_deadline(
                lambda: operation_entered.set(),
                deadline=101.0,
                timeout_error=TimeoutError("delayed worker deadline"),
                cancel_event=cancel_event,
            )

        assert cancel_event.is_set()
        assert not operation_entered.is_set()
        release_worker.set()
        assert worker_exited.wait(timeout=2)
        assert not operation_entered.is_set()

        # The cancelled worker must still run its finalizer and return the
        # engine's single-flight slot for the next foreground operation.
        monkeypatch.setattr(compaction_module.threading, "Thread", real_thread)
        assert engine._call_with_hard_foreground_deadline(
            lambda: "next call",
            deadline=clock() + 1.0,
            timeout_error=TimeoutError("slot leaked"),
        ) == "next call"
    finally:
        release_worker.set()
        worker_exited.wait(timeout=2)
        monkeypatch.setattr(compaction_module.threading, "Thread", real_thread)
        engine.shutdown()


def test_hard_deadline_wait_interruption_signals_live_worker_cancellation(
    tmp_path,
    monkeypatch,
):
    import hermes_lcm.compaction as compaction_module

    engine = _engine(tmp_path)
    operation_entered = threading.Event()
    release_operation = threading.Event()
    worker_exited = threading.Event()
    cancel_event = threading.Event()
    real_event = threading.Event
    real_thread = threading.Thread

    class InterruptedWaitEvent:
        def __init__(self):
            self._event = real_event()

        def is_set(self):
            return self._event.is_set()

        def set(self):
            self._event.set()

        def wait(self, timeout=None):
            del timeout
            raise KeyboardInterrupt("foreground caller interrupted")

    class CoordinatedThread:
        def __init__(self, *, target, name, daemon):
            self._thread = real_thread(target=target, name=name, daemon=daemon)

        def start(self):
            self._thread.start()
            assert operation_entered.wait(timeout=2)

    class InterruptedThreading:
        Event = InterruptedWaitEvent
        Thread = CoordinatedThread

    def operation():
        operation_entered.set()
        try:
            assert release_operation.wait(timeout=2)
            return "late result"
        finally:
            worker_exited.set()

    monkeypatch.setattr(compaction_module, "threading", InterruptedThreading)
    try:
        with pytest.raises(KeyboardInterrupt, match="foreground caller interrupted"):
            engine._call_with_hard_foreground_deadline(
                operation,
                deadline=time.monotonic() + 1.0,
                timeout_error=TimeoutError("foreground wall"),
                cancel_event=cancel_event,
            )
        assert cancel_event.is_set()
    finally:
        release_operation.set()
        assert worker_exited.wait(timeout=2)
        engine.shutdown()


def test_leaf_provider_shares_the_foreground_single_flight_slot(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    leaf_entered = threading.Event()

    def stuck_call():
        first_entered.set()
        assert release_first.wait(timeout=3)

    def forbidden_leaf(**_kwargs):
        leaf_entered.set()
        return "forbidden leaf", 1

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        forbidden_leaf,
    )
    try:
        with pytest.raises(TimeoutError):
            engine._call_with_hard_foreground_deadline(
                stuck_call,
                deadline=time.monotonic() + 0.05,
                timeout_error=TimeoutError("first deadline"),
            )
        assert first_entered.is_set()
        with pytest.raises(ForegroundProviderPermitUnavailable):
            engine._summarize_leaf_chunk_with_rescue(
                [{"role": "user", "content": "do not launch a second provider"}],
                deadline=time.monotonic() + 1.0,
            )
        assert not leaf_entered.is_set()
    finally:
        release_first.set()
        engine.shutdown()


def test_leaf_worker_scheduled_after_timeout_never_enters_provider(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    deferred_threads: list[threading.Thread] = []
    provider_entered = threading.Event()
    original_start = threading.Thread.start

    def defer_leaf_start(thread):
        if thread.name == "lcm-foreground-summary-deadline":
            deferred_threads.append(thread)
            return None
        return original_start(thread)

    def forbidden_provider(**_kwargs):
        provider_entered.set()
        return "forbidden", 1

    monkeypatch.setattr(threading.Thread, "start", defer_leaf_start)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        forbidden_provider,
    )
    try:
        with pytest.raises(TimeoutError):
            engine._summarize_leaf_chunk_with_rescue(
                [{"role": "user", "content": "never launch after timeout"}],
                deadline=time.monotonic() + 0.05,
            )
        assert len(deferred_threads) == 1
        deferred_threads[0].run()
        assert not provider_entered.is_set()
    finally:
        engine.shutdown()


def test_foreground_permit_completion_state_is_isolated_per_attempt(tmp_path):
    engine = _engine(tmp_path)
    permit_state = {
        "completion_lock": threading.Lock(),
        "provider_finished": False,
        "start_failure_finalizer": None,
        "start_failure_claimed": False,
        "completed": None,
    }
    first_completed = threading.Event()
    second_completed = threading.Event()
    try:
        first_attempt = engine._prepare_foreground_provider_attempt(
            permit_state,
            first_completed,
        )
        first_completed.set()
        second_attempt = engine._prepare_foreground_provider_attempt(
            permit_state,
            second_completed,
        )
        engine._mark_foreground_provider_finished(first_attempt)

        assert first_attempt["provider_finished"] is True
        assert second_attempt["provider_finished"] is False
        assert permit_state["active_attempt"] is second_attempt
    finally:
        engine.shutdown()


def test_extraction_keeps_a_strict_legacy_callable_compatible(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    entered = threading.Event()

    def legacy_extraction(
        serialized_messages,
        output_path,
        session_id="",
        model="",
        timeout=None,
    ):
        del serialized_messages, output_path, session_id, model, timeout
        entered.set()
        return True

    monkeypatch.setattr(
        "hermes_lcm.engine.extract_before_compaction",
        legacy_extraction,
    )
    try:
        engine._run_pre_compaction_extraction(
            [{"role": "user", "content": "legacy callable"}],
            timeout_seconds=0.5,
        )
        assert entered.is_set()
    finally:
        engine.shutdown()


def test_foreground_guardian_start_failure_transfers_one_shot_release(
    tmp_path,
):
    engine = _engine(tmp_path)
    completed = threading.Event()
    release_calls: list[str] = []

    class _Permit:
        job_id = "permit-job"
        lease_owner = "permit-owner"
        lease_generation = 7

    class _Helper:
        def _start_provider_permit_guardian(
            self,
            _completed,
            *,
            job_id,
            owner,
            generation,
            on_start_failure,
        ):
            assert (job_id, owner, generation) == (
                "permit-job",
                "permit-owner",
                7,
            )
            on_start_failure(lambda: release_calls.append("released"))
            return False

    state = {
        "permit": _Permit(),
        "guardian_owned": False,
        "completion_lock": threading.Lock(),
        "provider_finished": False,
        "start_failure_finalizer": None,
        "start_failure_claimed": False,
        "completed": completed,
    }
    engine._background_compaction = _Helper()
    engine._thread_context.foreground_emergency_provider_permit = state
    try:
        engine._handoff_foreground_emergency_provider_permit(completed)
        assert release_calls == []
        engine._mark_foreground_provider_finished(state)
        engine._mark_foreground_provider_finished(state)
        assert release_calls == ["released"]
    finally:
        engine.shutdown()


def test_foreground_permit_release_uses_captured_store_after_runtime_rebind(
    tmp_path,
):
    engine = _engine(tmp_path)
    release_calls: list[tuple[str, str, int]] = []

    class _Permit:
        job_id = "permit-job"
        lease_owner = "permit-owner"
        lease_generation = 7

    class _CapturedStore:
        def release_provider_permit(
            self,
            job_id,
            *,
            owner,
            generation,
            busy_timeout_ms,
        ):
            assert busy_timeout_ms == 0
            release_calls.append((job_id, owner, generation))
            return True

    captured_store = _CapturedStore()
    engine._thread_context.foreground_emergency_provider_permit = {
        "permit": _Permit(),
        "permit_store": captured_store,
        "guardian_owned": False,
    }
    engine._background_compaction = None
    try:
        engine._release_foreground_emergency_provider_permit()
        assert release_calls == [("permit-job", "permit-owner", 7)]
    finally:
        engine.shutdown()


def test_emergency_permit_python_lock_wait_respects_foreground_deadline(
    tmp_path,
):
    engine = _engine(tmp_path)
    helper = engine._background_compaction
    assert helper is not None
    fence_generation = engine._reconcile_background_publication_fence()
    assert fence_generation is not None
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_python_lock() -> None:
        with helper._lock:
            lock_held.set()
            assert release_lock.wait(timeout=3)

    def acquire_permit() -> None:
        try:
            result_box["value"] = engine._acquire_foreground_emergency_provider_permit(
                deadline=time.monotonic() + 0.05,
                expected_session_fence_generation=fence_generation,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    holder = threading.Thread(target=hold_python_lock)
    caller = threading.Thread(target=acquire_permit)
    try:
        holder.start()
        assert lock_held.wait(timeout=2)
        caller.start()
        assert caller_finished.wait(timeout=0.2)
        assert isinstance(
            result_box.get("error"),
            ForegroundProviderPermitUnavailable,
        )
        assert "value" not in result_box
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_foreground_permit_release_never_waits_for_store_python_lock(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    helper = engine._background_compaction
    assert helper is not None
    fence_generation = engine._reconcile_background_publication_fence()
    assert fence_generation is not None
    permit = helper.acquire_emergency_provider_permit(
        conversation_id="c",
        owner="foreground-release-owner",
        session_id="s",
        expected_session_fence_generation=fence_generation,
        lease_seconds=30,
        global_limit=1,
    )
    assert permit is not None

    lock_held = threading.Event()
    release_lock = threading.Event()
    foreground_started = threading.Event()
    foreground_finished = threading.Event()
    permit_released = threading.Event()
    result_box: dict[str, BaseException] = {}
    original_release = BackgroundCompactionStore.release_provider_permit

    def observed_release(self, *args, **kwargs):
        result = original_release(self, *args, **kwargs)
        permit_released.set()
        return result

    monkeypatch.setattr(
        BackgroundCompactionStore,
        "release_provider_permit",
        observed_release,
    )

    def hold_python_lock() -> None:
        with helper._lock:
            lock_held.set()
            assert release_lock.wait(timeout=3)

    def release_from_foreground() -> None:
        engine._thread_context.foreground_emergency_provider_permit = {
            "permit": permit,
            "permit_store": helper,
            "guardian_owned": False,
        }
        foreground_started.set()
        try:
            engine._release_foreground_emergency_provider_permit()
        except BaseException as exc:  # pragma: no cover - reported below
            result_box["error"] = exc
        finally:
            foreground_finished.set()

    holder = threading.Thread(target=hold_python_lock)
    foreground = threading.Thread(target=release_from_foreground)
    try:
        holder.start()
        assert lock_held.wait(timeout=2)
        foreground.start()
        assert foreground_started.wait(timeout=2)

        # The foreground path must finish while the process-local store lock is
        # still held.  Durable cleanup transfers to the independent guardian.
        assert foreground_finished.wait(timeout=1)
        assert not release_lock.is_set()
        assert result_box == {}
        assert permit_released.wait(timeout=2)
    finally:
        release_lock.set()
        holder.join(timeout=2)
        foreground.join(timeout=2)
        engine.shutdown()


def test_timeout_exception_text_is_not_exposed_in_deadline_status(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = [100.0]
    marker = "sensitive-marker-must-not-appear"
    engine._foreground_monotonic = lambda: clock[0]

    def timeout_after_deadline(*_args, **_kwargs):
        clock[0] = 200.0
        raise TimeoutError(marker)

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        15.0,
    )
    monkeypatch.setattr(engine, "_compress_impl", timeout_after_deadline)
    try:
        messages = [{"role": "user", "content": "preserve"}]
        assert engine.compress(messages) is messages
        assert marker not in str(engine._last_foreground_deadline)
        assert engine._last_foreground_deadline["detail"] == (
            "provider call exceeded the foreground deadline"
        )
    finally:
        engine.shutdown()


def test_abandoned_foreground_provider_cannot_launch_paid_fallbacks(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    engine._config.summary_model = "primary/model-a"
    engine._config.summary_fallback_models = ["fallback/model-b"]
    primary_entered = threading.Event()
    release_primary = threading.Event()
    escalation_finished = threading.Event()
    calls: list[str] = []

    def blocking_primary(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
    ):
        del timeout, route_info
        calls.append(model)
        if len(calls) > 1:
            return "forbidden follow-on provider result"
        primary_entered.set()
        assert release_primary.wait(timeout=2)
        return None

    real_escalation = escalation.summarize_with_escalation

    def observed_escalation(**kwargs):
        try:
            return real_escalation(**kwargs)
        finally:
            escalation_finished.set()

    monkeypatch.setattr(escalation, "_call_llm_for_summary", blocking_primary)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        observed_escalation,
    )
    try:
        with pytest.raises(TimeoutError):
            engine._summarize_leaf_chunk_with_rescue(
                [{"role": "user", "content": "bounded source"}],
                deadline=time.monotonic() + 0.05,
            )
        assert primary_entered.is_set()
        release_primary.set()
        assert escalation_finished.wait(timeout=2)
        assert calls == ["primary/model-a"]
    finally:
        release_primary.set()
        engine.shutdown()
