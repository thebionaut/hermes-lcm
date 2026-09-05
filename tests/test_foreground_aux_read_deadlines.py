"""Automatic compaction must bound DAG, assertion, and lifecycle side reads."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import ForegroundDeadlineExceeded


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> int:
        self.value += seconds
        return 0


def _engine(tmp_path, **overrides) -> LCMEngine:
    values = {
        "database_path": str(tmp_path / "foreground-aux-read-deadlines.db"),
        "background_compaction_enabled": False,
        "fresh_tail_count": 1,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 1,
        "deferred_maintenance_enabled": True,
    }
    values.update(overrides)
    engine = LCMEngine(LCMConfig(**values))
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    return engine


@contextmanager
def _held_lock(lock: threading.RLock) -> Iterator[None]:
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


def _call_before_releasing_lock(
    lock: threading.RLock,
    operation: Callable[[], object],
) -> tuple[bool, dict[str, object]]:
    finished = threading.Event()
    result: dict[str, object] = {}

    def invoke() -> None:
        try:
            result["value"] = operation()
        except BaseException as exc:
            result["error"] = exc
        finally:
            finished.set()

    with _held_lock(lock):
        caller = threading.Thread(target=invoke)
        caller.start()
        completed_while_contended = finished.wait(timeout=0.25)
    caller.join(timeout=2)
    assert not caller.is_alive()
    return completed_while_contended, result


def test_automatic_frontier_read_respects_shared_wall_without_provider_or_publish(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=True)
    messages = [
        {"role": "user", "content": "old raw source " * 8},
        {"role": "assistant", "content": "fresh tail"},
    ]
    provider_calls: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: provider_calls.append("provider") or ("summary", 1),
    )
    engine.threshold_tokens = 1
    try:
        completed, result = _call_before_releasing_lock(
            engine._dag._db_lock,
            lambda: engine.compress(messages, current_tokens=10_000),
        )

        assert completed
        assert result.get("error") is None
        assert result.get("value") is messages
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        engine.shutdown()


def test_automatic_assertion_reads_stop_before_worker_provider_and_publication(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        assertions_enabled=True,
        assertion_extraction_enabled=True,
        assertion_extraction_max_sources_per_pass=2,
    )
    assert engine._assertions is not None
    messages = [
        {"role": "user", "content": "old assertion source " * 8},
        {"role": "assistant", "content": "fresh tail"},
    ]
    assertion_workers: list[str] = []
    summary_providers: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        engine,
        "_run_assertion_extraction_batch",
        lambda *_args: assertion_workers.append("assertion-provider"),
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: summary_providers.append("summary-provider")
        or ("summary", 1),
    )
    engine.threshold_tokens = 1
    try:
        completed, result = _call_before_releasing_lock(
            engine._assertions._write_lock,
            lambda: engine.compress(messages, current_tokens=10_000),
        )

        assert completed
        assert result.get("error") is None
        assert result.get("value") is messages
        assert assertion_workers == []
        assert summary_providers == []
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        engine.shutdown()


def test_automatic_deferred_maintenance_read_respects_shared_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    messages = [
        {"role": "user", "content": "raw maintenance debt"},
        {"role": "assistant", "content": "fresh tail"},
    ]
    engine._lifecycle.record_debt("c", kind="raw_backlog", size_estimate=10)
    provider_calls: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: provider_calls.append("provider") or ("summary", 1),
    )
    engine.threshold_tokens = 99_999
    try:
        completed, result = _call_before_releasing_lock(
            engine._lifecycle._lock,
            lambda: engine.compress(messages, current_tokens=10_000),
        )

        assert completed
        assert result.get("error") is None
        assert result.get("value") is messages
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
        state = engine._lifecycle.get_by_conversation("c")
        assert state is not None
        assert state.last_maintenance_attempt_at is None
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("get_session_nodes", ("s",)),
        ("get_session_node_count", ("s",)),
        ("get_uncondensed_at_depth", ("s", 0)),
        ("get_active_frontier_nodes", ("s",)),
        ("get_source_time_window", ([1],)),
    ],
)
def test_summary_dag_read_apis_bound_the_process_lock(
    tmp_path,
    method_name,
    args,
):
    dag = SummaryDAG(tmp_path / "bounded-dag-reads.db")
    try:
        completed, result = _call_before_releasing_lock(
            dag._db_lock,
            lambda: getattr(dag, method_name)(
                *args,
                lock_timeout_seconds=0.05,
                deadline=time.monotonic() + 0.05,
                deadline_clock=time.monotonic,
                busy_timeout_ms=50,
            ),
        )

        assert completed
        assert isinstance(result.get("error"), TimeoutError)
        assert "value" not in result
    finally:
        dag.close()


def test_summary_dag_read_detects_fake_deadline_expiry_inside_sqlite(tmp_path):
    dag = SummaryDAG(tmp_path / "dag-read-expiry.db")
    clock = _FakeClock()
    assert dag.connection is not None
    dag.connection.set_progress_handler(lambda: clock.advance(1.0), 1)
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            dag.get_active_frontier_nodes(
                "s",
                deadline=100.5,
                deadline_clock=clock,
                busy_timeout_ms=0,
            )
    finally:
        dag.connection.set_progress_handler(None, 0)
        dag.close()


def test_lifecycle_maintenance_write_rolls_back_on_fake_deadline_expiry(
    tmp_path,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    connection = engine._lifecycle.connection
    assert connection is not None
    connection.create_function(
        "expire_lifecycle_maintenance_deadline",
        0,
        lambda: clock.advance(1.0),
    )
    connection.execute(
        """
        CREATE TEMP TRIGGER expire_lifecycle_maintenance_after_update
        AFTER UPDATE OF last_maintenance_attempt_at ON lcm_lifecycle_state
        BEGIN
          SELECT expire_lifecycle_maintenance_deadline();
        END
        """
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            engine._lifecycle.record_maintenance_attempt(
                "c",
                deadline=100.5,
                deadline_clock=clock,
                busy_timeout_ms=0,
            )
        state = engine._lifecycle.get_by_conversation("c")
        assert state is not None
        assert state.last_maintenance_attempt_at is None
    finally:
        engine.shutdown()


def test_condensation_source_window_lock_never_allows_late_dag_publication(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    nodes = [
        SummaryNode(
            session_id="s",
            depth=0,
            summary=f"source-{index}",
            token_count=4,
            source_token_count=10,
            source_ids=[],
            source_type="messages",
        )
        for index in range(2)
    ]
    for node in nodes:
        engine._dag.add_node(node)
    provider_calls: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: provider_calls.append("provider") or ("condensed", 1),
    )
    try:
        completed, result = _call_before_releasing_lock(
            engine._dag._db_lock,
            lambda: engine._condense_summary_nodes(
                nodes,
                deadline=time.monotonic() + 0.05,
            ),
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert provider_calls == ["provider"]
        assert engine._dag.get_session_node_count("s") == 2
    finally:
        engine.shutdown()
