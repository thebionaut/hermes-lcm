"""Deadline checks surround foreground leaf work and condensation publication."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import resolve_large_output_storage_dir
from hermes_lcm.reconcile import ForegroundDeadlineExceeded
from hermes_lcm.tokens import count_messages_tokens


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _engine(tmp_path) -> LCMEngine:
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "deadline-postwork.db"),
            background_compaction_enabled=True,
            background_compaction_workers=0,
            leaf_chunk_tokens=1,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    return engine


def _call_before_releasing_store_lock(engine, operation):
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result: dict[str, object] = {}

    def hold_store_lock() -> None:
        with engine._store._write_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    def invoke() -> None:
        try:
            result["value"] = operation()
        except BaseException as exc:
            result["error"] = exc
        finally:
            caller_finished.set()

    holder = threading.Thread(target=hold_store_lock)
    caller = threading.Thread(target=invoke)
    holder.start()
    assert lock_held.wait(timeout=2)
    caller.start()
    try:
        completed_while_contended = caller_finished.wait(timeout=0.2)
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
    assert not holder.is_alive()
    assert not caller.is_alive()
    return completed_while_contended, result


def test_leaf_expired_deadline_stops_before_token_count(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    count_calls: list[str] = []

    def forbidden_count(_messages):
        count_calls.append("counted")
        return 10

    monkeypatch.setattr(
        "hermes_lcm.engine.count_messages_tokens",
        forbidden_count,
    )
    try:
        with pytest.raises(ForegroundDeadlineExceeded):
            engine._summarize_leaf_chunk_with_rescue(
                [{"role": "user", "content": "preserve"}],
                deadline=clock.value,
            )
        assert count_calls == []
    finally:
        engine.shutdown()


def test_leaf_token_count_expiry_stops_before_serialization(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    count_calls: list[str] = []
    serialization_calls: list[str] = []

    def expiring_count(_messages):
        count_calls.append("counted")
        clock.advance(0.6)
        return 10

    def forbidden_serialization(_messages):
        serialization_calls.append("serialized")
        return "forbidden"

    monkeypatch.setattr(
        "hermes_lcm.engine.count_messages_tokens",
        expiring_count,
    )
    monkeypatch.setattr(engine, "_serialize_messages", forbidden_serialization)
    try:
        with pytest.raises(ForegroundDeadlineExceeded):
            engine._summarize_leaf_chunk_with_rescue(
                [{"role": "user", "content": "preserve"}],
                deadline=100.5,
            )
        assert count_calls == ["counted"]
        assert serialization_calls == []
    finally:
        engine.shutdown()


def test_leaf_serialization_expiry_never_starts_provider_or_rescue_retry(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    serialization_calls: list[str] = []
    provider_calls: list[str] = []

    monkeypatch.setattr(
        "hermes_lcm.engine.count_messages_tokens",
        lambda _messages: 100,
    )

    def expiring_serialization(_messages):
        serialization_calls.append("serialized")
        clock.advance(0.6)
        return "late serialization"

    def forbidden_provider(**_kwargs):
        provider_calls.append("provider")
        return "forbidden", 1

    monkeypatch.setattr(engine, "_serialize_messages", expiring_serialization)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        forbidden_provider,
    )
    try:
        with pytest.raises(ForegroundDeadlineExceeded):
            engine._summarize_leaf_chunk_with_rescue(
                [
                    {"role": "user", "content": f"message-{index}"}
                    for index in range(4)
                ],
                deadline=100.5,
            )
        assert serialization_calls == ["serialized"]
        assert provider_calls == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "expiry_step",
    ["source_window", "token_count", "expand_hint", "node_build"],
)
def test_condensation_expiry_during_post_provider_work_never_adds_node(
    tmp_path,
    monkeypatch,
    expiry_step,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    monkeypatch.setattr("hermes_lcm.engine.time.monotonic", clock)
    nodes = [
        SummaryNode(
            session_id="s",
            depth=0,
            summary=f"source-{index}",
            token_count=10,
            source_token_count=20,
            source_ids=[],
            source_type="messages",
        )
        for index in range(2)
    ]
    for node in nodes:
        engine._dag.add_node(node)

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: ("condensed", 1),
    )
    add_calls: list[int] = []
    original_add_node = engine._dag.add_node

    def observed_add_node(node, **kwargs):
        add_calls.append(node.depth)
        return original_add_node(node, **kwargs)

    monkeypatch.setattr(engine._dag, "add_node", observed_add_node)

    if expiry_step == "source_window":
        original_source_window = engine._dag.get_source_time_window

        def expiring_source_window(source_ids):
            result = original_source_window(source_ids)
            clock.advance(0.6)
            return result

        monkeypatch.setattr(
            engine._dag,
            "get_source_time_window",
            expiring_source_window,
        )
    elif expiry_step == "token_count":
        def expiring_token_count(_text):
            clock.advance(0.6)
            return 1

        monkeypatch.setattr(
            "hermes_lcm.engine.count_tokens",
            expiring_token_count,
        )
    elif expiry_step == "expand_hint":
        def expiring_expand_hint(_summary):
            clock.advance(0.6)
            return "hint"

        monkeypatch.setattr(engine, "_extract_expand_hint", expiring_expand_hint)
    else:
        def expiring_node_build(**kwargs):
            clock.advance(0.6)
            return SummaryNode(**kwargs)

        monkeypatch.setattr(
            "hermes_lcm.engine.SummaryNode",
            expiring_node_build,
        )

    try:
        with pytest.raises(ForegroundDeadlineExceeded):
            engine._condense_summary_nodes(nodes, deadline=100.5)
        assert add_calls == []
        assert engine._dag.get_session_node_count("s") == 2
    finally:
        engine.shutdown()


def test_background_emergency_publication_python_lock_wait_respects_deadline(
    tmp_path,
):
    engine = _engine(tmp_path)
    helper = engine._background_compaction
    assert helper is not None
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_python_lock() -> None:
        with helper._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    def publish() -> None:
        try:
            result_box["value"] = engine._publish_foreground_background_node(
                SummaryNode(
                    session_id="s",
                    depth=0,
                    summary="must not publish late",
                    token_count=4,
                    source_token_count=20,
                    source_ids=[1],
                    source_type="messages",
                ),
                expected_session_fence_generation=1,
                expected_frontier_store_id=0,
                consumed_store_ids=[1],
                expected_source_identity_hash="unused-before-lock",
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    holder = threading.Thread(target=hold_python_lock)
    caller = threading.Thread(target=publish)
    try:
        holder.start()
        assert lock_held.wait(timeout=2)
        caller.start()
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert isinstance(result_box.get("error"), ForegroundDeadlineExceeded)
        assert "value" not in result_box
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_background_emergency_publication_sqlite_wait_respects_deadline(tmp_path):
    engine = _engine(tmp_path)
    blocker = sqlite3.connect(str(tmp_path / "deadline-postwork.db"))
    result_box: dict[str, object] = {}
    started = time.monotonic()
    try:
        blocker.execute("BEGIN IMMEDIATE")
        try:
            result_box["value"] = engine._publish_foreground_background_node(
                SummaryNode(
                    session_id="s",
                    depth=0,
                    summary="must not publish through a locked writer",
                    token_count=4,
                    source_token_count=20,
                    source_ids=[1],
                    source_type="messages",
                ),
                expected_session_fence_generation=1,
                expected_frontier_store_id=0,
                consumed_store_ids=[1],
                expected_source_identity_hash="unused-before-write-lock",
                deadline=started + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert isinstance(result_box.get("error"), ForegroundDeadlineExceeded)
        assert "value" not in result_box
    finally:
        blocker.rollback()
        blocker.close()
        engine.shutdown()


def test_background_publication_contention_diagnostic_never_waits_on_python_lock(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    helper = engine._background_compaction
    assert helper is not None
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_python_lock() -> None:
        with helper._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    def locked_publication(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(helper, "publish_foreground_node", locked_publication)

    def publish() -> None:
        try:
            result_box["value"] = engine._publish_foreground_background_node(
                SummaryNode(
                    session_id="s",
                    depth=0,
                    summary="diagnostic must not outwait caller",
                    token_count=4,
                    source_token_count=20,
                    source_ids=[1],
                    source_type="messages",
                ),
                expected_session_fence_generation=1,
                expected_frontier_store_id=0,
                consumed_store_ids=[1],
                expected_source_identity_hash="unused-before-lock",
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    holder = threading.Thread(target=hold_python_lock)
    caller = threading.Thread(target=publish)
    try:
        holder.start()
        assert lock_held.wait(timeout=2)
        caller.start()
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert "error" not in result_box
        result = result_box.get("value")
        assert result is not None
        assert result.promoted is False
        assert result.reason == "sqlite_lock_contention"
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_background_write_deadline_rolls_back_before_commit(tmp_path):
    engine = _engine(tmp_path)
    helper = engine._background_compaction
    assert helper is not None
    assert helper.connection is not None
    clock = _FakeClock()
    helper.connection.execute(
        "CREATE TABLE foreground_deadline_probe(value TEXT NOT NULL)"
    )
    helper.connection.commit()
    try:
        with pytest.raises(TimeoutError):
            with helper._write_transaction(
                busy_timeout_ms=0,
                lock_timeout_seconds=0.5,
                write_deadline=100.5,
                deadline_clock=clock,
            ) as conn:
                conn.execute(
                    "INSERT INTO foreground_deadline_probe(value) VALUES('late')"
                )
                clock.advance(1.0)

        assert helper.connection.execute(
            "SELECT COUNT(*) FROM foreground_deadline_probe"
        ).fetchone()[0] == 0
    finally:
        engine.shutdown()


def test_condensation_dag_publication_lock_wait_respects_deadline(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    nodes = [
        SummaryNode(
            session_id="s",
            depth=0,
            summary=f"source-{index}",
            token_count=10,
            source_token_count=20,
            source_ids=[],
            source_type="messages",
        )
        for index in range(2)
    ]
    for node in nodes:
        engine._dag.add_node(node)

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: ("must not persist late", 1),
    )
    # Isolate the publication lock. The real lookup uses the same RLock and
    # would otherwise consume the barrier before add_node is reached.
    monkeypatch.setattr(
        engine._dag,
        "get_source_time_window",
        lambda _source_ids: (None, None),
    )

    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_dag_lock() -> None:
        with engine._dag._db_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    def condense() -> None:
        try:
            result_box["value"] = engine._condense_summary_nodes(
                nodes,
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    holder = threading.Thread(target=hold_dag_lock)
    caller = threading.Thread(target=condense)
    try:
        holder.start()
        assert lock_held.wait(timeout=2)
        caller.start()
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert isinstance(result_box.get("error"), ForegroundDeadlineExceeded)
        assert engine._dag.get_session_node_count("s") == 2
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_condensation_defers_rollup_drain_without_losing_invalidation(
    tmp_path,
    monkeypatch,
):
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "deadline-rollup-invalidation.db"),
            background_compaction_enabled=False,
            temporal_rollups_enabled=True,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    assert engine.drain_rollup_maintenance(timeout=2)
    nodes = [
        SummaryNode(
            session_id="s",
            depth=0,
            summary=f"source-{index}",
            token_count=10,
            source_token_count=20,
            source_ids=[],
            source_type="messages",
        )
        for index in range(2)
    ]
    for node in nodes:
        engine._dag.add_node(node)
    engine._dag.connection.execute("DELETE FROM lcm_rollup_invalidations")
    engine._dag.connection.commit()

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: ("deadline-safe condensation", 1),
    )
    invalidator_entered = threading.Event()
    release_invalidator = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def blocking_invalidator(*_args, **_kwargs):
        invalidator_entered.set()
        assert release_invalidator.wait(timeout=2)
        return 0

    monkeypatch.setattr(
        "hermes_lcm.engine.mark_stale_for_published_summary",
        blocking_invalidator,
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
            caller_finished.set()

    caller = threading.Thread(target=condense)
    try:
        caller.start()
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_invalidator.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert not invalidator_entered.is_set()
        assert "error" not in result_box
        assert engine._dag.get_session_node_count("s") == 3
        pending = engine._dag.connection.execute(
            "SELECT node_id, operation FROM lcm_rollup_invalidations"
        ).fetchall()
        assert [(int(row[0]), str(row[1])) for row in pending] == [
            (nodes[-1].node_id + 1, "insert")
        ]
    finally:
        release_invalidator.set()
        caller.join(timeout=2)
        engine.shutdown()


def test_foreground_dag_publication_sqlite_wait_respects_deadline(tmp_path):
    engine = _engine(tmp_path)
    blocker = sqlite3.connect(str(tmp_path / "deadline-postwork.db"))
    node = SummaryNode(
        session_id="s",
        depth=1,
        summary="must not publish through a locked writer",
        token_count=4,
        source_token_count=20,
        source_ids=[],
        source_type="nodes",
    )
    started = time.monotonic()
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(ForegroundDeadlineExceeded):
            engine._publish_foreground_dag_node(
                node,
                deadline=started + 0.05,
                stage="condensation",
            )
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert node.node_id == 0
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        blocker.rollback()
        blocker.close()
        engine.shutdown()


def test_deadline_expiry_inside_dag_write_rolls_back_before_commit(tmp_path):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    node = SummaryNode(
        session_id="s",
        depth=1,
        summary="must roll back",
        token_count=4,
        source_token_count=20,
        source_ids=[],
        source_type="nodes",
    )
    engine._dag._conn.create_function(
        "expire_foreground_deadline",
        0,
        lambda: clock.advance(1.0),
    )
    engine._dag._conn.execute(
        """
        CREATE TEMP TRIGGER expire_summary_insert
        BEFORE INSERT ON summary_nodes
        BEGIN
            SELECT expire_foreground_deadline();
        END
        """
    )
    try:
        with pytest.raises(ForegroundDeadlineExceeded):
            engine._publish_foreground_dag_node(
                node,
                deadline=100.5,
                stage="condensation",
            )

        assert node.node_id == 0
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        engine.shutdown()


def test_disabled_background_leaf_publication_lock_wait_respects_deadline(
    tmp_path,
    monkeypatch,
):
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "disabled-leaf-publication.db"),
            background_compaction_enabled=False,
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
            incremental_max_depth=0,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old source message"},
        {"role": "assistant", "content": "old source response"},
        {"role": "user", "content": "fresh tail"},
    ]
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_dag_lock() -> None:
        with engine._dag._db_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_dag_lock)

    def summarize(chunk, **_kwargs):
        holder.start()
        assert lock_held.wait(timeout=2)
        return (
            chunk,
            count_messages_tokens(chunk),
            "must not persist late",
            1,
            1,
        )

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)

    def compress() -> None:
        try:
            result_box["value"] = engine.compress(
                messages,
                current_tokens=count_messages_tokens(messages),
                force=True,
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=compress)
    try:
        caller.start()
        assert lock_held.wait(timeout=2)
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert "error" not in result_box
        assert result_box.get("value") is messages
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_disabled_leaf_frontier_lock_timeout_prevents_partial_publication(
    tmp_path,
    monkeypatch,
):
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "disabled-frontier-publication.db"),
            background_compaction_enabled=False,
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
            incremental_max_depth=0,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old source message"},
        {"role": "assistant", "content": "old source response"},
        {"role": "user", "content": "fresh tail"},
    ]
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_lifecycle_lock() -> None:
        with engine._lifecycle._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_lifecycle_lock)

    def summarize(chunk, **_kwargs):
        holder.start()
        assert lock_held.wait(timeout=2)
        return (
            chunk,
            count_messages_tokens(chunk),
            "must not partially persist",
            1,
            1,
        )

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)

    def compress() -> None:
        try:
            result_box["value"] = engine.compress(
                messages,
                current_tokens=count_messages_tokens(messages),
                force=True,
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=compress)
    try:
        caller.start()
        assert lock_held.wait(timeout=2)
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert "error" not in result_box
        assert result_box.get("value") is messages
        assert engine._dag.get_session_node_count("s") == 0
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.current_frontier_store_id == 0
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_disabled_leaf_optional_transcript_gc_never_waits_or_runs_late(
    tmp_path,
    monkeypatch,
):
    hermes_home = str(tmp_path / "hermes")
    config = LCMConfig(
        database_path=str(tmp_path / "disabled-gc-publication.db"),
        background_compaction_enabled=False,
        fresh_tail_count=0,
        leaf_chunk_tokens=50,
        dynamic_leaf_chunk_enabled=False,
        incremental_max_depth=0,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_transcript_gc_enabled=True,
    )
    resolve_large_output_storage_dir(config, hermes_home=hermes_home)
    engine = LCMEngine(config, hermes_home=hermes_home)
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    content = "RESULT:\n" + ("abcdef" * 2000)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "tool", "tool_call_id": "call_gc", "content": content},
    ]
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_store_lock() -> None:
        with engine._store._write_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_store_lock)

    def summarize(chunk, **_kwargs):
        holder.start()
        assert lock_held.wait(timeout=2)
        return (
            chunk,
            count_messages_tokens(chunk),
            "deadline-safe summary",
            1,
            1,
        )

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)

    def compress() -> None:
        try:
            result_box["value"] = engine.compress(
                messages,
                current_tokens=count_messages_tokens(messages),
                force=True,
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=compress)
    try:
        caller.start()
        assert lock_held.wait(timeout=2)
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert "error" not in result_box
        stored_tool = next(
            row
            for row in engine._store.get_range("s")
            if row["role"] == "tool"
        )
        assert stored_tool["content"].startswith("[Externalized tool output:")
        assert not stored_tool["content"].startswith(
            "[GC'd externalized tool output:"
        )
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_disabled_leaf_derived_placeholder_metadata_never_waits_or_runs_late(
    tmp_path,
    monkeypatch,
):
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "disabled-derived-metadata.db"),
            background_compaction_enabled=False,
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
            incremental_max_depth=0,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old source message"},
        {"role": "assistant", "content": "old source response"},
        {"role": "user", "content": "fresh tail"},
    ]
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def hold_store_lock() -> None:
        with engine._store._write_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_store_lock)

    def summarize(chunk, **_kwargs):
        return (
            chunk,
            count_messages_tokens(chunk),
            "deadline-safe summary",
            1,
            1,
        )

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    publish = engine._publish_foreground_dag_node

    def publish_then_block_derived_metadata(*args, **kwargs):
        result = publish(*args, **kwargs)
        holder.start()
        assert lock_held.wait(timeout=2)
        return result

    monkeypatch.setattr(
        engine,
        "_publish_foreground_dag_node",
        publish_then_block_derived_metadata,
    )

    def compress() -> None:
        try:
            result_box["value"] = engine.compress(
                messages,
                current_tokens=count_messages_tokens(messages),
                force=True,
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=compress)
    try:
        caller.start()
        assert lock_held.wait(timeout=2)
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert "error" not in result_box
        assert engine._dag.get_session_node_count("s") == 1
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.current_frontier_store_id > 0
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_background_emergency_source_time_bounds_lock_never_outwaits_wall_or_publishes(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 1
    engine._config.fresh_tail_max_tokens = 0
    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"emergency-source-{index} x x x",
        }
        for index in range(6)
    ]
    visible = engine._ingest_messages(messages)
    helper = engine._background_compaction
    assert helper is not None
    fresh_tail_start = engine._fresh_tail_start(visible)
    durable_rows = engine._store.get_session_messages("s")
    source_ids = [
        int(row["store_id"])
        for row in durable_rows[:fresh_tail_start]
    ]
    assert source_ids
    source_snapshot = tuple(durable_rows[:fresh_tail_start])
    session_fence_generation = helper.get_publishable_session_fence_generation(
        conversation_id="c",
        session_id="s",
    )
    assert session_fence_generation is not None
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    provider_returned = threading.Event()
    result_box: dict[str, object] = {}

    def hold_store_lock() -> None:
        with engine._store._write_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_store_lock)

    def summarize(chunk, **_kwargs):
        holder.start()
        assert lock_held.wait(timeout=2)
        provider_returned.set()
        return (
            chunk,
            count_messages_tokens(chunk),
            "must not publish after source-read contention",
            1,
            1,
        )

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    monkeypatch.setattr(
        engine,
        "_get_store_ids_for_messages",
        lambda *_args, **_kwargs: list(source_ids),
    )
    monkeypatch.setattr(
        helper,
        "capture_foreground_source_snapshot",
        lambda **_kwargs: source_snapshot,
    )
    monkeypatch.setattr(
        engine,
        "_reconcile_background_publication_fence",
        lambda **_kwargs: session_fence_generation,
    )
    monkeypatch.setattr(
        engine,
        "_maybe_enqueue_background_compaction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine,
        "_adopt_committed_background_frontier",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        engine,
        "_has_publishable_background_job",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        helper,
        "has_active_partial_atomic_repair",
        lambda **_kwargs: False,
    )

    def compress() -> None:
        try:
            result_box["value"] = engine.compress(
                visible,
                current_tokens=90_000,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=compress)
    try:
        caller.start()
        assert provider_returned.wait(timeout=2), result_box
        completed_within_wall = caller_finished.wait(timeout=0.2)
        assert completed_within_wall
        assert "error" not in result_box
        assert result_box.get("value") is visible
        assert engine._dag.get_session_node_count("s") == 0
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.current_frontier_store_id == 0
    finally:
        release_lock.set()
        if holder.ident is not None:
            holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_background_emergency_source_mapping_lock_stops_before_provider_or_publication(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 1
    engine._config.fresh_tail_max_tokens = 0
    stored_messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"mapped-source-{index} x x x",
        }
        for index in range(6)
    ]
    engine.ingest(stored_messages)
    messages = [
        engine._store.to_openai_msg(row)
        for row in engine._store.get_session_messages("s")
    ]
    helper = engine._background_compaction
    assert helper is not None
    provider_calls: list[str] = []
    session_fence_generation = helper.get_publishable_session_fence_generation(
        conversation_id="c",
        session_id="s",
    )
    assert session_fence_generation is not None

    def source_snapshot(**kwargs):
        source_ids = list(kwargs["expected_source_store_ids"])
        source_rows = engine._store.get_batch(source_ids)
        return tuple(source_rows[store_id] for store_id in source_ids)

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        engine,
        "_maybe_enqueue_background_compaction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine,
        "_adopt_committed_background_frontier",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        engine,
        "_has_publishable_background_job",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        helper,
        "has_active_partial_atomic_repair",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        helper,
        "capture_foreground_source_snapshot",
        source_snapshot,
    )
    monkeypatch.setattr(
        engine,
        "_reconcile_background_publication_fence",
        lambda **_kwargs: session_fence_generation,
    )
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda *_args, **_kwargs: provider_calls.append("provider")
        or ([], 1, "forbidden", 1, 1),
    )
    engine._active_replay_store_refs_by_message_id = {}

    try:
        completed, result = _call_before_releasing_store_lock(
            engine,
            lambda: engine.compress(messages, current_tokens=90_000),
        )
        assert completed
        assert result.get("error") is None
        assert result.get("value") == messages
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.current_frontier_store_id == 0
    finally:
        engine.shutdown()


def test_background_emergency_source_mapping_stops_between_expired_pages(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    engine._config.fresh_tail_count = 1
    engine._config.fresh_tail_max_tokens = 0
    stored_messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"paged-source-{index} x x x",
        }
        for index in range(6)
    ]
    engine.ingest(stored_messages)
    durable_messages = engine._store.get_session_messages("s")
    messages = [engine._store.to_openai_msg(row) for row in durable_messages]
    durable_row = durable_messages[0]
    helper = engine._background_compaction
    assert helper is not None
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    reads: list[float | None] = []
    provider_calls: list[str] = []

    def expiring_page(*_args, deadline=None, **_kwargs):
        reads.append(deadline)
        if len(reads) == 1:
            clock.advance(1.0)
            return [durable_row]
        return []

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.5,
    )
    monkeypatch.setattr(
        engine,
        "_ingest_messages",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        engine,
        "_maybe_enqueue_background_compaction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine,
        "_adopt_committed_background_frontier",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        engine,
        "_has_publishable_background_job",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        helper,
        "has_active_partial_atomic_repair",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        engine._store,
        "get_session_messages_after",
        expiring_page,
    )
    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda *_args, **_kwargs: provider_calls.append("provider")
        or ([], 1, "forbidden", 1, 1),
    )
    engine._active_replay_store_refs_by_message_id = {}

    try:
        result = engine.compress(messages, current_tokens=90_000)
        assert result == messages
        assert reads == [100.5]
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.current_frontier_store_id == 0
    finally:
        engine.shutdown()


def test_disabled_leaf_derived_debt_refresh_never_waits_after_publication(
    tmp_path,
    monkeypatch,
):
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "disabled-derived-debt.db"),
            background_compaction_enabled=False,
            deferred_maintenance_enabled=True,
            deferred_maintenance_max_passes=1,
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
            incremental_max_depth=0,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    engine._lifecycle.record_debt(
        "c",
        kind="raw_backlog",
        size_estimate=10,
    )
    messages = [{"role": "system", "content": "system"}]
    for index in range(6):
        messages.extend(
            [
                {"role": "user", "content": f"old source message {index}"},
                {"role": "assistant", "content": f"old source response {index}"},
            ]
        )
    messages.append({"role": "user", "content": "fresh tail"})
    lock_held = threading.Event()
    release_lock = threading.Event()
    caller_finished = threading.Event()
    refresh_entered = threading.Event()
    refresh_entered_while_locked = threading.Event()
    result_box: dict[str, object] = {}

    def hold_lifecycle_lock() -> None:
        with engine._lifecycle._lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_lifecycle_lock)

    monkeypatch.setattr(
        engine,
        "_summarize_leaf_chunk_with_rescue",
        lambda chunk, **_kwargs: (
            chunk,
            count_messages_tokens(chunk),
            "deadline-safe summary",
            1,
            1,
        ),
    )

    def start_holder_immediately_before_debt_refresh(*_args, **_kwargs) -> None:
        holder.start()
        assert lock_held.wait(timeout=2)

    monkeypatch.setattr(
        engine,
        "_maybe_condense",
        start_holder_immediately_before_debt_refresh,
    )
    refresh_raw_backlog_debt = engine._refresh_raw_backlog_debt

    def observe_refresh(*args, **kwargs) -> None:
        refresh_entered.set()
        if lock_held.is_set():
            refresh_entered_while_locked.set()
            assert not engine._lifecycle._lock.acquire(blocking=False)
            assert engine._config.deferred_maintenance_enabled
            assert engine._conversation_id == "c"
            assert kwargs.get("deadline") is not None
        refresh_raw_backlog_debt(*args, **kwargs)

    monkeypatch.setattr(engine, "_refresh_raw_backlog_debt", observe_refresh)

    def compress() -> None:
        try:
            result_box["value"] = engine.compress(
                messages,
                current_tokens=count_messages_tokens(messages),
                force=True,
                deadline=time.monotonic() + 1.0,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=compress)
    try:
        caller.start()
        assert lock_held.wait(timeout=2)
        completed_within_wall = caller_finished.wait(timeout=0.2)
        release_lock.set()
        caller.join(timeout=2)

        assert completed_within_wall
        assert "error" not in result_box
        assert refresh_entered.is_set()
        assert refresh_entered_while_locked.is_set()
        assert engine._dag.get_session_node_count("s") == 1
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.debt_size_estimate == 10
    finally:
        release_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        engine.shutdown()


def test_derived_debt_refresh_never_waits_or_writes_late_on_sqlite_contention(
    tmp_path,
):
    database_path = tmp_path / "derived-debt-sqlite.db"
    engine = LCMEngine(
        LCMConfig(
            database_path=str(database_path),
            background_compaction_enabled=False,
            deferred_maintenance_enabled=True,
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
        )
    )
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    engine._lifecycle.record_debt(
        "c",
        kind="raw_backlog",
        size_estimate=10,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old source message"},
        {"role": "assistant", "content": "old source response"},
        {"role": "user", "content": "fresh tail"},
    ]
    blocker = sqlite3.connect(str(database_path), timeout=0.1)
    blocker.execute("PRAGMA busy_timeout=100")
    blocker.execute("BEGIN IMMEDIATE")
    caller_finished = threading.Event()
    result_box: dict[str, object] = {}

    def refresh() -> None:
        try:
            engine._refresh_raw_backlog_debt(
                messages,
                observed_tokens=count_messages_tokens(messages),
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=refresh)
    try:
        caller.start()
        assert caller_finished.wait(timeout=0.2)
        assert "error" not in result_box
        blocker.execute("ROLLBACK")
        caller.join(timeout=2)

        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.debt_size_estimate == 10
    finally:
        if blocker.in_transaction:
            blocker.execute("ROLLBACK")
        blocker.close()
        caller.join(timeout=2)
        engine.shutdown()
