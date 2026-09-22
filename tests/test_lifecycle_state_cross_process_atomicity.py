"""Deterministic cross-connection races for lifecycle state transitions."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from typing import Any

import pytest

from hermes_lcm.lifecycle_state import LifecycleStateStore


class _ReadBarrierLifecycleStateStore(LifecycleStateStore):
    """Pause one lifecycle operation after its snapshot read."""

    def __init__(self, db_path):
        super().__init__(db_path)
        self._pause_next_conversation_read = False
        self.snapshot_read = threading.Event()
        self.release_snapshot = threading.Event()

    def pause_next_conversation_read(self) -> None:
        self.snapshot_read.clear()
        self.release_snapshot.clear()
        self._pause_next_conversation_read = True

    def get_by_conversation(self, conversation_id):
        state = super().get_by_conversation(conversation_id)
        if self._pause_next_conversation_read:
            self._pause_next_conversation_read = False
            self.snapshot_read.set()
            if not self.release_snapshot.wait(timeout=5):
                raise AssertionError("timed out waiting to release stale lifecycle snapshot")
        return state


def _run_after_snapshot(
    stale_store: _ReadBarrierLifecycleStateStore,
    stale_operation: Callable[[], Any],
    sibling_operation: Callable[[], Any],
) -> Any:
    """Run a sibling write after the stale operation reads but before it writes."""

    outcome: dict[str, Any] = {}

    def run_stale_operation() -> None:
        try:
            outcome["result"] = stale_operation()
        except BaseException as exc:  # pragma: no cover - re-raised in test thread
            outcome["error"] = exc

    stale_store.pause_next_conversation_read()
    thread = threading.Thread(target=run_stale_operation, name="stale-lifecycle-writer")
    thread.start()
    assert stale_store.snapshot_read.wait(timeout=5), "stale operation never read its snapshot"
    try:
        sibling_operation()
    finally:
        stale_store.release_snapshot.set()
        thread.join(timeout=5)
    assert not thread.is_alive(), "stale lifecycle operation did not finish"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def test_stale_finalize_cannot_rewind_sibling_frontier_advance(tmp_path):
    db_path = tmp_path / "finalize-frontier-race.db"
    stale = _ReadBarrierLifecycleStateStore(db_path)
    sibling = LifecycleStateStore(db_path)
    try:
        stale.bind_session("session-a", conversation_id="conversation")
        stale.advance_frontier("conversation", "session-a", 17)

        finalized = _run_after_snapshot(
            stale,
            lambda: stale.finalize_session("conversation", "session-a"),
            lambda: sibling.advance_frontier("conversation", "session-a", 73),
        )

        assert finalized is not None
        assert finalized.last_finalized_frontier_store_id == 73
        state = sibling.get_by_conversation("conversation")
        assert state is not None
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "session-a"
        assert state.last_finalized_frontier_store_id == 73
    finally:
        stale.close()
        sibling.close()


def test_stale_finalize_cannot_clear_newer_sibling_bind(tmp_path):
    db_path = tmp_path / "finalize-bind-race.db"
    stale = _ReadBarrierLifecycleStateStore(db_path)
    sibling = LifecycleStateStore(db_path)
    try:
        stale.bind_session("session-a", conversation_id="conversation")
        stale.advance_frontier("conversation", "session-a", 17)

        finalized = _run_after_snapshot(
            stale,
            lambda: stale.finalize_session("conversation", "session-a"),
            lambda: sibling.bind_session("session-b", conversation_id="conversation"),
        )

        assert finalized is not None
        assert finalized.current_session_id == "session-b"
        state = sibling.get_by_conversation("conversation")
        assert state is not None
        assert state.current_session_id == "session-b"
        assert state.last_finalized_session_id == "session-a"
        assert state.last_finalized_frontier_store_id == 17
    finally:
        stale.close()
        sibling.close()


def test_stale_bind_cannot_clobber_newer_cross_connection_session(tmp_path):
    db_path = tmp_path / "bind-session-race.db"
    stale = _ReadBarrierLifecycleStateStore(db_path)
    sibling = LifecycleStateStore(db_path)
    try:
        stale.bind_session("session-a", conversation_id="conversation")
        stale.finalize_session("conversation", "session-a", frontier_store_id=11)

        rebound = _run_after_snapshot(
            stale,
            lambda: stale.bind_session("session-a", conversation_id="conversation"),
            lambda: sibling.bind_session("session-b", conversation_id="conversation"),
        )

        assert rebound.current_session_id == "session-b"
        state = sibling.get_by_conversation("conversation")
        assert state is not None
        assert state.current_session_id == "session-b"
        assert state.current_frontier_store_id == 0
    finally:
        stale.close()
        sibling.close()


def test_stale_bind_cannot_rewind_newer_same_session_frontier(tmp_path):
    db_path = tmp_path / "bind-frontier-race.db"
    stale = _ReadBarrierLifecycleStateStore(db_path)
    sibling = LifecycleStateStore(db_path)
    try:
        stale.bind_session("session-a", conversation_id="conversation")
        stale.finalize_session("conversation", "session-a", frontier_store_id=11)

        def bind_and_advance_sibling() -> None:
            sibling.bind_session("session-a", conversation_id="conversation")
            sibling.advance_frontier("conversation", "session-a", 89)

        rebound = _run_after_snapshot(
            stale,
            lambda: stale.bind_session("session-a", conversation_id="conversation"),
            bind_and_advance_sibling,
        )

        assert rebound.current_frontier_store_id == 89
        state = sibling.get_by_conversation("conversation")
        assert state is not None
        assert state.current_session_id == "session-a"
        assert state.current_frontier_store_id == 89
    finally:
        stale.close()
        sibling.close()


def test_stale_rollover_cannot_overwrite_newer_cross_connection_rollover(tmp_path):
    db_path = tmp_path / "rollover-session-race.db"
    stale = _ReadBarrierLifecycleStateStore(db_path)
    sibling = LifecycleStateStore(db_path)
    try:
        stale.bind_session("session-a", conversation_id="conversation")

        rolled = _run_after_snapshot(
            stale,
            lambda: stale.record_rollover(
                "conversation",
                old_session_id="session-a",
                new_session_id="session-b",
            ),
            lambda: sibling.record_rollover(
                "conversation",
                old_session_id="session-a",
                new_session_id="session-c",
            ),
        )

        assert rolled.current_session_id == "session-c"
        assert rolled.last_finalized_session_id == "session-a"
        state = sibling.get_by_conversation("conversation")
        assert state is not None
        assert state.current_session_id == "session-c"
        assert state.last_finalized_session_id == "session-a"
    finally:
        stale.close()
        sibling.close()


def test_rollover_preserves_sibling_frontier_advance_after_snapshot(tmp_path):
    db_path = tmp_path / "rollover-frontier-race.db"
    stale = _ReadBarrierLifecycleStateStore(db_path)
    sibling = LifecycleStateStore(db_path)
    try:
        stale.bind_session("session-a", conversation_id="conversation")
        stale.advance_frontier("conversation", "session-a", 17)

        rolled = _run_after_snapshot(
            stale,
            lambda: stale.record_rollover(
                "conversation",
                old_session_id="session-a",
                new_session_id="session-b",
            ),
            lambda: sibling.advance_frontier(
                "conversation",
                "session-a",
                73,
            ),
        )

        assert rolled.current_session_id == "session-b"
        assert rolled.current_frontier_store_id == 0
        assert rolled.last_finalized_session_id == "session-a"
        assert rolled.last_finalized_frontier_store_id == 73
    finally:
        stale.close()
        sibling.close()


class _FailingConnectionProxy:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        fail_restore: bool = False,
        fail_rollback: bool = False,
    ) -> None:
        self._connection = connection
        self._fail_restore = fail_restore
        self._fail_rollback = fail_rollback
        self._busy_timeout_writes = 0

    def execute(self, statement, *args, **kwargs):
        normalized = " ".join(str(statement).split()).lower()
        if normalized.startswith("pragma busy_timeout="):
            self._busy_timeout_writes += 1
            if self._fail_restore and self._busy_timeout_writes > 1:
                raise sqlite3.OperationalError("busy_timeout restore failed")
        return self._connection.execute(statement, *args, **kwargs)

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        if self._fail_rollback:
            raise sqlite3.OperationalError("rollback failed")
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def test_read_busy_timeout_restore_failure_does_not_mask_primary_error(tmp_path):
    lifecycle = LifecycleStateStore(tmp_path / "read-restore-primary.db")
    real_connection = lifecycle.connection
    assert real_connection is not None
    lifecycle._conn = _FailingConnectionProxy(  # type: ignore[assignment]
        real_connection,
        fail_restore=True,
    )
    try:
        with pytest.raises(RuntimeError, match="primary read failure"):
            with lifecycle._bounded_read_connection(busy_timeout_ms=1):
                raise RuntimeError("primary read failure")
    finally:
        lifecycle._conn = real_connection
        lifecycle.close()


def test_rollback_failure_preserves_primary_error_and_closes_connection(tmp_path):
    lifecycle = LifecycleStateStore(tmp_path / "rollback-primary.db")
    real_connection = lifecycle.connection
    assert real_connection is not None
    lifecycle._conn = _FailingConnectionProxy(  # type: ignore[assignment]
        real_connection,
        fail_rollback=True,
    )

    with pytest.raises(RuntimeError, match="primary write failure"):
        with lifecycle._bounded_write_connection(busy_timeout_ms=1) as conn:
            conn.execute(
                """
                UPDATE lcm_lifecycle_state
                SET updated_at=updated_at
                """
            )
            raise RuntimeError("primary write failure")

    assert lifecycle.connection is None
