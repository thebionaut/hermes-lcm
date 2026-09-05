"""Regression coverage for failed MessageStore transaction compensation."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.store import MessageStore


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class _RollbackFailOnceConnection:
    """Connection proxy that expires after one mutation and fails rollback once."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        clock: _FakeClock | None = None,
        expire_on_sql: str = "",
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._expire_on_sql = expire_on_sql
        self.rollback_attempted = False

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            return self._connection.__exit__(exc_type, exc, traceback)
        self.rollback()
        return False

    def execute(self, sql, parameters=()):
        result = self._connection.execute(sql, parameters)
        if (
            self._clock is not None
            and self._expire_on_sql
            and self._expire_on_sql in str(sql)
        ):
            self._clock.value = 101.0
            self._expire_on_sql = ""
        return result

    def rollback(self) -> None:
        if not self.rollback_attempted:
            self.rollback_attempted = True
            raise sqlite3.OperationalError("synthetic rollback failure")
        self._connection.rollback()

    def __getattr__(self, name: str):
        return getattr(self._connection, name)


def _poison_store_connection(
    store: MessageStore,
    *,
    clock: _FakeClock | None = None,
    expire_on_sql: str = "",
) -> _RollbackFailOnceConnection:
    connection = store.connection
    assert connection is not None
    proxy = _RollbackFailOnceConnection(
        connection,
        clock=clock,
        expire_on_sql=expire_on_sql,
    )
    store._conn = proxy
    return proxy


def test_protected_batch_rollback_failure_poison_closes_transaction(tmp_path):
    db_path = tmp_path / "store-append-rollback-poison.db"
    store = MessageStore(db_path)
    clock = _FakeClock()
    proxy = _poison_store_connection(
        store,
        clock=clock,
        expire_on_sql="INSERT INTO messages",
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            store._append_protected_batch(
                "s",
                [{"role": "user", "content": "should roll back"}],
                [1],
                deadline=100.5,
                deadline_clock=clock,
            )

        assert proxy.rollback_attempted
        assert store.connection is None
        with pytest.raises(RuntimeError, match="closed"):
            store._append_protected_batch(
                "s",
                [{"role": "user", "content": "later"}],
                [1],
            )
        with sqlite3.connect(str(db_path)) as verifier:
            rows = verifier.execute("SELECT content FROM messages").fetchall()
        assert rows == []
    finally:
        store.close()


def test_metadata_rollback_failure_poison_closes_transaction(tmp_path):
    db_path = tmp_path / "store-metadata-rollback-poison.db"
    store = MessageStore(db_path)
    clock = _FakeClock()
    proxy = _poison_store_connection(
        store,
        clock=clock,
        expire_on_sql="INSERT INTO metadata",
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            store.write_metadata_json_batch_bounded(
                metadata_json_updates=[(["failed"], '"value"', False)],
                deadline=100.5,
                deadline_clock=clock,
            )

        assert proxy.rollback_attempted
        assert store.connection is None
        with pytest.raises(RuntimeError, match="closed"):
            store.write_metadata_json_batch_bounded(
                metadata_json_updates=[(["later"], '"value"', False)]
            )
        with sqlite3.connect(str(db_path)) as verifier:
            rows = verifier.execute(
                "SELECT key, value FROM metadata WHERE key IN ('failed', 'later')"
            ).fetchall()
        assert rows == []
    finally:
        store.close()


def test_transcript_gc_rollback_failure_poison_closes_transaction(tmp_path):
    db_path = tmp_path / "store-gc-rollback-poison.db"
    store = MessageStore(db_path)
    store.append(
        "s",
        {"role": "tool", "tool_call_id": "call-1", "content": "original"},
    )
    proxy = _poison_store_connection(store)

    def fail_before_commit(_connection: sqlite3.Connection, _store_id: int) -> None:
        raise TimeoutError("primary GC deadline")

    try:
        with pytest.raises(TimeoutError, match="primary GC deadline"):
            store.gc_externalized_tool_result(
                1,
                "[gc]",
                before_commit=fail_before_commit,
            )

        assert proxy.rollback_attempted
        assert store.connection is None
        with pytest.raises(RuntimeError, match="closed"):
            store._append_protected_batch(
                "s",
                [{"role": "user", "content": "later"}],
                [1],
            )
        with sqlite3.connect(str(db_path)) as verifier:
            rows = verifier.execute(
                "SELECT role, content FROM messages ORDER BY store_id"
            ).fetchall()
        assert rows == [("tool", "original")]
    finally:
        store.close()
