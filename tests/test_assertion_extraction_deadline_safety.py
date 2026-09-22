"""Deadline and durability contracts for optional foreground extraction work."""

from __future__ import annotations

import builtins
from datetime import datetime
import sqlite3
import sys
import threading
from types import ModuleType

import pytest

import hermes_lcm.assertion_extraction as assertion_extraction_module
import hermes_lcm.assertion_store as assertion_store_module
import hermes_lcm.extraction as extraction_module
import hermes_lcm.model_routing as model_routing_module
from hermes_lcm.assertion_extraction import ModelAssertionExtractor
from hermes_lcm.assertion_store import AssertionSchemaUnavailableError, AssertionStore
from hermes_lcm.store import MessageStore


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _WriteObserver:
    def __init__(
        self,
        handle,
        *,
        after_write,
        fail_after_prefix: bool = False,
        return_short_write: bool = False,
    ) -> None:
        self._handle = handle
        self._after_write = after_write
        self._fail_after_prefix = fail_after_prefix
        self._return_short_write = return_short_write

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *args):
        return self._handle.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def write(self, content: str | bytes) -> int:
        if self._fail_after_prefix:
            prefix = content[: max(1, len(content) // 2)]
            written = self._handle.write(prefix)
            self._handle.flush()
            self._after_write()
            raise OSError("synthetic partial write")
        written = self._handle.write(content)
        self._handle.flush()
        self._after_write()
        if self._return_short_write:
            return max(0, written - 1)
        return written


def _observe_text_writes(
    monkeypatch,
    *,
    after_write,
    fail_after_prefix=False,
    return_short_write=False,
) -> None:
    def observed_open(path, mode="r", *args, **kwargs):
        handle = builtins.open(path, mode, *args, **kwargs)
        if "w" in mode or "a" in mode:
            return _WriteObserver(
                handle,
                after_write=after_write,
                fail_after_prefix=fail_after_prefix,
                return_short_write=return_short_write,
            )
        return handle

    monkeypatch.setattr(extraction_module, "open", observed_open, raising=False)


def _daily_file(output_dir):
    return output_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"


def test_extraction_write_crossing_deadline_never_replaces_durable_note(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "extractions"
    output_dir.mkdir()
    target = _daily_file(output_dir)
    target.write_text("existing\n", encoding="utf-8")
    clock = _Clock()
    deadline = 101.0
    _observe_text_writes(
        monkeypatch,
        after_write=lambda: setattr(clock, "value", deadline),
    )

    assert extraction_module.persist_pre_compaction_extraction(
        extraction_module.PreparedExtraction(content="- late decision"),
        output_path=str(output_dir),
        deadline=deadline,
        now=clock,
    ) is False
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_extraction_replace_crossing_deadline_restores_the_prior_durable_note(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "extractions"
    output_dir.mkdir()
    target = _daily_file(output_dir)
    target.write_text("existing\n", encoding="utf-8")
    clock = _Clock()
    deadline = 101.0
    original_replace = extraction_module.os.replace
    replace_calls = 0

    def expiring_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        original_replace(source, destination)
        if replace_calls == 1:
            clock.value = deadline

    monkeypatch.setattr(extraction_module.os, "replace", expiring_replace)

    assert extraction_module.persist_pre_compaction_extraction(
        extraction_module.PreparedExtraction(content="- late decision"),
        output_path=str(output_dir),
        deadline=deadline,
        now=clock,
    ) is False
    assert replace_calls == 2
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_extraction_rollback_failure_never_reports_false_with_candidate_visible(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "extractions"
    output_dir.mkdir()
    target = _daily_file(output_dir)
    target.write_text("existing\n", encoding="utf-8")
    clock = _Clock()
    deadline = 101.0
    original_replace = extraction_module.os.replace
    replace_calls = 0

    def commit_then_refuse_rollback(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            original_replace(source, destination)
            clock.value = deadline
            return
        raise OSError("synthetic persistent rollback failure")

    monkeypatch.setattr(
        extraction_module.os,
        "replace",
        commit_then_refuse_rollback,
    )

    assert extraction_module.persist_pre_compaction_extraction(
        extraction_module.PreparedExtraction(content="- complete late decision"),
        output_path=str(output_dir),
        deadline=deadline,
        now=clock,
    ) is True
    assert replace_calls == 4
    assert "complete late decision" in target.read_text(encoding="utf-8")
    recovery_files = list(output_dir.glob(".*.rollback.*.tmp"))
    assert len(recovery_files) == 1
    assert recovery_files[0].read_text(encoding="utf-8") == "existing\n"


def test_partial_extraction_write_cannot_leave_a_visible_prefix(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "extractions"
    output_dir.mkdir()
    target = _daily_file(output_dir)
    target.write_text("existing\n", encoding="utf-8")
    _observe_text_writes(
        monkeypatch,
        after_write=lambda: None,
        fail_after_prefix=True,
    )

    assert extraction_module.persist_pre_compaction_extraction(
        extraction_module.PreparedExtraction(content="- atomic decision"),
        output_path=str(output_dir),
    ) is False
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_short_extraction_write_cannot_replace_the_durable_note(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "extractions"
    output_dir.mkdir()
    target = _daily_file(output_dir)
    target.write_text("existing\n", encoding="utf-8")
    _observe_text_writes(
        monkeypatch,
        after_write=lambda: None,
        return_short_write=True,
    )

    assert extraction_module.persist_pre_compaction_extraction(
        extraction_module.PreparedExtraction(content="- atomic decision"),
        output_path=str(output_dir),
    ) is False
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_failed_commit_fsync_restores_the_prior_durable_note(tmp_path, monkeypatch):
    output_dir = tmp_path / "extractions"
    output_dir.mkdir()
    target = _daily_file(output_dir)
    target.write_text("existing\n", encoding="utf-8")
    original_fsync = extraction_module.os.fsync
    fsync_calls = 0
    directory_failures = 0

    def fail_first_directory_fsync(fd):
        nonlocal directory_failures, fsync_calls
        fsync_calls += 1
        is_directory = extraction_module.stat.S_ISDIR(
            extraction_module.os.fstat(fd).st_mode
        )
        if is_directory and directory_failures == 0:
            directory_failures += 1
            raise OSError("synthetic directory fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(extraction_module.os, "fsync", fail_first_directory_fsync)

    assert extraction_module.persist_pre_compaction_extraction(
        extraction_module.PreparedExtraction(content="- atomic decision"),
        output_path=str(output_dir),
    ) is False
    assert directory_failures == 1
    assert fsync_calls >= 4
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_windows_extraction_lock_fallback_uses_a_locked_byte(tmp_path, monkeypatch):
    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self):
            self.calls: list[tuple[int, int]] = []

        def locking(self, _fd, mode, byte_count):
            self.calls.append((mode, byte_count))

    fallback = FakeMsvcrt()
    monkeypatch.setattr(extraction_module, "_fcntl", None)
    monkeypatch.setattr(extraction_module, "_msvcrt", fallback)
    lock_path = tmp_path / "fallback.lock"
    lock_fd = extraction_module.os.open(
        lock_path,
        extraction_module.os.O_CREAT | extraction_module.os.O_RDWR,
        0o600,
    )
    try:
        extraction_module._acquire_extraction_file_lock(lock_fd, blocking=False)
        extraction_module._release_extraction_file_lock(lock_fd)
    finally:
        extraction_module.os.close(lock_fd)

    assert lock_path.read_bytes() == b"\0"
    assert fallback.calls == [
        (fallback.LK_NBLCK, 1),
        (fallback.LK_UNLCK, 1),
    ]


def test_schema_ready_assertion_open_never_runs_configure_or_migrations(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    owner = AssertionStore(db_path)
    schema_ready_token = owner.issue_schema_ready_token()
    owner.close()

    def forbidden_schema_work(*_args, **_kwargs):
        raise AssertionError("schema-ready worker must not scan, configure, or migrate")

    for name in (
        "configure_connection",
        "ensure_assertion_tables",
        "run_versioned_migrations",
        "verify_assertion_schema",
    ):
        monkeypatch.setattr(assertion_store_module, name, forbidden_schema_work)
    worker = AssertionStore(
        db_path,
        initialize_schema=False,
        schema_ready_token=schema_ready_token,
        deadline=101.0,
        deadline_clock=_Clock(),
    )
    try:
        assert worker.query_assertions() == []
    finally:
        worker.close(commit=False)
        messages.close()


def test_schema_ready_assertion_open_fails_closed_without_mutating_missing_schema(
    tmp_path,
):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    before = {
        tuple(row)
        for row in messages._conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name LIKE 'lcm_assertion%' ORDER BY type, name"
        ).fetchall()
    }
    with pytest.raises(ValueError, match="schema_ready_token"):
        AssertionStore(
            db_path,
            initialize_schema=False,
            deadline=101.0,
            deadline_clock=_Clock(),
        )
    after = {
        tuple(row)
        for row in messages._conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name LIKE 'lcm_assertion%' ORDER BY type, name"
        ).fetchall()
    }
    assert after == before == set()
    messages.close()


def test_schema_ready_assertion_open_never_creates_a_missing_database(tmp_path):
    db_path = tmp_path / "missing-parent" / "lcm.db"

    with pytest.raises(ValueError, match="schema_ready_token"):
        AssertionStore(
            db_path,
            initialize_schema=False,
            deadline=101.0,
            deadline_clock=_Clock(),
        )

    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_deadline_owned_assertion_open_cannot_opt_into_direct_migrations(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "missing-parent" / "lcm.db"
    migration_calls: list[object] = []

    def forbidden_migration(conn):
        migration_calls.append(conn)
        raise AssertionError("deadline-owned opens must reject migration before SQLite open")

    monkeypatch.setattr(
        assertion_store_module,
        "run_versioned_migrations",
        forbidden_migration,
    )
    with pytest.raises(ValueError, match="initialize_schema=False"):
        AssertionStore(
            db_path,
            initialize_schema=True,
            deadline=101.0,
            deadline_clock=_Clock(),
        )

    assert migration_calls == []
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_schema_ready_token_rejects_same_name_malformed_schema(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    owner = AssertionStore(db_path)
    schema_ready_token = owner.issue_schema_ready_token()
    owner.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TRIGGER lcm_assertion_message_update")
        conn.execute(
            """
            CREATE TRIGGER lcm_assertion_message_update
            AFTER UPDATE OF content ON messages
            BEGIN
                SELECT 1;
            END
            """
        )

    with pytest.raises(
        AssertionSchemaUnavailableError,
        match="changed after startup validation",
    ):
        AssertionStore(
            db_path,
            initialize_schema=False,
            schema_ready_token=schema_ready_token,
            deadline=101.0,
            deadline_clock=_Clock(),
        )
    messages.close()


def test_valid_second_startup_does_not_invalidate_schema_ready_token(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    first_owner = AssertionStore(db_path)
    schema_ready_token = first_owner.issue_schema_ready_token()
    first_owner.close()

    # The dashboard and gateway initialize the same profile independently.
    # A no-op second startup must not churn SQLite's schema cookie and strand
    # a token already issued by the other live process.
    second_owner = AssertionStore(db_path)
    second_owner.close()

    worker = AssertionStore(
        db_path,
        initialize_schema=False,
        schema_ready_token=schema_ready_token,
        deadline=101.0,
        deadline_clock=_Clock(),
    )
    try:
        assert worker.query_assertions() == []
    finally:
        worker.close(commit=False)
        messages.close()


def test_schema_ready_open_stops_before_a_phase_that_starts_after_deadline(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    owner = AssertionStore(db_path)
    schema_ready_token = owner.issue_schema_ready_token()
    owner.close()
    clock = _Clock()
    deadline = 101.0
    statements: list[str] = []
    original_connect = assertion_store_module.sqlite3.connect

    class ObservedConnection:
        def __init__(self, connection):
            self._connection = connection

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(str(sql).split()).casefold()
            statements.append(normalized)
            result = self._connection.execute(sql, *args, **kwargs)
            if normalized == "pragma journal_mode":
                clock.value = deadline
            return result

        def close(self):
            return self._connection.close()

    def observed_connect(*args, **kwargs):
        return ObservedConnection(original_connect(*args, **kwargs))

    monkeypatch.setattr(assertion_store_module.sqlite3, "connect", observed_connect)
    with pytest.raises(TimeoutError, match="deadline"):
        AssertionStore(
            db_path,
            initialize_schema=False,
            schema_ready_token=schema_ready_token,
            deadline=deadline,
            deadline_clock=clock,
        )

    assert "pragma journal_mode" in statements
    assert "pragma synchronous=full" not in statements
    assert "pragma schema_version" not in statements
    messages.close()


def test_assertion_prompt_expiry_prevents_provider_start(tmp_path, monkeypatch):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": "I prefer tea."},
    )
    snapshot = assertions.snapshot_source(store_id)
    clock = _Clock()
    deadline = 101.0
    query_kwargs: list[dict[str, object]] = []
    original_query = assertions.query_assertions

    def expiring_history_query(**kwargs):
        query_kwargs.append(dict(kwargs))
        rows = original_query(**kwargs)
        clock.value = deadline
        return rows

    provider_calls: list[float] = []
    monkeypatch.setattr(assertions, "query_assertions", expiring_history_query)
    extractor = ModelAssertionExtractor(
        assertions,
        timeout_seconds=30.0,
        payload_call=lambda _prompt, _model, timeout: provider_calls.append(timeout),
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            extractor(
                snapshot,
                deadline=deadline,
                deadline_clock=clock,
            )
        assert provider_calls == []
        assert extractor.call_count == 0
        assert query_kwargs and query_kwargs[0]["deadline"] == deadline
    finally:
        assertions.close()
        messages.close()


def test_assertion_provider_timeout_uses_time_left_after_prompt_history(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": "I prefer tea."},
    )
    snapshot = assertions.snapshot_source(store_id)
    clock = _Clock()
    deadline = 101.0
    original_query = assertions.query_assertions

    def slow_history_query(**kwargs):
        rows = original_query(**kwargs)
        clock.value = 100.4
        return rows

    observed_timeouts: list[float] = []

    def provider(_prompt, _model, timeout):
        observed_timeouts.append(timeout)
        return (
            f'{{"schema_version":1,"source_store_id":{snapshot.store_id},'
            f'"source_content_sha256":"{snapshot.content_sha256}",'
            '"assertions":[],"relations":[]}',
            1,
            1,
        )

    monkeypatch.setattr(assertions, "query_assertions", slow_history_query)
    extractor = ModelAssertionExtractor(
        assertions,
        timeout_seconds=30.0,
        payload_call=provider,
    )
    try:
        extraction = extractor(
            snapshot,
            deadline=deadline,
            deadline_clock=clock,
        )
        assert extraction.assertions == ()
        assert observed_timeouts == [pytest.approx(0.6)]
    finally:
        assertions.close()
        messages.close()


def test_default_assertion_provider_rechecks_deadline_after_lazy_routing(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": "I prefer tea."},
    )
    snapshot = assertions.snapshot_source(store_id)
    clock = _Clock()
    deadline = 101.0
    provider_calls: list[dict[str, object]] = []

    def forbidden_provider(**kwargs):
        provider_calls.append(dict(kwargs))
        raise AssertionError("provider must not start after routing exhausts the budget")

    agent_module = ModuleType("agent")
    auxiliary_module = ModuleType("agent.auxiliary_client")
    auxiliary_module.call_llm = forbidden_provider
    agent_module.auxiliary_client = auxiliary_module
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary_module)

    def expiring_route(_call_kwargs, _model):
        clock.value = deadline

    monkeypatch.setattr(
        model_routing_module,
        "apply_lcm_model_route",
        expiring_route,
    )
    extractor = ModelAssertionExtractor(
        assertions,
        payload_call=assertion_extraction_module._call_structured_assertion_llm,
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            extractor(
                snapshot,
                deadline=deadline,
                deadline_clock=clock,
            )
        assert provider_calls == []
        assert extractor.call_count == 0
    finally:
        assertions.close()
        messages.close()


def test_kwargs_assertion_provider_gets_deadline_but_counts_at_callback_entry(
    tmp_path,
):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": "I prefer tea."},
    )
    snapshot = assertions.snapshot_source(store_id)
    clock = _Clock()
    deadline = 101.0
    received_kwargs: list[dict[str, object]] = []

    def provider(_prompt, _model, _timeout, **kwargs):
        received_kwargs.append(dict(kwargs))
        return (
            f'{{"schema_version":1,"source_store_id":{snapshot.store_id},'
            f'"source_content_sha256":"{snapshot.content_sha256}",'
            '"assertions":[],"relations":[]}',
            2,
            3,
        )

    extractor = ModelAssertionExtractor(assertions, payload_call=provider)
    try:
        extractor(snapshot, deadline=deadline, deadline_clock=clock)
        assert extractor.call_count == 1
        assert len(received_kwargs) == 1
        assert received_kwargs[0]["deadline"] == deadline
        assert received_kwargs[0]["deadline_clock"] is clock
        assert received_kwargs[0]["cancel_event"] is None
        assert "on_provider_start" not in received_kwargs[0]
    finally:
        assertions.close()
        messages.close()


def test_late_assertion_response_accounts_usage_before_rejecting_payload(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": "I prefer tea."},
    )
    snapshot = assertions.snapshot_source(store_id)
    clock = _Clock()
    deadline = 101.0

    def late_provider(_prompt, _model, _timeout):
        clock.value = deadline
        return (
            f'{{"schema_version":1,"source_store_id":{snapshot.store_id},'
            f'"source_content_sha256":"{snapshot.content_sha256}",'
            '"assertions":[],"relations":[]}',
            7,
            9,
        )

    extractor = ModelAssertionExtractor(assertions, payload_call=late_provider)
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            extractor(snapshot, deadline=deadline, deadline_clock=clock)
        assert extractor.call_count == 1
        assert extractor.total_input_tokens == 7
        assert extractor.total_output_tokens == 9
        assert extractor.last_metrics is not None
    finally:
        assertions.close()
        messages.close()


def test_assertion_history_query_rejects_an_expired_deadline(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            assertions.query_assertions(deadline=100.0, deadline_clock=_Clock())
    finally:
        assertions.close()
        messages.close()


@pytest.mark.parametrize("context_name", ["_bounded_read_connection", "_bounded_write_connection"])
def test_busy_timeout_restore_never_masks_the_primary_failure(context_name):
    class _Result:
        def fetchone(self):
            return (30_000,)

    class _Connection:
        def __init__(self):
            self.restore_attempted = False

        def execute(self, sql):
            if sql == "PRAGMA busy_timeout":
                return _Result()
            if sql == "PRAGMA busy_timeout=30000":
                self.restore_attempted = True
                raise sqlite3.OperationalError("synthetic restore failure")
            return _Result()

    store = object.__new__(AssertionStore)
    store.read_only = False
    store._write_lock = threading.RLock()
    store._conn = _Connection()
    context = getattr(store, context_name)(busy_timeout_ms=1)

    with pytest.raises(ValueError, match="primary failure"):
        with context:
            raise ValueError("primary failure")
    assert store._conn.restore_attempted is True


def test_deadline_worker_close_rolls_back_instead_of_committing(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    assertions.connection.execute("BEGIN")
    assertions.connection.execute(
        "INSERT INTO lcm_migration_state(step_name, completed_at) VALUES(?, ?)",
        ("must-not-commit", 1.0),
    )
    assertions.close(commit=False)

    observer = AssertionStore(db_path, read_only=True)
    try:
        assert observer.connection.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = ?",
            ("must-not-commit",),
        ).fetchone() is None
    finally:
        observer.close()
        messages.close()
