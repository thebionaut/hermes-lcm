"""Real Hermes process retirement must include plugin background work."""
import sqlite3
import threading
import time

import pytest

from hermes_cli import backend_retirement, web_server_idle_proof
from hermes_lcm import background_compaction as background, db_bootstrap
from hermes_lcm.config import LCMConfig


@pytest.fixture
def gate(monkeypatch):
    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    return fence


def _until(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def _fake_store(monkeypatch, *, work=None, close=None):
    opened = threading.Event()
    ran = threading.Event()

    class Store:
        def __init__(self, *args, **kwargs):
            opened.set()

        def recover_expired_leases(self):
            pass

        def run_next_job(self, **kwargs):
            ran.set()
            if work:
                work()
            return None

        def _next_maintenance_delay(self):
            return 30.0

        def close(self):
            if close:
                close()

    monkeypatch.setattr(background, "BackgroundCompactionStore", Store)
    return opened, ran


def test_active_compaction_and_database_cleanup_prevent_retirement(tmp_path, monkeypatch, gate):
    finish = threading.Event()
    closing, close_done = threading.Event(), threading.Event()

    def work():
        assert finish.wait(3)

    def close():
        closing.set()
        assert close_done.wait(3)

    _, ran = _fake_store(monkeypatch, work=work, close=close)
    runtime = background._BackgroundRuntime("retirement-active", tmp_path / "lcm.db", LCMConfig())
    try:
        assert ran.wait(3)
        assert gate.prepare()["ok"] is False
        finish.set()
        assert closing.wait(3)
        assert gate.prepare()["ok"] is False
        close_done.set()
        _until(lambda: gate.active_count() == 0)
        assert gate.prepare()["ok"] is True
    finally:
        finish.set()
        close_done.set()
        assert runtime.stop()


def test_prepared_retirement_defers_store_open_then_cancel_resumes(tmp_path, monkeypatch, gate):
    prepared = gate.prepare()
    opened, ran = _fake_store(monkeypatch)
    runtime = background._BackgroundRuntime("retirement-pending", tmp_path / "lcm.db", LCMConfig())
    try:
        assert not opened.wait(0.1)
        assert gate.cancel(prepared["token"])["ok"] is True
        runtime.notify()
        assert ran.wait(3)
    finally:
        assert runtime.stop()
    assert gate.active_count() == 0


def test_committed_retirement_prevents_new_background_work(tmp_path, monkeypatch, gate):
    prepared = gate.prepare()
    assert gate.commit(prepared["token"])["ok"] is True
    opened, _ = _fake_store(monkeypatch)
    runtime = background._BackgroundRuntime("retirement-committed", tmp_path / "lcm.db", LCMConfig())
    try:
        runtime.notify()
        assert not opened.wait(0.15)
    finally:
        assert runtime.stop()
    assert gate.active_count() == 0


def _fts_fixture(tmp_path):
    conn = sqlite3.connect(tmp_path / "scan.db")
    conn.executescript("CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT);"
                       "INSERT INTO messages VALUES(1, 'preserved data');")
    spec = db_bootstrap.ExternalContentFtsSpec(
        table_name="messages_fts", content_table="messages",
        content_rowid="store_id", indexed_column="content", trigger_sqls=(),
    )
    db_bootstrap.ensure_external_content_fts(conn, spec)
    return conn, spec


def test_deep_integrity_scan_prevents_retirement_until_metadata_finishes(tmp_path, monkeypatch, gate):
    conn, spec = _fts_fixture(tmp_path)
    entered, finish = threading.Event(), threading.Event()

    def check(*args):
        entered.set()
        assert finish.wait(3)
        return {"status": "pass"}

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", check)
    try:
        assert db_bootstrap._dispatch_background_integrity_scan(conn, spec)
        assert entered.wait(3)
        assert gate.prepare()["ok"] is False
        finish.set()
        db_bootstrap.join_background_integrity_scans(timeout=3)
        assert gate.active_count() == 0
        assert gate.prepare()["ok"] is True
        assert db_bootstrap._load_scan_started_at(conn, spec) is None
    finally:
        finish.set()
        db_bootstrap.join_background_integrity_scans(timeout=3)
        conn.close()


def test_retirement_defers_scan_without_claim_or_synchronous_fallback(tmp_path, monkeypatch, gate):
    conn, spec = _fts_fixture(tmp_path)
    entered = threading.Event()
    monkeypatch.setattr(db_bootstrap, "_run_background_integrity_scan", lambda *args: entered.set())
    prepared = gate.prepare()
    assert gate.commit(prepared["token"])["ok"] is True
    try:
        assert db_bootstrap._dispatch_background_integrity_scan(conn, spec) is True
        assert not entered.wait(0.1)
        assert db_bootstrap._load_scan_started_at(conn, spec) is None
    finally:
        db_bootstrap.join_background_integrity_scans(timeout=3)
        conn.close()


def test_timed_out_provider_retains_admission_after_runtime_stops(tmp_path, monkeypatch, gate):
    from hermes_lcm import escalation
    from hermes_lcm.dag import SummaryDAG
    from test_background_runtime_cancellation_isolation import _config, _seed_conversation, _enqueue

    db_path = tmp_path / "provider-retirement.db"
    SummaryDAG(db_path).close()
    ids = _seed_conversation(db_path, conversation="held", session="held", content="held provider")
    config = _config(db_path)
    writer = background.BackgroundCompactionStore(db_path, config=config)
    _enqueue(writer, ids, conversation="held", session="held")
    entered, finish = threading.Event(), threading.Event()

    def summarize(**kwargs):
        entered.set()
        assert finish.wait(8)
        return "preserved summary", 1

    monkeypatch.setattr(escalation, "summarize_with_escalation", summarize)
    runtime = background._BackgroundRuntime("provider-retirement", db_path, config)
    try:
        assert entered.wait(3)
        runtime.request_stop()
        assert runtime.wait_stopped(timeout=3)
        # The bounded job has returned, but its provider and durable-permit
        # cleanup guardian still exist and must block process retirement.
        assert not finish.is_set()
        assert gate.prepare()["ok"] is False
        finish.set()
        _until(lambda: gate.active_count() == 0)
        assert gate.prepare()["ok"] is True
    finally:
        finish.set()
        runtime.stop()
        writer.close()


def test_thread_start_failure_releases_admission(tmp_path, monkeypatch, gate):
    from hermes_lcm.host_admission import start_background_thread

    def fail(_thread):
        raise RuntimeError("fixture thread start failure")

    monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        start_background_thread(target=lambda: None, name="unstarted-fixture")
    assert gate.active_count() == 0
    conn, spec = _fts_fixture(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="fixture"):
            db_bootstrap._dispatch_background_integrity_scan(conn, spec)
        assert gate.active_count() == 0
        assert not any(key[0] == str(tmp_path / "scan.db") for key in db_bootstrap._integrity_scan_threads)
    finally:
        conn.close()


def test_missing_legacy_host_is_compatible_but_broken_host_import_is_not(monkeypatch):
    import builtins
    from hermes_lcm.host_admission import acquire_background_admission

    real_import = builtins.__import__
    missing = "hermes_cli.backend_retirement"

    def importing(name, *args, **kwargs):
        if name == "hermes_cli.backend_retirement":
            raise ModuleNotFoundError("fixture missing module", name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    permit = acquire_background_admission()
    assert permit is not None
    permit.release()
    permit.release()
    missing = "unrelated_missing_dependency"
    with pytest.raises(ModuleNotFoundError):
        acquire_background_admission()
