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
