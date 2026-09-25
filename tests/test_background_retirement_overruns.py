"""Admission follows real daemon completion, including queued maintenance."""
import threading
import time

import pytest

from hermes_cli import backend_retirement, web_server_idle_proof
from hermes_lcm import tools, tokens, embedding_provider
from hermes_lcm.engine import _RollupMaintenanceScheduler


@pytest.fixture
def gate(monkeypatch):
    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setattr(web_server_idle_proof, "idle_proof", lambda: {"idle": True})
    return fence


def _idle(gate):
    deadline = time.monotonic() + 3
    while gate.active_count() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert gate.active_count() == 0


@pytest.mark.parametrize("operation", ["semantic", "encoder"])
def test_timeout_keeps_admission_until_abandoned_worker_exits(operation, gate):
    entered, finish = threading.Event(), threading.Event()

    def held():
        entered.set()
        assert finish.wait(8)

    try:
        if operation == "semantic":
            with pytest.raises(TimeoutError):
                tools._run_within_deadline(held, remaining_s=0.02, name="fixture-semantic",
                                           worker_slots=threading.BoundedSemaphore(1))
        else:
            with pytest.raises(embedding_provider.EmbeddingProviderError):
                embedding_provider._run_blocking_with_deadline(held, timeout=0.02, provider="fixture")
        assert entered.wait(3)
        assert gate.prepare()["ok"] is False
    finally:
        finish.set()
        _idle(gate)


def test_rollup_queue_and_followup_hold_one_admission_per_key(gate):
    scheduler = _RollupMaintenanceScheduler()
    entered, finish, second = threading.Event(), threading.Event(), threading.Event()

    def held():
        entered.set()
        assert finish.wait(8)

    try:
        assert scheduler.schedule(("fixture", "a"), held)
        assert entered.wait(3)
        assert scheduler.schedule(("fixture", "a"), lambda: None)
        assert scheduler.schedule(("fixture", "a"), lambda: None)
        assert scheduler.schedule(("fixture", "b"), second.set)
        assert gate.active_count() == 2
        assert gate.prepare()["ok"] is False
        finish.set()
        assert second.wait(3)
        _idle(gate)
    finally:
        finish.set()
        scheduler.drain({("fixture", "a"), ("fixture", "b")}, timeout=3)


def test_prepared_retirement_declines_rollup_without_queueing(gate):
    scheduler = _RollupMaintenanceScheduler()
    called = threading.Event()
    prepared = gate.prepare()
    assert scheduler.schedule(("fixture", "new"), called.set) is False
    assert not called.is_set()
    assert not scheduler._jobs
    assert gate.cancel(prepared["token"])["ok"] is True
    assert scheduler.schedule(("fixture", "new"), called.set)
    assert called.wait(3)
    assert scheduler.drain({("fixture", "new")}, timeout=3)
    _idle(gate)


def test_lazy_token_loader_keeps_admission_but_defers_during_retirement(monkeypatch, gate):
    entered, finish = threading.Event(), threading.Event()

    def load():
        entered.set()
        assert finish.wait(8)
        return None

    monkeypatch.setattr(tokens, "_encoder_ready", False)
    monkeypatch.setattr(tokens, "_encoder_thread", None)
    monkeypatch.setattr(tokens, "_ENCODER_FIRST_WAIT_S", 0.02)
    monkeypatch.setattr(tokens, "_load_encoder", load)
    prepared = gate.prepare()
    try:
        assert tokens._get_encoder() is None
        assert not entered.is_set()
        assert gate.cancel(prepared["token"])["ok"] is True
        assert tokens._get_encoder() is None
        assert entered.wait(3)
        assert gate.prepare()["ok"] is False
    finally:
        finish.set()
        if tokens._encoder_thread is not None:
            tokens._encoder_thread.join(3)
        _idle(gate)
