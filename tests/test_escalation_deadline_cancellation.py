"""Absolute-deadline and cancellation contracts for summary escalation."""

from __future__ import annotations

import threading

import pytest

from hermes_lcm import escalation


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_late_primary_result_cannot_launch_fallback_or_l2(monkeypatch):
    clock = _Clock()
    calls: list[tuple[str, float | None]] = []

    def ignores_timeout(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
    ):
        del route_info
        calls.append((model, timeout))
        clock.advance(6.0)
        return "late provider summary"

    monkeypatch.setattr(escalation.time, "monotonic", clock)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", ignores_timeout)

    summary, level = escalation.summarize_with_escalation(
        "source material " * 100,
        source_tokens=500,
        token_budget=100,
        model="primary/model-a",
        fallback_models=["fallback/model-b"],
        timeout=30.0,
        deadline=105.0,
    )

    assert summary
    assert level == 3
    assert calls == [("primary/model-a", pytest.approx(5.0))]


def test_cancellation_after_primary_stops_fallback_and_l2(monkeypatch):
    cancelled = threading.Event()
    calls: list[str] = []

    def cancelled_primary(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
    ):
        del timeout, route_info
        calls.append(model)
        cancelled.set()
        return None

    monkeypatch.setattr(escalation, "_call_llm_for_summary", cancelled_primary)

    summary, level = escalation.summarize_with_escalation(
        "source material " * 100,
        source_tokens=500,
        token_budget=100,
        model="primary/model-a",
        fallback_models=["fallback/model-b"],
        cancel_event=cancelled,
    )

    assert summary
    assert level == 3
    assert calls == ["primary/model-a"]


def test_each_route_timeout_uses_absolute_remaining_budget(monkeypatch):
    clock = _Clock()
    calls: list[tuple[str, float | None]] = []

    def routed_summary(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
    ):
        del route_info
        calls.append((model, timeout))
        if model == "primary/model-a":
            clock.advance(3.0)
            return None
        return "short fallback summary"

    monkeypatch.setattr(escalation.time, "monotonic", clock)
    monkeypatch.setattr(escalation, "_call_llm_for_summary", routed_summary)

    assert escalation.summarize_with_escalation(
        "source material " * 100,
        source_tokens=500,
        token_budget=100,
        model="primary/model-a",
        fallback_models=["fallback/model-b"],
        timeout=30.0,
        deadline=110.0,
    ) == ("short fallback summary", 1)
    assert calls == [
        ("primary/model-a", pytest.approx(10.0)),
        ("fallback/model-b", pytest.approx(7.0)),
    ]


def test_legacy_custom_summary_callable_does_not_receive_new_keywords(monkeypatch):
    observed: dict[str, object] = {}

    def legacy_summary(_prompt, _max_tokens, model="", timeout=None):
        observed.update(model=model, timeout=timeout)
        return "legacy compatible summary"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", legacy_summary)

    assert escalation.summarize_with_escalation(
        "source material " * 100,
        source_tokens=500,
        token_budget=100,
        model="primary/model-a",
        timeout=30.0,
        deadline=escalation.time.monotonic() + 10.0,
    ) == ("legacy compatible summary", 1)
    assert observed["model"] == "primary/model-a"
    assert 0 < float(observed["timeout"]) <= 10.0
