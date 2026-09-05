"""Offline provider-route and outcome tracing tests for summarization."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm import escalation


def _response(content, **message_fields):
    message = SimpleNamespace(content=content, **message_fields)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _install_auxiliary_client(monkeypatch, call_llm):
    agent = ModuleType("agent")
    agent.__path__ = []
    auxiliary = ModuleType("agent.auxiliary_client")
    auxiliary.call_llm = call_llm
    agent.auxiliary_client = auxiliary
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)


def test_task_default_route_info_records_actual_host_fallback(monkeypatch):
    def fake_call_llm(*, route_info=None, **_kwargs):
        assert route_info is not None
        route_info.update(
            {"provider": "openai-codex", "model": "gpt-5.6-luna"}
        )
        return _response("short durable summary")

    _install_auxiliary_client(monkeypatch, fake_call_llm)
    trace = {}

    result = escalation.summarize_with_escalation(
        "sensitive fixture source " * 100,
        source_tokens=600,
        token_budget=60,
        trace=trace,
    )

    assert result == ("short durable summary", 1)
    assert trace["actual_provider"] == "openai-codex"
    assert trace["actual_model"] == "gpt-5.6-luna"
    assert trace["attribution_available"] is True
    # A task-default route reveals the final provider, but current Hermes does
    # not expose whether that route was its primary or internal fallback.
    assert trace["fallback_used"] is None
    attempt = trace["attempts"][0]
    assert attempt["actual_provider"] == "openai-codex"
    assert attempt["actual_model"] == "gpt-5.6-luna"
    assert attempt["logical_route_ordinal"] == 0
    assert attempt["outcome"] == "success"
    assert attempt["failure_reason"] == ""
    assert attempt["finished_at"] >= attempt["started_at"] > 0
    assert attempt["latency_ms"] >= 0
    assert "sensitive fixture source" not in json.dumps(trace)


def test_route_info_is_omitted_when_host_signature_does_not_declare_it(monkeypatch):
    calls = 0

    def legacy_call_llm(*, task, messages, temperature, max_tokens):
        nonlocal calls
        calls += 1
        assert task == "compression"
        assert messages and temperature == 0.3 and max_tokens == 200
        return _response("legacy host summary")

    _install_auxiliary_client(monkeypatch, legacy_call_llm)
    route_info = {}
    attempt = {}

    result = escalation._call_llm_for_summary(
        "fixture",
        200,
        route_info=route_info,
        trace=attempt,
    )

    assert result == "legacy host summary"
    assert calls == 1
    assert route_info == {}
    assert attempt["attribution_available"] is False
    assert attempt["actual_provider"] == ""
    assert attempt["actual_model"] == ""


def test_type_error_is_not_retried_or_persisted(monkeypatch):
    calls = 0

    def fake_call_llm(*, route_info=None, **_kwargs):
        nonlocal calls
        calls += 1
        route_info.update({"provider": "zai", "model": "glm-test"})
        raise TypeError("sensitive-marker-must-not-appear")

    _install_auxiliary_client(monkeypatch, fake_call_llm)
    attempt = {}

    result = escalation._call_llm_for_summary(
        "fixture",
        200,
        route_info={},
        trace=attempt,
    )

    assert result is None
    assert calls == 1
    assert attempt["failure_reason"] == "provider_error"
    assert "sensitive-marker-must-not-appear" not in json.dumps(attempt)


class _StatusFailure(Exception):
    def __init__(self, status_code):
        super().__init__("sensitive provider response")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        (TimeoutError("private timeout detail"), "provider_timeout"),
        (ConnectionError("private endpoint detail"), "provider_connection_error"),
        (_StatusFailure(401), "provider_auth_error"),
        (_StatusFailure(403), "provider_auth_error"),
        (_StatusFailure(402), "provider_quota"),
        (_StatusFailure(429), "provider_rate_limit"),
    ],
)
def test_provider_failures_use_safe_precise_reason_codes(
    monkeypatch,
    failure,
    expected_reason,
):
    def fake_call_llm(*, route_info=None, **_kwargs):
        route_info.update({"provider": "fixture-provider", "model": "fixture-model"})
        raise failure

    _install_auxiliary_client(monkeypatch, fake_call_llm)
    attempt = {}
    result = escalation._call_llm_for_summary(
        "fixture",
        200,
        route_info={},
        trace=attempt,
    )

    assert result is None
    assert attempt["failure_reason"] == expected_reason
    assert attempt["outcome"] == "failure"
    serialized = json.dumps(attempt)
    assert "private" not in serialized
    assert "sensitive provider response" not in serialized


def test_invalid_response_and_reasoning_only_are_distinct(monkeypatch):
    responses = iter(
        [
            SimpleNamespace(choices=[]),
            _response(None, reasoning="private chain of thought"),
        ]
    )

    def fake_call_llm(*, route_info=None, **_kwargs):
        route_info.update({"provider": "fixture-provider", "model": "fixture-model"})
        return next(responses)

    _install_auxiliary_client(monkeypatch, fake_call_llm)
    invalid_trace = {}
    reasoning_trace = {}

    assert escalation._call_llm_for_summary(
        "fixture", 200, route_info={}, trace=invalid_trace
    ) is None
    assert invalid_trace["failure_reason"] == "invalid_provider_response"

    assert escalation._call_llm_for_summary(
        "fixture", 200, route_info={}, trace=reasoning_trace
    ) == ""
    assert reasoning_trace["failure_reason"] == "reasoning_only_output"
    assert "private chain of thought" not in json.dumps(reasoning_trace)


def test_lcm_fallback_route_and_non_compressing_outcomes_are_traced(monkeypatch):
    calls = []

    def fake_summary_call(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
    ):
        del timeout
        calls.append(model)
        provider, _, model_name = model.partition("/")
        route_info.update({"provider": provider, "model": model_name or model})
        if model == "primary/model-a":
            return "verbose " * 1_000
        return "short fallback summary"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", fake_summary_call)
    trace = {}

    result = escalation.summarize_with_escalation(
        "source " * 100,
        source_tokens=200,
        token_budget=40,
        model="primary/model-a",
        fallback_models=["fallback/model-b"],
        trace=trace,
    )

    assert result == ("short fallback summary", 1)
    assert calls == ["primary/model-a", "fallback/model-b"]
    assert trace["attempts"][0]["failure_reason"] == "non_compressing_output"
    selected = trace["attempts"][1]
    assert selected["outcome"] == "success"
    assert selected["actual_provider"] == "fallback"
    assert selected["actual_model"] == "model-b"
    assert selected["fallback_used"] is True
    assert trace["terminal_reason"] == ""


@pytest.mark.parametrize(
    ("guard_factory", "expected_reason"),
    [
        (
            lambda: escalation.SummaryCircuitBreaker(
                failure_threshold=1,
                cooldown_seconds=60,
            ),
            "provider_circuit_open",
        ),
        (
            lambda: escalation.SummarySpendGuard(
                max_calls=1,
                window_seconds=60,
                backoff_seconds=60,
            ),
            "provider_spend_guard",
        ),
    ],
)
def test_open_guards_skip_provider_with_precise_reason(
    monkeypatch,
    guard_factory,
    expected_reason,
):
    calls = []

    def must_not_call(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("provider should be skipped")

    monkeypatch.setattr(escalation, "_call_llm_for_summary", must_not_call)
    guard = guard_factory()
    kwargs = {}
    if isinstance(guard, escalation.SummaryCircuitBreaker):
        guard.record_failure("primary/model")
        kwargs["circuit_breaker"] = guard
    else:
        guard.record_call()
        kwargs["spend_guard"] = guard
    trace = {}

    summary, level = escalation.summarize_with_escalation(
        "source " * 100,
        source_tokens=200,
        token_budget=40,
        model="primary/model",
        trace=trace,
        **kwargs,
    )

    assert summary and level == 3
    assert calls == []
    assert trace["terminal_reason"] == expected_reason
    assert {attempt["failure_reason"] for attempt in trace["attempts"]} == {
        expected_reason
    }
    assert {attempt["outcome"] for attempt in trace["attempts"]} == {"skipped"}


def test_non_compressing_and_insufficient_source_have_distinct_terminal_reasons(
    monkeypatch,
):
    def verbose_summary(*_args, **_kwargs):
        return "verbose " * 1_000

    monkeypatch.setattr(escalation, "_call_llm_for_summary", verbose_summary)
    non_compressing_trace = {}
    summary, level = escalation.summarize_with_escalation(
        "source " * 100,
        source_tokens=200,
        token_budget=40,
        trace=non_compressing_trace,
    )
    assert summary and level == 3
    assert non_compressing_trace["terminal_reason"] == "non_compressing_output"

    def must_not_call(*_args, **_kwargs):
        raise AssertionError("empty source must not call a provider")

    monkeypatch.setattr(escalation, "_call_llm_for_summary", must_not_call)
    insufficient_trace = {}
    assert escalation.summarize_with_escalation(
        "",
        source_tokens=0,
        token_budget=40,
        trace=insufficient_trace,
    ) == ("", 3)
    assert (
        insufficient_trace["terminal_reason"]
        == "insufficient_compressible_source"
    )
    assert insufficient_trace["attempts"][0]["outcome"] == "skipped"
