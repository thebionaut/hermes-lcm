import threading
import time
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm import model_switch_preparation as preparation
from hermes_lcm import summary_serialization as serialization


@pytest.fixture
def engine():
    config = LCMConfig()
    config.summary_model = 'primary/model'
    config.summary_fallback_models = ['fallback/model']
    return SimpleNamespace(_config=config, _summary_circuit_breaker=object(),
                           _summary_spend_guard=object())


def test_preparation_serializer_keeps_long_source_without_externalization(engine, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('preparation must not write externalized files')
    monkeypatch.setattr(serialization, 'maybe_externalize_tool_output', forbidden)
    source = [
        {'role':'user', 'content':'A' * 5000 + ' USER-END'},
        {'role':'assistant', 'content':'B' * 5000 + ' ASSISTANT-END',
         'tool_calls':[{'id':'call', 'function':{'name':'read', 'arguments':'C' * 1000 + ' ARG-END'}}]},
        {'role':'tool', 'tool_call_id':'call', 'content':'D' * 5000 + ' TOOL-END'},
    ]
    text = serialization.serialize_summary_messages(source, config=engine._config, preparation_only=True)
    for end in ('USER-END', 'ASSISTANT-END', 'ARG-END', 'TOOL-END'):
        assert end in text
    assert 'A' * 5000 in text
    assert 'B' * 5000 in text
    assert 'C' * 1000 in text
    assert 'D' * 5000 in text
    assert '[truncated]' not in text


def test_ordinary_serialization_keeps_existing_bounded_behavior(engine):
    text = serialization.serialize_summary_messages(
        [{'role':'user', 'content':'A' * 5000}], config=engine._config)
    assert '[truncated]' in text
    assert len(text) < 5000


def test_summary_checks_both_prompts_all_routes_and_reuses_policy(engine, monkeypatch):
    calls = []
    checks = []
    def summarize(**kwargs):
        calls.append(kwargs)
        return 'verified summary', 1
    monkeypatch.setattr(preparation, 'summarize_with_escalation', summarize)
    event = threading.Event()
    result = preparation.prepare_model_switch_summary(
        engine, [{'role':'user', 'content':'task context ' * 100}], 256,
        time.monotonic() + 30, event,
        admit_request=lambda prompt, output, route: checks.append((prompt, output, route)) or True,
    )
    assert result == ('verified summary', 1)
    assert len(checks) == 4
    assert {row[2] for row in checks} == {'primary/model', 'fallback/model'}
    assert len(calls) == 1
    assert calls[0]['spend_guard'] is engine._summary_spend_guard
    assert calls[0]['circuit_breaker'] is engine._summary_circuit_breaker
    assert calls[0]['cancel_event'] is event
    assert calls[0]['fallback_models'] == engine._config.summary_fallback_models
    assert 'tools' not in calls[0]


def test_rejected_auxiliary_budget_makes_no_provider_call(engine, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail('invalid request must not reach provider')
    monkeypatch.setattr(preparation, 'summarize_with_escalation', forbidden)
    with pytest.raises(ValueError, match='auxiliary route budget'):
        preparation.prepare_model_switch_summary(
            engine, [{'role':'user', 'content':'source'}], 256,
            time.monotonic() + 30, threading.Event(), admit_request=lambda *args: False)


def test_truncation_fallback_is_not_a_success(engine, monkeypatch):
    monkeypatch.setattr(preparation, 'summarize_with_escalation', lambda **kwargs: ('truncated', 3))
    with pytest.raises(ValueError, match='truncation fallback refused'):
        preparation.prepare_model_switch_summary(
            engine, [{'role':'user', 'content':'source'}], 256,
            time.monotonic() + 30, threading.Event(), admit_request=lambda *args: True)


def test_cancelled_provider_result_cannot_escape(engine, monkeypatch):
    event = threading.Event()
    def summarize(**kwargs):
        event.set()
        return 'late', 1
    monkeypatch.setattr(preparation, 'summarize_with_escalation', summarize)
    with pytest.raises(InterruptedError):
        preparation.prepare_model_switch_summary(
            engine, [{'role':'user', 'content':'source'}], 256,
            time.monotonic() + 30, event, admit_request=lambda *args: True)
