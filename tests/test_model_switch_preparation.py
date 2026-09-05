import threading
import time
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm import model_switch_preparation as preparation
from hermes_lcm import summary_serialization as serialization


def test_dedicated_preparation_task_is_scoped_and_keeps_global_policy(engine, monkeypatch):
    from agent import auxiliary_client as auxiliary
    from hermes_lcm import escalation
    monkeypatch.setattr(auxiliary, '_get_auxiliary_task_config', lambda task: {
        'provider': 'openai-codex', 'model': 'gpt-5.6-luna', 'timeout': 90,
    } if task == 'model_switch_preparation' else {})
    calls = []
    def summarize(**kwargs):
        calls.append((escalation._SUMMARY_TASK.get(), kwargs))
        return 'prepared summary', 1
    monkeypatch.setattr(preparation, 'summarize_with_escalation', summarize)
    preparation.prepare_model_switch_summary(
        engine, [{'role': 'user', 'content': 'source ' * 100}], 256,
        time.monotonic() + 120, threading.Event(), custom_providers=[],
        admit_request=lambda *args: True,
    )
    task, call = calls[0]
    assert task == 'model_switch_preparation'
    assert call['model'] == ''
    assert call['fallback_models'] == []
    assert 89 <= call['timeout'] <= 90
    assert escalation._SUMMARY_TASK.get() == 'compression'
    assert engine._config.summary_model == 'primary/model'


def test_summary_task_restores_after_exception():
    from hermes_lcm.escalation import _SUMMARY_TASK, summary_task
    with pytest.raises(RuntimeError):
        with summary_task('model_switch_preparation'):
            assert _SUMMARY_TASK.get() == 'model_switch_preparation'
            raise RuntimeError('test')
    assert _SUMMARY_TASK.get() == 'compression'


def test_dedicated_preparation_rejects_implicit_provider(engine, monkeypatch):
    from agent import auxiliary_client as auxiliary
    monkeypatch.setattr(auxiliary, '_get_auxiliary_task_config', lambda task: {
        'provider': 'auto', 'model': 'some-model',
    })
    with pytest.raises(ValueError, match='explicit provider'):
        preparation._summary_route_config(engine._config)


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
        time.monotonic() + 30, event, custom_providers=[],
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
            time.monotonic() + 30, threading.Event(), admit_request=lambda *args: False, custom_providers=[])


def test_truncation_fallback_is_not_a_success(engine, monkeypatch):
    monkeypatch.setattr(preparation, 'summarize_with_escalation', lambda **kwargs: ('truncated', 3))
    with pytest.raises(ValueError, match='truncation fallback refused'):
        preparation.prepare_model_switch_summary(
            engine, [{'role':'user', 'content':'source'}], 256,
            time.monotonic() + 30, threading.Event(), admit_request=lambda *args: True, custom_providers=[])


def test_cancelled_provider_result_cannot_escape(engine, monkeypatch):
    event = threading.Event()
    def summarize(**kwargs):
        event.set()
        return 'late', 1
    monkeypatch.setattr(preparation, 'summarize_with_escalation', summarize)
    with pytest.raises(InterruptedError):
        preparation.prepare_model_switch_summary(
            engine, [{'role':'user', 'content':'source'}], 256,
            time.monotonic() + 30, event, admit_request=lambda *args: True, custom_providers=[])


def test_host_fallback_is_guarded_inside_lcm_preparation(engine, monkeypatch):
    from agent import auxiliary_client, context_admission
    monkeypatch.setattr(context_admission, 'resolve_context_budget',
                        lambda **kw: SimpleNamespace(metadata_known=False))
    sent = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: sent.append(kw))))

    def summarize(**kwargs):
        return auxiliary_client._relay_sync_completion(
            client, {'model': 'host-fallback', 'messages': [{'role': 'user', 'content': 'source'}]},
            provider='zai')

    monkeypatch.setattr(preparation, 'summarize_with_escalation', summarize)
    with pytest.raises(ValueError, match='not admitted'):
        preparation.prepare_model_switch_summary(
            engine, [{'role': 'user', 'content': 'source'}], 256,
            time.monotonic() + 30, threading.Event(),
            admit_request=lambda *args: True, custom_providers=[])
    assert sent == []


def test_summary_policy_resolves_runtime_and_checks_complete_prompts(engine, monkeypatch):
    from agent import auxiliary_client as auxiliary, context_admission, model_metadata
    calls = []
    client = SimpleNamespace(base_url='https://configured.invalid/v1')
    monkeypatch.setattr(auxiliary, '_resolve_task_provider_model',
                        lambda task, **kw: ('zai', kw.get('model'), None, None, None))
    monkeypatch.setattr(auxiliary, '_get_cached_client', lambda provider, model, **kw: (client, model))
    monkeypatch.setattr(auxiliary, '_effective_provider_for_client', lambda client, provider: provider)
    monkeypatch.setattr(auxiliary, '_get_task_extra_body', lambda task: {})
    def build(provider, model, messages, **kw):
        calls.append((provider, model, messages, kw['max_tokens']))
        return {'model': model, 'messages': messages, 'max_tokens': kw['max_tokens']}
    monkeypatch.setattr(auxiliary, '_build_call_kwargs', build)
    monkeypatch.setattr(context_admission, 'resolve_context_budget', lambda **kw:
                        SimpleNamespace(metadata_known=True, safe_input_tokens=80000, context_window=100000))
    monkeypatch.setattr(model_metadata, 'estimate_request_tokens_rough', lambda *a, **kw: 1000)
    policy = preparation.model_switch_summary_policy(
        engine, custom_providers=[], deadline=time.monotonic() + 30, cancel_event=threading.Event())
    assert policy.source_limit == 80000
    assert policy.admit_chunk([{'role': 'user', 'content': 'original details'}])
    assert len(calls) == 4
    assert all('original details' in row[2][0]['content'] for row in calls)
    assert {row[3] for row in calls} == {24000, int(12000 * engine._config.l2_budget_ratio) * 2}
