"""Final SDK-wire capture for duplicate projection; no real provider calls."""
import json
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize('case', ['exact_replay', 'restamped_tool_replay', 'covered_frontier'])
def test_duplicate_history_and_legitimate_repeats_at_final_wire(tmp_path, monkeypatch, case):
    # The plugin's tools.py must not shadow the host's tools package.
    from pathlib import Path
    import agent
    monkeypatch.syspath_prepend(str(Path(agent.__file__).resolve().parent.parent))
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from agent.transports.chat_completions import ChatCompletionsTransport

    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / 'wire.db'),
        proactive_recall_enabled=False, sensitive_patterns_enabled=False))
    engine._session_id = 'wire-pilot'
    engine._conversation_id = 'wire-pilot'
    messages = []
    for turn in range(2):
        messages.extend([
            {'role': 'user', 'content': 'Run the read-only check again.', 'timestamp': 1790000000.0 + turn * 10},
            {'role': 'assistant', 'content': '', 'timestamp': 1790000001.0 + turn * 10,
             'tool_calls': [{'id': f'call-wire-{turn}', 'type': 'function',
                            'function': {'name': 'read_check', 'arguments': '{}'}}]},
            {'role': 'tool', 'content': 'ok', 'tool_call_id': f'call-wire-{turn}',
             'timestamp': 1790000002.0 + turn * 10},
            {'role': 'assistant', 'content': 'Check completed.', 'timestamp': 1790000003.0 + turn * 10},
        ])
    captures = []
    def capture(request):
        captures.append(bytes(request.content))
        return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion',
            'created': 0, 'model': 'fixture', 'choices': [
                {'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
    client = OpenAI(api_key='fixture-not-a-credential', base_url='https://fixture.invalid/v1',
                    max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(capture)))
    try:
        # Historical duplicate rows are deliberately retained. Only replay projects.
        first_ids = engine._store.append_batch('wire-pilot', messages)
        if case == 'restamped_tool_replay':
            replay = [dict(m, timestamp=1790500000.0) for m in messages]
        else:
            replay = messages
        engine._store.append_batch('wire-pilot', replay)
        frontier = first_ids[3] if case == 'covered_frontier' else 0
        projected = engine._project_durable_active_tail(frontier, preserve_messages=list(messages))
        transport = ChatCompletionsTransport()
        wire_messages = transport.convert_messages(projected)
        agent = SimpleNamespace(api_mode='chat_completions', provider='openai',
                                _goal_budget_contract=None)
        _dispatch_nonstreaming_api_request(agent,
            {'model': 'fixture', 'messages': wire_messages}, make_client=lambda *a, **k: client)
        assert len(captures) == 1
        body = json.loads(captures[0])
        calls = Counter(tc['id'] for m in body['messages'] for tc in m.get('tool_calls', []))
        results = Counter(m['tool_call_id'] for m in body['messages'] if m['role'] == 'tool')
        assert calls == results == ({'call-wire-1': 1} if case == 'covered_frontier' else
                                    {'call-wire-0': 1, 'call-wire-1': 1})
        if case == 'exact_replay':
            assert sum(m.get('content') == 'Run the read-only check again.' for m in body['messages']) == 2
        assert engine._store.get_session_count('wire-pilot') == len(messages) * 2
        (tmp_path / f'{case}-wire-capture.json').write_bytes(captures[0])
    finally:
        client.close()
        engine.shutdown()
