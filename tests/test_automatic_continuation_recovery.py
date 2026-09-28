import importlib.util
import sys
from pathlib import Path

import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.dag import SummaryNode
from hermes_lcm.selective_retrieval import automatic_recovery


@pytest.fixture
def pilot(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path/'pilot.db'),
                                     selective_retrieval_session_ids='pilot'))
    engine.on_session_start('pilot')
    raw = 'User correction: BX-008 quantity is 42.750 mL, not 40 mL. Do not edit the label.'
    sid = engine._store.append('pilot', {'role': 'user', 'content': raw})
    node = SummaryNode(session_id='pilot', depth=0, summary='Omitted areas: BX-008 quantity correction.',
                       source_ids=[sid], source_type='messages', token_count=15)
    nid = engine._dag.add_node(node)
    payload = {'session_id': 'pilot', 'user_message': 'What is the corrected BX-008 quantity?',
               'conversation_history': [{'role': 'user', 'content': f'[Recent Summary (d0, node {nid})]\n{node.summary}'}]}
    yield engine, payload, raw
    engine.shutdown()


def test_hook_automatically_recovers_missing_middle(pilot):
    engine, payload, raw = pilot
    spec = importlib.util.spec_from_file_location('hermes_lcm_hook_test', Path(__file__).resolve().parents[1]/'__init__.py', submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'hermes_lcm'
    spec.loader.exec_module(module)
    result = module._pre_llm_context(engine, 'policy', payload)
    assert raw in result['context']
    assert engine._last_selective_recovery_trace['retrieved'] == 1
    assert engine._query_views is None
    assert engine._adaptive_retrieval is None


def test_irrelevant_and_sufficient_trigger_no_reads(pilot, monkeypatch):
    engine, payload, raw = pilot
    monkeypatch.setattr(engine._store, 'get', lambda *a: pytest.fail('unexpected raw read'))
    assert automatic_recovery(engine, dict(payload, user_message='Write a poem'))['retrieved'] == 0
    history = payload['conversation_history'] + [{'role':'user','content':raw}]
    assert automatic_recovery(engine, dict(payload, conversation_history=history))['retrieved'] == 0


def test_exact_session_and_revocation(pilot):
    engine, payload, raw = pilot
    assert automatic_recovery(engine, dict(payload, session_id='other'))['retrieved'] == 0
    engine._config.selective_retrieval_session_ids = ''
    assert automatic_recovery(engine, payload)['retrieved'] == 0


def test_foreign_leaf_is_not_injected(pilot):
    engine, payload, raw = pilot
    sid = engine._store.append('foreign', {'role':'user','content':'BX-008 secret quantity'})
    node = SummaryNode(session_id='pilot', depth=0, summary='Omitted areas: BX-008 quantity.',
                       source_ids=[sid], source_type='messages', token_count=12)
    nid = engine._dag.add_node(node)
    payload['conversation_history'] = [{'role':'user','content': f'[Recent Summary (d0, node {nid})]\n{node.summary}'}]
    result = automatic_recovery(engine, payload)
    assert result['retrieved'] == 0 and 'secret' not in result['context']


def test_bounds_and_fabricated_marker(pilot):
    engine, payload, raw = pilot
    payload['conversation_history'][0]['content'] = '[Recent Summary (d0, node 999999)]\nOmitted areas: BX-008 quantity.'
    assert automatic_recovery(engine, payload)['retrieved'] == 0
