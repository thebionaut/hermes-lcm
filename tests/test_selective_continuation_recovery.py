"""Offline omitted-evidence pilot gates: no provider calls or shared archive edits."""
import json

import pytest

from hermes_lcm.adaptive_retrieval import MAX_RETRIEVAL_ROUNDS, _bounded_tool_args
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.selective_retrieval import controller_for, session_allowed


def test_exact_session_gate_and_default_off(tmp_path):
    assert session_allowed('pilot,second', 'pilot')
    assert not session_allowed('pilot,second', 'pilot-other')
    assert not session_allowed('*', 'pilot')
    assert not session_allowed('pilot', '')
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / 'off.db')))
    try:
        engine.on_session_start('pilot')
        assert controller_for(engine) is None
        result = json.loads(engine.handle_tool_call('lcm_retrieve', {'action': 'start'}))
        assert result['status'] == 'disabled'
        assert engine._query_views is None
    finally:
        engine.shutdown()


def test_omitted_correction_recovers_exact_source_without_optional_stores(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / 'pilot.db'),
                       selective_retrieval_session_ids='pilot')
    engine = LCMEngine(config=config)
    engine.on_session_start('pilot', conversation_id='pilot-conversation')
    try:
        raw = 'User correction: quantity is 42.750 mL, not 40 mL. Next action: verify label Q-7.'
        row_id = engine._store.append('pilot', {'role': 'user', 'content': raw})
        # The continuation omitted the correction, but kept a source handle.
        summary = f'Objective: reconcile label. Omitted areas: quantity correction. Recovery references: {row_id}'
        assert '42.750' not in summary
        assert engine._adaptive_retrieval is None
        assert engine._query_views is None
        assert not config.assertions_enabled and not config.query_views_enabled
        call = lambda **payload: json.loads(engine.handle_tool_call('lcm_retrieve', payload))
        start = call(action='start', question='What was the corrected quantity?',
                     identity={'intent_type': 'continuation', 'operation': 'evidence_only'},
                     requirements=[{'slot_id': 'quantity', 'description': 'omitted correction'}])
        assert start['status'] == 'active', start
        result = call(action='search', retrieval_id=start['retrieval_id'], missing_slot='quantity',
                      tool='lcm_expand', tool_args={'store_id': row_id, 'include_exact_ref': True})
        assert result['evidence'][0]['quote'] == raw, result
        ref = result['evidence'][0]['citation']
        finish = call(action='finish', retrieval_id=start['retrieval_id'],
                      selected_refs=[ref], resolved_slots=[{'slot_id': 'quantity', 'evidence_refs': [ref]}])
        assert finish['status'] != 'error', finish
        assert engine._store.get(row_id)['content'] == raw
        assert controller_for(engine)._query_views is None
        # Revocation takes effect on every tool call, not only new episodes.
        config.selective_retrieval_session_ids = ''
        assert call(action='status', retrieval_id=start['retrieval_id'])['status'] == 'disabled'
    finally:
        engine.shutdown()


def test_source_backed_answer_needs_no_episode_and_wrong_session_cannot_resume(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / 'pilot.db'),
                       selective_retrieval_session_ids='pilot,next')
    engine = LCMEngine(config=config)
    try:
        engine.on_session_start('pilot')
        # Current uncompressed source is already present: no omission to recover.
        source = 'User: quantity is 42.750 mL.'
        assert '42.750 mL' in source
        assert controller_for(engine) is not None
        assert len(controller_for(engine)._states) == 0
        # Only a real missing evidence slot should start an episode.
        started = json.loads(engine.handle_tool_call('lcm_retrieve', {
            'action': 'start', 'question': 'Which missing constraint?',
            'identity': {'intent_type': 'continuation', 'operation': 'evidence_only'},
            'requirements': [{'slot_id': 'constraint', 'description': 'unresolved constraint'}],
        }))
        assert started['status'] == 'active'
        engine.on_session_start('next')
        resumed = json.loads(engine.handle_tool_call('lcm_retrieve', {
            'action': 'status', 'retrieval_id': started['retrieval_id'],
        }))
        assert resumed['status'] == 'error', resumed
    finally:
        engine.shutdown()


def test_exact_ref_arguments_are_typed_and_retrieval_budget_is_hard():
    assert MAX_RETRIEVAL_ROUNDS == 3
    for tool, args in [('lcm_expand', {'store_id': 1}),
                       ('lcm_load_session', {'session_id': 'pilot'})]:
        assert _bounded_tool_args(tool, dict(args, include_exact_ref=True))['include_exact_ref'] is True
        with pytest.raises(ValueError, match='boolean'):
            _bounded_tool_args(tool, dict(args, include_exact_ref='false'))
        with pytest.raises(ValueError):
            _bounded_tool_args(tool, dict(args, ground_truth='malicious'))
