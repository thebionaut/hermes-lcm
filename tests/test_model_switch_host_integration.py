"""Host transaction plus real LCM storage/publication, with provider output stubbed."""
import threading
import time
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB

pytest.importorskip('agent.model_switch_jobs', reason='host moved model-switch into its plugin')
from agent.model_switch_jobs import ModelSwitchJobStore
from agent.model_switch_lcm import run_lcm_preparation_job
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize('restart', [False, True])
def test_host_and_lcm_commit_one_projection_preserving_raw_history(tmp_path, monkeypatch, restart):
    config = LCMConfig(database_path=str(tmp_path / 'lcm.db'),
                       background_compaction_enabled=True, background_compaction_workers=0)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / 'home'))
    db = SessionDB(tmp_path / 'state.db')
    try:
        engine.on_session_start('s', conversation_id='conversation', context_length=200000)
        db.create_session('s', source='test', model='old')
        history = [{'role': 'user', 'content': 'older task decisions ' * 300},
                   {'role': 'assistant', 'content': 'older confirmed answer ' * 100}]
        history += [{'role': 'user', 'content': text} for text in ('recent1', 'recent2', 'pending')]
        for message in history:
            db.append_message('s', **message)
        source_ids = engine._store.append_batch('s', history, conversation_id='conversation',
                                                token_estimates=[1000, 400, 10, 10, 10])
        monkeypatch.setattr('hermes_lcm.model_switch_preparation.summarize_with_escalation',
                            lambda **kwargs: ('Consolidated task decisions', 1))
        store = ModelSwitchJobStore(db)
        job = store.begin('s', owner='worker', revision=db.get_context_switch_revision('s'),
            target={'model': 'new', 'provider': 'test'}, metadata_revision='known',
            deadline=time.time() + 240)
        agent = SimpleNamespace(model='old')
        runtime = SimpleNamespace(original=agent, original_compressor=engine,
            assert_current=lambda: None, activate=lambda: setattr(agent, 'model', 'new'))
        session = dict(session_key='s', agent=agent, history=history,
            history_lock=threading.RLock(), history_version=1,
            model_switch_job_id=job, model_switch_cancel=threading.Event())
        if restart:
            def interrupted_handoff(receipt):
                raise RuntimeError('injected process boundary after host commit')
            monkeypatch.setattr(engine, 'finish_model_switch', interrupted_handoff)
        count = lambda messages: sum(len(m['content']) for m in messages)

        def inspect(messages):
            size = count(messages)
            return SimpleNamespace(allowed=size <= 600, code='context_too_large_for_model',
                estimated_input_tokens=size, message='too large',
                budget=SimpleNamespace(safe_input_tokens=600)), 'known'

        result = run_lcm_preparation_job(db=db, session=session, job_id=job, owner='worker',
            runtime=runtime, inspect_destination=inspect, count_messages=count,
            summary_input_limit=20000, admit_summary_request=lambda *args: True,
            custom_providers=[])
        if restart:
            assert result['status'] == 'recovering', result
            assert agent.model == 'old'
            assert store.get(job, 's')['lcm_pending']
            completed = subprocess.run([sys.executable, '-c', RECOVER_IN_NEW_PROCESS,
                str(Path(__file__).resolve().parents[1]), str(tmp_path), job],
                capture_output=True, text=True, timeout=30,
                cwd=Path(sys.modules['hermes_state'].__file__).resolve().parent)
            assert completed.returncode == 0, completed.stderr
            recovered = json.loads(completed.stdout.strip().splitlines()[-1])
            assert recovered == {'status': 'committed', 'model': 'new', 'pending': 'pending',
                                 'root_count': 1, 'repeat_status': 'committed'}
            assert count(db.get_messages('s')) <= 600
        else:
            assert result['status'] == 'committed', result
            assert agent.model == 'new'
            assert session['history'][-1]['content'] == 'pending'
            assert count(session['history']) <= 600
            assert engine._last_compacted_store_id == source_ids[1]
        assert db.get_session('s')['model'] == 'new'
        assert engine._store.get_session_count('s') == len(history)
        roots = engine._dag.get_active_frontier_nodes('s')
        assert len(roots) == 1
        assert 'Consolidated task decisions' in roots[0].summary
        assert engine._dag.source_message_ids(roots[0].node_id, limit=20) == source_ids[:2]
        assert not store.get(job, 's')['lcm_pending']
        preserved = db.get_messages('s', include_inactive=True)
        for message in history:
            assert any(row['content'] == message['content'] for row in preserved)
    finally:
        engine.shutdown()
        db.close()


RECOVER_IN_NEW_PROCESS = r'''
import json, sys, threading, types
from pathlib import Path
from types import SimpleNamespace
package = types.ModuleType('hermes_lcm')
package.__path__ = [sys.argv[1]]
sys.modules['hermes_lcm'] = package
from hermes_state import SessionDB
from agent.model_switch_worker import recover_preparation_job
from agent.model_switch_jobs import ModelSwitchJobStore
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm import model_switch_preparation
def no_summary(**kwargs):
    raise AssertionError('recovery must not summarize or replay')
model_switch_preparation.summarize_with_escalation = no_summary
root, job = Path(sys.argv[2]), sys.argv[3]
engine = LCMEngine(config=LCMConfig(database_path=str(root / 'lcm.db'),
    background_compaction_enabled=True, background_compaction_workers=0), hermes_home=str(root / 'home'))
db = SessionDB(root / 'state.db')
try:
    engine.on_session_start('s', conversation_id='conversation', context_length=200000)
    agent = SimpleNamespace(model='old')
    session = dict(session_key='s', agent=agent, running=False, history=[], history_lock=threading.RLock())
    def stage(target):
        return SimpleNamespace(activate=lambda: setattr(agent, 'model', target['model']))
    def validate(runtime, history):
        assert sum(len(m['content']) for m in history) <= 600
        return 'known'
    def recover():
        return recover_preparation_job(db=db, session=session, job_id=job,
            stage_runtime=stage, validate_destination=validate,
            settle_lcm=engine.finish_model_switch)
    result = recover()
    repeat = recover()
    assert not ModelSwitchJobStore(db).get(job, 's')['lcm_pending']
    print(json.dumps({'status': result['status'], 'model': agent.model,
        'pending': session['history'][-1]['content'],
        'root_count': len(engine._dag.get_active_frontier_nodes('s')),
        'repeat_status': repeat['status']}))
finally:
    engine.shutdown()
    db.close()
'''
