import time

import pytest

from hermes_lcm.background_compaction import BackgroundCompactionStore
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.model_switch_publication import ModelSwitchPublication
from hermes_lcm.model_switch_binding import prepare_model_switch_sources


def test_source_projection_covers_duplicate_gaps_without_copying_raw(setup):
    engine, publication, ids = setup
    engine._store.append_batch('session', [
        {'role': 'user', 'content': 'old source'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': 'new older request'},
        {'role': 'user', 'content': 'current pending'},
    ], conversation_id='conversation', token_estimates=[10]*4)
    prefix = [{'role': 'user', 'content': 'old source'},
              {'role': 'assistant', 'content': 'old answer'},
              {'role': 'user', 'content': 'new older request'}]
    tail = [{'role': 'user', 'content': 'current pending'}]
    before = engine._store.get_session_count('session')
    deadline = time.monotonic() + 30
    prepared = prepare_model_switch_sources(engine, prefix, tail, deadline=deadline)
    assert any('_lcm_switch_source' in m for m in prepared)
    engine.reserve_model_switch(job_id='job', host_revision='revision', prefix=prepared, tail=tail,
                                deadline=deadline, wall_deadline=time.time()+30)
    engine.finish_model_switch({**receipt('committed'), 'lcm_summary': 'All earlier context'})
    assert engine._store.get_session_count('session') == before
    roots = engine._dag.get_active_frontier_nodes('session')
    assert len(engine._dag.source_message_ids(roots[0].node_id, limit=20)) == before-1


def test_source_projection_adds_missing_canonical_root(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    engine.finish_model_switch({**receipt('committed'), 'lcm_summary': 'Canonical decision'})
    engine._store.append_batch('session', [{'role':'user', 'content':'new pending'}],
                              conversation_id='conversation', token_estimates=[10])
    prefix = [{'role':'user', 'content':'protected pending prompt'}]
    tail = [{'role':'user', 'content':'new pending'}]
    deadline = time.monotonic()+30
    prepared = prepare_model_switch_sources(engine, prefix, tail, deadline=deadline)
    assert 'Canonical decision' in prepared[0]['content']
    engine.reserve_model_switch(job_id='next', host_revision='revision2', prefix=prepared, tail=tail,
                                deadline=deadline, wall_deadline=time.time()+30)


def test_switch_reservation_spans_lifecycle_rollover(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / 'lcm.db'),
                       background_compaction_enabled=True, background_compaction_workers=0)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / 'home'))
    try:
        engine.on_session_start('session', conversation_id='conversation-1', context_length=200000)
        engine._store.append_batch('session', [
            {'role': 'user', 'content': 'older request'},
            {'role': 'assistant', 'content': 'older answer'},
        ], conversation_id='conversation-1', token_estimates=[10, 10])
        engine.on_session_start('session', conversation_id='conversation-2', context_length=200000)
        engine._store.append_batch('session', [
            {'role': 'user', 'content': 'new request'},
            {'role': 'assistant', 'content': 'new answer'},
            {'role': 'user', 'content': 'protected pending'},
        ], conversation_id='conversation-2', token_estimates=[10, 10, 10])
        prefix = [
            {'role': 'user', 'content': 'older request'},
            {'role': 'assistant', 'content': 'older answer'},
            {'role': 'user', 'content': 'new request'},
            {'role': 'assistant', 'content': 'new answer'},
        ]
        tail = [{'role': 'user', 'content': 'protected pending'}]
        deadline = time.monotonic() + 30

        prepared = prepare_model_switch_sources(engine, prefix, tail, deadline=deadline)
        engine.reserve_model_switch(
            job_id='rollover-job', host_revision='revision', prefix=prepared, tail=tail,
            deadline=deadline, wall_deadline=time.time() + 30,
        )
        engine.finish_model_switch({
            'id': 'rollover-job', 'session_id': 'session', 'revision': 'revision',
            'state': 'committed', 'lcm_summary': 'History across both conversations',
        })

        roots = engine._dag.get_active_frontier_nodes('session')
        assert len(roots) == 1
        assert engine._dag.source_message_ids(roots[0].node_id, limit=20)[:4]
    finally:
        engine.shutdown()


def test_quoted_source_cannot_be_forged(setup):
    engine, publication, ids = setup
    prefix = [{'role':'user', 'content':'old source'},
              {'role':'assistant', 'content':'old answer'},
              {'role':'user', 'content':'fabricated', '_lcm_switch_source':ids[0]}]
    with pytest.raises(ValueError, match='Quoted model switch source'):
        engine.reserve_model_switch(job_id='job', host_revision='revision', prefix=prefix, tail=[],
                                    deadline=time.monotonic()+30, wall_deadline=time.time()+30)


@pytest.fixture
def setup(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / 'lcm.db'),
                       background_compaction_enabled=True, background_compaction_workers=0)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / 'home'))
    engine.on_session_start('session', conversation_id='conversation', context_length=200000)
    ids = engine._store.append_batch('session', [
        {'role':'user', 'content':'old source'},
        {'role':'assistant', 'content':'old answer'},
        {'role':'user', 'content':'protected pending prompt'},
    ], conversation_id='conversation', token_estimates=[10, 10, 10])
    publication = ModelSwitchPublication(engine._background_compaction)
    yield engine, publication, ids
    engine.shutdown()


def reserve(publication, ids):
    return publication.reserve(job_id='job', conversation_id='conversation',
                               session_id='session', host_revision='revision',
                               expected_frontier=0, watermark=ids[1], source_ids=ids[:2],
                               deadline=time.time() + 240)


def receipt(state):
    return {'id':'job', 'session_id':'session', 'revision':'revision', 'state':state}


def test_large_source_reservation_uses_bounded_database_work(setup):
    from hermes_lcm.model_switch_publication import _source_rows
    engine, publication, ids = setup
    added = engine._store.append_batch('session',
        [{'role': 'user', 'content': f'source {i}'} for i in range(2000)],
        conversation_id='conversation', token_estimates=[10]*2000)
    with engine._background_compaction._write_transaction() as conn:
        callbacks = 0
        def budget():
            nonlocal callbacks
            callbacks += 1
            return int(callbacks > 200)
        conn.set_progress_handler(budget, 1000)
        try:
            rows = _source_rows(engine._background_compaction, conn,
                                session_id='session', source_ids=added)
        finally:
            conn.set_progress_handler(None, 0)
    assert [row['store_id'] for row in rows] == added


def test_engine_partition_binding_preserves_tail(setup):
    engine, publication, ids = setup
    prefix = [{'role': 'user', 'content': 'old source'},
              {'role': 'assistant', 'content': 'old answer'}]
    tail = [{'role': 'user', 'content': 'protected pending prompt'}]
    engine.reserve_model_switch(job_id='job', host_revision='revision', prefix=prefix, tail=tail,
                                deadline=time.monotonic() + 30, wall_deadline=time.time() + 30)
    engine.finish_model_switch({**receipt('committed'), 'lcm_summary': 'Old task decisions'})
    assert engine._last_compacted_store_id == ids[1]
    assert engine._store.get_session_count('session') == 3
    assert engine._dag.get_active_frontier_nodes('session')[0].summary == 'Old task decisions'


def test_engine_partition_rejects_unmapped_source(setup):
    engine, publication, ids = setup
    with pytest.raises(ValueError, match='without LCM source identity'):
        engine.reserve_model_switch(job_id='job', host_revision='revision',
                                    prefix=[{'role': 'user', 'content': 'not stored'}], tail=[],
                                    deadline=time.monotonic() + 30, wall_deadline=time.time() + 30)
    assert engine._background_compaction.get_publishable_session_fence_generation(
        conversation_id='conversation', session_id='session') is not None


def test_repeated_prefix_replay_uses_exact_reserved_identity(setup):
    engine, publication, ids = setup
    prefix = [{'role': 'user', 'content': 'old source'},
              {'role': 'user', 'content': 'old source'},
              {'role': 'assistant', 'content': 'old answer'}]
    tail = [{'role': 'user', 'content': 'protected pending prompt'}]
    engine.reserve_model_switch(job_id='job', host_revision='revision',
                                prefix=prefix, tail=tail,
                                deadline=time.monotonic()+30, wall_deadline=time.time()+30)
    engine.finish_model_switch({**receipt('committed'), 'lcm_summary': 'Old decisions'})
    assert engine._last_compacted_store_id == ids[1]
    assert engine._store.get_session_count('session') == 3


def test_repeated_prefix_cannot_borrow_protected_tail_identity(setup):
    engine, publication, ids = setup
    prefix = [{'role': 'user', 'content': 'old source'},
              {'role': 'assistant', 'content': 'old answer'},
              {'role': 'user', 'content': 'protected pending prompt'}]
    tail = [{'role': 'user', 'content': 'protected pending prompt'}]
    with pytest.raises(ValueError):
        engine.reserve_model_switch(job_id='job', host_revision='revision',
                                    prefix=prefix, tail=tail,
                                    deadline=time.monotonic()+30, wall_deadline=time.time()+30)


def test_engine_partition_rejects_hole_in_prefix(setup):
    engine, publication, ids = setup
    with pytest.raises(ValueError, match='protected tail|source range changed'):
        engine.reserve_model_switch(job_id='job', host_revision='revision',
                                    prefix=[{'role': 'assistant', 'content': 'old answer'}],
                                    tail=[{'role': 'user', 'content': 'old source'}],
                                    deadline=time.monotonic() + 30, wall_deadline=time.time() + 30)


def test_failed_intent_before_reservation_can_be_released(setup):
    engine, publication, ids = setup
    engine.finish_model_switch(receipt('failed'))
    assert engine._store.get_session_count('session') == 3
    with pytest.raises(ValueError, match='does not match reservation'):
        engine.finish_model_switch({**receipt('committed'), 'lcm_summary': 'unconfirmed'})


def test_partition_must_include_previous_root_and_replaces_it_once(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    engine.finish_model_switch({**receipt('committed'), 'lcm_summary': 'Earlier confirmed decisions'})
    added = engine._store.append_batch('session', [
        {'role': 'assistant', 'content': 'follow-up answer'},
        {'role': 'user', 'content': 'new pending prompt'},
    ], conversation_id='conversation', token_estimates=[10, 10])
    prefix = [{'role': 'user', 'content': 'protected pending prompt'},
              {'role': 'assistant', 'content': 'follow-up answer'}]
    tail = [{'role': 'user', 'content': 'new pending prompt'}]
    with pytest.raises(ValueError, match='active summary is not represented'):
        engine.reserve_model_switch(job_id='next', host_revision='next-revision', prefix=prefix, tail=tail,
                                    deadline=time.monotonic() + 30, wall_deadline=time.time() + 30)
    prefix.insert(0, {'role': 'user', 'content': '[Summary] Earlier confirmed decisions'})
    engine.reserve_model_switch(job_id='next', host_revision='next-revision', prefix=prefix, tail=tail,
                                deadline=time.monotonic() + 30, wall_deadline=time.time() + 30)
    engine.finish_model_switch({'id': 'next', 'revision': 'next-revision', 'session_id': 'session',
                                'state': 'committed', 'lcm_summary': 'One consolidated summary'})
    roots = engine._dag.get_active_frontier_nodes('session')
    assert len(roots) == 1 and roots[0].summary == 'One consolidated summary'
    assert engine._last_compacted_store_id == added[0]
    assert engine._store.get_session_count('session') == 5
    assert engine._dag.source_message_ids(roots[0].node_id, limit=20) == ids + [added[0]]


def test_adoption_preserves_raw_source_and_pending_tail(setup):
    engine, publication, ids = setup
    reservation = reserve(publication, ids)
    store = engine._background_compaction
    assert store.get_publishable_session_fence_generation(
        conversation_id='conversation', session_id='session') is None
    node_id = publication.finish(job_id='job', host_receipt=receipt('committed'),
                                 summary='Consolidated old decisions', summary_tokens=5)
    assert node_id
    roots = engine._dag.get_active_frontier_nodes('session')
    assert len(roots) == 1
    assert roots[0].summary == 'Consolidated old decisions'
    assert engine._store.get_session_count('session') == 3
    assert store.get_publishable_session_fence_generation(
        conversation_id='conversation', session_id='session') == reservation['generation']
    with store._write_transaction() as conn:
        row = conn.execute('SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id=?',
                           ('conversation',)).fetchone()
    assert row[0] == ids[1] < ids[2]
    assert publication.finish(job_id='job', host_receipt=receipt('committed'),
                              summary='Consolidated old decisions', summary_tokens=5) == node_id
    assert len(engine._dag.get_session_nodes('session')) == 2


@pytest.mark.parametrize('state', ['cancelled', 'expired', 'failed'])
def test_failed_host_job_releases_without_context_mutation(setup, state):
    engine, publication, ids = setup
    reserve(publication, ids)
    publication.finish(job_id='job', host_receipt=receipt(state))
    assert engine._dag.get_session_nodes('session') == []
    assert engine._store.get_session_count('session') == 3
    assert engine._background_compaction.get_publishable_session_fence_generation(
        conversation_id='conversation', session_id='session') is not None


def test_unknown_host_outcome_cannot_release_fence(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    with pytest.raises(ValueError, match='not terminal'):
        publication.finish(job_id='job', host_receipt=receipt('prepared'))
    assert engine._background_compaction.get_publishable_session_fence_generation(
        conversation_id='conversation', session_id='session') is None


def test_restart_recovers_from_committed_host_receipt(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    reopened = BackgroundCompactionStore(engine._config.database_path, config=engine._config)
    try:
        recovered = ModelSwitchPublication(reopened)
        assert recovered.finish(job_id='job', host_receipt=receipt('committed'),
                                summary='Recovered committed summary', summary_tokens=5)
        assert engine._dag.get_active_frontier_nodes('session')[0].summary == 'Recovered committed summary'
    finally:
        reopened.close()


def test_changed_source_prevents_adoption(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    with engine._background_compaction._write_transaction() as conn:
        conn.execute("UPDATE messages SET content='concurrent edit' WHERE store_id=?", (ids[0],))
    with pytest.raises(ValueError, match='lineage changed'):
        publication.finish(job_id='job', host_receipt=receipt('committed'), summary='summary', summary_tokens=5)
    assert engine._dag.get_session_nodes('session') == []


def test_adoption_transaction_rolls_back_on_failure(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    with engine._background_compaction._write_transaction() as conn:
        conn.execute("""CREATE TRIGGER fail_switch BEFORE UPDATE ON lcm_host_model_switches
                     WHEN NEW.state='adopted' BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
    with pytest.raises(Exception, match='injected failure'):
        publication.finish(job_id='job', host_receipt=receipt('committed'), summary='summary', summary_tokens=5)
    assert engine._dag.get_session_nodes('session') == []
    assert engine._background_compaction.get_publishable_session_fence_generation(
        conversation_id='conversation', session_id='session') is None


def test_repeated_switch_consolidates_prior_root_without_deleting_lineage(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    first = publication.finish(job_id='job', host_receipt=receipt('committed'),
                               summary='First summary', summary_tokens=5)
    extra = engine._store.append_batch('session', [
        {'role':'assistant', 'content':'new answer'},
        {'role':'user', 'content':'new pending prompt'},
    ], conversation_id='conversation', token_estimates=[10, 10])
    publication.reserve(job_id='next', conversation_id='conversation', session_id='session',
                        host_revision='next-revision', expected_frontier=ids[1], watermark=extra[0],
                        source_ids=[ids[2], extra[0]], deadline=time.time() + 240)
    second = publication.finish(job_id='next',
                                host_receipt={'id':'next', 'session_id':'session',
                                              'revision':'next-revision', 'state':'committed'},
                                summary='One consolidated summary', summary_tokens=5)
    roots = engine._dag.get_active_frontier_nodes('session')
    assert len(roots) == 1 and roots[0].node_id == second
    assert first in roots[0].source_ids
    assert engine._store.get_session_count('session') == 5
    assert len(engine._dag.get_session_nodes('session')) == 4


def test_reservation_supersedes_prior_background_job(setup):
    engine, publication, ids = setup
    store = engine._background_compaction
    job = store.enqueue_or_coalesce(
        conversation_id='conversation', session_id='session', expected_frontier_store_id=0,
        captured_watermark_store_id=ids[1], source_store_ids=ids[:2], source_token_count=20,
        policy_fingerprint=engine._background_policy_fingerprint(), expected_node_count=1)
    assert job is not None
    reserve(publication, ids)
    assert store.get_job(job.job_id).state == 'superseded'


def test_new_binding_cannot_reopen_unresolved_host_switch(setup):
    engine, publication, ids = setup
    reserve(publication, ids)
    store = engine._background_compaction
    with store._write_transaction() as conn:
        conn.execute('UPDATE lcm_lifecycle_state SET current_bound_at=current_bound_at+1 WHERE conversation_id=?',
                     ('conversation',))
        lifecycle = conn.execute('SELECT * FROM lcm_lifecycle_state WHERE conversation_id=?',
                                 ('conversation',)).fetchone()
    assert store.activate_session(conversation_id='conversation', session_id='session',
                                  expected_lifecycle_bound_at=lifecycle['current_bound_at'],
                                  expected_lifecycle_reset_at=lifecycle['last_reset_at']) is None
    assert store.get_publishable_session_fence_generation(
        conversation_id='conversation', session_id='session') is None
