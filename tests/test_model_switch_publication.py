import time

import pytest

from hermes_lcm.background_compaction import BackgroundCompactionStore
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.model_switch_publication import ModelSwitchPublication


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
