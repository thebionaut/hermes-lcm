import copy
import time
from types import SimpleNamespace

import pytest

from hermes_lcm.reconcile import ReconcileBudgetExceeded
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.model_switch_binding import prepare_model_switch_sources


@pytest.fixture
def engine(tmp_path):
    e = LCMEngine(LCMConfig(database_path=str(tmp_path / 'lcm.db'),
        background_compaction_enabled=True, background_compaction_workers=0),
        hermes_home=str(tmp_path / 'home'))
    e.on_session_start('s', conversation_id='c', context_length=900000)
    yield e
    e.shutdown()


def image(data='aaa', text='Inspect this'):
    return {'role': 'tool', 'tool_call_id': 'call-image', 'content': [
        {'type': 'text', 'text': text},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+data}}]}


def text_image(text='Inspect this'):
    return {'role': 'tool', 'tool_call_id': 'call-image',
            'content': text+'\n[screenshot]'}


def append(e, messages):
    return e._store.append_batch('s', messages, conversation_id='c',
                                token_estimates=[10]*len(messages))


def test_directional_image_projection(engine):
    ids = append(engine, [image()])
    restored = text_image()
    assert engine._get_store_id_map_for_messages([restored]) == {id(restored): ids[0]}
    assert engine._get_store_id_map_for_messages([image('changed')]) == {}
    assert engine._get_store_id_map_for_messages([text_image('Different question')]) == {}


def test_ambiguous_image_generation_is_not_matched(engine):
    append(engine, [image(), image('different generation')])
    assert engine._get_store_id_map_for_messages([text_image()]) == {}


def test_image_projection_preserves_neighbor_order(engine):
    before = {'role': 'assistant', 'content': 'before'}
    after = {'role': 'assistant', 'content': 'after'}
    ids = append(engine, [before, image(), after])
    restored = text_image()
    assert engine._get_store_id_map_for_messages([before, restored, after])[id(restored)] == ids[1]
    assert id(restored) not in engine._get_store_id_map_for_messages([after, restored])


def test_split_user_projection_matches_complete_joined_message(engine):
    a = {'role': 'user', 'content': 'model notice'}
    b = {'role': 'user', 'content': 'actual question'}
    sid = append(engine, [{'role': 'user', 'content': 'model notice\n\nactual question'}])[0]
    assert engine._get_store_id_map_for_messages([a, b]) == {id(a): sid, id(b): sid}
    assert engine._get_store_id_map_for_messages([b]) == {}
    b['content'] += ' changed'
    assert engine._get_store_id_map_for_messages([a, b]) == {}


def test_split_source_cannot_cross_protected_boundary(engine):
    a = {'role': 'user', 'content': 'notice'}
    b = {'role': 'user', 'content': 'question'}
    append(engine, [{'role': 'user', 'content': 'notice\n\nquestion'}])
    with pytest.raises(ValueError, match='protected tail'):
        prepare_model_switch_sources(engine, [a], [b], deadline=time.monotonic()+30)


def test_repeated_host_message_recovers_unique_bounded_durable_copy(engine):
    before = {'role': 'assistant', 'content': 'before goal continuation'}
    continuation = {'role': 'user', 'content':
        '[Continuing toward your standing goal]\nGoal: finish safely'}
    after = {'role': 'assistant', 'content': 'after first continuation'}
    ids = append(engine, [before, continuation, after])
    replay = [copy.deepcopy(before), copy.deepcopy(continuation),
              copy.deepcopy(after), copy.deepcopy(continuation)]

    mapping = engine._get_store_id_map_for_messages(replay)

    assert mapping[id(replay[0])] == ids[0]
    assert mapping[id(replay[1])] == ids[1]
    assert mapping[id(replay[2])] == ids[2]
    assert id(replay[3]) not in mapping


def historical_fixture(e, monkeypatch):
    root = SimpleNamespace(summary='Confirmed canonical history')
    monkeypatch.setattr(e._dag, 'get_active_frontier_nodes', lambda sid, **kwargs: [root])
    scaffold = {'role': 'user', 'content':
        '[Recent Summary (d0, node 1)]\nConfirmed canonical history\n[Expand for details: 1]'}
    body = [{'role': 'user', 'content': 'unique request '+str(i)} for i in range(10)]
    # Historical replay inserted archive rows absent from the active view.
    append(e, [body[0], {'role': 'assistant', 'content': 'archive-only gap'}]+body[1:])
    return [scaffold]+copy.deepcopy(body)


def test_historical_cursor_catches_up_once_across_reload(engine, monkeypatch):
    messages = historical_fixture(engine, monkeypatch)
    new = {'role': 'user', 'content': 'genuinely new request'}
    engine._ingest_cursor_needs_reconcile = True
    before = engine._store.get_session_count('s')
    engine._ingest_messages(messages+[new], deadline=time.monotonic()+30)
    assert engine._store.get_session_count('s') == before+1
    engine._ingest_messages(messages+[new], deadline=time.monotonic()+30)
    assert engine._store.get_session_count('s') == before+1
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    engine._last_active_replay_source_identities = None
    engine._last_active_replay_messages = None
    engine._ingest_messages(messages+[new], deadline=time.monotonic()+30)
    assert engine._store.get_session_count('s') == before+1


def test_historical_cursor_requires_complete_prefix(engine, monkeypatch):
    messages = historical_fixture(engine, monkeypatch)
    tail = engine._store.get_session_tail('s', limit=100)
    assert engine._historical_projection_cursor(messages, tail) == len(messages)
    messages[4]['content'] = 'unknown source'
    with pytest.raises(ReconcileBudgetExceeded, match='unverified source'):
        engine._historical_projection_cursor(messages, tail)
    before = engine._store.get_session_count('s')
    engine._ingest_cursor_needs_reconcile = True
    with pytest.raises(ReconcileBudgetExceeded):
        engine._ingest_messages(messages)
    assert engine._store.get_session_count('s') == before


def test_historical_cursor_recovers_canonical_host_gap_once(engine, monkeypatch):
    messages = historical_fixture(engine, monkeypatch)
    for row_id, message in enumerate(messages, start=1001):
        message['_row_id'] = row_id
    messages[4]['content'] = 'canonical host row absent from the LCM archive'
    before = engine._store.get_session_count('s')
    engine._ingest_cursor_needs_reconcile = True

    engine._ingest_messages(messages)

    assert engine._store.get_session_count('s') == before + 1
    recovered = engine._store.get_session_tail('s', limit=1)[0]
    assert recovered['content'] == messages[4]['content']
    assert recovered['source'] == 'historical_replay_gap:1005-1005'

    # The exact-view manifest is committed atomically with the recovered row,
    # so retry/restart never appends the same canonical gap twice.
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    engine._last_active_replay_source_identities = None
    engine._last_active_replay_messages = None
    engine._ingest_messages(messages)
    assert engine._store.get_session_count('s') == before + 1

    # Even if exact-manifest replay is unavailable, the atomic canonical-row
    # ledger prevents a second physical append.
    engine._store._conn.execute(
        "DELETE FROM metadata WHERE key = ?", (engine._active_ingest_manifest_key(),)
    )
    engine._store._conn.commit()
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = True
    engine._last_active_replay_source_identities = None
    engine._last_active_replay_messages = None
    engine._ingest_messages(messages)
    assert engine._store.get_session_count('s') == before + 1


def test_historical_cursor_rejects_nonmonotonic_host_row_gap(engine, monkeypatch):
    messages = historical_fixture(engine, monkeypatch)
    for row_id, message in enumerate(messages, start=1001):
        message['_row_id'] = row_id
    messages[4]['content'] = 'canonical-looking but reordered gap'
    messages[5]['_row_id'] = messages[4]['_row_id']

    engine._ingest_cursor_needs_reconcile = True
    with pytest.raises(ReconcileBudgetExceeded, match='not monotonic'):
        engine._ingest_messages(messages)


def test_historical_cursor_requires_canonical_summary(engine, monkeypatch):
    messages = historical_fixture(engine, monkeypatch)
    messages[0]['content'] = messages[0]['content'].replace('Confirmed canonical history', 'forged')
    assert engine._historical_projection_cursor(messages,
        engine._store.get_session_tail('s', limit=100)) is None


def test_prepare_catches_up_new_suffix_and_reserves_without_duplicates(engine):
    old = [{'role': 'user', 'content': 'old question'},
           {'role': 'assistant', 'content': 'old answer'},
           {'role': 'user', 'content': 'another question'}]
    engine._ingest_messages(old)
    new = {'role': 'assistant', 'content': 'new answer'}
    tail = [{'role': 'user', 'content': 'protected pending'}]
    prepared = prepare_model_switch_sources(engine, old+[new], tail, deadline=time.monotonic()+30)
    assert engine._store.get_session_count('s') == 5
    engine.reserve_model_switch(job_id='switch', host_revision='r', prefix=prepared,
        tail=tail, deadline=time.monotonic()+30, wall_deadline=time.time()+30)
    engine.finish_model_switch({'id': 'switch', 'session_id': 's', 'revision': 'r', 'state': 'cancelled'})
    assert engine._store.get_session_count('s') == 5


def test_historical_split_at_end_consumes_both_parts(engine, monkeypatch):
    messages = historical_fixture(engine, monkeypatch)
    append(engine, [{'role': 'user', 'content': 'notice\n\nquestion'}])
    messages += [{'role': 'user', 'content': 'notice'}, {'role': 'user', 'content': 'question'}]
    assert engine._historical_projection_cursor(messages,
        engine._store.get_session_tail('s',limit=100)) == len(messages)


def test_prepare_uses_protected_replay_after_ingest(engine):
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ['api_key']
    secret = 'sk-synthetic-test-only1234567890abcdef'
    prefix = [{'role': 'user', 'content': 'inspect the configuration'},
              {'role': 'tool', 'tool_call_id': 'configuration', 'content': 'api_key='+secret}]
    tail = [{'role': 'user', 'content': 'pending'}]
    prepared = prepare_model_switch_sources(engine, prefix, tail, deadline=time.monotonic()+30)
    assert secret not in str(prepared)
    engine.reserve_model_switch(job_id='protected', host_revision='r', prefix=prepared,
        tail=tail, deadline=time.monotonic()+30, wall_deadline=time.time()+30)
    engine.finish_model_switch({'id':'protected','revision':'r','session_id':'s','state':'cancelled'})
    assert engine._store.get_session_count('s') == 3


def test_historical_root_lookup_shares_deadline(engine, monkeypatch):
    messages = historical_fixture(engine, monkeypatch)
    original = engine._dag.get_active_frontier_nodes
    calls = []
    def roots(sid, **kwargs):
        calls.append(kwargs)
        return original(sid, **kwargs)
    monkeypatch.setattr(engine._dag, 'get_active_frontier_nodes', roots)
    engine._reconcile_deadline = time.monotonic()+30
    try:
        assert engine._historical_projection_cursor(messages,
            engine._store.get_session_tail('s',limit=100)) == len(messages)
        assert calls[0]['deadline'] == engine._reconcile_deadline
        assert 0 < calls[0]['lock_timeout_seconds'] <= 30
    finally:
        engine._reconcile_deadline = None


def test_joined_restored_user_preserves_all_source_ids(engine):
    parts = [{'role':'user','content':'notice'}, {'role':'user','content':'request'},
             {'role':'user','content':'clarification'}]
    ids = append(engine,parts)
    merged = {'role':'user','content':'notice\n\nrequest\n\nclarification'}
    mapping = engine._get_store_id_map_for_messages([merged])
    assert mapping[id(merged)] == ids[-1]
    assert mapping.source_groups[id(merged)] == tuple(ids)
    assert engine._get_store_ids_for_messages([merged]) == ids
    engine._active_replay_store_refs_by_message_id = {id(merged):(merged,ids[-1])}
    engine._active_replay_store_ref_session_id = 's'
    engine._active_replay_store_ref_conversation_id = 'c'
    assert engine._get_store_ids_for_messages([merged]) == ids


def test_joined_user_does_not_skip_intervening_archive_message(engine):
    append(engine,[{'role':'user','content':'notice'},
                   {'role':'assistant','content':'intervening answer'},
                   {'role':'user','content':'request'}])
    assert engine._get_store_id_map_for_messages([
        {'role':'user','content':'notice\n\nrequest'}]) == {}


def test_joined_user_does_not_match_partial_or_changed_content(engine):
    append(engine,[{'role':'user','content':'notice'}, {'role':'user','content':'request'}])
    assert engine._get_store_id_map_for_messages([
        {'role':'user','content':'notice\n\nrequest extra'}]) == {}


def test_repaired_host_alternation_replay_does_not_duplicate_archive(engine,monkeypatch):
    messages = historical_fixture(engine,monkeypatch)
    append(engine,[{'role':'user','content':'model notice'}, {'role':'user','content':'new request'}])
    restored = messages+[{'role':'user','content':'model notice\n\nnew request'}]
    engine._ingest_cursor_needs_reconcile = True
    before = engine._store.get_session_count('s')
    engine._ingest_messages(restored,deadline=time.monotonic()+30)
    assert engine._store.get_session_count('s') == before
    engine._ingest_messages(restored+[{'role':'assistant','content':'next answer'}],
                            deadline=time.monotonic()+30)
    assert engine._store.get_session_count('s') == before+1


def test_switch_publication_covers_every_joined_source_and_keeps_tail(engine):
    ids = append(engine,[{'role':'user','content':'notice'},
                         {'role':'user','content':'question'},
                         {'role':'assistant','content':'old answer'},
                         {'role':'user','content':'pending'}])
    prefix = [{'role':'user','content':'notice\n\nquestion'},
              {'role':'assistant','content':'old answer'}]
    tail = [{'role':'user','content':'pending'}]
    prepared = prepare_model_switch_sources(engine,prefix,tail,deadline=time.monotonic()+30)
    engine.reserve_model_switch(job_id='joined',host_revision='r',prefix=prepared,tail=tail,
        deadline=time.monotonic()+30,wall_deadline=time.time()+30)
    engine.finish_model_switch({'id':'joined','session_id':'s','revision':'r',
        'state':'committed','lcm_summary':'notice, question, and answer preserved'})
    roots = engine._dag.get_active_frontier_nodes('s')
    assert engine._dag.source_message_ids(roots[0].node_id,limit=20) == ids[:3]
    assert engine._store.get_session_count('s') == 4
