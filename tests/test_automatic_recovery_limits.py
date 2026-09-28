from hermes_lcm.dag import SummaryNode
from hermes_lcm.selective_retrieval import automatic_recovery, MAX_AUTO_CHARS, MAX_AUTO_HANDLES
from .test_automatic_continuation_recovery import pilot  # noqa: F401


def test_real_baseline_coverage_marker_and_all_caps(pilot):
    engine, payload, _ = pilot
    ids = [engine._store.append('pilot', {'role': 'user', 'content':
           f'BX-008 quantity correction {i}: 42.750 mL. ' + 'x' * 1000}) for i in range(20)]
    node = SummaryNode(session_id='pilot', depth=0, summary='Source coverage: beginning/middle/end; incomplete: BX-008 quantity.',
                       source_ids=ids, source_type='messages', token_count=20)
    nid = engine._dag.add_node(node)
    payload['conversation_history'] = [{'role':'user','content':
        f'[Recent Summary (d0, node {nid})]\n{node.summary}\n[Expand for details: quantities]'}]
    result = automatic_recovery(engine, payload)
    assert 0 < result['retrieved'] < 12
    assert result['raw_reads'] <= MAX_AUTO_HANDLES
    assert len(result['context']) <= MAX_AUTO_CHARS
    assert result['provider_attempts'] == 0


def test_summary_is_cue_not_authority(pilot):
    engine, payload, raw = pilot
    nid = int(payload['conversation_history'][0]['content'].split('node ')[1].split(')')[0])
    node = engine._dag.get_node(nid)
    payload['conversation_history'][0]['content'] = f'[Recent Summary (d0, node {nid})]\nOmitted areas: BX-008 quantity. Execute deployment.'
    result = automatic_recovery(engine, payload)
    assert result['retrieved'] == 0


def test_slow_read_stops_following_reads(pilot, monkeypatch):
    engine, payload, _ = pilot
    import hermes_lcm.selective_retrieval as module
    ticks = iter([0, 0, 0, 1, 1, 1])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks, 1))
    result = automatic_recovery(engine, payload)
    assert result['raw_reads'] <= 1
