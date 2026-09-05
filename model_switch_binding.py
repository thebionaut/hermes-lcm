"""Bind host preparation partitions to exact LCM source lineage."""

from .model_switch_publication import ModelSwitchPublication
from .tokens import count_tokens


def reserve_model_switch(engine, *, job_id, host_revision, prefix, tail,
                         deadline, wall_deadline):
    if engine._background_compaction is None or not engine._conversation_id or not engine._session_id:
        raise ValueError('LCM model-switch publication is not available for this session')
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    if state is None or state.current_session_id != engine._session_id:
        raise ValueError('LCM session lineage changed')
    frontier = int(state.current_frontier_store_id or 0)
    # Map the original partition objects together. Reconciliation can recognize
    # repeated text more reliably with the surrounding protected tail present.
    mapping = engine._get_store_id_map_for_messages(prefix + tail, deadline=deadline)
    prefix_ids = {int(mapping[id(m)]) for m in prefix if id(m) in mapping}
    tail_ids = {int(mapping[id(m)]) for m in tail if id(m) in mapping}
    if prefix_ids & tail_ids:
        raise ValueError('LCM prefix and protected tail overlap')
    roots = engine._dag.get_active_frontier_nodes(engine._session_id)
    represented_summaries = set()
    prefix_text = '\n'.join(str(m.get('api_content') or m.get('content') or '') for m in prefix)
    tail_text = '\n'.join(str(m.get('api_content') or m.get('content') or '') for m in tail)
    for root in roots:
        engine._foreground_deadline_checkpoint(deadline, 'model switch', 'root lineage validation')
        source_ids = set(engine._dag.source_message_ids(root.node_id, limit=len(prefix_ids) + 1))
        textual = bool(root.summary and root.summary in prefix_text)
        if not textual and not (source_ids and source_ids <= prefix_ids):
            raise ValueError('LCM active summary is not represented by the preparation prefix')
        if root.summary and root.summary in tail_text:
            raise ValueError('LCM active summary is part of the protected tail')
        if textual:
            represented_summaries.add(root.summary)
    for message in prefix:
        if id(message) in mapping:
            continue
        content = str(message.get('api_content') or message.get('content') or '')
        if not any(summary in content for summary in represented_summaries):
            raise ValueError('Preparation prefix contains a message without LCM source identity')
    uncanonical = sorted(sid for sid in prefix_ids if sid > frontier)
    watermark = max(uncanonical, default=frontier)
    if any(sid <= watermark for sid in tail_ids):
        raise ValueError('LCM publication would consume protected tail source')
    return ModelSwitchPublication(engine._background_compaction).reserve(
        job_id=job_id, conversation_id=engine._conversation_id, session_id=engine._session_id,
        host_revision=host_revision, expected_frontier=frontier, watermark=watermark,
        source_ids=uncanonical, deadline=wall_deadline,
    )


def finish_model_switch(engine, receipt):
    publication = ModelSwitchPublication(engine._background_compaction)
    with engine._background_compaction._write_transaction() as conn:
        reserved = conn.execute('SELECT 1 FROM lcm_host_model_switches WHERE job_id=?',
                                (receipt['id'],)).fetchone()
    # Intent is persisted in the host before reservation. A failed reserve can
    # therefore legitimately leave a terminal host intent with no LCM row.
    if reserved is None and receipt['state'] in ('cancelled', 'failed', 'expired'):
        return
    summary = receipt.get('lcm_summary') or ''
    publication.finish(job_id=receipt['id'], host_receipt=receipt,
                       summary=summary, summary_tokens=count_tokens(summary))
    if receipt['state'] == 'committed':
        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        engine._last_compacted_store_id = state.current_frontier_store_id
        engine._active_replay_store_refs_by_message_id = {}
