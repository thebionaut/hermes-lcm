"""Foreground leaf summarizes the durable interval when the view is not 1:1."""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

TS = 1_790_000_000.0


def _config(db_path):
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=2,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=120,
        background_compaction_lease_seconds=300,
        background_compaction_backlog_tokens=1,
        background_compaction_min_provider_source_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=1,
        context_reserve_tokens=100_000,
        sensitive_patterns_enabled=False,
    )


def _ids(engine, session="s"):
    return [
        int(row[0])
        for row in engine._store.connection.execute(
            "SELECT store_id FROM messages WHERE session_id=? ORDER BY store_id",
            (session,),
        ).fetchall()
    ]


def _host():
    return [
        {"role": "user", "content": "first question", "timestamp": TS},
        {"role": "assistant", "content": "first answer", "timestamp": TS + 1},
        {"role": "user", "content": "[System: model changed]"},
        {"role": "user", "content": "second question", "timestamp": TS + 2},
        {"role": "assistant", "content": "second answer", "timestamp": TS + 3},
        {"role": "user", "content": "fresh tail", "timestamp": TS + 4},
    ]


def _seed_damaged_store(engine, host):
    store = engine._store
    store.append_batch("s", host[:3], conversation_id="c")
    # Older builds: prefix copies and a same-timestamp content variant stored
    # before the next real turn, so they sit inside the caller's interval.
    store.append_batch("s", host[:2], conversation_id="c")
    store.append_batch(
        "s",
        [{"role": "user", "content": "first question (expanded @file)", "timestamp": TS}],
        conversation_id="c",
    )
    store.append_batch("s", host[3:5], conversation_id="c")


def test_interval_with_duplicates_variants_and_untimed_rows_summarizes(
    tmp_path, monkeypatch
):
    engine = LCMEngine(_config(tmp_path / "interval.db"))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    host = _host()
    _seed_damaged_store(engine, host)
    provider_inputs: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: provider_inputs.append(kwargs["text"])
        or ("interval summary", 1),
    )
    try:
        stored_before = _ids(engine)
        result = engine.compress(host, current_tokens=340_000)
        assert len(provider_inputs) == 1
        text = provider_inputs[0]
        # Each distinct stored message is summarized exactly once, including
        # the variant the host no longer shows.
        assert text.count("first answer") == 1
        assert text.count("second answer") == 1
        assert "first question (expanded @file)" in text
        nodes = engine._dag.get_session_nodes("s")
        assert len(nodes) == 1
        # Every stored row up to the watermark (duplicates included) is
        # covered by the node lineage and the frontier.
        stored_after = _ids(engine)
        frontier = engine._last_compacted_store_id
        assert set(stored_before) <= set(nodes[0].source_ids)
        assert nodes[0].source_ids == [i for i in stored_after if i <= frontier]
        # The fresh tail row stays raw, above the frontier.
        tail_id = int(
            engine._store.connection.execute(
                "SELECT store_id FROM messages WHERE content='fresh tail'"
            ).fetchone()[0]
        )
        assert tail_id > frontier
        contents = [str(message.get("content")) for message in result]
        assert "fresh tail" in contents
        assert not any(c in contents for c in ("first answer", "second answer"))
    finally:
        engine.shutdown()


def test_interval_refuses_when_caller_row_lies_outside_interval(tmp_path, monkeypatch):
    engine = LCMEngine(_config(tmp_path / "outside.db"))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    host = _host()
    _seed_damaged_store(engine, host)
    # A caller message whose only stored row belongs to another conversation
    # of the session is outside the (conversation-scoped) interval.
    foreign = {"role": "assistant", "content": "foreign row", "timestamp": TS + 3.5}
    engine._store.append_batch("s", [foreign], conversation_id="other")
    calls: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: calls.append(kwargs["text"]) or ("x", 1),
    )
    monkeypatch.setattr(
        engine._store,
        "latest_canonical_store_ids_for_messages",
        _without_ingest_rows(engine._store, foreign),
    )
    try:
        engine.compress(host[:5] + [foreign, host[5]], current_tokens=340_000)
        assert calls == []
        assert engine._dag.get_session_nodes("s") == []
        assert engine._last_compacted_store_id == 0
    finally:
        engine.shutdown()


def _without_ingest_rows(store, foreign):
    """Map ``foreign`` to its other-conversation row, not a fresh ingest copy."""
    original = store.latest_canonical_store_ids_for_messages
    foreign_id = int(
        store.connection.execute(
            "SELECT store_id FROM messages WHERE conversation_id='other'"
        ).fetchone()[0]
    )

    def locate(session_id, messages, **kwargs):
        located = original(session_id, messages, **kwargs)
        return [
            foreign_id if message.get("content") == "foreign row" else store_id
            for message, store_id in zip(messages, located)
        ]

    return locate


def test_partial_interval_consumption_cuts_at_last_summarized_row(tmp_path, monkeypatch):
    engine = LCMEngine(_config(tmp_path / "partial.db"))
    try:
        snapshot = [{"store_id": store_id} for store_id in (11, 12, 13, 14, 15)]
        aliases = {13: 11}
        summary_input = [{"content": "a"}, {"content": "b"}, {"content": "c"}, {"content": "d"}]
        summary_ids = [11, 12, 14, 15]
        chunk = [{"id": 1}, {"id": 2}, {"id": 3}]
        monkeypatch.setattr(
            engine,
            "_foreground_chunk_source_ids",
            lambda _chunk, deadline=None: [(11,), (13,), (15,)],
        )
        consumed, retired, _tokens = engine._interval_consumed_prefix(
            chunk, snapshot, aliases, summary_input, summary_ids, summary_input[:2]
        )
        # Cut after row 12: the duplicate 13 (alias of 11) is not yet consumed,
        # but its caller message is retired because its original is covered.
        assert consumed == [11, 12]
        assert retired == 2
        # A non-prefix summarizer result is refused.
        assert engine._interval_consumed_prefix(
            chunk, snapshot, aliases, summary_input, summary_ids, summary_input[1:3]
        ) is None
    finally:
        engine.shutdown()


def _hidden_row_setup(tmp_path, monkeypatch, hidden):
    engine = LCMEngine(_config(tmp_path / "hidden.db"))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    host = _host()
    store = engine._store
    store.append_batch("s", host[:2], conversation_id="c")
    # A row the caller no longer shows, stored between rows it does show.
    store.append_batch("s", [hidden], conversation_id="c")
    store.append_batch("s", host[2:5], conversation_id="c")
    provider_inputs: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: provider_inputs.append(kwargs["text"])
        or ("interval summary", 1),
    )
    return engine, host, provider_inputs


def test_hidden_older_host_variant_below_caller_max_is_summarized(tmp_path, monkeypatch):
    hidden = {"role": "user", "content": "dropped older turn", "timestamp": TS + 1.5}
    engine, host, provider_inputs = _hidden_row_setup(tmp_path, monkeypatch, hidden)
    try:
        engine.compress(host, current_tokens=340_000)
        assert len(provider_inputs) == 1
        assert "dropped older turn" in provider_inputs[0]
        assert engine._dag.get_session_node_count("s") == 1
    finally:
        engine.shutdown()


def test_hidden_row_with_newer_host_timestamp_refuses(tmp_path, monkeypatch):
    hidden = {"role": "user", "content": "from the future", "timestamp": TS + 999}
    engine, host, provider_inputs = _hidden_row_setup(tmp_path, monkeypatch, hidden)
    try:
        engine.compress(host, current_tokens=340_000)
        assert provider_inputs == []
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        engine.shutdown()
