"""Emergency behavior while a partial atomic-group repair is pending."""

from __future__ import annotations

import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


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
        background_compaction_prepare_ratio=0.0,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=10_000,
        context_reserve_tokens=100_000,
    )


def _tool_group():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "first", "arguments": "{}"},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "second", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "first result"},
        {"role": "tool", "tool_call_id": "call-b", "content": "second result"},
        {"role": "user", "content": "protected fresh tail"},
    ]


def test_pending_partial_repair_stub_intercepts_before_foreground_provider(
    tmp_path,
    monkeypatch,
):
    """The emergency gate itself is fail-closed independent of repair setup."""
    engine = LCMEngine(_config(tmp_path / "partial-emergency-stub.db"))
    engine.on_session_start(
        "session",
        conversation_id="conversation",
        context_length=372_000,
    )
    messages = _tool_group()
    provider_calls: list[dict] = []
    try:
        engine.ingest(messages)
        raw_count_before = engine._store.get_session_count("session")
        node_count_before = engine._dag.get_session_node_count("session")
        frontier_before = engine._lifecycle.get_by_conversation(
            "conversation"
        ).current_frontier_store_id
        coverage_before = engine._background_compaction.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_source_coverage"
        ).fetchone()[0]
        monkeypatch.setattr(
            engine._background_compaction,
            "has_active_partial_atomic_repair",
            lambda **_kwargs: True,
        )
        monkeypatch.setattr(
            engine,
            "_acquire_foreground_emergency_provider_permit",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("foreground provider permit must not be acquired")
            ),
        )
        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: provider_calls.append(kwargs) or ("must not run", 1),
        )

        started = time.monotonic()
        result = engine.compress(messages, current_tokens=340_000)

        assert time.monotonic() - started < 1.0
        assert result
        assert provider_calls == []
        assert engine.last_compression_noop_reason == (
            "partial atomic repair pending"
        )
        assert engine._store.get_session_count("session") == raw_count_before
        assert engine._dag.get_session_node_count("session") == node_count_before
        assert engine._lifecycle.get_by_conversation(
            "conversation"
        ).current_frontier_store_id == frontier_before
        assert engine._background_compaction.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_source_coverage"
        ).fetchone()[0] == coverage_before
    finally:
        engine.shutdown()


def test_emergency_requeues_after_small_publication_before_partial_check(
    tmp_path,
    monkeypatch,
):
    """A publication that stays critical queues work at its new live frontier."""
    engine = LCMEngine(_config(tmp_path / "partial-emergency-follow-on.db"))
    engine.on_session_start(
        "session",
        conversation_id="conversation",
        context_length=372_000,
    )
    messages = _tool_group()
    enqueue_calls: list[tuple[int, bool]] = []
    inspected_frontiers: list[int] = []
    provider_calls: list[dict] = []

    def record_enqueue(_messages, observed_tokens, **kwargs):
        enqueue_calls.append(
            (
                int(observed_tokens),
                bool(kwargs.get("allow_canonical_prefix_adoption", True)),
            )
        )
        return None

    def publish_small_batch(_messages, *, current_tokens):
        assert current_tokens == 340_000
        engine._lifecycle._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_frontier_store_id=3
            WHERE conversation_id='conversation'
            """
        )
        engine._lifecycle._conn.commit()
        engine._last_compacted_store_id = 3
        return list(messages)

    def pending_at_live_frontier(**kwargs):
        inspected_frontiers.append(int(kwargs["expected_frontier_store_id"]))
        return True

    monkeypatch.setattr(engine, "_maybe_enqueue_background_compaction", record_enqueue)
    monkeypatch.setattr(engine, "_has_publishable_background_job", lambda: True)
    monkeypatch.setattr(
        engine,
        "_publish_prepared_background_compaction",
        publish_small_batch,
    )
    monkeypatch.setattr(
        engine._background_compaction,
        "has_active_partial_atomic_repair",
        pending_at_live_frontier,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: provider_calls.append(kwargs) or ("must not run", 1),
    )
    try:
        result = engine.compress(messages, current_tokens=340_000)

        assert result
        assert provider_calls == []
        assert len(enqueue_calls) == 2
        assert enqueue_calls[0] == (340_000, False)
        assert enqueue_calls[1][1] is False
        assert inspected_frontiers == [3]
        assert engine.last_compression_noop_reason == (
            "partial atomic repair pending"
        )
    finally:
        engine.shutdown()


def test_pending_partial_repair_emergency_uses_no_provider_and_later_publishes(
    tmp_path,
    monkeypatch,
):
    """Critical pressure stays bounded and fail-closed until repair is prepared."""
    db_path = tmp_path / "partial-emergency.db"
    engine = LCMEngine(_config(db_path))
    engine.on_session_start(
        "session",
        conversation_id="conversation",
        context_length=372_000,
    )
    messages = _tool_group()
    provider_calls: list[dict] = []
    try:
        engine.ingest(messages)
        source_ids = [
            int(row[0])
            for row in engine._store.connection.execute(
                "SELECT store_id FROM messages WHERE session_id='session' "
                "ORDER BY store_id"
            ).fetchall()
        ]
        legacy_node_id = engine._dag.add_node(
            SummaryNode(
                session_id="session",
                summary="legacy partial tool exchange",
                token_count=4,
                source_token_count=20,
                source_ids=source_ids[:2],
                source_type="messages",
            )
        )
        canonical_before = [
            tuple(row)
            for row in engine._background_compaction.connection.execute(
                "SELECT node_id, store_id FROM lcm_compaction_node_sources "
                "WHERE session_id='session' ORDER BY node_id, store_id"
            ).fetchall()
        ]
        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: provider_calls.append(kwargs) or ("must not run", 1),
        )

        started = time.monotonic()
        result = engine.compress(messages, current_tokens=340_000)
        assert time.monotonic() - started < 1.0

        assert result
        assert provider_calls == []
        assert engine._store.get_session_count("session") == len(messages)
        assert engine._dag.get_session_node_count("session") == 1
        assert engine._lifecycle.get_by_conversation(
            "conversation"
        ).current_frontier_store_id == 0
        assert [
            tuple(row)
            for row in engine._background_compaction.connection.execute(
                "SELECT node_id, store_id FROM lcm_compaction_node_sources "
                "WHERE session_id='session' ORDER BY node_id, store_id"
            ).fetchall()
        ] == canonical_before
        assert engine._background_compaction.has_active_partial_atomic_repair(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
        )
        assert engine.last_compression_noop_reason == "partial atomic repair pending"
        result_tool_ids = {
            str(message.get("tool_call_id") or "")
            for message in result
            if message.get("role") == "tool"
        }
        result_assistants = [
            message for message in result if message.get("role") == "assistant"
        ]
        if result_assistants:
            declared = {
                str(call.get("id") or "")
                for call in result_assistants[0].get("tool_calls") or []
            }
            assert not declared or declared <= result_tool_ids

        prepared = engine.run_background_compaction_once(
            summarizer=lambda **_kwargs: ("repaired atomic exchange", 1)
        )
        assert prepared is not None and prepared.state == "prepared"
        published = engine.promote_prepared_compaction(
            prepared.job_id,
            messages,
            current_tokens=340_000,
        )
        assert published.promoted is True
        assert len(published.node_ids) == 2
        assert engine._lifecycle.get_by_conversation(
            "conversation"
        ).current_frontier_store_id == source_ids[2]
        assert engine._store.get_session_count("session") == len(messages)
        assert engine._dag.source_message_ids(
            published.node_ids[-1], limit=100
        ) == source_ids[:3]
        assert legacy_node_id not in {
            node.node_id for node in engine._dag.get_active_frontier_nodes("session")
        }
    finally:
        engine.shutdown()
