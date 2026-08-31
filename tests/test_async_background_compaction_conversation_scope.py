"""Conversation isolation for post-publication durable replay."""

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_durable_tail_projection_excludes_reused_session_foreign_conversation(tmp_path):
    db_path = tmp_path / "conversation-tail.db"
    engine = LCMEngine(
        LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        )
    )
    engine.on_session_start("shared-session", conversation_id="conversation-a")
    first = engine._store.append_batch(
        "shared-session",
        [{"role": "user", "content": "visible-a"}],
        token_estimates=[1],
        conversation_id="conversation-a",
    )[0]
    engine._store.append_batch(
        "shared-session",
        [{"role": "assistant", "content": "must-not-leak-b"}],
        token_estimates=[1],
        conversation_id="conversation-b",
    )
    last = engine._store.append_batch(
        "shared-session",
        [{"role": "user", "content": "tail-a"}],
        token_estimates=[1],
        conversation_id="conversation-a",
    )[0]
    try:
        rows = engine._all_durable_messages_after(first)
        assert [row["store_id"] for row in rows] == [last]
        projected = engine._project_durable_active_tail(first)
        assert [message["content"] for message in projected] == ["tail-a"]
    finally:
        engine.shutdown()
