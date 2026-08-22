from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path):
    config = LCMConfig(
        sensitive_patterns_enabled=False,
        proactive_recall_enabled=False,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine._session_id = "amplification-session"
    engine._session_platform = "desktop"
    engine._conversation_id = "amplification-session"
    return engine


def test_restart_ingest_never_persists_generated_context_projection(tmp_path):
    engine = _engine(tmp_path)
    projection = {
        "role": "assistant",
        "content": (
            "[Recent Summary (d0, node 42)]\n"
            "generated summary\n"
            "[Expand for details: node=42]"
        ),
        "display_kind": "context_projection",
        "display_metadata": {"engine": "lcm"},
    }
    fresh = {"role": "user", "content": "new user message", "timestamp": 123.0}

    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    engine._ingest_messages([projection, fresh])

    rows = engine._store.get_session_messages("amplification-session")
    assert [row["content"] for row in rows] == ["new user message"]


def test_generated_todo_and_proactive_memory_scaffold_are_not_sources(tmp_path):
    engine = _engine(tmp_path)
    messages = [
        {
            "role": "assistant",
            "content": "[Your active task list was preserved across context compression]\n- generated",
        },
        {"role": "user", "content": "<relevant-memories>\ngenerated recall\n</relevant-memories>"},
        {"role": "user", "content": "real source", "timestamp": 456.0},
    ]

    engine._ingest_messages(messages)

    rows = engine._store.get_session_messages("amplification-session")
    assert [row["content"] for row in rows] == ["real source"]


def test_assembled_summary_carries_explicit_projection_provenance(tmp_path):
    engine = _engine(tmp_path)
    engine._dag.add_node(
        SummaryNode(
            session_id="amplification-session",
            depth=0,
            summary="node-backed summary",
            token_count=5,
            source_token_count=10,
            source_ids=[1],
            source_type="messages",
            expand_hint="node=1",
        )
    )

    assembled = engine._assemble_context(None, [])

    summary = next(message for message in assembled if "node-backed summary" in message.get("content", ""))
    assert summary["display_kind"] == "context_projection"
    assert summary["display_metadata"] == {"engine": "lcm"}
