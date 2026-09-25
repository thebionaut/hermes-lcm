"""Idempotent ingest and duplicate-safe durable replay.

Regression for sessions whose raw store accumulated whole-transcript copies
after process restarts and active prefix rewrites, which later made the
durable replay (and therefore compression) larger than its input.
"""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore, message_dedupe_identity

SESSION = "dedupe-session"
CLEARED = "[Old tool output cleared to save context space]"


def _config(tmp_path):
    return LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        sensitive_patterns_enabled=False,
        proactive_recall_enabled=False,
    )


def _engine(tmp_path):
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "hermes"))
    engine._session_id = SESSION
    engine._session_platform = "desktop"
    engine._conversation_id = SESSION
    return engine


def _transcript(turns: int, *, base: float = 1_790_000_000.0):
    messages = []
    for turn in range(turns):
        ts = base + turn * 10
        call_id = f"call_{turn}"
        messages.extend(
            [
                {"role": "user", "content": f"question {turn}", "timestamp": ts},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "terminal", "arguments": f'{{"n": {turn}}}'},
                        }
                    ],
                    "timestamp": ts + 1,
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"tool output {turn} " + "x" * 200,
                    "timestamp": ts + 2,
                },
                {"role": "assistant", "content": f"answer {turn}", "timestamp": ts + 3},
            ]
        )
    return messages


def _row_count(store, session_id=SESSION):
    return store.get_session_count(session_id)


def test_restart_reingest_of_same_history_stores_no_new_rows(tmp_path):
    messages = _transcript(5)
    first = _engine(tmp_path)
    try:
        first._ingest_messages(messages)
        assert _row_count(first._store) == len(messages)
    finally:
        first.shutdown()

    # A fresh process (dashboard/gateway restart) whose cursor reconciliation
    # cannot prove the replay ("persisted ambiguous delta" -> cursor 0).
    second = _engine(tmp_path)
    try:
        second._ingest_cursor = 0
        second._ingest_cursor_needs_reconcile = False
        new_turn = {"role": "user", "content": "brand new", "timestamp": 1_790_100_000.0}
        second._ingest_messages(messages + [new_turn])
        assert _row_count(second._store) == len(messages) + 1
        rows = second._store.get_session_messages(SESSION)
        assert [row["content"] for row in rows][-1] == "brand new"
    finally:
        second.shutdown()


def test_prefix_rewrite_with_cleared_tool_output_stores_no_duplicates(tmp_path):
    messages = _transcript(6)
    engine = _engine(tmp_path)
    try:
        engine._ingest_messages(messages)
        assert _row_count(engine._store) == len(messages)

        # Host rewrites the active prefix: old tool output is cleared in place
        # and the first turn is replaced by a compaction summary message.
        rewritten = []
        for index, message in enumerate(messages):
            if index < 4:
                continue
            copy = dict(message)
            if copy["role"] == "tool" and index < 16:
                copy["content"] = CLEARED
            rewritten.append(copy)
        rewritten = [{"role": "user", "content": "[CONTEXT SUMMARY] earlier turns"}] + rewritten
        rewritten.append({"role": "user", "content": "follow-up", "timestamp": 1_790_200_000.0})
        engine._ingest_messages(rewritten)

        rows = engine._store.get_session_messages(SESSION)
        identities = [message_dedupe_identity(row, stored_row=True) for row in rows]
        keyed = [identity for identity in identities if identity is not None]
        assert len(keyed) == len(set(keyed))
        assert not any(row["content"] == CLEARED for row in rows)
        assert sum(row["content"] == "follow-up" for row in rows) == 1
    finally:
        engine.shutdown()


def test_identical_text_with_new_timestamp_is_still_a_new_message(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine._ingest_messages([{"role": "user", "content": "go", "timestamp": 1.5e9}])
        engine._ingest_messages(
            [
                {"role": "user", "content": "go", "timestamp": 1.5e9},
                {"role": "user", "content": "go", "timestamp": 1.5e9 + 60},
            ]
        )
        assert _row_count(engine._store) == 2
    finally:
        engine.shutdown()


def test_durable_replay_skips_preexisting_duplicate_rows(tmp_path):
    messages = _transcript(4)
    store = MessageStore(tmp_path / "lcm.db")
    try:
        # Simulate the damaged store left behind by older builds: the whole
        # transcript re-inserted twice, once with cleared tool output.
        first_ids = store.append_batch(SESSION, messages, conversation_id=SESSION)
        store.append_batch(SESSION, messages, conversation_id=SESSION)
        cleared = [
            dict(m, content=CLEARED) if m["role"] == "tool" else dict(m)
            for m in messages
        ]
        store.append_batch(SESSION, cleared, conversation_id=SESSION)
        assert store.get_session_count(SESSION) == 3 * len(messages)
    finally:
        store.close()

    engine = _engine(tmp_path)
    try:
        projected = engine._project_durable_active_tail(0)
        assert len(projected) == len(messages)
        assert [m.get("content") for m in projected] == [m["content"] for m in messages]

        # Rows at/below the frontier are covered by summaries: their later
        # copies must not come back as raw tail either.
        frontier = first_ids[7]
        projected = engine._project_durable_active_tail(frontier)
        assert [m.get("content") for m in projected] == [
            m["content"] for m in messages[8:]
        ]

        # Active messages that map to duplicate rows are not resurrected.
        projected = engine._project_durable_active_tail(0, preserve_messages=list(messages))
        assert len(projected) == len(messages)
    finally:
        engine.shutdown()
