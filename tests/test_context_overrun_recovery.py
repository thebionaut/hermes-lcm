"""Archive amplification cannot make emergency recovery depend on full replay."""
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from hermes_lcm.summary_serialization import project_active_replay_messages


@pytest.mark.parametrize("trigger", ["archive", "deadline", "hard"])
@pytest.mark.parametrize("safe", [8_000, 16_000])
def test_emergency_preserves_archive_and_recovers_without_full_projection(tmp_path, monkeypatch, trigger, safe):
    engine = LCMEngine(LCMConfig(
        database_path=str(tmp_path / "recovery.db"),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    ))
    try:
        engine.on_session_start("s", conversation_id="c", context_length=safe * 2)
        engine.bind_context_budget(SimpleNamespace(context_window=safe * 2, safe_input_tokens=safe))
        history = [{"role": "user", "content": f"archived-{i}"} for i in range(1100 if trigger == "archive" else 2)]
        engine.ingest(history)
        messages = [{"role": "user", "content": "Original objective"}]
        messages += [{"role": "assistant", "content": f"step-{i} " + " token" * (safe // 20)} for i in range(30)]
        messages += [
            {"role": "user", "content": "LATEST USER: finish this objective"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "latest-tool", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "latest-tool", "content": "LATEST RESULT"},
        ]
        engine.ingest(messages)
        if trigger != "hard":
            safe = int(count_messages_tokens(messages) / 0.97)
            engine.bind_context_budget(SimpleNamespace(context_window=safe * 2, safe_input_tokens=safe))
        before = engine._store.get_session_messages("s")
        frontier = engine._last_compacted_store_id
        if trigger == "deadline":
            engine._last_foreground_deadline = {"status": "exhausted", "session_id": "s", "conversation_id": "c"}

        def forbidden(*args, **kwargs):
            pytest.fail("emergency recovery attempted full durable replay")

        monkeypatch.setattr(engine, "_project_durable_active_tail", forbidden)
        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        assert engine._last_compression_status == "overflow_recovery"
        assert count_messages_tokens(result) <= engine._effective_assembly_token_cap() < safe
        assert any(m.get("content") == "LATEST USER: finish this objective" for m in result)
        assert any(m.get("tool_call_id") == "latest-tool" for m in result)
        assert any(m.get("tool_calls") for m in result)
        assert engine._last_compacted_store_id == frontier
        assert engine._store.get_session_messages("s") == before
        engine.ingest(result)
        assert engine._store.get_session_count("s") == len(before)
        engine.shutdown()
        engine = LCMEngine(engine._config)
        engine.on_session_start("s", conversation_id="c", context_length=safe * 2)
        engine.ingest(result)
        assert engine._store.get_session_count("s") == len(before)
    finally:
        engine.shutdown()


def test_projection_cache_keeps_independent_values_and_markers():
    config = LCMConfig()
    messages = [{"role": "user", "content": [{"type": "text", "text": "same"}], "_lcm_replay_store_id": i} for i in (1, 2)]
    result = project_active_replay_messages(messages, config=config)
    assert result == messages
    result[0]["content"][0]["text"] = "changed"
    assert result[1]["content"][0]["text"] == "same"
    assert messages[0]["content"][0]["text"] == "same"


def test_projection_non_json_extension_remains_supported():
    message = {"role": "user", "content": "text", "extension": b"bytes"}
    assert project_active_replay_messages([message], config=LCMConfig()) == [message]


def test_repeated_multi_edits_and_restart_do_not_reingest(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "restart.db"), background_compaction_workers=0)
    engine = LCMEngine(config)
    messages = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}, {"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]
    try:
        engine.on_session_start("s", conversation_id="c")
        engine.ingest(messages)
        for i in range(10):
            messages = [dict(m) for m in messages]
            messages[0]["content"] = f"A-{i}"
            messages[2]["content"] = f"C-{i}"
            engine.ingest(messages)
        assert engine._store.get_session_count("s") == 24
        engine.shutdown()
        engine = LCMEngine(config)
        engine.on_session_start("s", conversation_id="c")
        engine.ingest(messages)
        assert engine._store.get_session_count("s") == 24
        # A genuinely new identical message must still be persisted.
        engine.ingest([*messages, dict(messages[-1])])
        assert engine._store.get_session_count("s") == 25
    finally:
        engine.shutdown()
