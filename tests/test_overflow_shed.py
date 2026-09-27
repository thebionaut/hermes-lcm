"""Deterministic overflow shed: a compaction result that still does not fit is brought under the
target by externalizing old tool payloads, never by dropping them (regression: session
20260927_154754_cbb61c ended with context_overflow at ~313K vs a 272K window after summarization
was unavailable)."""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        sensitive_patterns_enabled=False,
        proactive_recall_enabled=False,
        large_output_externalization_enabled=True,
        large_output_active_replay_stubbing_enabled=True,
        large_output_externalization_path=str(tmp_path / "large"),
        fresh_tail_count=4,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine._session_id = "shed-session"
    engine._conversation_id = "shed-session"
    return engine


def _transcript(turns: int, payload_words: int):
    messages = [{"role": "system", "content": "system prompt"}]
    for n in range(turns):
        call_id = f"call_{n}"
        messages += [
            {"role": "user", "content": f"step {n}"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id, "content": f"output {n} " + ("word%d " % n) * payload_words},
            {"role": "assistant", "content": f"done {n}"},
        ]
    return messages


def test_unchanged_over_target_result_is_shed_to_fit(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        messages = _transcript(12, 3_000)
        before = count_messages_tokens(messages)
        engine.threshold_tokens = before // 3
        engine.context_length = before // 2
        # Summarization unavailable: the compaction pass hands back the caller view unchanged.
        monkeypatch.setattr(engine, "_compress_with_terminal_status", lambda msgs, **_: msgs)

        result = engine.compress(messages, current_tokens=before)

        assert count_messages_tokens(result) <= engine._overflow_shed_target()
        assert len(result) == len(messages)  # nothing dropped
        assert [m.get("tool_call_id") for m in result] == [m.get("tool_call_id") for m in messages]
        stubbed = [m for m in result if m.get("role") == "tool" and str(m["content"]).startswith("[Externalized")]
        assert stubbed
        # The protected fresh tail stays verbatim.
        assert result[-4:] == messages[-4:]
    finally:
        engine.shutdown()


def test_result_already_under_target_is_untouched(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    try:
        messages = _transcript(3, 50)
        engine.threshold_tokens = 100_000
        engine.context_length = 200_000
        monkeypatch.setattr(engine, "_compress_with_terminal_status", lambda msgs, **_: msgs)
        assert engine.compress(messages) is messages
    finally:
        engine.shutdown()
