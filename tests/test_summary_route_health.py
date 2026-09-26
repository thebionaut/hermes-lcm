"""Summary-route circuit health and caller-cleared tool output on rebuild."""

from __future__ import annotations

import logging

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

CLEARED = "[Old tool output cleared to save context space]"


def test_non_compressing_output_does_not_open_the_circuit(monkeypatch, caplog):
    monkeypatch.setattr(
        escalation,
        "_call_llm_for_summary",
        lambda prompt, max_tokens, model="", timeout=None: "verbose " * 400,
    )
    breaker = escalation.SummaryCircuitBreaker(failure_threshold=2, cooldown_seconds=300)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.escalation"):
        for _ in range(3):
            escalation.summarize_with_escalation(
                "source text " * 80,
                source_tokens=200,
                token_budget=50,
                circuit_breaker=breaker,
            )
    assert breaker.allows("")
    assert "non-compressing output" in caplog.text
    assert "circuit opened" not in caplog.text


def test_provider_failures_still_open_the_circuit_with_logged_reason(monkeypatch, caplog):
    monkeypatch.setattr(
        escalation,
        "_call_llm_for_summary",
        lambda prompt, max_tokens, model="", timeout=None: None,
    )
    breaker = escalation.SummaryCircuitBreaker(failure_threshold=2, cooldown_seconds=300)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.escalation"):
        escalation.summarize_with_escalation(
            "source text " * 80,
            source_tokens=200,
            token_budget=50,
            circuit_breaker=breaker,
        )
    assert not breaker.allows("")
    assert "failed: reason=provider_error" in caplog.text


def test_durable_rebuild_keeps_caller_cleared_tool_output(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            sensitive_patterns_enabled=False,
            proactive_recall_enabled=False,
        ),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine._session_id = "s"
    engine._conversation_id = "s"
    stored = [
        {"role": "user", "content": "run it", "timestamp": 1.7e9},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
            ],
            "timestamp": 1.7e9 + 1,
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "full output line\n" * 500,
            "timestamp": 1.7e9 + 2,
        },
        {"role": "assistant", "content": "done", "timestamp": 1.7e9 + 3},
    ]
    try:
        engine._store.append_batch("s", stored, conversation_id="s")
        caller = [dict(m) for m in stored]
        caller[2]["content"] = CLEARED
        rebuilt = engine._project_durable_active_tail(0, preserve_messages=caller)
        tool = next(m for m in rebuilt if m.get("role") == "tool")
        assert tool["content"] == CLEARED
        assert count_messages_tokens(rebuilt) <= count_messages_tokens(caller)
        # The full output stays durable for recall.
        rows = engine._store.get_session_messages("s")
        assert any(r["content"].startswith("full output line") for r in rows)
    finally:
        engine.shutdown()
