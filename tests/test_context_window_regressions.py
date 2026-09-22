"""Regressions for production context-window and summary serialization bugs."""

from hermes_lcm.codex_routing import _codex_oauth_context_cap
from hermes_lcm.config import LCMConfig
from hermes_lcm.summary_serialization import serialize_summary_messages


def test_summary_serialization_accepts_explicit_null_tool_calls():
    rendered = serialize_summary_messages(
        [
            {
                "role": "assistant",
                "content": "A normal answer without a tool call.",
                "tool_calls": None,
            }
        ],
        config=LCMConfig(),
        session_id="session-null-tool-calls",
    )

    assert rendered == "[ASSISTANT]: A normal answer without a tool call."


def test_gpt56_codex_oauth_uses_current_total_context_window():
    assert _codex_oauth_context_cap("gpt-5.6-sol", "openai-codex") == 900_000
    assert _codex_oauth_context_cap("openai/gpt-5.6", "openai-codex") == 900_000

