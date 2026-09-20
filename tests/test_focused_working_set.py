from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens
from hermes_lcm.working_set import (
    WORKING_SET_REFERENCE_PREFIX,
    select_phase_working_set,
)


def _text(label: str, words: int = 180) -> str:
    return label + " " + " ".join(f"token{index % 17}" for index in range(words))


def _history(blocks: int = 24) -> list[dict]:
    messages: list[dict] = []
    for index in range(blocks):
        messages.extend(
            [
                {"role": "user", "content": _text(f"historical request {index}")},
                {"role": "assistant", "content": _text(f"historical answer {index}")},
            ]
        )
    messages.extend(
        [
            {"role": "assistant", "content": _text("verified checkpoint build 41")},
            {"role": "assistant", "content": _text("remaining item database proof")},
            {"role": "user", "content": _text("Goal: finish current objective")},
        ]
    )
    return messages


def _decision(messages, **overrides):
    values = {
        "target_tokens": 2_400,
        "trigger_tokens": 2_800,
        "min_reclaim_tokens": 500,
        "min_reclaim_ratio": 0.30,
        "cache_state": "unknown",
        "expected_remaining_calls": 4,
    }
    values.update(overrides)
    return select_phase_working_set(messages, **values)


def test_long_replay_selects_bounded_lossless_reference_and_required_state():
    messages = [{"role": "system", "content": "immutable authority"}, *_history()]

    decision = _decision(messages)

    assert decision.reason == "selected"
    assert decision.messages is not None
    assert decision.selected_tokens < decision.source_tokens * 0.30
    assert decision.messages[0] == messages[0]
    rendered = "\n".join(str(message.get("content") or "") for message in decision.messages)
    assert WORKING_SET_REFERENCE_PREFIX in rendered
    assert "verified checkpoint build 41" in rendered
    assert "remaining item database proof" in rendered
    assert "Goal: finish current objective" in rendered
    assert decision.source_sha256 in rendered
    assert decision.omitted_ranges


def test_below_trigger_is_byte_preserving_noop():
    messages = [{"role": "system", "content": "authority"}, {"role": "user", "content": "short"}]
    decision = _decision(messages)
    assert decision.messages is None
    assert decision.reason == "below_trigger"


def test_tool_call_and_results_are_selected_or_omitted_as_one_block():
    old_pair = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-old", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-old", "content": _text("old result", 500)},
    ]
    current_pair = [
        {"role": "user", "content": "Goal: inspect current result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-new", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-new", "content": _text("current result", 80)},
    ]
    messages = [{"role": "system", "content": "authority"}, *old_pair]
    messages.extend(
        {"role": "user", "content": _text(f"filler {index}", 200)}
        for index in range(20)
    )
    messages.extend(current_pair)

    decision = _decision(messages, target_tokens=900, trigger_tokens=1_000)

    assert decision.messages is not None
    selected_ids = {
        str(message.get("tool_call_id") or "")
        for message in decision.messages
        if message.get("role") == "tool"
    }
    assistant_ids = {
        call["id"]
        for message in decision.messages
        for call in message.get("tool_calls") or []
    }
    assert selected_ids == assistant_ids
    assert "call-new" in selected_ids


def test_hot_cache_requires_projected_reseed_payback():
    messages = [{"role": "system", "content": "authority"}, *_history(8)]
    decision = _decision(
        messages,
        target_tokens=4_500,
        trigger_tokens=4_500,
        min_reclaim_tokens=100,
        min_reclaim_ratio=0.10,
        cache_state="hot",
        expected_remaining_calls=1,
    )
    assert decision.messages is None
    assert decision.reason == "cache_payback_not_reached"
    assert decision.payback_calls > 1


def test_required_request_only_prefill_is_never_dropped():
    messages = [
        {"role": "system", "content": "authority"},
        {"role": "user", "content": _text("host profile projection", 500)},
        *_history(),
    ]
    decision = _decision(
        messages,
        target_tokens=3_500,
        trigger_tokens=3_800,
        required_indices={1},
    )
    assert decision.messages is not None
    assert any(
        "host profile projection" in str(message.get("content") or "")
        for message in decision.messages
    )


def test_engine_selection_ingests_once_and_turn_completion_appends(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        focused_working_set_enabled=True,
        focused_working_set_target_tokens=2_400,
        focused_working_set_trigger_tokens=2_800,
        focused_working_set_min_reclaim_tokens=500,
        focused_working_set_min_reclaim_ratio=0.30,
        focused_working_set_payback_calls=4,
    )
    engine = LCMEngine(config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("session", conversation_id="conversation", context_length=200_000)
    conversation = _history()
    request = [{"role": "system", "content": "authority"}, *conversation]

    selected = engine.select_context(
        request,
        conversation_messages=conversation,
        incoming_message=conversation[-1],
        budget_tokens=200_000,
    )
    assert selected is not None
    assert count_messages_tokens(selected) < count_messages_tokens(request) * 0.30
    assert engine._store.get_session_count("session") == len(conversation)

    selected_again = engine.select_context(
        request,
        conversation_messages=conversation,
        incoming_message=conversation[-1],
        budget_tokens=200_000,
    )
    assert selected_again == selected
    assert engine._store.get_session_count("session") == len(conversation)

    completed = [*conversation, {"role": "assistant", "content": "final answer"}]
    engine.on_turn_complete(completed, usage={"input_tokens": 100})
    assert engine._store.get_session_count("session") == len(completed)
    engine.shutdown()


def test_working_set_reference_is_never_reingested(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    engine = LCMEngine(config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("session", conversation_id="conversation", context_length=200_000)
    reference = {
        "role": "user",
        "content": f"{WORKING_SET_REFERENCE_PREFIX}\nsource_sha256=abc",
        "display_kind": "context_projection",
    }
    engine._ingest_messages([reference, {"role": "user", "content": "real source"}])
    rows = engine._store.get_session_messages("session")
    assert [row["content"] for row in rows] == ["real source"]
    engine.shutdown()
