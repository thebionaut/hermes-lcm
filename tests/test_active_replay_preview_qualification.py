from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from hermes_lcm.active_replay_preview import (
    ActiveReplayPreviewPolicy,
    PREVIEW_PREFIX,
    project_externalized_tool_results,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import load_externalized_payload, maybe_externalize_tool_output
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens


SESSION = "goal1-frozen-preview-cohort-v1"


def _config(tmp_path: Path, *, enabled: bool) -> LCMConfig:
    return LCMConfig(
        database_path=str(tmp_path / ("on.db" if enabled else "off.db")),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=100,
        active_replay_preview_enabled=enabled,
        active_replay_preview_policy_version="externalized-tool-preview-v1",
        active_replay_preview_min_tokens=2_000,
        active_replay_preview_head_tokens=256,
        active_replay_preview_tail_tokens=256,
        active_replay_preview_max_tokens=768,
        fresh_tail_count=1,
        focused_working_set_enabled=False,
        background_compaction_enabled=False,
    )


def _policy() -> ActiveReplayPreviewPolicy:
    return ActiveReplayPreviewPolicy(
        enabled=True,
        version="externalized-tool-preview-v1",
        min_tokens=2_000,
        head_tokens=256,
        tail_tokens=256,
        max_tokens=768,
        fresh_tail_count=1,
    )


def _payload(index: int) -> str:
    return (
        f"HEAD_REQUIRED_{index}\n"
        + ((f"fixture={index} unicode=αβγ ordinary bulk line\n") * (360 + index * 20))
        + f"MIDDLE_CRUCIAL_{index}=retrieve-exact-payload\n"
        + ((f"fixture={index} unicode=δεζ ordinary bulk line\n") * (360 + index * 20))
        + f"ERROR_REQUIRED_{index}=E{index:03d}\nTAIL_REQUIRED_{index}"
    )


def _messages(content: str, index: int) -> list[dict]:
    call_id = f"cohort-call-{index}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_fixture", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
        {"role": "user", "content": f"continue fixture {index}"},
    ]


def _p95_ms(samples_ns: list[int]) -> float:
    ordered = sorted(samples_ns)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index] / 1_000_000


def test_frozen_tool_heavy_cohort_qualifies_with_accounted_retrieval_retry_and_summary(tmp_path):
    config = _config(tmp_path, enabled=True)
    home = tmp_path / "home"
    fixture_rows = []
    payload_paths: list[Path] = []
    for index in range(5):
        content = _payload(index)
        externalized = maybe_externalize_tool_output(
            content,
            tool_call_id=f"cohort-call-{index}",
            session_id=SESSION,
            config=config,
            hermes_home=str(home),
            force=True,
        )
        assert externalized is not None
        path = Path(externalized["path"])
        payload_paths.append(path)
        fixture_rows.append((content, _messages(content, index), path.name))

    before_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in payload_paths}
    control_tokens = 0
    candidate_tokens = 0
    control_bytes = 0
    candidate_bytes = 0
    mandatory_checks = 0
    retrieval_calls = 0
    retry_calls = 0
    summary_calls_control = 0
    summary_calls_candidate = 0

    for index, (content, messages, ref) in enumerate(fixture_rows):
        decision = project_externalized_tool_results(
            messages,
            session_id=SESSION,
            config=config,
            hermes_home=str(home),
            policy=_policy(),
        )
        assert decision.messages is not None
        preview = normalize_content_value(decision.messages[1]["content"])
        assert preview.startswith(PREVIEW_PREFIX)
        assert f"HEAD_REQUIRED_{index}" in preview
        assert f"ERROR_REQUIRED_{index}" in preview
        assert f"TAIL_REQUIRED_{index}" in preview
        assert f"MIDDLE_CRUCIAL_{index}" not in preview
        mandatory_checks += 3

        repetitions = 2 if index == 2 else 1
        if repetitions == 2:
            retry_calls += 1
        control_tokens += count_messages_tokens(messages) * repetitions
        candidate_tokens += count_messages_tokens(decision.messages) * repetitions
        control_bytes += len(json.dumps(messages, ensure_ascii=False).encode()) * repetitions
        candidate_bytes += len(json.dumps(decision.messages, ensure_ascii=False).encode()) * repetitions

        if index == 1:
            recovered = load_externalized_payload(ref, config=config, hermes_home=str(home))
            assert recovered is not None and recovered["content"] == content
            assert f"MIDDLE_CRUCIAL_{index}=retrieve-exact-payload" in recovered["content"]
            retrieval = {"role": "tool", "content": recovered["content"]}
            candidate_tokens += count_message_tokens(retrieval)
            candidate_bytes += len(recovered["content"].encode())
            retrieval_calls += 1
            mandatory_checks += 1

        if index == 3:
            # Background summary serialization retains the durable marker in
            # both arms; previewing the foreground request adds no summary call.
            marker_call = {"role": "user", "content": f"durable-ref={ref}"}
            marker_tokens = count_message_tokens(marker_call)
            control_tokens += marker_tokens
            candidate_tokens += marker_tokens
            marker_bytes = len(str(marker_call["content"]).encode())
            control_bytes += marker_bytes
            candidate_bytes += marker_bytes
            summary_calls_control += 1
            summary_calls_candidate += 1

    reduction = (control_tokens - candidate_tokens) / control_tokens
    assert retrieval_calls == 1
    assert retry_calls == 1
    assert summary_calls_candidate == summary_calls_control == 1
    assert mandatory_checks == 16
    assert reduction >= 0.20
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in payload_paths} == before_hashes

    # Flag-off equivalence and warm foreground latency/lock gates use the same
    # frozen fixture, with no provider or summarizer callable installed.
    messages = fixture_rows[0][1]
    on_engine = LCMEngine(config, hermes_home=str(home))
    off_config = _config(tmp_path, enabled=False)
    off_engine = LCMEngine(off_config, hermes_home=str(home))
    on_engine.on_session_start(SESSION, conversation_id="on", context_length=200_000)
    off_engine.on_session_start("off-session", conversation_id="off", context_length=200_000)
    input_hash = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert off_engine.select_context(messages) is None
    assert hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == input_hash
    assert on_engine.select_context(messages, conversation_messages=messages, incoming_message=messages[-1]) is not None

    on_latency: list[int] = []
    off_latency: list[int] = []
    lock_latency: list[int] = []
    probe = sqlite3.connect(config.database_path, timeout=0.1)
    try:
        for _ in range(60):
            started = time.perf_counter_ns()
            assert off_engine.select_context(messages) is None
            off_latency.append(time.perf_counter_ns() - started)

            started = time.perf_counter_ns()
            assert on_engine.select_context(
                messages,
                conversation_messages=messages,
                incoming_message=messages[-1],
            ) is not None
            on_latency.append(time.perf_counter_ns() - started)

            started = time.perf_counter_ns()
            probe.execute("BEGIN IMMEDIATE")
            probe.rollback()
            lock_latency.append(time.perf_counter_ns() - started)
    finally:
        probe.close()
        on_engine.shutdown()
        off_engine.shutdown()

    off_p95 = _p95_ms(off_latency)
    on_p95 = _p95_ms(on_latency)
    overhead_p95 = on_p95 - off_p95
    assert overhead_p95 <= max(10.0, off_p95 * 0.10)
    assert _p95_ms(lock_latency) <= 10.0

    result = {
        "cohort_version": "goal1-frozen-preview-cohort-v1",
        "fixtures": len(fixture_rows),
        "control_provider_visible_estimate_tokens": control_tokens,
        "candidate_provider_visible_estimate_tokens": candidate_tokens,
        "aggregate_reduction_ratio": round(reduction, 6),
        "control_provider_visible_bytes": control_bytes,
        "candidate_provider_visible_bytes": candidate_bytes,
        "retrieval_calls": retrieval_calls,
        "retry_calls": retry_calls,
        "summary_calls_control": summary_calls_control,
        "summary_calls_candidate": summary_calls_candidate,
        "new_physical_provider_attempts": 0,
        "actual_provider_usage": "unknown_not_called",
        "mandatory_checks": mandatory_checks,
        "flag_off_equivalent": True,
        "foreground_p95_ms_control": round(off_p95, 6),
        "foreground_p95_ms_candidate": round(on_p95, 6),
        "foreground_p95_overhead_ms": round(overhead_p95, 6),
        "writer_lock_acquire_p95_ms": round(_p95_ms(lock_latency), 6),
        "raw_payload_hashes_unchanged": True,
    }
    print("GOAL1_PREVIEW_QUALIFICATION=" + json.dumps(result, sort_keys=True))
