"""Lossless request-only previews for old externalized tool results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Collection, Mapping, Sequence

from .externalize import (
    find_externalized_payload_for_message,
    load_externalized_payload,
)
from .message_content import normalize_content_value
from .tokens import count_message_tokens, count_tokens


PREVIEW_PREFIX = "[LCM Active Replay Preview v1]"
RECOVERY_TOOLS = frozenset({"lcm_describe", "lcm_expand"})
TEXT_BLOCK_TYPES = frozenset({"text", "input_text", "output_text"})


@dataclass(frozen=True)
class ActiveReplayPreviewPolicy:
    enabled: bool = False
    version: str = "externalized-tool-preview-v1"
    min_tokens: int = 2_000
    head_tokens: int = 256
    tail_tokens: int = 256
    max_tokens: int = 768
    fresh_tail_count: int = 0


@dataclass(frozen=True)
class ActiveReplayPreviewDecision:
    messages: list[dict[str, Any]] | None
    reason: str
    policy_version: str
    eligible_results: int = 0
    projected_results: int = 0
    source_tokens: int = 0
    projected_tokens: int = 0
    tokens_saved: int = 0
    refs: tuple[str, ...] = ()


def _textual_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not content:
        return None
    for block in content:
        if isinstance(block, str):
            continue
        if not isinstance(block, Mapping):
            return None
        if str(block.get("type") or "").lower() not in TEXT_BLOCK_TYPES:
            return None
    return normalize_content_value(content) or ""


def _replace_text_shape(original: Any, preview: str) -> Any:
    if not isinstance(original, list):
        return preview
    for block in original:
        if isinstance(block, str):
            return [preview]
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type") or "").lower()
        if block_type not in TEXT_BLOCK_TYPES:
            continue
        for key in ("text", "content"):
            value = block.get(key)
            if isinstance(value, str):
                return [{"type": block_type, key: preview}]
            if isinstance(value, Mapping):
                for nested in ("value", "content"):
                    if isinstance(value.get(nested), str):
                        return [{"type": block_type, key: {nested: preview}}]
    return [{"type": "text", "text": preview}]


def _clip_prefix(text: str, token_budget: int) -> str:
    if token_budget <= 0 or not text:
        return ""
    if count_tokens(text) <= token_budget:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid]) <= token_budget:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def _clip_suffix(text: str, token_budget: int) -> str:
    if token_budget <= 0 or not text:
        return ""
    if count_tokens(text) <= token_budget:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[len(text) - mid :]) <= token_budget:
            low = mid
        else:
            high = mid - 1
    return text[len(text) - low :] if low else ""


def _preview_text(content: str, *, ref: str, policy: ActiveReplayPreviewPolicy) -> str:
    digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
    marker = (
        f"{PREVIEW_PREFIX}\n"
        f"policy={policy.version}; original_chars={len(content)}; "
        f"original_sha256={digest}; ref={ref}\n"
    )
    head = _clip_prefix(content, policy.head_tokens)
    tail = _clip_suffix(content, policy.tail_tokens)
    overlap = max(0, len(head) + len(tail) - len(content))
    if overlap:
        tail = tail[overlap:]
    omitted = max(0, len(content) - len(head) - len(tail))
    omission = (
        f"\n[... {omitted} original characters omitted; retrieve exact "
        f"payload with ref={ref} ...]\n"
    )
    rendered = marker + head + omission + tail
    if count_tokens(rendered) <= policy.max_tokens:
        return rendered

    # The marker/reference are mandatory. Split remaining capacity evenly and
    # remeasure the complete rendered preview, shrinking deterministically.
    mandatory_tokens = count_tokens(marker + omission)
    remaining = max(0, policy.max_tokens - mandatory_tokens)
    head_budget = min(policy.head_tokens, remaining // 2)
    tail_budget = min(policy.tail_tokens, remaining - head_budget)
    while True:
        head = _clip_prefix(content, head_budget)
        tail = _clip_suffix(content, tail_budget)
        overlap = max(0, len(head) + len(tail) - len(content))
        if overlap:
            tail = tail[overlap:]
        omitted = max(0, len(content) - len(head) - len(tail))
        omission = (
            f"\n[... {omitted} original characters omitted; retrieve exact "
            f"payload with ref={ref} ...]\n"
        )
        rendered = marker + head + omission + tail
        if count_tokens(rendered) <= policy.max_tokens or not (head_budget or tail_budget):
            return rendered
        if head_budget >= tail_budget and head_budget:
            head_budget -= 1
        elif tail_budget:
            tail_budget -= 1


def _paired_tool_results(
    messages: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[str, str]]:
    """Return result-index -> (call id, tool name) for exact ordered groups."""
    paired: dict[int, tuple[str, str]] = {}
    index = 0
    while index < len(messages):
        message = messages[index]
        if str(message.get("role") or "") != "assistant":
            index += 1
            continue
        calls = message.get("tool_calls") or []
        declared: list[tuple[str, str]] = []
        for call in calls:
            if not isinstance(call, Mapping):
                declared = []
                break
            call_id = str(call.get("id") or call.get("tool_call_id") or "")
            function = call.get("function") or {}
            tool_name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
            if not call_id:
                declared = []
                break
            declared.append((call_id, tool_name))
        if not declared:
            index += 1
            continue
        candidates: list[tuple[int, str, str]] = []
        valid = True
        for offset, (call_id, tool_name) in enumerate(declared, start=1):
            result_index = index + offset
            if result_index >= len(messages):
                valid = False
                break
            result = messages[result_index]
            if (
                str(result.get("role") or "") != "tool"
                or str(result.get("tool_call_id") or "") != call_id
            ):
                valid = False
                break
            candidates.append((result_index, call_id, tool_name))
        if valid:
            for result_index, call_id, tool_name in candidates:
                paired[result_index] = (call_id, tool_name)
            index += len(declared) + 1
        else:
            index += 1
    return paired


def project_externalized_tool_results(
    messages: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    config: Any,
    hermes_home: str,
    policy: ActiveReplayPreviewPolicy,
    excluded_result_indices: Collection[int] = (),
) -> ActiveReplayPreviewDecision:
    source = [dict(message) for message in messages]
    source_tokens = sum(count_message_tokens(message) for message in source)
    if not policy.enabled:
        return ActiveReplayPreviewDecision(
            None, "disabled", policy.version, source_tokens=source_tokens,
            projected_tokens=source_tokens,
        )
    paired = _paired_tool_results(source)
    excluded = {int(index) for index in excluded_result_indices}
    eligible_end = max(0, len(source) - max(0, int(policy.fresh_tail_count)))
    projected = list(source)
    refs: list[str] = []
    eligible = 0
    changed = 0
    for index in sorted(paired):
        if index in excluded:
            continue
        if index >= eligible_end:
            continue
        call_id, tool_name = paired[index]
        if tool_name in RECOVERY_TOOLS:
            continue
        message = source[index]
        content = _textual_content(message.get("content"))
        if not content or content.startswith(PREVIEW_PREFIX):
            continue
        if count_tokens(content) < max(1, int(policy.min_tokens)):
            continue
        eligible += 1
        externalized = find_externalized_payload_for_message(
            content,
            tool_call_id=call_id,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
        )
        if externalized is None:
            continue
        payload = load_externalized_payload(
            str(externalized.get("ref") or ""),
            config=config,
            hermes_home=hermes_home,
        )
        if (
            payload is None
            or str(payload.get("tool_call_id") or "") != call_id
            or str(payload.get("session_id") or "") != str(session_id or "")
            or payload.get("content") != content
        ):
            continue
        preview = _preview_text(
            content,
            ref=str(externalized["ref"]),
            policy=policy,
        )
        replacement = dict(message)
        replacement["content"] = _replace_text_shape(message.get("content"), preview)
        projected[index] = replacement
        refs.append(str(externalized["ref"]))
        changed += 1
    if not changed:
        return ActiveReplayPreviewDecision(
            None,
            "no_qualified_payload",
            policy.version,
            eligible_results=eligible,
            source_tokens=source_tokens,
            projected_tokens=source_tokens,
        )
    projected_tokens = sum(count_message_tokens(message) for message in projected)
    if projected_tokens >= source_tokens:
        return ActiveReplayPreviewDecision(
            None,
            "no_reclaim",
            policy.version,
            eligible_results=eligible,
            source_tokens=source_tokens,
            projected_tokens=projected_tokens,
        )
    return ActiveReplayPreviewDecision(
        projected,
        "projected",
        policy.version,
        eligible_results=eligible,
        projected_results=changed,
        source_tokens=source_tokens,
        projected_tokens=projected_tokens,
        tokens_saved=max(0, source_tokens - projected_tokens),
        refs=tuple(refs),
    )


__all__ = [
    "ActiveReplayPreviewDecision",
    "ActiveReplayPreviewPolicy",
    "PREVIEW_PREFIX",
    "project_externalized_tool_results",
]
