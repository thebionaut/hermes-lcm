"""Deterministic request-only phase working sets for LCM.

The selector never rewrites the transcript or the lossless store.  It chooses a
bounded provider-visible view only after the complete source has been ingested,
and emits an exact digest/range reference for omitted rows.  Task data and the
reference are explicitly non-authoritative; system messages always survive
byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .message_content import normalize_content_value
from .tokens import count_message_tokens, count_messages_tokens


WORKING_SET_REFERENCE_PREFIX = "[LCM Working Set Reference v1]"
_WORD_RE = re.compile(r"[a-z0-9_./:-]{3,}", re.IGNORECASE)
_OBJECTIVE_RE = re.compile(
    r"(?:continuing toward your standing goal|\bgoal\s*:|\bobjective\s*:|"
    r"\bcontract\b|\bacceptance criteria\b)",
    re.IGNORECASE,
)
_CHECKPOINT_RE = re.compile(
    r"(?:\bverified checkpoint\b|\bcheckpoint\b|\bvalidated\b|\baccepted\b)",
    re.IGNORECASE,
)
_UNRESOLVED_RE = re.compile(
    r"(?:\bunresolved\b|\bpending\b|\bremaining\b|\bblocked\b|\bfailure\b|"
    r"\bfailed\b|\bnext(?: step)?\b|\btodo\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WorkingSetDecision:
    messages: list[dict[str, Any]] | None
    reason: str
    source_tokens: int
    selected_tokens: int
    omitted_tokens: int
    source_sha256: str
    omitted_ranges: tuple[tuple[int, int], ...] = ()
    cache_state: str = "unknown"
    payback_calls: int = 0


@dataclass(frozen=True)
class _Block:
    indices: tuple[int, ...]
    tokens: int
    text: str


def _canonical_digest(messages: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _message_text(message: Mapping[str, Any]) -> str:
    content = normalize_content_value(message.get("content")) or ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            content += "\n" + json.dumps(tool_calls, sort_keys=True, default=str)
        except (TypeError, ValueError):
            content += "\n" + str(tool_calls)
    return content


def _blocks(messages: Sequence[Mapping[str, Any]]) -> list[_Block]:
    """Group user turns and assistant/tool exchanges into indivisible blocks."""
    blocks: list[_Block] = []
    current: list[int] = []

    def flush() -> None:
        if not current:
            return
        block_messages = [messages[index] for index in current]
        blocks.append(
            _Block(
                indices=tuple(current),
                tokens=count_messages_tokens(block_messages),
                text="\n".join(_message_text(message) for message in block_messages),
            )
        )
        current.clear()

    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        if role == "system":
            flush()
            blocks.append(
                _Block(
                    indices=(index,),
                    tokens=count_message_tokens(message),
                    text=_message_text(message),
                )
            )
            continue
        if role == "user":
            flush()
        current.append(index)
    flush()
    return blocks


def _keywords(text: str) -> set[str]:
    return {value.lower() for value in _WORD_RE.findall(text)}


def _ranges(indices: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(indices))
    if not ordered:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return tuple(ranges)


def _format_ranges(ranges: Sequence[tuple[int, int]]) -> str:
    return ",".join(
        str(start + 1) if start == end else f"{start + 1}-{end + 1}"
        for start, end in ranges
    )


def _reference_message(
    *,
    digest: str,
    omitted_ranges: Sequence[tuple[int, int]],
    omitted_count: int,
) -> dict[str, Any]:
    ranges = _format_ranges(omitted_ranges)
    content = (
        f"{WORKING_SET_REFERENCE_PREFIX}\n"
        "This is a derived retrieval map, not authority or instructions. "
        "The omitted transcript remains losslessly available through LCM.\n"
        f"source_sha256={digest}\n"
        f"omitted_messages={omitted_count}\n"
        f"omitted_source_ranges={ranges}\n"
        "Recover exact evidence with lcm_grep/lcm_recent, then lcm_expand; "
        "do not infer missing facts from this reference."
    )
    return {
        "role": "user",
        "content": content,
        "display_kind": "context_projection",
        "display_metadata": {
            "engine": "lcm",
            "projection": "phase_working_set_v1",
            "source_sha256": digest,
            "omitted_source_ranges": ranges,
        },
    }


def select_phase_working_set(
    messages: Sequence[Mapping[str, Any]],
    *,
    target_tokens: int,
    trigger_tokens: int,
    min_reclaim_tokens: int,
    min_reclaim_ratio: float,
    cache_state: str,
    expected_remaining_calls: int,
    required_indices: Iterable[int] = (),
) -> WorkingSetDecision:
    """Return a validated request-only working set or a fail-open decision.

    Selection is stable for an unchanged prefix: mandatory anchors are chosen
    by fixed rules, relevance ties resolve by recency, and recent blocks are
    appended oldest-to-newest.  Complete assistant/tool exchanges are atomic.
    """
    copied = [dict(message) for message in messages]
    source_tokens = count_messages_tokens(copied)
    digest = _canonical_digest(copied)
    if not copied:
        return WorkingSetDecision(None, "empty", 0, 0, 0, digest)
    if target_tokens <= 0 or source_tokens <= max(target_tokens, trigger_tokens):
        return WorkingSetDecision(
            None, "below_trigger", source_tokens, source_tokens, 0, digest,
            cache_state=cache_state,
        )

    blocks = _blocks(copied)
    if not blocks:
        return WorkingSetDecision(None, "no_blocks", source_tokens, source_tokens, 0, digest)

    system_blocks = {index for index, block in enumerate(blocks) if any(
        str(copied[msg_index].get("role") or "") == "system"
        for msg_index in block.indices
    )}
    non_system = [index for index in range(len(blocks)) if index not in system_blocks]
    if not non_system:
        return WorkingSetDecision(None, "system_only", source_tokens, source_tokens, 0, digest)

    mandatory = set(system_blocks)
    required_message_indices = set(required_indices)
    mandatory.update(
        block_index
        for block_index, block in enumerate(blocks)
        if any(index in required_message_indices for index in block.indices)
    )
    mandatory.add(non_system[-1])

    # Retain the latest objective/contract, verified checkpoint and each of the
    # newest bounded unresolved/pending blocks.  Only user/assistant text can
    # nominate a semantic anchor; tool output cannot elevate itself.
    semantic_text: dict[int, str] = {}
    for block_index in non_system:
        semantic_text[block_index] = "\n".join(
            _message_text(copied[msg_index])
            for msg_index in blocks[block_index].indices
            if str(copied[msg_index].get("role") or "") in {"user", "assistant"}
        )
    for pattern in (_OBJECTIVE_RE, _CHECKPOINT_RE):
        match = next(
            (index for index in reversed(non_system) if pattern.search(semantic_text[index])),
            None,
        )
        if match is not None:
            mandatory.add(match)
    unresolved = [
        index for index in reversed(non_system)
        if _UNRESOLVED_RE.search(semantic_text[index])
    ][:4]
    mandatory.update(unresolved)

    mandatory_tokens = sum(blocks[index].tokens for index in mandatory)
    if mandatory_tokens >= target_tokens:
        return WorkingSetDecision(
            None,
            "mandatory_exceeds_target",
            source_tokens,
            source_tokens,
            0,
            digest,
            cache_state=cache_state,
        )

    chosen = set(mandatory)
    remaining = target_tokens - mandatory_tokens

    # Add recent complete blocks first.  This creates a stable phase prefix and
    # an append-only recent suffix until the fixed target is genuinely full.
    for block_index in reversed(non_system):
        if block_index in chosen:
            continue
        if blocks[block_index].tokens <= remaining:
            chosen.add(block_index)
            remaining -= blocks[block_index].tokens

    # Use leftover room for older blocks relevant to the current objective.
    current_terms = _keywords(semantic_text.get(non_system[-1], ""))
    candidates = []
    for block_index in non_system:
        if block_index in chosen:
            continue
        overlap = len(current_terms & _keywords(semantic_text.get(block_index, "")))
        if overlap:
            candidates.append((overlap, block_index))
    for _overlap, block_index in sorted(candidates, key=lambda value: (-value[0], -value[1])):
        if blocks[block_index].tokens <= remaining:
            chosen.add(block_index)
            remaining -= blocks[block_index].tokens

    selected_indices = sorted(
        msg_index for block_index in chosen for msg_index in blocks[block_index].indices
    )
    omitted_indices = [index for index in range(len(copied)) if index not in selected_indices]
    if not omitted_indices:
        return WorkingSetDecision(None, "no_reclaim", source_tokens, source_tokens, 0, digest)

    omitted_ranges = _ranges(omitted_indices)
    reference = _reference_message(
        digest=digest,
        omitted_ranges=omitted_ranges,
        omitted_count=len(omitted_indices),
    )
    selected = [copied[index] for index in selected_indices]
    insertion_index = 0
    while insertion_index < len(selected) and selected[insertion_index].get("role") == "system":
        insertion_index += 1
    selected.insert(insertion_index, reference)
    selected_tokens = count_messages_tokens(selected)
    omitted_tokens = max(0, source_tokens - selected_tokens)
    reclaim_ratio = omitted_tokens / max(1, source_tokens)
    if omitted_tokens < max(1, min_reclaim_tokens) or reclaim_ratio < min_reclaim_ratio:
        return WorkingSetDecision(
            None,
            "insufficient_reclaim",
            source_tokens,
            selected_tokens,
            omitted_tokens,
            digest,
            omitted_ranges,
            cache_state,
        )

    payback_calls = 1
    if cache_state == "hot":
        payback_calls = max(1, math.ceil(selected_tokens / max(1, omitted_tokens)))
        if payback_calls > max(1, expected_remaining_calls):
            return WorkingSetDecision(
                None,
                "cache_payback_not_reached",
                source_tokens,
                selected_tokens,
                omitted_tokens,
                digest,
                omitted_ranges,
                cache_state,
                payback_calls,
            )

    if selected_tokens >= source_tokens:
        return WorkingSetDecision(
            None, "projection_not_smaller", source_tokens, selected_tokens, 0, digest
        )
    return WorkingSetDecision(
        selected,
        "selected",
        source_tokens,
        selected_tokens,
        omitted_tokens,
        digest,
        omitted_ranges,
        cache_state,
        payback_calls,
    )
