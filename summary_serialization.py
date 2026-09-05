"""Shared, policy-aware preparation of messages for summary providers."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .externalize import (
    ExternalizationOwnership,
    load_externalized_payload,
    maybe_externalize_tool_output,
)
from .extraction import (
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
)
from .ingest_protection import (
    extract_all_externalized_payload_refs,
    redact_sensitive_value,
)
from .message_analysis import (
    _is_synthetic_assistant_noise,
    _matched_tool_call_ids,
    _tool_call_id,
)
from .message_content import stored_text_content_for_pattern_matching
from .message_patterns import compile_message_patterns, matches_message_pattern


def _require_projection_active(
    *,
    deadline: float | None,
    cancel_event: threading.Event | None,
    now: Callable[[], float] | None,
) -> None:
    clock = now or time.monotonic
    if (
        (cancel_event is not None and cancel_event.is_set())
        or (deadline is not None and float(clock()) >= float(deadline))
    ):
        raise TimeoutError("summary serialization deadline exceeded")


def _externalized_text_parts(
    message: Mapping[str, Any],
    *,
    config: Any,
    session_id: str,
    hermes_home: str,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    now: Callable[[], float] | None = None,
) -> list[str]:
    _require_projection_active(
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    ref_sources: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        ref_sources.append(content)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            ref_sources.append(json.dumps(tool_calls, ensure_ascii=False))
        except (TypeError, ValueError):
            ref_sources.append(str(tool_calls))
    refs: list[str] = []
    for source in ref_sources:
        for ref in extract_all_externalized_payload_refs(source):
            if ref not in refs:
                refs.append(ref)
    parts: list[str] = []
    message_session_id = str(message.get("session_id") or session_id or "")
    for ref in refs:
        _require_projection_active(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
        )
        payload = load_externalized_payload(
            ref,
            config=config,
            hermes_home=hermes_home,
        )
        _require_projection_active(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
        )
        if not payload:
            continue
        payload_session_id = str(payload.get("session_id") or "")
        if (
            message_session_id
            and payload_session_id
            and payload_session_id != message_session_id
        ):
            continue
        payload_content = payload.get("content")
        if isinstance(payload_content, str):
            parts.append(payload_content)
    return parts


def filter_summary_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    config: Any,
    session_id: str = "",
    hermes_home: str = "",
    resolve_externalized: bool = True,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    now: Callable[[], float] | None = None,
) -> list[dict[str, Any]]:
    """Apply stored-row ignore policy and dependent-reply suppression."""
    _require_projection_active(
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    compiled = compile_message_patterns(
        list(getattr(config, "ignore_message_patterns", ()) or ())
    )
    if not compiled:
        return [dict(message) for message in messages]

    kept: list[dict[str, Any]] = []
    drop_dependent_reply = False
    for value in messages:
        _require_projection_active(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
        )
        message = dict(value)
        role = str(message.get("role") or "")
        content = stored_text_content_for_pattern_matching(message.get("content")) or ""
        ignored = matches_message_pattern(content, compiled)
        if not ignored and resolve_externalized:
            ignored = any(
                matches_message_pattern(part, compiled)
                for part in _externalized_text_parts(
                    message,
                    config=config,
                    session_id=session_id,
                    hermes_home=hermes_home,
                    deadline=deadline,
                    cancel_event=cancel_event,
                    now=now,
                )
                if part
            )
        if ignored:
            if role in {"user", "system", "assistant", "tool"}:
                drop_dependent_reply = True
            continue
        if drop_dependent_reply and role in {"assistant", "tool"}:
            continue
        if role in {"user", "system"}:
            drop_dependent_reply = False
        kept.append(message)
    _require_projection_active(
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    return kept


def serialize_summary_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    config: Any,
    session_id: str = "",
    hermes_home: str = "",
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    now: Callable[[], float] | None = None,
    ownership: ExternalizationOwnership | None = None,
) -> str:
    """Serialize messages through the canonical summary-provider scrub."""
    _require_projection_active(
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    normalized = [dict(message) for message in messages]
    parts: list[str] = []
    matched_tool_ids = _matched_tool_call_ids(normalized)
    for message in normalized:
        _require_projection_active(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
        )
        role = message.get("role", "unknown")
        content = redact_sensitive_value(
            message.get("content") or "",
            config,
            parse_json_strings=False,
        )
        if role == "tool":
            tool_id = str(message.get("tool_call_id") or "").strip()
            externalized = maybe_externalize_tool_output(
                content,
                tool_call_id=tool_id,
                session_id=session_id,
                config=config,
                hermes_home=hermes_home,
                deadline=deadline,
                cancel_event=cancel_event,
                now=now,
                ownership=ownership,
            )
            _require_projection_active(
                deadline=deadline,
                cancel_event=cancel_event,
                now=now,
            )
            if externalized:
                content = externalized["placeholder"]
            else:
                content = sanitize_pre_compaction_content(content)
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[TOOL RESULT {tool_id}]: {content}")
            continue

        content = sanitize_pre_compaction_content(content)
        if role == "assistant":
            # Persisted Hermes rows use SQL NULL for assistant messages that
            # did not make a tool call.  ``dict.get(..., [])`` only applies
            # the default when the key is absent, so normalize an explicit
            # None before iterating.
            tool_calls = message.get("tool_calls") or []
            matched_tool_calls = [
                tool_call
                for tool_call in tool_calls
                if not _tool_call_id(tool_call)
                or _tool_call_id(tool_call) in matched_tool_ids
            ]
            if _is_synthetic_assistant_noise(content):
                if not matched_tool_calls:
                    continue
                content = ""
            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            if matched_tool_calls:
                call_parts: list[str] = []
                for tool_call in matched_tool_calls:
                    _require_projection_active(
                        deadline=deadline,
                        cancel_event=cancel_event,
                        now=now,
                    )
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function", {})
                    name = function.get("name", "?")
                    arguments = redact_sensitive_value(
                        function.get("arguments", ""),
                        config,
                        parse_json_strings=True,
                    )
                    arguments = sanitize_pre_compaction_tool_arguments(arguments)
                    if len(arguments) > 500:
                        arguments = arguments[:400] + "..."
                    call_parts.append(f"  {name}({arguments})")
                content += "\n[Tool calls:\n" + "\n".join(call_parts) + "\n]"
            parts.append(f"[ASSISTANT]: {content}")
            continue

        if len(content) > 3000:
            content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
        parts.append(f"[{str(role).upper()}]: {content}")
    _require_projection_active(
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    return "\n\n".join(parts)


def project_active_replay_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    config: Any,
    session_id: str = "",
    hermes_home: str = "",
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
    now: Callable[[], float] | None = None,
) -> list[dict[str, Any]]:
    """Project durable raw rows into a policy-safe active replay view.

    Active replay is not summary-provider input.  Preserve provider-visible
    content and structured media exactly as stored; only apply the configured
    ignore/dependent-message filter and sensitive-value redaction here.  The
    summary serializer owns pre-compaction cleanup and truncation.
    """
    _require_projection_active(
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    filtered = filter_summary_messages(
        messages,
        config=config,
        session_id=session_id,
        hermes_home=hermes_home,
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    projected: list[dict[str, Any]] = []
    for message in filtered:
        _require_projection_active(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
        )
        protected = redact_sensitive_value(
            dict(message),
            config,
            parse_json_strings=True,
        )
        projected.append(protected)
    _require_projection_active(
        deadline=deadline,
        cancel_event=cancel_event,
        now=now,
    )
    return projected


__all__ = [
    "filter_summary_messages",
    "project_active_replay_messages",
    "serialize_summary_messages",
]
