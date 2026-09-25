"""Leaf-compaction pipeline for the LCM engine (WS5 Seam 6).

The ``CompactionMixin`` holds the compaction gate + pipeline: ``should_compress``
/ ``should_compress_preflight`` (public), the leaf-candidate and chunk-selection
helpers, and the main ``compress`` entry point. These methods were lifted
verbatim out of ``LCMEngine`` and continue to run bound to the engine instance
(``self`` is the ``LCMEngine``), so they read and write the engine's runtime
state (``_ingest_cursor``, ``_store``, ``_dag``, ``_lifecycle``, status/telemetry
fields, per-turn caches) and call back into engine helpers (ingest,
reconciliation, placeholder-ledger, the summarize-with-rescue step, assembly,
lifecycle) through normal attribute lookup. ``LCMEngine`` mixes this in ahead of
``ContextEngine`` so the mixin's ``compress`` / ``should_compress`` /
``should_compress_preflight`` override the ContextEngine protocol defaults.
"""

from __future__ import annotations

import logging
import json
import hashlib
import inspect
import sqlite3
import threading
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from .background_compaction import ForegroundProviderPermitUnavailable
from .dag import SummaryNode
from .message_content import text_content_for_pattern_matching
from .reconcile import ForegroundDeadlineExceeded
from .sanitize import _contains_sensitive_redaction
from .sqlite_util import _is_sqlite_locked_error
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

_AUTOMATIC_FOREGROUND_MAX_PASSES = 1
_AUTOMATIC_FOREGROUND_MAX_SECONDS = 15.0
_MANUAL_FULL_SWEEP_MAX_PASSES = 12
_MANUAL_FULL_SWEEP_MAX_SECONDS = 120.0


class CompactionMixin:
    def _compaction_payback_status(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
    ) -> tuple[bool, str]:
        """Estimate whether elective leaf compaction repays its total work.

        Summary input/output and a hot-cache reseed are charged before future
        replay savings. Emergency overflow recovery is deliberately evaluated
        by callers first and never reaches this gate.
        """
        if not getattr(self._config, "compaction_payback_enabled", False):
            return True, "disabled"
        if not messages:
            return False, "no source messages"
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail"
        candidate = messages[leading_anchor_count:fresh_tail_start]
        raw_tokens = count_messages_tokens(candidate)
        if raw_tokens <= 0:
            return False, "no eligible raw backlog outside fresh tail"
        chunk_tokens = min(
            raw_tokens,
            self._working_leaf_chunk_tokens(raw_tokens),
        )
        summary_tokens = max(
            256,
            int(chunk_tokens * float(self._config.compaction_payback_summary_ratio)),
        )
        reclaim_tokens = max(0, chunk_tokens - summary_tokens)
        minimum_reclaim = max(
            1,
            int(self._config.compaction_payback_min_reclaim_tokens),
        )
        source_tokens = max(
            count_messages_tokens(messages),
            int(observed_tokens or 0),
        )
        cache_hot = bool(
            getattr(self, "last_cache_read_tokens", 0) > 0
            or getattr(self, "last_cache_write_tokens", 0) > 0
        )
        reseed_tokens = max(0, source_tokens - reclaim_tokens) if cache_hot else 0
        one_time_tokens = chunk_tokens + summary_tokens + reseed_tokens
        expected_calls = max(1, int(self._config.compaction_payback_expected_calls))
        projected_savings = reclaim_tokens * expected_calls
        source_digest = hashlib.sha256(
            json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        allowed = bool(
            reclaim_tokens >= minimum_reclaim
            and projected_savings > one_time_tokens
        )
        previous = getattr(self, "_last_compaction_payback", {}) or {}
        same_denial_count = 0
        if not allowed:
            same_denial_count = (
                int(previous.get("same_source_denials", 0) or 0) + 1
                if previous.get("source_sha256") == source_digest
                and previous.get("allowed") is False
                else 1
            )
        reason = "payback_reached" if allowed else "payback_not_reached"
        if not allowed and same_denial_count >= 2:
            reason = "unchanged_source_checkpoint_required"
        self._last_compaction_payback = {
            "allowed": allowed,
            "reason": reason,
            "source_sha256": source_digest,
            "source_tokens": source_tokens,
            "chunk_tokens": chunk_tokens,
            "estimated_summary_tokens": summary_tokens,
            "estimated_reclaim_tokens": reclaim_tokens,
            "estimated_reseed_tokens": reseed_tokens,
            "estimated_one_time_tokens": one_time_tokens,
            "expected_phase_calls": expected_calls,
            "projected_savings_tokens": projected_savings,
            "cache_state": "hot" if cache_hot else (
                "cold" if getattr(self, "cache_metrics_available", False) else "unknown"
            ),
            "same_source_denials": same_denial_count,
        }
        return allowed, reason

    def _maybe_reclassify_late_auxiliary_before_compaction_write(
        self,
        *,
        deadline: float | None = None,
    ) -> None:
        maybe_reclassify = getattr(
            self,
            "_maybe_reclassify_current_session_as_auxiliary_before_message_ingest",
            None,
        )
        if callable(maybe_reclassify):
            try:
                self._call_with_foreground_deadline(
                    maybe_reclassify,
                    deadline=deadline,
                )
            except TimeoutError as exc:
                if deadline is None:
                    raise
                raise ForegroundDeadlineExceeded(
                    "session classification",
                    "auxiliary classification exceeded the foreground deadline",
                ) from exc

    def _background_publication_trigger_tokens(self) -> int:
        """Fixed-reserve publication boundary for the effective context window."""
        reserve = max(0, int(self._config.context_reserve_tokens or 0))
        if reserve > 0 and self.context_length > 0:
            return max(1, int(self.context_length) - reserve)
        return max(0, int(self.threshold_tokens or 0))

    def _background_emergency_trigger_tokens(self) -> int:
        """Bounded synchronous fallback boundary, independent of preparation."""
        context_length = max(0, int(self.context_length or 0))
        if context_length <= 0:
            return 0
        backlog_floor = max(32_000, int(context_length * 0.05))
        return max(1, context_length - backlog_floor)

    def _background_preflight_requested(
        self,
        observed_tokens: int,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Whether foreground work is justified by publish or emergency pressure."""
        self._foreground_deadline_checkpoint(
            deadline,
            "background preflight inspection",
        )
        try:
            if self._call_with_foreground_deadline(
                self._has_newer_committed_background_frontier,
                deadline=deadline,
            ):
                return True
        except (ForegroundDeadlineExceeded, TimeoutError):
            if deadline is not None:
                raise
        except Exception:
            logger.warning(
                "LCM could not inspect the durable background frontier",
                exc_info=True,
            )
        emergency_trigger = self._background_emergency_trigger_tokens()
        if emergency_trigger > 0 and observed_tokens >= emergency_trigger:
            return True
        publication_trigger = self._background_publication_trigger_tokens()
        publishable = False
        if publication_trigger > 0 and observed_tokens >= publication_trigger:
            try:
                publishable = self._call_with_foreground_deadline(
                    self._has_publishable_background_job,
                    deadline=deadline,
                )
            except (ForegroundDeadlineExceeded, TimeoutError):
                if deadline is not None:
                    raise
            except Exception:
                logger.warning(
                    "LCM could not inspect prepared background work",
                    exc_info=True,
                )
        self._foreground_deadline_checkpoint(
            deadline,
            "background preflight inspection",
        )
        return bool(
            publication_trigger > 0
            and observed_tokens >= publication_trigger
            and publishable
        )

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._bypasses_lcm_context_management():
            if self._compression_boundary_cooldown_active():
                return False
            if prompt_tokens is not None:
                tokens = prompt_tokens
            else:
                auxiliary_session_id = self._thread_context_session_id()
                if auxiliary_session_id:
                    tokens = self._current_auxiliary_prompt_tokens(auxiliary_session_id)
                else:
                    tokens = self.last_prompt_tokens
            if self._should_force_overflow_recovery(observed_tokens=tokens):
                return True
            if self.threshold_tokens <= 0:
                return False
            return tokens >= self.threshold_tokens
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._background_compaction_enabled():
            if self._should_force_overflow_recovery(observed_tokens=tokens):
                return True
            return self._background_preflight_requested(max(0, int(tokens or 0)))
        if self._compression_boundary_cooldown_active():
            return False
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        if tokens < self.threshold_tokens:
            return False
        replay = list(getattr(self, "_last_active_replay_messages", ()) or ())
        if replay:
            allowed, reason = self._compaction_payback_status(
                replay,
                observed_tokens=int(tokens or 0),
            )
            if not allowed:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = reason
            return allowed
        return True

    def _clear_foreground_preflight_deadline_handoff(self) -> None:
        context = getattr(self, "_thread_context", None)
        if context is not None and hasattr(
            context,
            "foreground_preflight_deadline_handoff",
        ):
            del context.foreground_preflight_deadline_handoff

    def _remember_foreground_preflight_deadline_handoff(
        self,
        messages: List[Dict[str, Any]],
        *,
        started_at: float,
        deadline: float,
    ) -> None:
        context = getattr(self, "_thread_context", None)
        if context is None:
            return
        context.foreground_preflight_deadline_handoff = {
            "messages_id": id(messages),
            "session_id": str(getattr(self, "_session_id", "") or ""),
            "conversation_id": str(getattr(self, "_conversation_id", "") or ""),
            "started_at": float(started_at),
            "deadline": float(deadline),
        }

    def _take_foreground_preflight_deadline_handoff(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[float, float] | None:
        context = getattr(self, "_thread_context", None)
        handoff = (
            getattr(context, "foreground_preflight_deadline_handoff", None)
            if context is not None
            else None
        )
        self._clear_foreground_preflight_deadline_handoff()
        if not isinstance(handoff, Mapping):
            return None
        if (
            int(handoff.get("messages_id") or 0) != id(messages)
            or str(handoff.get("session_id") or "")
            != str(getattr(self, "_session_id", "") or "")
            or str(handoff.get("conversation_id") or "")
            != str(getattr(self, "_conversation_id", "") or "")
        ):
            return None
        try:
            return float(handoff["started_at"]), float(handoff["deadline"])
        except (KeyError, TypeError, ValueError):
            return None

    def should_compress_preflight(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> bool:
        """Run host preflight inside the automatic turn's single wall."""
        started_at = self._foreground_now()
        local_deadline = started_at + _AUTOMATIC_FOREGROUND_MAX_SECONDS
        foreground_deadline = (
            local_deadline
            if deadline is None
            else min(float(deadline), local_deadline)
        )
        self._clear_foreground_preflight_deadline_handoff()
        try:
            requested = self._should_compress_preflight_impl(
                messages,
                deadline=foreground_deadline,
            )
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "preflight completion",
            )
        except ForegroundDeadlineExceeded as exc:
            self._record_foreground_deadline_exhaustion(
                messages,
                exc,
                started_at=started_at,
                deadline=foreground_deadline,
            )
            return False
        except TimeoutError as exc:
            self._record_foreground_deadline_exhaustion(
                messages,
                ForegroundDeadlineExceeded(
                    "preflight",
                    "operation exceeded the foreground deadline",
                ),
                started_at=started_at,
                deadline=foreground_deadline,
            )
            logger.debug("LCM preflight deadline timeout", exc_info=exc)
            return False
        if requested:
            self._remember_foreground_preflight_deadline_handoff(
                messages,
                started_at=started_at,
                deadline=foreground_deadline,
            )
        return bool(requested)

    def _should_compress_preflight_impl(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float,
    ) -> bool:
        """Pre-flight check — also ingests messages into the store."""
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        self._maybe_reclassify_late_auxiliary_before_compaction_write(
            deadline=deadline,
        )
        if self._call_with_foreground_deadline(
            self._bypasses_lcm_context_management,
            deadline=deadline,
        ):
            session_id = self._call_with_foreground_deadline(
                self._bypass_lcm_session_id,
                deadline=deadline,
            )
            self._foreground_deadline_checkpoint(deadline, "preflight bypass setup")
            self._remember_lcm_bypass_message_prefix(session_id, messages)
            rough = count_messages_tokens(messages)
            self._foreground_deadline_checkpoint(deadline, "preflight token count")
            if self._compression_boundary_cooldown_active():
                return False
            if self._should_force_overflow_recovery(observed_tokens=rough, messages=messages):
                return True
            return self.threshold_tokens > 0 and rough >= self.threshold_tokens
        rough = count_messages_tokens(messages)
        self._foreground_deadline_checkpoint(deadline, "preflight token count")
        background_mode = self._background_compaction_enabled()
        background_pressure = (
            max(rough, int(self.last_prompt_tokens or 0))
            if background_mode
            else rough
        )
        pre_ingest_placeholder_ambiguous_noop = False
        pre_ingest_noop_reason = ""
        if (
            self.threshold_tokens > 0
            and rough >= self.threshold_tokens
            and not self._compiled_ignore_message_patterns
            and any(
                self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                for msg in messages
            )
        ):
            eligible, reason = self._leaf_compaction_candidate_status(
                messages,
                allow_partial_leaf=self._config.threshold_full_sweep_enabled,
                deadline=deadline,
            )
            pre_ingest_placeholder_ambiguous_noop = not eligible
            pre_ingest_noop_reason = reason
        replay_messages = None
        if self._session_id and messages:
            try:
                replay_messages = self._ingest_messages_with_foreground_deadline(
                    messages,
                    deadline,
                )
                self._foreground_deadline_checkpoint(
                    deadline,
                    "preflight ingest",
                )
                self._record_ingest_success()
            except ForegroundDeadlineExceeded:
                raise
            except Exception as e:
                # Fail closed for NORMAL threshold compaction: the store did not
                # accept this turn, so do not compact against a store missing the
                # latest messages - that could rebuild active context without
                # them. But still honor emergency overflow recovery, whose whole
                # job is to keep the prompt under the provider limit; it converges
                # via deterministic L3 truncation without needing the store write.
                self._record_ingest_failure("preflight", e)
                self._foreground_deadline_checkpoint(
                    deadline,
                    "preflight ingest failure",
                )
                if background_mode and self._call_with_foreground_deadline(
                    self._background_preflight_requested,
                    background_pressure,
                    deadline=deadline,
                ):
                    return self._mark_preflight_compression_requested()
                if self._should_force_overflow_recovery(
                    observed_tokens=background_pressure if background_mode else rough
                ):
                    return True
                return False
        if background_mode:
            replay_view = replay_messages if replay_messages is not None else messages
            try:
                self._call_with_foreground_deadline(
                    self._maybe_enqueue_background_compaction,
                    replay_view,
                    background_pressure,
                    deadline=deadline,
                )
            except ForegroundDeadlineExceeded:
                raise
            except Exception:
                # Preparation is opportunistic.  Durable ingest has already
                # succeeded, so queue contention must not block this turn.
                logger.warning(
                    "LCM could not enqueue background compaction",
                    exc_info=True,
                )
            self._foreground_deadline_checkpoint(
                deadline,
                "background compaction enqueue",
            )
            if (
                self._should_force_overflow_recovery(
                    observed_tokens=background_pressure,
                    messages=replay_view,
                )
                or self._call_with_foreground_deadline(
                    self._background_preflight_requested,
                    background_pressure,
                    deadline=deadline,
                )
            ):
                return self._mark_preflight_compression_requested()
            if replay_messages is not None and replay_messages != messages:
                # Preserve the existing safety path for redaction,
                # externalization, and tool-result cleanup.  This is not a
                # summary-model compaction request.
                if self._replay_diff_requests_ingest_cleanup(messages, replay_messages):
                    return self._mark_preflight_compression_requested()
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "background preparation queued"
            return False
        if replay_messages is not None and replay_messages != messages:
            replay_rough = count_messages_tokens(replay_messages)
            cleanup_requested = self._replay_diff_requests_ingest_cleanup(
                messages,
                replay_messages,
            )
            force_overflow_requested = self._should_force_overflow_recovery(
                observed_tokens=rough,
                messages=messages,
            ) or self._should_force_overflow_recovery(
                observed_tokens=replay_rough,
                messages=replay_messages,
            )
            if cleanup_requested:
                if (
                    not force_overflow_requested
                    and self._compression_boundary_cooldown_active()
                ):
                    self._preflight_cleanup_only_due_to_boundary_cooldown = True
                return self._mark_preflight_compression_requested()
            if force_overflow_requested:
                return self._mark_preflight_compression_requested()
            # A boundary skip cools down summary-producing leaf/condensation
            # work. It must not prevent the host from adopting a replay cleanup
            # that ingest has already made durable (for example a live tool
            # result stub); those returns above are deterministic and add no
            # summarizer spend.
            if self._compression_boundary_cooldown_active():
                return False
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(
                replay_messages,
                allow_partial_leaf=bool(
                    self._config.threshold_full_sweep_enabled
                    and self.threshold_tokens > 0
                    and replay_rough >= self.threshold_tokens
                ),
                deadline=deadline,
            )
            if eligible:
                payback_allowed, payback_reason = self._compaction_payback_status(
                    replay_messages,
                    observed_tokens=replay_rough,
                )
                if payback_allowed:
                    return self._mark_preflight_compression_requested()
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = payback_reason
                logger.info("LCM preflight compression no-op: %s", payback_reason)
                return False
            if self._has_ignored_backlog_outside_fresh_tail(
                replay_messages,
                deadline=deadline,
            ):
                return self._mark_preflight_compression_requested()
            if self.threshold_tokens > 0 and replay_rough >= self.threshold_tokens:
                if self._should_run_deferred_maintenance(
                    replay_messages,
                    observed_tokens=replay_rough,
                    deadline=deadline,
                ):
                    return self._mark_preflight_compression_requested()
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = reason
                logger.info("LCM preflight compression no-op: %s", reason)
                return False
            self._refresh_raw_backlog_debt(
                replay_messages,
                observed_tokens=replay_rough,
                deadline=deadline,
            )
            if self._should_run_deferred_maintenance(
                replay_messages,
                observed_tokens=replay_rough,
                deadline=deadline,
            ):
                return self._mark_preflight_compression_requested()
            return False
        if self._compression_boundary_cooldown_active():
            return False
        if self._should_force_overflow_recovery(observed_tokens=rough):
            return self._mark_preflight_compression_requested()
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            if pre_ingest_placeholder_ambiguous_noop:
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = pre_ingest_noop_reason
                logger.info("LCM preflight compression no-op: %s", pre_ingest_noop_reason)
                return False
            eligible, reason = self._leaf_compaction_candidate_status(
                messages,
                allow_partial_leaf=self._config.threshold_full_sweep_enabled,
                deadline=deadline,
            )
            if eligible:
                payback_allowed, payback_reason = self._compaction_payback_status(
                    messages,
                    observed_tokens=rough,
                )
                if payback_allowed:
                    return self._mark_preflight_compression_requested()
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = payback_reason
                logger.info("LCM preflight compression no-op: %s", payback_reason)
                return False
            if self._has_ignored_backlog_outside_fresh_tail(
                messages,
                deadline=deadline,
            ):
                return self._mark_preflight_compression_requested()
            if self._should_run_deferred_maintenance(
                messages,
                observed_tokens=rough,
                deadline=deadline,
            ):
                return self._mark_preflight_compression_requested()
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
            return False
        self._refresh_raw_backlog_debt(
            messages,
            observed_tokens=rough,
            deadline=deadline,
        )
        if self._should_run_deferred_maintenance(
            messages,
            observed_tokens=rough,
            deadline=deadline,
        ):
            return self._mark_preflight_compression_requested()
        return False

    def _replay_diff_requests_ingest_cleanup(
        self,
        original_messages: List[Dict[str, Any]],
        replay_messages: List[Dict[str, Any]],
    ) -> bool:
        if len(original_messages) != len(replay_messages):
            return True
        for original_msg, replay_msg in zip(original_messages, replay_messages):
            original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
            replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
            if original_text != replay_text:
                if replay_text.startswith("[Externalized LCM ingest payload:"):
                    return True
                if replay_text.startswith("[Externalized payload: kind=raw_payload;"):
                    return True
                if replay_text.startswith("[Externalized tool output:"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: assistant output quarantined;"):
                    return True
                if replay_text.startswith("[LCM active replay placeholder: message ignored;"):
                    return True
                if "[LCM sensitive redaction:" in replay_text:
                    return True
            if original_msg.get("content") != replay_msg.get("content") and _contains_sensitive_redaction(
                replay_msg.get("content")
            ):
                return True
            if original_msg.get("tool_calls") != replay_msg.get("tool_calls") and _contains_sensitive_redaction(
                replay_msg.get("tool_calls")
            ):
                return True
        return False

    def _has_ignored_backlog_outside_fresh_tail(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> bool:
        self._foreground_deadline_checkpoint(
            deadline,
            "ignored backlog inspection",
        )
        if not self._compiled_ignore_message_patterns or not messages:
            return False
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False
        previous_store_id_map = self._current_compress_store_ids_by_message_id
        self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(
            messages[leading_anchor_count:fresh_tail_start],
            deadline=deadline,
        )
        try:
            return any(
                self._matches_ignore_message_patterns(msg)
                or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                for msg in messages[leading_anchor_count:fresh_tail_start]
            )
        finally:
            self._current_compress_store_ids_by_message_id = previous_store_id_map

    def _leaf_compaction_candidate_status(
        self,
        messages: List[Dict[str, Any]],
        *,
        force_overflow: bool = False,
        allow_partial_leaf: bool = False,
        deadline: float | None = None,
    ) -> tuple[bool, str]:
        """Return whether a normal leaf compaction pass can actually run.

        The host asks ``should_compress_preflight`` before it emits user-visible
        compression status. A session can be over the global context threshold
        while all pressure sits in the protected fresh tail, or while the raw
        backlog outside that tail is still smaller than the configured leaf
        chunk. In that case ``compress()`` would immediately no-op, so preflight
        should not advertise a compaction attempt yet.
        """
        self._foreground_deadline_checkpoint(
            deadline,
            "preflight leaf inspection",
        )
        if not messages:
            return False, "empty message list"
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail"

        candidate_raw = messages[leading_anchor_count:fresh_tail_start]
        if not candidate_raw:
            return False, "no eligible raw backlog outside fresh tail"
        generated_placeholder_hashes = self._load_generated_ignored_placeholder_hashes()
        self._foreground_deadline_checkpoint(
            deadline,
            "preflight placeholder metadata",
        )
        if self._compiled_ignore_message_patterns or generated_placeholder_hashes:
            previous_store_id_map = self._current_compress_store_ids_by_message_id
            self._current_compress_store_ids_by_message_id = (
                self._get_store_id_map_for_messages(
                    candidate_raw,
                    deadline=deadline,
                )
            )
            try:
                filtered_candidate_raw: list[Dict[str, Any]] = []
                for index, msg in enumerate(candidate_raw):
                    if index % 32 == 0:
                        self._foreground_deadline_checkpoint(
                            deadline,
                            "preflight leaf inspection",
                        )
                    content_text = text_content_for_pattern_matching(msg.get("content")) or ""
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in generated_placeholder_hashes
                    )
                    if (
                        self._matches_ignore_message_patterns(msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(msg)
                        or self._is_ignored_active_replay_placeholder(msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        continue
                    filtered_candidate_raw.append(msg)
            finally:
                self._current_compress_store_ids_by_message_id = previous_store_id_map
            candidate_raw = filtered_candidate_raw
            if not candidate_raw:
                return False, "no eligible raw backlog outside fresh tail"

        if force_overflow:
            return True, "forced overflow recovery"

        raw_tokens_outside_tail = count_messages_tokens(candidate_raw)
        self._foreground_deadline_checkpoint(
            deadline,
            "preflight leaf token count",
        )
        if allow_partial_leaf:
            return True, "eligible partial threshold-sweep leaf"
        if self._config.dynamic_leaf_chunk_enabled:
            working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
        else:
            working_leaf_chunk_tokens = self._config.leaf_chunk_tokens
        if raw_tokens_outside_tail < working_leaf_chunk_tokens:
            return False, "raw backlog outside fresh tail is below leaf chunk threshold"
        return True, "eligible raw backlog outside fresh tail"

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_leaf_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    def _load_dependent_reply_records_with_foreground_deadline(
        self,
        *,
        deadline: float | None,
    ) -> list[dict[str, str]]:
        """Bound the legacy metadata read on automatic foreground paths."""
        if deadline is None:
            return self._load_generated_ignored_dependent_reply_records()

        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is None or remaining <= 0:
            raise ForegroundDeadlineExceeded("foreground compaction planning")
        store_lock = self._store._write_lock
        acquired = bool(store_lock.acquire(timeout=max(0.0, remaining)))
        if not acquired:
            raise ForegroundDeadlineExceeded(
                "foreground compaction planning",
                "message-store metadata lock wait exceeded the shared deadline",
            )
        try:
            self._foreground_deadline_checkpoint(
                deadline,
                "foreground compaction planning",
                "before dependent-reply metadata read",
            )
            records = self._load_generated_ignored_dependent_reply_records()
            self._foreground_deadline_checkpoint(
                deadline,
                "foreground compaction planning",
                "after dependent-reply metadata read",
            )
            return records
        finally:
            store_lock.release()

    def _record_foreground_deadline_exhaustion(
        self,
        messages: List[Dict[str, Any]],
        exc: ForegroundDeadlineExceeded,
        *,
        started_at: float,
        deadline: float,
    ) -> List[Dict[str, Any]]:
        """Fail safe without replacing or truncating the caller's live context."""
        reason = f"foreground deadline exhausted during {exc.stage}"
        self._last_compression_status = "noop"
        self._last_compression_noop_reason = reason
        self._last_foreground_deadline = {
            "status": "exhausted",
            "stage": exc.stage,
            "detail": exc.detail,
            "budget_seconds": max(0.0, float(deadline) - float(started_at)),
            "elapsed_seconds": max(0.0, self._foreground_now() - float(started_at)),
            "session_id": str(getattr(self, "_session_id", "") or ""),
            "conversation_id": str(getattr(self, "_conversation_id", "") or ""),
            "message_count": len(messages),
        }
        logger.warning("LCM %s", reason)

        # Keep an operator-visible durable breadcrumb when the background store
        # is immediately available. The non-blocking Python-lock probe and a
        # zero SQLite busy timeout ensure diagnostics cannot create a second
        # foreground stall after the main deadline has already expired.
        background = getattr(self, "_background_compaction", None)
        recorder = getattr(background, "record_diagnostic_event", None)
        write_lock = getattr(background, "_lock", None)
        acquired = False
        if (
            self._foreground_now() < float(deadline)
            and callable(recorder)
            and write_lock is not None
        ):
            try:
                acquired = bool(write_lock.acquire(blocking=False))
                if acquired:
                    recorder_kwargs: dict[str, Any] = {
                        "conversation_id": str(
                            getattr(self, "_conversation_id", "") or ""
                        ),
                        "session_id": str(
                            getattr(self, "_session_id", "") or ""
                        ),
                        "detail": f"stage={exc.stage}; detail={exc.detail}",
                        "deduplicate_seconds": 1.0,
                        "write_timeout_ms": 0,
                    }
                    if self._summary_callable_accepts_keyword(
                        recorder,
                        "lock_timeout_seconds",
                    ):
                        recorder_kwargs["lock_timeout_seconds"] = 0.0
                    if self._summary_callable_accepts_keyword(
                        recorder,
                        "write_deadline",
                    ):
                        recorder_kwargs["write_deadline"] = deadline
                    if self._summary_callable_accepts_keyword(
                        recorder,
                        "deadline_clock",
                    ):
                        recorder_kwargs["deadline_clock"] = self._foreground_now
                    recorder(
                        "foreground_deadline_exhausted",
                        **recorder_kwargs,
                    )
            except Exception:
                logger.debug(
                    "LCM could not persist foreground deadline diagnostics",
                    exc_info=True,
                )
            finally:
                if acquired:
                    write_lock.release()
        return messages

    def _persist_active_placeholder_metadata(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None,
    ) -> None:
        """Best-effort derived replay metadata within the foreground wall."""

        def persist() -> None:
            self._foreground_deadline_checkpoint(
                deadline,
                "active replay metadata",
                "before placeholder count derivation",
            )
            counts = self._generated_placeholder_digest_budget_for_active_replay(
                messages,
                deadline=deadline,
            )
            self._foreground_deadline_checkpoint(
                deadline,
                "active replay metadata",
                "before placeholder count persistence",
            )
            self._write_generated_ignored_placeholder_hash_counts(counts)
            self._foreground_deadline_checkpoint(
                deadline,
                "active replay metadata",
                "before placeholder ordinal derivation",
            )
            ordinals = (
                self._generated_placeholder_digest_ordinals_for_active_replay(
                    messages,
                    deadline=deadline,
                )
            )
            self._foreground_deadline_checkpoint(
                deadline,
                "active replay metadata",
                "before placeholder ordinal persistence",
            )
            self._write_generated_ignored_placeholder_hash_ordinals(ordinals)
            self._store.write_metadata_json(
                [self._active_ingest_manifest_key()],
                json.dumps(self._active_ingest_manifest(messages)),
                skip_unchanged=True,
            )

        if deadline is None:
            persist()
            return

        # Use the store's bounded transaction lane instead of the legacy
        # metadata writer. The transaction checks the same absolute wall before
        # commit and rolls back on expiry, so it cannot leave a late derived
        # write after the foreground caller returns. These values remain
        # best-effort: raw messages are already durable source truth.
        self._foreground_deadline_checkpoint(
            deadline,
            "active replay metadata",
            "before bounded metadata derivation",
        )
        store_ids_by_message_id = self._get_store_id_map_for_messages(
            messages,
            deadline=deadline,
        )
        durable_indices = {
            index
            for index, message in enumerate(messages)
            if id(message) in store_ids_by_message_id
        }
        updates, list_additions = (
            self._active_placeholder_metadata_transaction_updates(
                messages,
                durable_indices=durable_indices,
                deadline=deadline,
            )
        )
        updates.append((
            [self._active_ingest_manifest_key()],
            json.dumps(self._active_ingest_manifest(messages, deadline=deadline)),
            True,
        ))
        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is None or remaining <= 0:
            return
        try:
            self._store.write_metadata_json_batch_bounded(
                metadata_json_updates=updates,
                metadata_json_list_additions=list_additions,
                deadline=deadline,
                deadline_clock=self._foreground_now,
                lock_timeout_seconds=remaining,
                busy_timeout_ms=max(0, int(remaining * 1000)),
            )
        except TimeoutError:
            logger.debug(
                "LCM active replay metadata write exceeded its foreground budget"
            )
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            logger.debug(
                "LCM active replay metadata write deferred by SQLite contention"
            )

    def _distinct_durable_probe_rows(
        self,
        durable_probe: List[Dict[str, Any]],
        *,
        replay_limit: int,
        deadline: float | None = None,
    ) -> int:
        """Count archive rows after the frontier, ignoring stored duplicates.

        Older builds re-inserted already-stored messages, so the raw row count
        can exceed ``replay_limit`` for a session whose real archive is small.
        """
        if len(durable_probe) <= replay_limit:
            return len(durable_probe)
        rows = self._all_durable_messages_after(
            self._last_compacted_store_id,
            deadline=deadline,
        )
        aliases = self._durable_replay_duplicate_aliases(rows, deadline=deadline)
        return len(rows) - len(aliases)

    def _ingest_messages_with_foreground_deadline(
        self,
        messages: List[Dict[str, Any]],
        deadline: float | None,
    ) -> List[Dict[str, Any]]:
        """Pass the new deadline contract without breaking legacy test/host shims."""
        ingest = self._ingest_messages
        try:
            parameters = inspect.signature(ingest).parameters.values()
            accepts_deadline = any(
                parameter.name == "deadline"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_deadline = True
        if accepts_deadline:
            return ingest(messages, deadline=deadline)
        return ingest(messages)

    @staticmethod
    def _call_with_foreground_deadline(
        operation,
        *args,
        deadline: float | None,
        **kwargs,
    ):
        """Pass the deadline to aware helpers while preserving legacy shims."""
        if deadline is not None:
            try:
                parameters = inspect.signature(operation).parameters.values()
                accepts_deadline = any(
                    parameter.name == "deadline"
                    or parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                accepts_deadline = True
            if accepts_deadline:
                kwargs["deadline"] = deadline
        return operation(*args, **kwargs)

    @staticmethod
    def _summary_callable_accepts_keyword(operation, name: str) -> bool:
        """Keep injected test/host summary callables backward compatible."""
        try:
            parameters = inspect.signature(operation).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == name
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _call_with_hard_foreground_deadline(
        self,
        operation,
        *,
        deadline: float | None,
        timeout_error: BaseException,
        cancel_event: threading.Event | None = None,
    ):
        """Enforce an absolute wall around a potentially blocking call."""
        if cancel_event is not None and cancel_event.is_set():
            raise timeout_error
        if deadline is not None:
            remaining = self._foreground_deadline_remaining(deadline)
        else:
            remaining = None
        if deadline is not None and (remaining is None or remaining <= 0):
            if cancel_event is not None:
                cancel_event.set()
            raise timeout_error

        hard_call_slot = self._foreground_hard_call_slot
        if not hard_call_slot.acquire(blocking=False):
            if cancel_event is not None:
                cancel_event.set()
            raise timeout_error

        if deadline is None:
            try:
                return operation()
            finally:
                hard_call_slot.release()

        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is None or remaining <= 0:
            hard_call_slot.release()
            if cancel_event is not None:
                cancel_event.set()
            raise timeout_error

        completed = threading.Event()
        result_box: dict[str, Any] = {}

        def invoke() -> None:
            try:
                # Thread creation does not guarantee immediate scheduling.  The
                # foreground caller may already have timed out and cancelled
                # while this worker was waiting to run; never enter the late
                # operation in that case.
                if (
                    (cancel_event is not None and cancel_event.is_set())
                    or self._foreground_now() >= float(deadline)
                ):
                    result_box["cancelled"] = True
                    return
                result_box["value"] = operation()
            except BaseException as exc:
                result_box["error"] = exc
            finally:
                hard_call_slot.release()
                completed.set()

        worker = threading.Thread(
            target=invoke,
            name="lcm-foreground-deadline-call",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            hard_call_slot.release()
            raise
        remaining = self._foreground_deadline_remaining(deadline)
        try:
            finished = (
                remaining is not None
                and completed.wait(timeout=max(0.0, remaining))
            )
        except BaseException:
            if cancel_event is not None:
                cancel_event.set()
            raise
        if not finished:
            if cancel_event is not None:
                cancel_event.set()
            raise timeout_error
        if self._foreground_now() >= float(deadline):
            if cancel_event is not None:
                cancel_event.set()
            raise timeout_error
        if result_box.get("cancelled"):
            if cancel_event is not None:
                cancel_event.set()
            raise timeout_error
        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("value")

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: Optional[str] = None,
                 force: bool = False,
                 deadline: float | None = None) -> List[Dict[str, Any]]:
        """Run compaction and leave a terminal public status on every failure."""
        started_at = self._foreground_now()
        preflight_handoff = self._take_foreground_preflight_deadline_handoff(messages)
        if preflight_handoff is not None and not force:
            started_at, preflight_deadline = preflight_handoff
        else:
            preflight_deadline = None
        foreground_deadline = float(deadline) if deadline is not None else None
        if not force:
            local_deadline = self._foreground_now() + _AUTOMATIC_FOREGROUND_MAX_SECONDS
            foreground_deadline = (
                local_deadline
                if foreground_deadline is None
                else min(foreground_deadline, local_deadline)
            )
            if preflight_deadline is not None:
                foreground_deadline = min(
                    foreground_deadline,
                    preflight_deadline,
                )
        try:
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "compression entry",
            )
            return self._compress_impl(
                messages,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
                force=force,
                foreground_deadline=foreground_deadline,
            )
        except ForegroundDeadlineExceeded as exc:
            if foreground_deadline is None:
                raise
            return self._record_foreground_deadline_exhaustion(
                messages,
                exc,
                started_at=started_at,
                deadline=foreground_deadline,
            )
        except TimeoutError:
            if (
                foreground_deadline is None
                or self._foreground_now() < foreground_deadline
            ):
                self._last_compression_status = "error"
                self._last_compression_noop_reason = ""
                raise
            return self._record_foreground_deadline_exhaustion(
                messages,
                ForegroundDeadlineExceeded(
                    "summarization",
                    "provider call exceeded the foreground deadline",
                ),
                started_at=started_at,
                deadline=foreground_deadline,
            )
        except BaseException:
            self._last_compression_status = "error"
            self._last_compression_noop_reason = ""
            raise
        finally:
            release_permit = getattr(
                self,
                "_release_foreground_emergency_provider_permit",
                None,
            )
            if callable(release_permit):
                try:
                    parameters = inspect.signature(release_permit).parameters.values()
                    accepts_deadline = any(
                        parameter.name == "deadline"
                        or parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters
                    )
                except (TypeError, ValueError):
                    accepts_deadline = False
                if accepts_deadline:
                    release_permit(deadline=foreground_deadline)
                else:
                    release_permit()

    def _compress_impl(self, messages: List[Dict[str, Any]],
                       current_tokens: int = None,
                       focus_topic: Optional[str] = None,
                       force: bool = False,
                       foreground_deadline: float | None = None) -> List[Dict[str, Any]]:
        """Main compaction entry point.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail
        """
        if not messages:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "empty message list"
            return messages

        self._last_compression_status = "running"
        self._last_compression_noop_reason = ""
        _compress_started = time.perf_counter()

        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "setup",
        )
        self._maybe_reclassify_late_auxiliary_before_compaction_write(
            deadline=foreground_deadline,
        )
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "session classification",
        )
        try:
            bypasses_lcm = self._call_with_foreground_deadline(
                self._bypasses_lcm_context_management,
                deadline=foreground_deadline,
            )
        except TimeoutError as exc:
            if foreground_deadline is None:
                raise
            raise ForegroundDeadlineExceeded(
                "session classification",
                "auxiliary session lookup exceeded the foreground deadline",
            ) from exc
        if bypasses_lcm:
            bypass_current_tokens = current_tokens
            if bypass_current_tokens is None or bypass_current_tokens <= 0:
                auxiliary_session_id = self._call_with_foreground_deadline(
                    self._thread_context_session_id,
                    deadline=foreground_deadline,
                )
                if auxiliary_session_id:
                    auxiliary_prompt_tokens = self._call_with_foreground_deadline(
                        self._current_auxiliary_prompt_tokens,
                        auxiliary_session_id,
                        deadline=foreground_deadline,
                    )
                    if auxiliary_prompt_tokens > 0:
                        bypass_current_tokens = auxiliary_prompt_tokens
            bypassed = self._call_with_foreground_deadline(
                self._compress_lcm_bypassed_session,
                messages,
                current_tokens=bypass_current_tokens,
                focus_topic=focus_topic,
                force=force,
                deadline=foreground_deadline,
            )
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "bypass compression",
            )
            return bypassed

        observed_prompt_tokens = current_tokens if current_tokens is not None else None
        background_feature_enabled = self._background_compaction_enabled()
        background_mode = background_feature_enabled and not force
        background_message_tokens = count_messages_tokens(messages)
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "token counting",
        )
        background_observed_tokens = max(
            background_message_tokens,
            int(
                observed_prompt_tokens
                if observed_prompt_tokens is not None
                else self.last_prompt_tokens or 0
            ),
        )
        background_prompt_overhead = max(
            0,
            background_observed_tokens - background_message_tokens,
        )
        if background_mode:
            observed_prompt_tokens = background_observed_tokens
        host_force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        emergency_trigger = self._background_emergency_trigger_tokens()
        background_emergency = bool(
            background_mode
            and (
                host_force_overflow
                or (
                    emergency_trigger > 0
                    and background_observed_tokens >= emergency_trigger
                )
            )
        )
        force_overflow = host_force_overflow or background_emergency
        # NOTE: deliberately do NOT clear the spend guard on force_overflow.
        # force_overflow is automatic (set every turn the prompt exceeds the
        # assembly cap), which is exactly the sustained-over-cap state a runaway
        # compaction loop produces - clearing it per turn would defeat the guard
        # in the case it exists for. A tripped guard still converges the
        # emergency via deterministic L3 truncation (no LLM spend).
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        # Step 1: Ingest new messages into the immutable store. Work from a
        # replay-safe view so quarantined assistant loops do not enter summaries
        # or provider context after the durable row has been written.
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "ingest",
        )
        working_messages = self._ingest_messages_with_foreground_deadline(
            messages,
            foreground_deadline,
        )
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "ingest",
        )
        ingest_cleanup_changed_active_context = working_messages != messages
        # A failed archive reconstruction must not be a prerequisite for
        # emergency admission recovery. All live messages have just been
        # durably ingested. Build the bounded view from those messages and
        # existing DAG summaries, preserving raw history for LCM retrieval.
        # Normal prepared-publication behavior remains unchanged.
        if background_mode and background_emergency:
            prior_deadline = getattr(self, "_last_foreground_deadline", {}) or {}
            replay_limit = max(1000, len(working_messages) * 4)
            durable_probe = self._store.get_session_messages_after(
                self._session_id,
                self._last_compacted_store_id,
                limit=replay_limit + 1,
                conversation_id=self._conversation_id,
                lock_timeout_seconds=self._foreground_deadline_remaining(foreground_deadline),
                deadline=foreground_deadline,
                deadline_clock=self._foreground_now,
            )
            same_session_deadline = (
                prior_deadline.get("status") == "exhausted"
                and prior_deadline.get("session_id") == self._session_id
                and prior_deadline.get("conversation_id") == self._conversation_id
            )
            safe_input = int(getattr(self, "_admission_safe_input_tokens", 0) or 0)
            hard_admission_pressure = safe_input > 0 and background_observed_tokens > safe_input
            archive_rows = self._distinct_durable_probe_rows(
                durable_probe,
                replay_limit=replay_limit,
                deadline=foreground_deadline,
            )
            # The active-view recovery only helps when it can bound the view.
            # Without an assembly cap it re-emits the live transcript unchanged
            # and skips summarization, so the host refuses it as non-shrinking.
            if recovery_assembly_cap is not None and (
                hard_admission_pressure
                or same_session_deadline
                or archive_rows > replay_limit
            ):
                leading_count = self._leading_anchor_count(working_messages)
                compressed = self._assemble_overflow_recovery_context(
                    working_messages[0] if leading_count else None,
                    working_messages[leading_count:],
                    assembly_cap_override=recovery_assembly_cap,
                    deadline=foreground_deadline,
                )
                result = self._finalize_forced_overflow_result(
                    working_messages,
                    compressed,
                    assembly_cap_override=recovery_assembly_cap,
                    ingest_cleanup_changed_active_context=ingest_cleanup_changed_active_context,
                    deadline=foreground_deadline,
                )
                self._persist_active_placeholder_metadata(result, deadline=foreground_deadline)
                logger.warning(
                    "LCM emergency active-view recovery: live=%d archive_probe=%d cap=%s status=%s; raw history retained",
                    len(working_messages), archive_rows, recovery_assembly_cap,
                    self._last_compression_status,
                )
                return result
        if background_mode:
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "frontier adoption",
            )
            try:
                adopted_messages = self._call_with_foreground_deadline(
                    self._adopt_committed_background_frontier,
                    working_messages,
                    deadline=foreground_deadline,
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "frontier adoption",
                )
                if adopted_messages is None:
                    # A pre-background canonical node can legitimately exist
                    # above a lagging lifecycle frontier after an old-process
                    # crash.  Reconcile at most one bounded prefix per
                    # foreground boundary, then reuse the normal durable
                    # frontier-adoption path to rebuild the active context.
                    self._call_with_foreground_deadline(
                        self._maybe_adopt_existing_canonical_prefix,
                        deadline=foreground_deadline,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "canonical prefix adoption",
                    )
                    adopted_messages = self._call_with_foreground_deadline(
                        self._adopt_committed_background_frontier,
                        working_messages,
                        deadline=foreground_deadline,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "frontier adoption",
                    )
            except ForegroundDeadlineExceeded:
                raise
            except Exception:
                logger.warning(
                    "LCM could not adopt a committed background frontier",
                    exc_info=True,
                )
                adopted_messages = None
            if adopted_messages is not None:
                working_messages = adopted_messages
                ingest_cleanup_changed_active_context = True
                # A sibling publication can remove most of the raw prefix that
                # made this turn look critical.  Preserve only the prompt
                # overhead that publication cannot change, then recalculate the
                # emergency gate from the newly active context before spending
                # on the synchronous fallback.
                background_observed_tokens = (
                    count_messages_tokens(working_messages)
                    + background_prompt_overhead
                )
                observed_prompt_tokens = background_observed_tokens
                host_force_overflow = self._should_force_overflow_recovery(
                    observed_tokens=background_observed_tokens,
                    messages=working_messages,
                )
                background_emergency = bool(
                    host_force_overflow
                    or (
                        emergency_trigger > 0
                        and background_observed_tokens >= emergency_trigger
                    )
                )
                force_overflow = host_force_overflow or background_emergency
                recovery_assembly_cap = (
                    self._overflow_recovery_assembly_cap(
                        observed_tokens=background_observed_tokens,
                        messages=working_messages,
                    )
                    if force_overflow
                    else None
                )
        cleanup_only_due_to_boundary_cooldown = bool(
            self._preflight_cleanup_only_due_to_boundary_cooldown
            and not force_overflow
        )
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        if cleanup_only_due_to_boundary_cooldown:
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "active context cleanup",
            )
            sanitized_messages = self._call_foreground_projection(
                self._sanitize_active_context_messages,
                working_messages,
                insert_missing_tool_stubs=False,
                deadline=foreground_deadline,
                stage="active context cleanup",
            )
            self._refresh_raw_backlog_debt(
                sanitized_messages,
                observed_tokens=observed_prompt_tokens,
                deadline=foreground_deadline,
            )
            self._align_ingest_cursor_to_active_context(
                sanitized_messages,
                deadline=foreground_deadline,
            )
            self._last_compression_status = "sanitized"
            self._last_compression_noop_reason = ""
            self._persist_active_placeholder_metadata(
                sanitized_messages,
                deadline=foreground_deadline,
            )
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "active context cleanup",
            )
            return sanitized_messages
        if background_mode:
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "background enqueue",
            )
            try:
                self._call_with_foreground_deadline(
                    self._maybe_enqueue_background_compaction,
                    working_messages,
                    background_observed_tokens,
                    allow_canonical_prefix_adoption=False,
                    deadline=foreground_deadline,
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "background enqueue",
                )
            except ForegroundDeadlineExceeded:
                raise
            except Exception:
                logger.warning(
                    "LCM could not enqueue background compaction during compress",
                    exc_info=True,
                )

            publication_trigger = self._background_publication_trigger_tokens()
            publication = None
            publishable = False
            if publication_trigger > 0 and background_observed_tokens >= publication_trigger:
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "prepared publication inspection",
                )
                try:
                    publishable = self._call_with_foreground_deadline(
                        self._has_publishable_background_job,
                        deadline=foreground_deadline,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "prepared publication inspection",
                    )
                except ForegroundDeadlineExceeded:
                    raise
                except Exception:
                    logger.warning(
                        "LCM could not inspect prepared background work during compress",
                        exc_info=True,
                    )
            if (
                publication_trigger > 0
                and background_observed_tokens >= publication_trigger
                and publishable
            ):
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "prepared publication",
                )
                try:
                    publication = self._call_with_foreground_deadline(
                        self._publish_prepared_background_compaction,
                        working_messages,
                        current_tokens=background_observed_tokens,
                        deadline=foreground_deadline,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "prepared publication",
                    )
                except ForegroundDeadlineExceeded:
                    raise
                except Exception:
                    # A failed compare-and-swap or short SQLite contention must
                    # never turn ordinary publication pressure into a slow
                    # synchronous model request.  Emergency pressure below is
                    # the only automatic route into the legacy pipeline.
                    logger.warning(
                        "LCM prepared background publication failed",
                        exc_info=True,
                    )

            published_messages = None
            if isinstance(publication, list):
                published_messages = publication
            elif isinstance(publication, Mapping):
                if publication.get("promoted"):
                    published_messages = publication.get("messages")
            elif publication is not None and getattr(publication, "promoted", False):
                published_messages = getattr(publication, "messages", None)
            if published_messages is not None:
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "published context rebuild",
                )
                published_messages = self._call_foreground_projection(
                    self._sanitize_active_context_messages,
                    list(published_messages),
                    deadline=foreground_deadline,
                    stage="published context cleanup",
                )
                self._align_ingest_cursor_to_active_context(
                    published_messages,
                    deadline=foreground_deadline,
                )
                self._last_compression_status = "compacted"
                self._last_compression_noop_reason = ""
                self._persist_active_placeholder_metadata(
                    published_messages,
                    deadline=foreground_deadline,
                )
                published_tokens = count_messages_tokens(published_messages)
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "published context rebuild",
                )
                published_pressure_tokens = (
                    published_tokens + background_prompt_overhead
                )
                if (
                    not background_emergency
                    or emergency_trigger <= 0
                    or published_pressure_tokens < emergency_trigger
                ):
                    return published_messages
                # A small prepared batch won the CAS but did not clear emergency
                # pressure. Continue into the one-pass shared-deadline fallback.
                working_messages = published_messages
                try:
                    self._call_with_foreground_deadline(
                        self._maybe_enqueue_background_compaction,
                        working_messages,
                        published_pressure_tokens,
                        allow_canonical_prefix_adoption=False,
                        deadline=foreground_deadline,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "background enqueue",
                    )
                except ForegroundDeadlineExceeded:
                    raise
                except Exception:
                    logger.warning(
                        "LCM could not enqueue follow-on background compaction "
                        "after publication",
                        exc_info=True,
                    )

            if background_emergency:
                partial_repair_pending = False
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "partial repair inspection",
                )
                try:
                    remaining = self._foreground_deadline_remaining(
                        foreground_deadline
                    )
                    lifecycle = self._lifecycle.get_by_conversation(
                        self._conversation_id,
                        lock_timeout_seconds=remaining,
                        deadline=foreground_deadline,
                        deadline_clock=self._foreground_now,
                    )
                    if (
                        lifecycle is not None
                        and lifecycle.current_session_id == self._session_id
                    ):
                        partial_repair_pending = (
                            self._background_compaction.has_active_partial_atomic_repair(
                                conversation_id=self._conversation_id,
                                session_id=self._session_id,
                                expected_frontier_store_id=int(
                                    lifecycle.current_frontier_store_id or 0
                                ),
                                lock_timeout_seconds=(
                                    self._foreground_deadline_remaining(
                                        foreground_deadline
                                    )
                                ),
                                deadline=foreground_deadline,
                                deadline_clock=self._foreground_now,
                            )
                        )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "partial repair inspection",
                    )
                except ForegroundDeadlineExceeded:
                    raise
                except TimeoutError as exc:
                    raise ForegroundDeadlineExceeded(
                        "partial repair inspection",
                        "store lock wait exceeded the shared deadline",
                    ) from exc
                except Exception:
                    logger.warning(
                        "LCM could not inspect pending partial atomic repair",
                        exc_info=True,
                    )
                if partial_repair_pending:
                    leading_anchor_count = self._leading_anchor_count(
                        working_messages
                    )
                    compressed = self._call_with_foreground_deadline(
                        self._assemble_overflow_recovery_context,
                        (
                            working_messages[0]
                            if leading_anchor_count
                            else None
                        ),
                        working_messages[leading_anchor_count:],
                        assembly_cap_override=recovery_assembly_cap,
                        deadline=foreground_deadline,
                    )
                    result = self._finalize_forced_overflow_result(
                        working_messages,
                        compressed,
                        assembly_cap_override=recovery_assembly_cap,
                        ingest_cleanup_changed_active_context=(
                            ingest_cleanup_changed_active_context
                        ),
                        deadline=foreground_deadline,
                    )
                    self._last_compression_noop_reason = (
                        "partial atomic repair pending"
                    )
                    return result

            if not background_emergency:
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "active context cleanup",
                )
                sanitized_messages = self._call_foreground_projection(
                    self._sanitize_active_context_messages,
                    working_messages,
                    insert_missing_tool_stubs=False,
                    deadline=foreground_deadline,
                    stage="active context cleanup",
                )
                self._refresh_raw_backlog_debt(
                    sanitized_messages,
                    observed_tokens=background_observed_tokens,
                    deadline=foreground_deadline,
                )
                if sanitized_messages != working_messages or ingest_cleanup_changed_active_context:
                    self._align_ingest_cursor_to_active_context(
                        sanitized_messages,
                        deadline=foreground_deadline,
                    )
                    self._last_compression_status = "sanitized"
                    self._last_compression_noop_reason = ""
                else:
                    self._last_compression_status = "noop"
                    self._last_compression_noop_reason = (
                        "prepared background summary unavailable below emergency trigger"
                    )
                self._persist_active_placeholder_metadata(
                    sanitized_messages,
                    deadline=foreground_deadline,
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "active context cleanup",
                )
                return sanitized_messages
        anchor_source_messages = list(working_messages)
        pressure_messages = messages if len(messages) == len(working_messages) else working_messages
        leaf_compacted_this_turn = False
        dropped_replayed_scaffold_messages = False
        leaf_passes = 0
        estimated_active_tokens = (
            observed_prompt_tokens
            if observed_prompt_tokens is not None and observed_prompt_tokens > 0
            else count_messages_tokens(messages)
        )
        threshold_full_sweep_active = bool(
            self._config.threshold_full_sweep_enabled
            and not force_overflow
            and self.threshold_tokens > 0
            and estimated_active_tokens >= self.threshold_tokens
        )
        automatic_foreground = not force
        sweep_pass_budget = (
            _AUTOMATIC_FOREGROUND_MAX_PASSES
            if automatic_foreground
            else _MANUAL_FULL_SWEEP_MAX_PASSES
        )
        sweep_seconds_budget = (
            _AUTOMATIC_FOREGROUND_MAX_SECONDS
            if automatic_foreground
            else _MANUAL_FULL_SWEEP_MAX_SECONDS
        )
        sweep_deadline = (
            foreground_deadline
            if foreground_deadline is not None
            else time.monotonic() + sweep_seconds_budget
        )
        configured_sweep_target = int(self._config.summary_prefix_target_tokens)
        sweep_target_tokens = max(
            1,
            configured_sweep_target
            if configured_sweep_target > 0
            else int(self._config.leaf_chunk_tokens),
        )
        sweep_summary_prefix_before = (
            self._summary_frontier_tokens(deadline=foreground_deadline)
            if threshold_full_sweep_active
            else 0
        )
        if threshold_full_sweep_active:
            self._last_threshold_full_sweep = {
                "status": "running",
                "leaf_passes": 0,
                "condensation_passes": 0,
                "total_passes": 0,
                "duration_ms": 0.0,
                "tokens_before": estimated_active_tokens,
                "tokens_after": estimated_active_tokens,
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": sweep_summary_prefix_before,
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": "",
                "budget_exhausted": False,
            }
        critical_budget_pressure = self._critical_budget_pressure_reached(
            observed_tokens=observed_prompt_tokens,
            messages=working_messages,
        )
        deferred_maintenance_active = (
            not background_mode
            and not force_overflow
            and not threshold_full_sweep_active
            and self._should_run_deferred_maintenance(
                working_messages,
                observed_tokens=observed_prompt_tokens,
                deadline=foreground_deadline,
            )
        )
        if deferred_maintenance_active:
            remaining = self._foreground_deadline_remaining(
                foreground_deadline
            )
            try:
                self._lifecycle.record_maintenance_attempt(
                    self._conversation_id,
                    lock_timeout_seconds=remaining,
                    deadline=foreground_deadline,
                    deadline_clock=self._foreground_now,
                    busy_timeout_ms=(
                        max(0, int(remaining * 1000))
                        if remaining is not None
                        else None
                    ),
                )
            except TimeoutError as exc:
                raise ForegroundDeadlineExceeded(
                    "deferred maintenance persistence",
                    "lifecycle lock wait exceeded the shared deadline",
                ) from exc
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_locked_error(exc):
                    raise
                raise ForegroundDeadlineExceeded(
                    "deferred maintenance persistence",
                    "SQLite contention prevented a bounded lifecycle write",
                ) from exc
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "deferred maintenance persistence",
            )
        base_max_leaf_passes = (
            (2 if automatic_foreground else 4)
            if self._config.dynamic_leaf_chunk_enabled
            else 1
        )
        max_leaf_passes = base_max_leaf_passes
        if background_emergency and automatic_foreground:
            max_leaf_passes = _AUTOMATIC_FOREGROUND_MAX_PASSES
        if threshold_full_sweep_active:
            max_leaf_passes = sweep_pass_budget
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)
            if automatic_foreground:
                max_leaf_passes = min(
                    max_leaf_passes,
                    _AUTOMATIC_FOREGROUND_MAX_PASSES,
                )

        explicit_focus_topic = focus_topic is not None

        noop_reason = "no eligible raw backlog outside fresh tail"
        sweep_stop_reason = ""
        sweep_raw_drained = False
        dependent_reply_message_ids: set[int] = set()
        preexisting_dependent_reply_records = (
            self._load_dependent_reply_records_with_foreground_deadline(
                deadline=foreground_deadline,
            )
        )
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "foreground compaction planning",
        )

        while leaf_passes < max_leaf_passes:
            if self._foreground_now() >= sweep_deadline:
                if foreground_deadline is not None:
                    raise ForegroundDeadlineExceeded("foreground compaction sweep")
                sweep_stop_reason = "time_budget_exhausted"
                break
            fresh_tail_start = self._fresh_tail_start(pressure_messages)

            # Keep only a real system prompt anchored. Gateway sessions may
            # pass only conversation messages, so index 0 can be an old user
            # turn; that must remain eligible for compaction instead of being
            # replayed forever as fresh-looking intent.
            leading_anchor_count = self._leading_anchor_count(working_messages)
            if fresh_tail_start <= leading_anchor_count:
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            candidate_start = leading_anchor_count
            while (
                candidate_start < fresh_tail_start
                and self._is_replayed_context_scaffold_message(working_messages[candidate_start])
            ):
                candidate_start += 1
            if candidate_start > leading_anchor_count:
                dropped_replayed_scaffold_messages = True
                working_messages = working_messages[:leading_anchor_count] + working_messages[candidate_start:]
                pressure_messages = pressure_messages[:leading_anchor_count] + pressure_messages[candidate_start:]
                candidate_start = leading_anchor_count
                fresh_tail_start = self._fresh_tail_start(pressure_messages)
                if fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            if candidate_start < fresh_tail_start:
                self._current_compress_store_ids_by_message_id = (
                    self._call_with_foreground_deadline(
                        self._get_store_id_map_for_messages,
                        working_messages[leading_anchor_count:],
                        deadline=foreground_deadline,
                    )
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "source mapping",
                )
                compactable_pairs = list(
                    zip(
                        working_messages[candidate_start:fresh_tail_start],
                        pressure_messages[candidate_start:fresh_tail_start],
                    )
                )
                kept_working: list[Dict[str, Any]] = []
                kept_pressure: list[Dict[str, Any]] = []
                dropped_ignored_backlog = False
                drop_dependent_reply = False
                for pair_index, (working_msg, pressure_msg) in enumerate(compactable_pairs):
                    if pair_index % 32 == 0:
                        self._foreground_deadline_checkpoint(
                            foreground_deadline,
                            "foreground compaction filtering",
                        )
                    role = str(working_msg.get("role") or "")
                    content_text = text_content_for_pattern_matching(working_msg.get("content")) or ""
                    generated_dependent_reply = self._is_generated_ignored_dependent_reply(
                        working_msg,
                        content_text,
                    )
                    volatile_digest = self._active_replay_placeholder_digest(content_text)
                    generated_volatile_placeholder = (
                        self._is_volatile_ignored_quarantine_placeholder(working_msg, content_text)
                        and volatile_digest is not None
                        and volatile_digest in self._load_generated_ignored_placeholder_hashes()
                    )
                    if (
                        self._matches_ignore_message_patterns(working_msg)
                        or self._matches_ignore_message_patterns(pressure_msg)
                        or self._mapped_stored_row_matches_ignore_message_patterns(working_msg)
                        or self._is_ignored_active_replay_placeholder(working_msg, content_text)
                        or generated_volatile_placeholder
                    ):
                        dropped_ignored_backlog = True
                        if role in {"user", "system", "tool", "assistant"}:
                            drop_dependent_reply = True
                        continue
                    if generated_dependent_reply:
                        dependent_reply_message_ids.add(id(working_msg))
                        if role in {"assistant", "tool"}:
                            drop_dependent_reply = True
                    if drop_dependent_reply and role in {"assistant", "tool"}:
                        dependent_reply_message_ids.add(id(working_msg))
                        self._remember_generated_ignored_dependent_reply(working_msg, content_text)
                    if role in {"user", "system"}:
                        drop_dependent_reply = False
                    kept_working.append(working_msg)
                    kept_pressure.append(pressure_msg)
                drop_dependent_reply_into_tail = drop_dependent_reply
                if dropped_ignored_backlog:
                    dropped_replayed_scaffold_messages = True
                    working_messages = (
                        working_messages[:candidate_start]
                        + kept_working
                        + working_messages[fresh_tail_start:]
                    )
                    pressure_messages = (
                        pressure_messages[:candidate_start]
                        + kept_pressure
                        + pressure_messages[fresh_tail_start:]
                    )
                    fresh_tail_start = self._fresh_tail_start(pressure_messages)
                if drop_dependent_reply_into_tail:
                    tail_scan_start = max(fresh_tail_start, leading_anchor_count)
                    pending_tail_dependents: list[tuple[Dict[str, Any], str]] = []
                    saw_tail_boundary = False
                    for tail_msg in working_messages[tail_scan_start:]:
                        if not isinstance(tail_msg, dict):
                            continue
                        tail_role = str(tail_msg.get("role") or "")
                        if tail_role in {"user", "system"}:
                            saw_tail_boundary = True
                            break
                        if tail_role in {"assistant", "tool"}:
                            tail_text = text_content_for_pattern_matching(tail_msg.get("content")) or ""
                            self._remember_generated_ignored_dependent_reply(tail_msg, tail_text)
                            pending_tail_dependents.append((tail_msg, tail_text))
                    if saw_tail_boundary or leading_anchor_count > 0 or kept_working:
                        for tail_msg, _tail_text in pending_tail_dependents:
                            dependent_reply_message_ids.add(id(tail_msg))
                if dropped_ignored_backlog and fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            # Auto-derive focus topic from the post-filter compaction view when
            # not explicitly provided.  The derived focus is summarizer-visible,
            # so it must follow the same ignored-message filtering as the leaf
            # chunk itself.
            if not explicit_focus_topic:
                focus_topic = self._derive_auto_focus_topic(working_messages)

            candidate_raw = working_messages[leading_anchor_count:fresh_tail_start]
            if not candidate_raw:
                noop_reason = "no eligible raw backlog outside fresh tail"
                if threshold_full_sweep_active:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                break

            pressure_candidate_raw = pressure_messages[leading_anchor_count:fresh_tail_start]
            raw_tokens_outside_tail = count_messages_tokens(pressure_candidate_raw)
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "foreground token counting",
            )
            if threshold_full_sweep_active:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(
                    raw_tokens_outside_tail
                )
                to_compact = self._select_oldest_leaf_chunk(
                    candidate_raw,
                    working_leaf_chunk_tokens,
                )
            elif self._config.dynamic_leaf_chunk_enabled:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
                if raw_tokens_outside_tail < working_leaf_chunk_tokens and not force_overflow:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                if force_overflow:
                    to_compact = candidate_raw
                else:
                    to_compact = self._select_oldest_leaf_chunk(candidate_raw, working_leaf_chunk_tokens)
            else:
                if raw_tokens_outside_tail < self._config.leaf_chunk_tokens and not force_overflow:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                to_compact = candidate_raw

            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            selected_raw_chunk = to_compact
            foreground_source_snapshot = None
            if background_feature_enabled:
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "foreground source snapshot",
                )
                prospective_store_ids = self._call_with_foreground_deadline(
                    self._get_store_ids_for_messages,
                    selected_raw_chunk,
                    deadline=foreground_deadline,
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "foreground source snapshot",
                )
                if not prospective_store_ids:
                    noop_reason = "foreground durable source interval is unavailable"
                    break
                try:
                    foreground_source_snapshot = (
                        self._background_compaction.capture_foreground_source_snapshot(
                            conversation_id=self._conversation_id,
                            session_id=self._session_id,
                            expected_frontier_store_id=self._last_compacted_store_id,
                            captured_watermark_store_id=max(prospective_store_ids),
                            expected_source_store_ids=prospective_store_ids,
                            lock_timeout_seconds=(
                                self._foreground_deadline_remaining(
                                    foreground_deadline
                                )
                            ),
                            deadline=foreground_deadline,
                            deadline_clock=self._foreground_now,
                        )
                    )
                except TimeoutError as exc:
                    raise ForegroundDeadlineExceeded(
                        "foreground source snapshot",
                        "source snapshot lock wait exceeded the shared deadline",
                    ) from exc
                except sqlite3.OperationalError as exc:
                    if not _is_sqlite_locked_error(exc):
                        raise
                    raise ForegroundDeadlineExceeded(
                        "foreground source snapshot",
                        "SQLite contention prevented a bounded source snapshot",
                    ) from exc
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "foreground source snapshot",
                )
                if foreground_source_snapshot is None:
                    # A sibling ingest or mutation changed the durable interval
                    # represented by the caller's active messages. Never ask a
                    # provider to summarize one view and retire another.
                    noop_reason = "foreground durable source interval changed"
                    break
                # Bind the caller view to the durable snapshot. If any source
                # row changed between the first active-to-store mapping and
                # the snapshot read, the same active messages no longer map to
                # the same IDs and provider I/O must not begin.
                if self._call_with_foreground_deadline(
                    self._get_store_ids_for_messages,
                    selected_raw_chunk,
                    deadline=foreground_deadline,
                ) != prospective_store_ids:
                    noop_reason = "foreground active source snapshot changed"
                    break
            foreground_session_fence_generation: int | None = None
            if background_feature_enabled:
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "publication fence",
                )
                foreground_session_fence_generation = (
                    self._reconcile_background_publication_fence(
                        deadline=foreground_deadline,
                    )
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "publication fence",
                )
                if foreground_session_fence_generation is None:
                    noop_reason = "background publication session is reset or rebound"
                    break
            summary_input_chunk = [
                message for message in selected_raw_chunk if id(message) not in dependent_reply_message_ids
            ]
            if not summary_input_chunk:
                compacted_chunk = selected_raw_chunk
                source_tokens = count_messages_tokens(selected_raw_chunk)
                summary_text = (
                    "Filtered replies derived from ignored messages.\n"
                    "[Expand for details: ignored-dependent reply]"
                )
                _level = 0
                _rescue_attempts = 0
            else:
                # Pre-compaction extraction: best-effort, never blocks compaction.
                # Use the same dependency-filtered view as summarization so ignored
                # turns cannot leak through derived assistant/tool replies.
                if self._config.extraction_enabled:
                    self._run_pre_compaction_extraction(
                        summary_input_chunk,
                        deadline=sweep_deadline,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "pre-compaction extraction",
                    )
                if bool(
                    getattr(
                        self._config,
                        "assertion_extraction_enabled",
                        False,
                    )
                ):
                    self._schedule_pre_compaction_assertions(
                        summary_input_chunk,
                        deadline=foreground_deadline,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "assertion scheduling",
                    )

                try:
                    summary_kwargs: dict[str, Any] = {
                        "focus_topic": focus_topic,
                        "deadline": sweep_deadline,
                    }
                    if background_emergency:
                        summary_kwargs["require_durable_provider_permit"] = True
                        summary_kwargs["expected_session_fence_generation"] = (
                            foreground_session_fence_generation
                        )
                    (
                        compacted_chunk,
                        source_tokens,
                        summary_text,
                        _level,
                        _rescue_attempts,
                    ) = self._summarize_leaf_chunk_with_rescue(
                        summary_input_chunk,
                        **summary_kwargs,
                    )
                    self._foreground_deadline_checkpoint(
                        foreground_deadline,
                        "summarization",
                    )
                except ForegroundProviderPermitUnavailable:
                    # Provider capacity is deliberately non-blocking in the
                    # foreground.  The forced-overflow branch below converges
                    # with the existing deterministic, no-provider fallback.
                    noop_reason = "foreground emergency provider permit unavailable"
                    break
                except Exception as exc:
                    if threshold_full_sweep_active and leaf_compacted_this_turn:
                        sweep_stop_reason = "leaf_summary_error"
                        logger.warning(
                            "LCM threshold full sweep stopped after %d persisted leaf pass(es): %s",
                            leaf_passes,
                            exc,
                        )
                        break
                    raise
            compacted_summary_ids = {id(message) for message in compacted_chunk}
            compacted_positions = [
                idx for idx, message in enumerate(selected_raw_chunk) if id(message) in compacted_summary_ids
            ]
            last_compacted_raw_pos = max(compacted_positions) if compacted_positions else len(compacted_chunk) - 1
            last_consumed_raw_pos = last_compacted_raw_pos
            while (
                last_consumed_raw_pos + 1 < len(selected_raw_chunk)
                and id(selected_raw_chunk[last_consumed_raw_pos + 1]) in dependent_reply_message_ids
            ):
                last_consumed_raw_pos += 1
            source_lookup_chunk = selected_raw_chunk[: last_consumed_raw_pos + 1]
            selected_raw_len = len(source_lookup_chunk)
            remaining_messages = working_messages[leading_anchor_count + selected_raw_len:]
            source_tokens = count_messages_tokens(source_lookup_chunk)
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "lineage mapping",
            )

            source_lineage_chunk = [
                message for message in source_lookup_chunk if id(message) not in dependent_reply_message_ids
            ]
            summary_source_store_ids = self._call_with_foreground_deadline(
                self._get_store_ids_for_messages,
                source_lineage_chunk,
                deadline=foreground_deadline,
            )
            summary_source_store_ids = sorted(dict.fromkeys(summary_source_store_ids))
            consumed_store_ids = self._call_with_foreground_deadline(
                self._get_store_ids_for_messages,
                source_lookup_chunk,
                deadline=foreground_deadline,
            )
            consumed_store_ids = sorted(dict.fromkeys(consumed_store_ids))
            if background_feature_enabled:
                snapshot_ids = [
                    int(message.get("store_id") or 0)
                    for message in (foreground_source_snapshot or ())
                ]
                expected_consumed_ids = snapshot_ids[: len(source_lookup_chunk)]
                if (
                    len(consumed_store_ids) != len(source_lookup_chunk)
                    or consumed_store_ids != expected_consumed_ids
                ):
                    # Never remove N caller messages after publishing lineage
                    # for a shorter mapped prefix. A concurrent source change
                    # is a bounded no-op; durable raw remains authoritative.
                    noop_reason = "foreground consumed source mapping changed"
                    break
            # Foreground emergency publication advances the durable frontier
            # across every mapped raw row in ``source_lookup_chunk``.  Its
            # canonical DAG node must carry that exact lineage and matching
            # time bounds even when provider policy filtered a dependent row.
            # Retain the legacy subset in disabled mode because the synchronous
            # tool-result GC below expects source IDs aligned to compacted input.
            source_store_ids = (
                consumed_store_ids
                if background_feature_enabled
                else summary_source_store_ids
            )
            foreground_source_identity_hash = ""
            if background_feature_enabled:
                foreground_source_identity_hash = str(
                    self._background_compaction.foreground_source_identity_hash(
                        foreground_source_snapshot or (),
                        consumed_store_ids,
                    )
                    or ""
                )
                if not foreground_source_identity_hash:
                    noop_reason = "foreground durable source identity is unavailable"
                    break
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "lineage mapping",
            )
            try:
                remaining = self._foreground_deadline_remaining(
                    foreground_deadline
                )
                earliest_at, latest_at = self._store.get_time_bounds(
                    source_store_ids,
                    deadline=foreground_deadline,
                    deadline_clock=self._foreground_now,
                    lock_timeout_seconds=remaining,
                    busy_timeout_ms=(
                        max(0, int(remaining * 1000))
                        if remaining is not None
                        else None
                    ),
                )
            except TimeoutError as exc:
                raise ForegroundDeadlineExceeded(
                    "summary publication preparation",
                    "source time-bound read exceeded the shared deadline",
                ) from exc
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_locked_error(exc):
                    raise
                raise ForegroundDeadlineExceeded(
                    "summary publication preparation",
                    "SQLite contention prevented a bounded source time-bound read",
                ) from exc
            summary_tokens = count_tokens(summary_text)
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "summary publication preparation",
            )

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            if background_feature_enabled:
                expected_frontier = self._last_compacted_store_id
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "foreground publication",
                )
                foreground_publication = self._publish_foreground_background_node(
                    node,
                    expected_session_fence_generation=int(
                        foreground_session_fence_generation or 0
                    ),
                    expected_frontier_store_id=expected_frontier,
                    consumed_store_ids=consumed_store_ids,
                    expected_source_identity_hash=(
                        foreground_source_identity_hash
                    ),
                    deadline=foreground_deadline,
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "foreground publication",
                )
                if not foreground_publication.promoted:
                    try:
                        lifecycle = self._lifecycle.get_by_conversation(
                            self._conversation_id,
                            lock_timeout_seconds=(
                                self._foreground_deadline_remaining(
                                    foreground_deadline
                                )
                            ),
                            deadline=foreground_deadline,
                            deadline_clock=self._foreground_now,
                        )
                    except TimeoutError as exc:
                        raise ForegroundDeadlineExceeded(
                            "foreground publication",
                            "lost-race lifecycle lookup exceeded the shared deadline",
                        ) from exc
                    if (
                        lifecycle is not None
                        and lifecycle.current_session_id == self._session_id
                        and lifecycle.current_frontier_store_id > expected_frontier
                    ):
                        # A prepared publisher won the serialized frontier CAS.
                        # Rebuild from its committed frontier; never drop raw rows
                        # beyond the range it actually published.
                        committed_frontier = int(
                            lifecycle.current_frontier_store_id
                        )
                        rebuilt_tail = self._call_with_foreground_deadline(
                            self._project_durable_active_tail,
                            committed_frontier,
                            preserve_messages=working_messages,
                            deadline=foreground_deadline,
                        )
                        self._foreground_deadline_checkpoint(
                            foreground_deadline,
                            "published context rebuild",
                        )
                        rebuilt = working_messages[:leading_anchor_count] + rebuilt_tail
                        working_messages = self._call_foreground_projection(
                            self._sanitize_active_context_messages,
                            rebuilt,
                            deadline=foreground_deadline,
                            stage="published context cleanup",
                        )
                        self._last_compacted_store_id = committed_frontier
                        pressure_messages = list(working_messages)
                        leaf_compacted_this_turn = True
                        leaf_passes += 1
                        sweep_stop_reason = "prepared_publication_won_race"
                        break
                    noop_reason = foreground_publication.reason
                    break
            else:
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "summary publication",
                )
                next_frontier_store_id = (
                    max(consumed_store_ids) if consumed_store_ids else 0
                )
                self._publish_foreground_dag_node(
                    node,
                    deadline=foreground_deadline,
                    stage="summary publication",
                    frontier_store_id=next_frontier_store_id,
                )
                self._last_compacted_store_id = next_frontier_store_id
                self._invalidate_rollups_for_published_node(
                    node,
                    deadline=foreground_deadline,
                )
                self._maybe_gc_compacted_tool_results(
                    compacted_chunk,
                    source_store_ids,
                    deadline=foreground_deadline,
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "summary publication",
                )

            pressure_remaining_messages = pressure_messages[leading_anchor_count + selected_raw_len:]
            working_messages = working_messages[:leading_anchor_count] + remaining_messages
            pressure_messages = pressure_messages[:leading_anchor_count] + pressure_remaining_messages
            leaf_compacted_this_turn = True
            leaf_passes += 1
            estimated_active_tokens = max(0, estimated_active_tokens - source_tokens + summary_tokens)
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "foreground compaction sweep",
            )

            if threshold_full_sweep_active:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    sweep_raw_drained = True
                    sweep_stop_reason = "raw_prefix_drained"
                    break
                continue

            if not self._config.dynamic_leaf_chunk_enabled:
                break

            if not force_overflow:
                if (not deferred_maintenance_active) and self.threshold_tokens > 0 and estimated_active_tokens < self.threshold_tokens:
                    break
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_fresh_tail_start = self._fresh_tail_start(pressure_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                if not remaining_raw:
                    break
                pressure_remaining_raw = pressure_messages[
                    leading_anchor_count:remaining_fresh_tail_start
                ]
                remaining_raw_tokens = count_messages_tokens(pressure_remaining_raw)
                remaining_threshold = self._working_leaf_chunk_tokens(remaining_raw_tokens)
                if remaining_raw_tokens < remaining_threshold:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        break

        if (
            threshold_full_sweep_active
            and not sweep_raw_drained
            and not sweep_stop_reason
            and leaf_passes >= max_leaf_passes
        ):
            sweep_stop_reason = "pass_budget_exhausted"

        if not leaf_compacted_this_turn:
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "active context finalization",
            )
            self._refresh_raw_backlog_debt(
                working_messages,
                observed_tokens=observed_prompt_tokens,
                deadline=foreground_deadline,
            )
            if force_overflow and len(messages) >= 1:
                logger.info(
                    "LCM forced overflow found no summarizable leaf: session=%s reason=%s",
                    self._session_id,
                    noop_reason,
                )
                leading_anchor_count = self._leading_anchor_count(working_messages)
                compressed = self._call_with_foreground_deadline(
                    self._assemble_overflow_recovery_context,
                    working_messages[0] if leading_anchor_count else None,
                    working_messages[leading_anchor_count:],
                    assembly_cap_override=recovery_assembly_cap,
                    deadline=foreground_deadline,
                )
                self._foreground_deadline_checkpoint(
                    foreground_deadline,
                    "overflow context assembly",
                )
                return self._finalize_forced_overflow_result(
                    working_messages,
                    compressed,
                    assembly_cap_override=recovery_assembly_cap,
                    ingest_cleanup_changed_active_context=ingest_cleanup_changed_active_context,
                    deadline=foreground_deadline,
                )
            active_context_messages = self._drop_preexisting_generated_ignored_dependent_eof_replies(
                working_messages,
                preexisting_dependent_reply_records,
                deadline=foreground_deadline,
            )
            if dropped_replayed_scaffold_messages:
                leading_anchor_count = self._leading_anchor_count(active_context_messages)
                anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
                self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
                try:
                    sanitized_messages = self._call_with_foreground_deadline(
                        self._assemble_context,
                        active_context_messages[0] if leading_anchor_count else None,
                        active_context_messages[leading_anchor_count:],
                        assembly_cap_override=recovery_assembly_cap,
                        deadline=foreground_deadline,
                    )
                finally:
                    self._pending_context_anchor_messages = None
            else:
                sanitized_messages = self._call_foreground_projection(
                    self._sanitize_active_context_messages,
                    active_context_messages,
                    insert_missing_tool_stubs=False,
                    deadline=foreground_deadline,
                    stage="active context finalization",
                )
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "active context finalization",
            )
            if sanitized_messages != working_messages or ingest_cleanup_changed_active_context:
                # _ingest_messages() already advanced the cursor to the original
                # active-context length. If the host continues from a sanitized
                # or reassembled context, keeping the old cursor could make the
                # next appended messages look already ingested. This applies to
                # content-only cleanup as well as dropped-message cleanup.
                self._align_ingest_cursor_to_active_context(
                    sanitized_messages,
                    deadline=foreground_deadline,
                )
                self._last_compression_status = "sanitized"
                self._last_compression_noop_reason = ""
            else:
                if dropped_replayed_scaffold_messages:
                    # The active context changed even though no new leaf node was
                    # written. Keep the cursor aligned with the returned context
                    # so the next appended turn is ingested instead of skipped.
                    self._align_ingest_cursor_to_active_context(
                        sanitized_messages,
                        deadline=foreground_deadline,
                    )
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = noop_reason
                logger.info("LCM compression no-op: %s", noop_reason)
            if threshold_full_sweep_active:
                duration_ms = (time.perf_counter() - _compress_started) * 1000.0
                self._last_threshold_full_sweep = {
                    **self._last_threshold_full_sweep,
                    "status": "noop",
                    "duration_ms": round(duration_ms, 3),
                    "stop_reason": sweep_stop_reason or noop_reason,
                    "budget_exhausted": sweep_stop_reason
                    in {"pass_budget_exhausted", "time_budget_exhausted"},
                }
            self._persist_active_placeholder_metadata(
                sanitized_messages,
                deadline=foreground_deadline,
            )
            self._foreground_deadline_checkpoint(
                foreground_deadline,
                "active context finalization",
            )
            return sanitized_messages

        # Step 6: Check if condensation is needed. A threshold full sweep only
        # condenses after the eligible raw prefix has been drained, and shares
        # the same total pass/deadline budget as its leaf work.
        condensation_passes = 0
        if threshold_full_sweep_active:
            if sweep_raw_drained:
                remaining_passes = max(
                    0,
                    sweep_pass_budget - leaf_passes,
                )
                condensation_passes, sweep_stop_reason = (
                    self._run_threshold_sweep_condensation(
                        target_tokens=sweep_target_tokens,
                        pass_budget=remaining_passes,
                        deadline=sweep_deadline,
                        focus_topic=focus_topic,
                    )
                )
        elif not background_emergency:
            self._maybe_condense(
                focus_topic=focus_topic,
                leaf_compacted_this_turn=True,
                force_overflow=force_overflow,
                critical_budget_pressure=critical_budget_pressure,
                deadline=sweep_deadline,
            )
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "condensation",
        )

        # Step 7: Assemble new active context
        self._refresh_raw_backlog_debt(
            working_messages,
            observed_tokens=observed_prompt_tokens,
            deadline=foreground_deadline,
        )
        leading_anchor_count = self._leading_anchor_count(working_messages)
        anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
        self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
        try:
            compressed = self._call_with_foreground_deadline(
                self._assemble_context,
                working_messages[0] if leading_anchor_count else None,
                working_messages[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
                deadline=foreground_deadline,
            )
        finally:
            self._pending_context_anchor_messages = None
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "active context assembly",
        )
        self.compression_count += 1
        self._last_compaction_duration_ms = (time.perf_counter() - _compress_started) * 1000.0
        try:
            dag_node_count = self._call_bounded_foreground_dag_read(
                self._dag.get_session_node_count,
                self._session_id,
                deadline=foreground_deadline,
                stage="compaction telemetry",
            )
        except ForegroundDeadlineExceeded:
            dag_node_count = -1
        logger.info(
            "LCM leaf compaction finished in %.1fms", self._last_compaction_duration_ms
        )
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = ""
        if recovery_assembly_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > recovery_assembly_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d after compaction; returning best-effort context (%d tokens)",
                    recovery_assembly_cap,
                    count_messages_tokens(compressed),
                )
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._align_ingest_cursor_to_active_context(
            compressed,
            deadline=foreground_deadline,
        )

        logger.info(
            "LCM compaction #%d: %d messages → %d (%d leaf pass%s, %d→%d tokens, %d DAG nodes%s)",
            self.compression_count,
            len(messages),
            len(compressed),
            leaf_passes,
            "es" if leaf_passes != 1 else "",
            count_messages_tokens(messages),
            count_messages_tokens(compressed),
            dag_node_count,
            ", forced overflow recovery" if force_overflow else "",
        )

        # ── Active-context cleanup / tool-pair guardrail (same as _assemble_context) ──
        # compress() output is consumed directly by the main loop in some
        # edge cases (e.g. forced overflow recovery bypassing _assemble_context).
        compressed = self._call_foreground_projection(
            self._sanitize_active_context_messages,
            compressed,
            deadline=foreground_deadline,
            stage="active context finalization",
        )
        if threshold_full_sweep_active:
            total_passes = leaf_passes + condensation_passes
            duration_ms = (time.perf_counter() - _compress_started) * 1000.0
            final_stop_reason = sweep_stop_reason or "raw_prefix_drained"
            partial_stop_reasons = {
                "pass_budget_exhausted",
                "time_budget_exhausted",
                "leaf_summary_error",
                "condensation_error",
                "condensation_no_progress",
                "no_same_depth_condensation_group",
            }
            self._last_threshold_full_sweep = {
                "status": "partial" if final_stop_reason in partial_stop_reasons else "completed",
                "leaf_passes": leaf_passes,
                "condensation_passes": condensation_passes,
                "total_passes": total_passes,
                "duration_ms": round(duration_ms, 3),
                "tokens_before": self._last_threshold_full_sweep["tokens_before"],
                "tokens_after": count_messages_tokens(compressed),
                "summary_prefix_tokens_before": sweep_summary_prefix_before,
                "summary_prefix_tokens_after": self._summary_frontier_tokens(
                    deadline=foreground_deadline,
                ),
                "summary_prefix_target_tokens": sweep_target_tokens,
                "stop_reason": final_stop_reason,
                "budget_exhausted": final_stop_reason
                in {"pass_budget_exhausted", "time_budget_exhausted"},
            }
        self._persist_active_placeholder_metadata(
            compressed,
            deadline=foreground_deadline,
        )
        self._foreground_deadline_checkpoint(
            foreground_deadline,
            "active context finalization",
        )

        return compressed
