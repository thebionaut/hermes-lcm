"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor
with a DAG-based summarization system that preserves every message.
"""

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.context_engine import ContextEngine

from .codex_routing import (
    _codex_oauth_context_cap,
    _is_codex_gpt55_route,
)
from .config import LCMConfig
from .background_compaction import (
    BackgroundCompactionStore,
    ForegroundProviderPermitUnavailable,
    PromotionResult,
    ProviderPermit,
    acquire_background_runtime,
    background_policy_fingerprint,
)
from .dag import SummaryDAG, SummaryNode
from .diagnostics import _enforce_state_db_containment
from .engine_registry import (
    _ACTIVE_ENGINE_REGISTRY_LOCK,
    _ACTIVE_ENGINES_BY_CONVERSATION_ID,
    _ACTIVE_ENGINES_BY_SESSION_ID,
    _remove_registry_entries_for_engine,
    resolve_active_lcm_engine,  # noqa: F401  (re-exported: hosts import it from .engine)
)
from .escalation import (
    SummaryCircuitBreaker,
    SummarySpendGuard,
    summarize_with_escalation,
)
from .externalize import (
    ExternalizationOwnership,
    build_transcript_gc_placeholder,
    extract_externalized_ref,
    find_externalized_payload_for_message,
    find_externalized_tool_result_content_for_call,
    get_large_output_storage_dir,
    is_externalized_placeholder,
    load_externalized_payload,
    maybe_externalize_tool_output,
)
from .extraction import (
    PreparedExtraction,
    extract_before_compaction,
    persist_pre_compaction_extraction,
    sanitize_pre_compaction_content,
    strip_injected_context_blocks,
)
from .ingest_protection import (
    _expected_persisted_output_chars,
    _has_lossy_sensitive_redaction,
    _contains_media_payload,
    _is_hermes_persisted_output_marker,
    _persisted_output_inline_preview_sha256,
    _persisted_output_preview_prefix_digest,
    _persisted_output_saved_path,
    assistant_output_quarantine_reason,
    extract_all_externalized_payload_refs,
    extract_ingest_externalized_refs,
    protect_inline_payloads_in_text,
    protect_messages_for_ingest,
    quarantine_suspicious_assistant_messages,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_text,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
    sensitive_pattern_status,
)
from .runtime_identity import (
    _PLUGIN_ROOT,
    _git_runtime_identity,
    _plugin_metadata,
)
from .rollup_builder import (
    initialize_rollup_invalidation_outbox,
    mark_stale_for_published_summary,
    run_rollup_maintenance,
)
from .assertion_extraction import ModelAssertionExtractor
from .assertion_store import (
    AssertionSchemaReadyToken,
    AssertionStore,
    SourceSnapshot,
)
from .adaptive_retrieval import AdaptiveRetrievalRegistry
from .query_view_store import QueryViewStore
from .schemas import (
    LCM_DESCRIBE,
    LCM_DOCTOR,
    LCM_EXPAND,
    LCM_EXPAND_QUERY,
    LCM_GREP,
    LCM_INSPECT,
    LCM_LOAD_SESSION,
    LCM_COMPUTE,
    LCM_COMPILE_EVIDENCE,
    LCM_EVIDENCE_PACK,
    LCM_QUERY_STATE,
    LCM_RECALL,
    LCM_RECENT,
    LCM_RETRIEVE,
    LCM_STATUS,
)
from .sanitize import (
    _clean_active_assistant_message,
    _should_drop_active_assistant_message,
)
from .session_patterns import (
    build_session_match_keys,
    compile_session_patterns,
    matches_session_pattern,
)
from .message_analysis import (
    _tool_call_id,
)
from .fresh_tail import FreshTailBoundary, resolve_fresh_tail_boundary
from .message_patterns import compile_message_patterns, matches_message_pattern
from .aux_session import AuxiliarySessionMixin
from .placeholder_ledger import PlaceholderLedgerMixin
from .reconcile import (
    ForegroundDeadlineExceeded,
    ReconcileBudgetExceeded,
    ReconcileMixin,
    _PRESERVED_OBJECTIVE_CONTEXT_PREFIX,
)
from .compaction import CompactionMixin
from .reset_state import ResetStateMixin
from .bypass import BypassMixin
from .lifecycle_state import LifecycleStateStore
from .message_content import (
    normalize_content_value,
    stored_text_content_for_pattern_matching,
    text_content_for_pattern_matching,
)
from .sqlite_util import (
    _is_sqlite_locked_error,
    _temporary_sqlite_busy_timeout,
)
from .store import MessageStore
from .summary_serialization import (
    project_active_replay_messages,
    serialize_summary_messages,
)
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from . import tools as lcm_tools
from .working_set import (
    WORKING_SET_REFERENCE_PREFIX,
    WorkingSetDecision,
    select_phase_working_set,
)
from .active_replay_preview import (
    ActiveReplayPreviewPolicy,
    project_externalized_tool_results,
    session_matches_preview_allowlist,
)

logger = logging.getLogger(__name__)

_ASSERTION_EXTRACTION_PROCESS_SLOT = threading.BoundedSemaphore(1)
_FOREGROUND_PROJECTION_MAX_WORKERS = 4
_FOREGROUND_PROJECTION_PROCESS_SLOT = threading.BoundedSemaphore(
    _FOREGROUND_PROJECTION_MAX_WORKERS
)

class _RollupMaintenanceScheduler:
    """Run deduplicated rollup jobs on one process-wide worker.

    Session binding only enqueues work. The worker is shared by every engine so
    rapid gateway binds cannot create one thread per session. A key that is
    already queued is ignored; a key requested while active gets at most one
    follow-up pass, which preserves eventual progress when new staleness arrives
    during an in-flight build without allowing concurrent duplicate builds.
    """

    def __init__(self, max_pending_jobs: int = 64) -> None:
        self._condition = threading.Condition()
        self._max_pending_jobs = max(1, int(max_pending_jobs))
        self._jobs: deque[
            tuple[tuple[str, str], Callable[[], None]]
        ] = deque()
        self._queued_keys: set[tuple[str, str]] = set()
        self._active_keys: set[tuple[str, str]] = set()
        # One worker means at most one active key. Keep its requested rerun in
        # a literal single slot so queue saturation cannot discard the
        # documented active-pass follow-up guarantee or grow hidden state.
        self._follow_up_key: tuple[str, str] | None = None
        self._follow_up_job: Callable[[], None] | None = None
        self._running_follow_up = False
        self._exclusive_keys: set[tuple[str, str]] = set()
        self._owned_keys: dict[object, set[tuple[str, str]]] = {}
        self._key_owners: dict[tuple[str, str], set[object]] = {}
        self._worker: threading.Thread | None = None

    def _track_owner_locked(
        self,
        key: tuple[str, str],
        owner: object | None,
    ) -> None:
        if owner is None:
            return
        self._owned_keys.setdefault(owner, set()).add(key)
        self._key_owners.setdefault(key, set()).add(owner)

    def _release_key_owners_if_idle_locked(self, key: tuple[str, str]) -> None:
        if (
            key in self._queued_keys
            or key in self._active_keys
            or key in self._exclusive_keys
            or self._follow_up_key == key
        ):
            return
        for owner in self._key_owners.pop(key, set()):
            owned = self._owned_keys.get(owner)
            if owned is None:
                continue
            owned.discard(key)
            if not owned:
                self._owned_keys.pop(owner, None)

    def schedule(
        self,
        key: tuple[str, str],
        job: Callable[[], None],
        *,
        owner: object | None = None,
    ) -> bool:
        with self._condition:
            if key in self._exclusive_keys:
                logger.info(
                    "LCM temporal rollup maintenance deferred while an operator "
                    "rebuild owns database=%s scope=%s",
                    key[0],
                    key[1],
                )
                return False
            if key in self._queued_keys:
                self._track_owner_locked(key, owner)
                return True
            if key in self._active_keys and not self._running_follow_up:
                self._follow_up_key = key
                self._follow_up_job = job
                self._track_owner_locked(key, owner)
                self._condition.notify_all()
                return True
            if len(self._jobs) >= self._max_pending_jobs:
                logger.warning(
                    "LCM temporal rollup maintenance queue is full; "
                    "deferring database=%s scope=%s until a later bind",
                    key[0],
                    key[1],
                )
                return False
            if self._worker is None or not self._worker.is_alive():
                worker = threading.Thread(
                    target=self._run,
                    name="lcm-rollup-maintenance",
                    daemon=True,
                )
                worker.start()
                self._worker = worker
            self._jobs.append((key, job))
            self._queued_keys.add(key)
            self._track_owner_locked(key, owner)
            self._condition.notify_all()
            return True

    def try_acquire_exclusive(self, key: tuple[str, str]) -> bool:
        """Reserve one idle key for a synchronous operator rebuild.

        This is intentionally non-blocking: a manual rebuild must not race a
        provider-backed maintenance pass, but it also must not wait behind one
        on the gateway thread. The caller can ask the operator to retry once the
        background pass completes.
        """
        with self._condition:
            busy_keys = self._queued_keys | self._active_keys | self._exclusive_keys
            if key in busy_keys or self._follow_up_key == key:
                return False
            if len(self._exclusive_keys) >= self._max_pending_jobs:
                return False
            self._exclusive_keys.add(key)
            return True

    def release_exclusive(self, key: tuple[str, str]) -> None:
        with self._condition:
            self._exclusive_keys.discard(key)
            self._condition.notify_all()

    def drain(
        self,
        keys: set[tuple[str, str]],
        timeout: float | None = None,
    ) -> bool:
        """Wait until none of ``keys`` is queued or active."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while keys & (
                self._queued_keys
                | self._active_keys
                | self._exclusive_keys
                | ({self._follow_up_key} if self._follow_up_key else set())
            ):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def drain_owner(
        self,
        owner: object,
        timeout: float | None = None,
    ) -> bool:
        """Wait until every outstanding key accepted for ``owner`` is idle."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._owned_keys.get(owner):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._jobs:
                    self._condition.wait()
                key, job = self._jobs.popleft()
                self._queued_keys.discard(key)
                self._active_keys.add(key)
                self._running_follow_up = False
            while True:
                try:
                    job()
                except (Exception, asyncio.CancelledError):
                    logger.warning(
                        "LCM background temporal rollup maintenance failed for database=%s scope=%s",
                        key[0],
                        key[1],
                        exc_info=True,
                    )
                with self._condition:
                    if self._follow_up_key == key and self._follow_up_job is not None:
                        job = self._follow_up_job
                        self._follow_up_key = None
                        self._follow_up_job = None
                        self._running_follow_up = True
                        self._condition.notify_all()
                        continue
                    self._active_keys.discard(key)
                    self._running_follow_up = False
                    self._release_key_owners_if_idle_locked(key)
                    self._condition.notify_all()
                    break


_ROLLUP_MAINTENANCE_SCHEDULER = _RollupMaintenanceScheduler()

_SESSION_END_BUSY_TIMEOUT_MS = 50
_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85

# Auto-focus topic derivation: infer a compact focus hint from the most recent
# real user turns so that summarization can prioritise current user intent.
# Mirrors Hermes upstream fix/compression-auto-focus-topic (#44687 branch).
_AUTO_FOCUS_MAX_TURNS = 3
_AUTO_FOCUS_TURN_MAX_CHARS = 260
_AUTO_FOCUS_MAX_CHARS = 700

_PRESERVED_TODO_CONTEXT_PREFIX = "[Your active task list was preserved across context compression]"
_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT = 8


class LCMEngine(CompactionMixin, ResetStateMixin, ReconcileMixin, AuxiliarySessionMixin, PlaceholderLedgerMixin, BypassMixin, ContextEngine):
    """Lossless Context Management engine.

    Automatic LCM compaction is routine background maintenance. Hosts that
    support user-visible compaction status opt-outs should keep successful
    automatic LCM passes silent unless the user explicitly asks for diagnostics.

    Architecture:
      1. Every message is persisted verbatim in an immutable MessageStore
      2. When context pressure builds, older messages outside the fresh tail
         are summarized into leaf nodes (D0) in a SummaryDAG
      3. When enough nodes accumulate at a depth, they're condensed into
         higher-depth nodes (D1, D2, ...)
      4. The agent gets tools (lcm_grep, lcm_load_session, lcm_describe,
         lcm_expand) to search and drill into compacted history
      5. Active context = system prompt + DAG summaries + fresh tail
    """

    # Explicit host capability: LCM validates and sizes every summarizer request
    # against its selected auxiliary route. The host may therefore keep the main
    # model's context threshold independent from a smaller summarizer's window.
    # Core must check this opt-in, never infer it from a non-ContextCompressor type.
    owns_summary_request_budget = True

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home
        self._background_compaction: BackgroundCompactionStore | None = None
        self._background_runtime_handle: Any = None
        # A timed-out Python thread may remain inside a provider that ignores
        # cancellation. Keep all foreground provider-like calls single-flight
        # until the physical call returns instead of accumulating orphans.
        self._foreground_hard_call_slot = threading.BoundedSemaphore(1)
        self._foreground_projection_call_slot = threading.BoundedSemaphore(1)
        self._extraction_persistence_slot = threading.BoundedSemaphore(1)
        self._extraction_persistence_idle = threading.Event()
        self._extraction_persistence_idle.set()
        self._assertion_extraction_metrics_lock = threading.RLock()
        # Restart reconciliation temporarily installs per-call memo/deadline
        # state on the engine. Serialize it so concurrent post-LLM hooks cannot
        # overwrite or clear another call's guardrails.
        self._reconcile_lock = threading.Lock()
        self._assertion_extraction_idle = threading.Event()
        self._assertion_extraction_idle.set()
        self._assertion_extraction_batches_scheduled = 0
        self._assertion_extraction_batches_skipped_busy = 0
        self._assertion_extraction_sources_scheduled = 0
        self._assertion_extraction_sources_completed = 0
        self._assertion_extraction_sources_failed = 0
        self._assertion_extraction_provider_calls = 0
        self._assertion_extraction_input_tokens = 0
        self._assertion_extraction_output_tokens = 0
        self._assertion_extraction_last_duration_ms = 0.0
        self._assertion_extraction_last_error = ""
        self._assertion_extraction_last_model = ""

        db_path = self._resolve_db_path(hermes_home)
        self._bind_storage(db_path, hermes_home)

        self._session_id: str = ""
        self._session_platform: str = ""
        # Tracks the most recent non-ignored, non-stateless binding so that
        # user-facing tools (lcm_status, lcm_grep default scope, lcm_describe,
        # lcm_expand_query, lcm_doctor) keep showing the foreground session
        # even while a side-channel session (cron, debug) temporarily owns the
        # engine's _session_id binding. Updated alongside _session_id only
        # when _refresh_session_filters classifies the new session as a real
        # foreground (neither ignored nor stateless). Read via the
        # `current_session_id` / `current_session_platform` properties and
        # `current_session_ignored` / `current_session_stateless` /
        # `side_channel_active` companion predicates.
        self._foreground_session_id: str = ""
        self._foreground_session_platform: str = ""
        self._foreground_conversation_id: str = ""
        self._foreground_rebind_session_id: str = ""
        self._foreground_rebind_previous_session_id: str = ""
        self._foreground_rebind_previous_platform: str = ""
        self._foreground_rebind_previous_conversation_id: str = ""
        self._foreground_rebind_parent_session_id: str = ""
        self._conversation_id: str = ""
        self._session_match_keys: list[str] = []
        self._session_ignored = False
        self._session_stateless = False
        self._compiled_ignore_session_patterns = compile_session_patterns(
            self._config.ignore_session_patterns
        )
        self._compiled_stateless_session_patterns = compile_session_patterns(
            self._config.stateless_session_patterns
        )
        self._compiled_ignore_message_patterns = compile_message_patterns(
            self._config.ignore_message_patterns
        )
        self._ignored_message_count: int = 0
        # Raw messages permanently dropped because they matched
        # ignore_message_patterns. These are NOT persisted anywhere, so an
        # over-broad operator pattern silently discards substantive turns from
        # the "lossless" store. Count + log them so the loss is at least
        # visible; full lossless retention (store with ignored=1) is a larger
        # follow-up that touches cursor reconciliation and FTS.
        self._ignore_pattern_dropped_count: int = 0

        # Track which store_ids have been ingested into the DAG
        self._last_compacted_store_id: int = 0

        # Cursor: index in the current messages list up to which all
        # messages have been persisted.  After compress() shortens the
        # list, the cursor resets to len(compressed) so that only
        # genuinely new messages (appended after compaction) get ingested.
        # The cursor is process-local; existing sessions rebound after a
        # gateway restart reconcile it against the durable store on the
        # next ingest.
        self._ingest_cursor: int = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation: Dict[str, Any] = {
            "action": "none",
            "reason": "not run",
        }

        # State required by ContextEngine ABC and run_agent.py compatibility
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""
        self.raw_context_length = 0
        self.context_length = 0
        self.effective_context_length_cap: int | None = None
        self.effective_context_length_reason = ""
        self._context_length_source = ""
        self._update_model_pending_session_start = False
        self.threshold_tokens = 0
        self.context_threshold = self._config.context_threshold
        self.threshold_percent = self.context_threshold
        self._context_threshold_source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        self._context_threshold_autoraised: dict[str, float] | None = None
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self.compression_count = 0
        # Wall-clock of the last leaf compaction (ms); surfaced via telemetry only.
        self._last_compaction_duration_ms = 0.0
        # run_agent.py reads these for preflight checks
        self.protect_first_n = 3
        self.protect_last_n = self._config.fresh_tail_count
        # run_agent.py reads these for context probing
        self._context_probed = False
        self._context_probe_persistable = False
        # Host compatibility: LCM treats successful automatic compaction as
        # silent maintenance. Manual /lcm diagnostics and warning/error paths
        # remain explicit.
        self.emit_automatic_compaction_status = False
        self.quiet_mode = True
        self.summary_model = self._config.summary_model
        self._summary_circuit_breaker = SummaryCircuitBreaker(
            failure_threshold=self._config.summary_circuit_breaker_failure_threshold,
            cooldown_seconds=self._config.summary_circuit_breaker_cooldown_seconds,
        )
        # Summary spend guard: process-local sliding window so a loop that
        # keeps succeeding cannot burn auxiliary-model budget without bound. When
        # tripped, escalation falls back to deterministic L3 truncation. Set
        # summary_spend_max_calls=0 to disable.
        self._summary_spend_guard = SummarySpendGuard(
            max_calls=int(self._config.summary_spend_max_calls),
            window_seconds=float(self._config.summary_spend_window_seconds),
            backoff_seconds=float(self._config.summary_spend_backoff_seconds),
        )
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_threshold_full_sweep: dict[str, Any] = {
            "status": "never_run",
            "leaf_passes": 0,
            "condensation_passes": 0,
            "total_passes": 0,
            "duration_ms": 0.0,
            "tokens_before": 0,
            "tokens_after": 0,
            "summary_prefix_tokens_before": 0,
            "summary_prefix_tokens_after": 0,
            "summary_prefix_target_tokens": 0,
            "stop_reason": "",
            "budget_exhausted": False,
        }
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        self._last_working_set_decision: dict[str, Any] = {
            "reason": "never_run",
            "source_tokens": 0,
            "selected_tokens": 0,
            "omitted_tokens": 0,
            "source_sha256": "",
            "cache_state": "unknown",
            "payback_calls": 0,
        }
        self._active_replay_preview_policies: dict[
            str, ActiveReplayPreviewPolicy
        ] = {}
        self._last_active_replay_preview: dict[str, Any] = {
            "reason": "never_run",
            "policy_version": "externalized-tool-preview-v1",
            "eligible_results": 0,
            "projected_results": 0,
            "source_tokens": 0,
            "projected_tokens": 0,
            "tokens_saved": 0,
            "refs": [],
        }
        self._last_compaction_payback: dict[str, Any] = {
            "allowed": None,
            "reason": "never_evaluated",
            "same_source_denials": 0,
        }
        self._last_foreground_deadline: Dict[str, Any] = {
            "status": "never_exhausted"
        }
        # Ingest-failure tracking. The core promise is that nothing is ever
        # lost, but a swallowed persistence error (disk full, DB locked,
        # corruption) silently breaks it: the turn continues while messages
        # exist only in the volatile host list. Surface it instead of hiding
        # it in a debug log so get_status()/doctor can escalate. Store-scoped,
        # not session-scoped, so it is not cleared on session reset.
        self._ingest_failure_count = 0
        self._consecutive_ingest_failures = 0
        # Proactive-recall injection telemetry (SPEC F). Store-scoped counters so
        # a session reset does not zero the operator's running totals; surfaced
        # through lcm_status. injected = a block was placed this assembly;
        # skipped = ran but nothing survived the floor/dedupe; timeout = the
        # recall query hit its deadline (inject nothing, never block assembly).
        self._proactive_recall_injected_count = 0
        self._proactive_recall_skipped_count = 0
        self._proactive_recall_timeout_count = 0
        self._last_ingest_error = ""
        self._last_ingest_error_time: float = 0
        # Cooldown timestamp to prevent compression cascade after boundary skip.
        # Set when skip-carry-over path is taken in _continue_compression_boundary.
        self._last_boundary_skip_time: float = 0
        # One-shot handoff from preflight: adopt an already-durable replay
        # cleanup during boundary cooldown without running summary work.
        self._preflight_cleanup_only_due_to_boundary_cooldown = False
        # Temporary source window used only while compress() assembles context.
        # _assemble_context also serves tests and recovery paths directly, so
        # keep anchoring opt-in rather than changing its public behavior.
        self._pending_context_anchor_messages: Optional[List[Dict[str, Any]]] = None
        self._current_compress_store_ids_by_message_id: dict[int, int] = {}
        self._current_compress_placeholder_identity_counts: dict[tuple[str, str, str, str], int] = {}
        self._last_active_replay_source_identities: list[tuple[Any, ...]] = []
        self._last_active_replay_messages: list[Dict[str, Any]] = []
        self._last_active_replay_store_ids: list[Optional[int]] = []
        self._last_active_replay_explicit_store_id_indices: set[int] = set()
        self._active_replay_store_refs_by_message_id: dict[
            int,
            tuple[Dict[str, Any], Optional[int]],
        ] = {}
        self._active_replay_store_ref_session_id = ""
        self._active_replay_store_ref_conversation_id = ""
        self._generated_ignored_active_replay_placeholder_message_ids: set[int] = set()
        self._logged_filter_config = False
        self._pending_reset_session_id: str = ""
        self._pending_reset_conversation_id: str = ""
        self._pending_reset_frontier_store_id: int = 0
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget: dict[str, int] = {}
        self._compression_boundary_active_placeholder_digest_ordinals: dict[str, set[int]] = {}
        self._compression_boundary_stored_placeholder_digest_counts: dict[str, int] = {}
        self._thread_context = threading.local()
        self._auxiliary_session_ids: set[str] = set()
        self._auxiliary_lineage_session_ids: set[str] = set()
        self._auxiliary_last_prompt_tokens: dict[str, int] = {}
        self._auxiliary_session_generations: dict[str, int] = {}
        self._auxiliary_generation_tokens: dict[int, tuple[Any, int]] = {}
        self._auxiliary_next_generation_token = 0
        self._auxiliary_direct_end_guard_session_ids: set[str] = set()
        self._auxiliary_handoff_parent_session_ids: dict[str, str] = {}
        self._auxiliary_retired_session_generations: dict[str, set[int]] = {}
        self._auxiliary_foreground_reused_session_ids: set[str] = set()
        self._lcm_bypass_lineage_session_ids: set[str] = set()
        self._lcm_bypass_lineage_platforms: dict[str, set[str]] = {}
        self._lcm_non_bypass_platforms: dict[str, set[str]] = {}
        self._lcm_session_last_platform: dict[str, str] = {}
        self._lcm_session_last_normal_platform: dict[str, str] = {}
        self._lcm_session_last_bypassed: dict[str, bool] = {}
        self._lcm_session_last_conversation_id: dict[str, str] = {}
        self._lcm_session_last_normal_conversation_id: dict[str, str] = {}
        self._lcm_bypass_message_prefix_fingerprints: dict[
            str, list[tuple[list[str], bool]]
        ] = {}
        self._lcm_normal_message_prefix_fingerprints: dict[tuple[str, str], list[str]] = {}
        self._lcm_current_start_allows_bypass_lineage = False
        self._auxiliary_session_lock = threading.RLock()
        self._host_fallback_compressor: Any = None
        self._host_fallback_session_id = ""
        self._host_fallback_import_warning_logged = False
        # The scheduler associates this identity only with outstanding work, so
        # diagnostic drains do not retain every historical session key forever.
        self._rollup_maintenance_owner = object()

    def clone_for_agent(self) -> "LCMEngine":
        """Return a fresh runtime engine for one AIAgent instance.

        Hermes registers plugin context engines process-wide, while gateway
        runtimes may keep multiple cached AIAgent instances alive at once
        (different platforms, chats, cron jobs, etc.).  LCM stores mutable
        session binding and ingest cursor state on the engine object itself, so
        sharing one registered instance across agents can let one conversation
        rebind another conversation's raw-message ingest and lifecycle state.

        The clone shares the same durable SQLite database path/configuration,
        but gets independent session/cursor/lifecycle runtime state. Runtime
        model and context-window metadata is copied so the clone is immediately
        budget-aware even before a compatible Hermes host calls update_model().
        """
        clone = type(self)(
            config=copy.deepcopy(self._config),
            hermes_home=self._hermes_home,
        )
        clone.model = self.model
        clone.base_url = self.base_url
        clone.api_key = self.api_key
        clone.provider = self.provider
        clone.api_mode = self.api_mode
        if self._context_length_source:
            clone._set_context_length(
                self.raw_context_length,
                source=self._context_length_source,
                model=self.model,
                provider=self.provider,
            )
        elif self.raw_context_length or self.context_length:
            clone._set_context_length(
                self.raw_context_length or self.context_length,
                source="clone_for_agent",
                model=self.model,
                provider=self.provider,
            )
        # ``update_model()`` authority is a per-runtime lifecycle edge, not
        # durable metadata.  Compatible hosts call update_model() on the clone
        # before binding it; hosts that bind only through on_session_start()
        # must still be able to replace the copied prototype route.
        clone._update_model_pending_session_start = False
        clone._lcm_current_start_allows_bypass_lineage = False
        return clone

    def __deepcopy__(self, memo: dict[int, object]) -> "LCMEngine":
        """Copy the plugin runtime without pickling SQLite-backed helpers.

        Hermes core may deepcopy plugin context engines while creating isolated
        AIAgent instances. A default object deepcopy walks into MessageStore,
        SummaryDAG, and LifecycleStateStore sqlite3.Connection handles, which
        cannot be pickled. LCM already exposes clone_for_agent() as the safe
        boundary: share durable configuration/database path, but allocate fresh
        per-agent runtime/storage helper objects.
        """
        clone = self.clone_for_agent()
        memo[id(self)] = clone
        return clone

    def _resolve_db_path(self, hermes_home: str = "") -> Path:
        """Resolve the SQLite path for the active Hermes profile/home."""
        if self._config.database_path:
            return Path(self._config.database_path)
        if hermes_home:
            return Path(hermes_home) / "lcm.db"
        return Path.home() / ".hermes" / "lcm.db"

    def _bind_storage(self, db_path: str | Path, hermes_home: str = "") -> None:
        """Bind store/DAG/lifecycle helpers to one SQLite database."""
        self._assertions = None
        self._query_views = None
        self._adaptive_retrieval = None
        self._assertion_extractor = None
        self._assertion_schema_ready_token = None
        try:
            self._store = MessageStore(
                db_path,
                ingest_protection_config=self._config,
                hermes_home=hermes_home,
            )
            # Resolve the configured target without creating it. Automatic
            # foreground projections are abandonable and therefore may use an
            # already-provisioned directory, but must never mkdir one. Manual
            # and background writes retain their normal lazy-create path.
            self._foreground_externalization_storage_dir = None
            try:
                self._foreground_externalization_storage_dir = (
                    get_large_output_storage_dir(
                        self._config,
                        hermes_home=hermes_home,
                        create=False,
                    )
                )
            except (OSError, ValueError) as exc:
                logger.warning(
                    "LCM externalization storage unavailable at engine bind: %s",
                    exc,
                )
            self._dag = SummaryDAG(db_path)
            if self._config.temporal_rollups_enabled:
                # Install the transaction-coupled summary mutation triggers before
                # this engine can publish or delete a DAG node.
                initialize_rollup_invalidation_outbox(self._dag)
            self._lifecycle = LifecycleStateStore(db_path)
            if (
                self._config.background_compaction_enabled
                and not self._background_compaction_compatibility_blocker()
            ):
                self._background_compaction = BackgroundCompactionStore(
                    db_path,
                    config=self._config,
                )
                self._background_runtime_handle = acquire_background_runtime(
                    db_path,
                    config=self._config,
                )
            self._assertions = (
                AssertionStore(db_path)
                if bool(getattr(self._config, "assertions_enabled", False))
                else None
            )
            self._query_views = (
                QueryViewStore(db_path)
                if bool(getattr(self._config, "query_views_enabled", False))
                or bool(getattr(self._config, "adaptive_retrieval_enabled", False))
                else None
            )
            self._adaptive_retrieval = (
                AdaptiveRetrievalRegistry(self._query_views)
                if bool(
                    getattr(self._config, "adaptive_retrieval_enabled", False)
                )
                else None
            )
            if (
                self._assertions is not None
                and bool(getattr(self._config, "assertion_extraction_enabled", False))
            ):
                # All profile stores have finished startup DDL. Issue one exact
                # inode/schema-cookie proof so deadline-owned workers can skip
                # migrations without weakening schema validation.
                self._assertion_schema_ready_token = (
                    self._assertions.issue_schema_ready_token()
                )
                self._assertion_extractor = ModelAssertionExtractor(
                    self._assertions,
                    model=self._assertion_extraction_model(),
                    timeout_seconds=self._assertion_extraction_timeout(),
                )
        except Exception:
            self._close_storage()
            raise

    def _close_storage(self) -> None:
        """Best-effort close of currently bound SQLite helpers."""
        self._assertion_schema_ready_token = None
        self._foreground_externalization_storage_dir = None
        runtime_handle = getattr(self, "_background_runtime_handle", None)
        self._background_runtime_handle = None
        release = getattr(runtime_handle, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                logger.debug(
                    "LCM failed releasing background compaction runtime",
                    exc_info=True,
                )
        for attr in (
            "_background_compaction",
            "_adaptive_retrieval",
            "_store",
            "_dag",
            "_lifecycle",
            "_assertions",
            "_query_views",
        ):
            helper = getattr(self, attr, None)
            close = getattr(helper, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("LCM failed closing %s during profile rebind", attr, exc_info=True)
            if attr == "_background_compaction":
                self._background_compaction = None

    def _assertion_extraction_model(self) -> str:
        return str(
            getattr(self._config, "assertion_extraction_model", "")
            or getattr(self._config, "extraction_model", "")
            or getattr(self._config, "summary_model", "")
            or ""
        )

    def _assertion_extraction_timeout(self) -> float:
        value = float(
            getattr(self._config, "assertion_extraction_timeout_seconds", 30.0)
        )
        return min(120.0, max(0.1, value))

    def _reset_profile_runtime_state(self) -> None:
        """Clear process-local session state that cannot cross profile homes."""
        if self._adaptive_retrieval is not None:
            self._adaptive_retrieval.clear()
        self._unregister_active_engine_binding()
        self._session_id = ""
        self._session_platform = ""
        self._foreground_session_id = ""
        self._foreground_session_platform = ""
        self._foreground_conversation_id = ""
        self._clear_foreground_rebind_candidate()
        self._conversation_id = ""
        self._session_match_keys = []
        self._session_ignored = False
        self._session_stateless = False
        self._clear_pending_reset_boundary()
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.clear()
            self._auxiliary_lineage_session_ids.clear()
            self._auxiliary_last_prompt_tokens.clear()
            self._auxiliary_session_generations.clear()
            self._auxiliary_generation_tokens.clear()
            self._auxiliary_next_generation_token = 0
            self._auxiliary_direct_end_guard_session_ids.clear()
            self._auxiliary_handoff_parent_session_ids.clear()
            self._auxiliary_retired_session_generations.clear()
            self._auxiliary_foreground_reused_session_ids.clear()
            self._lcm_bypass_lineage_session_ids.clear()
            self._lcm_bypass_lineage_platforms.clear()
            self._lcm_non_bypass_platforms.clear()
            self._lcm_session_last_platform.clear()
            self._lcm_session_last_normal_platform.clear()
            self._lcm_session_last_bypassed.clear()
            self._lcm_session_last_conversation_id.clear()
            self._lcm_session_last_normal_conversation_id.clear()
            self._lcm_bypass_message_prefix_fingerprints.clear()
            self._lcm_normal_message_prefix_fingerprints.clear()
        self._lcm_current_start_allows_bypass_lineage = False
        self._host_fallback_compressor = None
        self._host_fallback_session_id = ""
        self._host_fallback_import_warning_logged = False
        self._clear_thread_context_stateless()
        self._reset_session_scoped_runtime_state()

    def _rebind_storage_for_home(self, hermes_home: str = "") -> bool:
        """Switch SQLite-backed state when a reused engine serves another profile.

        Hermes core passes the active ``hermes_home`` on session start.  Older
        Hermes versions may still reuse the same plugin/context-engine object
        after ``HERMES_HOME`` changes, so the plugin must not assume the store
        captured during ``register()`` is still correct.
        """
        if not hermes_home:
            return False
        if self._config.database_path:
            current_home = str(self._hermes_home or "")
            current_store_home = str(getattr(getattr(self, "_store", None), "_hermes_home", "") or "")
            if current_home == str(hermes_home) and current_store_home == str(hermes_home):
                return False
            self._hermes_home = hermes_home
            store = getattr(self, "_store", None)
            if store is not None:
                store._hermes_home = hermes_home
            self._reset_profile_runtime_state()
            logger.info("LCM rebound Hermes home for configured database path %s", hermes_home)
            return True

        db_path = self._resolve_db_path(hermes_home)
        current_db = Path(getattr(getattr(self, "_store", None), "db_path", ""))
        if current_db == db_path and str(self._hermes_home or "") == str(hermes_home):
            return False

        self._close_storage()
        self._hermes_home = hermes_home
        self._bind_storage(db_path, hermes_home)
        self._reset_profile_runtime_state()
        logger.info("LCM rebound storage for Hermes home %s", hermes_home)
        return True

    def _runtime_context_threshold(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[float, str, dict[str, float] | None]:
        configured = float(self._config.context_threshold)
        source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        explicit_lcm_override = source in {
            "env:LCM_CONTEXT_THRESHOLD",
            "config_yaml:lcm.context_threshold",
        }
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        if (
            _is_codex_gpt55_route(route_model, route_provider)
            and self._config.codex_gpt55_autoraise_enabled
            and not explicit_lcm_override
            and configured < _CODEX_GPT55_COMPACTION_THRESHOLD
        ):
            return (
                _CODEX_GPT55_COMPACTION_THRESHOLD,
                "codex_gpt55_autoraise",
                {"from": configured, "to": _CODEX_GPT55_COMPACTION_THRESHOLD},
            )
        return configured, source, None

    def _effective_context_length(
        self,
        raw_context_length: int,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[int, int | None, str]:
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        cap = _codex_oauth_context_cap(route_model, route_provider)
        if cap is not None and raw_context_length > cap:
            return (
                cap,
                cap,
                "codex_oauth_context_cap",
            )
        return raw_context_length, None, ""

    def _effective_threshold_tokens(self, context_threshold_tokens: int) -> int:
        """Return the host-visible preflight trigger token count.

        Hermes core uses ``threshold_tokens`` as a cheap gate before it pays for
        the full request estimate that includes system prompt and tool schemas.
        LCM can enforce a stricter active-context assembly cap than the normal
        context-threshold value, so expose the stricter cap here; otherwise a
        tool/schema-heavy request can skip host preflight entirely.
        """
        publication_cap = getattr(self, "_admission_publication_tokens", 0)
        if publication_cap:
            context_threshold_tokens = min(context_threshold_tokens, publication_cap) if context_threshold_tokens > 0 else publication_cap
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is not None and assembly_cap > 0:
            if context_threshold_tokens > 0:
                return min(context_threshold_tokens, assembly_cap)
            return assembly_cap
        return context_threshold_tokens

    def _set_context_length(
        self,
        context_length: Any,
        *,
        source: str,
        model: str | None = None,
        provider: str | None = None,
    ) -> bool:
        try:
            parsed_context_length = int(context_length)
        except (TypeError, ValueError):
            logger.debug("LCM ignored invalid %s context_length: %r", source, context_length)
            return False
        if parsed_context_length <= 0:
            logger.debug(
                "LCM cleared non-positive %s context_length: %r",
                source,
                context_length,
            )
            self.raw_context_length = 0
            self.context_length = 0
            self.effective_context_length_cap = None
            self.effective_context_length_reason = ""
            self._context_length_source = source
            self.threshold_tokens = 0
            self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
                self._runtime_context_threshold(model=model, provider=provider)
            )
            self.threshold_percent = self.context_threshold
            return True
        self.raw_context_length = parsed_context_length
        effective_context_length, cap, reason = self._effective_context_length(
            parsed_context_length,
            model=model,
            provider=provider,
        )
        self.context_length = effective_context_length
        self.effective_context_length_cap = cap
        self.effective_context_length_reason = reason
        self._context_length_source = source
        self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
            self._runtime_context_threshold(model=model, provider=provider)
        )
        self.threshold_percent = self.context_threshold
        reserve_tokens = max(0, int(self._config.context_reserve_tokens or 0))
        if reserve_tokens > 0:
            context_threshold_tokens = max(
                1,
                effective_context_length - reserve_tokens,
            )
        else:
            context_threshold_tokens = int(
                effective_context_length * self.context_threshold
            )
        self.threshold_tokens = self._effective_threshold_tokens(
            context_threshold_tokens
        )
        return True

    def _session_metadata_matches_active_runtime(
        self,
        kwargs: Dict[str, Any],
        *,
        ignore_empty_optional: bool = False,
    ) -> bool:
        if "model" in kwargs and str(kwargs.get("model") or "") != self.model:
            return False
        for key in ("provider", "base_url", "api_key", "api_mode"):
            if key not in kwargs:
                continue
            incoming = str(kwargs.get(key) or "")
            if ignore_empty_optional and not incoming:
                continue
            if incoming != str(getattr(self, key, "") or ""):
                return False
        return True

    @property
    def name(self) -> str:
        return "lcm"

    @staticmethod
    def _working_set_message_identity(message: Dict[str, Any]) -> str:
        return json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _working_set_required_request_indices(
        self,
        request_messages: List[Dict[str, Any]],
        conversation_messages: Optional[List[Dict[str, Any]]],
    ) -> set[int]:
        """Preserve request-only authority/profile/prefill scaffolding.

        The persisted transcript is reference data. Messages present only in
        the assembled request include the system prompt, policy/profile
        projections, prefill and other host-owned scaffolding, so they are
        mandatory even when their text is not recognized by a heuristic.
        """
        remaining: dict[str, int] = {}
        for message in conversation_messages or []:
            identity = self._working_set_message_identity(message)
            remaining[identity] = remaining.get(identity, 0) + 1
        required: set[int] = set()
        for index, message in enumerate(request_messages):
            if message.get("role") == "system":
                required.add(index)
                continue
            identity = self._working_set_message_identity(message)
            available = remaining.get(identity, 0)
            if available > 0:
                remaining[identity] = available - 1
            else:
                required.add(index)
        return required

    def _working_set_cache_state(self) -> str:
        if self.last_cache_read_tokens > 0 or self.last_cache_write_tokens > 0:
            return "hot"
        if self.cache_metrics_available:
            return "cold"
        return "unknown"

    def _active_replay_preview_policy(self) -> ActiveReplayPreviewPolicy:
        """Return the immutable policy snapshot for the bound session."""
        session_id = str(self._session_id or "")
        existing = self._active_replay_preview_policies.get(session_id)
        if existing is not None:
            return existing
        allowlist = set(self._config.active_replay_preview_session_allowlist)
        prefix_allowlist = tuple(
            self._config.active_replay_preview_session_prefix_allowlist
        )
        allowlisted = (
            not allowlist and not prefix_allowlist
        ) or session_matches_preview_allowlist(
            session_id,
            exact_sessions=allowlist,
            prefix_patterns=prefix_allowlist,
        )
        policy = ActiveReplayPreviewPolicy(
            enabled=bool(self._config.active_replay_preview_enabled)
            and allowlisted,
            version=str(
                self._config.active_replay_preview_policy_version
                or "externalized-tool-preview-v1"
            ),
            min_tokens=int(self._config.active_replay_preview_min_tokens),
            head_tokens=int(self._config.active_replay_preview_head_tokens),
            tail_tokens=int(self._config.active_replay_preview_tail_tokens),
            max_tokens=int(self._config.active_replay_preview_max_tokens),
            fresh_tail_count=int(self._config.fresh_tail_count),
        )
        self._active_replay_preview_policies[session_id] = policy
        while len(self._active_replay_preview_policies) > 1_024:
            self._active_replay_preview_policies.pop(
                next(iter(self._active_replay_preview_policies))
            )
        return policy

    def _record_active_replay_preview(self, decision: Any) -> None:
        self._last_active_replay_preview = {
            "reason": str(decision.reason),
            "policy_version": str(decision.policy_version),
            "eligible_results": int(decision.eligible_results),
            "projected_results": int(decision.projected_results),
            "source_tokens": int(decision.source_tokens),
            "projected_tokens": int(decision.projected_tokens),
            "tokens_saved": int(decision.tokens_saved),
            "refs": list(decision.refs),
        }

    def _record_working_set_decision(self, decision: WorkingSetDecision) -> None:
        self._last_working_set_decision = {
            "reason": decision.reason,
            "source_tokens": decision.source_tokens,
            "selected_tokens": decision.selected_tokens,
            "omitted_tokens": decision.omitted_tokens,
            "source_sha256": decision.source_sha256,
            "omitted_ranges": [list(value) for value in decision.omitted_ranges],
            "cache_state": decision.cache_state,
            "payback_calls": decision.payback_calls,
        }

    def _active_replay_preview_excluded_indices(
        self,
        messages: List[Dict[str, Any]],
    ) -> set[int]:
        """Exclude content altered by existing privacy/ignore policy."""
        excluded: set[int] = set()
        for index, message in enumerate(messages):
            if self._matches_ignore_message_patterns(message):
                excluded.add(index)
                continue
            original = normalize_content_value(message.get("content")) or ""
            redacted = normalize_content_value(
                redact_sensitive_value(
                    message.get("content"),
                    self._config,
                    parse_json_strings=False,
                )
            ) or ""
            if redacted != original:
                excluded.add(index)
        return excluded

    def _ingest_for_working_set(
        self,
        messages: List[Dict[str, Any]],
    ) -> bool:
        if not messages or self._bypasses_lcm_context_management():
            return False
        timeout = max(
            0.1,
            float(self._config.focused_working_set_ingest_timeout_seconds),
        )
        deadline = self._foreground_now() + timeout
        try:
            self._ingest_messages(messages, deadline=deadline)
            return True
        except Exception as exc:
            self._record_ingest_failure("focused working set", exc)
            logger.warning(
                "LCM focused working-set ingest failed; retaining full request",
                exc_info=True,
            )
            return False

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        incoming_message: Optional[Dict[str, Any]] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Select a bounded phase view while keeping raw history lossless."""
        preview_policy = self._active_replay_preview_policy()
        working_set_enabled = bool(self._config.focused_working_set_enabled)
        if not working_set_enabled and not preview_policy.enabled:
            return None
        if self._bypasses_lcm_context_management():
            return None
        if (
            preview_policy.enabled
            and self._config.ignore_message_patterns
            and not self._compiled_ignore_message_patterns
        ):
            # A configured ignore policy that could not be compiled must not
            # become a privacy bypass merely because the optional bounded
            # regex engine is unavailable. Disable only this request-local
            # preview; established working-set behavior remains unchanged.
            preview_policy = ActiveReplayPreviewPolicy(
                enabled=False,
                version=preview_policy.version,
                min_tokens=preview_policy.min_tokens,
                head_tokens=preview_policy.head_tokens,
                tail_tokens=preview_policy.tail_tokens,
                max_tokens=preview_policy.max_tokens,
                fresh_tail_count=preview_policy.fresh_tail_count,
            )
        durable_source = list(conversation_messages or [])
        if isinstance(incoming_message, dict) and incoming_message not in durable_source:
            durable_source.append(incoming_message)
        if durable_source and not self._ingest_for_working_set(durable_source):
            return None

        preview_decision = project_externalized_tool_results(
            request_messages,
            session_id=str(self._session_id or ""),
            config=self._config,
            hermes_home=self._hermes_home,
            policy=preview_policy,
            excluded_result_indices=self._active_replay_preview_excluded_indices(
                request_messages
            ),
        )
        self._record_active_replay_preview(preview_decision)
        projected_messages = preview_decision.messages or request_messages
        if not working_set_enabled:
            return preview_decision.messages

        target = int(self._config.focused_working_set_target_tokens)
        if budget_tokens > 0:
            # Never spend more than half of a model window on replay when the
            # configured target is larger, while retaining a useful floor.
            target = min(target, max(8_000, int(budget_tokens * 0.50)))
        required_indices = self._working_set_required_request_indices(
            projected_messages,
            conversation_messages,
        )
        decision = select_phase_working_set(
            projected_messages,
            target_tokens=target,
            trigger_tokens=max(
                target,
                int(self._config.focused_working_set_trigger_tokens),
            ),
            min_reclaim_tokens=int(
                self._config.focused_working_set_min_reclaim_tokens
            ),
            min_reclaim_ratio=float(
                self._config.focused_working_set_min_reclaim_ratio
            ),
            cache_state=self._working_set_cache_state(),
            expected_remaining_calls=int(
                self._config.focused_working_set_payback_calls
            ),
            required_indices=required_indices,
        )
        self._record_working_set_decision(decision)
        if decision.messages is None:
            return preview_decision.messages
        selected = self._sanitize_active_context_messages(decision.messages)
        if not selected or count_messages_tokens(selected) >= decision.source_tokens:
            self._record_working_set_decision(
                WorkingSetDecision(
                    None,
                    "post_validation_failed",
                    decision.source_tokens,
                    count_messages_tokens(selected),
                    0,
                    decision.source_sha256,
                    decision.omitted_ranges,
                    decision.cache_state,
                    decision.payback_calls,
                )
            )
            return preview_decision.messages
        return selected

    def on_turn_complete(
        self,
        messages: List[Dict[str, Any]],
        usage: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Durably ingest each finalized turn without triggering a model call."""
        if not self._config.focused_working_set_enabled:
            return
        self._ingest_for_working_set(messages)

    @property
    def last_compression_status(self) -> str:
        """Public status for the most recent compression/preflight attempt.

        Host runtimes use this to distinguish a real compaction boundary from
        an LCM no-op (for example, when request pressure is high but all
        compactable raw backlog is protected by the fresh tail).
        """
        return self._last_compression_status

    @property
    def last_compression_noop_reason(self) -> str:
        """Human-readable reason for the latest no-op compression decision."""
        return self._last_compression_noop_reason

    @property
    def last_compression_was_noop(self) -> bool:
        """Whether the most recent compression/preflight decision was a no-op."""
        return self._last_compression_status == "noop"

    def _background_publication_trigger_tokens(self) -> int:
        """Publish before the host admission budget, retaining earlier policy."""
        window = max(0, int(self.context_length or 0))
        reserve = max(0, int(self._config.context_reserve_tokens or 0))
        trigger = max(1, window - reserve) if window > 0 and reserve > 0 else max(0, int(self.threshold_tokens or 0))
        bound = getattr(self, "_admission_publication_tokens", 0)
        if bound:
            trigger = min(trigger or bound, bound, self._effective_assembly_token_cap() or bound)
        return trigger

    def _background_emergency_trigger_tokens(self) -> int:
        """Recover with headroom before assembly and final host admission."""
        window = max(0, int(self.context_length or 0))
        if window <= 0:
            return 0
        trigger = max(1, window - max(32_000, int(window * 0.05)))
        bound = getattr(self, "_admission_emergency_tokens", 0)
        if bound:
            trigger = min(trigger, bound, self._effective_assembly_token_cap() or bound)
        return trigger

    def _background_compaction_enabled(self) -> bool:
        return bool(
            self._config.background_compaction_enabled
            and self._background_compaction is not None
            and not self._background_compaction_compatibility_blocker()
        )

    def _background_compaction_compatibility_blocker(self) -> str:
        """Return why this configuration must retain foreground derivation.

        File/assertion extraction are separate model-data egress paths.  Until
        an operator explicitly adopts an asynchronous policy for those paths,
        keep their existing foreground semantics rather than silently skipping
        them or sending captured content from a new background worker.
        """
        if bool(getattr(self._config, "extraction_enabled", False)):
            return "file_extraction_requires_foreground_compaction"
        if bool(getattr(self._config, "assertions_enabled", False)) and bool(
            getattr(self._config, "assertion_extraction_enabled", False)
        ):
            return "assertion_extraction_requires_foreground_compaction"
        return ""

    def _background_policy_fingerprint(self) -> str:
        return background_policy_fingerprint(self._config)

    def _reconcile_background_publication_fence(
        self,
        lifecycle_state: Any | None = None,
        *,
        log_contention: bool = False,
        deadline: float | None = None,
    ) -> int | None:
        """Activate the currently bound lifecycle epoch with a bounded CAS.

        This is intentionally safe to call again on a later foreground turn.
        The store will not reopen an intentional reset fence for the same
        lifecycle binding, but it can recover from a transient activation lock
        after a newer durable bind.
        """
        if (
            not self._background_compaction_enabled()
            or not self._conversation_id
            or not self._session_id
        ):
            return None
        remaining = self._foreground_deadline_remaining(deadline)
        if deadline is not None and (remaining is None or remaining <= 0):
            raise ForegroundDeadlineExceeded(
                "publication fence",
                "before lifecycle lookup",
            )
        try:
            state = lifecycle_state or self._lifecycle.get_by_conversation(
                self._conversation_id,
                lock_timeout_seconds=remaining,
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                "publication fence",
                "lifecycle lookup exceeded the shared deadline",
            ) from exc
        if (
            state is None
            or state.current_session_id != self._session_id
            or state.current_bound_at is None
        ):
            return None
        try:
            write_timeout_ms = 100
            remaining = self._foreground_deadline_remaining(deadline)
            if remaining is not None:
                write_timeout_ms = min(
                    write_timeout_ms,
                    max(0, int(remaining * 1000)),
                )
            return self._background_compaction.activate_session(
                conversation_id=self._conversation_id,
                session_id=self._session_id,
                expected_lifecycle_bound_at=float(state.current_bound_at),
                expected_lifecycle_reset_at=state.last_reset_at,
                write_timeout_ms=write_timeout_ms,
                lock_timeout_seconds=remaining,
                write_deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                "publication fence",
                "activation lock wait exceeded the shared deadline",
            ) from exc
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            if log_contention:
                logger.warning(
                    "LCM background publication fence activation deferred: %s",
                    exc,
                )
            return None

    def _maybe_enqueue_background_compaction(
        self,
        messages: List[Dict[str, Any]],
        observed_tokens: int | None = None,
        *,
        allow_canonical_prefix_adoption: bool = True,
        deadline: float | None = None,
    ):
        """Capture/coalesce eligible durable raw work without model I/O."""
        if (
            not self._background_compaction_enabled()
            or not self._session_id
            or not self._conversation_id
            or self._call_with_foreground_deadline(
                self._bypasses_lcm_context_management,
                deadline=deadline,
            )
        ):
            return None
        remaining = self._foreground_deadline_remaining(deadline)
        try:
            lifecycle = self._lifecycle.get_by_conversation(
                self._conversation_id,
                lock_timeout_seconds=remaining,
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except TimeoutError:
            return None
        if lifecycle is None or lifecycle.current_session_id != self._session_id:
            return None
        try:
            fence_generation = self._reconcile_background_publication_fence(
                lifecycle,
                deadline=deadline,
            )
        except ForegroundDeadlineExceeded:
            return None
        if fence_generation is None:
            return None
        if allow_canonical_prefix_adoption:
            adopted_frontier = self._maybe_adopt_existing_canonical_prefix(
                lifecycle,
                expected_session_fence_generation=fence_generation,
                deadline=deadline,
            )
            if adopted_frontier > int(lifecycle.current_frontier_store_id or 0):
                # The durable frontier changed atomically.  Do not capture a
                # stale job in this boundary; preflight/compress will rebuild
                # the active view from the canonical DAG before considering
                # the still-uncovered suffix on a later bounded boundary.
                return None
            remaining = self._foreground_deadline_remaining(deadline)
            try:
                latest_lifecycle = self._lifecycle.get_by_conversation(
                    self._conversation_id,
                    lock_timeout_seconds=remaining,
                    deadline=deadline,
                    deadline_clock=self._foreground_now,
                )
            except TimeoutError:
                return None
            if (
                latest_lifecycle is None
                or latest_lifecycle.current_session_id != self._session_id
                or int(latest_lifecycle.current_frontier_store_id or 0)
                != int(lifecycle.current_frontier_store_id or 0)
            ):
                return None
        active_tokens = (
            int(observed_tokens)
            if observed_tokens is not None and int(observed_tokens) >= 0
            else count_messages_tokens(messages)
        )
        try:
            remaining = self._foreground_deadline_remaining(deadline)
            job = self._background_compaction.capture_and_enqueue(
                conversation_id=self._conversation_id,
                session_id=self._session_id,
                expected_frontier_store_id=lifecycle.current_frontier_store_id,
                active_tokens=active_tokens,
                effective_context_window=max(0, int(getattr(self, "_admission_safe_input_tokens", 0) or self.context_length or 0)),
                policy_fingerprint=self._background_policy_fingerprint(),
                lock_timeout_seconds=remaining,
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except TimeoutError:
            return None
        except sqlite3.OperationalError as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning(
                    "LCM background compaction enqueue deferred by SQLite lock contention"
                )
                return None
            raise
        runtime = self._background_runtime_handle
        notify = getattr(runtime, "notify", None)
        if job is not None and callable(notify):
            notify()
        return job

    def _maybe_adopt_existing_canonical_prefix(
        self,
        lifecycle_state: Any | None = None,
        *,
        expected_session_fence_generation: int | None = None,
        deadline: float | None = None,
    ) -> int:
        """Boundedly reconcile legacy canonical coverage above the frontier.

        Older foreground publication used separate DAG/frontier transactions,
        so a crash could leave an already-canonical raw prefix above the
        lifecycle marker.  Advance only across the normalized, contiguous
        prefix before any provider work; the following foreground compression
        boundary adopts that durable frontier into the active context.
        """
        if (
            not self._background_compaction_enabled()
            or not self._conversation_id
            or not self._session_id
        ):
            return 0
        remaining = self._foreground_deadline_remaining(deadline)
        try:
            lifecycle = lifecycle_state or self._lifecycle.get_by_conversation(
                self._conversation_id,
                lock_timeout_seconds=remaining,
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except TimeoutError:
            return 0
        if lifecycle is None or lifecycle.current_session_id != self._session_id:
            return 0
        frontier = int(lifecycle.current_frontier_store_id or 0)
        fence_generation = expected_session_fence_generation
        if fence_generation is None:
            fence_generation = self._reconcile_background_publication_fence(
                lifecycle,
                deadline=deadline,
            )
        if fence_generation is None:
            return frontier
        try:
            remaining = self._foreground_deadline_remaining(deadline)
            write_timeout_ms = 100
            if remaining is not None:
                write_timeout_ms = min(
                    write_timeout_ms,
                    max(0, int(remaining * 1000)),
                )
            return int(
                self._background_compaction.adopt_contiguous_canonical_prefix(
                    conversation_id=self._conversation_id,
                    session_id=self._session_id,
                    expected_frontier_store_id=frontier,
                    expected_session_fence_generation=int(fence_generation),
                    write_timeout_ms=write_timeout_ms,
                    lock_timeout_seconds=remaining,
                    write_deadline=deadline,
                    deadline_clock=self._foreground_now,
                )
                or frontier
            )
        except TimeoutError:
            return frontier
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            logger.warning(
                "LCM canonical-prefix adoption deferred by SQLite lock contention"
            )
            return frontier

    def _has_publishable_background_job(
        self,
        *,
        deadline: float | None = None,
    ) -> bool:
        if not self._background_compaction_enabled() or not self._conversation_id:
            return False
        remaining = self._foreground_deadline_remaining(deadline)
        lifecycle = self._lifecycle.get_by_conversation(
            self._conversation_id,
            lock_timeout_seconds=remaining,
            deadline=deadline,
            deadline_clock=self._foreground_now,
        )
        expected_frontier = (
            lifecycle.current_frontier_store_id if lifecycle is not None else None
        )
        return self._background_compaction.find_prepared_job(
            conversation_id=self._conversation_id,
            session_id=self._session_id,
            expected_frontier_store_id=expected_frontier,
            policy_fingerprint=self._background_policy_fingerprint(),
            lock_timeout_seconds=remaining,
            deadline=deadline,
            deadline_clock=self._foreground_now,
        ) is not None

    def _has_newer_committed_background_frontier(
        self,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Return whether this process still needs to adopt a sibling commit."""
        if not self._background_compaction_enabled() or not self._conversation_id:
            return False
        remaining = self._foreground_deadline_remaining(deadline)
        lifecycle = self._lifecycle.get_by_conversation(
            self._conversation_id,
            lock_timeout_seconds=remaining,
            deadline=deadline,
            deadline_clock=self._foreground_now,
            busy_timeout_ms=(
                max(0, int(remaining * 1000))
                if remaining is not None
                else None
            ),
        )
        return bool(
            lifecycle is not None
            and lifecycle.current_session_id == self._session_id
            and int(lifecycle.current_frontier_store_id or 0)
            > int(self._last_compacted_store_id or 0)
        )

    def _all_durable_messages_after(
        self,
        store_id: int,
        *,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Page the complete durable suffix; MessageStore defaults to 10k rows."""
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "durable tail scan",
        )
        rows: List[Dict[str, Any]] = []
        cursor = max(0, int(store_id or 0))
        while True:
            self._foreground_deadline_checkpoint(
                deadline,
                "context reconstruction",
                "durable tail page",
            )
            try:
                page = self._store.get_session_messages_after(
                    self._session_id,
                    cursor,
                    limit=1_000,
                    conversation_id=self._conversation_id,
                    lock_timeout_seconds=(
                        self._foreground_deadline_remaining(deadline)
                    ),
                    deadline=deadline,
                    deadline_clock=self._foreground_now,
                )
            except TimeoutError as exc:
                raise ForegroundDeadlineExceeded(
                    "context reconstruction",
                    "durable tail read exceeded the shared deadline",
                ) from exc
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_locked_error(exc):
                    raise
                raise ForegroundDeadlineExceeded(
                    "context reconstruction",
                    "SQLite contention prevented a bounded durable tail read",
                ) from exc
            self._foreground_deadline_checkpoint(
                deadline,
                "context reconstruction",
                "durable tail page",
            )
            if not page:
                break
            rows.extend(page)
            next_cursor = max(int(row.get("store_id") or 0) for row in page)
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(page) < 1_000:
                break
        return rows

    def _durable_replay_duplicate_aliases(
        self,
        durable_rows: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> dict[int, int]:
        """Map duplicate durable rows to the earliest row with their identity."""
        lookup = getattr(self._store, "duplicate_store_id_aliases", None)
        if not durable_rows or not self._session_id or not callable(lookup):
            return {}
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "durable duplicate scan",
        )
        try:
            aliases = lookup(
                self._session_id,
                durable_rows,
                deadline=deadline,
                deadline_clock=self._foreground_now,
                lock_timeout_seconds=self._foreground_deadline_remaining(deadline),
            )
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                "context reconstruction",
                "durable duplicate scan exceeded the shared deadline",
            ) from exc
        except sqlite3.OperationalError as exc:
            if _is_sqlite_locked_error(exc):
                raise ForegroundDeadlineExceeded(
                    "context reconstruction",
                    "SQLite contention prevented the durable duplicate scan",
                ) from exc
            logger.warning("LCM durable duplicate scan failed", exc_info=True)
            return {}
        if aliases:
            logger.info(
                "LCM replay skipped %d duplicate stored message(s): session=%s",
                len(aliases),
                self._session_id,
            )
        return aliases

    def _project_durable_active_tail(
        self,
        store_id: int,
        *,
        preserve_messages: List[Dict[str, Any]] | None = None,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Return the complete replay-safe suffix above ``store_id``.

        Durable rows define ordering and make sibling-process writes visible.
        When the caller also has an in-memory active view, overlay matching raw
        rows with those exact objects so provider-only fields, structured media,
        and already-established replay representations are not degraded by a
        storage round-trip.
        """
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "durable replay projection",
        )
        durable_rows = self._call_with_foreground_deadline(
            self._all_durable_messages_after,
            store_id,
            deadline=deadline,
        )
        durable_suffix_store_ids = {
            int(row.get("store_id") or 0) for row in durable_rows
        }
        # Older builds could re-insert already-stored host messages after a
        # restart or active prefix rewrite. Replay each message once, from its
        # earliest row; a copy of a row at/below the frontier is already
        # covered by the summary DAG and is dropped entirely.
        duplicate_aliases = self._durable_replay_duplicate_aliases(
            durable_rows,
            deadline=deadline,
        )
        if duplicate_aliases:
            durable_rows = [
                row
                for row in durable_rows
                if int(row.get("store_id") or 0) not in duplicate_aliases
            ]
        replay: List[Dict[str, Any]] = []
        for row_index, row in enumerate(durable_rows):
            if row_index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "context reconstruction",
                    "durable replay conversion",
                )
            message = self._store.to_openai_msg(row)
            if row.get("observed_at") is not None:
                # Keep the host timestamp: it is the replay identity that
                # later foreground lineage (interval summaries, ingest dedupe)
                # uses to locate this message's stored rows.
                message["timestamp"] = row.get("observed_at")
            message["_lcm_replay_store_id"] = int(row.get("store_id") or 0)
            replay.append(message)
        projected = self._call_foreground_projection(
            project_active_replay_messages,
            replay,
            deadline=deadline,
            stage="durable replay projection",
            config=self._config,
            session_id=self._session_id,
            hermes_home=self._hermes_home,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "durable replay projection",
        )
        if not preserve_messages:
            for message_index, message in enumerate(projected):
                if message_index % 32 == 0:
                    self._foreground_deadline_checkpoint(
                        deadline,
                        "context reconstruction",
                        "durable replay cleanup",
                    )
                message.pop("_lcm_replay_store_id", None)
            return projected

        leading_count = (
            1
            if preserve_messages and preserve_messages[0].get("role") == "system"
            else 0
        )
        active_tail = preserve_messages[leading_count:]
        ids_by_message_id = self._get_store_id_map_for_messages(
            active_tail,
            deadline=deadline,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "active replay source mapping",
        )
        exact_by_store_id: dict[int, Dict[str, Any]] = {}
        last_mapped_active_index = -1
        for active_index, message in enumerate(active_tail):
            if active_index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "context reconstruction",
                    "active replay source overlay",
                )
            message_store_id = ids_by_message_id.get(id(message))
            if message_store_id is None:
                continue
            last_mapped_active_index = active_index
            message_store_id = duplicate_aliases.get(
                int(message_store_id),
                int(message_store_id),
            )
            if int(message_store_id) > int(store_id or 0):
                exact_by_store_id[int(message_store_id)] = dict(message)
        merged: List[Dict[str, Any]] = []
        represented_store_ids: set[int] = set()
        for projected_index, durable_message in enumerate(projected):
            if projected_index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "context reconstruction",
                    "durable replay merge",
                )
            durable_store_id = int(
                durable_message.pop("_lcm_replay_store_id", 0) or 0
            )
            represented_store_ids.add(durable_store_id)
            exact_message = exact_by_store_id.get(durable_store_id)
            merged.append(
                dict(exact_message) if exact_message is not None else durable_message
            )

        # A normal publication/adoption follows ingestion, so every live raw
        # message should map durably.  Preserve any genuinely new unmapped
        # caller suffix defensively, while excluding synthetic context scaffolds
        # that the freshly assembled DAG prefix will replace.
        for index, message in enumerate(active_tail):
            if index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "context reconstruction",
                    "active replay suffix merge",
                )
            mapped_store_id = ids_by_message_id.get(id(message))
            if mapped_store_id is not None:
                mapped_store_id = int(mapped_store_id)
                if mapped_store_id in represented_store_ids:
                    continue
                if mapped_store_id <= int(store_id or 0):
                    continue
                if mapped_store_id in durable_suffix_store_ids:
                    # The durable projector considered this row.  If it is not
                    # represented above, replay policy deliberately filtered it
                    # (for example, an ignored turn or its dependent reply).
                    # Never resurrect or move that row behind later kept rows.
                    continue
            elif index <= last_mapped_active_index:
                # Unmapped ignore/quarantine placeholders can sit inside the
                # consumed range.  Their position does not prove they are a
                # new suffix, so fail closed instead of moving them beyond the
                # published frontier.  Only caller objects after every mapped
                # durable row are demonstrably newer.
                continue
            if (
                message.get("display_kind") == "context_projection"
                or self._is_replayed_context_scaffold_message(message)
            ):
                continue
            merged.append(dict(message))
        return merged

    def _adopt_committed_background_frontier(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]] | None:
        """Adopt a sibling process's committed frontier at a foreground boundary."""
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "frontier adoption",
        )
        if not self._background_compaction_enabled() or not self._conversation_id:
            return None
        lifecycle = self._lifecycle.get_by_conversation(
            self._conversation_id,
            lock_timeout_seconds=self._foreground_deadline_remaining(deadline),
            deadline=deadline,
            deadline_clock=self._foreground_now,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "frontier lookup",
        )
        if (
            lifecycle is None
            or lifecycle.current_session_id != self._session_id
            or int(lifecycle.current_frontier_store_id or 0)
            <= int(self._last_compacted_store_id or 0)
        ):
            return None
        committed_frontier = int(lifecycle.current_frontier_store_id or 0)
        tail_messages = self._call_with_foreground_deadline(
            self._project_durable_active_tail,
            committed_frontier,
            preserve_messages=messages,
            deadline=deadline,
        )
        leading = (
            dict(messages[0])
            if messages and messages[0].get("role") == "system"
            else None
        )
        assembled = self._call_with_foreground_deadline(
            self._assemble_context,
            leading,
            tail_messages,
            deadline=deadline,
        )
        assembled = self._call_foreground_projection(
            self._sanitize_active_context_messages,
            assembled,
            deadline=deadline,
            stage="frontier adoption cleanup",
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "frontier adoption finalization",
        )
        self._last_compacted_store_id = committed_frontier
        self._align_ingest_cursor_to_active_context(
            assembled,
            deadline=deadline,
        )
        return assembled

    def run_background_compaction_once(self, summarizer=None):
        """Claim and prepare one durable job (deterministic tests/operators)."""
        if not self._background_compaction_enabled():
            return None
        if self._session_id and self._conversation_id:
            self._maybe_enqueue_background_compaction(
                self._store.get_session_messages(self._session_id),
                observed_tokens=self.last_prompt_tokens,
            )
        owner = f"manual-{os.getpid()}-{threading.get_ident()}"
        job = self._background_compaction.run_next_job(
            owner=owner,
            summarizer=summarizer,
            circuit_breaker=self._summary_circuit_breaker,
            spend_guard=self._summary_spend_guard,
        )
        return job

    def promote_prepared_compaction(
        self,
        job_id: str,
        messages: List[Dict[str, Any]],
        *,
        current_tokens: int | None = None,
        deadline: float | None = None,
    ) -> PromotionResult:
        """Atomically publish one prepared range, then assemble a new list."""
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "prepared publication entry",
        )
        if not self._background_compaction_enabled():
            return PromotionResult(
                promoted=False,
                reason="background_compaction_disabled",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        remaining = self._foreground_deadline_remaining(deadline)
        try:
            scoped_job = self._background_compaction.get_job(
                str(job_id or ""),
                lock_timeout_seconds=remaining,
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except (TimeoutError, sqlite3.OperationalError) as exc:
            if isinstance(exc, sqlite3.OperationalError) and not _is_sqlite_locked_error(
                exc
            ):
                raise
            return PromotionResult(
                promoted=False,
                reason="publication_inspection_deferred",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        if scoped_job is None:
            return PromotionResult(
                promoted=False,
                reason="job_not_found",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        if (
            scoped_job.conversation_id != self._conversation_id
            or scoped_job.session_id != self._session_id
        ):
            return PromotionResult(
                promoted=False,
                reason="job_scope_mismatch",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        observed_tokens = (
            max(0, int(current_tokens))
            if current_tokens is not None
            else max(0, int(count_messages_tokens(messages)))
        )
        publication_trigger = self._background_publication_trigger_tokens()
        if publication_trigger <= 0 or observed_tokens < publication_trigger:
            return PromotionResult(
                promoted=False,
                reason="publication_trigger_not_reached",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        owner = f"publish-{os.getpid()}-{threading.get_ident()}"
        remaining = self._foreground_deadline_remaining(deadline)
        write_timeout_ms = 100
        if remaining is not None:
            write_timeout_ms = min(
                write_timeout_ms,
                max(0, int(remaining * 1000)),
            )
        try:
            lease = self._background_compaction.acquire_publication_lease(
                job_id,
                owner=owner,
                write_timeout_ms=write_timeout_ms,
                lock_timeout_seconds=remaining,
                write_deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except (TimeoutError, sqlite3.OperationalError) as exc:
            if isinstance(exc, sqlite3.OperationalError) and not _is_sqlite_locked_error(
                exc
            ):
                raise
            return PromotionResult(
                promoted=False,
                reason="publication_lease_deferred",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        if lease is None:
            return PromotionResult(
                promoted=False,
                reason="publication_lease_unavailable",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        remaining = self._foreground_deadline_remaining(deadline)
        write_timeout_ms = 100
        if remaining is not None:
            write_timeout_ms = min(
                write_timeout_ms,
                max(0, int(remaining * 1000)),
            )
        try:
            result = self._background_compaction.publish_prepared(
                job_id,
                owner=owner,
                generation=lease.lease_generation,
                policy_fingerprint=self._background_policy_fingerprint(),
                write_timeout_ms=write_timeout_ms,
                observed_active_tokens=observed_tokens,
                effective_context_window=max(0, int(getattr(self, "_admission_safe_input_tokens", 0) or self.context_length or 0)),
                publication_trigger_tokens=publication_trigger,
                emergency_trigger_tokens=self._background_emergency_trigger_tokens(),
                trigger_policy=(
                    "fixed_reserve"
                    if int(self.context_length or 0) > 0
                    and int(self._config.context_reserve_tokens or 0) > 0
                    else "threshold"
                ),
                lock_timeout_seconds=remaining,
                write_deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except (TimeoutError, sqlite3.OperationalError) as exc:
            if isinstance(exc, sqlite3.OperationalError) and not _is_sqlite_locked_error(
                exc
            ):
                raise
            return PromotionResult(
                promoted=False,
                reason="prepared_publication_deferred",
                job_id=str(job_id or ""),
                messages=list(messages),
            )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "prepared publication commit",
        )
        if not result.promoted:
            result.messages = list(messages)
            return result

        committed_frontier = int(
            result.captured_watermark_store_id or lease.captured_watermark_store_id
        )
        try:
            tail_messages = self._call_with_foreground_deadline(
                self._project_durable_active_tail,
                committed_frontier,
                preserve_messages=messages,
                deadline=deadline,
            )
            leading = (
                dict(messages[0])
                if messages and messages[0].get("role") == "system"
                else None
            )
            assembled = self._call_with_foreground_deadline(
                self._assemble_context,
                leading,
                tail_messages,
                deadline=deadline,
            )
            assembled = self._call_foreground_projection(
                self._sanitize_active_context_messages,
                assembled,
                deadline=deadline,
                stage="prepared publication cleanup",
            )
            self._foreground_deadline_checkpoint(
                deadline,
                "context reconstruction",
                "prepared publication assembly",
            )
        except ForegroundDeadlineExceeded:
            raise
        except Exception:
            # The SQLite publication/frontier CAS is already committed and is
            # authoritative.  Do not turn a local replay/assembly fault into an
            # apparent failed publication, and do not advance the process-local
            # marker: the next foreground boundary can adopt the durable
            # frontier again.  Returning the caller view preserves the current
            # turn without deleting or duplicating raw storage.
            logger.exception(
                "LCM published background compaction but deferred local context rebuild"
            )
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "published context rebuild deferred"
            result.reason = "published_context_rebuild_deferred"
            result.messages = list(messages)
            return result
        self._last_compacted_store_id = committed_frontier
        self._align_ingest_cursor_to_active_context(
            assembled,
            deadline=deadline,
        )
        self.compression_count += 1
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = ""
        result.messages = assembled
        return result

    def _publish_prepared_background_compaction(
        self,
        messages: List[Dict[str, Any]],
        *,
        current_tokens: int,
        deadline: float | None = None,
    ) -> PromotionResult | None:
        if not self._background_compaction_enabled():
            return None
        job = self._background_compaction.find_prepared_job(
            conversation_id=self._conversation_id,
            session_id=self._session_id,
            expected_frontier_store_id=self._last_compacted_store_id,
            policy_fingerprint=self._background_policy_fingerprint(),
            lock_timeout_seconds=self._foreground_deadline_remaining(deadline),
            deadline=deadline,
            deadline_clock=self._foreground_now,
        )
        if job is None:
            return None
        return self.promote_prepared_compaction(
            job.job_id,
            messages,
            current_tokens=current_tokens,
            deadline=deadline,
        )

    def _supersede_background_jobs_after_foreground(self) -> int:
        if not self._background_compaction_enabled() or not self._conversation_id:
            return 0
        return self._background_compaction.supersede_overlapping(
            conversation_id=self._conversation_id,
            new_frontier_store_id=self._last_compacted_store_id,
            reason="foreground_emergency_compaction",
        )

    def _publish_foreground_background_node(
        self,
        node: SummaryNode,
        *,
        expected_session_fence_generation: int,
        expected_frontier_store_id: int,
        consumed_store_ids: List[int],
        expected_source_identity_hash: str,
        deadline: float | None = None,
    ) -> PromotionResult:
        """Atomically fence a foreground node against prepared publishers."""
        if not self._background_compaction_enabled() or not consumed_store_ids:
            return PromotionResult(False, "background_compaction_disabled")
        self._foreground_deadline_checkpoint(
            deadline,
            "foreground publication",
            "before durable publication",
        )
        remaining = self._foreground_deadline_remaining(deadline)
        write_timeout_ms = 100
        sqlite_wait_uses_deadline = False
        if remaining is not None:
            if remaining <= 0:
                raise ForegroundDeadlineExceeded(
                    "foreground publication",
                    "before durable publication",
                )
            sqlite_wait_uses_deadline = remaining <= (write_timeout_ms / 1000.0)
            write_timeout_ms = min(write_timeout_ms, max(0, int(remaining * 1000)))
        watermark = max(int(store_id) for store_id in consumed_store_ids)
        sqlite_wait_started = self._foreground_now()
        try:
            result = self._background_compaction.publish_foreground_node(
                node,
                conversation_id=self._conversation_id,
                session_id=self._session_id,
                expected_session_fence_generation=expected_session_fence_generation,
                expected_frontier_store_id=expected_frontier_store_id,
                captured_watermark_store_id=watermark,
                expected_source_identity_hash=expected_source_identity_hash,
                debt_tokens=int(node.source_token_count or 0),
                reason="foreground_emergency_compaction",
                write_timeout_ms=write_timeout_ms,
                lock_timeout_seconds=remaining,
                write_deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                "foreground publication",
                "durable publication lock wait exceeded the shared deadline",
            ) from exc
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            sqlite_wait_finished = self._foreground_now()
            deadline_wait_exhausted = bool(
                deadline is not None and sqlite_wait_finished >= float(deadline)
            )
            if sqlite_wait_uses_deadline and remaining is not None:
                # SQLite accepts only whole-millisecond busy timeouts. Treat a
                # real wait that reaches that rounded boundary as exhausting
                # the shared wall, while preserving the existing immediate
                # contention result for callers that still have time to react.
                deadline_wait_exhausted = deadline_wait_exhausted or (
                    sqlite_wait_finished - sqlite_wait_started
                    >= max(0.0, remaining - 0.002)
                )
            if deadline_wait_exhausted:
                raise ForegroundDeadlineExceeded(
                    "foreground publication",
                    "SQLite wait consumed the shared deadline",
                ) from exc
            logger.warning(
                "LCM foreground emergency publication deferred by SQLite lock contention"
            )
            try:
                self._background_compaction.record_diagnostic_event(
                    "sqlite_lock_contention",
                    conversation_id=self._conversation_id,
                    session_id=self._session_id,
                    detail=exc,
                    deduplicate_seconds=60.0,
                    # Diagnostics are optional. Never spend the caller's
                    # remaining wall waiting on the same contended writer or
                    # this helper's process-local serialization lock.
                    write_timeout_ms=0,
                    lock_timeout_seconds=0.0,
                    write_deadline=deadline,
                    deadline_clock=self._foreground_now,
                )
            except (sqlite3.OperationalError, TimeoutError):
                # Contention diagnostics must never extend the bounded
                # foreground path merely because the same writer is busy.
                pass
            return PromotionResult(False, "sqlite_lock_contention")
        if result.promoted:
            node.node_id = result.node_ids[0]
            self._last_compacted_store_id = watermark
        return result

    def _publish_foreground_dag_node(
        self,
        node: SummaryNode,
        *,
        deadline: float | None,
        stage: str,
        frontier_store_id: int | None = None,
    ) -> int:
        """Publish one legacy DAG node without outwaiting the foreground wall.

        A disabled-background leaf also advances its lifecycle frontier in the
        same SQLite transaction.  Holding the lifecycle store's process-local
        lock across that transaction preserves its serialization contract while
        eliminating the old node-then-marker partial-publication window.
        """

        lifecycle_lock = None
        lifecycle_lock_acquired = False
        before_commit = None
        if frontier_store_id is not None:
            lifecycle_lock = self._lifecycle._lock
            if deadline is None:
                lifecycle_lock_acquired = bool(lifecycle_lock.acquire())
            else:
                remaining = self._foreground_deadline_remaining(deadline)
                if remaining is None or remaining <= 0:
                    raise ForegroundDeadlineExceeded(
                        stage,
                        "before lifecycle frontier serialization",
                    )
                lifecycle_lock_acquired = bool(
                    lifecycle_lock.acquire(timeout=max(0.0, remaining))
                )
            if not lifecycle_lock_acquired:
                raise ForegroundDeadlineExceeded(
                    stage,
                    "lifecycle frontier lock wait exceeded the shared deadline",
                )

            def advance_frontier_in_publication(
                conn: sqlite3.Connection,
            ) -> None:
                self._foreground_deadline_checkpoint(
                    deadline,
                    stage,
                    "before lifecycle frontier publication",
                )
                conn.execute(
                    """
                    UPDATE lcm_lifecycle_state
                    SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                        updated_at = ?
                    WHERE conversation_id = ? AND current_session_id = ?
                    """,
                    (
                        max(0, int(frontier_store_id)),
                        time.time(),
                        self._conversation_id,
                        self._session_id,
                    ),
                )

            before_commit = advance_frontier_in_publication

        try:
            if deadline is None:
                return self._dag.add_node(
                    node,
                    before_commit=before_commit,
                )

            self._foreground_deadline_checkpoint(
                deadline,
                stage,
                "before DAG publication",
            )
            remaining = self._foreground_deadline_remaining(deadline)
            if remaining is None or remaining <= 0:
                raise ForegroundDeadlineExceeded(stage, "before DAG publication")
            node_id = self._dag.add_node(
                node,
                lock_timeout_seconds=remaining,
                write_timeout_ms=max(0, int(remaining * 1000)),
                write_deadline=deadline,
                deadline_clock=self._foreground_now,
                before_commit=before_commit,
            )
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                stage,
                "DAG publication lock wait exceeded the shared deadline",
            ) from exc
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            raise ForegroundDeadlineExceeded(
                stage,
                "SQLite wait prevented DAG publication inside the shared deadline",
            ) from exc
        finally:
            if lifecycle_lock_acquired and lifecycle_lock is not None:
                lifecycle_lock.release()
        return node_id

    def get_background_compaction_status(self) -> Dict[str, Any]:
        """Return durable worker/job status, or an inert disabled view."""
        compatibility_blocker = self._background_compaction_compatibility_blocker()
        configured_enabled = bool(
            getattr(self._config, "background_compaction_enabled", False)
        )
        helper = getattr(self, "_background_compaction", None)
        if helper is not None:
            status = helper.status(
                conversation_id=str(self.current_conversation_id or ""),
                publication_trigger_tokens=self._background_publication_trigger_tokens(),
                emergency_trigger_tokens=self._background_emergency_trigger_tokens(),
            )
        else:
            status = {
                "enabled": configured_enabled and not compatibility_blocker,
                "worker_count": int(
                    getattr(self._config, "background_compaction_workers", 1)
                ),
                "global_limit": int(
                    getattr(self._config, "background_compaction_global_limit", 2)
                ),
                "queued_jobs": 0,
                "running_jobs": 0,
                "prepared_jobs": 0,
                "retry_jobs": 0,
                "published_jobs": 0,
                "superseded_jobs": 0,
                "failed_jobs": 0,
                "oldest_queued_job_age_seconds": None,
                "current_job": None,
                "current_debt_tokens": 0,
                "last_preparation_result": None,
                "last_publication_result": None,
                "durable_frontier_store_id": 0,
                "active_coverage": "raw",
            }
        status["publication_trigger_tokens"] = (
            self._background_publication_trigger_tokens()
        )
        status["emergency_trigger_tokens"] = (
            self._background_emergency_trigger_tokens()
        )
        status["configured_enabled"] = configured_enabled
        if compatibility_blocker:
            status["enabled"] = False
            status["worker_count"] = 0
            status["worker_available"] = False
            status["compatibility_blocked_reason"] = compatibility_blocker
        durable_frontier = max(
            0,
            int(status.get("durable_frontier_store_id") or 0),
        )
        status["active_coverage"] = (
            "published_summary"
            if durable_frontier > 0
            and int(self._last_compacted_store_id or 0) >= durable_frontier
            else "raw"
        )
        return status

    def _mark_preflight_compression_requested(self) -> bool:
        """Record that preflight found work and clear any stale no-op reason."""
        self._last_compression_status = "pending"
        self._last_compression_noop_reason = ""
        return True

    @property
    def bound_session_id(self) -> str:
        """Session id this engine is actively servicing for ingest/lifecycle.

        This differs from ``current_session_id`` while a side-channel session is
        bound but operator-facing tools should keep showing the foreground
        session. Host lifecycle hooks must compare against this value before
        deciding whether a post-turn ingest needs to rebind the engine.
        """
        return self._session_id

    @property
    def current_session_id(self) -> str:
        """User-facing "current session" id surfaced by LCM tools.

        Returns the most recent foreground binding (the last session id that
        ``_refresh_session_filters`` classified as neither ignored nor
        stateless). Falls back to ``_session_id`` when no foreground has
        ever been bound, so unattended cron-only or stateless-only processes
        remain observable via ``lcm_status``.

        Lifecycle paths (compress, ingest, on_session_end, etc.) must keep
        reading ``_session_id`` directly because those paths must follow the
        binding the engine is actually servicing. Only tool-surface code
        paths that report a "current session" view to operators should read
        this property.
        """
        return self._foreground_session_id or self._session_id

    @property
    def current_session_platform(self) -> str:
        """Platform string paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_session_platform
        return self._session_platform

    @property
    def current_conversation_id(self) -> str:
        """Conversation id paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_conversation_id
        return self._conversation_id

    @property
    def side_channel_active(self) -> bool:
        """True when an ignored or stateless session has temporarily rebound
        ``_session_id`` while a real foreground binding still exists.

        Operators reading lcm_status during this window see the foreground
        session id and counts (because tools read ``current_session_id``)
        but the engine itself is servicing the side channel. This predicate
        lets diagnostic surfaces (lcm_status, /lcm command) make the
        divergence explicit without recomputing the underlying invariant.
        """
        return bool(self._foreground_session_id) and self._foreground_session_id != self._session_id

    @property
    def current_session_ignored(self) -> bool:
        """``_session_ignored`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-ignored; otherwise this is the bound session's ignore flag.
        """
        if self.side_channel_active:
            return False
        return self._session_ignored

    @property
    def current_session_stateless(self) -> bool:
        """``_session_stateless`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-stateless; otherwise this is the bound session's stateless flag.
        """
        if self.side_channel_active:
            return False
        return self._session_stateless

    # -- ContextEngine required methods ------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._thread_context_stateless():
            auxiliary_session_id = self._thread_context_session_id()
            if auxiliary_session_id:
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                caller_generation = self._in_process_auxiliary_caller_generation(
                    auxiliary_session_id
                )
                with self._auxiliary_session_lock:
                    if auxiliary_session_id not in self._auxiliary_session_ids:
                        return
                    if self._auxiliary_generation_is_retired(
                        auxiliary_session_id,
                        caller_generation,
                    ):
                        return
                    active_generation = self._auxiliary_session_generations.get(
                        auxiliary_session_id
                    )
                    if active_generation is None and caller_generation:
                        expected_parent = self._auxiliary_handoff_parent_session_ids.get(
                            auxiliary_session_id
                        )
                        if expected_parent and self._in_process_parent_session_id(
                            {},
                            session_id=auxiliary_session_id,
                            include_explicit=False,
                        ) != expected_parent:
                            return
                        if auxiliary_session_id in self._auxiliary_last_prompt_tokens:
                            self._auxiliary_direct_end_guard_session_ids.add(
                                auxiliary_session_id
                            )
                        self._auxiliary_last_prompt_tokens.pop(auxiliary_session_id, None)
                        if self._host_fallback_session_id == auxiliary_session_id:
                            self._end_host_fallback_compressor_for_session(
                                auxiliary_session_id,
                                [],
                                current_session_bypasses=True,
                            )
                        self._auxiliary_session_generations[
                            auxiliary_session_id
                        ] = caller_generation
                        active_generation = caller_generation
                    stack = self._thread_context_auxiliary_stack()
                    stack_marks_current_session = bool(
                        active_generation is not None
                        and caller_generation == 0
                        and stack
                        and stack[-1] == auxiliary_session_id
                    )
                    generation_matches = (
                        caller_generation == 0
                        if active_generation is None
                        else caller_generation == active_generation or stack_marks_current_session
                    )
                    if generation_matches:
                        self._auxiliary_last_prompt_tokens[auxiliary_session_id] = prompt_tokens
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        cache_keys = {"cache_read_tokens", "cache_write_tokens"}
        self.cache_metrics_available = any(key in usage for key in cache_keys)
        self.last_input_tokens = int(usage.get("input_tokens", self.last_prompt_tokens) or 0)
        self.last_output_tokens = int(
            usage.get("output_tokens", self.last_completion_tokens) or 0
        )
        self.last_cache_read_tokens = int(usage.get("cache_read_tokens", 0) or 0)
        self.last_cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
        self.last_reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
        self._record_turn_compaction_telemetry()

    @property
    def cache_read_ratio(self) -> float:
        if self.last_prompt_tokens <= 0:
            return 0.0
        return self.last_cache_read_tokens / self.last_prompt_tokens

    def _record_turn_compaction_telemetry(self) -> None:
        """Persist a per-conversation compaction-telemetry snapshot for this turn.

        Best-effort and diagnostic only: any failure is logged at debug and never
        affects the turn. Turns with no token or cache signal are skipped so idle
        turns do not churn the record. The since-compaction accumulators reset off
        the monotonic ``compression_count`` (which also drops to 0 on a session
        reset) rather than instrumenting the compaction hot path.
        """
        conversation_id = self._conversation_id
        if not conversation_id:
            return
        prompt_tokens = self.last_prompt_tokens
        cache_read = self.last_cache_read_tokens
        cache_write = self.last_cache_write_tokens
        if (
            prompt_tokens <= 0
            and cache_read <= 0
            and cache_write <= 0
            and not self.cache_metrics_available
        ):
            return
        try:
            existing = self._store.read_compaction_telemetry(conversation_id) or {}

            if cache_read > 0 or cache_write > 0:
                cache_state = "hot"
            elif self.cache_metrics_available:
                cache_state = "cold"
            else:
                cache_state = "unknown"
            cold_streak = int(existing.get("consecutive_cold_observations", 0) or 0)
            if cache_state == "hot":
                cold_streak = 0
            elif cache_state == "cold":
                cold_streak += 1

            prev_count = int(existing.get("compression_count_at_record", 0) or 0)
            compacted = self.compression_count > prev_count
            rebaselined = self.compression_count != prev_count  # compaction or session reset
            if rebaselined:
                turns_since = 0
                peak_tokens_since = prompt_tokens
            else:
                turns_since = int(existing.get("turns_since_leaf_compaction", 0) or 0) + 1
                peak_tokens_since = max(
                    int(existing.get("peak_prompt_tokens_since_leaf_compaction", 0) or 0),
                    prompt_tokens,
                )
            total_compactions = int(existing.get("total_compactions", 0) or 0)
            if compacted:
                total_compactions += self.compression_count - prev_count
                last_leaf_compaction_at = time.time()
                last_compaction_duration_ms = round(self._last_compaction_duration_ms, 3)
            else:
                last_leaf_compaction_at = existing.get("last_leaf_compaction_at")
                last_compaction_duration_ms = existing.get("last_compaction_duration_ms")

            record = dict(existing)
            record.update({
                "conversation_id": conversation_id,
                "last_observed_prompt_tokens": prompt_tokens,
                "last_observed_cache_read": cache_read,
                "last_observed_cache_write": cache_write,
                "cache_state": cache_state,
                "consecutive_cold_observations": cold_streak,
                "turns_since_leaf_compaction": turns_since,
                "peak_prompt_tokens_since_leaf_compaction": peak_tokens_since,
                # Reserved carry-forward field; no live 'medium'/'high' computation yet.
                "activity_band": existing.get("activity_band", "low"),
                "provider": self.provider or existing.get("provider"),
                "model": self.model or existing.get("model"),
                "last_api_call_at": time.time(),
                "last_leaf_compaction_at": last_leaf_compaction_at,
                "last_compaction_duration_ms": last_compaction_duration_ms,
                "total_compactions": total_compactions,
                "compression_count_at_record": self.compression_count,
            })
            if cache_state == "hot":
                record["last_cache_hit_at"] = time.time()
            self._store.write_compaction_telemetry(conversation_id, record)
        except Exception:
            logger.debug("LCM compaction telemetry update failed", exc_info=True)

    def _compression_boundary_cooldown_active(self) -> bool:
        """Return true while a boundary skip is in its short no-compress window."""
        if self._last_boundary_skip_time <= 0:
            return False
        elapsed = time.time() - self._last_boundary_skip_time
        if elapsed < 60:
            logger.debug(
                "LCM compression cooldown active: %.1f seconds since boundary skip",
                elapsed,
            )
            return True
        self._last_boundary_skip_time = 0
        return False

    def _record_ingest_success(self) -> None:
        self._consecutive_ingest_failures = 0

    def _record_ingest_failure(self, where: str, error: Exception) -> None:
        """Track a swallowed ingest error so it is operator-visible.

        Escalates to error level once failures are consecutive: a single
        transient lock is a warning, but a sustained inability to persist
        means the lossless guarantee is broken and must not stay hidden.
        """
        self._ingest_failure_count += 1
        self._consecutive_ingest_failures += 1
        self._last_ingest_error = f"{type(error).__name__}: {error}"
        self._last_ingest_error_time = time.time()
        message = "LCM ingest failed (%s): %s [consecutive=%d, total=%d]"
        args = (
            where,
            error,
            self._consecutive_ingest_failures,
            self._ingest_failure_count,
        )
        if self._consecutive_ingest_failures >= 3:
            logger.error(message, *args)
        else:
            logger.warning(message, *args)

    def _clear_foreground_rebind_candidate(self) -> None:
        self._foreground_rebind_session_id = ""
        self._foreground_rebind_previous_session_id = ""
        self._foreground_rebind_previous_platform = ""
        self._foreground_rebind_previous_conversation_id = ""
        self._foreground_rebind_parent_session_id = ""

    def _clear_foreground_rebind_candidate_if_bound_session_confirmed(self) -> None:
        if self._foreground_rebind_session_id == self._session_id:
            self._clear_foreground_rebind_candidate()

    def _session_has_foreground_branch_marker(
        self,
        session_id: str,
        parent_session_id: str,
    ) -> bool:
        """Return True when Hermes state.db records ``session_id`` as a real branch.

        An un-ingested foreground branch and a provisional late-marked auxiliary
        child can both expose the same in-process parent id before either has
        durable LCM rows. The canonical host signal for real foreground branches
        is the ``sessions.model_config._branched_from`` marker in state.db; use
        it to decide whether a displaced un-ingested candidate is safe to restore.
        """
        session_id = str(session_id or "")
        parent_session_id = str(parent_session_id or "")
        if not session_id or not parent_session_id:
            return False
        path = self._state_db_path()
        if not path.exists():
            return False
        try:
            uri = path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT parent_session_id, model_config
                    FROM sessions
                    WHERE id = ?
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            logger.debug("LCM foreground branch marker probe failed", exc_info=True)
            return False
        if not row:
            return False
        row_parent_id = str(row[0] or "")
        if row_parent_id != parent_session_id:
            return False
        try:
            model_config = json.loads(row[1] or "{}")
        except (TypeError, ValueError):
            return False
        if not isinstance(model_config, dict):
            return False
        return str(model_config.get("_branched_from") or "") == parent_session_id

    def _remember_foreground_rebind_candidate(self, session_id: str) -> None:
        """Remember the foreground view displaced by a provisional normal bind.

        Bind-time auxiliary detection can miss background-review children when
        the host seeds its marker late. If that happens, ``on_session_start``
        temporarily classifies the child as a normal foreground and overwrites
        the operator-facing foreground pointer. The first-ingest recheck may
        later prove the child is auxiliary; this snapshot lets that recovery
        restore the parent foreground instead of falling back to the child.
        """
        session_id = str(session_id or "")
        if not session_id:
            self._clear_foreground_rebind_candidate()
            return
        if self._foreground_rebind_session_id == session_id:
            return
        if (
            self._foreground_rebind_session_id
            and self._foreground_rebind_previous_session_id
            and self._foreground_session_id == self._foreground_rebind_session_id
        ):
            parent_session_id = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
                require_auxiliary_frame=False,
            )
            rebind_candidate_is_foreground_branch = bool(
                self._foreground_rebind_parent_session_id
                and self._session_has_foreground_branch_marker(
                    self._foreground_rebind_session_id,
                    self._foreground_rebind_parent_session_id,
                )
            )
            if (
                (
                    parent_session_id == self._foreground_rebind_session_id
                    and (
                        not self._foreground_rebind_parent_session_id
                        or rebind_candidate_is_foreground_branch
                    )
                )
                or (parent_session_id and not self._foreground_rebind_parent_session_id)
                or (
                    parent_session_id
                    and parent_session_id == self._foreground_rebind_parent_session_id
                    and rebind_candidate_is_foreground_branch
                )
            ):
                self._foreground_rebind_previous_session_id = self._foreground_session_id
                self._foreground_rebind_previous_platform = self._foreground_session_platform
                self._foreground_rebind_previous_conversation_id = self._foreground_conversation_id
            self._foreground_rebind_session_id = session_id
            self._foreground_rebind_parent_session_id = parent_session_id
            return
        if self._foreground_session_id and self._foreground_session_id != session_id:
            parent_session_id = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
                require_auxiliary_frame=False,
            )
            self._foreground_rebind_session_id = session_id
            self._foreground_rebind_previous_session_id = self._foreground_session_id
            self._foreground_rebind_previous_platform = self._foreground_session_platform
            self._foreground_rebind_previous_conversation_id = self._foreground_conversation_id
            self._foreground_rebind_parent_session_id = parent_session_id
            return
        self._clear_foreground_rebind_candidate()

    def _restore_foreground_after_late_auxiliary_reclassification(self, session_id: str) -> None:
        if self._foreground_session_id != session_id:
            if self._foreground_rebind_session_id == session_id:
                self._clear_foreground_rebind_candidate()
            return
        if (
            self._foreground_rebind_session_id == session_id
            and self._foreground_rebind_previous_session_id
        ):
            parent_session_id = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            if (
                parent_session_id
                and parent_session_id != session_id
                and parent_session_id == self._foreground_rebind_previous_session_id
                and self._lcm_session_last_normal_conversation_id.get(parent_session_id)
            ):
                self._foreground_session_id = parent_session_id
                self._foreground_session_platform = self._lcm_session_last_normal_platform.get(
                    parent_session_id,
                    self._foreground_rebind_previous_platform,
                )
                self._foreground_conversation_id = self._lcm_session_last_normal_conversation_id[
                    parent_session_id
                ]
            else:
                self._foreground_session_id = self._foreground_rebind_previous_session_id
                self._foreground_session_platform = self._foreground_rebind_previous_platform
                self._foreground_conversation_id = self._foreground_rebind_previous_conversation_id
        else:
            self._foreground_session_id = ""
            self._foreground_session_platform = ""
            self._foreground_conversation_id = ""
        self._clear_foreground_rebind_candidate()

    def _maybe_reclassify_current_session_as_auxiliary_before_message_ingest(
        self,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Defense-in-depth for host markers that arrive after session binding.

        Older Hermes Agent background-review forks seed ``_memory_write_origin``
        too late for ``on_session_start`` frame-walk detection. By the first
        message-writing entry point the marker is present on the running agent
        frame, so re-check only while the bound session is still empty. That
        keeps normal foreground session starts/resets writable and avoids
        reclassifying sessions after real data has already been stored.
        """
        session_id = str(self._session_id or "")
        if not session_id:
            return False
        if (
            self._session_ignored
            or self._session_stateless
            or self._thread_context_stateless(
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        ):
            return False
        if self._ingest_cursor > 0:
            return False
        try:
            stored_count = self._store.get_session_count(
                session_id,
                deadline=deadline,
                deadline_clock=self._foreground_now,
                busy_timeout_ms=0 if deadline is not None else None,
            )
        except TimeoutError:
            raise
        except Exception:
            logger.debug("LCM first-ingest auxiliary recheck count probe failed", exc_info=True)
            return False
        if stored_count != 0:
            return False
        if not self._in_process_auxiliary_caller_generation(
            session_id,
            deadline=deadline,
            deadline_clock=self._foreground_now,
        ):
            return False

        self._foreground_deadline_checkpoint(
            deadline,
            "session classification",
            "before auxiliary state mutation",
        )
        self._mark_thread_context_stateless(
            session_id,
            deadline=deadline,
            deadline_clock=self._foreground_now,
        )
        self._restore_foreground_after_late_auxiliary_reclassification(session_id)
        logger.info(
            "LCM reclassified session %s as auxiliary at first ingest after bind-time detection missed",
            session_id,
        )
        return True

    def ingest(self, messages: List[Dict[str, Any]]) -> None:
        """Persist messages to the durable store every turn.

        Called by the post_llm_call plugin hook so messages land in LCM
        regardless of whether compression triggers — short WebUI
        conversations never hit the compression threshold and never
        expire like Telegram sessions do, so without this they'd never
        be ingested.

        Uses the same _ingest_messages cursor as compress(), so if
        compression runs later the same turn, already-ingested messages
        are skipped (no duplicates).
        """
        if self._maybe_reclassify_current_session_as_auxiliary_before_message_ingest():
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            return
        if self._bypasses_lcm_context_management():
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            return
        if self._session_id and messages:
            try:
                self._remember_lcm_normal_message_prefix(
                    self._session_id,
                    messages,
                    conversation_id=self._conversation_id,
                )
                self._ingest_messages(messages)
                self._record_ingest_success()
                self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            except Exception as e:
                self._record_ingest_failure("per-turn ingest()", e)
                return
            try:
                self._maybe_enqueue_background_compaction(
                    messages,
                    observed_tokens=max(
                        int(self.last_prompt_tokens or 0),
                        count_messages_tokens(messages),
                    ),
                )
            except Exception as e:
                # Immutable raw ingest already committed successfully.  Queue
                # maintenance is opportunistic and must not corrupt ingest
                # health accounting or cause the host to retry this raw batch.
                logger.warning(
                    "LCM background compaction enqueue failed after raw ingest: %s",
                    e,
                )
            logger.debug(
                "Per-turn ingest OK: session=%s msgs=%d cursor=%d",
                self._session_id, len(messages), self._ingest_cursor,
            )

    def _is_retry_worthy_leaf_summary_error(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).lower()
        retry_markers = (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "input too long",
            "request too large",
            "timed out",
            "timeout",
        )
        return any(marker in message for marker in retry_markers)

    def _next_leaf_rescue_chunk(
        self,
        current_chunk: List[Dict[str, Any]],
        current_source_tokens: int,
    ) -> List[Dict[str, Any]]:
        if len(current_chunk) <= 1:
            return []

        floor_tokens = max(1, self._config.leaf_chunk_tokens)
        shrink_targets = [
            max(floor_tokens, int(current_source_tokens * 0.75)),
            max(floor_tokens, int(current_source_tokens * 0.50)),
        ]

        for target in shrink_targets:
            if target >= current_source_tokens:
                continue
            smaller = self._select_oldest_leaf_chunk(current_chunk, target)
            if smaller and len(smaller) < len(current_chunk):
                return smaller

        return current_chunk[:-1]

    def _acquire_foreground_emergency_provider_permit(
        self,
        *,
        deadline: float,
        expected_session_fence_generation: int,
    ) -> ProviderPermit:
        """Reserve durable provider capacity immediately before thread start."""
        state = getattr(
            self._thread_context,
            "foreground_emergency_provider_permit",
            None,
        )
        if state is not None:
            if state.get("guardian_owned"):
                completed = state.get("completed")
                if not isinstance(completed, threading.Event) or not completed.is_set():
                    raise ForegroundProviderPermitUnavailable(
                        "foreground emergency provider call is still physically running"
                    )
                try:
                    del self._thread_context.foreground_emergency_provider_permit
                except AttributeError:
                    pass
            else:
                return state["permit"]
        helper = self._background_compaction
        if helper is None or not self._conversation_id:
            raise ForegroundProviderPermitUnavailable(
                "foreground emergency provider permit is unavailable"
            )
        # The first lease covers the complete foreground wall plus a short
        # publication cushion.  If the physical call outlives that wall, the
        # guardian begins heartbeating before this reservation can expire.
        remaining = max(0.0, float(deadline) - self._foreground_now())
        if remaining <= 0:
            raise ForegroundProviderPermitUnavailable(
                "foreground emergency provider deadline is exhausted"
            )
        lease_seconds = max(
            float(
                getattr(
                    self._config,
                    "background_compaction_lease_seconds",
                    300,
                )
                or 300
            ),
            (remaining * 2.0) + 5.0,
        )
        owner = f"foreground-{os.getpid()}-{threading.get_ident()}"
        acquire_kwargs: dict[str, Any] = {
            "conversation_id": self._conversation_id,
            "owner": owner,
            "session_id": self._session_id,
            "expected_session_fence_generation": expected_session_fence_generation,
            "lease_seconds": lease_seconds,
            "global_limit": int(
                getattr(
                    self._config,
                    "background_compaction_global_limit",
                    2,
                )
                or 0
            ),
        }
        if self._summary_callable_accepts_keyword(
            helper.acquire_emergency_provider_permit,
            "lock_timeout_seconds",
        ):
            acquire_kwargs["lock_timeout_seconds"] = remaining
        if self._summary_callable_accepts_keyword(
            helper.acquire_emergency_provider_permit,
            "write_deadline",
        ):
            acquire_kwargs["write_deadline"] = deadline
        if self._summary_callable_accepts_keyword(
            helper.acquire_emergency_provider_permit,
            "deadline_clock",
        ):
            acquire_kwargs["deadline_clock"] = self._foreground_now
        if self._summary_callable_accepts_keyword(
            helper.acquire_emergency_provider_permit,
            "overlap_background",
        ):
            # A running background preparation must not starve the bounded
            # emergency path; the publication CAS arbitrates the two results.
            acquire_kwargs["overlap_background"] = True
        try:
            permit = helper.acquire_emergency_provider_permit(**acquire_kwargs)
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            permit = None
        except TimeoutError:
            permit = None
        if permit is not None and self._foreground_now() >= float(deadline):
            completed = threading.Event()
            completed.set()
            helper._start_provider_permit_guardian(
                completed,
                job_id=permit.job_id,
                owner=permit.lease_owner,
                generation=permit.lease_generation,
            )
            permit = None
        if permit is None:
            raise ForegroundProviderPermitUnavailable(
                "foreground emergency provider capacity is unavailable"
            )
        self._thread_context.foreground_emergency_provider_permit = {
            "permit": permit,
            "permit_store": helper,
            "guardian_owned": False,
            "completion_lock": threading.Lock(),
            "provider_finished": False,
            "start_failure_finalizer": None,
            "start_failure_claimed": False,
            "completed": None,
        }
        return permit

    @staticmethod
    def _prepare_foreground_provider_attempt(
        state: dict[str, Any],
        completed: threading.Event,
    ) -> dict[str, Any]:
        """Create race state owned by exactly one physical provider attempt."""
        attempt = {
            "completion_lock": threading.Lock(),
            "provider_finished": False,
            "start_failure_finalizer": None,
            "start_failure_claimed": False,
            "completed": completed,
        }
        with state["completion_lock"]:
            state["active_attempt"] = attempt
            state["completed"] = completed
        return attempt

    @staticmethod
    def _mark_foreground_provider_finished(state: dict[str, Any]) -> None:
        """Run a guardian-start fallback only after the provider returns."""
        finalizer = None
        with state["completion_lock"]:
            state["provider_finished"] = True
            if (
                state.get("start_failure_finalizer") is not None
                and not state.get("start_failure_claimed")
            ):
                state["start_failure_claimed"] = True
                finalizer = state["start_failure_finalizer"]
        if finalizer is not None:
            finalizer()

    @staticmethod
    def _arm_foreground_guardian_start_failure(
        state: dict[str, Any],
        finalizer: Callable[[], None],
    ) -> None:
        """Transfer durable release when the guardian thread cannot start."""
        run_now = False
        with state["completion_lock"]:
            state["start_failure_finalizer"] = finalizer
            if state.get("provider_finished") and not state.get(
                "start_failure_claimed"
            ):
                state["start_failure_claimed"] = True
                run_now = True
        if run_now:
            finalizer()

    def _handoff_foreground_emergency_provider_permit(
        self,
        completed: threading.Event,
    ) -> None:
        """Keep a timed-out physical call fenced until it actually returns."""
        state = getattr(
            self._thread_context,
            "foreground_emergency_provider_permit",
            None,
        )
        if state is None or state.get("guardian_owned"):
            return
        permit: ProviderPermit = state["permit"]
        state["guardian_owned"] = True
        attempt_state = state.get("active_attempt") or state
        helper = state.get("permit_store") or self._background_compaction
        if helper is None:
            return
        helper._start_provider_permit_guardian(
            completed,
            job_id=permit.job_id,
            owner=permit.lease_owner,
            generation=permit.lease_generation,
            on_start_failure=lambda finalizer: (
                self._arm_foreground_guardian_start_failure(
                    attempt_state,
                    finalizer,
                )
            ),
        )

    def _release_foreground_emergency_provider_permit(
        self,
        *,
        deadline: float | None = None,
    ) -> None:
        """Release this foreground thread's permit after publication or failure."""
        state = getattr(
            self._thread_context,
            "foreground_emergency_provider_permit",
            None,
        )
        if state is None:
            return
        try:
            del self._thread_context.foreground_emergency_provider_permit
        except AttributeError:
            pass
        if state.get("guardian_owned"):
            return
        permit: ProviderPermit = state["permit"]
        helper = state.get("permit_store") or self._background_compaction
        if helper is None:
            return
        completed = threading.Event()
        completed.set()
        if deadline is not None and self._foreground_now() >= float(deadline):
            helper._start_provider_permit_guardian(
                completed,
                job_id=permit.job_id,
                owner=permit.lease_owner,
                generation=permit.lease_generation,
            )
            return
        try:
            release_kwargs: dict[str, Any] = {
                "owner": permit.lease_owner,
                "generation": permit.lease_generation,
                "busy_timeout_ms": 0,
            }
            if self._summary_callable_accepts_keyword(
                helper.release_provider_permit,
                "lock_timeout_seconds",
            ):
                release_kwargs["lock_timeout_seconds"] = 0.0
            if self._summary_callable_accepts_keyword(
                helper.release_provider_permit,
                "write_deadline",
            ):
                release_kwargs["write_deadline"] = deadline
            if self._summary_callable_accepts_keyword(
                helper.release_provider_permit,
                "deadline_clock",
            ):
                release_kwargs["deadline_clock"] = self._foreground_now
            helper.release_provider_permit(permit.job_id, **release_kwargs)
        except BaseException:
            logger.warning(
                "LCM deferred release of a foreground emergency provider permit",
                exc_info=True,
            )
            helper._start_provider_permit_guardian(
                completed,
                job_id=permit.job_id,
                owner=permit.lease_owner,
                generation=permit.lease_generation,
            )

    def _summarize_leaf_chunk_with_rescue(
        self,
        initial_chunk: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
        deadline: Optional[float] = None,
        require_durable_provider_permit: bool = False,
        expected_session_fence_generation: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int, str, int, int]:
        attempt_chunk = list(initial_chunk)
        max_attempts = 3
        attempt_number = 0

        while attempt_chunk and attempt_number < max_attempts:
            self._foreground_deadline_checkpoint(
                deadline,
                "leaf token counting",
                "before token count",
            )
            attempt_number += 1
            source_tokens = count_messages_tokens(attempt_chunk)
            self._foreground_deadline_checkpoint(
                deadline,
                "leaf token counting",
                "after token count",
            )
            self._foreground_deadline_checkpoint(
                deadline,
                "leaf serialization",
                "before serialization",
            )
            serialized = self._call_with_foreground_deadline(
                self._serialize_messages,
                attempt_chunk,
                deadline=deadline,
            )
            self._foreground_deadline_checkpoint(
                deadline,
                "leaf serialization",
                "after serialization",
            )
            token_budget = max(2000, int(source_tokens * 0.20))
            token_budget = min(token_budget, 12000)

            try:
                timeout_seconds = self._config.summary_timeout_ms / 1000
                if deadline is not None:
                    remaining_seconds = deadline - self._foreground_now()
                    if remaining_seconds <= 0:
                        raise TimeoutError("threshold full sweep time budget exhausted")
                    timeout_seconds = min(timeout_seconds, remaining_seconds)
                call_kwargs = {
                    "text": serialized,
                    "source_tokens": source_tokens,
                    "token_budget": token_budget,
                    "depth": 0,
                    "model": self._config.summary_model,
                    "fallback_models": self._config.summary_fallback_models,
                    "circuit_breaker": self._summary_circuit_breaker,
                    "spend_guard": self._summary_spend_guard,
                    "timeout": timeout_seconds,
                    "l2_budget_ratio": self._config.l2_budget_ratio,
                    "l3_truncate_tokens": self._config.l3_truncate_tokens,
                    "focus_topic": focus_topic or "",
                    "custom_instructions": self._config.custom_instructions,
                }
                provider_cancel_event = threading.Event()
                if deadline is not None and self._summary_callable_accepts_keyword(
                    summarize_with_escalation,
                    "deadline",
                ):
                    call_kwargs["deadline"] = deadline
                if deadline is not None and self._summary_callable_accepts_keyword(
                    summarize_with_escalation,
                    "cancel_event",
                ):
                    call_kwargs["cancel_event"] = provider_cancel_event
                hard_call_slot = self._foreground_hard_call_slot
                if not hard_call_slot.acquire(blocking=False):
                    provider_cancel_event.set()
                    raise ForegroundProviderPermitUnavailable(
                        "another foreground provider call is still physically running"
                    )
                if deadline is None:
                    try:
                        summary_text, level = summarize_with_escalation(**call_kwargs)
                    finally:
                        hard_call_slot.release()
                else:
                    completed = threading.Event()
                    result_box: dict[str, Any] = {}
                    foreground_permit_state: dict[str, Any] | None = None
                    foreground_attempt_state: dict[str, Any] | None = None

                    if require_durable_provider_permit:
                        if expected_session_fence_generation is None:
                            hard_call_slot.release()
                            raise ForegroundProviderPermitUnavailable(
                                "foreground emergency session fence is unavailable"
                            )
                        try:
                            self._acquire_foreground_emergency_provider_permit(
                                deadline=deadline,
                                expected_session_fence_generation=(
                                    expected_session_fence_generation
                                ),
                            )
                            foreground_permit_state = getattr(
                                self._thread_context,
                                "foreground_emergency_provider_permit",
                                None,
                            )
                            if foreground_permit_state is None:
                                raise ForegroundProviderPermitUnavailable(
                                    "foreground emergency provider state is unavailable"
                                )
                            foreground_attempt_state = (
                                self._prepare_foreground_provider_attempt(
                                    foreground_permit_state,
                                    completed,
                                )
                            )
                        except BaseException:
                            hard_call_slot.release()
                            self._release_foreground_emergency_provider_permit()
                            raise

                    def call_provider() -> None:
                        try:
                            if (
                                provider_cancel_event.is_set()
                                or self._foreground_now() >= float(deadline)
                            ):
                                result_box["cancelled"] = True
                                return
                            result_box["value"] = summarize_with_escalation(
                                **call_kwargs
                            )
                        except BaseException as provider_error:
                            result_box["error"] = provider_error
                        finally:
                            hard_call_slot.release()
                            completed.set()
                            if foreground_attempt_state is not None:
                                self._mark_foreground_provider_finished(
                                    foreground_attempt_state
                                )

                    provider_thread = threading.Thread(
                        target=call_provider,
                        name="lcm-foreground-summary-deadline",
                        daemon=True,
                    )
                    try:
                        provider_thread.start()
                    except BaseException:
                        hard_call_slot.release()
                        if require_durable_provider_permit:
                            self._release_foreground_emergency_provider_permit()
                        raise
                    try:
                        finished = completed.wait(
                            timeout=max(
                                0.0,
                                float(deadline) - self._foreground_now(),
                            )
                        )
                    except BaseException:
                        provider_cancel_event.set()
                        if require_durable_provider_permit and not completed.is_set():
                            self._handoff_foreground_emergency_provider_permit(
                                completed
                            )
                        raise
                    if not finished or self._foreground_now() >= float(deadline):
                        provider_cancel_event.set()
                        if require_durable_provider_permit:
                            self._handoff_foreground_emergency_provider_permit(
                                completed
                            )
                        raise TimeoutError(
                            "foreground compaction provider deadline exceeded"
                        )
                    if result_box.get("cancelled"):
                        raise TimeoutError(
                            "foreground compaction provider cancelled before start"
                        )
                    if "error" in result_box:
                        raise result_box["error"]
                    summary_text, level = result_box["value"]
                return attempt_chunk, source_tokens, summary_text, level, attempt_number
            except Exception as exc:
                if (
                    deadline is not None
                    and self._foreground_now() >= float(deadline)
                ):
                    raise
                if attempt_number >= max_attempts or not self._is_retry_worthy_leaf_summary_error(exc):
                    raise
                smaller_chunk = self._next_leaf_rescue_chunk(attempt_chunk, source_tokens)
                self._foreground_deadline_checkpoint(
                    deadline,
                    "leaf rescue",
                    "after retry selection",
                )
                if not smaller_chunk or len(smaller_chunk) >= len(attempt_chunk):
                    raise
                logger.warning(
                    "LCM leaf summarization retrying with smaller oldest chunk after retry-worthy failure: %s (attempt %d/%d, %d→%d messages)",
                    exc,
                    attempt_number,
                    max_attempts,
                    len(attempt_chunk),
                    len(smaller_chunk),
                )
                attempt_chunk = smaller_chunk

        raise RuntimeError("adaptive leaf rescue exhausted without a valid chunk")

    # -- ContextEngine optional methods ------------------------------------

    def _rollup_maintenance_key(self, scope: str) -> tuple[str, str]:
        raw_database_path = str(self._dag.db_path)
        if raw_database_path == ":memory:":
            database_identity = f":memory:{id(self._dag)}"
        else:
            database_identity = str(Path(raw_database_path).resolve())
        return database_identity, str(scope)

    def try_acquire_rollup_operator_lease(
        self,
        scope: str,
    ) -> tuple[str, str] | None:
        """Reserve this database/scope for a synchronous rollup rebuild."""
        key = self._rollup_maintenance_key(scope)
        if not _ROLLUP_MAINTENANCE_SCHEDULER.try_acquire_exclusive(key):
            return None
        return key

    def release_rollup_operator_lease(self, key: tuple[str, str]) -> None:
        _ROLLUP_MAINTENANCE_SCHEDULER.release_exclusive(key)

    def _schedule_rollup_maintenance(self, scope: str) -> None:
        """Enqueue one best-effort rollup pass using private SQLite helpers."""
        try:
            raw_database_path = str(self._dag.db_path)
            if raw_database_path == ":memory:":
                logger.warning(
                    "LCM cannot run background temporal rollup maintenance for "
                    "an isolated in-memory SQLite database; maintenance skipped"
                )
                return
            database_path = Path(raw_database_path).resolve()
            key = self._rollup_maintenance_key(scope)
            config = copy.deepcopy(self._config)
            circuit_breaker = self._summary_circuit_breaker
            spend_guard = self._summary_spend_guard

            def maintain() -> None:
                private_dag = SummaryDAG(database_path)
                try:
                    run_rollup_maintenance(
                        private_dag,
                        config,
                        scope,
                        circuit_breaker=circuit_breaker,
                        spend_guard=spend_guard,
                    )
                finally:
                    private_dag.close()

            _ROLLUP_MAINTENANCE_SCHEDULER.schedule(
                key,
                maintain,
                owner=self._rollup_maintenance_owner,
            )
        except Exception:
            # Maintenance is opportunistic; a scheduler/setup failure must never
            # turn a successful foreground session bind into a host failure.
            logger.warning(
                "LCM could not schedule background temporal rollup maintenance",
                exc_info=True,
            )

    def drain_rollup_maintenance(self, timeout: float | None = None) -> bool:
        """Wait for rollup jobs scheduled by this engine (tests and diagnostics)."""
        return _ROLLUP_MAINTENANCE_SCHEDULER.drain_owner(
            self._rollup_maintenance_owner,
            timeout=timeout,
        )

    def _bind_lifecycle_state(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        state = self._lifecycle.bind_session(session_id, conversation_id=conversation_id)
        self._conversation_id = state.conversation_id
        self._lcm_session_last_conversation_id[session_id] = state.conversation_id
        self._last_compacted_store_id = state.current_frontier_store_id
        self._register_active_engine_binding()
        if not self._session_ignored and not self._session_stateless:
            self._remember_foreground_rebind_candidate(session_id)
            self._lcm_session_last_normal_conversation_id[session_id] = state.conversation_id
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
            self._foreground_conversation_id = state.conversation_id

        if self._background_compaction_enabled():
            self._reconcile_background_publication_fence(
                state,
                log_contention=True,
            )

        # Garbage-collect empty lifecycle rows when the table exceeds threshold.
        # Gateway restarts, ephemeral cron ticks, and crash-loops all create
        # lifecycle rows that never ingest data — prune them here so they
        # don't accumulate forever.
        if (
            self._config.empty_lifecycle_gc_enabled
            and self._lifecycle.row_count() > self._config.empty_lifecycle_gc_threshold
        ):
            protected = {str(self._session_id)} if self._session_id else None
            max_age = self._config.empty_lifecycle_gc_max_age_hours
            try:
                deleted = self._lifecycle.prune_empty_sessions(
                    protected_session_ids=protected,
                    max_age_hours=max_age,
                )
            except Exception:
                deleted = 0
            if deleted:
                logger.info(
                    "LCM pruned %d lifecycle rows with zero stored data "
                    "(table exceeded threshold of %d rows)",
                    deleted,
                    self._config.empty_lifecycle_gc_threshold,
                )
        # Bypassed/stateless sessions skip every LCM write, so they must also skip
        # rollup maintenance — otherwise the bind-time hook would build rollups for
        # a session whose ingest is suppressed (maintainer #388: gate on the same
        # not-bypassed condition the ingest write uses).
        if (
            self._config.temporal_rollups_enabled
            and not self._session_ignored
            and not self._session_stateless
        ):
            self._schedule_rollup_maintenance(session_id)

    def _invalidate_rollups_for_published_node(
        self,
        node: "SummaryNode",
        *,
        deadline: float | None = None,
    ) -> None:
        """Stale the rollups covering EVERY UTC day a just-published node spans.

        Rollups consume published summary nodes, so publication — not raw ingest
        — is the load-bearing staleness signal (maintainer #388 blocker 1). This
        is called after every ``_dag.add_node`` on the engine so a later summary
        cannot leave an older rollup ``ready`` and apparently current. The node's
        ``earliest_at``/``latest_at`` coverage span is passed through so a summary
        crossing midnight stales BOTH days, not only its newest (maintainer #388
        blocker 2 / B2).
        """
        if not self._config.temporal_rollups_enabled:
            return
        if deadline is not None:
            # SummaryDAG's mutation trigger wrote the invalidation event in the
            # node-publication transaction.  Recent-rollup reads already refuse
            # ready rows while this outbox has debt, so synchronous draining is
            # optional maintenance and must not outwait a foreground caller.
            self._foreground_deadline_checkpoint(
                deadline,
                "rollup invalidation",
                "after durable invalidation enqueue",
            )
            return
        mark_stale_for_published_summary(
            self._dag,
            str(node.session_id or ""),
            node.latest_at,
            node.created_at,
            earliest_at=node.earliest_at,
        )

    def _register_active_engine_binding(self) -> None:
        session_id = str(self._session_id or "")
        conversation_id = str(self._conversation_id or "")
        if not session_id:
            return
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(
                self,
                keep_session_id=session_id,
                keep_conversation_id=conversation_id,
            )
            _ACTIVE_ENGINES_BY_SESSION_ID[session_id] = self
            if conversation_id:
                _ACTIVE_ENGINES_BY_CONVERSATION_ID[conversation_id] = self

    def _unregister_active_engine_binding(self) -> None:
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(self)

    def _persist_frontier_marker(self) -> None:
        if not self._session_id or not self._conversation_id:
            return
        self._lifecycle.advance_frontier(
            self._conversation_id,
            self._session_id,
            self._last_compacted_store_id,
        )

    def _has_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> bool:
        with self._auxiliary_session_lock:
            if session_id not in self._lcm_bypass_lineage_session_ids:
                return False
            if platform is None:
                return True
            platforms = self._lcm_bypass_lineage_platforms.get(session_id) or set()
            return not platforms or platform in platforms

    def _mark_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> None:
        if not session_id:
            return
        platform = self._session_platform if platform is None else str(platform or "")
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.add(session_id)
            self._lcm_bypass_lineage_platforms.setdefault(session_id, set()).add(platform)
            self._lcm_session_last_platform[session_id] = platform
            self._lcm_session_last_bypassed[session_id] = True

    def _unmark_lcm_bypass_lineage_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.discard(session_id)
            self._lcm_bypass_lineage_platforms.pop(session_id, None)

    def _handoff_lcm_bypass_lineage(
        self,
        old_session_id: str,
        new_session_id: str,
        *,
        new_platform: str = "",
    ) -> None:
        with self._auxiliary_session_lock:
            if old_session_id:
                self._lcm_bypass_lineage_session_ids.add(old_session_id)
            if new_session_id:
                new_platform = str(new_platform or "")
                self._lcm_bypass_lineage_session_ids.add(new_session_id)
                self._lcm_bypass_lineage_platforms.setdefault(new_session_id, set()).add(new_platform)
                self._lcm_session_last_platform[new_session_id] = new_platform
                self._lcm_session_last_bypassed[new_session_id] = True

    def _compression_boundary_from_lcm_bypassed_session(self, old_session_id: str) -> bool:
        if not old_session_id:
            return False
        if old_session_id in self._lcm_session_last_bypassed:
            return bool(self._lcm_session_last_bypassed.get(old_session_id))
        if old_session_id == self._session_id:
            return bool(
                self._bypasses_lcm_context_management()
                or self._session_id_matches_lcm_bypass_filters(
                    old_session_id,
                    platform=self._session_platform,
                )
            )
        return bool(
            self._has_lcm_bypass_lineage_session(old_session_id)
            or self._session_id_matches_lcm_bypass_filters(old_session_id)
        )

    def _get_allowed_hermes_base(self) -> Path | None:
        """Get the allowed base directory for hermes_home, or None if not restricted."""
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            return Path(env_base).expanduser().resolve()
        return None  # No restriction when env var not set

    def _state_db_path(self, kwargs: Dict[str, Any] | None = None) -> Path:
        kwargs = kwargs or {}
        hermes_home = str(kwargs.get("hermes_home") or self._hermes_home or "")
        if hermes_home:
            return _enforce_state_db_containment(
                Path(hermes_home) / "state.db",
                description=f"hermes_home {hermes_home}",
            )
        db_path = Path(self._store.db_path)
        return _enforce_state_db_containment(
            db_path.parent / "state.db",
            description=f"state database fallback from LCM database {db_path}",
        )

    def _clear_pending_reset_boundary(self) -> None:
        self._pending_reset_session_id = ""
        self._pending_reset_conversation_id = ""
        self._pending_reset_frontier_store_id = 0

    def _finalize_pending_reset_boundary(self, session_id: str) -> None:
        if not self._pending_reset_session_id:
            return
        if self._pending_reset_session_id != session_id:
            self._clear_pending_reset_boundary()
            return
        if not self._pending_reset_conversation_id:
            self._clear_pending_reset_boundary()
            return
        state = self._lifecycle.get_by_conversation(self._pending_reset_conversation_id)
        frontier_store_id = self._pending_reset_frontier_store_id
        if state is not None and state.current_session_id == session_id:
            frontier_store_id = max(
                frontier_store_id,
                int(state.current_frontier_store_id or 0),
            )
        self._lifecycle.finalize_session(
            self._pending_reset_conversation_id,
            self._pending_reset_session_id,
            frontier_store_id=frontier_store_id,
        )
        self._clear_pending_reset_boundary()

    def _raw_backlog_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fresh_tail_start = self._fresh_tail_start(messages)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return []
        return messages[leading_anchor_count:fresh_tail_start]

    def _fresh_tail_boundary(self, messages: List[Dict[str, Any]]) -> FreshTailBoundary:
        return resolve_fresh_tail_boundary(
            messages,
            fresh_tail_count=self._config.fresh_tail_count,
            fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
        )

    def _fresh_tail_start(self, messages: List[Dict[str, Any]]) -> int:
        return self._fresh_tail_boundary(messages).start

    def _get_session_fresh_tail(
        self,
        session_id: str,
        *,
        minimum_count: int = 0,
    ) -> tuple[List[Dict[str, Any]], FreshTailBoundary]:
        """Load and resolve a stored tail, expanding backward for tool pairing."""
        total_count = int(self._store.get_session_count(session_id))
        configured_count = max(minimum_count, int(self._config.fresh_tail_count or 0))
        if self._config.fresh_tail_max_tokens > 0:
            configured_count = max(1, configured_count)
        if total_count <= 0 or configured_count <= 0:
            return [], resolve_fresh_tail_boundary(
                [],
                fresh_tail_count=configured_count,
                fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
            )

        load_limit = min(total_count, configured_count)
        while True:
            rows = self._store.get_session_tail(session_id, load_limit)
            boundary = resolve_fresh_tail_boundary(
                rows,
                fresh_tail_count=configured_count,
                fresh_tail_max_tokens=self._config.fresh_tail_max_tokens,
            )
            selected = rows[boundary.start:]
            unresolved_tool_boundary = bool(
                selected
                and selected[0].get("role") == "tool"
                and not boundary.tool_group_extended
                and load_limit < total_count
            )
            if not unresolved_tool_boundary:
                return selected, boundary
            load_limit = min(total_count, max(load_limit + 1, load_limit * 2))

    @staticmethod
    def _leading_anchor_count(messages: List[Dict[str, Any]]) -> int:
        """Return the number of non-compactable leading messages.

        Only the system prompt is a safe permanent anchor. Hermes gateway
        sessions can begin with a user message when core passes conversation
        history without a system prompt; preserving that first user turn as raw
        active context lets stale requests look current after later compaction.
        """
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return 1
        return 0

    def _raw_backlog_tokens(self, messages: List[Dict[str, Any]]) -> int:
        backlog = self._raw_backlog_messages(messages)
        if not backlog:
            return 0
        return count_messages_tokens(backlog)

    def _raw_backlog_threshold(self, raw_tokens: int) -> int:
        if self._config.dynamic_leaf_chunk_enabled:
            return self._working_leaf_chunk_tokens(raw_tokens)
        return max(1, self._config.leaf_chunk_tokens)

    def _has_raw_backlog_debt(self, *, deadline: float | None = None) -> bool:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return False
        remaining = self._foreground_deadline_remaining(deadline)
        try:
            state = self._lifecycle.get_by_conversation(
                self._conversation_id,
                lock_timeout_seconds=remaining,
                deadline=deadline,
                deadline_clock=self._foreground_now,
                busy_timeout_ms=(
                    max(0, int(remaining * 1000))
                    if remaining is not None
                    else None
                ),
            )
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                "deferred maintenance inspection",
                "lifecycle lock wait exceeded the shared deadline",
            ) from exc
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            raise ForegroundDeadlineExceeded(
                "deferred maintenance inspection",
                "SQLite contention prevented a bounded lifecycle read",
            ) from exc
        return bool(state and state.debt_kind == "raw_backlog" and state.debt_size_estimate > 0)

    def _budget_pressure_ratio(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> float | None:
        if self.context_length <= 0:
            return None
        token_count: int | None = None
        if observed_tokens is not None and observed_tokens > 0:
            token_count = observed_tokens
        elif messages is not None:
            token_count = count_messages_tokens(messages)
        elif self.last_prompt_tokens > 0:
            token_count = self.last_prompt_tokens
        if token_count is None or token_count <= 0:
            return None
        return token_count / self.context_length

    def _critical_budget_pressure_reached(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> bool:
        threshold = self._config.critical_budget_pressure_ratio
        if threshold <= 0:
            return False
        pressure = self._budget_pressure_ratio(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        return pressure is not None and pressure >= threshold

    def _should_run_deferred_maintenance(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
        deadline: float | None = None,
    ) -> bool:
        if not self._has_raw_backlog_debt(deadline=deadline):
            return False
        self._foreground_deadline_checkpoint(
            deadline,
            "deferred maintenance inspection",
            "before raw backlog sizing",
        )
        raw_tokens = self._raw_backlog_tokens(messages)
        self._foreground_deadline_checkpoint(
            deadline,
            "deferred maintenance inspection",
            "after raw backlog sizing",
        )
        if raw_tokens <= 0:
            return False
        if raw_tokens >= self._raw_backlog_threshold(raw_tokens):
            return True
        return self._critical_budget_pressure_reached(
            observed_tokens=observed_tokens,
            messages=messages,
        )

    def _refresh_raw_backlog_debt(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
        deadline: float | None = None,
    ) -> None:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return

        def refresh() -> None:
            raw_tokens = self._raw_backlog_tokens(messages)
            threshold = (
                self._raw_backlog_threshold(raw_tokens) if raw_tokens > 0 else 0
            )
            keep_under_critical_pressure = (
                raw_tokens > 0
                and self._has_raw_backlog_debt(deadline=deadline)
                and self._critical_budget_pressure_reached(
                    observed_tokens=observed_tokens,
                    messages=messages,
                )
            )
            if raw_tokens > 0 and (
                raw_tokens >= threshold or keep_under_critical_pressure
            ):
                self._lifecycle.record_debt(
                    self._conversation_id,
                    kind="raw_backlog",
                    size_estimate=raw_tokens,
                    lock_timeout_seconds=(
                        self._foreground_deadline_remaining(deadline)
                    ),
                    deadline=deadline,
                    deadline_clock=self._foreground_now,
                    busy_timeout_ms=(0 if deadline is not None else None),
                )
                return
            if self._has_raw_backlog_debt(deadline=deadline):
                self._lifecycle.clear_debt(
                    self._conversation_id,
                    lock_timeout_seconds=(
                        self._foreground_deadline_remaining(deadline)
                    ),
                    deadline=deadline,
                    deadline_clock=self._foreground_now,
                    busy_timeout_ms=(0 if deadline is not None else None),
                )

        if deadline is None:
            refresh()
            return

        # Maintenance debt is derived from immutable raw rows and can be
        # reconstructed on the next turn.  Never let this bookkeeping wait
        # behind another same-process lifecycle writer after the summary node
        # and frontier have already committed.
        self._foreground_deadline_checkpoint(
            deadline,
            "raw backlog debt persistence",
            "before lifecycle lock probe",
        )
        lifecycle_lock = getattr(self._lifecycle, "_lock", None)
        connection = getattr(self._lifecycle, "connection", None)
        if lifecycle_lock is None or connection is None:
            return
        if not lifecycle_lock.acquire(blocking=False):
            return

        prior_busy_timeout: int | None = None
        try:
            prior_busy_timeout = int(
                connection.execute("PRAGMA busy_timeout").fetchone()[0]
            )
            connection.execute("PRAGMA busy_timeout=0")
            self._foreground_deadline_checkpoint(
                deadline,
                "raw backlog debt persistence",
                "before derived debt write",
            )
            refresh()
            self._foreground_deadline_checkpoint(
                deadline,
                "raw backlog debt persistence",
                "after derived debt write",
            )
        except (TimeoutError, ForegroundDeadlineExceeded):
            if connection.in_transaction:
                connection.rollback()
            # Derived maintenance debt is safely reconstructed next turn.
            return
        except sqlite3.OperationalError as exc:
            if connection.in_transaction:
                connection.rollback()
            if not _is_sqlite_locked_error(exc):
                raise
            # A later turn can derive the same marker from durable raw rows.
            return
        finally:
            try:
                if prior_busy_timeout is not None:
                    connection.execute(
                        f"PRAGMA busy_timeout={prior_busy_timeout}"
                    )
            except sqlite3.Error:
                logger.debug(
                    "LCM could not restore SQLite busy_timeout after bounded "
                    "raw backlog debt persistence",
                    exc_info=True,
                )
            finally:
                lifecycle_lock.release()

    def _apply_session_start_metadata(self, session_id: str, kwargs: Dict[str, Any]) -> None:
        self._session_id = session_id
        self._session_platform = str(kwargs.get("platform") or "")
        self._refresh_session_filters()
        if hasattr(self, "_active_replay_preview_policies"):
            self._active_replay_preview_policy()
        # Hold the foreground view stable when the new binding is a side
        # channel (cron tick inside the gateway process, debug probe, etc.).
        # Tools that report "current session" to operators must keep pointing
        # at the real foreground rather than the ignored/stateless session
        # that just stole _session_id. Lifecycle paths still read _session_id
        # directly so cron's compress short-circuits correctly via the
        # _session_ignored / _session_stateless gates.
        if not self._session_ignored and not self._session_stateless:
            self._remember_foreground_rebind_candidate(session_id)
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
        if "hermes_home" in kwargs:
            self._hermes_home = kwargs["hermes_home"]

        update_model_is_authoritative = (
            self._context_length_source == "update_model"
            and self._update_model_pending_session_start
        )

        # Pick up context_length from kwargs if provided, but do not let stale
        # session metadata undo the authoritative runtime update_model() call.
        # Hermes Agent calls update_model() with the resolver output before it
        # binds a fresh agent/session.  Older or buggy host paths can still pass
        # a context_length copied from the previously bound runtime; treating
        # that as authoritative makes /model switches keep compressing against
        # the old model window.
        if "context_length" in kwargs:
            incoming_context_length = kwargs["context_length"]
            try:
                parsed_context_length = int(incoming_context_length)
            except (TypeError, ValueError):
                logger.debug(
                    "LCM ignored invalid session-start context_length: %r",
                    incoming_context_length,
                )
                self._update_model_pending_session_start = False
                return
            if parsed_context_length <= 0:
                if update_model_is_authoritative:
                    if self._session_metadata_matches_active_runtime(
                        kwargs,
                        ignore_empty_optional=True,
                    ):
                        logger.debug(
                            "LCM ignored missing session-start context_length=%r for model=%s; active update_model context_length=%s",
                            incoming_context_length,
                            self.model or str(kwargs.get("model") or ""),
                            self.context_length,
                        )
                    else:
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                    self._update_model_pending_session_start = False
                    return
                self._set_context_length(parsed_context_length, source="session_start")
                update_model_is_authoritative = False
            else:
                if (
                    update_model_is_authoritative
                    and parsed_context_length not in {self.context_length, self.raw_context_length}
                ):
                    logger.warning(
                        "LCM ignored stale session-start context_length=%s for model=%s; active update_model raw_context_length=%s effective_context_length=%s",
                        parsed_context_length,
                        self.model or str(kwargs.get("model") or ""),
                        self.raw_context_length,
                        self.context_length,
                    )
                    self._update_model_pending_session_start = False
                    return
                if update_model_is_authoritative:
                    if not self._session_metadata_matches_active_runtime(kwargs):
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                        self._update_model_pending_session_start = False
                        return
                else:
                    self._set_context_length(
                        parsed_context_length,
                        source="session_start",
                        model=str(kwargs.get("model") or self.model),
                        provider=str(kwargs.get("provider") or self.provider),
                    )
                    update_model_is_authoritative = False
        if (
            update_model_is_authoritative
            and not self._session_metadata_matches_active_runtime(kwargs)
        ):
            logger.warning(
                "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                str(kwargs.get("model") or ""),
                self.model,
            )
            self._update_model_pending_session_start = False
            return
        if "model" in kwargs:
            self.model = str(kwargs.get("model") or "")
        route_affects_context = "model" in kwargs or "provider" in kwargs
        for key in ("base_url", "api_key", "provider", "api_mode"):
            if key in kwargs:
                setattr(self, key, str(kwargs.get(key) or ""))
        if (
            "context_length" not in kwargs
            and route_affects_context
            and (self.raw_context_length or self.context_length)
        ):
            self._set_context_length(
                self.raw_context_length or self.context_length,
                source=self._context_length_source or "session_start",
                model=self.model,
                provider=self.provider,
            )
        self._update_model_pending_session_start = False

    def _continue_compression_boundary(
        self,
        session_id: str,
        old_session_id: str,
        kwargs: Dict[str, Any],
    ) -> None:
        previous_session_id = self._session_id
        requested_conversation_id = kwargs.get("conversation_id")
        session_state = self._lifecycle.get_by_session(old_session_id)
        conversation_state = self._lifecycle.get_by_conversation(old_session_id)

        def _state_conversation_matches(state: Any) -> bool:
            return bool(
                state
                and (
                    not requested_conversation_id
                    or state.conversation_id == requested_conversation_id
                )
            )

        def _has_summary_nodes(candidate_session_id: str | None) -> bool:
            return bool(candidate_session_id and self._dag.get_session_nodes(candidate_session_id))

        def _host_source_from_conversation_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.conversation_id == old_session_id
                and state.current_session_id
                and _has_summary_nodes(state.current_session_id)
            ):
                return state.current_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id
                and _has_summary_nodes(state.last_finalized_session_id)
            ):
                return state.last_finalized_session_id, state
            return "", None

        def _host_source_from_session_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id == old_session_id
                and _has_summary_nodes(old_session_id)
            ):
                return old_session_id, state
            return "", None

        host_source_session_id, host_source_state = _host_source_from_conversation_state(
            conversation_state
        )
        if not host_source_session_id:
            host_source_session_id, host_source_state = _host_source_from_session_state(
                session_state
            )

        source_session_id = host_source_session_id or old_session_id
        source_state = host_source_state or session_state

        if previous_session_id and previous_session_id != old_session_id:
            # Hermes passes the session that actually crossed the compression
            # boundary as old_session_id. A different bound session can be a
            # short-lived subagent/cron/WebUI side channel that ran after the
            # foreground compaction. Prefer the host-authoritative source when
            # durable lifecycle + DAG evidence proves it belongs to LCM, then
            # fall back to the older bound-session recovery path. When the host
            # old_session_id is the durable conversation id, use that row's
            # current/finalized LCM source instead of unrelated auxiliary rows
            # where the id appears only as last_finalized_session_id.
            if host_source_session_id:
                logger.warning(
                    "LCM compression boundary using host old_session_id %s as carry-over source=%s despite bound session drift=%s",
                    old_session_id,
                    host_source_session_id,
                    previous_session_id,
                )
            else:
                bound_state = self._lifecycle.get_by_session(previous_session_id)
                bound_conversation_matches = bool(
                    bound_state
                    and (not self._conversation_id or bound_state.conversation_id == self._conversation_id)
                    and (
                        not requested_conversation_id
                        or bound_state.conversation_id == requested_conversation_id
                    )
                )
                bound_is_active_source = bool(
                    bound_state and bound_state.current_session_id == previous_session_id
                )
                bound_is_finalized_source = bool(
                    bound_state
                    and bound_state.current_session_id is None
                    and bound_state.last_finalized_session_id == previous_session_id
                )
                bound_has_summary_nodes = bool(self._dag.get_session_nodes(previous_session_id))
                if (
                    bound_conversation_matches
                    and (bound_is_active_source or bound_is_finalized_source)
                    and bound_has_summary_nodes
                ):
                    source_session_id = previous_session_id
                    source_state = bound_state
                    logger.warning(
                        "LCM compression boundary using bound session %s as carry-over source; host old_session_id=%s does not match",
                        previous_session_id,
                        old_session_id,
                    )
                else:
                    # Fallback: sibling chain with zero-DAG parent.
                    # When stale old_session_id has no DAG nodes AND the
                    # bound session belongs to a different conversation_id
                    # but shares the same last_finalized_session_id
                    # (parent) — prefer the bound session despite the
                    # conversation_id mismatch. This handles the lifecycle
                    # fork case where two sessions on the same channel
                    # received different conversation_ids.
                    bound_shares_parent_with_host = bool(
                        bound_state
                        and bound_state.last_finalized_session_id == old_session_id
                    )
                    host_has_no_dag = not bool(
                        self._dag.get_session_nodes(old_session_id)
                    )
                    if (
                        bound_shares_parent_with_host
                        and host_has_no_dag
                        and (bound_is_active_source or bound_is_finalized_source)
                        and bound_has_summary_nodes
                    ):
                        source_session_id = previous_session_id
                        source_state = bound_state
                        logger.warning(
                            "LCM compression boundary using bound session %s on sibling chain as carry-over source; host old_session_id=%s has zero DAG, parent=%s matches",
                            previous_session_id,
                            old_session_id,
                            bound_state.last_finalized_session_id,
                        )
                    else:
                        source_session_id = ""
                        source_state = None

        conversation_id = (
            (source_state.conversation_id if source_state else None)
            or kwargs.get("conversation_id")
            or self._conversation_id
            or source_session_id
            or old_session_id
            or session_id
        )
        process_local_frontier = (
            int(self._last_compacted_store_id or 0)
            if source_session_id and previous_session_id == source_session_id
            else 0
        )
        pending_reset_frontier = int(
            self._pending_reset_frontier_store_id
            if self._pending_reset_session_id
            and self._pending_reset_session_id == source_session_id
            else 0
        )
        frontier = max(
            process_local_frontier,
            int(source_state.current_frontier_store_id if source_state else 0),
            int(source_state.last_finalized_frontier_store_id if source_state else 0),
            pending_reset_frontier,
        )
        can_reassign = bool(
            source_session_id
            and session_id
            and source_session_id != session_id
        )
        boundary_placeholder_budget = {}
        boundary_placeholder_ordinals: dict[str, set[int]] = {}
        if can_reassign:
            if previous_session_id == source_session_id:
                boundary_placeholder_budget = self._active_replay_generated_placeholder_digest_budget()
                boundary_placeholder_ordinals = self._generated_placeholder_digest_ordinals_for_active_replay(
                    self._last_active_replay_messages
                )
            if not boundary_placeholder_budget:
                boundary_placeholder_budget = self._load_generated_ignored_placeholder_hash_counts(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_counts",
                        source_session_id,
                    )
                )
            if not boundary_placeholder_ordinals:
                boundary_placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_ordinals",
                        source_session_id,
                    )
                )
            for digest, ordinals in boundary_placeholder_ordinals.items():
                boundary_placeholder_budget[digest] = max(
                    boundary_placeholder_budget.get(digest, 0),
                    len(ordinals),
                )
            self._compression_boundary_stored_placeholder_digest_counts = (
                self._stored_active_replay_placeholder_digest_counts(
                    source_session_id,
                    after_store_id=frontier,
                )
            )

        if can_reassign and self._background_compaction_enabled():
            try:
                can_reassign = (
                    self._background_compaction.begin_session_reset(
                        conversation_id=conversation_id,
                        session_id=source_session_id,
                        reason="compression_rollover",
                    )
                    is not None
                )
            except sqlite3.OperationalError as exc:
                can_reassign = False
                logger.warning(
                    "LCM compression carry-over skipped because the old-session publication fence could not be acquired: %s",
                    exc,
                )

        if can_reassign:
            self._lifecycle.finalize_session(
                conversation_id,
                source_session_id,
                frontier_store_id=frontier,
            )
            self._copy_generated_ignore_hashes_to_session(
                source_session_id,
                session_id,
                copy_dependent_content=True,
                source_frontier_store_id=frontier,
            )
            self._write_generated_ignored_placeholder_hash_counts(
                boundary_placeholder_budget,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_counts",
                    session_id,
                ),
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                boundary_placeholder_ordinals,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_ordinals",
                    session_id,
                ),
            )
            # Compression rollover carries derived context forward, but raw
            # messages remain owned by the session that produced them. Moving
            # raw rows here makes session-scoped transcript recovery report the
            # old/child session as missing even though its payload was only
            # reassigned to the next compression segment.
            moved_nodes = self._dag.reassign_session_nodes(source_session_id, session_id)
            logger.debug(
                "LCM compression boundary continued %s -> %s: carried %d DAG nodes; preserved raw message ownership",
                source_session_id,
                session_id,
                moved_nodes,
            )
        elif old_session_id:
            logger.warning(
                "LCM compression boundary skipped carry-over: old_session_id=%s does not match bound session=%s",
                old_session_id,
                previous_session_id,
            )
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
            self._last_boundary_skip_time = time.time()
            self._apply_session_start_metadata(session_id, kwargs)
            self._bind_lifecycle_state(
                session_id,
                conversation_id=kwargs.get("conversation_id"),
            )
            self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            self._schedule_ingest_cursor_reconciliation()
            self._clear_pending_reset_boundary()
            self._log_session_filter_diagnostics()
            return

        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(session_id, conversation_id=conversation_id)
        self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
        if frontier > 0:
            state = self._lifecycle.advance_frontier(
                self._conversation_id,
                session_id,
                frontier,
            )
            if state is not None:
                self._last_compacted_store_id = state.current_frontier_store_id
        self._clear_pending_reset_boundary()
        self._compression_boundary_ingest_pending = can_reassign
        self._compression_boundary_active_placeholder_digest_budget = boundary_placeholder_budget
        self._compression_boundary_active_placeholder_digest_ordinals = boundary_placeholder_ordinals
        self._log_session_filter_diagnostics()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        if "hermes_home" in kwargs:
            self._rebind_storage_for_home(str(kwargs.get("hermes_home") or ""))

        boundary_reason = str(kwargs.get("boundary_reason") or "")
        old_session_id = str(kwargs.get("old_session_id") or "")
        previous_session_id = self._session_id
        self._lcm_current_start_allows_bypass_lineage = False
        requested_platform = str(kwargs.get("platform") or self._session_platform or "")
        pre_reset_preserve_ambiguous_no_frame_old_session = False
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            old_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(
                old_session_id
            )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(session_id)
            with self._auxiliary_session_lock:
                active_old_auxiliary_generation = self._auxiliary_session_generations.get(
                    old_session_id
                )
                old_session_auxiliary_generation_is_stale = bool(
                    (
                        old_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            old_session_id,
                            old_session_auxiliary_generation,
                        )
                    )
                    or (
                        new_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            session_id,
                            new_session_auxiliary_generation,
                        )
                    )
                    or (
                        active_old_auxiliary_generation is not None
                        and (
                            (
                                old_session_auxiliary_generation
                                and active_old_auxiliary_generation != old_session_auxiliary_generation
                            )
                        )
                    )
                )
                old_session_has_retired_generation = bool(
                    self._auxiliary_retired_session_generations.get(old_session_id)
                )
            if old_session_auxiliary_generation_is_stale:
                logger.info(
                    "LCM ignored stale auxiliary compression boundary from %s to %s",
                    old_session_id,
                    session_id,
                )
                return
            pre_reset_preserve_ambiguous_no_frame_old_session = bool(
                active_old_auxiliary_generation is not None
                and not old_session_auxiliary_generation
                and old_session_id != self._session_id
                and old_session_has_retired_generation
                and old_session_id not in self._auxiliary_direct_end_guard_session_ids
                and new_session_auxiliary_parent != old_session_id
            )
        if self._host_fallback_compressor is not None and (
            self._host_fallback_session_id != session_id or requested_platform != self._session_platform
        ) and not (
            pre_reset_preserve_ambiguous_no_frame_old_session
            and self._host_fallback_session_id == old_session_id
        ):
            compressor = self._host_fallback_compressor
            fallback_session_id = self._host_fallback_session_id or previous_session_id
            on_session_end = getattr(compressor, "on_session_end", None)
            if callable(on_session_end) and fallback_session_id:
                try:
                    on_session_end(fallback_session_id, [])
                except Exception:
                    logger.debug("LCM host fallback compressor session-start reset failed", exc_info=True)
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            old_session_is_suppressed_foreground = self._auxiliary_lineage_suppressed_as_foreground(
                old_session_id
            )
            old_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(
                old_session_id
            )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(session_id)
            with self._auxiliary_session_lock:
                active_old_auxiliary_generation = self._auxiliary_session_generations.get(
                    old_session_id
                )
                old_session_auxiliary_generation_is_stale = bool(
                    (
                        old_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            old_session_id,
                            old_session_auxiliary_generation,
                        )
                    )
                    or (
                        new_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            session_id,
                            new_session_auxiliary_generation,
                        )
                    )
                    or (
                        active_old_auxiliary_generation is not None
                        and (
                            (
                                old_session_auxiliary_generation
                                and active_old_auxiliary_generation != old_session_auxiliary_generation
                            )
                        )
                    )
                )
                old_session_has_retired_generation = bool(
                    self._auxiliary_retired_session_generations.get(old_session_id)
                )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_is_auxiliary_continuation = new_session_auxiliary_parent == old_session_id
            preserve_ambiguous_no_frame_old_session = bool(
                active_old_auxiliary_generation is not None
                and not old_session_auxiliary_generation
                and old_session_id != self._session_id
                and old_session_has_retired_generation
                and old_session_id not in self._auxiliary_direct_end_guard_session_ids
                and not new_session_is_auxiliary_continuation
            )
            if old_session_auxiliary_generation_is_stale:
                logger.info(
                    "LCM ignored stale auxiliary compression boundary from %s to %s",
                    old_session_id,
                    session_id,
                )
                return
            if (
                self._has_auxiliary_lineage_session(old_session_id)
                and not old_session_auxiliary_generation_is_stale
                and (
                    old_session_id != self._session_id
                    or old_session_auxiliary_generation
                    or new_session_is_auxiliary_continuation
                )
                and (
                    not old_session_is_suppressed_foreground
                    or old_session_auxiliary_generation
                    or new_session_is_auxiliary_continuation
                )
            ):
                self._handoff_auxiliary_session(
                    old_session_id,
                    session_id,
                    preserve_old_session=(
                        old_session_id == self._session_id
                        or preserve_ambiguous_no_frame_old_session
                    ),
                    preserve_old_foreground_marker=old_session_is_suppressed_foreground,
                )
                logger.info(
                    "LCM auxiliary session %s compressed to %s — keeping boundary stateless",
                    old_session_id,
                    session_id,
                )
                return
            if self._compression_boundary_from_lcm_bypassed_session(old_session_id):
                self._handoff_lcm_bypass_lineage(
                    old_session_id,
                    session_id,
                    new_platform=str(kwargs.get("platform") or ""),
                )
                self._clear_thread_context_stateless()
                if previous_session_id and previous_session_id != session_id:
                    self._finalize_pending_reset_boundary(previous_session_id)
                    self._reset_session_scoped_runtime_state()
                else:
                    self._clear_pending_reset_boundary()
                    self._ingest_cursor = 0
                    self._last_compacted_store_id = 0
                    self._last_overflow_recovery_failed = False
                    self._last_condensation_suppressed_reason = ""
                self._lcm_current_start_allows_bypass_lineage = True
                self._apply_session_start_metadata(session_id, kwargs)
                self._bind_lifecycle_state(
                    session_id,
                    conversation_id=kwargs.get("conversation_id"),
                )
                self._schedule_ingest_cursor_reconciliation()
                self._log_session_filter_diagnostics()
                logger.info(
                    "LCM compression boundary %s -> %s stayed stateless because the source session bypasses LCM storage",
                    old_session_id,
                    session_id,
                )
                return
            self._clear_thread_context_stateless()
            self._continue_compression_boundary(session_id, old_session_id, kwargs)
            return

        if self._is_live_auxiliary_child_session(session_id, previous_session_id, kwargs):
            explicit_parent_id = str(kwargs.get("parent_session_id") or "")
            preserve_foreground_reuse_marker = bool(
                (
                    explicit_parent_id
                    and self._lcm_session_last_bypassed.get(explicit_parent_id)
                )
                or self._lcm_session_last_normal_conversation_id.get(session_id)
            )
            if preserve_foreground_reuse_marker:
                if self._lcm_session_last_normal_conversation_id.get(session_id):
                    with self._auxiliary_session_lock:
                        self._auxiliary_foreground_reused_session_ids.add(session_id)
                self._mark_thread_context_stateless(
                    session_id,
                    preserve_foreground_reuse_marker=True,
                )
            else:
                self._register_auxiliary_session(session_id)
            logger.info(
                "LCM session %s is a live child of bound session %s — treating it as auxiliary/stateless",
                session_id,
                previous_session_id,
            )
            return
        start_platform = str(kwargs.get("platform") or "")
        side_channel_rebind = self._session_id_matches_lcm_bypass_filters(
            session_id,
            platform=start_platform,
        ) or self._has_lcm_bypass_lineage_session(session_id, platform=start_platform)
        self._unmark_thread_context_auxiliary_session(
            session_id,
            suppress_as_foreground_reuse=not side_channel_rebind,
        )
        self._clear_thread_context_stateless()

        # The stock host reports two non-invalidating lifecycle edges with the
        # already-bound session id: an ordinary same-conversation rebind and an
        # in-place compression commit (old_session_id == session_id).  The
        # process-local ingest cursor already describes the host's active list
        # in both cases.  Rewinding it here makes the next turn scan/reappend the
        # entire transcript when replay cleanup prevents an exact prefix match.
        #
        # Preserve runtime progress only when the durable lifecycle row proves
        # the same logical conversation.  Explicit resets, platform changes,
        # side-channel/bypass binds, conversation reassignment, and all session
        # rotations continue through the invalidating path below.
        requested_conversation_id = str(kwargs.get("conversation_id") or "")
        same_session_lifecycle = (
            self._lifecycle.get_by_conversation(self._conversation_id)
            if previous_session_id == session_id and self._conversation_id
            else None
        )
        stable_same_session_rebind = bool(
            previous_session_id == session_id
            and not self._pending_reset_session_id
            and not self._session_ignored
            and not self._session_stateless
            and not side_channel_rebind
            and requested_platform == self._session_platform
            and (
                (not boundary_reason and not old_session_id)
                or (
                    boundary_reason == "compression"
                    and old_session_id == session_id
                )
            )
            and (
                not requested_conversation_id
                or requested_conversation_id == self._conversation_id
            )
            and same_session_lifecycle is not None
            and same_session_lifecycle.conversation_id == self._conversation_id
            and (
                same_session_lifecycle.current_session_id == session_id
                or (
                    same_session_lifecycle.current_session_id is None
                    and same_session_lifecycle.last_finalized_session_id == session_id
                )
            )
        )
        if stable_same_session_rebind:
            stable_kwargs = dict(kwargs)
            stable_kwargs.setdefault("platform", self._session_platform)
            stable_kwargs.setdefault("conversation_id", self._conversation_id)
            self._apply_session_start_metadata(session_id, stable_kwargs)
            self._bind_lifecycle_state(
                session_id,
                conversation_id=self._conversation_id,
            )
            self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            self._log_session_filter_diagnostics()
            logger.debug(
                "LCM preserved runtime progress across same-session %s rebind: "
                "session=%s conversation=%s cursor=%d frontier=%d",
                boundary_reason or "ordinary",
                session_id,
                self._conversation_id,
                self._ingest_cursor,
                self._last_compacted_store_id,
            )
            return

        if previous_session_id and previous_session_id != session_id:
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
        else:
            self._clear_pending_reset_boundary()
            self._ingest_cursor = 0
            self._last_compacted_store_id = 0
            self._last_overflow_recovery_failed = False
            self._last_condensation_suppressed_reason = ""
        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(
            session_id,
            conversation_id=kwargs.get("conversation_id"),
        )
        self._schedule_ingest_cursor_reconciliation()
        self._log_session_filter_diagnostics()

    def _session_end_matches_current_store_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        prefix_count = self._session_end_store_prefix_count(session_id, messages)
        return prefix_count is not None and prefix_count > 0

    def _session_end_prefix_compare_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                key: self._session_end_prefix_compare_value(child, session_id=session_id)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                self._session_end_prefix_compare_value(child, session_id=session_id)
                for child in value
            ]
        if not isinstance(value, str):
            return value

        text = restore_ingest_payload_placeholders(
            value,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )
        stripped = text.strip()
        ingest_refs = extract_ingest_externalized_refs(stripped)
        if (
            len(ingest_refs) == 1
            and stripped.startswith("[Externalized LCM ingest payload:")
            and stripped.endswith("]")
        ):
            payload = load_externalized_payload(
                ingest_refs[0],
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content

        if is_externalized_placeholder(stripped):
            ref = extract_externalized_ref(stripped)
            payload = load_externalized_payload(
                ref or "",
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content
        return text

    def _session_end_prefix_compare_content(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
    ) -> str:
        content = self._session_end_prefix_compare_value(
            (message or {}).get("content"),
            session_id=session_id,
        )
        content = redact_sensitive_value(
            content,
            self._config,
            parse_json_strings=False,
        )
        return normalize_content_value(content)

    def _session_end_prefix_compare_tool_calls(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
    ) -> str:
        tool_calls = self._session_end_prefix_compare_value(
            (message or {}).get("tool_calls"),
            session_id=session_id,
        )
        tool_calls = redact_sensitive_value(
            tool_calls,
            self._config,
            parse_json_strings=True,
        )
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        return json.dumps(
            tool_calls,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _session_end_prefix_compare_identity(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
    ) -> tuple[str, str, str, str, str]:
        return (
            str((message or {}).get("role") or ""),
            self._session_end_prefix_compare_content(message, session_id=session_id),
            str((message or {}).get("tool_call_id") or ""),
            str((message or {}).get("tool_name") or ""),
            self._session_end_prefix_compare_tool_calls(message, session_id=session_id),
        )

    def _session_end_store_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> Optional[int]:
        try:
            stored_messages = self._store.get_range(
                session_id,
                limit=max(1, len(messages)),
                conversation_id=conversation_id,
            )
        except Exception:
            logger.debug("LCM session-end prefix check failed", exc_info=True)
            return None
        if not stored_messages:
            return 0
        if len(messages) < len(stored_messages):
            return None
        for idx, stored_msg in enumerate(stored_messages):
            msg = messages[idx]
            try:
                message_identity = self._session_end_prefix_compare_identity(
                    msg,
                    session_id=session_id,
                )
                stored_identity = self._session_end_prefix_compare_identity(
                    stored_msg,
                    session_id=session_id,
                )
            except Exception:
                logger.debug("LCM session-end prefix compare normalization failed", exc_info=True)
                return None
            if message_identity != stored_identity:
                return None
        return len(stored_messages)

    @staticmethod
    def _lcm_bypass_message_fingerprint(message: Dict[str, Any]) -> str:
        tool_calls = message.get("tool_calls")
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        payload = {
            "role": message.get("role"),
            "content": normalize_content_value(message.get("content")),
            "tool_call_id": message.get("tool_call_id"),
            "tool_calls": tool_calls,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()

    def _remember_lcm_bypass_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            remembered = self._lcm_bypass_message_prefix_fingerprints.setdefault(session_id, [])
            truncated = len(messages) > _LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT
            retained: list[tuple[list[str], bool]] = []
            for existing_fingerprints, existing_truncated in remembered:
                if existing_fingerprints == fingerprints:
                    truncated = truncated or bool(existing_truncated)
                    continue
                retained.append((existing_fingerprints, existing_truncated))
            retained.append((fingerprints, truncated))
            remembered[:] = retained

    def _remember_lcm_normal_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            self._lcm_normal_message_prefix_fingerprints[
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ] = fingerprints

    def _lcm_normal_prefix_key(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> tuple[str, str]:
        return (
            session_id,
            str(
                conversation_id
                or self._lcm_session_last_normal_conversation_id.get(session_id)
                or ""
            ),
        )

    def _messages_match_fingerprint_prefix(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_fingerprint_prefix_count(fingerprints, messages) > 0

    def _matching_fingerprint_prefix_count(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> int:
        if not fingerprints or not messages:
            return 0
        compare_count = min(len(fingerprints), len(messages))
        if compare_count <= 0:
            return 0
        candidate = [self._lcm_bypass_message_fingerprint(msg) for msg in messages[:compare_count]]
        if candidate == fingerprints[:compare_count]:
            return compare_count
        return 0

    def _messages_match_lcm_bypass_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_lcm_bypass_prefix_count(session_id, messages) > 0

    def _matching_lcm_bypass_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        count, _truncated = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
        return count

    def _matching_lcm_bypass_prefix_evidence(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[int, bool]:
        best_count = 0
        best_truncated = False
        for fingerprints, truncated in self._lcm_bypass_message_prefix_fingerprints.get(session_id, []):
            count = self._matching_fingerprint_prefix_count(fingerprints, messages)
            count_truncated = bool(truncated and count > 0 and count == len(fingerprints))
            if count > best_count:
                best_count = count
                best_truncated = count_truncated
            elif count == best_count:
                best_truncated = best_truncated or count_truncated
        return best_count, best_truncated

    def _messages_match_lcm_normal_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> bool:
        return self._matching_lcm_normal_prefix_count(
            session_id,
            messages,
            conversation_id=conversation_id,
        ) > 0

    def _matching_lcm_normal_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> int:
        return self._matching_fingerprint_prefix_count(
            self._lcm_normal_message_prefix_fingerprints.get(
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ) or [],
            messages,
        )

    def _append_off_current_session_end_suffix(
        self,
        session_id: str,
        suffix: List[Dict[str, Any]],
        *,
        source: str,
        conversation_id: str,
    ) -> list[int]:
        if not session_id or not suffix:
            return []
        kept: list[Dict[str, Any]] = []
        for msg in suffix:
            if self._matches_ignore_message_patterns(msg):
                self._ignored_message_count += 1
                excerpt = (text_content_for_pattern_matching(msg.get("content")) or "")[:80].replace("\n", " ")
                logger.debug(
                    "LCM ignore_message_patterns dropped late session-end %s message: %r",
                    msg.get("role", "unknown"),
                    excerpt,
                )
                continue
            kept.append(msg)
        if not kept:
            return []
        protected_messages = protect_messages_for_ingest(
            kept,
            session_id=session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return self._store._append_protected_batch(
            session_id,
            protected_messages,
            [count_message_tokens(msg) for msg in protected_messages],
            source=source,
            conversation_id=conversation_id,
            dedupe_existing=True,
        )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        ended_generation = self._in_process_auxiliary_caller_generation(session_id)
        active_auxiliary_end = session_id in self._active_auxiliary_session_ids()
        if (
            self._has_auxiliary_lineage_session(session_id)
            and session_id != self._session_id
            and (
                active_auxiliary_end
                or not self._auxiliary_lineage_suppressed_as_foreground(session_id)
                or ended_generation
                or (
                    session_id in self._auxiliary_last_prompt_tokens
                    and not self._auxiliary_lineage_suppressed_as_foreground(session_id)
                )
            )
        ):
            current_thread_session_id = self._thread_context_session_id()
            deactivated = self._deactivate_auxiliary_session(
                session_id,
                generation=ended_generation,
            )
            if deactivated:
                if current_thread_session_id == session_id or active_auxiliary_end:
                    self._remember_lcm_bypass_message_prefix(session_id, messages)
                self._end_host_fallback_compressor_for_session(
                    session_id,
                    messages,
                    current_session_bypasses=False,
                )
                if current_thread_session_id == session_id:
                    self._clear_thread_context_stateless(session_id)
            return
        current_session_bypasses = session_id == self._session_id and self._bypasses_lcm_context_management()
        ended_session_directly_bypasses = self._ended_session_directly_bypasses_lcm(session_id)
        direct_bypass_normal_conversation_id = self._lcm_session_last_normal_conversation_id.get(session_id)
        direct_bypass_normal_prefix_count = None
        if (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and direct_bypass_normal_conversation_id
            and not ended_generation
        ):
            direct_bypass_normal_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=direct_bypass_normal_conversation_id,
            )
        direct_bypass_is_suppressed_reused_normal = (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and bool(direct_bypass_normal_conversation_id)
            and not ended_generation
            and session_id != self._thread_context_session_id()
            and direct_bypass_normal_prefix_count is not None
            and direct_bypass_normal_prefix_count > 0
        )
        if ended_session_directly_bypasses and not direct_bypass_is_suppressed_reused_normal:
            self._remember_lcm_bypass_message_prefix(session_id, messages)
            self._end_host_fallback_compressor_for_session(
                session_id,
                messages,
                current_session_bypasses=current_session_bypasses,
            )
            if session_id == self._thread_context_session_id():
                self._deactivate_auxiliary_session(session_id, generation=ended_generation)
                self._clear_thread_context_stateless(session_id)
            return
        same_id_has_bypass_lineage = (
            session_id == self._session_id
            and not current_session_bypasses
            and self._has_lcm_bypass_lineage_session(session_id)
        )
        same_id_normal_prefix_count = None
        same_id_recorded_normal_prefix_count = 0
        same_id_bypass_prefix_count = 0
        same_id_bypass_prefix_truncated = False
        if same_id_has_bypass_lineage:
            same_id_conversation_id = (
                self._conversation_id
                or self._lcm_session_last_normal_conversation_id.get(session_id)
                or None
            )
            (
                same_id_bypass_prefix_count,
                same_id_bypass_prefix_truncated,
            ) = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
            same_id_normal_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=same_id_conversation_id,
            )
            same_id_recorded_normal_prefix_count = self._matching_lcm_normal_prefix_count(
                session_id,
                messages,
                conversation_id=same_id_conversation_id,
            )
        same_id_store_prefix_positive = (
            same_id_normal_prefix_count is not None
            and same_id_normal_prefix_count > 0
        )
        same_id_strongest_normal_prefix_count = max(
            same_id_recorded_normal_prefix_count,
            same_id_normal_prefix_count if same_id_store_prefix_positive else 0,
        )
        same_id_truncated_bypass_prefix_ambiguous = (
            same_id_bypass_prefix_truncated
            and same_id_bypass_prefix_count > 0
            and same_id_strongest_normal_prefix_count >= same_id_bypass_prefix_count
            and len(messages) > same_id_bypass_prefix_count
        )
        same_id_matches_stronger_normal_prefix = (
            same_id_strongest_normal_prefix_count > 0
            and not same_id_truncated_bypass_prefix_ambiguous
            and (
                same_id_bypass_prefix_count <= 0
                or same_id_strongest_normal_prefix_count >= same_id_bypass_prefix_count
            )
        )
        off_current_auxiliary_reused_normal = (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and bool(direct_bypass_normal_conversation_id)
            and not ended_generation
        )
        off_current_lineage = (
            session_id != self._session_id
            and (
                self._has_lcm_bypass_lineage_session(session_id)
                or off_current_auxiliary_reused_normal
            )
        )
        off_current_normal_conversation_id = (
            self._lcm_session_last_normal_conversation_id.get(session_id)
            if off_current_lineage
            else ""
        )
        off_current_store_prefix_count = None
        off_current_recorded_prefix_count = 0
        off_current_bypass_prefix_count = 0
        off_current_bypass_prefix_truncated = False
        if off_current_lineage:
            (
                off_current_bypass_prefix_count,
                off_current_bypass_prefix_truncated,
            ) = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
        if off_current_lineage and off_current_normal_conversation_id:
            off_current_store_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=off_current_normal_conversation_id,
            )
            off_current_recorded_prefix_count = self._matching_lcm_normal_prefix_count(
                session_id,
                messages,
                conversation_id=off_current_normal_conversation_id,
            )
        off_current_prefix_count = None
        off_current_store_prefix_positive = (
            off_current_store_prefix_count is not None
            and off_current_store_prefix_count > 0
        )
        off_current_store_prefix_for_append = int(off_current_store_prefix_count or 0)
        off_current_recorded_prefix_for_append = 0
        if off_current_recorded_prefix_count > 0 and off_current_normal_conversation_id:
            try:
                stored_normal_rows = self._store.get_range(
                    session_id,
                    limit=off_current_recorded_prefix_count + 1,
                    conversation_id=off_current_normal_conversation_id,
                )
            except Exception:
                logger.debug("LCM off-current recorded-prefix row-count probe failed", exc_info=True)
                stored_normal_rows = []
            if len(stored_normal_rows) == off_current_recorded_prefix_count:
                off_current_recorded_prefix_for_append = off_current_recorded_prefix_count
        off_current_strongest_normal_prefix_count = max(
            off_current_store_prefix_for_append if off_current_store_prefix_positive else 0,
            off_current_recorded_prefix_for_append,
        )
        off_current_truncated_bypass_prefix_ambiguous = (
            off_current_bypass_prefix_truncated
            and off_current_bypass_prefix_count > 0
            and off_current_strongest_normal_prefix_count >= off_current_bypass_prefix_count
            and len(messages) > off_current_bypass_prefix_count
        )
        if (
            off_current_store_prefix_positive
            and not off_current_truncated_bypass_prefix_ambiguous
            and (
                off_current_bypass_prefix_count <= 0
                or off_current_store_prefix_for_append > off_current_bypass_prefix_count
            )
        ):
            off_current_prefix_count = off_current_store_prefix_for_append
        elif (
            off_current_recorded_prefix_for_append > 0
            and not off_current_truncated_bypass_prefix_ambiguous
            and (
                off_current_bypass_prefix_count <= 0
                or off_current_recorded_prefix_for_append > off_current_bypass_prefix_count
            )
        ):
            off_current_prefix_count = off_current_recorded_prefix_for_append
        if (
            off_current_lineage
            and not off_current_auxiliary_reused_normal
            and off_current_normal_conversation_id
            and off_current_store_prefix_count == 0
            and off_current_bypass_prefix_count <= 0
            and self._lcm_session_last_bypassed.get(session_id) is False
        ):
            off_current_prefix_count = 0
        same_id_should_bypass = (
            same_id_has_bypass_lineage
            and same_id_bypass_prefix_count > 0
            and not same_id_matches_stronger_normal_prefix
        )
        off_current_matches_bypass_prefix = (
            session_id != self._session_id
            and self._has_lcm_bypass_lineage_session(session_id)
            and off_current_bypass_prefix_count > 0
            and off_current_prefix_count is None
        )
        ended_lineage_bypasses = (
            session_id != self._session_id
            and self._has_lcm_bypass_lineage_session(session_id)
            and bool(self._lcm_session_last_bypassed.get(session_id))
            and not self._session_end_matches_current_store_prefix(session_id, messages)
        )
        off_current_should_bypass = off_current_lineage and off_current_prefix_count is None
        if off_current_prefix_count is not None:
            prefix_count = off_current_prefix_count
            suffix = messages[prefix_count:]
            if suffix:
                self._append_off_current_session_end_suffix(
                    session_id,
                    suffix,
                    source=(
                        self._lcm_session_last_normal_platform.get(session_id)
                        or self._lcm_session_last_platform.get(session_id, self._session_platform)
                    ),
                    conversation_id=off_current_normal_conversation_id,
                )
            try:
                state = self._lifecycle.get_by_conversation(off_current_normal_conversation_id)
                frontier_store_id = state.current_frontier_store_id if state is not None else 0
                self._lifecycle.finalize_session(
                    off_current_normal_conversation_id,
                    session_id,
                    frontier_store_id=frontier_store_id,
                )
            except Exception:
                logger.debug("LCM off-current session-end lifecycle finalization failed", exc_info=True)
            return
        if (
            current_session_bypasses
            or same_id_should_bypass
            or off_current_should_bypass
            or off_current_matches_bypass_prefix
            or ended_lineage_bypasses
        ):
            self._end_host_fallback_compressor_for_session(
                session_id,
                messages,
                current_session_bypasses=current_session_bypasses,
            )
            return
        try:
            with _temporary_sqlite_busy_timeout(
                [
                    getattr(self._store, "_conn", None),
                    getattr(self._lifecycle, "_conn", None),
                ],
                _SESSION_END_BUSY_TIMEOUT_MS,
            ):
                try:
                    # Best-effort final flush. Keep this path bounded because
                    # host gateways call session-end hooks from lifecycle paths
                    # that must not wait through SQLite's normal busy timeout.
                    self._ingest_messages(messages)
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end raw-message ingest interrupted; "
                        "final messages may be absent from the plugin-local store"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end raw-message ingest skipped due to SQLite lock after short wait; "
                            "final messages may be absent from the plugin-local store: %s",
                            exc,
                        )
                        return
                    raise

                try:
                    self._lifecycle.finalize_session(
                        self._conversation_id,
                        session_id,
                        frontier_store_id=self._last_compacted_store_id,
                    )
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end lifecycle finalization interrupted; "
                        "raw messages may be ingested but lifecycle state may be finalized later"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end lifecycle finalization skipped due to SQLite lock after short wait; "
                            "raw messages were ingested but lifecycle state may be finalized later: %s",
                            exc,
                        )
                        return
                    raise
        except KeyboardInterrupt:
            logger.warning("LCM session-end ingest/finalize interrupted before bounded flush completed")
            return
        except Exception as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning(
                    "LCM session-end ingest/finalize skipped due to SQLite lock before bounded flush: %s",
                    exc,
                )
                return
            raise

    def on_session_reset(self) -> None:
        if self._host_fallback_compressor is not None:
            compressor = self._host_fallback_compressor
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        self._pending_reset_session_id = self._session_id
        self._pending_reset_conversation_id = self._conversation_id
        self._pending_reset_frontier_store_id = self._last_compacted_store_id
        background_reset_fenced = True
        if (
            self._background_compaction_enabled()
            and self._session_id
            and self._conversation_id
        ):
            try:
                background_reset_fenced = (
                    self._background_compaction.begin_session_reset(
                        conversation_id=self._conversation_id,
                        session_id=self._session_id,
                    )
                    is not None
                )
            except sqlite3.OperationalError as exc:
                # Never free canonical coverage while publication could still
                # race the batched reset. Preserving the derived nodes is the
                # fail-closed outcome; immutable raw messages are unaffected.
                background_reset_fenced = False
                logger.warning(
                    "LCM reset preserved DAG nodes because the background publication fence could not be acquired: %s",
                    exc,
                )
        super().on_session_reset()
        self._reset_session_scoped_runtime_state()

        # Retain DAG nodes across sessions based on config.
        #   -1  → keep all nodes
        #    0  → delete everything
        #    N  → keep nodes at depth >= N (e.g. 2 keeps d2+)
        retain = self._config.new_session_retain_depth
        if self._session_id and retain != -1 and background_reset_fenced:
            if retain == 0:
                self._dag.delete_session_nodes(
                    self._session_id,
                    on_deleted_batch=self._purge_embeddings_for_nodes,
                )
            else:
                self._dag.delete_below_depth(
                    self._session_id,
                    retain,
                    on_deleted_batch=self._purge_embeddings_for_nodes,
                )
        elif self._session_id and retain != -1 and not background_reset_fenced:
            logger.warning(
                "LCM reset skipped DAG pruning because publication was not durably fenced"
            )
        # Publish the new reset epoch only after all batched DAG mutation is
        # complete. A sibling process cannot reopen the closed fence while
        # deletion is in progress; the next foreground bind/reconciliation may
        # safely activate this completed epoch.
        self._lifecycle.record_reset(self._conversation_id)

    def _purge_embeddings_for_nodes(
        self,
        node_ids: "list[int]",
        *,
        connection: "sqlite3.Connection | None" = None,
    ) -> None:
        """Purge stored embeddings for deleted summary nodes (best effort).

        No-op unless embeddings are enabled. Opens a short-lived VectorStore on
        the shared DB; any failure is swallowed so a purge problem never breaks
        session reset — the summary_nodes join still keeps orphaned vectors out
        of ranking.
        """
        if not node_ids:
            return
        if not bool(getattr(self._config, "embeddings_enabled", False)):
            return
        try:
            from .vector_store import VectorStore

            if connection is not None:
                VectorStore.purge_embedding_batch_on_connection(connection, node_ids)
                return
            store = VectorStore(self._store.db_path, config=self._config)
            try:
                store.purge_embeddings_for_nodes(node_ids)
            finally:
                store.close()
        except Exception:  # pragma: no cover - defensive; purge is best-effort
            logger.debug(
                "LCM embedding purge for deleted nodes failed", exc_info=True
            )

    def _archive_chunks_for_messages(
        self,
        store_ids: "list[int]",
        *,
        connection: "sqlite3.Connection | None" = None,
    ) -> None:
        """Soft-archive raw-history chunks for purged/GC'd messages.

        Mirrors ``_purge_embeddings_for_nodes`` but for the chunk corpus: when a
        message is deleted or its content is GC-rewritten, its chunks no longer
        map to live content, so they are archived (dropped from ranking, rows
        retained). No-op unless embeddings are enabled. Standalone maintenance
        remains best-effort, but failures on a caller-supplied transaction must
        propagate so the paired raw rewrite is rolled back atomically.
        """
        if not store_ids:
            return
        if not bool(getattr(self._config, "embeddings_enabled", False)):
            return
        from .vector_store import VectorStore

        if connection is not None:
            for offset in range(0, len(store_ids), 256):
                VectorStore.archive_chunks_for_messages_on_connection(
                    connection, store_ids[offset:offset + 256]
                )
            return
        try:
            store = VectorStore(self._store.db_path, config=self._config)
            try:
                store.archive_chunks_for_messages(store_ids)
            finally:
                store.close()
        except Exception:  # pragma: no cover - defensive; archive is best-effort
            logger.debug(
                "LCM chunk archive for purged messages failed", exc_info=True
            )

    def carry_over_new_session_context(self, old_session_id: str, new_session_id: str) -> int:
        """Move retained summaries from the old session into the new one.

        This reassigns session ownership for retained summary nodes, but it does
        not rewrite the nodes' descendant raw-message lineage. Retrieval under
        ``session_scope='current'`` may therefore include a carried-over node in
        the new session, while ``source`` filtering still evaluates against the
        node's original descendant message sources.

        Temporal rollups are deliberately NOT re-scoped here: they are keyed by
        (period_kind, period_start, scope=session_id) with a UNIQUE constraint, so
        rewriting scope on rollover could collide with the new session's own
        rollups and would need a core-schema change to do safely. Rotation is the
        documented rollup scope boundary; ``lcm_recent`` compensates at read time
        by spanning the same current + last-finalized sessions its leaf fallback
        uses (see ``_recent_ready_rollups``), so no window content is dropped.
        """
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return 0
        if self._session_ignored and new_session_id == self._session_id:
            logger.debug(
                "LCM carry-over skipped for ignored session %s",
                new_session_id,
            )
            return 0
        return self._dag.reassign_session_nodes(old_session_id, new_session_id)

    def rollover_session(
        self,
        old_session_id: str,
        new_session_id: str,
        previous_messages: List[Dict[str, Any]] | None = None,
        carry_over_context: bool = True,
        **kwargs,
    ) -> int:
        """Complete a Hermes-style `/new` rollover for this engine.

        This is a small helper for host/runtime integrations that need the
        correct lifecycle ordering in one call:
        1. flush old-session messages into the store
        2. prune/reset retained DAG state on the old session
        3. bind the engine to the new session
        4. optionally move retained summaries into the new session
        """
        previous_messages = previous_messages or []
        boundary_reason = str(kwargs.get("boundary_reason") or "")
        conversation_id = self._conversation_id or old_session_id or new_session_id
        bound_session_id = self._session_id
        can_carry_over = bool(
            old_session_id and bound_session_id and old_session_id == bound_session_id
        )

        if carry_over_context and boundary_reason == "compression" and old_session_id and old_session_id != new_session_id:
            before_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            if can_carry_over:
                self.on_session_end(old_session_id, previous_messages)
            else:
                logger.warning(
                    "LCM compression rollover old_session_id=%s does not match bound session=%s; using boundary handler fallback",
                    old_session_id,
                    bound_session_id,
                )
            self.on_session_start(
                new_session_id,
                old_session_id=old_session_id,
                **kwargs,
            )
            after_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            return len(after_node_ids - before_node_ids)

        if old_session_id and can_carry_over:
            self.on_session_end(old_session_id, previous_messages)
            self.on_session_reset()
        elif old_session_id and not carry_over_context:
            logger.warning(
                "LCM rollover skipped old-session finalization: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )
        elif old_session_id and not can_carry_over:
            logger.warning(
                "LCM carry-over skipped: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )

        self.on_session_start(new_session_id, conversation_id=conversation_id, **kwargs)

        if not carry_over_context:
            return 0
        if old_session_id and not can_carry_over:
            return 0
        return self.carry_over_new_session_context(old_session_id, new_session_id)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            LCM_GREP,
            LCM_RECALL,
            LCM_QUERY_STATE,
            LCM_COMPUTE,
            LCM_COMPILE_EVIDENCE,
            LCM_EVIDENCE_PACK,
            LCM_RETRIEVE,
            LCM_RECENT,
            LCM_LOAD_SESSION,
            LCM_DESCRIBE,
            LCM_EXPAND,
            LCM_EXPAND_QUERY,
            LCM_STATUS,
            LCM_INSPECT,
            LCM_DOCTOR,
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        # Ingest live messages if passed (enables current-turn search)
        messages = kwargs.get("messages")

        if name != "lcm_inspect" and messages and self._session_id:
            if self._maybe_reclassify_current_session_as_auxiliary_before_message_ingest():
                self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            elif not (
                self._session_ignored or self._session_stateless or self._thread_context_stateless()
            ):
                try:
                    self._ingest_messages(messages)
                    self._record_ingest_success()
                    self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
                except Exception as e:
                    self._record_ingest_failure("tool-call ingest", e)

        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_recall": lcm_tools.lcm_recall,
            "lcm_query_state": lcm_tools.lcm_query_state,
            "lcm_compute": lcm_tools.lcm_compute,
            "lcm_compile_evidence": lcm_tools.lcm_compile_evidence,
            "lcm_evidence_pack": lcm_tools.lcm_evidence_pack,
            "lcm_retrieve": lcm_tools.lcm_retrieve,
            "lcm_recent": lcm_tools.lcm_recent,
            "lcm_load_session": lcm_tools.lcm_load_session,
            "lcm_describe": lcm_tools.lcm_describe,
            "lcm_expand": lcm_tools.lcm_expand,
            "lcm_expand_query": lcm_tools.lcm_expand_query,
            "lcm_status": lcm_tools.lcm_status,
            "lcm_inspect": lcm_tools.lcm_inspect,
            "lcm_doctor": lcm_tools.lcm_doctor,
        }
        handler = handlers.get(name)
        if handler:
            return handler(args, engine=self)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    def _database_path_source(self) -> str:
        if self._config.database_path:
            return "config.database_path"
        if self._hermes_home:
            return "hermes_home"
        return "default_home"

    def get_runtime_identity(self) -> Dict[str, Any]:
        """Return operator-facing identity for the loaded LCM runtime.

        The public identity follows the same foreground-session view as
        ``lcm_status`` and other tools. When a side-channel session is bound,
        the bound session details are still exposed separately for diagnostics.
        """
        metadata = _plugin_metadata()
        git_identity = _git_runtime_identity(_PLUGIN_ROOT)
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = None
        lifecycle_error = ""
        if conversation_id:
            try:
                lifecycle_state = self._lifecycle.get_by_conversation(conversation_id)
            except Exception as exc:  # pragma: no cover - defensive
                lifecycle_error = str(exc)

        identity: Dict[str, Any] = {
            "engine": self.name,
            "plugin_name": metadata.get("name", "hermes-lcm"),
            "plugin_version": metadata.get("version", "unknown"),
            "plugin_path": str(_PLUGIN_ROOT),
            "module_path": str(Path(__file__).resolve()),
            "hermes_home": str(self._hermes_home or ""),
            "database_path": str(self._store.db_path),
            "database_path_source": self._database_path_source(),
            "session_id": session_id,
            "session_platform": self.current_session_platform,
            "session_bound": bool(session_id),
            "conversation_id": conversation_id,
            "lifecycle_current_session_id": "",
            "lifecycle_last_finalized_session_id": "",
        }
        if self.side_channel_active:
            identity.update({
                "bound_session_id": self._session_id,
                "bound_session_platform": self._session_platform,
                "bound_conversation_id": self._conversation_id,
            })
        identity.update(git_identity)
        if lifecycle_state is not None:
            identity.update({
                "lifecycle_current_session_id": lifecycle_state.current_session_id or "",
                "lifecycle_last_finalized_session_id": lifecycle_state.last_finalized_session_id or "",
            })
        if lifecycle_error:
            identity["lifecycle_error"] = lifecycle_error
        return identity

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "compression_count": self.compression_count,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_total_tokens": self.last_total_tokens,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_cache_read_tokens": self.last_cache_read_tokens,
            "last_cache_write_tokens": self.last_cache_write_tokens,
            "last_reasoning_tokens": self.last_reasoning_tokens,
            "cache_metrics_available": self.cache_metrics_available,
            "cache_read_ratio": round(self.cache_read_ratio, 4),
            "raw_context_length": self.raw_context_length,
            "context_length": self.context_length,
            "effective_context_length_cap": self.effective_context_length_cap,
            "effective_context_length_reason": self.effective_context_length_reason,
            "threshold_tokens": self.threshold_tokens,
            "compression_policy": {
                "engine": "lcm",
                "ignored_stock_settings": [
                    "compression.threshold_tokens", "compression.target_ratio",
                ],
                "host_preflight_trigger_tokens": self.threshold_tokens,
                "publication_trigger_tokens": self._background_publication_trigger_tokens(),
                "emergency_trigger_tokens": self._background_emergency_trigger_tokens(),
                "assembly_cap_tokens": self._effective_assembly_token_cap(),
                "safe_input_tokens": getattr(self, "_admission_safe_input_tokens", 0),
                "target_policy": "LCM summary budgets and protected recent messages; not a fixed stock target ratio",
            },
            "last_compression_status": self._last_compression_status,
            "last_compression_noop_reason": self._last_compression_noop_reason,
            "last_foreground_deadline": dict(self._last_foreground_deadline),
            "threshold_full_sweep": dict(self._last_threshold_full_sweep),
            "ingest_failure_count": self._ingest_failure_count,
            "consecutive_ingest_failures": self._consecutive_ingest_failures,
            "last_ingest_error": self._last_ingest_error,
            "last_ingest_error_time": self._last_ingest_error_time,
            "model": self.model,
            "provider": self.provider,
            "context_length_source": self._context_length_source,
            "configured_context_threshold": self._config.context_threshold,
            "context_threshold": self.context_threshold,
            "context_reserve_tokens": self._config.context_reserve_tokens,
            "threshold_policy": (
                "fixed_reserve"
                if self._config.context_reserve_tokens > 0
                else "ratio"
            ),
            "context_threshold_source": self._context_threshold_source,
            "context_threshold_autoraised": self._context_threshold_autoraised,
            "config_sources": dict(getattr(self._config, "config_sources", {}) or {}),
            "config_source_warnings": list(getattr(self._config, "config_source_warnings", []) or []),
            "ignored_config_yaml_lcm_keys": list(getattr(self._config, "ignored_config_yaml_lcm_keys", []) or []),
            "focused_working_set": {
                "enabled": bool(self._config.focused_working_set_enabled),
                "target_tokens": int(self._config.focused_working_set_target_tokens),
                "trigger_tokens": int(self._config.focused_working_set_trigger_tokens),
                "min_reclaim_tokens": int(self._config.focused_working_set_min_reclaim_tokens),
                "min_reclaim_ratio": float(self._config.focused_working_set_min_reclaim_ratio),
                "expected_payback_calls": int(self._config.focused_working_set_payback_calls),
                "last_decision": dict(self._last_working_set_decision),
            },
            "active_replay_preview": {
                "enabled_for_current_session": bool(
                    self._active_replay_preview_policy().enabled
                ),
                "policy_version": self._active_replay_preview_policy().version,
                "last_decision": dict(self._last_active_replay_preview),
            },
            "compaction_payback": {
                "enabled": bool(self._config.compaction_payback_enabled),
                "expected_phase_calls": int(self._config.compaction_payback_expected_calls),
                "min_reclaim_tokens": int(self._config.compaction_payback_min_reclaim_tokens),
                "summary_ratio": float(self._config.compaction_payback_summary_ratio),
                "last_decision": dict(self._last_compaction_payback),
            },
        })
        status["background_compaction"] = self.get_background_compaction_status()
        with self._assertion_extraction_metrics_lock:
            status["assertion_extraction"] = {
                "store_enabled": self._assertions is not None,
                "enabled": bool(
                    getattr(self._config, "assertion_extraction_enabled", False)
                ),
                "model": self._assertion_extraction_model(),
                "extractor": (
                    getattr(self._assertion_extractor, "kind", "")
                    if self._assertion_extractor is not None
                    else ""
                ),
                "busy": not self._assertion_extraction_idle.is_set(),
                "batches_scheduled": self._assertion_extraction_batches_scheduled,
                "batches_skipped_busy": self._assertion_extraction_batches_skipped_busy,
                "sources_scheduled": self._assertion_extraction_sources_scheduled,
                "sources_completed": self._assertion_extraction_sources_completed,
                "sources_failed": self._assertion_extraction_sources_failed,
                "provider_calls": self._assertion_extraction_provider_calls,
                "input_tokens": self._assertion_extraction_input_tokens,
                "output_tokens": self._assertion_extraction_output_tokens,
                "last_duration_ms": round(
                    self._assertion_extraction_last_duration_ms, 3
                ),
                "last_error": self._assertion_extraction_last_error,
                "last_model": self._assertion_extraction_last_model,
            }
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = self._lifecycle.get_by_conversation(conversation_id) if conversation_id else None
        status["engine"] = "lcm"
        status["runtime_identity"] = self.get_runtime_identity()
        status["ingest_protection"] = sensitive_pattern_status(self._config)
        try:
            status["source_lineage"] = self._store.get_source_stats(session_id or None)
        except Exception as exc:  # pragma: no cover - defensive
            status["source_lineage"] = {"error": str(exc)}
        try:
            status["lifecycle_fragmentation"] = self._lifecycle.get_fragmentation_stats(
                state_db_path=self._state_db_path()
            )
        except Exception as exc:  # pragma: no cover - defensive
            status["lifecycle_fragmentation"] = {"error": str(exc), "read_only": True}
        try:
            rotate_backup_path = self.rotate_backup_path()
            status["rotate_backup_path"] = str(rotate_backup_path)
            # Single stat() to avoid a TOCTOU window where the rolling slot
            # could be atomically replaced between separate mtime and size reads.
            try:
                rotate_stat = rotate_backup_path.stat()
            except FileNotFoundError:
                rotate_stat = None
            if rotate_stat is not None:
                status["last_rotate_at"] = rotate_stat.st_mtime
                status["rotate_backup_size"] = rotate_stat.st_size
            else:
                status["last_rotate_at"] = None
                status["rotate_backup_size"] = 0
        except Exception as exc:  # pragma: no cover - defensive
            status["rotate_backup_path"] = None
            status["last_rotate_at"] = None
            status["rotate_backup_size"] = 0
            status["rotate_backup_error"] = str(exc)
        if session_id:
            status["store_messages"] = self._store.get_session_count(session_id)
            status["dag_nodes"] = self._dag.get_session_node_count(session_id)
            status["session_platform"] = self.current_session_platform
            status["session_ignored"] = self.current_session_ignored
            status["session_stateless"] = self.current_session_stateless
            status["ignore_session_patterns"] = list(self._config.ignore_session_patterns)
            status["stateless_session_patterns"] = list(self._config.stateless_session_patterns)
            status["ignore_message_patterns"] = list(self._config.ignore_message_patterns)
            status["ignore_session_patterns_source"] = self._config.ignore_session_patterns_source
            status["stateless_session_patterns_source"] = self._config.stateless_session_patterns_source
            status["ignore_message_patterns_source"] = self._config.ignore_message_patterns_source
            status["ignored_message_count"] = self._ignored_message_count
            status["ignore_pattern_dropped_count"] = self._ignore_pattern_dropped_count
            status["ingest_reconciliation"] = dict(self._last_ingest_reconciliation)
            status["overflow_recovery_failed"] = self._last_overflow_recovery_failed
            status["condensation_suppressed_reason"] = self._last_condensation_suppressed_reason
            status["conversation_id"] = conversation_id
            if lifecycle_state is not None:
                status["lifecycle"] = {
                    "conversation_id": lifecycle_state.conversation_id,
                    "current_session_id": lifecycle_state.current_session_id,
                    "last_finalized_session_id": lifecycle_state.last_finalized_session_id,
                    "current_frontier_store_id": lifecycle_state.current_frontier_store_id,
                    "last_finalized_frontier_store_id": lifecycle_state.last_finalized_frontier_store_id,
                    "debt_kind": lifecycle_state.debt_kind,
                    "debt_size_estimate": lifecycle_state.debt_size_estimate,
                    "current_bound_at": lifecycle_state.current_bound_at,
                    "last_finalized_at": lifecycle_state.last_finalized_at,
                    "debt_updated_at": lifecycle_state.debt_updated_at,
                    "last_maintenance_attempt_at": lifecycle_state.last_maintenance_attempt_at,
                    "last_rollover_at": lifecycle_state.last_rollover_at,
                    "last_reset_at": lifecycle_state.last_reset_at,
                    "updated_at": lifecycle_state.updated_at,
                }
            try:
                telemetry = self._store.read_compaction_telemetry(conversation_id)
            except Exception:
                telemetry = None
            if telemetry:
                status["compaction_telemetry"] = {
                    "cache_state": telemetry.get("cache_state", "unknown"),
                    "consecutive_cold_observations": telemetry.get(
                        "consecutive_cold_observations", 0
                    ),
                    "turns_since_leaf_compaction": telemetry.get(
                        "turns_since_leaf_compaction", 0
                    ),
                    "peak_prompt_tokens_since_leaf_compaction": telemetry.get(
                        "peak_prompt_tokens_since_leaf_compaction", 0
                    ),
                    "last_observed_prompt_tokens": telemetry.get(
                        "last_observed_prompt_tokens", 0
                    ),
                    "last_observed_cache_read": telemetry.get("last_observed_cache_read", 0),
                    "last_observed_cache_write": telemetry.get("last_observed_cache_write", 0),
                    "activity_band": telemetry.get("activity_band", "low"),
                    "total_compactions": telemetry.get("total_compactions", 0),
                    "last_leaf_compaction_at": telemetry.get("last_leaf_compaction_at"),
                    "last_compaction_duration_ms": telemetry.get("last_compaction_duration_ms"),
                    "provider": telemetry.get("provider"),
                    "model": telemetry.get("model"),
                    "last_api_call_at": telemetry.get("last_api_call_at"),
                }
        return status

    def bind_context_budget(self, budget: Any) -> None:
        super().bind_context_budget(budget)
        self._set_context_length(
            budget.context_window,
            source="host_context_budget",
            model=self.model,
            provider=self.provider,
        )

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "",
                     api_mode: str = "") -> None:
        parent_session_id = self._in_process_parent_session_id({})
        if parent_session_id:
            logger.debug(
                "LCM model update ignored for auxiliary child of %s",
                parent_session_id,
            )
            return
        for field in ("_admission_safe_input_tokens", "_admission_publication_tokens", "_admission_emergency_tokens", "_admission_assembly_tokens"):
            setattr(self, field, 0)
        self.model = str(model or "")
        self.base_url = str(base_url or "")
        self.api_key = str(api_key or "")
        self.provider = str(provider or "")
        self.api_mode = str(api_mode or "")
        self._set_context_length(context_length, source="update_model")
        self._update_model_pending_session_start = True

    def _refresh_session_filters(self) -> None:
        self._session_match_keys = build_session_match_keys(
            self._session_id,
            platform=self._session_platform,
        )
        self._session_ignored = matches_session_pattern(
            self._session_match_keys,
            self._compiled_ignore_session_patterns,
        )
        self._session_stateless = (
            not self._session_ignored
            and (
                (
                    self._lcm_current_start_allows_bypass_lineage
                    and self._has_lcm_bypass_lineage_session(self._session_id, platform=self._session_platform)
                )
                or matches_session_pattern(
                    self._session_match_keys,
                    self._compiled_stateless_session_patterns,
                )
            )
        )
        if self._session_id:
            self._lcm_session_last_platform[self._session_id] = self._session_platform
            self._lcm_session_last_bypassed[self._session_id] = bool(self._session_ignored or self._session_stateless)
            if not self._session_ignored and not self._session_stateless:
                self._lcm_non_bypass_platforms.setdefault(self._session_id, set()).add(self._session_platform)
                self._lcm_session_last_normal_platform[self._session_id] = self._session_platform
        if self._session_ignored or self._session_stateless:
            self._mark_lcm_bypass_lineage_session(self._session_id, platform=self._session_platform)

    def _log_session_filter_diagnostics(self) -> None:
        if not self._logged_filter_config:
            if self._config.ignore_session_patterns:
                logger.info(
                    "LCM ignore_session_patterns from %s: %s",
                    self._config.ignore_session_patterns_source,
                    ", ".join(self._config.ignore_session_patterns),
                )
            if self._config.stateless_session_patterns:
                logger.info(
                    "LCM stateless_session_patterns from %s: %s",
                    self._config.stateless_session_patterns_source,
                    ", ".join(self._config.stateless_session_patterns),
                )
            if self._config.ignore_message_patterns:
                logger.info(
                    "LCM ignore_message_patterns from %s: %s",
                    self._config.ignore_message_patterns_source,
                    ", ".join(self._config.ignore_message_patterns),
                )
            self._logged_filter_config = True
        if self._session_ignored:
            logger.info(
                "LCM session %s matched ignore_session_patterns via %s — skipping writes and compaction",
                self._session_id,
                ", ".join(self._session_match_keys),
            )
        elif self._session_stateless:
            logger.info(
                "LCM session %s matched stateless_session_patterns via %s — read-only mode (no LCM writes)",
                self._session_id,
                ", ".join(self._session_match_keys),
            )

    # -- Internal: message ingestion ---------------------------------------

    def _schedule_ingest_cursor_reconciliation(self) -> None:
        """Mark existing-session rebinds for cursor repair on next ingest."""
        self._ingest_cursor_needs_reconcile = False
        if not self._session_id or self._session_ignored or self._session_stateless:
            return
        try:
            self._ingest_cursor_needs_reconcile = self._store.get_session_count(self._session_id) > 0
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation probe failed: %s", exc)
            self._ingest_cursor_needs_reconcile = False

    def _stored_row_externalized_text_parts_for_pattern_matching(self, msg: Dict[str, Any]) -> list[str]:
        ref_sources: list[str] = []
        content = msg.get("content")
        if isinstance(content, str):
            ref_sources.append(content)
        tool_calls = msg.get("tool_calls")
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
        session_id = str(msg.get("session_id") or self._session_id or "")
        for ref in refs:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if not payload:
                continue
            payload_session_id = str(payload.get("session_id") or "")
            if session_id and payload_session_id and payload_session_id != session_id:
                continue
            payload_content = payload.get("content")
            if isinstance(payload_content, str):
                parts.append(payload_content)
        return parts

    def _stored_row_externalized_text_for_pattern_matching(self, msg: Dict[str, Any]) -> str:
        return "\n".join(self._stored_row_externalized_text_parts_for_pattern_matching(msg))

    def _is_cached_active_replay_message_at_index(self, idx: int, msg: Dict[str, Any]) -> bool:
        if idx < 0 or idx >= len(self._last_active_replay_messages):
            return False
        return self._message_replay_identity(msg) == self._message_replay_identity(
            self._last_active_replay_messages[idx]
        )

    def _matches_ignore_message_patterns(self, msg: Dict[str, Any], *, stored_row: bool = False) -> bool:
        if not self._compiled_ignore_message_patterns:
            return False
        content = msg.get("content")
        text = (
            stored_text_content_for_pattern_matching(content)
            if stored_row
            else text_content_for_pattern_matching(content)
        ) or ""
        if matches_message_pattern(text, self._compiled_ignore_message_patterns):
            return True
        if stored_row:
            externalized_parts = self._stored_row_externalized_text_parts_for_pattern_matching(msg)
            for externalized_text in externalized_parts:
                if externalized_text and matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns):
                    return True
            externalized_text = "\n".join(externalized_parts)
            if externalized_text and externalized_text != text:
                return matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns)
        return False

    def _content_has_externalized_placeholder_ref(self, content: str) -> bool:
        return bool(extract_externalized_ref(content) or extract_ingest_externalized_refs(content))

    def _has_prior_raw_externalized_placeholder_row(
        self,
        store_id: int,
        msg: Dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> bool:
        if not self._session_id:
            return False
        raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
        after_store_id = 0
        while True:
            rows = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=after_store_id,
                limit=1000,
                conversation_id=self._conversation_id,
                lock_timeout_seconds=(
                    self._foreground_deadline_remaining(deadline)
                ),
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
            if not rows:
                return False
            for row_index, row in enumerate(rows):
                if row_index % 32 == 0:
                    self._foreground_deadline_checkpoint(
                        deadline,
                        "ingest filtering",
                        "prior externalized-row scan",
                    )
                row_store_id = int(row.get("store_id") or 0)
                if row_store_id >= store_id:
                    return False
                if self._raw_externalized_placeholder_replay_identity(row) == raw_identity:
                    return True
                after_store_id = max(after_store_id, row_store_id)

    def _mapped_stored_row_matches_ignore_message_patterns(
        self,
        msg: Dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> bool:
        if not self._compiled_ignore_message_patterns:
            # Every branch below ends in _matches_ignore_message_patterns,
            # which is False without patterns; skip the store lookups.
            return False
        store_id = msg.get("store_id")
        content = normalize_content_value(msg.get("content")) or ""
        has_externalized_placeholder = self._content_has_externalized_placeholder_ref(content)
        mapped_from_active_placeholder = False
        if store_id is None:
            store_id = self._current_compress_store_ids_by_message_id.get(id(msg))
            mapped_from_active_placeholder = has_externalized_placeholder and store_id is not None
        if store_id is None:
            return False
        try:
            if (
                mapped_from_active_placeholder
                and self._has_prior_raw_externalized_placeholder_row(
                    int(store_id),
                    msg,
                    deadline=deadline,
                )
            ):
                raw_identity = self._raw_externalized_placeholder_replay_identity(
                    msg
                )
                if (
                    self._current_compress_placeholder_identity_counts.get(
                        raw_identity,
                        0,
                    )
                    <= 1
                ):
                    return False
            stored = self._store.get(
                int(store_id),
                lock_timeout_seconds=(
                    self._foreground_deadline_remaining(deadline)
                ),
                deadline=deadline,
                deadline_clock=self._foreground_now,
            )
        except TimeoutError as exc:
            if deadline is not None:
                raise ForegroundDeadlineExceeded(
                    "ingest filtering",
                    "message-store lookup exceeded the shared deadline",
                ) from exc
            logger.debug("LCM stored ignore-pattern lookup failed", exc_info=True)
            return False
        except sqlite3.OperationalError as exc:
            if deadline is not None and _is_sqlite_locked_error(exc):
                raise ForegroundDeadlineExceeded(
                    "ingest filtering",
                    "SQLite contention prevented a bounded message lookup",
                ) from exc
            logger.debug("LCM stored ignore-pattern lookup failed", exc_info=True)
            return False
        except Exception:
            logger.debug("LCM stored ignore-pattern lookup failed", exc_info=True)
            return False
        return bool(stored and self._matches_ignore_message_patterns(stored, stored_row=True))

    def _copy_active_replay_messages_preserving_generated_ids(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        copied_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in active_replay_messages:
            copied_message = dict(message)
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(copied_message))
            copied_replay_messages.append(copied_message)
        return copied_replay_messages

    def _remember_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
        active_replay_messages: List[Dict[str, Any]],
        *,
        store_ids_by_index: Optional[List[Optional[int]]] = None,
        explicit_store_id_indices: Optional[set[int]] = None,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]]:
        if (
            store_ids_by_index is None
            or len(store_ids_by_index) != len(active_replay_messages)
        ):
            store_ids_by_message_id = ReconcileMixin._get_store_id_map_for_messages(
                self,
                original_messages,
                deadline=deadline,
            )
            store_ids_by_index = [
                store_ids_by_message_id.get(id(message))
                for message in original_messages
            ]
            if len(store_ids_by_index) < len(active_replay_messages):
                store_ids_by_index.extend(
                    [None] * (len(active_replay_messages) - len(store_ids_by_index))
                )
            elif len(store_ids_by_index) > len(active_replay_messages):
                store_ids_by_index = store_ids_by_index[: len(active_replay_messages)]

        self._last_active_replay_source_identities = [
            self._message_replay_identity(message) for message in original_messages
        ]
        cached_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
            active_replay_messages
        )
        self._last_active_replay_messages = cached_replay_messages
        self._last_active_replay_store_ids = list(store_ids_by_index)
        self._last_active_replay_explicit_store_id_indices = set(
            explicit_store_id_indices or ()
        )
        self._remember_active_replay_store_refs(
            original_messages,
            active_replay_messages,
            cached_replay_messages,
            store_ids_by_index=store_ids_by_index,
            explicit_store_id_indices=(
                self._last_active_replay_explicit_store_id_indices
            ),
        )
        # Placeholder counts/ordinals are regenerable replay metadata, not raw
        # source truth. A deadline-bound ingest has already committed its raw
        # rows by the time this cache is remembered; synchronously persisting
        # derived metadata here could extend the caller wall and mutate SQLite
        # after timeout. Manual/background callers keep the historical durable
        # update, while automatic foreground turns safely defer it to a later
        # no-deadline maintenance path.
        if deadline is None:
            self._write_generated_ignored_placeholder_hash_counts(
                self._generated_placeholder_digest_budget_for_active_replay(
                    active_replay_messages,
                    deadline=None,
                )
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                self._generated_placeholder_digest_ordinals_for_active_replay(
                    active_replay_messages,
                    deadline=None,
                )
            )
        return active_replay_messages

    def _remember_active_replay_store_refs(
        self,
        original_messages: List[Dict[str, Any]],
        active_replay_messages: List[Dict[str, Any]],
        cached_replay_messages: List[Dict[str, Any]],
        *,
        store_ids_by_index: List[Optional[int]],
        explicit_store_id_indices: set[int],
    ) -> None:
        """Keep exact-object lineage for an active view with reordered rows."""
        refs: dict[int, tuple[Dict[str, Any], Optional[int]]] = {}
        for index, store_id in enumerate(store_ids_by_index):
            if index not in explicit_store_id_indices:
                continue
            for collection in (
                original_messages,
                active_replay_messages,
                cached_replay_messages,
            ):
                if index >= len(collection):
                    continue
                message = collection[index]
                refs[id(message)] = (message, store_id)
        self._active_replay_store_refs_by_message_id = refs
        self._active_replay_store_ref_session_id = str(self._session_id or "")
        self._active_replay_store_ref_conversation_id = str(
            self._conversation_id or ""
        )

    def _align_ingest_cursor_to_active_context(
        self,
        active_messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> None:
        """Make a transformed context the exact basis for future appends."""
        ids_by_message_id = ReconcileMixin._get_store_id_map_for_messages(
            self,
            active_messages,
            deadline=deadline,
        )
        store_ids_by_index = [
            ids_by_message_id.get(id(message)) for message in active_messages
        ]
        cached_replay_messages = (
            self._copy_active_replay_messages_preserving_generated_ids(
                active_messages
            )
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "active context alignment",
            "before runtime cursor update",
        )
        self._ingest_cursor = len(active_messages)
        self._ingest_cursor_needs_reconcile = False
        self._last_active_replay_source_identities = [
            self._message_replay_identity(message) for message in active_messages
        ]
        self._last_active_replay_messages = cached_replay_messages
        self._last_active_replay_store_ids = store_ids_by_index
        self._last_active_replay_explicit_store_id_indices = set()
        self._remember_active_replay_store_refs(
            active_messages,
            active_messages,
            cached_replay_messages,
            store_ids_by_index=store_ids_by_index,
            explicit_store_id_indices=set(),
        )

    def supports_bounded_admission_recovery(self) -> bool:
        """Advertise the automatic, deterministic recovery lane to the host."""
        return self._background_compaction_enabled()

    def _reused_cached_suffix_indices(
        self,
        previous_identities: List[tuple[Any, ...]],
        incoming_identities: List[tuple[Any, ...]],
        *,
        common_prefix: int,
        deadline: float | None,
    ) -> dict[int, int]:
        """Map unchanged previous blocks onto their incoming positions.

        The host can replace, insert, or delete messages inside an already
        ingested active list while preserving its later messages.  Find the
        longest suffix of the previous list that still occurs contiguously in
        the incoming list, using a linear KMP scan over the reversed identities.
        Unique ordered anchors also recover unchanged islands between edits.
        Values map incoming indices to their cached indices.
        """
        previous_tail = previous_identities[common_prefix:]
        incoming_tail = incoming_identities[common_prefix:]
        if not previous_tail or not incoming_tail:
            return {}

        pattern = list(reversed(previous_tail))
        failure = [0] * len(pattern)
        for index in range(1, len(pattern)):
            if index % 64 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "ingest replay validation",
                )
            matched = failure[index - 1]
            while matched and pattern[matched] != pattern[index]:
                matched = failure[matched - 1]
            if pattern[matched] == pattern[index]:
                matched += 1
            failure[index] = matched

        matched = 0
        best_length = 0
        best_incoming_start = -1
        incoming_tail_length = len(incoming_tail)
        for reverse_index, identity in enumerate(reversed(incoming_tail)):
            if reverse_index % 64 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "ingest replay validation",
                )
            while matched and pattern[matched] != identity:
                matched = failure[matched - 1]
            if pattern[matched] == identity:
                matched += 1
            incoming_start = (
                common_prefix + incoming_tail_length - 1 - reverse_index
            )
            if matched > best_length or (
                matched == best_length
                and matched > 0
                and (
                    best_incoming_start < 0
                    or incoming_start < best_incoming_start
                )
            ):
                best_length = matched
                best_incoming_start = incoming_start
            if matched == len(pattern):
                matched = failure[matched - 1]

        reused = {}
        if best_length > 0:
            previous_start = len(previous_identities) - best_length
            reused = {
                best_incoming_start + offset: previous_start + offset
                for offset in range(best_length)
            }

        # More than one host rewrite can leave unchanged islands before the
        # final suffix. Reusing only that suffix re-ingests those islands on
        # every bind. Unique identities provide ordered, unambiguous anchors;
        # repeated text alone is never enough to suppress a new occurrence.
        from collections import Counter
        previous_counts = Counter(previous_tail)
        incoming_counts = Counter(incoming_tail)
        previous_positions = {
            identity: common_prefix + index
            for index, identity in enumerate(previous_tail)
            if previous_counts[identity] == 1
        }
        last_previous = common_prefix - 1
        incoming_limit = best_incoming_start if reused else len(incoming_identities)
        previous_limit = min(reused.values()) if reused else len(previous_identities)
        for incoming_index in range(common_prefix, incoming_limit):
            if incoming_index % 64 == 0:
                self._foreground_deadline_checkpoint(deadline, "ingest replay validation")
            identity = incoming_identities[incoming_index]
            previous_index = previous_positions.get(identity)
            if (
                incoming_counts[identity] == 1
                and previous_index is not None
                and last_previous < previous_index < previous_limit
            ):
                reused[incoming_index] = previous_index
                last_previous = previous_index
        # Extend exact adjacent blocks between anchors without a quadratic
        # diff or changing the order/multiplicity of repeated messages.
        anchors = [(common_prefix - 1, common_prefix - 1), *sorted(reused.items()),
                   (len(incoming_identities), len(previous_identities))]
        for (left_i, left_p), (right_i, right_p) in zip(anchors, anchors[1:]):
            i, pidx = left_i + 1, left_p + 1
            while i < right_i and pidx < right_p and incoming_identities[i] == previous_identities[pidx]:
                if i % 64 == 0:
                    self._foreground_deadline_checkpoint(deadline, "ingest replay validation")
                reused[i] = pidx
                i += 1
                pidx += 1
        return reused

    def _cached_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        identities = [self._message_replay_identity(message) for message in original_messages]
        if identities == getattr(self, "_last_active_replay_source_identities", None):
            cached = getattr(self, "_last_active_replay_messages", None)
            if cached is not None:
                copied = self._copy_active_replay_messages_preserving_generated_ids(cached)
                store_ids = list(
                    getattr(self, "_last_active_replay_store_ids", [])
                )
                if len(store_ids) != len(copied):
                    store_ids = [None] * len(copied)
                explicit_store_id_indices = set(
                    getattr(
                        self,
                        "_last_active_replay_explicit_store_id_indices",
                        set(),
                    )
                )
                self._remember_active_replay_store_refs(
                    original_messages,
                    copied,
                    cached,
                    store_ids_by_index=store_ids,
                    explicit_store_id_indices=explicit_store_id_indices,
                )
                return copied
        return None

    def _is_replayed_context_scaffold_message(self, msg: Dict[str, Any]) -> bool:
        """Return true for active-context scaffolding that should not be re-ingested."""
        role = str(msg.get("role") or "")
        content = normalize_content_value(msg.get("content")) or ""
        if role == "system":
            return (
                "[Note: This conversation uses Lossless Context Management (LCM)." in content
                and "Earlier turns have been compacted into hierarchical summaries below." in content
            )
        if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX):
            return True
        if content.lstrip().startswith(_PRESERVED_TODO_CONTEXT_PREFIX):
            return True
        if content.lstrip().startswith("<relevant-memories>"):
            return True
        if content.lstrip().startswith(WORKING_SET_REFERENCE_PREFIX):
            return True
        if "[Expand for details:" not in content:
            return False
        return bool(
            re.search(
                r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]",
                content,
            )
        )

    def _restore_ingest_payload_placeholders_in_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                self._restore_ingest_payload_placeholders_in_value(key, session_id=session_id)
                if isinstance(key, str)
                else key: self._restore_ingest_payload_placeholders_in_value(val, session_id=session_id)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._restore_ingest_payload_placeholders_in_value(item, session_id=session_id) for item in value]
        if isinstance(value, str):
            return restore_ingest_payload_placeholders(
                value,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        return value

    def _restore_ingest_payload_placeholders_in_content_identity(self, content: str, *, session_id: str) -> str:
        if not content:
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return restore_ingest_payload_placeholders(
                content,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        restore_as_structured = False
        if isinstance(decoded, (dict, list)) and normalize_content_value(decoded) == content:
            for ref in extract_ingest_externalized_refs(content):
                payload = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                payload_session_id = (payload or {}).get("session_id") or ""
                if session_id and payload_session_id and payload_session_id != session_id:
                    continue
                field_path = str((payload or {}).get("field_path") or "")
                if field_path and field_path != "content":
                    restore_as_structured = True
                    break
        if restore_as_structured:
            restored = self._restore_ingest_payload_placeholders_in_value(decoded, session_id=session_id)
            return normalize_content_value(restored) or ""
        return restore_ingest_payload_placeholders(
            content,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )

    def _recovered_content_matches_durable_identity(self, recovered_content: str, durable_content: str) -> bool:
        recovered_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                self._config,
                parse_json_strings=False,
            )
        )
        if recovered_identity_content == durable_content:
            return True
        redaction_names = sorted(set(re.findall(r"\[LCM sensitive redaction: name=([^;\]]+)", durable_content)))
        if not redaction_names or bool(getattr(self._config, "sensitive_patterns_enabled", False)):
            return False
        compat_config = copy.copy(self._config)
        compat_config.sensitive_patterns_enabled = True
        compat_config.sensitive_patterns = redaction_names
        compat_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                compat_config,
                parse_json_strings=False,
            )
        )
        return compat_identity_content == durable_content

    @staticmethod
    def _persisted_output_marker_replay_proof(content: str) -> tuple[str | None, bool]:
        inline_preview_sha256 = _persisted_output_inline_preview_sha256(content)
        preview_sha256 = inline_preview_sha256 or _persisted_output_preview_prefix_digest(content)
        if not preview_sha256:
            return None, False
        allow_redacted_preview_match = inline_preview_sha256 is None and not _has_lossy_sensitive_redaction(content)
        return preview_sha256, allow_redacted_preview_match

    def _has_any_durable_persisted_output_payload_for_marker(self, msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if expected_chars is None or not persisted_output_source_path or not persisted_output_preview_sha256:
            return False
        if recover_hermes_persisted_output_with_file_stat(content) is None:
            return False
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=str(msg.get("session_id") or self._session_id or ""),
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return durable_content is not None

    @classmethod
    def _is_active_context_droppable_identity(cls, identity: tuple[str, str, str, str]) -> bool:
        """Return true for durable rows sanitized out of active replay only."""
        role, content, _tool_call_id, tool_calls = identity
        if role != "assistant" or tool_calls:
            return False
        return _should_drop_active_assistant_message({
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        })

    def _ignored_message_is_quarantinable_assistant(self, msg: Dict[str, Any]) -> bool:
        if self._is_volatile_ignored_quarantine_placeholder(
            msg,
            text_content_for_pattern_matching(msg.get("content")) or "",
        ):
            return True
        identity = self._message_replay_identity(msg)
        if self._is_quarantined_assistant_replay_identity(identity):
            return True
        if not self._matches_ignore_message_patterns(msg):
            return False
        if identity[0] != "assistant":
            return False
        content = normalize_content_value(msg.get("content")) or ""
        return assistant_output_quarantine_reason(content) is not None

    def _redact_active_replay_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        redacted_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in messages:
            redacted_message = dict(message)
            if "content" in redacted_message:
                redacted_content = redact_sensitive_value(
                    redacted_message.get("content"),
                    self._config,
                    parse_json_strings=False,
                )
                redacted_message["content"] = redacted_content

            if "tool_calls" in redacted_message:
                redacted_message["tool_calls"] = redact_sensitive_value(
                    redacted_message.get("tool_calls"),
                    self._config,
                    parse_json_strings=True,
                )
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(redacted_message))
            redacted_replay_messages.append(redacted_message)
        return redacted_replay_messages

    def _call_foreground_projection(
        self,
        operation,
        *args,
        deadline: float | None,
        stage: str,
        **kwargs,
    ):
        """Bound a foreground-owned file projection by the caller's wall."""
        if deadline is None:
            return operation(*args, **kwargs)
        cancel_event = threading.Event()
        guarded_kwargs = dict(kwargs)
        timeout_error = ForegroundDeadlineExceeded(
            stage,
            "foreground projection exceeded the shared deadline",
        )
        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is None or remaining <= 0:
            cancel_event.set()
            raise timeout_error
        try:
            accepts_ownership = "ownership" in inspect.signature(
                operation
            ).parameters
        except (TypeError, ValueError):
            accepts_ownership = False
        projection_ownership = guarded_kwargs.get("ownership")
        owns_projection = accepts_ownership and projection_ownership is None
        if owns_projection:
            projection_ownership = self._new_foreground_externalization_ownership(
                deadline,
            )
            guarded_kwargs["ownership"] = projection_ownership
        for name, value in (
            ("deadline", deadline),
            ("cancel_event", cancel_event),
            ("now", self._foreground_now),
        ):
            if self._summary_callable_accepts_keyword(operation, name):
                guarded_kwargs[name] = value
        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is None or remaining <= 0:
            cancel_event.set()
            if owns_projection:
                projection_ownership.rollback()
            raise timeout_error
        slot = self._foreground_projection_call_slot
        if not slot.acquire(blocking=False):
            cancel_event.set()
            if owns_projection:
                projection_ownership.rollback()
            raise timeout_error
        process_slot = _FOREGROUND_PROJECTION_PROCESS_SLOT
        if not process_slot.acquire(blocking=False):
            slot.release()
            cancel_event.set()
            if owns_projection:
                projection_ownership.rollback()
            raise timeout_error

        completed = threading.Event()
        result_box: dict[str, Any] = {}

        def invoke() -> None:
            try:
                if cancel_event.is_set() or self._foreground_now() >= float(deadline):
                    return
                result_box["value"] = operation(*args, **guarded_kwargs)
            except BaseException as exc:
                result_box["error"] = exc
            finally:
                process_slot.release()
                slot.release()
                completed.set()

        worker = threading.Thread(
            target=invoke,
            name="lcm-foreground-projection",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            process_slot.release()
            slot.release()
            if owns_projection:
                projection_ownership.rollback()
            raise
        remaining = self._foreground_deadline_remaining(deadline)
        try:
            finished = (
                remaining is not None
                and completed.wait(timeout=max(0.0, remaining))
            )
        except BaseException:
            cancel_event.set()
            if owns_projection:
                projection_ownership.rollback()
            raise
        if not finished or self._foreground_now() >= float(deadline):
            cancel_event.set()
            if owns_projection:
                projection_ownership.rollback()
            raise timeout_error
        if "error" in result_box:
            if owns_projection:
                projection_ownership.rollback()
            raise result_box["error"]
        if owns_projection:
            try:
                projection_ownership.publish(
                    deadline=deadline,
                    now=self._foreground_now,
                )
            except TimeoutError:
                cancel_event.set()
                projection_ownership.rollback()
                raise timeout_error
            except BaseException:
                projection_ownership.rollback()
                raise
            if self._foreground_now() >= float(deadline):
                cancel_event.set()
                projection_ownership.rollback()
                raise timeout_error
            projection_ownership.commit()
        return result_box.get("value")

    def _new_foreground_externalization_ownership(
        self,
        deadline: float,
    ) -> ExternalizationOwnership:
        """Prepare storage before an abandonable worker can queue file intents."""
        self._foreground_deadline_checkpoint(
            deadline,
            "externalization storage preparation",
        )
        storage_dir = getattr(
            self,
            "_foreground_externalization_storage_dir",
            None,
        )
        if storage_dir is None:
            try:
                # Resolve the configured name without creating or chmodding
                # anything. A worker may perform bounded read-only existence
                # checks, but only the joined foreground caller may write.
                storage_dir = get_large_output_storage_dir(
                    self._config,
                    hermes_home=self._hermes_home,
                    create=False,
                )
            except (OSError, ValueError):
                storage_dir = None
        self._foreground_deadline_checkpoint(
            deadline,
            "externalization storage preparation",
        )
        return ExternalizationOwnership(
            defer_publication=True,
            storage_dir=storage_dir,
        )

    def _ingest_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]]:
        if deadline is None:
            return self._ingest_messages_with_externalization_scope(
                messages,
                deadline=None,
                ownership=None,
            )

        ownership = self._new_foreground_externalization_ownership(deadline)
        try:
            replay = self._ingest_messages_with_externalization_scope(
                messages,
                deadline=deadline,
                ownership=ownership,
            )
            ownership.publish(
                deadline=deadline,
                now=self._foreground_now,
            )
            self._foreground_deadline_checkpoint(deadline, "ingest publication")
        except BaseException:
            ownership.rollback()
            raise
        ownership.commit()
        return replay

    def _ingest_messages_with_externalization_scope(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None,
        ownership: ExternalizationOwnership | None,
    ) -> List[Dict[str, Any]]:
        """Persist new messages to the store.

        Uses a cursor to track which portion of the current messages list
        has already been persisted.  After compress() shortens the list,
        the cursor is reset to len(compressed), so only messages appended
        after compaction are ingested — regardless of how the store count
        compares to the current list length.

        Returns a replay-safe copy of ``messages`` with obviously broken
        assistant loops replaced by quarantine placeholders. Existing callers may
        ignore the return value when they only need durable persistence.
        """
        self._foreground_deadline_checkpoint(deadline, "ingest")
        if not self._session_id:
            logger.debug("Ingest skipped: no session_id")
            return self._redact_active_replay_messages(messages)

        if self._session_ignored or self._session_stateless:
            logger.debug(
                "Ingest skipped for %s session %s",
                "ignored" if self._session_ignored else "stateless",
                self._session_id,
            )
            return self._redact_active_replay_messages(messages)

        n = len(messages)
        previous_cursor = max(self._ingest_cursor, 0)
        cursor = min(previous_cursor, n)
        reused_cached_replay_indices: dict[int, int] = {}
        active_store_ids_by_index: List[Optional[int]] = [None] * n
        explicit_store_id_indices: set[int] = set()
        cached_active_store_ids = list(
            getattr(self, "_last_active_replay_store_ids", [])
        )
        cached_explicit_store_id_indices = set(
            getattr(
                self,
                "_last_active_replay_explicit_store_id_indices",
                set(),
            )
        )
        if cursor > 0 and not self._ingest_cursor_needs_reconcile:
            cached_source_identities = getattr(
                self,
                "_last_active_replay_source_identities",
                None,
            )
            cached_active_replay_messages = getattr(
                self,
                "_last_active_replay_messages",
                None,
            )
            if (
                cached_source_identities is None
                or cached_active_replay_messages is None
                or len(cached_source_identities) < previous_cursor
                or len(cached_active_replay_messages) < previous_cursor
            ):
                self._ingest_cursor_needs_reconcile = True
            else:
                matching_cursor = 0
                for index, message in enumerate(messages[:cursor]):
                    if index % 32 == 0:
                        self._foreground_deadline_checkpoint(
                            deadline,
                            "ingest replay validation",
                        )
                    if (
                        self._message_replay_identity(message)
                        != cached_source_identities[index]
                    ):
                        break
                    matching_cursor += 1
                if matching_cursor < cursor:
                    logger.info(
                        "LCM rewound the in-memory ingest cursor after an active "
                        "prefix rewrite: session=%s cursor=%d->%d",
                        self._session_id,
                        cursor,
                        matching_cursor,
                    )
                    self._ingest_cursor = matching_cursor
                    cursor = matching_cursor
                    incoming_identities: List[tuple[Any, ...]] = []
                    for index, message in enumerate(messages):
                        if index % 32 == 0:
                            self._foreground_deadline_checkpoint(
                                deadline,
                                "ingest replay validation",
                            )
                        incoming_identities.append(
                            self._message_replay_identity(message)
                        )
                    reused_cached_replay_indices = (
                        self._reused_cached_suffix_indices(
                            cached_source_identities[:previous_cursor],
                            incoming_identities,
                            common_prefix=matching_cursor,
                            deadline=deadline,
                        )
                    )
                    if reused_cached_replay_indices:
                        explicit_store_id_indices.update(
                            reused_cached_replay_indices
                        )
                        for incoming_index, cached_index in (
                            reused_cached_replay_indices.items()
                        ):
                            if cached_index < len(cached_active_store_ids):
                                active_store_ids_by_index[incoming_index] = (
                                    cached_active_store_ids[cached_index]
                                )
                        logger.info(
                            "LCM reused %d unchanged active suffix messages "
                            "after an in-memory prefix rewrite: session=%s",
                            len(reused_cached_replay_indices),
                            self._session_id,
                        )
                for index in range(matching_cursor):
                    if index < len(cached_active_store_ids):
                        active_store_ids_by_index[index] = (
                            cached_active_store_ids[index]
                        )
                    if index in cached_explicit_store_id_indices:
                        explicit_store_id_indices.add(index)
        scan_start = 0 if self._ingest_cursor_needs_reconcile else cursor
        ignored_original_messages = [False] * n
        if self._compiled_ignore_message_patterns:
            previous_store_id_map = self._current_compress_store_ids_by_message_id
            self._current_compress_store_ids_by_message_id = (
                self._get_store_id_map_for_messages(
                    messages,
                    deadline=deadline,
                )
            )
            try:
                for idx in range(scan_start, n):
                    if idx % 32 == 0:
                        self._foreground_deadline_checkpoint(
                            deadline,
                            "ingest filtering",
                        )
                    mapped_ignore = (
                        self._mapped_stored_row_matches_ignore_message_patterns(
                            messages[idx],
                            deadline=deadline,
                        )
                    )
                    ignored_original_messages[idx] = (
                        self._matches_ignore_message_patterns(messages[idx])
                        or mapped_ignore
                    )
            finally:
                self._current_compress_store_ids_by_message_id = previous_store_id_map
        externalize_messages = [False] * n
        prefer_existing_externalized = [False] * n
        for idx in range(scan_start, n):
            if idx % 64 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "ingest preparation",
                )
            externalize_messages[idx] = not ignored_original_messages[idx]
        for idx in range(0, scan_start):
            prefer_existing_externalized[idx] = not ignored_original_messages[idx]
        replay_messages = self._call_foreground_projection(
            quarantine_suspicious_assistant_messages,
            messages,
            deadline=deadline,
            stage="ingest quarantine",
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
            externalize=externalize_messages,
            prefer_existing_externalized=prefer_existing_externalized,
            ownership=ownership,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "ingest quarantine",
        )
        replay_messages = self._redact_active_replay_messages(replay_messages)
        replay_messages = self._apply_ignored_active_replay_placeholders(
            messages,
            replay_messages,
            scan_start=scan_start,
            ignored_messages=ignored_original_messages,
            persist_metadata=deadline is None,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "ingest replay cleanup",
        )
        if reused_cached_replay_indices:
            cached_replay = self._copy_active_replay_messages_preserving_generated_ids(
                self._last_active_replay_messages
            )
            replay_messages = list(replay_messages)
            for incoming_index, cached_index in reused_cached_replay_indices.items():
                replay_messages[incoming_index] = cached_replay[cached_index]
        if self._ingest_cursor_needs_reconcile:
            reconcile_messages = [
                original_msg
                if (
                    (
                        str(original_msg.get("role") or "") == "tool"
                        and _is_hermes_persisted_output_marker(
                            normalize_content_value(original_msg.get("content")) or ""
                        )
                        and self._has_any_durable_persisted_output_payload_for_marker(original_msg)
                    )
                    or (
                        self._compiled_ignore_message_patterns
                        and ignored_original_messages[idx]
                    )
                )
                else replay_msg
                for idx, (original_msg, replay_msg) in enumerate(zip(messages, replay_messages))
            ]
            try:
                if deadline is None:
                    self._ingest_cursor = self._reconcile_ingest_cursor_from_store(
                        reconcile_messages
                    )
                else:
                    self._ingest_cursor = self._reconcile_ingest_cursor_from_store(
                        reconcile_messages,
                        deadline=deadline,
                    )
                self._ingest_cursor_needs_reconcile = False
            except ReconcileBudgetExceeded:
                # Reconciliation outlived its budget: abort this ingest batch
                # entirely (never ingest duplicates past an unreconciled
                # cursor), keep the needs-reconcile flag so the next turn
                # retries with a warm payload index, and unblock turn
                # finalization — the post-LLM hook must never wedge a turn.
                logger.warning(
                    "LCM ingest reconcile budget exceeded: session=%s incoming=%d — "
                    "deferring ingest to next turn",
                    self._session_id,
                    len(reconcile_messages),
                )
                raise
        cursor = min(max(self._ingest_cursor, 0), n)
        self._foreground_deadline_checkpoint(deadline, "ingest")
        if cursor > 0:
            cached_source_identities = getattr(self, "_last_active_replay_source_identities", None)
            cached_active_replay_messages = getattr(self, "_last_active_replay_messages", None)
            if (
                cached_source_identities is not None
                and cached_active_replay_messages is not None
                and len(cached_source_identities) >= cursor
                and len(cached_active_replay_messages) >= cursor
            ):
                current_prefix_identities = []
                for index, message in enumerate(messages[:cursor]):
                    if index % 32 == 0:
                        self._foreground_deadline_checkpoint(
                            deadline,
                            "ingest replay validation",
                        )
                    current_prefix_identities.append(
                        self._message_replay_identity(message)
                    )
                if current_prefix_identities == cached_source_identities[:cursor]:
                    replay_messages = (
                        self._copy_active_replay_messages_preserving_generated_ids(
                            cached_active_replay_messages[:cursor]
                        )
                        + replay_messages[cursor:]
                    )
            if any(
                active_store_ids_by_index[index] is None
                for index in range(cursor)
            ):
                prefix_store_ids = ReconcileMixin._get_store_id_map_for_messages(
                    self,
                    messages[:cursor],
                    deadline=deadline,
                )
                for index, message in enumerate(messages[:cursor]):
                    if active_store_ids_by_index[index] is None:
                        active_store_ids_by_index[index] = prefix_store_ids.get(
                            id(message)
                        )
        logger.debug(
            "Ingest: session=%s cursor=%d incoming=%d",
            self._session_id, cursor, n,
        )

        new_messages = replay_messages[cursor:] if cursor < n else []
        original_new_messages = messages[cursor:] if cursor < n else []

        if not new_messages:
            cached_replay = self._cached_active_replay_messages(messages)
            self._compression_boundary_ingest_pending = False
            self._compression_boundary_active_placeholder_digest_budget = {}
            self._compression_boundary_active_placeholder_digest_ordinals = {}
            self._compression_boundary_stored_placeholder_digest_counts = {}
            self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            if cached_replay is not None:
                return cached_replay
            return self._remember_active_replay_messages(
                messages,
                replay_messages,
                store_ids_by_index=active_store_ids_by_index,
                explicit_store_id_indices=explicit_store_id_indices,
                deadline=deadline,
            )

        active_replay_messages = replay_messages
        compression_boundary_ingest_pending = self._compression_boundary_ingest_pending
        empty_session_placeholder_budget: dict[str, int] = {}
        empty_session_placeholder_ordinals: dict[str, set[int]] = {}
        if not compression_boundary_ingest_pending and self._session_id:
            self._foreground_deadline_checkpoint(
                deadline,
                "ingest durable count",
            )
            try:
                if self._store.get_session_count(
                    self._session_id,
                    lock_timeout_seconds=(
                        self._foreground_deadline_remaining(deadline)
                    ),
                    deadline=deadline,
                    deadline_clock=self._foreground_now,
                ) == 0:
                    empty_session_placeholder_budget = self._load_generated_ignored_placeholder_hash_counts()
                    empty_session_placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals()
            except Exception:
                empty_session_placeholder_budget = {}
                empty_session_placeholder_ordinals = {}
            self._foreground_deadline_checkpoint(
                deadline,
                "ingest durable count",
            )
        messages_to_store_with_index: list[tuple[int, Dict[str, Any]]] = [
            (cursor + offset, replay_msg)
            for offset, replay_msg in enumerate(new_messages)
        ]
        if messages_to_store_with_index:
            kept: list[tuple[int, Dict[str, Any]]] = []
            boundary_placeholder_seen: dict[str, int] = {}
            boundary_seen_synthetic_summary_before = False
            empty_session_placeholder_seen: dict[str, int] = {}
            if empty_session_placeholder_ordinals and cursor > 0:
                for replay_msg in replay_messages[:cursor]:
                    replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
                    digest = self._active_replay_placeholder_digest(replay_text)
                    if digest:
                        empty_session_placeholder_seen[digest] = empty_session_placeholder_seen.get(digest, 0) + 1
            boundary_all_placeholder_replay_batch = (
                compression_boundary_ingest_pending
                and len(new_messages) > 1
                and all(
                    self._is_ignored_active_replay_placeholder(
                        msg,
                        text_content_for_pattern_matching(msg.get("content")) or "",
                    )
                    for msg in new_messages
                )
            )
            if compression_boundary_ingest_pending:
                boundary_budget = self._compression_boundary_active_placeholder_digest_budget
                stored_counts = self._compression_boundary_stored_placeholder_digest_counts
                if boundary_budget and stored_counts:
                    incoming_counts: dict[str, int] = {}
                    relevant_digests = set(boundary_budget) | set(stored_counts)
                    for msg in new_messages:
                        text = text_content_for_pattern_matching(msg.get("content")) or ""
                        digest = self._active_replay_placeholder_digest(text)
                        if digest in relevant_digests:
                            incoming_counts[digest] = incoming_counts.get(digest, 0) + 1
                    adjusted_budget: dict[str, int] = {}
                    for digest, count in boundary_budget.items():
                        parsed_count = max(0, int(count or 0))
                        incoming_count = max(0, int(incoming_counts.get(digest, 0) or 0))
                        stored_count = max(0, int(stored_counts.get(digest, 0) or 0))
                        remaining = min(parsed_count, max(0, incoming_count - stored_count))
                        if remaining > 0:
                            adjusted_budget[digest] = remaining
                    self._compression_boundary_active_placeholder_digest_budget = adjusted_budget
            empty_session_all_placeholder_replay_batch = (
                bool(empty_session_placeholder_ordinals)
                and len(new_messages) > 1
                and all(
                    self._is_ignored_active_replay_placeholder(
                        msg,
                        text_content_for_pattern_matching(msg.get("content")) or "",
                    )
                    for msg in new_messages
                )
            )
            for offset, (original_msg, replay_msg) in enumerate(zip(original_new_messages, new_messages)):
                if offset % 32 == 0:
                    self._foreground_deadline_checkpoint(
                        deadline,
                        "ingest batch preparation",
                    )
                absolute_idx = cursor + offset
                replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
                original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
                generated_context_scaffold = self._is_replayed_context_scaffold_message(replay_msg)
                volatile_placeholder = self._is_volatile_ignored_quarantine_placeholder(
                    replay_msg,
                    replay_text,
                )
                volatile_digest = self._active_replay_placeholder_digest(replay_text)
                generated_volatile_placeholder = volatile_placeholder and (
                    original_text != replay_text
                    or (
                        volatile_digest is not None
                        and volatile_digest in self._load_generated_ignored_placeholder_hashes()
                    )
                )
                active_replay_placeholder = self._is_ignored_active_replay_placeholder(replay_msg, replay_text)
                active_replay_placeholder_digest = self._active_replay_placeholder_digest(replay_text)
                if not active_replay_placeholder:
                    replay_text_stripped = replay_text.strip()
                    if (
                        self._is_context_summary_content(replay_text)
                        or replay_text_stripped.startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                        or replay_text_stripped.startswith(_PRESERVED_TODO_CONTEXT_PREFIX)
                    ):
                        boundary_seen_synthetic_summary_before = True
                # Generated projections are model-facing replay scaffold, not
                # new source history. Record their compression-boundary ordering
                # signal above before excluding them, otherwise carried ignored
                # placeholders lose their ordinal/provenance protection.
                if generated_context_scaffold:
                    logger.debug(
                        "LCM skipped generated context projection during ingest: session=%s index=%d",
                        self._session_id,
                        absolute_idx,
                    )
                    continue
                compression_carried_active_placeholder = False
                metadata_replayed_active_placeholder = False
                if (
                    empty_session_placeholder_budget
                    and empty_session_placeholder_ordinals
                    and active_replay_placeholder
                    and active_replay_placeholder_digest is not None
                ):
                    empty_session_placeholder_seen[active_replay_placeholder_digest] = (
                        empty_session_placeholder_seen.get(active_replay_placeholder_digest, 0) + 1
                    )
                    ordinal = empty_session_placeholder_seen[active_replay_placeholder_digest]
                    remaining = empty_session_placeholder_budget.get(active_replay_placeholder_digest, 0)
                    if (
                        remaining > 0
                        and ordinal in empty_session_placeholder_ordinals.get(
                            active_replay_placeholder_digest,
                            set(),
                        )
                        and (ordinal > 1 or empty_session_all_placeholder_replay_batch)
                    ):
                        metadata_replayed_active_placeholder = True
                        if remaining == 1:
                            empty_session_placeholder_budget.pop(active_replay_placeholder_digest, None)
                        else:
                            empty_session_placeholder_budget[active_replay_placeholder_digest] = remaining - 1
                if (
                    compression_boundary_ingest_pending
                    and active_replay_placeholder
                    and active_replay_placeholder_digest is not None
                ):
                    boundary_placeholder_seen[active_replay_placeholder_digest] = (
                        boundary_placeholder_seen.get(active_replay_placeholder_digest, 0) + 1
                    )
                    current_placeholder_ordinal = boundary_placeholder_seen[active_replay_placeholder_digest]
                    boundary_budget = self._compression_boundary_active_placeholder_digest_budget
                    boundary_ordinals = self._compression_boundary_active_placeholder_digest_ordinals
                    generated_message_ids = getattr(
                        self,
                        "_generated_ignored_active_replay_placeholder_message_ids",
                        set(),
                    )
                    has_generated_provenance = (
                        id(replay_msg) in generated_message_ids
                        or id(original_msg) in generated_message_ids
                    )
                    ordinal_matches_generated = (
                        current_placeholder_ordinal in boundary_ordinals.get(
                            active_replay_placeholder_digest,
                            set(),
                        )
                        and (
                            current_placeholder_ordinal > 1
                            or boundary_seen_synthetic_summary_before
                            or boundary_all_placeholder_replay_batch
                        )
                    )
                    if boundary_budget and (
                        has_generated_provenance
                        or (not has_generated_provenance and ordinal_matches_generated)
                    ):
                        remaining = boundary_budget.get(active_replay_placeholder_digest, 0)
                        if remaining > 0:
                            compression_carried_active_placeholder = True
                            if remaining == 1:
                                boundary_budget.pop(active_replay_placeholder_digest, None)
                            else:
                                boundary_budget[active_replay_placeholder_digest] = remaining - 1
                replayed_active_placeholder = active_replay_placeholder and (
                    self._is_cached_active_replay_message_at_index(absolute_idx, replay_msg)
                    or compression_carried_active_placeholder
                    or metadata_replayed_active_placeholder
                )
                if (
                    ignored_original_messages[absolute_idx]
                    or generated_volatile_placeholder
                    or replayed_active_placeholder
                ):
                    self._ignored_message_count += 1
                    if generated_volatile_placeholder and volatile_digest is not None:
                        self._remember_generated_ignored_placeholder_hash(
                            volatile_digest,
                            persist=deadline is None,
                        )
                    replay_preserves_ignore_decision = (
                        self._is_volatile_ignored_quarantine_placeholder(replay_msg, replay_text)
                        or self._is_ignored_active_replay_placeholder(replay_msg, replay_text)
                    )
                    if ignored_original_messages[absolute_idx] and not replay_preserves_ignore_decision:
                        if active_replay_messages is replay_messages:
                            active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                                replay_messages
                            )
                        active_message = dict(active_replay_messages[absolute_idx])
                        active_message["content"] = self._ignored_active_replay_placeholder(original_text)
                        active_replay_messages[absolute_idx] = active_message
                    excerpt = original_text[:80].replace("\n", " ")
                    if ignored_original_messages[absolute_idx]:
                        # A raw message matched ignore_message_patterns and is
                        # discarded here - never persisted anywhere. Count and
                        # log it (INFO) so an over-broad pattern silently eating
                        # substantive turns is at least visible to the operator.
                        self._ignore_pattern_dropped_count += 1
                        logger.info(
                            "LCM ignore_message_patterns dropped %s message "
                            "(not persisted; total dropped=%d): %r",
                            original_msg.get("role", "unknown"),
                            self._ignore_pattern_dropped_count,
                            excerpt,
                        )
                    else:
                        logger.debug(
                            "LCM ignore_message_patterns dropped %s message: %r",
                            original_msg.get("role", "unknown"),
                            excerpt,
                        )
                    continue
                store_msg = replay_msg
                if (
                    str(original_msg.get("role") or "") == "tool"
                    and _is_hermes_persisted_output_marker(
                        normalize_content_value(original_msg.get("content")) or ""
                    )
                ):
                    store_msg = original_msg
                if absolute_idx not in reused_cached_replay_indices:
                    kept.append((absolute_idx, store_msg))
            messages_to_store_with_index = kept

        if not messages_to_store_with_index:
            self._ingest_cursor = n
            self._compression_boundary_ingest_pending = False
            self._compression_boundary_active_placeholder_digest_budget = {}
            self._compression_boundary_active_placeholder_digest_ordinals = {}
            self._compression_boundary_stored_placeholder_digest_counts = {}
            self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
            return self._remember_active_replay_messages(
                messages,
                active_replay_messages,
                store_ids_by_index=active_store_ids_by_index,
                explicit_store_id_indices=explicit_store_id_indices,
                deadline=deadline,
            )

        protected_messages = self._call_foreground_projection(
            protect_messages_for_ingest,
            [msg for _idx, msg in messages_to_store_with_index],
            deadline=deadline,
            stage="ingest protection",
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
            ownership=ownership,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "ingest protection",
        )
        recovery_tool_call_ids = self._active_replay_recovery_tool_call_ids(
            active_replay_messages
        )
        for protected_index, ((absolute_idx, _replay_msg), protected_msg) in enumerate(zip(
            messages_to_store_with_index,
            protected_messages,
        )):
            if protected_index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "ingest active replay protection",
                )
            if self._protected_message_uses_raw_payload_active_stub(protected_msg):
                if active_replay_messages is replay_messages:
                    active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                        replay_messages
                    )
                active_message = dict(active_replay_messages[absolute_idx])
                active_message["content"] = protected_msg["content"]
                active_replay_messages[absolute_idx] = active_message
                continue

            active_message = active_replay_messages[absolute_idx]
            stubbed_message = self._call_foreground_projection(
                self._maybe_stub_active_tool_result,
                active_message,
                deadline=deadline,
                stage="ingest active replay stubbing",
                recovery_tool_call_ids=recovery_tool_call_ids,
                ownership=ownership,
            )
            if stubbed_message is not None:
                if active_replay_messages is replay_messages:
                    active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                        replay_messages
                    )
                active_replay_messages[absolute_idx] = stubbed_message

        estimates: list[int] = []
        for estimate_index, (protected_message, (absolute_idx, _replay_message)) in enumerate(
            zip(protected_messages, messages_to_store_with_index)
        ):
            if estimate_index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "ingest token counting",
                )
            # When previewing is enabled, the stored row may contain a small
            # lossless externalization marker. Keep backlog/compaction accounting
            # tied to the original source message so request-only projections
            # cannot make raw work disappear. Flag-off sessions retain the prior
            # marker-estimate behavior exactly.
            # The integer estimate contains no payload and does not weaken the
            # protected/externalized storage boundary.
            source_message = protected_message
            if self._active_replay_preview_policy().enabled:
                source_message = (
                    messages[absolute_idx]
                    if 0 <= absolute_idx < len(messages)
                    else protected_message
                )
            estimates.append(count_message_tokens(source_message))
        self._foreground_deadline_checkpoint(deadline, "ingest commit")
        metadata_json_updates = None
        metadata_json_list_additions = None
        if deadline is not None:
            durable_active_indices = {
                index
                for index, store_id in enumerate(active_store_ids_by_index)
                if store_id is not None
            }
            durable_active_indices.update(
                absolute_idx
                for absolute_idx, _message in messages_to_store_with_index
            )
            (
                metadata_json_updates,
                metadata_json_list_additions,
            ) = self._active_placeholder_metadata_transaction_updates(
                active_replay_messages,
                durable_indices=durable_active_indices,
                deadline=deadline,
            )
        # Persist replay ordering atomically with its raw rows. Otherwise a
        # restart after multiple edits loses the only proof that unchanged
        # islands were already ingested and appends the entire view again.
        # Intentionally filtered rows are accepted too; bind the proof to
        # the complete policy so changing filters forces reconciliation.
        metadata_json_updates = list(metadata_json_updates or [])
        metadata_json_updates.append((
            [self._active_ingest_manifest_key()],
            json.dumps(self._active_ingest_manifest(messages, deadline=deadline)),
            True,
        ))
        if ownership is not None:
            try:
                ownership.publish(
                    deadline=deadline,
                    now=self._foreground_now,
                )
            except TimeoutError as exc:
                raise ForegroundDeadlineExceeded(
                    "ingest",
                    "payload publication exceeded the shared deadline",
                ) from exc
            self._foreground_deadline_checkpoint(deadline, "ingest publication")
        appended_store_ids = self._append_ingest_batch_with_deadline(
            protected_messages,
            estimates,
            deadline=deadline,
            metadata_json_updates=metadata_json_updates,
            metadata_json_list_additions=metadata_json_list_additions,
        )
        # The raw append is the ownership boundary: placeholders now have a
        # durable consumer, so a later foreground checkpoint must not remove
        # their payload files during compensation.
        if ownership is not None:
            ownership.commit()
        for (absolute_idx, _message), store_id in zip(
            messages_to_store_with_index,
            appended_store_ids,
        ):
            active_store_ids_by_index[absolute_idx] = int(store_id)
        # Rollup staleness is driven by summary-node PUBLICATION
        # (_invalidate_rollups_for_published_node at every add_node site), not by
        # raw ingest: marking a period stale before its covering summary exists
        # would let a rebuild publish 'ready' from old sources and omit the leaf
        # (maintainer #388 P1).
        self._ingest_cursor = n
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
        self._foreground_deadline_checkpoint(deadline, "ingest commit")
        logger.debug("Ingested %d messages into LCM store", len(messages_to_store_with_index))
        self._clear_foreground_rebind_candidate_if_bound_session_confirmed()
        # Most ``protected_messages`` changes are storage-only: inline media and
        # data/base64 substrings stay provider-usable in active replay. The
        # exceptions are whole-message ``raw_payload`` externalization and the
        # separately opt-in textual tool-result interceptor above.
        return self._remember_active_replay_messages(
            messages,
            active_replay_messages,
            store_ids_by_index=active_store_ids_by_index,
            explicit_store_id_indices=explicit_store_id_indices,
            deadline=deadline,
        )

    def _active_placeholder_metadata_transaction_updates(
        self,
        active_replay_messages: List[Dict[str, Any]],
        *,
        durable_indices: set[int],
        deadline: float,
    ) -> tuple[
        list[tuple[list[str], str, bool]],
        list[tuple[list[str], list[str], int]],
    ]:
        """Build derived placeholder metadata for the raw append transaction."""
        generated_hashes = set(
            getattr(
                self,
                "_generated_ignored_active_replay_placeholder_hashes",
                set(),
            )
        )
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        counts: dict[str, int] = {}
        ordinals: dict[str, set[int]] = {}
        occurrence_by_digest: dict[str, int] = {}
        for index, message in enumerate(active_replay_messages):
            if index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "ingest placeholder metadata",
                )
            text = text_content_for_pattern_matching(message.get("content")) or ""
            digest = self._active_replay_placeholder_digest(text)
            if not digest or (
                digest not in generated_hashes
                and id(message) not in generated_message_ids
            ):
                continue
            generated_hashes.add(digest)
            occurrence_by_digest[digest] = occurrence_by_digest.get(digest, 0) + 1
            if index in durable_indices:
                continue
            counts[digest] = counts.get(digest, 0) + 1
            ordinals.setdefault(digest, set()).add(
                occurrence_by_digest[digest]
            )

        serialized_counts = json.dumps(counts, sort_keys=True)
        serialized_ordinals = json.dumps(
            {
                digest: sorted(values)
                for digest, values in ordinals.items()
                if values
            },
            sort_keys=True,
        )
        updates = [
            (
                self._ignored_placeholder_count_metadata_keys(),
                serialized_counts,
                True,
            ),
            (
                self._ignored_placeholder_ordinal_metadata_keys(),
                serialized_ordinals,
                True,
            ),
        ]
        list_additions = [
            (
                self._ignored_placeholder_metadata_keys(),
                sorted(generated_hashes),
                512,
            )
        ]
        return updates, list_additions

    def _append_ingest_batch_with_deadline(
        self,
        protected_messages: List[Dict[str, Any]],
        estimates: List[int],
        *,
        deadline: float | None,
        metadata_json_updates: list[
            tuple[list[str], str, bool]
        ] | None = None,
        metadata_json_list_additions: list[
            tuple[list[str], list[str], int]
        ] | None = None,
    ) -> List[int]:
        """Atomically append raw rows without waiting past a foreground wall."""
        if deadline is None:
            return self._store._append_protected_batch(
                self._session_id,
                protected_messages,
                estimates,
                source=self._session_platform,
                conversation_id=self._conversation_id,
                metadata_json_updates=metadata_json_updates,
                metadata_json_list_additions=metadata_json_list_additions,
                dedupe_existing=True,
            )

        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is None or remaining <= 0:
            raise ForegroundDeadlineExceeded("ingest", "before raw append")
        try:
            return self._store._append_protected_batch(
                self._session_id,
                protected_messages,
                estimates,
                source=self._session_platform,
                conversation_id=self._conversation_id,
                deadline=deadline,
                deadline_clock=self._foreground_now,
                lock_timeout_seconds=remaining,
                busy_timeout_ms=max(0, int(remaining * 1000)),
                metadata_json_updates=metadata_json_updates,
                metadata_json_list_additions=metadata_json_list_additions,
                dedupe_existing=True,
            )
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                "ingest",
                "message-store work exceeded the shared deadline",
            ) from exc
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            raise ForegroundDeadlineExceeded(
                "ingest",
                "SQLite wait consumed the shared deadline",
            ) from exc

    @staticmethod
    def _protected_message_uses_raw_payload_active_stub(message: Dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, str) and content.startswith(
            "[Externalized payload: kind=raw_payload;"
        )

    def _get_store_id_map_for_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> dict[int, int]:
        """Map active objects, including a suffix reused across a middle edit."""
        self._foreground_deadline_checkpoint(
            deadline,
            "source mapping",
            "before active replay reference scan",
        )
        explicit: dict[int, Optional[int]] = {}
        refs = (
            getattr(self, "_active_replay_store_refs_by_message_id", {})
            if (
                str(self._session_id or "")
                == getattr(self, "_active_replay_store_ref_session_id", "")
                and str(self._conversation_id or "")
                == getattr(
                    self,
                    "_active_replay_store_ref_conversation_id",
                    "",
                )
            )
            else {}
        )
        all_explicit = True
        for message_index, message in enumerate(messages):
            if message_index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "source mapping",
                    "active replay reference scan",
                )
            cached = refs.get(id(message))
            if cached is None or cached[0] is not message:
                all_explicit = False
                continue
            explicit[id(message)] = cached[1]

        possible_joined_user = any(
            m.get('role') == 'user' and isinstance(m.get('content'), str)
            and '\n\n' in m['content'] for m in messages)
        if all_explicit and not possible_joined_user:
            self._foreground_deadline_checkpoint(
                deadline,
                "source mapping",
                "after active replay reference scan",
            )
            return {
                message_id: int(store_id)
                for message_id, store_id in explicit.items()
                if store_id is not None
            }

        mapped = ReconcileMixin._get_store_id_map_for_messages(
            self,
            messages,
            deadline=deadline,
        )
        for message_id, store_id in explicit.items():
            if store_id is None:
                mapped.pop(message_id, None)
            else:
                mapped[message_id] = int(store_id)
                groups = getattr(mapped, 'source_groups', {})
                if message_id in groups and groups[message_id][-1] != int(store_id):
                    groups.pop(message_id)
        self._foreground_deadline_checkpoint(
            deadline,
            "source mapping",
            "after durable source mapping",
        )
        return mapped

    def _get_store_ids_for_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> List[int]:
        ids_by_message_id = self._get_store_id_map_for_messages(
            messages,
            deadline=deadline,
        )
        from .historical_projection import source_ids
        return [source for msg in messages for source in source_ids(ids_by_message_id, msg)]

    # -- Internal: summarization -------------------------------------------

    def _schedule_pre_compaction_extraction_persistence(
        self,
        prepared: PreparedExtraction,
        *,
        output_path: str,
        session_id: str,
        cancel_event: threading.Event,
        deadline: float,
    ) -> None:
        """Persist extraction off-path with one bounded, deadline-aware worker."""
        if not prepared.content or self._foreground_now() >= float(deadline):
            return
        if not self._extraction_persistence_slot.acquire(blocking=False):
            logger.debug(
                "Pre-compaction extraction persistence skipped while a prior write is active"
            )
            return
        self._extraction_persistence_idle.clear()

        def persist() -> None:
            try:
                persist_pre_compaction_extraction(
                    prepared,
                    output_path=output_path,
                    session_id=session_id,
                    cancel_event=cancel_event,
                    deadline=deadline,
                    now=self._foreground_now,
                )
            finally:
                self._extraction_persistence_slot.release()
                self._extraction_persistence_idle.set()

        worker = threading.Thread(
            target=persist,
            name="lcm-extraction-persistence",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException as exc:
            self._extraction_persistence_slot.release()
            self._extraction_persistence_idle.set()
            logger.warning(
                "Pre-compaction extraction persistence worker could not start "
                "(error_type=%s)",
                type(exc).__name__,
            )

    def _run_pre_compaction_extraction(
        self,
        messages: List[Dict[str, Any]],
        *,
        timeout_seconds: Optional[float] = None,
        deadline: float | None = None,
    ) -> None:
        """Best-effort extraction of decisions before compaction."""
        cancel_event = threading.Event()
        try:
            started_at = self._foreground_now()
            configured_timeout = (
                float(timeout_seconds)
                if timeout_seconds is not None
                else self._config.summary_timeout_ms / 1000
            )
            local_deadline = started_at + max(0.0, configured_timeout)
            effective_deadline = (
                local_deadline
                if deadline is None
                else min(float(deadline), local_deadline)
            )
            if self._foreground_now() >= effective_deadline:
                raise TimeoutError(
                    "pre-compaction extraction budget expired before serialization"
                )
            serialized = self._call_with_foreground_deadline(
                self._serialize_messages,
                messages,
                deadline=effective_deadline,
            )
            remaining = max(
                0.0,
                effective_deadline - self._foreground_now(),
            )
            if remaining <= 0:
                raise TimeoutError(
                    "pre-compaction extraction budget expired during serialization"
                )
            output_path = self._config.extraction_output_path
            if not output_path:
                base = self._hermes_home or os.path.expanduser("~/.hermes")
                output_path = os.path.join(base, "lcm-extractions")
            extraction_model = self._config.extraction_model or self._config.summary_model
            extraction_kwargs: dict[str, Any] = {
                "serialized_messages": serialized,
                "output_path": output_path,
                "session_id": self._session_id or "",
                "model": extraction_model,
            }
            if self._summary_callable_accepts_keyword(
                extract_before_compaction,
                "timeout",
            ):
                extraction_kwargs["timeout"] = remaining
            if self._summary_callable_accepts_keyword(
                extract_before_compaction,
                "cancel_event",
            ):
                extraction_kwargs["cancel_event"] = cancel_event
            defers_persistence = self._summary_callable_accepts_keyword(
                extract_before_compaction,
                "defer_persistence",
            )
            if defers_persistence:
                extraction_kwargs["defer_persistence"] = True

            result = self._call_with_hard_foreground_deadline(
                lambda: extract_before_compaction(**extraction_kwargs),
                deadline=effective_deadline,
                timeout_error=TimeoutError(
                    "pre-compaction extraction exceeded its foreground budget"
                ),
                cancel_event=cancel_event,
            )
            if isinstance(result, PreparedExtraction):
                if self._foreground_now() >= effective_deadline:
                    cancel_event.set()
                    return
                self._schedule_pre_compaction_extraction_persistence(
                    result,
                    output_path=output_path,
                    session_id=self._session_id or "",
                    cancel_event=cancel_event,
                    deadline=effective_deadline,
                )
        except Exception as e:
            logger.warning(
                "Pre-compaction extraction failed or timed out (non-blocking): %s",
                type(e).__name__,
            )

    def _run_assertion_extraction_batch(
        self,
        db_path: str,
        snapshots: tuple[SourceSnapshot, ...],
        model: str,
        timeout_seconds: float,
        schema_ready_token: AssertionSchemaReadyToken,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Derive one bounded batch on a daemon worker outside raw writes."""
        clock = deadline_clock or time.monotonic

        def remaining_seconds() -> float | None:
            if cancel_event is not None and cancel_event.is_set():
                return 0.0
            if deadline is None:
                return None
            return max(0.0, float(deadline) - float(clock()))

        def require_active() -> float | None:
            remaining = remaining_seconds()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("assertion origin deadline exceeded")
            return remaining

        started = time.perf_counter()
        completed = 0
        failed = 0
        last_error = ""
        extractor = None
        writer = None
        try:
            initial_remaining = require_active()
            writer = AssertionStore(
                db_path,
                deadline=deadline,
                deadline_clock=clock,
                busy_timeout_ms=(
                    None
                    if initial_remaining is None
                    else max(0, int(initial_remaining * 1000))
                ),
                initialize_schema=False,
                schema_ready_token=schema_ready_token,
            )
            extractor = ModelAssertionExtractor(
                writer,
                model=model,
                timeout_seconds=(
                    timeout_seconds
                    if initial_remaining is None
                    else min(timeout_seconds, initial_remaining)
                ),
            )
            for snapshot in snapshots:
                try:
                    remaining = require_active()
                    read_kwargs = (
                        {
                            "lock_timeout_seconds": remaining,
                            "deadline": deadline,
                            "deadline_clock": clock,
                            "busy_timeout_ms": max(0, int(remaining * 1000)),
                        }
                        if remaining is not None
                        else {}
                    )
                    if writer.has_current_receipt(snapshot, **read_kwargs):
                        completed += 1
                        continue
                    remaining = require_active()
                    if remaining is not None:
                        # The constructor keeps legacy standalone calls at a
                        # 100ms floor. Origin-bounded workers must instead pass
                        # the exact smaller remainder through to the provider.
                        extractor.timeout_seconds = max(
                            0.001,
                            min(float(timeout_seconds), remaining),
                        )
                    extraction = extractor(
                        snapshot,
                        deadline=deadline,
                        deadline_clock=clock,
                        cancel_event=cancel_event,
                    )
                    remaining = require_active()
                    publish_kwargs = (
                        {
                            "lock_timeout_seconds": remaining,
                            "deadline": deadline,
                            "deadline_clock": clock,
                            "busy_timeout_ms": max(0, int(remaining * 1000)),
                            "cancel_event": cancel_event,
                        }
                        if remaining is not None
                        else {}
                    )
                    writer.publish_source(
                        snapshot,
                        extraction.assertions,
                        relations=extraction.relations,
                        **publish_kwargs,
                    )
                    completed += 1
                except Exception as exc:
                    failed += 1
                    last_error = f"{type(exc).__name__}: {exc}"[:300]
                    logger.warning(
                        "Structured assertion extraction failed for store_id %s: %s",
                        snapshot.store_id,
                        last_error,
                    )
                    if isinstance(exc, TimeoutError) or (
                        cancel_event is not None and cancel_event.is_set()
                    ):
                        failed += max(0, len(snapshots) - completed - failed)
                        break
        except Exception as exc:
            failed += max(1, len(snapshots) - completed)
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("Structured assertion batch failed: %s", last_error)
        finally:
            if writer is not None:
                try:
                    # Every durable assertion publication commits explicitly.
                    # A deadline-owned worker must never turn an interrupted
                    # transaction into a late commit during cleanup.
                    writer.close(commit=False)
                except Exception:
                    logger.warning(
                        "Structured assertion writer close failed",
                        exc_info=True,
                    )
            duration_ms = (time.perf_counter() - started) * 1000
            with self._assertion_extraction_metrics_lock:
                self._assertion_extraction_sources_completed += completed
                self._assertion_extraction_sources_failed += failed
                self._assertion_extraction_last_duration_ms = duration_ms
                self._assertion_extraction_last_error = last_error
                self._assertion_extraction_last_model = model
                if extractor is not None:
                    self._assertion_extraction_provider_calls += extractor.call_count
                    self._assertion_extraction_input_tokens += extractor.total_input_tokens
                    self._assertion_extraction_output_tokens += extractor.total_output_tokens
            self._assertion_extraction_idle.set()
            _ASSERTION_EXTRACTION_PROCESS_SLOT.release()

    def _schedule_pre_compaction_assertions(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> bool:
        """Queue exact persisted rows without blocking the compaction path."""
        if (
            not bool(getattr(self._config, "assertion_extraction_enabled", False))
            or self._assertions is None
            or not messages
        ):
            return False
        max_sources = min(
            8,
            max(
                1,
                int(
                    getattr(
                        self._config,
                        "assertion_extraction_max_sources_per_pass",
                        4,
                    )
                ),
            ),
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "assertion scheduling",
            "before source mapping",
        )
        store_ids = sorted(
            dict.fromkeys(
                self._call_with_foreground_deadline(
                    self._get_store_ids_for_messages,
                    messages,
                    deadline=deadline,
                )
            )
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "assertion scheduling",
            "after source mapping",
        )
        snapshots: list[SourceSnapshot] = []
        for store_id in store_ids:
            self._foreground_deadline_checkpoint(
                deadline,
                "assertion scheduling",
                "before assertion source read",
            )
            remaining = self._foreground_deadline_remaining(deadline)
            read_kwargs = (
                {
                    "lock_timeout_seconds": remaining,
                    "deadline": deadline,
                    "deadline_clock": self._foreground_now,
                    "busy_timeout_ms": max(0, int(remaining * 1000)),
                }
                if remaining is not None
                else {}
            )
            try:
                snapshot = self._assertions.snapshot_source(
                    store_id,
                    **read_kwargs,
                )
                if snapshot.role not in {"user", "assistant"}:
                    continue
                if not self._assertions.has_current_receipt(
                    snapshot,
                    **read_kwargs,
                ):
                    snapshots.append(snapshot)
            except (KeyError, sqlite3.Error):
                continue
            except TimeoutError as exc:
                raise ForegroundDeadlineExceeded(
                    "assertion scheduling",
                    "assertion store lock wait exceeded the shared deadline",
                ) from exc
            if len(snapshots) >= max_sources:
                break
        self._foreground_deadline_checkpoint(
            deadline,
            "assertion scheduling",
            "after assertion source reads",
        )
        if not snapshots:
            return False
        self._foreground_deadline_checkpoint(
            deadline,
            "assertion scheduling",
            "before assertion worker admission",
        )
        if not _ASSERTION_EXTRACTION_PROCESS_SLOT.acquire(blocking=False):
            with self._assertion_extraction_metrics_lock:
                self._assertion_extraction_batches_skipped_busy += 1
            return False

        model = self._assertion_extraction_model()
        timeout_seconds = self._assertion_extraction_timeout()
        schema_ready_token = self._assertion_schema_ready_token
        if schema_ready_token is None:
            _ASSERTION_EXTRACTION_PROCESS_SLOT.release()
            logger.warning(
                "Structured assertion worker skipped without a startup schema proof"
            )
            return False
        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is not None:
            if remaining <= 0:
                _ASSERTION_EXTRACTION_PROCESS_SLOT.release()
                raise ForegroundDeadlineExceeded(
                    "assertion scheduling",
                    "deadline expired after worker admission",
                )
            timeout_seconds = min(timeout_seconds, remaining)
        cancel_event = threading.Event()
        try:
            self._foreground_deadline_checkpoint(
                deadline,
                "assertion scheduling",
                "before assertion worker start",
            )
        except BaseException:
            cancel_event.set()
            _ASSERTION_EXTRACTION_PROCESS_SLOT.release()
            raise
        with self._assertion_extraction_metrics_lock:
            self._assertion_extraction_batches_scheduled += 1
            self._assertion_extraction_sources_scheduled += len(snapshots)
        self._assertion_extraction_idle.clear()
        worker = threading.Thread(
            target=self._run_assertion_extraction_batch,
            args=(
                str(self._store.db_path),
                tuple(snapshots),
                model,
                timeout_seconds,
                schema_ready_token,
                deadline,
                self._foreground_now,
                cancel_event,
            ),
            name="lcm-assertion-extraction",
            daemon=True,
        )
        try:
            worker.start()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            with self._assertion_extraction_metrics_lock:
                self._assertion_extraction_sources_failed += len(snapshots)
                self._assertion_extraction_last_error = last_error
                self._assertion_extraction_last_model = model
            self._assertion_extraction_idle.set()
            _ASSERTION_EXTRACTION_PROCESS_SLOT.release()
            logger.warning(
                "Structured assertion worker could not start: %s",
                last_error,
            )
            return False
        return True

    def _maybe_gc_compacted_tool_results(
        self,
        compacted_chunk: List[Dict[str, Any]],
        source_store_ids: List[int],
        *,
        deadline: float | None = None,
    ) -> None:
        if not getattr(self._config, "large_output_transcript_gc_enabled", False):
            return
        if not compacted_chunk or not source_store_ids:
            return
        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is not None and remaining < 0.1:
            # Optional cleanup should not start when there is no useful budget
            # for both discovery and a precommit-checked rewrite.
            return

        def checkpoint(detail: str) -> None:
            self._foreground_deadline_checkpoint(
                deadline,
                "transcript GC",
                detail,
            )

        def _archive_in_rewrite_txn(conn: "sqlite3.Connection", sid: int) -> None:
            # Runs inside gc_externalized_tool_result's write transaction, right
            # after the content rewrite and before its commit: archive this row's
            # now-stale chunks ATOMICALLY with the rewrite so a recall can never
            # slice the new (short) content at the old chunk offsets (F2).
            self._archive_chunks_for_messages([sid], connection=conn)
            checkpoint("before transcript GC commit")

        def plan_gc(
            *,
            deadline: float | None = None,
            cancel_event: threading.Event | None = None,
            now: Callable[[], float] | None = None,
        ) -> list[tuple[int, str]]:
            clock = now or self._foreground_now

            def plan_checkpoint(detail: str) -> None:
                if (
                    (cancel_event is not None and cancel_event.is_set())
                    or (
                        deadline is not None
                        and float(clock()) >= float(deadline)
                    )
                ):
                    raise TimeoutError(
                        f"transcript GC planning deadline exceeded {detail}"
                    )

            plan_checkpoint("before source lookup")
            if deadline is None:
                stored_by_id = self._store.get_batch(source_store_ids)
            else:
                remaining_for_read = max(
                    0.0,
                    float(deadline) - float(clock()),
                )
                stored_by_id = self._store.get_batch(
                    source_store_ids,
                    deadline=deadline,
                    deadline_clock=clock,
                    lock_timeout_seconds=remaining_for_read,
                    busy_timeout_ms=max(0, int(remaining_for_read * 1000)),
                )
            plan_checkpoint("after source lookup")
            planned: list[tuple[int, str]] = []

            for store_id in source_store_ids:
                plan_checkpoint("before source lookup")
                stored = stored_by_id.get(store_id)
                if not stored or stored.get("session_id") != self._session_id:
                    continue

                if stored.get("role") != "tool":
                    continue
                content = stored.get("content", "") or ""
                tool_call_id = stored.get("tool_call_id", "") or ""
                if not content:
                    continue

                # Only take the fast ref-branch when the ENTIRE row is the
                # externalized placeholder. A ref merely embedded in surrounding
                # text (e.g. a recall-tool result that quotes a placeholder) must
                # fall through to the content-equality lookup below, which tombstones
                # only when the full row content matches the stored payload -
                # otherwise the surrounding, never-externalized text is lost.
                ref = (
                    extract_externalized_ref(content)
                    if is_externalized_placeholder(content)
                    else None
                )
                if ref:
                    externalized = load_externalized_payload(
                        ref,
                        config=self._config,
                        hermes_home=self._hermes_home,
                    )
                    plan_checkpoint("after externalized payload lookup")
                    if (
                        externalized is not None
                        and externalized.get("kind", "tool_result")
                        == "tool_result"
                    ):
                        placeholder = build_transcript_gc_placeholder(externalized)
                        planned.append((store_id, placeholder))
                        continue

                lookup_candidates = []
                sanitized_content = sanitize_pre_compaction_content(content)
                if sanitized_content and sanitized_content != content:
                    lookup_candidates.append(sanitized_content)
                lookup_candidates.append(content)

                externalized = None
                for candidate in lookup_candidates:
                    externalized = find_externalized_payload_for_message(
                        candidate,
                        tool_call_id=tool_call_id,
                        session_id=self._session_id,
                        config=self._config,
                        hermes_home=self._hermes_home,
                    )
                    plan_checkpoint("after externalized payload search")
                    if externalized is not None:
                        break
                if externalized is not None:
                    placeholder = build_transcript_gc_placeholder(externalized)
                    planned.append((store_id, placeholder))
            plan_checkpoint("after source planning")
            return planned

        if deadline is None:
            planned_rewrites = plan_gc()
        else:
            try:
                planned_rewrites = self._call_foreground_projection(
                    plan_gc,
                    deadline=deadline,
                    stage="transcript GC planning",
                )
            except (ForegroundDeadlineExceeded, TimeoutError):
                return
            except sqlite3.OperationalError as exc:
                if _is_sqlite_locked_error(exc):
                    return
                raise

        for store_id, placeholder in planned_rewrites:
            try:
                checkpoint("before source rewrite")
                remaining_for_write = self._foreground_deadline_remaining(deadline)
                write_kwargs: dict[str, Any] = {}
                if deadline is not None:
                    write_kwargs = {
                        "deadline": deadline,
                        "deadline_clock": self._foreground_now,
                        "lock_timeout_seconds": remaining_for_write,
                        "busy_timeout_ms": max(
                            0,
                            int(float(remaining_for_write or 0.0) * 1000),
                        ),
                    }
                self._store.gc_externalized_tool_result(
                    store_id,
                    placeholder,
                    before_commit=_archive_in_rewrite_txn,
                    **write_kwargs,
                )
            except (ForegroundDeadlineExceeded, TimeoutError):
                # Each rewrite is its own transaction. A deadline raised by the
                # store or archive callback rolls the current one back; earlier
                # completed cleanup remains valid and raw content never vanishes.
                return
            except sqlite3.OperationalError as exc:
                if _is_sqlite_locked_error(exc):
                    return
                raise

    def reserve_model_switch(self, **kwargs):
        from .model_switch_binding import reserve_model_switch
        return reserve_model_switch(self, **kwargs)

    def finish_model_switch(self, receipt):
        from .model_switch_binding import finish_model_switch
        return finish_model_switch(self, receipt)

    def model_switch_summary_policy(self, *, custom_providers, deadline, cancel_event,
                                    registry_version=None):
        from .model_switch_preparation import model_switch_summary_policy
        return model_switch_summary_policy(
            self, custom_providers=custom_providers, deadline=deadline,
            cancel_event=cancel_event, registry_version=registry_version)

    def prepare_model_switch_summary(
        self, messages, token_budget, deadline, cancel_event, *, admit_request,
        custom_providers, registry_version=None
    ):
        """Prepare a detached summary without changing LCM or host history."""
        from .model_switch_preparation import prepare_model_switch_summary

        return prepare_model_switch_summary(
            self, messages, token_budget, deadline, cancel_event,
            admit_request=admit_request, custom_providers=custom_providers,
            registry_version=registry_version,
        )

    def _serialize_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> str:
        """Serialize messages into labeled text for the summarizer."""
        return self._call_foreground_projection(
            serialize_summary_messages,
            messages,
            deadline=deadline,
            stage="summary serialization",
            config=self._config,
            session_id=self._session_id,
            hermes_home=self._hermes_home,
        )

    # -- Internal: tool-pair sanitization ------------------------------------

    def _context_projection_checkpoint(
        self,
        *,
        deadline: float | None,
        cancel_event: threading.Event | None,
        now: Callable[[], float] | None,
        operation: str,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ForegroundDeadlineExceeded(
                "context reconstruction",
                f"{operation} was cancelled at the shared deadline",
            )
        if deadline is None:
            return
        clock = now or self._foreground_now
        if clock() >= float(deadline):
            raise ForegroundDeadlineExceeded(
                "context reconstruction",
                f"{operation} exceeded the shared deadline",
            )

    def _sanitize_active_context_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> List[Dict[str, Any]]:
        """Drop unsafe assistant-only noise, then repair tool sequencing.

        This is intentionally active-context-only: callers pass the selected
        provider replay context, and this helper never mutates stored rows,
        source mappings, or DAG nodes.
        """
        cleaned: list[Dict[str, Any]] = []
        dropped_assistant_messages = 0
        stripped_assistant_messages = 0
        for message_index, msg in enumerate(messages):
            if message_index % 32 == 0:
                self._context_projection_checkpoint(
                    deadline=deadline,
                    cancel_event=cancel_event,
                    now=now,
                    operation="active context cleanup",
                )
            msg = self._sanitize_active_preserved_objective_message(
                msg,
                deadline=deadline,
                cancel_event=cancel_event,
                now=now,
                ownership=ownership,
            )
            if msg.get("role") == "assistant":
                cleaned_msg = _clean_active_assistant_message(msg)
                if cleaned_msg is None:
                    dropped_assistant_messages += 1
                    continue
                if cleaned_msg is not msg:
                    stripped_assistant_messages += 1
                cleaned.append(cleaned_msg)
                continue
            cleaned.append(msg)

        if dropped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: dropped %d assistant message(s) with no visible content",
                dropped_assistant_messages,
            )
        if stripped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: stripped internal content from %d assistant message(s)",
                stripped_assistant_messages,
            )

        sanitized = self._sanitize_tool_pairs(
            cleaned,
            insert_missing_tool_stubs=insert_missing_tool_stubs,
        )
        self._context_projection_checkpoint(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            operation="active context cleanup",
        )
        return sanitized

    @staticmethod
    def _active_tool_stub_content(original_content: Any, placeholder: str) -> Any:
        """Preserve compatible structured text-block shape when inserting a ref."""
        if not isinstance(original_content, list):
            return placeholder
        for block in original_content:
            if isinstance(block, str):
                return [placeholder]
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").lower()
            if block_type not in {"text", "input_text", "output_text"}:
                continue
            for value_key in ("text", "content"):
                text_value = block.get(value_key)
                if isinstance(text_value, str):
                    return [{"type": block_type, value_key: placeholder}]
                if not isinstance(text_value, dict):
                    continue
                for nested_key in ("value", "content"):
                    if isinstance(text_value.get(nested_key), str):
                        return [{
                            "type": block_type,
                            value_key: {nested_key: placeholder},
                        }]
        # _is_textual_tool_result_content() only admits supported shapes, so
        # this fallback is defensive rather than a normal provider path.
        return [{"type": "text", "text": placeholder}]

    @staticmethod
    def _structured_text_block_value(block: Dict[str, Any]) -> str | None:
        for value_key in ("text", "content"):
            text_value = block.get(value_key)
            if isinstance(text_value, str):
                return text_value
            if not isinstance(text_value, dict):
                continue
            for nested_key in ("value", "content"):
                nested = text_value.get(nested_key)
                if isinstance(nested, str):
                    return nested
        return None

    @staticmethod
    def _is_textual_tool_result_content(content: Any) -> bool:
        """Return whether active stubbing can preserve provider semantics."""
        if _contains_media_payload(content):
            return False
        if isinstance(content, str):
            return True
        if not isinstance(content, list) or not content:
            return False
        for block in content:
            if isinstance(block, str):
                continue
            if not isinstance(block, dict):
                return False
            if str(block.get("type") or "").lower() not in {
                "text",
                "input_text",
                "output_text",
            }:
                return False
            if LCMEngine._structured_text_block_value(block) is None:
                return False
        return True

    def _active_replay_recovery_tool_call_ids(
        self,
        messages: List[Dict[str, Any]],
    ) -> set[str]:
        recovery_tool_call_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = _tool_call_id(tool_call)
                function = tool_call.get("function") or {}
                tool_name = str(function.get("name") or "") if isinstance(function, dict) else ""
                if call_id and tool_name in {"lcm_describe", "lcm_expand"}:
                    recovery_tool_call_ids.add(call_id)
        return recovery_tool_call_ids

    def _maybe_stub_active_tool_result(
        self,
        message: Dict[str, Any],
        *,
        recovery_tool_call_ids: set[str],
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> Dict[str, Any] | None:
        clock = now or time.monotonic
        if (
            (cancel_event is not None and cancel_event.is_set())
            or (deadline is not None and float(clock()) >= float(deadline))
        ):
            raise TimeoutError("active replay projection deadline exceeded")
        if not getattr(self._config, "large_output_active_replay_stubbing_enabled", False):
            return None
        if not getattr(self._config, "large_output_externalization_enabled", False):
            return None
        if not isinstance(message, dict) or message.get("role") != "tool":
            return None
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        if not tool_call_id or tool_call_id in recovery_tool_call_ids:
            return None
        content = message.get("content")
        if not self._is_textual_tool_result_content(content):
            return None
        normalized_content = normalize_content_value(content) or ""
        if not normalized_content or is_externalized_placeholder(normalized_content):
            return None
        threshold = max(
            1,
            int(
                getattr(
                    self._config,
                    "large_output_active_replay_stub_threshold_tokens",
                    25_000,
                )
                or 0
            ),
        )
        if count_tokens(normalized_content) <= threshold:
            return None
        externalized = maybe_externalize_tool_output(
            normalized_content,
            tool_call_id=tool_call_id,
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
            force=True,
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            ownership=ownership,
        )
        if (
            (cancel_event is not None and cancel_event.is_set())
            or (deadline is not None and float(clock()) >= float(deadline))
        ):
            raise TimeoutError("active replay projection deadline exceeded")
        if externalized is None:
            return None
        replacement = dict(message)
        replacement["content"] = self._active_tool_stub_content(
            content,
            externalized["placeholder"],
        )
        return replacement

    def _stub_large_tool_results_for_active_replay(
        self,
        messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> List[Dict[str, Any]]:
        """Replace eligible old tool payloads with durable refs for assembly.

        This is provider-replay-only. It never mutates the input messages, raw
        SQLite rows, or DAG lineage. The newest ``fresh_tail_count`` messages
        stay inline, matching Lossless Claw's protected-tail contract.
        """
        if not getattr(self._config, "large_output_active_replay_stubbing_enabled", False):
            return messages
        if not getattr(self._config, "large_output_externalization_enabled", False):
            return messages
        protected_tail_count = max(0, int(getattr(self._config, "fresh_tail_count", 0) or 0))
        eligible_end = max(0, len(messages) - protected_tail_count)
        if eligible_end <= 0:
            return messages

        recovery_tool_call_ids = self._active_replay_recovery_tool_call_ids(messages)

        result = list(messages)
        stubbed_count = 0
        tokens_saved = 0
        for idx, message in enumerate(messages[:eligible_end]):
            clock = now or time.monotonic
            if (
                (cancel_event is not None and cancel_event.is_set())
                or (deadline is not None and float(clock()) >= float(deadline))
            ):
                raise TimeoutError("active replay projection deadline exceeded")
            replacement = self._maybe_stub_active_tool_result(
                message,
                recovery_tool_call_ids=recovery_tool_call_ids,
                deadline=deadline,
                cancel_event=cancel_event,
                now=now,
                ownership=ownership,
            )
            if replacement is None:
                continue
            result[idx] = replacement
            stubbed_count += 1
            tokens_saved += max(
                0,
                count_message_tokens(message) - count_message_tokens(replacement),
            )

        if stubbed_count:
            logger.info(
                "LCM active replay stubbing: replaced %d evictable tool result(s), saved about %d tokens",
                stubbed_count,
                tokens_saved,
            )
        return result

    def _sanitize_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return provider-safe active-context tool-call/result sequencing.

        Raw store and DAG history remain lossless. This guardrail only sanitizes
        the active context emitted back to providers, where assistant tool calls
        must be followed immediately by their contiguous tool results. Late,
        duplicate, out-of-order, and orphan tool results are dropped; missing
        direct results get synthetic stubs.
        """
        sanitized: List[Dict[str, Any]] = []
        dropped_tool_results = 0
        inserted_stub_results = 0

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "tool":
                dropped_tool_results += 1
                i += 1
                continue

            sanitized.append(msg)

            if msg.get("role") == "assistant":
                expected_ids = [
                    call_id
                    for call_id in (_tool_call_id(tool_call) for tool_call in (msg.get("tool_calls") or []))
                    if call_id
                ]

                for expected_id in expected_ids:
                    matched_direct_result = False
                    while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                        next_msg = messages[i + 1]
                        next_id = str(next_msg.get("tool_call_id") or "").strip()
                        if next_id == expected_id:
                            sanitized.append(next_msg)
                            i += 1
                            matched_direct_result = True
                            break
                        dropped_tool_results += 1
                        i += 1

                    if not matched_direct_result and insert_missing_tool_stubs:
                        sanitized.append({
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": expected_id,
                        })
                        inserted_stub_results += 1

                while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    dropped_tool_results += 1
                    i += 1

            i += 1

        if dropped_tool_results:
            logger.info(
                "LCM tool-pair guardrail: dropped %d late/orphan/duplicate tool result(s)",
                dropped_tool_results,
            )
        if inserted_stub_results:
            logger.info(
                "LCM tool-pair guardrail: inserted %d missing tool-result stub(s)",
                inserted_stub_results,
            )

        return sanitized

    # -- Internal: condensation --------------------------------------------

    def _should_allow_follow_on_condensation(
        self,
        *,
        uncondensed_count: int,
        leaf_compacted_this_turn: bool,
        force_overflow: bool,
        critical_budget_pressure: bool = False,
    ) -> tuple[bool, str]:
        if not leaf_compacted_this_turn:
            return True, ""
        if not self._config.cache_friendly_condensation_enabled:
            return True, ""
        if force_overflow:
            return True, ""
        if critical_budget_pressure:
            return True, ""

        fanin = max(1, self._config.condensation_fanin)
        debt_threshold = fanin * max(1, self._config.cache_friendly_min_debt_groups)
        if uncondensed_count >= debt_threshold:
            return True, ""
        if uncondensed_count == fanin:
            return False, "cache_friendly_single_group"
        return False, "cache_friendly_low_debt"

    def _maybe_condense(
        self,
        focus_topic: Optional[str] = None,
        *,
        leaf_compacted_this_turn: bool = False,
        force_overflow: bool = False,
        critical_budget_pressure: bool = False,
        deadline: Optional[float] = None,
    ) -> None:
        """Check if any depth level has enough nodes for condensation."""
        self._last_condensation_suppressed_reason = ""

        max_depth = self._config.incremental_max_depth
        if max_depth == 0:
            return  # condensation disabled

        # When max_depth is -1 (unlimited), derive the upper bound from
        # the deepest existing node + 1, so condensation can always
        # create the next depth level.
        if max_depth < 0:
            all_nodes = self._call_bounded_foreground_dag_read(
                self._dag.get_session_nodes,
                self._session_id,
                deadline=deadline,
                stage="condensation",
            )
            upper = (max(n.depth for n in all_nodes) + 1) if all_nodes else 1
        else:
            upper = max_depth

        condensed_any = False
        suppression_reason = ""
        fanin = max(1, self._config.condensation_fanin)

        for depth in range(upper):
            if deadline is not None and time.monotonic() >= deadline:
                break
            uncondensed = self._call_bounded_foreground_dag_read(
                self._dag.get_uncondensed_at_depth,
                self._session_id,
                depth,
                deadline=deadline,
                stage="condensation",
            )
            if len(uncondensed) < fanin:
                continue

            allow_condense, reason = self._should_allow_follow_on_condensation(
                uncondensed_count=len(uncondensed),
                leaf_compacted_this_turn=leaf_compacted_this_turn,
                force_overflow=force_overflow,
                critical_budget_pressure=critical_budget_pressure,
            )
            if not allow_condense:
                suppression_reason = reason or suppression_reason
                continue

            # Take the first fanin nodes and condense
            to_condense = uncondensed[:fanin]
            source_tokens, summary_tokens, level = self._condense_summary_nodes(
                to_condense,
                focus_topic=focus_topic,
                deadline=deadline,
            )
            condensed_any = True

            logger.info(
                "LCM condensation: d%d × %d → d%d (L%d, %d→%d tokens)",
                depth, len(to_condense), depth + 1, level,
                source_tokens, summary_tokens,
            )

            if leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
                break

        if not condensed_any and leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
            self._last_condensation_suppressed_reason = suppression_reason

    def _condense_summary_nodes(
        self,
        nodes: List[SummaryNode],
        *,
        focus_topic: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> tuple[int, int, int]:
        """Persist one same-depth condensation and return source/output tokens and level."""
        if not nodes:
            raise ValueError("condensation requires at least one summary node")
        depth = nodes[0].depth
        if any(node.depth != depth for node in nodes):
            raise ValueError("condensation requires same-depth summary nodes")
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "before input preparation",
        )
        combined_text = "\n\n---\n\n".join(node.summary for node in nodes)
        source_tokens = sum(node.token_count for node in nodes)
        token_budget = max(1000, int(source_tokens * 0.40))
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "after input preparation",
        )
        timeout_seconds = self._config.summary_timeout_ms / 1000
        if deadline is not None:
            remaining_seconds = float(deadline) - self._foreground_now()
            if remaining_seconds <= 0:
                raise TimeoutError("threshold full sweep time budget exhausted")
            timeout_seconds = min(timeout_seconds, remaining_seconds)
        cancel_event = threading.Event()
        summarize_kwargs: dict[str, Any] = {
            "text": combined_text,
            "source_tokens": source_tokens,
            "token_budget": token_budget,
            "depth": depth + 1,
            "model": self._config.summary_model,
            "fallback_models": self._config.summary_fallback_models,
            "circuit_breaker": self._summary_circuit_breaker,
            "spend_guard": self._summary_spend_guard,
            "timeout": timeout_seconds,
            "l2_budget_ratio": self._config.l2_budget_ratio,
            "l3_truncate_tokens": self._config.l3_truncate_tokens,
            "focus_topic": focus_topic or "",
            "custom_instructions": self._config.custom_instructions,
        }
        if deadline is not None and self._summary_callable_accepts_keyword(
            summarize_with_escalation,
            "deadline",
        ):
            summarize_kwargs["deadline"] = deadline
        if deadline is not None and self._summary_callable_accepts_keyword(
            summarize_with_escalation,
            "cancel_event",
        ):
            summarize_kwargs["cancel_event"] = cancel_event
        summary_text, level = self._call_with_hard_foreground_deadline(
            lambda: summarize_with_escalation(**summarize_kwargs),
            deadline=deadline,
            timeout_error=TimeoutError(
                "condensation provider exceeded the foreground deadline"
            ),
            cancel_event=cancel_event,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "after provider",
        )
        source_node_ids = [node.node_id for node in nodes]
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "after source projection",
        )
        earliest_at, latest_at = self._call_bounded_foreground_dag_read(
            self._dag.get_source_time_window,
            source_node_ids,
            deadline=deadline,
            stage="condensation",
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "after source time lookup",
        )
        summary_tokens = count_tokens(summary_text)
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "after output token count",
        )
        expand_hint = self._extract_expand_hint(summary_text)
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "after expand-hint extraction",
        )
        created_at = time.time()
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "after node timestamp",
        )
        condensed_node = SummaryNode(
            session_id=self._session_id,
            depth=depth + 1,
            summary=summary_text,
            token_count=summary_tokens,
            source_token_count=source_tokens,
            source_ids=source_node_ids,
            source_type="nodes",
            created_at=created_at,
            earliest_at=earliest_at,
            latest_at=latest_at,
            expand_hint=expand_hint,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "condensation",
            "before DAG publication",
        )
        self._publish_foreground_dag_node(
            condensed_node,
            deadline=deadline,
            stage="condensation",
        )
        self._invalidate_rollups_for_published_node(
            condensed_node,
            deadline=deadline,
        )
        return source_tokens, summary_tokens, level

    def _call_bounded_foreground_dag_read(
        self,
        operation,
        *args,
        deadline: float | None,
        stage: str,
    ):
        """Call a deadline-aware DAG read while preserving legacy test shims."""
        if deadline is None:
            return operation(*args)
        self._foreground_deadline_checkpoint(
            deadline,
            stage,
            "before DAG read",
        )
        remaining = self._foreground_deadline_remaining(deadline)
        if remaining is None or remaining <= 0:
            raise ForegroundDeadlineExceeded(stage, "before DAG read")
        available: set[str] = set()
        accepts_kwargs = False
        try:
            parameters = inspect.signature(operation).parameters.values()
            available = {parameter.name for parameter in parameters}
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_kwargs = True
        bounded_kwargs = {
            "lock_timeout_seconds": remaining,
            "deadline": deadline,
            "deadline_clock": self._foreground_now,
            "busy_timeout_ms": max(0, int(remaining * 1000)),
        }
        kwargs = {
            name: value
            for name, value in bounded_kwargs.items()
            if accepts_kwargs or name in available
        }
        try:
            result = operation(*args, **kwargs)
        except TimeoutError as exc:
            raise ForegroundDeadlineExceeded(
                stage,
                "DAG read lock wait exceeded the shared deadline",
            ) from exc
        except sqlite3.OperationalError as exc:
            if not _is_sqlite_locked_error(exc):
                raise
            raise ForegroundDeadlineExceeded(
                stage,
                "SQLite contention prevented a bounded DAG read",
            ) from exc
        self._foreground_deadline_checkpoint(
            deadline,
            stage,
            "after DAG read",
        )
        return result

    def _summary_frontier_nodes(
        self,
        *,
        deadline: float | None = None,
    ) -> List[SummaryNode]:
        """Return all provider-visible summary frontier nodes for the active session."""
        return self._call_bounded_foreground_dag_read(
            self._dag.get_active_frontier_nodes,
            self._session_id,
            deadline=deadline,
            stage="summary frontier read",
        )

    def _summary_frontier_tokens(self, *, deadline: float | None = None) -> int:
        return sum(
            node.token_count
            for node in self._summary_frontier_nodes(deadline=deadline)
        )

    def _select_threshold_sweep_condensation_group(
        self,
        *,
        deadline: float | None = None,
    ) -> List[SummaryNode]:
        """Prefer routine fanin/depth, then allow bounded pressure condensation."""
        by_depth: dict[int, list[SummaryNode]] = {}
        for node in self._summary_frontier_nodes(deadline=deadline):
            by_depth.setdefault(node.depth, []).append(node)
        if not by_depth:
            return []
        fanin = max(2, self._config.condensation_fanin)
        preferred_max_depth = self._config.incremental_max_depth
        for depth in sorted(by_depth):
            nodes = by_depth[depth]
            within_preferred_depth = preferred_max_depth < 0 or depth < preferred_max_depth
            if within_preferred_depth and len(nodes) >= fanin:
                return nodes[:fanin]
        # The frontier still exceeds its sweep target but no routine group is
        # available. Permit a same-depth partial group or depth beyond the
        # preferred routine maximum; the outer sweep budget keeps this bounded.
        for depth in sorted(by_depth):
            nodes = by_depth[depth]
            if len(nodes) >= 2:
                return nodes[: min(fanin, len(nodes))]
        return []

    def _run_threshold_sweep_condensation(
        self,
        *,
        target_tokens: int,
        pass_budget: int,
        deadline: float,
        focus_topic: Optional[str] = None,
    ) -> tuple[int, str]:
        """Condense an oversized summary frontier within the remaining sweep budget."""
        passes = 0
        while self._summary_frontier_tokens(deadline=deadline) > target_tokens:
            if passes >= pass_budget:
                return passes, "pass_budget_exhausted"
            if time.monotonic() >= deadline:
                return passes, "time_budget_exhausted"
            group = self._select_threshold_sweep_condensation_group(
                deadline=deadline,
            )
            if not group:
                return passes, "no_same_depth_condensation_group"
            before = self._summary_frontier_tokens(deadline=deadline)
            try:
                self._condense_summary_nodes(
                    group,
                    focus_topic=focus_topic,
                    deadline=deadline,
                )
            except Exception as exc:
                logger.warning(
                    "LCM threshold full sweep condensation stopped after %d pass(es): %s",
                    passes,
                    exc,
                )
                return passes, "condensation_error"
            passes += 1
            after = self._summary_frontier_tokens(deadline=deadline)
            if after >= before:
                return passes, "condensation_no_progress"
        return passes, "summary_prefix_target_reached"

    # -- Internal: context assembly ----------------------------------------

    @staticmethod
    def _append_lcm_note_to_content(content: Any) -> Any:
        note = (
            "\n\n[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "Use lcm_grep to search history, lcm_describe to inspect the DAG, "
            "and lcm_expand to recover original details from any summary.]"
        )
        if isinstance(content, str):
            return content + note
        note_part = {"type": "text", "text": note.lstrip()}
        if content is None:
            return note.lstrip()
        if isinstance(content, list):
            return list(content) + [note_part]
        normalized = normalize_content_value(content) or ""
        return normalized + note

    @staticmethod
    def _is_preserved_todo_context_message(message: Dict[str, Any]) -> bool:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content.lstrip().startswith(_PRESERVED_TODO_CONTEXT_PREFIX)

    @staticmethod
    def _preserved_objective_context_content(message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX) else ""

    def _sanitized_preserved_objective_context_content(
        self,
        message: Dict[str, Any],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> str:
        preserved_objective = self._preserved_objective_context_content(message)
        if not preserved_objective:
            return ""
        return self._sanitize_preserved_objective_content(
            preserved_objective,
            role=str(message.get("role") or "user"),
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            ownership=ownership,
        )

    def _sanitize_active_preserved_objective_message(
        self,
        message: Dict[str, Any],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> Dict[str, Any]:
        sanitized_content = self._sanitized_preserved_objective_context_content(
            message,
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            ownership=ownership,
        )
        if not sanitized_content or sanitized_content == message.get("content"):
            return message
        sanitized = dict(message)
        sanitized["content"] = sanitized_content
        return sanitized

    def _sanitize_preserved_objective_content(
        self,
        content: str,
        role: str = "user",
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> str:
        self._context_projection_checkpoint(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            operation="preserved objective projection",
        )
        content = strip_injected_context_blocks(content)
        content = protect_inline_payloads_in_text(
            content,
            role=role,
            session_id=self._session_id,
            field_path="preserved_objective.content",
            config=self._config,
            hermes_home=self._hermes_home,
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            ownership=ownership,
        )
        self._context_projection_checkpoint(
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            operation="preserved objective projection",
        )
        return content

    def _build_preserved_objective_summary_part(
        self,
        message: Dict[str, Any],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        content = self._sanitize_preserved_objective_content(
            content,
            role=str(message.get("role") or "user"),
            deadline=deadline,
            cancel_event=cancel_event,
            now=now,
            ownership=ownership,
        )
        return f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\n{content}"

    def _latest_user_context_anchor(
        self,
        messages: List[Dict[str, Any]],
        selected_tail: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        now: Callable[[], float] | None = None,
        ownership: ExternalizationOwnership | None = None,
    ) -> Optional[str]:
        """Return a scaffolded newest real user objective omitted from the tail.

        Tool-heavy turns can push the operative user request outside the fresh
        tail while retaining only assistant/tool traces from that turn.  The
        returned text is active-context scaffolding, not raw conversation: it is
        emitted inside the summary block so restart reconciliation ignores it
        instead of ingesting a duplicate non-contiguous user message.

        Previous preserved-objective scaffolds are derived context, not real
        user turns, so they are not eligible as the next anchor source. Once a
        reverse scan reaches one, older user turns are stale relative to that
        synthetic continuity marker and must not be promoted as current intent.
        """
        selected_tail_messages = [msg for msg in selected_tail if isinstance(msg, dict)]
        for message_index, message in enumerate(reversed(messages)):
            if message_index % 32 == 0:
                self._context_projection_checkpoint(
                    deadline=deadline,
                    cancel_event=cancel_event,
                    now=now,
                    operation="context anchor scan",
                )
            if not isinstance(message, dict):
                continue
            content_text = text_content_for_pattern_matching(message.get("content")) or ""
            if (
                self._matches_ignore_message_patterns(message)
                or self._mapped_stored_row_matches_ignore_message_patterns(
                    message,
                    deadline=deadline,
                )
                or self._is_volatile_ignored_quarantine_placeholder(
                    message,
                    content_text,
                )
                or self._is_ignored_active_replay_placeholder(message, content_text)
            ):
                continue
            if self._preserved_objective_context_content(message):
                return None
            if message.get("role") != "user":
                continue
            if self._is_preserved_todo_context_message(message):
                continue
            if any(message == selected for selected in selected_tail_messages):
                return None
            return self._build_preserved_objective_summary_part(
                message,
                deadline=deadline,
                cancel_event=cancel_event,
                now=now,
                ownership=ownership,
            )
        return None

    @staticmethod
    def _newest_user_message_text(messages: List[Dict[str, Any]]) -> str:
        """Return the newest real user turn's text (SPEC F injection query).

        Scans from the tail so the block reflects the turn the host is about to
        answer. Returns "" when the tail carries no user text (e.g. a tool-only
        continuation) — the caller then leaves the feature inert.
        """
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = (text_content_for_pattern_matching(message.get("content")) or "").strip()
            if text:
                return text
        return ""

    def _build_proactive_recall_message(
        self,
        tail_messages: List[Dict[str, Any]],
        summary_role: str,
        active_node_ids: set,
        *,
        deadline: float | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the single proactive "relevant memories" block, or None.

        SPEC F: at assembly time embed the newest user message, run the
        lcm_recall pipeline (k=6, moderate scope bias), drop hits already in the
        active context (summary-prefix node ids + current session) and hits
        below the relevance floor, then render ONE budget-capped block. Any
        failure or timeout injects nothing — assembly is never blocked. Wrapped
        in <relevant-memories> so it is stripped before ingest/summarization and
        never re-enters the lossless store as a real turn.
        """
        config = self._config
        if not getattr(config, "proactive_recall_enabled", False):
            return None
        # Injection is an embedding-query feature: silently inert when embeddings
        # are disabled/unwarmed (no provider to embed the newest message).
        if not getattr(config, "embeddings_enabled", False):
            return None

        query = self._newest_user_message_text(tail_messages)
        if not query:
            return None

        # SPEC F retrieval constants: small k, moderate scope bias (favor — never
        # hard-filter — the current conversation).
        recall_k = 6
        scope_bias = 0.3
        min_score = float(getattr(config, "proactive_recall_min_score", 0.01) or 0.0)
        budget_tokens = int(getattr(config, "proactive_recall_budget_tokens", 500) or 0)
        if budget_tokens <= 0:
            return None
        provider_override = getattr(config, "proactive_recall_provider", "") or ""

        try:
            recall_kwargs: dict[str, Any] = {
                "engine": self,
                "provider_override": provider_override,
            }
            if deadline is not None:
                recall_kwargs["deadline"] = deadline

            def recall_operation() -> str:
                return lcm_tools.lcm_recall(
                    {
                        "query": query,
                        "limit": recall_k,
                        "scope_bias": scope_bias,
                        "include": "all",
                    },
                    **recall_kwargs,
                )
            if deadline is None:
                raw = recall_operation()
            else:
                recall_cancel_event = threading.Event()
                raw = self._call_with_hard_foreground_deadline(
                    recall_operation,
                    deadline=deadline,
                    timeout_error=TimeoutError(
                        "proactive recall exceeded the shared foreground deadline"
                    ),
                    cancel_event=recall_cancel_event,
                )
                if self._foreground_now() >= float(deadline):
                    raise TimeoutError(
                        "proactive recall completed after the shared foreground deadline"
                    )
            payload = json.loads(raw)
        except TimeoutError:
            self._proactive_recall_timeout_count += 1
            return None
        except Exception:  # noqa: BLE001 - never let recall break assembly
            logger.debug("LCM proactive recall failed; injecting nothing", exc_info=True)
            self._proactive_recall_skipped_count += 1
            return None

        if payload.get("timeout"):
            # Deadline hit: inject nothing this turn (contract), count it.
            self._proactive_recall_timeout_count += 1
            return None

        surviving: list[dict] = []
        for hit in payload.get("hits", []):
            if not isinstance(hit, dict):
                continue
            # Dedupe: skip anything already represented in the active context —
            # the current session (its summary prefix + recent tail window) and
            # any summary node already placed in the prefix.
            if hit.get("from_current_session"):
                continue
            node_id = hit.get("node_id")
            if node_id is not None and node_id in active_node_ids:
                continue
            # Relevance floor.
            if float(hit.get("score") or 0.0) < min_score:
                continue
            snippet = (hit.get("snippet") or "").strip()
            if not snippet:
                continue
            surviving.append(hit)

        if not surviving:
            self._proactive_recall_skipped_count += 1
            return None

        # Render 1-3 items inside the token budget. The header labels the block
        # as retrieved memory (provenance-honest, never asserted as fact).
        header = (
            "Possibly relevant memories (retrieved from earlier conversations — "
            "context only, not asserted as fact):"
        )
        rendered_lines: list[str] = []
        for hit in surviving[:3]:
            ts = hit.get("timestamp") or 0
            try:
                when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(ts))) if ts else "unknown time"
            except (TypeError, ValueError, OSError):
                when = "unknown time"
            snippet = (hit.get("snippet") or "").strip().replace("\n", " ")
            expand = hit.get("expand_hint") or ""
            line = f"- [{when}] {snippet}"
            if expand:
                line += f" (expand: {expand})"
            candidate_lines = rendered_lines + [line]
            block = self._wrap_relevant_memories(header, candidate_lines)
            if count_message_tokens({"role": summary_role, "content": block}) > budget_tokens:
                break
            rendered_lines = candidate_lines

        if not rendered_lines:
            self._proactive_recall_skipped_count += 1
            return None

        content = self._wrap_relevant_memories(header, rendered_lines)
        self._proactive_recall_injected_count += 1
        return {"role": summary_role, "content": content}

    @staticmethod
    def _wrap_relevant_memories(header: str, lines: List[str]) -> str:
        body = "\n".join([header, *lines])
        return f"<relevant-memories>\n{body}\n</relevant-memories>"

    def _assemble_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
        include_lcm_note: bool = True,
        *,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Build the active context from DAG summaries + fresh tail.

        Structure:
          [leading anchor, normally system prompt]
          [highest-depth summary nodes first, then lower]
          [fresh tail messages]
        """
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "assembly entry",
        )
        result = []

        # Leading anchor with optional LCM annotation. Only a true system prompt
        # is a safe permanent anchor; gateway sessions can start directly with
        # user messages, and those user turns must remain compactable.
        leading_msg = system_msg.copy() if system_msg is not None else None
        if leading_msg is not None:
            if (
                leading_msg.get("role") == "system"
                and self.compression_count == 0
                and include_lcm_note
            ):
                leading_msg["content"] = self._append_lcm_note_to_content(
                    leading_msg.get("content", "")
                )
            result.append(leading_msg)

        assembly_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )

        # Stub durably externalized evictable tool payloads before the assembly
        # budget pass so the selector sees their reduced provider-visible cost.
        # The helper protects the configured fresh tail and is fail-open.
        assembly_tail_messages = self._call_foreground_projection(
            self._stub_large_tool_results_for_active_replay,
            tail_messages,
            deadline=deadline,
            stage="context replay projection",
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "tool replay projection",
        )
        tail_selected = assembly_tail_messages
        anchor_source = getattr(self, "_pending_context_anchor_messages", None)
        if anchor_source is None:
            anchor_source = tail_messages
        anchor_part: Optional[str] = None
        summary_budget = None
        if assembly_cap is not None:
            used = count_message_tokens(leading_msg) if leading_msg is not None else 0
            kept_tail_reversed: list[Dict[str, Any]] = []
            tail_token_total = 0
            tail_for_selection = self._call_foreground_projection(
                self._sanitize_active_context_messages,
                assembly_tail_messages,
                deadline=deadline,
                stage="context tail cleanup",
                insert_missing_tool_stubs=False,
            )
            skipped_tail_gap = False
            for tail_index, msg in enumerate(reversed(tail_for_selection)):
                if tail_index % 32 == 0:
                    self._foreground_deadline_checkpoint(
                        deadline,
                        "context reconstruction",
                        "tail budget selection",
                    )
                msg_tokens = count_message_tokens(msg)
                if used + tail_token_total + msg_tokens > assembly_cap:
                    if self._is_budget_droppable_tail_message(msg):
                        skipped_tail_gap = True
                        continue
                    break
                if skipped_tail_gap:
                    break
                kept_tail_reversed.append(msg)
                tail_token_total += msg_tokens
            tail_selected = list(reversed(kept_tail_reversed))
            summary_budget = max(0, assembly_cap - used - tail_token_total)
        if anchor_source is not None:
            anchor_part = self._call_foreground_projection(
                self._latest_user_context_anchor,
                anchor_source,
                tail_selected,
                deadline=deadline,
                stage="context objective projection",
            )
            self._foreground_deadline_checkpoint(
                deadline,
                "context reconstruction",
                "context anchor selection",
            )

        # Collect DAG summaries — highest depth first for context hierarchy
        summary_parts: list[str] = []
        last_role = result[-1].get("role", "system") if result else "system"
        if not result or result[-1].get("role") == "system":
            # The summary becomes the first provider-visible message: either no
            # leading anchor exists (gateway-style assembly) or the system
            # prompt is the only anchor, which Anthropic extracts into a
            # separate field. Either way messages[0] must be role "user"; an
            # assistant summary here is rejected with HTTP 400 after the second
            # compaction.
            summary_role = "user"
        else:
            summary_role = "assistant" if last_role != "assistant" else "user"
        if anchor_part is not None:
            anchor_msg = {"role": summary_role, "content": anchor_part}
            if summary_budget is None or count_message_tokens(anchor_msg) <= summary_budget:
                summary_parts.append(anchor_part)

        # Node ids placed in the summary prefix — used to dedupe proactive-recall
        # injection against summaries already visible in the active context.
        active_summary_node_ids: set = set()
        frontier_nodes = self._summary_frontier_nodes(deadline=deadline)
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "summary frontier load",
        )
        for node_index, node in enumerate(frontier_nodes):
            if node_index % 32 == 0:
                self._foreground_deadline_checkpoint(
                    deadline,
                    "context reconstruction",
                    "summary frontier assembly",
                )
            active_summary_node_ids.add(node.node_id)
            depth_label = {
                0: "Recent",
                1: "Session Arc",
                2: "Durable",
            }.get(node.depth, f"Depth-{node.depth}")
            summary_parts.append(
                f"[{depth_label} Summary (d{node.depth}, node {node.node_id})]\n"
                f"{node.summary}\n"
                f"[Expand for details: {node.expand_hint}]"
            )

        if summary_parts:
            selected_parts = summary_parts
            if summary_budget is not None:
                selected_parts = []
                for part_index, part in enumerate(summary_parts):
                    if part_index % 32 == 0:
                        self._foreground_deadline_checkpoint(
                            deadline,
                            "context reconstruction",
                            "summary budget selection",
                        )
                    candidate = "\n\n---\n\n".join(selected_parts + [part])
                    candidate_msg = {"role": summary_role, "content": candidate}
                    if count_message_tokens(candidate_msg) > summary_budget:
                        if part == anchor_part:
                            continue
                        continue
                    selected_parts.append(part)
            if selected_parts:
                combined = "\n\n---\n\n".join(selected_parts)
                result.append(
                    {
                        "role": summary_role,
                        "content": combined,
                        # Host persistence may use this explicit provenance to
                        # exclude/supersede intermediate model projections while
                        # retaining raw source turns and the current active view.
                        "display_kind": "context_projection",
                        "display_metadata": {"engine": "lcm"},
                    }
                )

        # Proactive memory injection (SPEC F, default-off). One bounded block is
        # placed adjacent to the summary prefix — a stable position below the
        # cache-stable summaries and above the volatile fresh tail — so the
        # already-volatile tail region absorbs its per-turn variability and the
        # cached summary prefix is left intact. Never inside the fresh tail.
        proactive_msg = self._build_proactive_recall_message(
            tail_messages,
            summary_role,
            active_summary_node_ids,
            deadline=deadline,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "proactive recall assembly",
        )
        if proactive_msg is not None:
            result.append(proactive_msg)

        # Fresh tail
        result.extend(tail_selected)

        # ── Active-context cleanup / tool-pair guardrail ──
        # Drop assistant turns that carry only blank/internal structured content,
        # then ensure provider-valid tool-call/result sequencing.
        result = self._call_foreground_projection(
            self._sanitize_active_context_messages,
            result,
            deadline=deadline,
            stage="active context cleanup",
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "active context cleanup",
        )
        if leading_msg is None:
            while result and result[0].get("role") in {"assistant", "tool"}:
                result = result[1:]
        if (
            assembly_cap is not None
            and anchor_part is not None
            and count_messages_tokens(result) > assembly_cap
        ):
            trimmed_result: list[Dict[str, Any]] = []
            for result_index, msg in enumerate(result):
                if result_index % 32 == 0:
                    self._foreground_deadline_checkpoint(
                        deadline,
                        "context reconstruction",
                        "objective anchor trimming",
                    )
                content = normalize_content_value(msg.get("content")) or ""
                if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX not in content:
                    trimmed_result.append(msg)
                    continue
                parts = [
                    part for part in content.split("\n\n---\n\n")
                    if not part.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                ]
                if parts:
                    trimmed = msg.copy()
                    trimmed["content"] = "\n\n---\n\n".join(parts)
                    trimmed_result.append(trimmed)
            result = self._call_foreground_projection(
                self._sanitize_active_context_messages,
                trimmed_result,
                deadline=deadline,
                stage="objective anchor cleanup",
            )

        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "assembly completion",
        )
        return result

    def _is_budget_droppable_tail_message(self, message: Dict[str, Any]) -> bool:
        """Return whether an over-budget tail message may be evicted.

        User turns are prompt-bearing context and stop tail selection when they
        cannot fit. Assistant/tool turns are derived context; if one bulky turn
        blocks older prompt material, skip it and keep scanning for budgetable
        user intent or compact status that still fits.
        """
        role = message.get("role")
        if role not in {"assistant", "tool"}:
            return False
        content = normalize_content_value(message.get("content")) or ""
        if _PRESERVED_TODO_CONTEXT_PREFIX in content:
            return False
        if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX in content:
            return False
        return True

    def _finalize_forced_overflow_result(
        self,
        original_messages: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
        ingest_cleanup_changed_active_context: bool = False,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]]:
        if compressed != original_messages or ingest_cleanup_changed_active_context:
            self._last_compression_status = "overflow_recovery"
            self._last_compression_noop_reason = ""
            self._align_ingest_cursor_to_active_context(
                compressed,
                deadline=deadline,
            )
            logger.info(
                "LCM assembly guardrail recovery: %d messages → %d (no new summary node)",
                len(original_messages),
                len(compressed),
            )
        else:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = (
                "forced overflow recovery found no droppable active-context messages"
            )

        effective_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )
        if effective_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > effective_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d; returning best-effort context (%d tokens)",
                    effective_cap,
                    count_messages_tokens(compressed),
                )
        return compressed

    def _should_force_overflow_recovery(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return False

        tokens = self._overflow_recovery_signal_tokens(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if tokens is None:
            return False
        return tokens >= assembly_cap

    def _overflow_recovery_signal_tokens(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        candidates: list[int] = []
        if observed_tokens is not None and observed_tokens > 0:
            candidates.append(observed_tokens)
        if messages is not None:
            candidates.append(count_messages_tokens(messages))
        if not candidates:
            return None
        return max(candidates)

    def _overflow_recovery_assembly_cap(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return None
        if messages is None or observed_tokens is None or observed_tokens <= 0:
            return assembly_cap

        message_tokens = count_messages_tokens(messages)
        overhead_tokens = max(0, observed_tokens - message_tokens)
        return max(1, assembly_cap - overhead_tokens)

    def _effective_assembly_token_cap(self) -> Optional[int]:
        """Return the active assembly cap, if any.

        Two knobs can constrain the assembled active context:
        - max_assembly_tokens: explicit hard cap
        - reserve_tokens_floor: keep headroom inside context_length
        """
        caps: list[int] = []
        admission_cap = getattr(self, "_admission_assembly_tokens", 0)
        if admission_cap:
            caps.append(admission_cap)


        if self._config.max_assembly_tokens > 0:
            caps.append(self._config.max_assembly_tokens)

        if self.context_length > 0 and self._config.reserve_tokens_floor > 0:
            reserve_cap = self.context_length - self._config.reserve_tokens_floor
            if reserve_cap > 0:
                caps.append(reserve_cap)
            else:
                logger.warning(
                    "LCM reserve_tokens_floor=%d disables reserve-based assembly cap because context_length=%d",
                    self._config.reserve_tokens_floor,
                    self.context_length,
                )

        if not caps:
            return None

        return max(1, min(caps))

    def _overflow_retained_raw_reference(
        self,
        tail_messages: List[Dict[str, Any]],
        *,
        deadline: float | None = None,
    ) -> Dict[str, Any] | None:
        """Return a bounded reference only when every raw tail row is durable.

        Overflow assembly may hide raw rows from the provider-visible request,
        but it must never imply that an unpersisted region is covered. Synthetic
        context projections are already derived from canonical DAG state and are
        therefore excluded from this raw-row proof.
        """
        raw_messages = [
            message
            for message in tail_messages
            if not (
                message.get("display_kind") == "context_projection"
                or self._looks_like_active_summary_blob(
                    normalize_content_value(message.get("content")) or ""
                )
            )
        ]
        if not raw_messages:
            return None
        id_map = self._get_store_id_map_for_messages(
            raw_messages,
            deadline=deadline,
        )
        from .historical_projection import source_ids

        durable_ids: list[int] = []
        for message in raw_messages:
            mapped = source_ids(id_map, message)
            if not mapped:
                return None
            durable_ids.extend(int(store_id) for store_id in mapped)
        durable_ids = sorted(dict.fromkeys(durable_ids))
        if not durable_ids:
            return None
        session_id = str(self._session_id or "")
        content = (
            "[LCM active recall incomplete; raw archive retained outside active context] "
            f"session={session_id} archived_range={durable_ids[0]}..{durable_ids[-1]} "
            f"mapped_rows={len(durable_ids)}. This is an archive locator, not a "
            "summary or claim of semantic preservation; inspect raw rows explicitly."
        )
        return {
            "role": "user",
            "content": content,
            "display_kind": "context_projection",
            "display_metadata": {
                "engine": "lcm",
                "projection": "retained_raw_reference",
                "session_id": session_id,
                "first_store_id": durable_ids[0],
                "last_store_id": durable_ids[-1],
                "mapped_rows": len(durable_ids),
                "active_recall_incomplete": True,
                "semantic_coverage": False,
            },
        }

    # -- Internal: helpers -------------------------------------------------

    def _assemble_overflow_recovery_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
        *,
        deadline: float | None = None,
    ) -> List[Dict[str, Any]]:
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "overflow recovery entry",
        )
        retained_reference: Dict[str, Any] | None = None
        effective_cap = assembly_cap_override
        requires_budget_eviction = False
        if effective_cap is not None and tail_messages:
            projected_tail = self._call_foreground_projection(
                self._stub_large_tool_results_for_active_replay,
                tail_messages,
                deadline=deadline,
                stage="overflow coverage projection probe",
            )
            projected_tail = list(projected_tail or tail_messages)
            projected_tokens = count_messages_tokens(projected_tail)
            if system_msg is not None:
                projected_tokens += count_message_tokens(system_msg)
            requires_budget_eviction = projected_tokens > int(effective_cap)
        if requires_budget_eviction:
            retained_reference = self._overflow_retained_raw_reference(
                tail_messages,
                deadline=deadline,
            )
            if retained_reference is None:
                raise RuntimeError(
                    "overflow recovery refused to evict an unarchived raw tail"
                )
            reference_tokens = count_message_tokens(retained_reference)
            effective_cap = int(effective_cap or 0) - reference_tokens
            if effective_cap <= 0:
                raise RuntimeError(
                    "overflow recovery cap cannot fit the durable archive locator"
                )

        def attach_retained_reference(
            candidate: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            if retained_reference is None:
                return candidate
            projected = list(candidate)
            insert_at = (
                1
                if projected and projected[0].get("role") == "system"
                else 0
            )
            projected.insert(insert_at, retained_reference)
            cleaned = self._call_foreground_projection(
                self._sanitize_active_context_messages,
                projected,
                deadline=deadline,
                stage="overflow retained-raw reference cleanup",
            )
            result = list(cleaned or projected)
            if assembly_cap_override is not None and count_messages_tokens(result) > int(
                assembly_cap_override
            ):
                raise RuntimeError(
                    "overflow recovery exceeded its hard assembly cap"
                )
            return result

        if tail_messages:
            first = tail_messages[0]
            content = first.get("content") or ""
            role = first.get("role") or ""
            if role == "assistant" and self._looks_like_active_summary_blob(content):
                candidate = self._call_with_foreground_deadline(
                    self._assemble_context,
                    system_msg,
                    tail_messages[1:],
                    assembly_cap_override=effective_cap,
                    include_lcm_note=False,
                    deadline=deadline,
                )
                self._foreground_deadline_checkpoint(
                    deadline,
                    "context reconstruction",
                    "overflow summary replacement",
                )
                if any(
                    (msg.get("content") or "") == content
                    for msg in (candidate[1:] if system_msg is not None else candidate)
                ):
                    return attach_retained_reference(candidate)

        candidate = self._call_with_foreground_deadline(
            self._assemble_context,
            system_msg,
            tail_messages,
            assembly_cap_override=effective_cap,
            include_lcm_note=False,
            deadline=deadline,
        )
        self._foreground_deadline_checkpoint(
            deadline,
            "context reconstruction",
            "overflow recovery assembly",
        )
        minimum_candidate_len = 1 if system_msg is not None else 0
        if len(candidate) == minimum_candidate_len and tail_messages:
            if retained_reference is not None:
                # The bounded locator is the only provider-visible replacement
                # for evicted durable raw. Never re-add an unbounded raw row.
                return attach_retained_reference(candidate)
            fallback = ([system_msg] if system_msg is not None else []) + [tail_messages[-1]]
            fallback = self._call_foreground_projection(
                self._sanitize_active_context_messages,
                fallback,
                deadline=deadline,
                stage="overflow recovery cleanup",
            )
            self._foreground_deadline_checkpoint(
                deadline,
                "context reconstruction",
                "overflow recovery fallback",
            )
            return attach_retained_reference(list(fallback or []))
        return attach_retained_reference(candidate)

    @staticmethod
    def _looks_like_active_summary_blob(content: str) -> bool:
        if not isinstance(content, str) or not content:
            return False
        block = (
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]\n"
            r".*?\n"
            r"\[Expand for details: .*?\]"
        )
        pattern = rf"^{block}(?:\n\n---\n\n{block})*$"
        return re.fullmatch(pattern, content, flags=re.DOTALL) is not None

    def _derive_auto_focus_topic(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Infer a compact focus hint from the most recent real user turns.

        Walks the message list backwards, collecting up to
        ``_AUTO_FOCUS_MAX_TURNS`` user messages (skipping context summaries
        and empty turns).  Returns a brief text block suitable for injection
        into the summarizer prompt as ``focus_topic``.

        IMPORTANT: The ``messages`` parameter must be ``working_messages``
        (output of ``_ingest_messages``), not raw messages.  ``working_messages``
        has already been redacted by ``_redact_active_replay_messages``.

        As an additional safety layer, text extracted by
        ``text_content_for_pattern_matching`` is run through
        ``redact_sensitive_text`` with the active config.  This covers
        sensitive values that ``_redact_active_replay_messages`` misses
        (e.g., dict/JSON token content deserialized into text,
        bearer-style auth text that survived structured-content flattening).

        Mirrors Hermes upstream ``ContextCompressor._derive_auto_focus_topic``
        from ``fix/compression-auto-focus-topic``.
        """
        candidates: list[str] = []
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            # Skip context compaction summaries — they are synthetic, not
            # real user intent.
            if self._is_context_summary_content(content):
                continue
            text = (text_content_for_pattern_matching(content) or "").strip()
            if self._matches_ignore_message_patterns(msg) or self._is_volatile_ignored_quarantine_placeholder(
                msg,
                text,
            ) or self._is_ignored_active_replay_placeholder(msg, text):
                continue
            # Additional redaction safety net: run extracted text through the
            # configured redaction path.  _redact_active_replay_messages uses
            # parse_json_strings=False for content, so structured content
            # (dict/JSON tokens, bearer-style auth text) may not be fully
            # covered.  This extra pass ensures the same redaction rules apply
            # to whatever text is extracted for the focus topic.
            text = redact_sensitive_text(text, self._config)
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) > _AUTO_FOCUS_TURN_MAX_CHARS:
                text = text[: _AUTO_FOCUS_TURN_MAX_CHARS - 1].rstrip() + "…"
            candidates.append(text)
            if len(candidates) >= _AUTO_FOCUS_MAX_TURNS:
                break

        if not candidates:
            return None

        candidates.reverse()
        focus = "Recent user focus:\n" + "\n".join(f"- {item}" for item in candidates)
        if len(focus) > _AUTO_FOCUS_MAX_CHARS:
            focus = focus[: _AUTO_FOCUS_MAX_CHARS - 1].rstrip() + "…"
        return focus

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        """Check whether message content is a synthetic context summary.

        Only checks string content — LCM/ Hermes compression summaries are
        always stored as plain strings, never as structured multimodal parts.
        """
        if not isinstance(content, str):
            return False
        return (
            "CONTEXT COMPACTION" in content
            or "CONTEXT SUMMARY" in content
            or "Earlier turns have been compacted" in content
            or "Earlier turns were compacted" in content
        )

    @staticmethod
    def _extract_expand_hint(summary: str) -> str:
        """Extract the 'Expand for details about:' line from a summary."""
        marker = "Expand for details about:"
        idx = summary.rfind(marker)
        if idx >= 0:
            hint = summary[idx + len(marker):].strip()
            # Take first line only
            return hint.split("\n")[0].strip()
        return ""

    # -- Rotate ------------------------------------------------------------

    def backup_dir(self) -> Path:
        """Return the directory where LCM backup snapshots are written.

        Centralized so the timestamped ``/lcm backup`` slot and the rolling
        ``/lcm rotate apply`` slot share the same directory derivation.
        """
        db_path = Path(self._store.db_path)
        backup_root = (
            Path(self._hermes_home).expanduser()
            if getattr(self, "_hermes_home", "")
            else db_path.parent
        )
        return backup_root / "backups" / "lcm"

    def rotate_backup_path(self) -> Path:
        """Return the rolling rotate-latest SQLite backup path for this engine.

        Centralized so command.py (which writes the backup) and get_status()
        (which reads its mtime to surface last_rotate_at) cannot drift.
        """
        db_path = Path(self._store.db_path)
        return self.backup_dir() / f"{db_path.stem}-rotate-latest.sqlite3"

    def rotate_active_session(
        self,
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Compact the active session in-place without changing identity.

        Read-only by default (``apply=False``). Returns a preview describing
        what would change. When ``apply=True``, advances the lifecycle frontier
        marker past the pre-tail raw messages so they are no longer replayed
        into active context on subsequent bootstrap. Raw messages remain in
        the SQLite store and are recoverable through ``lcm_load_session`` and
        ``lcm_expand`` — the lossless raw recovery contract is preserved.

        Refuses on sessions that are unbound, ignored, or stateless.

        Two frontier markers are intentionally kept separate:

        - The **persisted lifecycle frontier**
          (``lifecycle_state.current_frontier_store_id``) is the
          bootstrap signal — on next session start, raw rows at or
          below it are not replayed into the active context. Rotate
          advances this marker.
        - The **in-process source-mapping marker**
          (``self._last_compacted_store_id``) tracks raw rows that the
          *current process* has already moved into summary DAG nodes.
          ``_get_store_ids_for_messages`` uses it to filter candidates
          when mapping in-memory active messages back to ``store_id``.
          Rotate deliberately does NOT advance this marker: pre-tail
          raw messages remain in the in-memory active context until
          the host rebuilds it, so a normal ``compress()`` later in
          the same process can still summarize them with correct
          ``source_ids`` lineage. On next process start,
          ``_bind_lifecycle_state`` reads the persisted frontier into
          the in-process marker — at that point the active context is
          being built from scratch, so the contract holds.

        Refusal/no-op reason codes (returned as ``reason``):

        - ``no_active_session``: engine has no bound session or conversation.
        - ``session_ignored``: foreground session matched
          ``LCM_IGNORE_SESSION_PATTERNS``.
        - ``session_stateless``: foreground session matched
          ``LCM_STATELESS_SESSION_PATTERNS``.
        - ``no_pre_tail_content``: no stored messages precede the resolved
          count/token-bounded fresh tail; nothing to rotate.
        - ``empty_tail``: tail query returned no rows despite a non-zero
          count (concurrent deletion race); rotate cannot compute a boundary.
        - ``frontier_already_ahead``: lifecycle frontier is already at or
          past the proposed new frontier; rotate is a no-op.
        - ``stale_lifecycle_state``: apply requested but lifecycle's
          ``current_session_id`` did not match this engine's session, so
          ``advance_frontier`` did not persist the change.
        """
        session_id = self._session_id
        conversation_id = self._conversation_id

        if not session_id or not conversation_id:
            return {"ok": False, "reason": "no_active_session"}
        if self._session_ignored:
            return {"ok": False, "reason": "session_ignored", "session_id": session_id}
        if self._session_stateless:
            return {"ok": False, "reason": "session_stateless", "session_id": session_id}

        fresh_tail_count = max(1, int(self._config.fresh_tail_count))
        total_count = int(self._store.get_session_count(session_id))
        tail, fresh_tail_boundary = self._get_session_fresh_tail(
            session_id,
            minimum_count=1,
        )
        effective_fresh_tail_count = len(tail)

        state = self._lifecycle.get_by_conversation(conversation_id)
        current_frontier = int(state.current_frontier_store_id) if state else 0

        base = {
            "ok": True,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "total_message_count": total_count,
            "fresh_tail_count": fresh_tail_count,
            "fresh_tail_max_tokens": self._config.fresh_tail_max_tokens,
            "effective_fresh_tail_count": effective_fresh_tail_count,
            "effective_fresh_tail_tokens": fresh_tail_boundary.tokens,
            "fresh_tail_token_limited": fresh_tail_boundary.token_limited,
            "fresh_tail_tool_group_extended": fresh_tail_boundary.tool_group_extended,
            "current_frontier_store_id": current_frontier,
            "mode": "apply" if apply else "preview",
        }

        if total_count <= effective_fresh_tail_count:
            return {
                **base,
                "noop": True,
                "reason": "no_pre_tail_content",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        if not tail:
            # Concurrent deletion can empty the tail after the count check.
            # Surface the same shape callers expect for any other no-op so
            # downstream formatters can render it without KeyError.
            return {
                **base,
                "noop": True,
                "reason": "empty_tail",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        smallest_tail_store_id = int(tail[0].get("store_id") or 0)
        new_frontier = max(0, smallest_tail_store_id - 1)
        pre_tail_count = max(0, total_count - len(tail))

        is_noop = new_frontier <= current_frontier
        result = {
            **base,
            "pre_tail_message_count": pre_tail_count,
            "new_frontier_store_id": new_frontier,
            "noop": is_noop,
        }
        if is_noop:
            # Set the reason for both preview and apply so downstream
            # formatters can render a stable explanation. Preview previously
            # omitted the reason, which left _rotate_apply_text's preflight
            # check unable to distinguish frontier-already-ahead from other
            # no-ops.
            result["reason"] = "frontier_already_ahead"

        if not apply:
            return result

        if is_noop:
            return result

        new_state = self._lifecycle.advance_frontier(
            conversation_id,
            session_id,
            new_frontier,
        )
        # advance_frontier silently returns the unchanged state when its
        # session_id check fails (lifecycle_state.py:557-559). Detect that
        # by checking whether the persisted frontier actually advanced; only
        # promote the in-process marker on a confirmed persist.
        persisted_frontier = (
            int(new_state.current_frontier_store_id) if new_state else current_frontier
        )
        if persisted_frontier < new_frontier:
            return {
                **{k: v for k, v in result.items() if k != "ok"},
                "ok": False,
                "noop": False,
                "reason": "stale_lifecycle_state",
                "applied_frontier_store_id": persisted_frontier,
            }
        # Deliberately do NOT touch self._last_compacted_store_id here.
        # The in-process source-mapping marker must stay aligned with the
        # in-memory active context the host is still using. Pre-tail raw
        # messages remain in that active context until the host rebuilds
        # it; advancing the marker would make
        # _get_store_ids_for_messages filter out those rows on the next
        # in-process compress(), producing summary nodes whose text
        # covers pre-rotate messages but whose source_ids reference only
        # post-rotate rows. The persisted lifecycle frontier we just
        # advanced is the bootstrap signal for the next process start,
        # where _bind_lifecycle_state will read it into the marker
        # against a freshly-built active context.
        result["applied_frontier_store_id"] = persisted_frontier
        return result

    # -- Lifecycle ---------------------------------------------------------

    def shutdown(self):
        self._unregister_active_engine_binding()
        self._close_storage()
