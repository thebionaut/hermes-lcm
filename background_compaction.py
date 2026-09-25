"""Durable asynchronous preparation and atomic publication for Hermes LCM.

Raw messages remain authoritative.  This module stores only durable job
metadata and derived, inactive summaries until a short publication transaction
successfully compare-and-swaps the conversation frontier.
"""

from __future__ import annotations

import _thread
import hashlib
import inspect
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .config import LCMConfig
from .db_bootstrap import (
    configure_connection,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .host_admission import acquire_background_admission, start_background_thread
from .fresh_tail import resolve_fresh_tail_boundary
from .summary_serialization import filter_summary_messages, serialize_summary_messages
from .tokens import count_message_tokens, count_tokens


logger = logging.getLogger(__name__)

BACKGROUND_COMPACTION_MIGRATION_STEP = "background_compaction_v1"
BACKGROUND_COMPACTION_COVERAGE_MIGRATION_STEP = (
    "background_compaction_source_coverage_v1"
)
BACKGROUND_COMPACTION_NODE_SOURCES_MIGRATION_STEP = (
    "background_compaction_node_sources_v1"
)
BACKGROUND_COMPACTION_NODE_EDGES_MIGRATION_STEP = (
    "background_compaction_node_edges_v1"
)
BACKGROUND_COMPACTION_PROVIDER_OWNER_MIGRATION_STEP = (
    "background_compaction_provider_owner_v1"
)
BACKGROUND_COMPACTION_RANGE_DISPOSITIONS_MIGRATION_STEP = (
    "background_compaction_range_dispositions_v1"
)
BACKGROUND_COMPACTION_TELEMETRY_MIGRATION_STEP = "background_compaction_telemetry_v1"
BACKGROUND_PREPARATION_POLICY_VERSION = "background-preparation-v1"
LCM_PHYSICAL_RECEIPT_SCHEMA_VERSION = "lcm-physical-attempt-v1"
LCM_SUMMARY_PROMPT_VERSION = "lcm-summary-prompt-v1"
LCM_ACTIVE_REPLAY_PROJECTION_VERSION = "disabled-v1"
_ACTIVE_STATES = ("queued", "running", "prepared", "retry")
_FINAL_STATES = ("published", "superseded", "failed")
_MAX_ERROR_CHARS = 512
_MAX_EVENT_ROWS = 1_000
_CAPTURE_ROW_LIMIT = 512
_CAPTURE_GROUP_LOOKAHEAD = 256
_CAPTURE_TAIL_ROW_LIMIT = 255
_CANONICAL_PREFIX_ADOPTION_ROW_LIMIT = 512
_PARTIAL_REPAIR_JOB_KIND = "partial_atomic_repair"
_LOSSLESS_RAW_PROJECTION_JOB_KIND = "lossless_raw_projection"
_PARTIAL_REPAIR_PRIORITY = 1_000_000
_PARTIAL_REPAIR_MAX_GROUP_ROWS = 256
_PARTIAL_REPAIR_MAX_OWNERS = 64
_PARTIAL_REPAIR_MAX_ROOTS = 64
_PARTIAL_REPAIR_MAX_ANCESTRY = 512
_PARTIAL_REPAIR_MAX_TOPOLOGY_EDGES = 32_768
_PARTIAL_REPAIR_BACKFILL_BATCH = 128
_RUNTIME_STOP_TIMEOUT_SECONDS = 5.0
_MAX_GLOBAL_PROVIDER_PERMITS = 2
_EVENT_REASON_CODES = frozenset(
    {
        "background_summary_prepared",
        "prepared_summary_published",
        "degraded_reference_published",
        "prepared_summary_superseded",
        "foreground_emergency_compaction",
        "provider_timeout",
        "lease_expiration",
        "sqlite_lock_contention",
        "worker_unavailable",
        "no_eligible_raw_range",
        "session_reset_fence",
        "canonical_prefix_adopted",
        "partial_atomic_repair_pending",
        "partial_atomic_repair_published",
        "insufficient_compressible_source",
        "non_compressing_output",
        "provider_connection_error",
        "provider_error",
        "provider_auth_error",
        "provider_quota",
        "provider_rate_limit",
        "invalid_provider_response",
        "provider_circuit_open",
        "provider_spend_guard",
        "reasoning_only_output",
        "foreground_deadline_exhausted",
        "stale_lease",
    }
)
_TRANSIENT_RANGE_DISPOSITION_REASONS = frozenset(
    {
        "provider_timeout",
        "provider_error",
        "provider_connection_error",
        "provider_auth_error",
        "provider_quota",
        "provider_rate_limit",
        "invalid_provider_response",
        "provider_circuit_open",
        "provider_spend_guard",
    }
)
_EXTENDABLE_RANGE_DISPOSITION_REASONS = frozenset(
    {
        "non_compressing_output",
        "reasoning_only_output",
        *_TRANSIENT_RANGE_DISPOSITION_REASONS,
    }
)


@dataclass
class PreparedSummaryNode:
    ordinal: int
    summary: str
    source_store_ids: tuple[int, ...]
    source_token_count: int
    summary_token_count: int
    creation_model: str = ""
    creation_provider: str = ""
    route_latency_ms: int = 0
    attempt_number: int = 0
    escalation_level: int = 0
    attribution_available: bool = False
    fallback_used: bool | None = None
    prepared_node_id: int = 0
    prepared_at: float = 0.0


@dataclass
class SourceChunk:
    ordinal: int
    source_store_ids: tuple[int, ...]
    source_token_count: int
    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    serialized_text: str | None = None


@dataclass
class CompactionJob:
    job_id: str
    conversation_id: str
    session_id: str
    session_fence_generation: int
    expected_frontier_store_id: int
    captured_watermark_store_id: int
    source_store_ids: tuple[int, ...]
    source_token_count: int
    source_identity_hash: str
    state: str
    priority: int
    created_at: float
    updated_at: float
    lease_owner: str | None
    lease_expires_at: float | None
    heartbeat_at: float | None
    lease_generation: int
    attempt_count: int
    next_retry_at: float | None
    last_error: str | None
    policy_fingerprint: str
    expected_node_count: int
    prepared_node_count: int
    completion_reason: str | None
    prepared_at: float | None
    published_at: float | None
    job_kind: str = "standard"
    repair_group_store_ids: tuple[int, ...] = field(default_factory=tuple)
    repair_group_identity_hash: str = ""
    repair_covered_store_ids: tuple[int, ...] = field(default_factory=tuple)
    repair_uncovered_store_ids: tuple[int, ...] = field(default_factory=tuple)
    repair_owner_node_ids: tuple[int, ...] = field(default_factory=tuple)
    repair_owner_map: str = "{}"
    repair_capture_dag_epoch: int = 0
    repair_root_node_ids: tuple[int, ...] = field(default_factory=tuple)
    repair_ancestry_node_ids: tuple[int, ...] = field(default_factory=tuple)
    repair_topology_hash: str = ""
    repair_root_identity_hash: str = ""
    repair_bound_dag_epoch: int = 0
    prepared_node_ids: list[int] = field(default_factory=list)

    @property
    def debt_tokens(self) -> int:
        return self.source_token_count


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    job_id: str = ""
    captured_watermark_store_id: int = 0
    node_ids: list[int] = field(default_factory=list)
    messages: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ProviderPermit:
    """One durable provider-call reservation shared by every process."""

    job_id: str
    conversation_id: str
    lease_owner: str
    lease_generation: int
    lease_expires_at: float
    session_fence_generation: int = 0


class ForegroundProviderPermitUnavailable(RuntimeError):
    """Raised when emergency fallback must avoid provider I/O and converge locally."""


# Compatibility name used by early design notes.
PublicationResult = PromotionResult


def _json_ids(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def _load_ids(value: Any) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        return tuple(int(item) for item in decoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _linux_process_start_ticks(pid: int) -> str | None:
    """Return Linux procfs start ticks, which fence PID reuse."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        # Field two is parenthesized and may itself contain spaces or ')'.
        # The tail starts at field three, making starttime (field 22) index 19.
        tail = stat.rsplit(")", 1)[1].split()
        return str(int(tail[19]))
    except (IndexError, TypeError, ValueError):
        return None


def _linux_boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return None
    return value or None


def _current_linux_process_identity() -> tuple[int, str, str] | None:
    """Return a durable same-boot process identity on Linux."""
    pid = os.getpid()
    start_ticks = _linux_process_start_ticks(pid)
    boot_id = _linux_boot_id()
    if not start_ticks or not boot_id:
        return None
    return pid, start_ticks, boot_id


def _linux_process_identity_is_live(
    pid: int,
    start_ticks: str,
    boot_id: str,
) -> bool | None:
    """Check a durable Linux process identity.

    ``None`` means liveness could not be determined. Callers deliberately
    treat that as live so provider concurrency fails closed.
    """
    current_boot_id = _linux_boot_id()
    if current_boot_id is None:
        return None
    if str(boot_id or "") != current_boot_id:
        return False
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (OSError, ValueError):
        return None
    try:
        tail = stat.rsplit(")", 1)[1].split()
        observed_start_ticks = str(int(tail[19]))
    except (IndexError, TypeError, ValueError):
        return None
    return observed_start_ticks == str(start_ticks or "")


def _protected_provider_owner_is_live(row: Any) -> bool:
    """Return whether an expired active-provider permit must remain fenced."""
    if int(row["provider_active"] or 0) != 1:
        return False
    try:
        pid = int(row["owner_pid"] or 0)
    except (TypeError, ValueError):
        return False
    start_ticks = str(row["owner_start_ticks"] or "")
    boot_id = str(row["owner_boot_id"] or "")
    if pid <= 0 or not start_ticks or not boot_id:
        return False
    # An unreadable procfs is an unknown result, not proof of death. Keeping
    # the permit is the safe direction for a global physical-call limit.
    return _linux_process_identity_is_live(pid, start_ticks, boot_id) is not False


def _provider_permit_owned_by_process(
    row: Any,
    identity: tuple[int, str, str] | None,
) -> bool:
    if int(row["provider_active"] or 0) != 1 or identity is None:
        return False
    pid, start_ticks, boot_id = identity
    return (
        int(row["owner_pid"] or 0) == int(pid)
        and str(row["owner_start_ticks"] or "") == str(start_ticks)
        and str(row["owner_boot_id"] or "") == str(boot_id)
    )


_LOCAL_DIAGNOSTIC_EXCEPTION_TYPES = (
    TypeError,
    AttributeError,
    KeyError,
    IndexError,
    NameError,
    ValueError,
)
_LOCAL_ERROR_CREDENTIAL_LABEL = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|authorization|bearer)\b"
    r"(?:\s*[:=]\s*|\s+)?[^\s,;]*"
)
_LOCAL_ERROR_OPAQUE_RUN = re.compile(r"[A-Za-z0-9+/=]{20,}")
_LOCAL_ERROR_EXCERPT_CHARS = 200


def _local_error_excerpt(error: BaseException) -> str:
    text = " ".join(str(error).split())
    text = _LOCAL_ERROR_CREDENTIAL_LABEL.sub("<redacted>", text)
    text = _LOCAL_ERROR_OPAQUE_RUN.sub("<redacted>", text)
    return text[:_LOCAL_ERROR_EXCERPT_CHARS]


def _sanitize_error(error: BaseException | str | None) -> str:
    if isinstance(error, BaseException):
        type_name = re.sub(
            r"[^A-Za-z0-9_.-]",
            "",
            type(error).__name__,
        )[:64] or "ProviderError"
        if isinstance(error, TimeoutError):
            category = "timeout"
        elif isinstance(error, ConnectionError):
            category = "connection_error"
        elif isinstance(error, sqlite3.OperationalError) and any(
            marker in str(error).lower() for marker in ("locked", "busy")
        ):
            category = "sqlite_busy"
        else:
            category = "provider_error"
        status = getattr(error, "status_code", None)
        if status is None:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
        status_hint = ""
        if isinstance(status, int) and 100 <= status <= 999:
            status_hint = f":http_{status}"
        # Keep the established Type:category prefix stable. Only local defect
        # classes receive a bounded, credential-scrubbed diagnostic excerpt;
        # provider-origin exception bodies retain the historical short form.
        prefix = f"{type_name}:{category}{status_hint}"
        if isinstance(error, _LOCAL_DIAGNOSTIC_EXCEPTION_TYPES):
            excerpt = _local_error_excerpt(error)
            if excerpt:
                return f"{prefix}:{excerpt}"[:_MAX_ERROR_CHARS]
        return prefix[:_MAX_ERROR_CHARS]
    text = str(error or "")
    text = re.sub(
        r"(?i)\b(api[-_ ]?key|access[-_ ]?token|authorization|bearer|password|secret|token)\b\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)bearer\s+[a-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)\b(?:sk-ant-|sk-|gh[pousr]_|xox[baprs]-)[a-z0-9._=-]{8,}",
        "[REDACTED CREDENTIAL]",
        text,
    )
    text = " ".join(text.split())
    return text[:_MAX_ERROR_CHARS]


def _owner_hint(owner: str | None) -> str | None:
    if not owner:
        return None
    return hashlib.sha256(owner.encode("utf-8", "replace")).hexdigest()[:12]


def background_policy_fingerprint(config: LCMConfig | Any) -> str:
    """Return a stable hash of every setting that changes prepared output."""
    names = (
        "leaf_chunk_tokens",
        "dynamic_leaf_chunk_enabled",
        "dynamic_leaf_chunk_max",
        "fresh_tail_count",
        "fresh_tail_max_tokens",
        "background_compaction_max_passes",
        "background_compaction_min_provider_source_tokens",
        "cache_friendly_condensation_enabled",
        "summary_model",
        "summary_fallback_models",
        "summary_timeout_ms",
        "l2_budget_ratio",
        "l3_truncate_tokens",
        "custom_instructions",
        "ignore_message_patterns",
        "ignore_message_patterns_source",
        "sensitive_patterns_enabled",
        "sensitive_patterns",
        "large_output_externalization_enabled",
        "large_output_externalization_threshold_chars",
        "large_output_active_replay_stubbing_enabled",
        "large_output_active_replay_stub_threshold_tokens",
        "large_output_transcript_gc_enabled",
        "background_compaction_backlog_tokens",
        "background_compaction_prepare_ratio",
    )
    payload = {name: getattr(config, name, None) for name in names}
    payload["preparation_policy_version"] = BACKGROUND_PREPARATION_POLICY_VERSION
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_SOURCE_COVERAGE_TRIGGER_NAMES = (
    "lcm_compaction_summary_coverage_insert",
    "lcm_compaction_summary_coverage_delete",
    "lcm_compaction_summary_coverage_update_before",
    "lcm_compaction_summary_coverage_update_after",
)


def _migration_step_is_complete(conn: sqlite3.Connection, step_name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name=?",
        (step_name,),
    ).fetchone() is not None


def _indexed_coverage_triggers_are_installed(conn: sqlite3.Connection) -> bool:
    installed = {
        str(row[0]): str(row[1] or "")
        for row in conn.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type='trigger' AND name IN (?,?,?,?)
            """,
            _SOURCE_COVERAGE_TRIGGER_NAMES,
        ).fetchall()
    }
    if set(installed) != set(_SOURCE_COVERAGE_TRIGGER_NAMES):
        return False
    delete_sql = installed["lcm_compaction_summary_coverage_delete"]
    update_before_sql = installed[
        "lcm_compaction_summary_coverage_update_before"
    ]
    insert_sql = installed["lcm_compaction_summary_coverage_insert"]
    update_after_sql = installed[
        "lcm_compaction_summary_coverage_update_after"
    ]
    return (
        all(
            "lcm_compaction_node_sources AS survivor" in sql
            and "FROM summary_nodes AS candidate" not in sql
            and "json_each" not in sql
            for sql in (delete_sql, update_before_sql)
        )
        and all(
            "INSERT OR IGNORE INTO lcm_compaction_node_sources" in sql
            for sql in (insert_sql, update_after_sql)
        )
    )


def _replace_source_coverage_triggers(
    conn: sqlite3.Connection,
    *,
    indexed_rehome: bool,
) -> None:
    """Install migration-safe or final indexed node-coverage triggers."""
    if indexed_rehome and _indexed_coverage_triggers_are_installed(conn):
        return
    for trigger_name in _SOURCE_COVERAGE_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")

    conn.execute(
        """
        CREATE TRIGGER lcm_compaction_summary_coverage_insert
        AFTER INSERT ON summary_nodes
        WHEN NEW.source_type='messages'
        BEGIN
            INSERT OR IGNORE INTO lcm_compaction_node_sources(
                node_id, session_id, store_id
            )
            SELECT NEW.node_id, NEW.session_id, CAST(value AS INTEGER)
            FROM json_each(
                CASE WHEN json_valid(NEW.source_ids)
                     THEN NEW.source_ids ELSE '[]' END
            );

            INSERT INTO lcm_compaction_source_coverage(
                session_id, store_id, publication_id,
                publication_kind, published_at
            )
            SELECT NEW.session_id, CAST(value AS INTEGER),
                   'summary-node:' || NEW.node_id, 'canonical', NEW.created_at
            FROM json_each(
                CASE WHEN json_valid(NEW.source_ids)
                     THEN NEW.source_ids ELSE '[]' END
            );
        END
        """
    )

    if indexed_rehome:
        rehome_sql = """
            DELETE FROM lcm_compaction_node_sources
            WHERE node_id=OLD.node_id;

            UPDATE lcm_compaction_source_coverage
            SET publication_id=COALESCE(
                    (
                        SELECT 'summary-node:' || survivor.node_id
                        FROM lcm_compaction_node_sources AS survivor
                        WHERE survivor.session_id=
                                  lcm_compaction_source_coverage.session_id
                          AND survivor.store_id=
                                  lcm_compaction_source_coverage.store_id
                        ORDER BY survivor.node_id LIMIT 1
                    ),
                    publication_id
                ),
                publication_kind='canonical'
            WHERE publication_id IN (
                'summary-node:' || OLD.node_id,
                'legacy-node:' || OLD.node_id
            );

            DELETE FROM lcm_compaction_source_coverage
            WHERE publication_id IN (
                'summary-node:' || OLD.node_id,
                'legacy-node:' || OLD.node_id
            );
        """
    else:
        # This definition exists only while the normalized map is being
        # backfilled.  It preserves old coverage across concurrent lifecycle
        # mutations; the final marker transaction replaces it atomically.
        rehome_sql = """
            DELETE FROM lcm_compaction_node_sources
            WHERE node_id=OLD.node_id;

            DELETE FROM lcm_compaction_source_coverage
            WHERE publication_id IN (
                'summary-node:' || OLD.node_id,
                'legacy-node:' || OLD.node_id
            );

            INSERT OR IGNORE INTO lcm_compaction_source_coverage(
                session_id, store_id, publication_id,
                publication_kind, published_at
            )
            SELECT candidate.session_id, CAST(source.value AS INTEGER),
                   'summary-node:' || candidate.node_id,
                   'canonical', candidate.created_at
            FROM summary_nodes AS candidate,
                 json_each(
                     CASE WHEN json_valid(candidate.source_ids)
                          THEN candidate.source_ids ELSE '[]' END
                 ) AS source
            WHERE candidate.node_id<>OLD.node_id
              AND candidate.source_type='messages'
              AND candidate.session_id=OLD.session_id
              AND CAST(source.value AS INTEGER) IN (
                  SELECT CAST(value AS INTEGER)
                  FROM json_each(
                      CASE WHEN json_valid(OLD.source_ids)
                           THEN OLD.source_ids ELSE '[]' END
                  )
              )
            ORDER BY candidate.node_id, CAST(source.value AS INTEGER);
        """

    conn.execute(
        f"""
        CREATE TRIGGER lcm_compaction_summary_coverage_delete
        AFTER DELETE ON summary_nodes
        WHEN OLD.source_type='messages'
        BEGIN
            {rehome_sql}
        END
        """
    )
    conn.execute(
        f"""
        CREATE TRIGGER lcm_compaction_summary_coverage_update_before
        BEFORE UPDATE OF session_id, source_ids, source_type ON summary_nodes
        WHEN OLD.source_type='messages'
        BEGIN
            {rehome_sql}
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER lcm_compaction_summary_coverage_update_after
        AFTER UPDATE OF session_id, source_ids, source_type ON summary_nodes
        WHEN NEW.source_type='messages'
        BEGIN
            INSERT OR IGNORE INTO lcm_compaction_node_sources(
                node_id, session_id, store_id
            )
            SELECT NEW.node_id, NEW.session_id, CAST(value AS INTEGER)
            FROM json_each(
                CASE WHEN json_valid(NEW.source_ids)
                     THEN NEW.source_ids ELSE '[]' END
            );

            -- Retained raw lineage may be moving with a whole legacy
            -- duplicate group during session carry-over.  Preserve one
            -- elected owner without rejecting that pre-existing overlap.
            INSERT OR IGNORE INTO lcm_compaction_source_coverage(
                session_id, store_id, publication_id,
                publication_kind, published_at
            )
            SELECT NEW.session_id, CAST(source.value AS INTEGER),
                   'summary-node:' || NEW.node_id,
                   'canonical', NEW.created_at
            FROM json_each(
                CASE WHEN json_valid(NEW.source_ids)
                     THEN NEW.source_ids ELSE '[]' END
            ) AS source
            WHERE OLD.source_type='messages'
              AND CAST(source.value AS INTEGER) IN (
                  SELECT CAST(value AS INTEGER)
                  FROM json_each(
                      CASE WHEN json_valid(OLD.source_ids)
                           THEN OLD.source_ids ELSE '[]' END
                  )
              );

            -- A newly added raw source is not legacy overlap.  Use a strict
            -- insert so UPDATE cannot bypass the overlap fence.
            INSERT INTO lcm_compaction_source_coverage(
                session_id, store_id, publication_id,
                publication_kind, published_at
            )
            SELECT NEW.session_id, CAST(source.value AS INTEGER),
                   'summary-node:' || NEW.node_id,
                   'canonical', NEW.created_at
            FROM json_each(
                CASE WHEN json_valid(NEW.source_ids)
                     THEN NEW.source_ids ELSE '[]' END
            ) AS source
            WHERE OLD.source_type<>'messages'
               OR CAST(source.value AS INTEGER) NOT IN (
                   SELECT CAST(value AS INTEGER)
                   FROM json_each(
                       CASE WHEN json_valid(OLD.source_ids)
                            THEN OLD.source_ids ELSE '[]' END
                   )
               );
        END
        """
    )


def _ensure_source_coverage_backfill(conn: sqlite3.Connection) -> None:
    """Backfill canonical ownership and its indexed node-source map."""
    if conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='summary_nodes'
        """
    ).fetchone() is None:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        coverage_complete = _migration_step_is_complete(
            conn,
            BACKGROUND_COMPACTION_COVERAGE_MIGRATION_STEP,
        )
        node_sources_complete = _migration_step_is_complete(
            conn,
            BACKGROUND_COMPACTION_NODE_SOURCES_MIGRATION_STEP,
        )
        _replace_source_coverage_triggers(
            conn,
            indexed_rehome=coverage_complete and node_sources_complete,
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    if coverage_complete and node_sources_complete:
        return

    cursor = 0
    while True:
        if (
            _migration_step_is_complete(
                conn,
                BACKGROUND_COMPACTION_COVERAGE_MIGRATION_STEP,
            )
            and _migration_step_is_complete(
                conn,
                BACKGROUND_COMPACTION_NODE_SOURCES_MIGRATION_STEP,
            )
        ):
            return
        legacy_node = conn.execute(
            """
            SELECT node_id, session_id, source_ids, created_at
            FROM summary_nodes
            WHERE source_type='messages' AND node_id>?
            ORDER BY node_id LIMIT 1
            """,
            (cursor,),
        ).fetchone()
        if legacy_node is None:
            conn.execute("BEGIN IMMEDIATE")
            try:
                newer = conn.execute(
                    """
                    SELECT 1 FROM summary_nodes
                    WHERE source_type='messages' AND node_id>?
                    LIMIT 1
                    """,
                    (cursor,),
                ).fetchone()
                if newer is None:
                    mark_migration_step_complete(
                        conn,
                        BACKGROUND_COMPACTION_COVERAGE_MIGRATION_STEP,
                    )
                    mark_migration_step_complete(
                        conn,
                        BACKGROUND_COMPACTION_NODE_SOURCES_MIGRATION_STEP,
                    )
                    _replace_source_coverage_triggers(
                        conn,
                        indexed_rehome=True,
                    )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            if newer is None:
                return
            continue

        node_id = int(legacy_node[0])
        source_ids = _load_ids(legacy_node[2])
        publication_id = f"legacy-node:{node_id}"
        for offset in range(0, len(source_ids), 500):
            batch = source_ids[offset:offset + 500]
            conn.execute("BEGIN IMMEDIATE")
            try:
                current_node = conn.execute(
                    """
                    SELECT session_id, source_ids, source_type, created_at
                    FROM summary_nodes WHERE node_id=?
                    """,
                    (node_id,),
                ).fetchone()
                metadata_matches = (
                    current_node is not None
                    and str(current_node[0]) == str(legacy_node[1])
                    and str(current_node[1] or "") == str(legacy_node[2] or "")
                    and str(current_node[2]) == "messages"
                )
                if not metadata_matches:
                    conn.execute("COMMIT")
                    break
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO lcm_compaction_node_sources(
                        node_id, session_id, store_id
                    ) VALUES(?,?,?)
                    """,
                    (
                        (node_id, str(current_node[0]), int(store_id))
                        for store_id in batch
                    ),
                )
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO lcm_compaction_source_coverage(
                        session_id, store_id, publication_id,
                        publication_kind, published_at
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        (
                            str(current_node[0]),
                            int(store_id),
                            publication_id,
                            "legacy",
                            float(current_node[3] or 0.0),
                        )
                        for store_id in batch
                    ),
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        cursor = node_id


_NODE_EDGE_TRIGGER_NAMES = (
    "lcm_compaction_dag_edge_insert",
    "lcm_compaction_dag_edge_delete",
    "lcm_compaction_dag_edge_update_before",
    "lcm_compaction_dag_edge_update_after",
    "lcm_compaction_dag_edge_session_update",
    "lcm_compaction_dag_epoch_update",
)


def _node_edge_epoch_triggers_are_installed(conn: sqlite3.Connection) -> bool:
    installed = {
        str(row[0]): str(row[1] or "")
        for row in conn.execute(
            f"""
            SELECT name, sql FROM sqlite_master
            WHERE type='trigger' AND name IN (
                {','.join('?' for _ in _NODE_EDGE_TRIGGER_NAMES)}
            )
            """,
            _NODE_EDGE_TRIGGER_NAMES,
        ).fetchall()
    }
    if set(installed) != set(_NODE_EDGE_TRIGGER_NAMES):
        return False
    combined = "\n".join(installed.values())
    return (
        "INSERT OR IGNORE INTO lcm_compaction_node_edges" in combined
        and "UPDATE lcm_compaction_node_edges" in combined
        and "FROM summary_nodes AS parent" not in combined
        and "lcm_compaction_dag_epochs" in combined
    )


def _replace_node_edge_epoch_triggers(conn: sqlite3.Connection) -> None:
    """Install exact normalized DAG edges and per-session mutation epochs."""
    if _node_edge_epoch_triggers_are_installed(conn):
        return
    for trigger_name in _NODE_EDGE_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")

    outgoing_new = """
        INSERT OR IGNORE INTO lcm_compaction_node_edges(
            parent_node_id, child_node_id, session_id
        )
        SELECT NEW.node_id, CAST(source.value AS INTEGER), NEW.session_id
        FROM json_each(
                 CASE WHEN json_valid(NEW.source_ids)
                      THEN NEW.source_ids ELSE '[]' END
             ) AS source
        WHERE NEW.source_type='nodes';
    """
    conn.execute(
        f"""
        CREATE TRIGGER lcm_compaction_dag_edge_insert
        AFTER INSERT ON summary_nodes
        BEGIN
            INSERT INTO lcm_compaction_dag_epochs(session_id, mutation_epoch)
            VALUES(NEW.session_id, 1)
            ON CONFLICT(session_id) DO UPDATE SET
                mutation_epoch=mutation_epoch+1;
            {outgoing_new}
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER lcm_compaction_dag_edge_delete
        AFTER DELETE ON summary_nodes
        BEGIN
            DELETE FROM lcm_compaction_node_edges
            WHERE parent_node_id=OLD.node_id OR child_node_id=OLD.node_id;
            INSERT INTO lcm_compaction_dag_epochs(session_id, mutation_epoch)
            VALUES(OLD.session_id, 1)
            ON CONFLICT(session_id) DO UPDATE SET
                mutation_epoch=mutation_epoch+1;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER lcm_compaction_dag_edge_update_before
        BEFORE UPDATE OF source_ids, source_type
        ON summary_nodes
        BEGIN
            DELETE FROM lcm_compaction_node_edges
            WHERE parent_node_id=OLD.node_id;
        END
        """
    )
    conn.execute(
        f"""
        CREATE TRIGGER lcm_compaction_dag_edge_update_after
        AFTER UPDATE OF source_ids, source_type
        ON summary_nodes
        BEGIN
            {outgoing_new}
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER lcm_compaction_dag_edge_session_update
        AFTER UPDATE OF session_id ON summary_nodes
        WHEN OLD.session_id<>NEW.session_id
        BEGIN
            -- session_id on an edge is the parent's session.  A child moved
            -- alone is ignored by traversal's live-child session join.
            UPDATE lcm_compaction_node_edges
            SET session_id=NEW.session_id
            WHERE parent_node_id=NEW.node_id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER lcm_compaction_dag_epoch_update
        AFTER UPDATE OF session_id, depth, summary, token_count,
                        source_token_count, source_ids, source_type,
                        created_at, earliest_at, latest_at, expand_hint
        ON summary_nodes
        BEGIN
            INSERT INTO lcm_compaction_dag_epochs(session_id, mutation_epoch)
            VALUES(NEW.session_id, 1)
            ON CONFLICT(session_id) DO UPDATE SET
                mutation_epoch=mutation_epoch+1;
            INSERT INTO lcm_compaction_dag_epochs(session_id, mutation_epoch)
            SELECT OLD.session_id, 1
            WHERE OLD.session_id<>NEW.session_id
            ON CONFLICT(session_id) DO UPDATE SET
                mutation_epoch=mutation_epoch+1;
        END
        """
    )


def _ensure_node_edge_backfill(conn: sqlite3.Connection) -> None:
    """Backfill normalized DAG edges with triggers live throughout migration."""
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='summary_nodes'"
    ).fetchone() is None:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        _replace_node_edge_epoch_triggers(conn)
        complete = _migration_step_is_complete(
            conn,
            BACKGROUND_COMPACTION_NODE_EDGES_MIGRATION_STEP,
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    if complete:
        return

    cursor = 0
    while True:
        if _migration_step_is_complete(
            conn,
            BACKGROUND_COMPACTION_NODE_EDGES_MIGRATION_STEP,
        ):
            return
        rows = conn.execute(
            """
            SELECT node_id FROM summary_nodes
            WHERE node_id>? ORDER BY node_id LIMIT ?
            """,
            (cursor, _PARTIAL_REPAIR_BACKFILL_BATCH),
        ).fetchall()
        if not rows:
            conn.execute("BEGIN IMMEDIATE")
            try:
                newer = conn.execute(
                    "SELECT 1 FROM summary_nodes WHERE node_id>? LIMIT 1",
                    (cursor,),
                ).fetchone()
                if newer is None:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO lcm_compaction_dag_epochs(
                            session_id, mutation_epoch
                        )
                        SELECT DISTINCT session_id, 1 FROM summary_nodes
                        """
                    )
                    mark_migration_step_complete(
                        conn,
                        BACKGROUND_COMPACTION_NODE_EDGES_MIGRATION_STEP,
                    )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            if newer is None:
                return
            continue

        node_ids = tuple(int(row[0]) for row in rows)
        conn.execute("BEGIN IMMEDIATE")
        try:
            for node_id in node_ids:
                current = conn.execute(
                    """
                    SELECT node_id, session_id, source_type, source_ids
                    FROM summary_nodes WHERE node_id=?
                    """,
                    (node_id,),
                ).fetchone()
                conn.execute(
                    "DELETE FROM lcm_compaction_node_edges WHERE parent_node_id=?",
                    (node_id,),
                )
                if current is None:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO lcm_compaction_dag_epochs(
                        session_id, mutation_epoch
                    ) VALUES(?,1)
                    """,
                    (str(current["session_id"] or ""),),
                )
                if str(current["source_type"] or "") == "nodes":
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO lcm_compaction_node_edges(
                            parent_node_id, child_node_id, session_id
                        )
                        SELECT ?, CAST(source.value AS INTEGER), ?
                        FROM json_each(
                                 CASE WHEN json_valid(?) THEN ? ELSE '[]' END
                             ) AS source
                        """,
                        (
                            node_id,
                            str(current["session_id"] or ""),
                            current["source_ids"],
                            current["source_ids"],
                        ),
                    )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        cursor = node_ids[-1]


def _ensure_background_schema(conn: sqlite3.Connection) -> None:
    """Create the optional schema and marker in one process-safe transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_jobs (
                job_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                session_fence_generation INTEGER NOT NULL DEFAULT 0,
                expected_frontier_store_id INTEGER NOT NULL DEFAULT 0,
                captured_watermark_store_id INTEGER NOT NULL DEFAULT 0,
                source_store_ids TEXT NOT NULL DEFAULT '[]',
                source_token_count INTEGER NOT NULL DEFAULT 0,
                source_identity_hash TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                lease_owner TEXT,
                lease_expires_at REAL,
                heartbeat_at REAL,
                lease_generation INTEGER NOT NULL DEFAULT 0,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_retry_at REAL,
                last_error TEXT,
                policy_fingerprint TEXT NOT NULL,
                expected_node_count INTEGER NOT NULL DEFAULT 0,
                prepared_node_count INTEGER NOT NULL DEFAULT 0,
                completion_reason TEXT,
                prepared_at REAL,
                published_at REAL,
                job_kind TEXT NOT NULL DEFAULT 'standard',
                repair_group_store_ids TEXT NOT NULL DEFAULT '[]',
                repair_group_identity_hash TEXT NOT NULL DEFAULT '',
                repair_covered_store_ids TEXT NOT NULL DEFAULT '[]',
                repair_uncovered_store_ids TEXT NOT NULL DEFAULT '[]',
                repair_owner_node_ids TEXT NOT NULL DEFAULT '[]',
                repair_owner_map TEXT NOT NULL DEFAULT '{}',
                repair_capture_dag_epoch INTEGER NOT NULL DEFAULT 0,
                repair_root_node_ids TEXT NOT NULL DEFAULT '[]',
                repair_ancestry_node_ids TEXT NOT NULL DEFAULT '[]',
                repair_topology_hash TEXT NOT NULL DEFAULT '',
                repair_root_identity_hash TEXT NOT NULL DEFAULT '',
                repair_bound_dag_epoch INTEGER NOT NULL DEFAULT 0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_session_fences (
                conversation_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                lifecycle_bound_at REAL NOT NULL DEFAULT 0,
                lifecycle_reset_at REAL NOT NULL DEFAULT 0,
                generation INTEGER NOT NULL DEFAULT 1,
                publishable INTEGER NOT NULL DEFAULT 1,
                updated_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_provider_leases (
                job_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                lease_generation INTEGER NOT NULL,
                lease_owner TEXT NOT NULL,
                started_at REAL NOT NULL,
                heartbeat_at REAL NOT NULL,
                lease_expires_at REAL NOT NULL,
                timed_out_at REAL,
                provider_active INTEGER NOT NULL DEFAULT 0,
                owner_pid INTEGER,
                owner_start_ticks TEXT,
                owner_boot_id TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                job_id TEXT NOT NULL DEFAULT '',
                reason_code TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_prepared_nodes (
                prepared_node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                summary TEXT NOT NULL,
                source_store_ids TEXT NOT NULL,
                source_token_count INTEGER NOT NULL DEFAULT 0,
                summary_token_count INTEGER NOT NULL DEFAULT 0,
                creation_model TEXT NOT NULL DEFAULT '',
                creation_provider TEXT NOT NULL DEFAULT '',
                route_latency_ms INTEGER NOT NULL DEFAULT 0,
                attempt_number INTEGER NOT NULL DEFAULT 0,
                escalation_level INTEGER NOT NULL DEFAULT 0,
                attribution_available INTEGER NOT NULL DEFAULT 0,
                fallback_used INTEGER,
                prepared_at REAL NOT NULL,
                UNIQUE(job_id, ordinal),
                FOREIGN KEY(job_id) REFERENCES lcm_compaction_jobs(job_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_range_dispositions (
                conversation_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                session_fence_generation INTEGER NOT NULL,
                expected_frontier_store_id INTEGER NOT NULL,
                captured_watermark_store_id INTEGER NOT NULL,
                policy_fingerprint TEXT NOT NULL,
                source_identity_hash TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                source_token_count INTEGER NOT NULL DEFAULT 0,
                prepared_job_id TEXT NOT NULL DEFAULT '',
                capture_started_at REAL NOT NULL DEFAULT 0,
                first_seen_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                resolved_at REAL,
                PRIMARY KEY(
                    conversation_id, session_id, session_fence_generation,
                    expected_frontier_store_id, captured_watermark_store_id,
                    policy_fingerprint, source_identity_hash
                )
            ) WITHOUT ROWID
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                job_attempt INTEGER NOT NULL,
                chunk_ordinal INTEGER NOT NULL,
                escalation_level INTEGER NOT NULL,
                logical_route_ordinal INTEGER NOT NULL DEFAULT 0,
                actual_provider TEXT NOT NULL DEFAULT '',
                actual_model TEXT NOT NULL DEFAULT '',
                attribution_available INTEGER NOT NULL DEFAULT 0,
                fallback_used INTEGER,
                outcome TEXT NOT NULL,
                failure_reason TEXT NOT NULL DEFAULT '',
                source_token_count INTEGER NOT NULL DEFAULT 0,
                summary_token_count INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                started_at REAL NOT NULL,
                finished_at REAL NOT NULL,
                UNIQUE(
                    job_id, job_attempt, chunk_ordinal,
                    escalation_level, logical_route_ordinal
                ),
                FOREIGN KEY(job_id) REFERENCES lcm_compaction_jobs(job_id)
                    ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_attempt_receipts (
                physical_attempt_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                job_attempt INTEGER NOT NULL,
                chunk_ordinal INTEGER NOT NULL,
                escalation_level INTEGER NOT NULL DEFAULT 0,
                logical_route_ordinal INTEGER NOT NULL DEFAULT 0,
                send_state TEXT NOT NULL,
                transport_outcome TEXT NOT NULL DEFAULT 'started',
                transport_reason TEXT NOT NULL DEFAULT '',
                summary_outcome TEXT NOT NULL DEFAULT 'pending',
                summary_reason TEXT NOT NULL DEFAULT '',
                requested_provider TEXT NOT NULL DEFAULT '',
                requested_model TEXT NOT NULL DEFAULT '',
                served_provider TEXT NOT NULL DEFAULT '',
                served_model TEXT NOT NULL DEFAULT '',
                effective_timeout_ms INTEGER,
                timeout_source TEXT NOT NULL DEFAULT '',
                source_version TEXT NOT NULL,
                config_version TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                projection_version TEXT NOT NULL,
                provider_input_total INTEGER,
                provider_output_total INTEGER,
                cache_read_input INTEGER,
                cache_write_input INTEGER,
                reasoning_output INTEGER,
                normalized_input INTEGER,
                normalized_output INTEGER,
                usage_convention TEXT NOT NULL DEFAULT 'unknown',
                usage_normalization_version TEXT NOT NULL DEFAULT '',
                usage_coverage TEXT NOT NULL DEFAULT 'unavailable',
                source_input_estimate INTEGER,
                serialized_input_estimate INTEGER,
                framed_input_estimate INTEGER,
                estimate_method TEXT NOT NULL DEFAULT '',
                finish_reason TEXT NOT NULL DEFAULT '',
                latency_ms INTEGER NOT NULL DEFAULT 0,
                started_at REAL NOT NULL,
                finished_at REAL
            ) WITHOUT ROWID
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_publications (
                job_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                publication_fence_generation INTEGER NOT NULL,
                range_identity TEXT NOT NULL,
                observed_active_tokens INTEGER,
                effective_context_window INTEGER,
                publication_trigger_tokens INTEGER,
                emergency_trigger_tokens INTEGER,
                trigger_policy TEXT NOT NULL DEFAULT '',
                prepared_to_published_ms INTEGER,
                source_token_count INTEGER NOT NULL DEFAULT 0,
                summary_token_count INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                published_at REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES lcm_compaction_jobs(job_id)
                    ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_source_coverage (
                session_id TEXT NOT NULL,
                store_id INTEGER NOT NULL,
                publication_id TEXT NOT NULL,
                publication_kind TEXT NOT NULL,
                published_at REAL NOT NULL,
                PRIMARY KEY(session_id, store_id)
            ) WITHOUT ROWID
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_node_sources (
                node_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                store_id INTEGER NOT NULL,
                PRIMARY KEY(node_id, store_id)
            ) WITHOUT ROWID
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_node_edges (
                parent_node_id INTEGER NOT NULL,
                child_node_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                PRIMARY KEY(parent_node_id, child_node_id)
            ) WITHOUT ROWID
            """,
            """
            CREATE TABLE IF NOT EXISTS lcm_compaction_dag_epochs (
                session_id TEXT PRIMARY KEY,
                mutation_epoch INTEGER NOT NULL DEFAULT 0
            ) WITHOUT ROWID
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_lcm_compaction_one_active_conversation
                ON lcm_compaction_jobs(conversation_id)
                WHERE state IN ('queued', 'running', 'prepared', 'retry')
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_claim
                ON lcm_compaction_jobs(state, next_retry_at, priority DESC, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_lease
                ON lcm_compaction_jobs(state, lease_expires_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_prepared_job
                ON lcm_compaction_prepared_nodes(job_id, ordinal)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_provider_lease_expiry
                ON lcm_compaction_provider_leases(lease_expires_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_provider_lease_conversation
                ON lcm_compaction_provider_leases(conversation_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_events_scope
                ON lcm_compaction_events(conversation_id, session_id, event_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_events_reason
                ON lcm_compaction_events(reason_code, event_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_coverage_publication
                ON lcm_compaction_source_coverage(publication_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_node_sources_lookup
                ON lcm_compaction_node_sources(session_id, store_id, node_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_node_edges_reverse
                ON lcm_compaction_node_edges(session_id, child_node_id, parent_node_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_node_edges_forward
                ON lcm_compaction_node_edges(session_id, parent_node_id, child_node_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_active_repair
                ON lcm_compaction_jobs(
                    conversation_id, session_id, expected_frontier_store_id,
                    job_kind, state
                )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_dispositions_scope
                ON lcm_compaction_range_dispositions(
                    conversation_id, session_id, last_seen_at
                )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_attempts_outcome_time
                ON lcm_compaction_attempts(finished_at, outcome)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_compaction_attempts_route_time
                ON lcm_compaction_attempts(
                    actual_provider, actual_model, finished_at
                )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_attempt_receipts_job
                ON lcm_compaction_attempt_receipts(
                    job_id, job_attempt, chunk_ordinal, escalation_level,
                    logical_route_ordinal
                )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lcm_attempt_receipts_started
                ON lcm_compaction_attempt_receipts(started_at, send_state)
            """,
        )
        conn.execute(statements[0])
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(lcm_compaction_jobs)").fetchall()
        }
        additive_columns = {
            "source_identity_hash": "TEXT NOT NULL DEFAULT ''",
            "session_fence_generation": "INTEGER NOT NULL DEFAULT 0",
            "job_kind": "TEXT NOT NULL DEFAULT 'standard'",
            "repair_group_store_ids": "TEXT NOT NULL DEFAULT '[]'",
            "repair_group_identity_hash": "TEXT NOT NULL DEFAULT ''",
            "repair_covered_store_ids": "TEXT NOT NULL DEFAULT '[]'",
            "repair_uncovered_store_ids": "TEXT NOT NULL DEFAULT '[]'",
            "repair_owner_node_ids": "TEXT NOT NULL DEFAULT '[]'",
            "repair_owner_map": "TEXT NOT NULL DEFAULT '{}'",
            "repair_capture_dag_epoch": "INTEGER NOT NULL DEFAULT 0",
            "repair_root_node_ids": "TEXT NOT NULL DEFAULT '[]'",
            "repair_ancestry_node_ids": "TEXT NOT NULL DEFAULT '[]'",
            "repair_topology_hash": "TEXT NOT NULL DEFAULT ''",
            "repair_root_identity_hash": "TEXT NOT NULL DEFAULT ''",
            "repair_bound_dag_epoch": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, declaration in additive_columns.items():
            if column not in columns:
                conn.execute(
                    f"ALTER TABLE lcm_compaction_jobs ADD COLUMN {column} {declaration}"
                )
        for statement in statements[1:]:
            conn.execute(statement)
        disposition_columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(lcm_compaction_range_dispositions)"
            ).fetchall()
        }
        if "capture_started_at" not in disposition_columns:
            conn.execute(
                "ALTER TABLE lcm_compaction_range_dispositions "
                "ADD COLUMN capture_started_at REAL NOT NULL DEFAULT 0"
            )
            # Legacy rows did not persist the capture/failure interval separately.
            # Recover the latest matching job's durable creation time when possible;
            # falling back to first_seen_at retains prior conservative behavior.
            conn.execute(
                """
                UPDATE lcm_compaction_range_dispositions
                SET capture_started_at=COALESCE(
                    (
                        SELECT MAX(job.created_at)
                        FROM lcm_compaction_jobs AS job
                        WHERE job.conversation_id=
                                  lcm_compaction_range_dispositions.conversation_id
                          AND job.session_id=
                                  lcm_compaction_range_dispositions.session_id
                          AND job.session_fence_generation=
                                  lcm_compaction_range_dispositions.session_fence_generation
                          AND job.expected_frontier_store_id=
                                  lcm_compaction_range_dispositions.expected_frontier_store_id
                          AND job.captured_watermark_store_id=
                                  lcm_compaction_range_dispositions.captured_watermark_store_id
                          AND job.policy_fingerprint=
                                  lcm_compaction_range_dispositions.policy_fingerprint
                          AND job.source_identity_hash=
                                  lcm_compaction_range_dispositions.source_identity_hash
                          AND job.created_at<=
                                  lcm_compaction_range_dispositions.first_seen_at
                    ),
                    first_seen_at
                )
                WHERE capture_started_at<=0
                """
            )
        prepared_columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(lcm_compaction_prepared_nodes)"
            ).fetchall()
        }
        prepared_additive_columns = {
            "route_latency_ms": "INTEGER NOT NULL DEFAULT 0",
            "attempt_number": "INTEGER NOT NULL DEFAULT 0",
            "escalation_level": "INTEGER NOT NULL DEFAULT 0",
            "attribution_available": "INTEGER NOT NULL DEFAULT 0",
            "fallback_used": "INTEGER",
        }
        for column, declaration in prepared_additive_columns.items():
            if column not in prepared_columns:
                conn.execute(
                    "ALTER TABLE lcm_compaction_prepared_nodes "
                    f"ADD COLUMN {column} {declaration}"
                )
        provider_columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(lcm_compaction_provider_leases)"
            ).fetchall()
        }
        provider_additive_columns = {
            "provider_active": "INTEGER NOT NULL DEFAULT 0",
            "owner_pid": "INTEGER",
            "owner_start_ticks": "TEXT",
            "owner_boot_id": "TEXT",
        }
        for column, declaration in provider_additive_columns.items():
            if column not in provider_columns:
                conn.execute(
                    "ALTER TABLE lcm_compaction_provider_leases "
                    f"ADD COLUMN {column} {declaration}"
                )
        fence_columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(lcm_compaction_session_fences)"
            ).fetchall()
        }
        if "lifecycle_bound_at" not in fence_columns:
            conn.execute(
                "ALTER TABLE lcm_compaction_session_fences "
                "ADD COLUMN lifecycle_bound_at REAL NOT NULL DEFAULT 0"
            )
        if "lifecycle_reset_at" not in fence_columns:
            conn.execute(
                "ALTER TABLE lcm_compaction_session_fences "
                "ADD COLUMN lifecycle_reset_at REAL NOT NULL DEFAULT 0"
            )
        mark_migration_step_complete(conn, BACKGROUND_COMPACTION_MIGRATION_STEP)
        mark_migration_step_complete(
            conn,
            BACKGROUND_COMPACTION_PROVIDER_OWNER_MIGRATION_STEP,
        )
        mark_migration_step_complete(
            conn,
            BACKGROUND_COMPACTION_RANGE_DISPOSITIONS_MIGRATION_STEP,
        )
        mark_migration_step_complete(
            conn,
            BACKGROUND_COMPACTION_TELEMETRY_MIGRATION_STEP,
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    _ensure_source_coverage_backfill(conn)
    _ensure_node_edge_backfill(conn)


@contextmanager
def _bounded_thread_lock(
    lock: threading.RLock,
    timeout_seconds: float | None,
) -> Iterator[None]:
    """Acquire a process-local store lock without escaping a caller wall."""
    if timeout_seconds is None:
        acquired = bool(lock.acquire())
    else:
        acquired = bool(lock.acquire(timeout=max(0.0, float(timeout_seconds))))
    if not acquired:
        raise TimeoutError("background compaction store lock deadline exceeded")
    try:
        yield
    finally:
        lock.release()


class BackgroundCompactionStore:
    """SQLite-backed durable job, lease, preparation, and publication store."""

    def __init__(
        self,
        db_path: str | Path,
        config: LCMConfig | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._hermes_home = (
            str(self.db_path.parent)
            if str(db_path) != ":memory:" and self.db_path.name == "lcm.db"
            else ""
        )
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or LCMConfig(database_path=str(db_path))
        self._now = now or time.time
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            refuse_schema_version_too_new(self._conn)
            configure_connection(self._conn)
            self._conn.row_factory = sqlite3.Row
            run_versioned_migrations(self._conn)
            self._conn.commit()
            _ensure_background_schema(self._conn)
            message_columns = {
                str(row[1])
                for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            self._message_arrival_sql = (
                "COALESCE(ingested_at, timestamp)"
                if "ingested_at" in message_columns
                else "timestamp"
            )
        except BaseException:
            self._conn.close()
            self._conn = None
            raise

    @property
    def connection(self) -> sqlite3.Connection | None:
        return self._conn

    @contextmanager
    def _write_transaction(
        self,
        *,
        busy_timeout_ms: int | None = None,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> Iterator[sqlite3.Connection]:
        clock = deadline_clock or time.monotonic

        def remaining_seconds() -> float | None:
            if write_deadline is None:
                return None
            return max(0.0, float(write_deadline) - float(clock()))

        def check_deadline() -> None:
            remaining = remaining_seconds()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(
                    "background compaction write deadline exceeded"
                )

        effective_lock_timeout = lock_timeout_seconds
        remaining = remaining_seconds()
        if remaining is not None:
            if remaining <= 0:
                raise TimeoutError(
                    "background compaction write deadline exceeded"
                )
            effective_lock_timeout = (
                remaining
                if effective_lock_timeout is None
                else min(max(0.0, float(effective_lock_timeout)), remaining)
            )

        with _bounded_thread_lock(self._lock, effective_lock_timeout):
            check_deadline()
            conn = self._conn
            if conn is None:
                raise RuntimeError("background compaction store is closed")
            prior_busy_timeout = None
            effective_busy_timeout_ms = busy_timeout_ms
            remaining = remaining_seconds()
            if remaining is not None:
                remaining_ms = max(0, int(remaining * 1000))
                effective_busy_timeout_ms = (
                    remaining_ms
                    if effective_busy_timeout_ms is None
                    else min(max(0, int(effective_busy_timeout_ms)), remaining_ms)
                )
            if effective_busy_timeout_ms is not None:
                prior_busy_timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
                conn.execute(
                    f"PRAGMA busy_timeout={max(0, int(effective_busy_timeout_ms))}"
                )
            primary_error: BaseException | None = None
            try:
                try:
                    check_deadline()
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        check_deadline()
                        yield conn
                        check_deadline()
                        conn.execute("COMMIT")
                    except BaseException as exc:
                        primary_error = exc
                        try:
                            if conn.in_transaction:
                                conn.execute("ROLLBACK")
                        except BaseException:
                            logger.warning(
                                "LCM could not roll back a failed write transaction",
                                exc_info=True,
                            )
                        raise
                except BaseException as exc:
                    primary_error = exc
                    raise
            finally:
                if prior_busy_timeout is not None:
                    try:
                        conn.execute(f"PRAGMA busy_timeout={prior_busy_timeout}")
                    except BaseException:
                        if primary_error is None:
                            raise
                        logger.warning(
                            "LCM could not restore SQLite busy_timeout after a failed "
                            "write transaction",
                            exc_info=True,
                        )

    @contextmanager
    def _bounded_read_connection(
        self,
        *,
        lock_timeout_seconds: float | None = None,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        busy_timeout_ms: int | None = None,
    ) -> Iterator[sqlite3.Connection]:
        """Borrow the shared connection for a bounded foreground read."""
        clock = deadline_clock or time.monotonic

        def remaining_seconds() -> float | None:
            if deadline is None:
                return None
            return max(0.0, float(deadline) - float(clock()))

        remaining = remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("background compaction read deadline exceeded")
        effective_lock_timeout = lock_timeout_seconds
        if remaining is not None:
            effective_lock_timeout = (
                remaining
                if effective_lock_timeout is None
                else min(max(0.0, float(effective_lock_timeout)), remaining)
            )
        with _bounded_thread_lock(self._lock, effective_lock_timeout):
            conn = self._conn
            if conn is None:
                raise RuntimeError("background compaction store is closed")
            remaining = remaining_seconds()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("background compaction read deadline exceeded")
            effective_busy_timeout_ms = busy_timeout_ms
            if remaining is not None:
                remaining_ms = max(0, int(remaining * 1000))
                effective_busy_timeout_ms = (
                    remaining_ms
                    if effective_busy_timeout_ms is None
                    else min(max(0, int(effective_busy_timeout_ms)), remaining_ms)
                )
            prior_busy_timeout: int | None = None
            try:
                if effective_busy_timeout_ms is not None:
                    prior_busy_timeout = int(
                        conn.execute("PRAGMA busy_timeout").fetchone()[0]
                    )
                    conn.execute(
                        f"PRAGMA busy_timeout={max(0, int(effective_busy_timeout_ms))}"
                    )
                yield conn
                remaining = remaining_seconds()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "background compaction read deadline exceeded"
                    )
            finally:
                if prior_busy_timeout is not None:
                    conn.execute(f"PRAGMA busy_timeout={prior_busy_timeout}")

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            if conn is None:
                return
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None

    def __enter__(self) -> "BackgroundCompactionStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @staticmethod
    def _record_event_under_tx(
        conn: sqlite3.Connection,
        reason_code: str,
        *,
        conversation_id: str = "",
        session_id: str = "",
        job_id: str = "",
        detail: BaseException | str | None = None,
        created_at: float,
        deduplicate_seconds: float = 0.0,
    ) -> None:
        reason = str(reason_code or "").strip()
        if reason not in _EVENT_REASON_CODES:
            raise ValueError(f"unsupported compaction event reason: {reason}")
        conversation = str(conversation_id or "")
        session = str(session_id or "")
        job = str(job_id or "")
        if deduplicate_seconds > 0:
            duplicate = conn.execute(
                """
                SELECT 1 FROM lcm_compaction_events
                WHERE reason_code=? AND conversation_id=? AND session_id=? AND job_id=?
                  AND created_at>=?
                LIMIT 1
                """,
                (
                    reason,
                    conversation,
                    session,
                    job,
                    float(created_at) - float(deduplicate_seconds),
                ),
            ).fetchone()
            if duplicate is not None:
                return
        conn.execute(
            """
            INSERT INTO lcm_compaction_events(
                conversation_id, session_id, job_id, reason_code, detail, created_at
            ) VALUES(?,?,?,?,?,?)
            """,
            (
                conversation,
                session,
                job,
                reason,
                _sanitize_error(detail),
                float(created_at),
            ),
        )
        conn.execute(
            """
            DELETE FROM lcm_compaction_events
            WHERE event_id NOT IN (
                SELECT event_id FROM lcm_compaction_events
                ORDER BY event_id DESC LIMIT ?
            )
            """,
            (_MAX_EVENT_ROWS,),
        )

    def record_diagnostic_event(
        self,
        reason_code: str,
        *,
        conversation_id: str = "",
        session_id: str = "",
        job_id: str = "",
        detail: BaseException | str | None = None,
        now: float | None = None,
        deduplicate_seconds: float = 0.0,
        write_timeout_ms: int | None = None,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> None:
        current = float(self._now() if now is None else now)
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            self._record_event_under_tx(
                conn,
                reason_code,
                conversation_id=conversation_id,
                session_id=session_id,
                job_id=job_id,
                detail=detail,
                created_at=current,
                deduplicate_seconds=deduplicate_seconds,
            )

    @staticmethod
    def _supersede_active_jobs_under_tx(
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        reason: str,
        current: float,
        session_id: str | None = None,
    ) -> int:
        where = "conversation_id=? AND state IN ('queued','running','prepared','retry')"
        args: list[Any] = [str(conversation_id)]
        if session_id is not None:
            where += " AND session_id=?"
            args.append(str(session_id))
        affected = conn.execute(
            f"SELECT job_id, session_id FROM lcm_compaction_jobs WHERE {where}",
            args,
        ).fetchall()
        cursor = conn.execute(
            f"""
            UPDATE lcm_compaction_jobs
            SET state='superseded', completion_reason=?, updated_at=?,
                lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                lease_generation=lease_generation+1
            WHERE {where}
            """,
            (str(reason), float(current), *args),
        )
        for row in affected:
            BackgroundCompactionStore._record_event_under_tx(
                conn,
                "prepared_summary_superseded",
                conversation_id=str(conversation_id),
                session_id=str(row["session_id"] or ""),
                job_id=str(row["job_id"] or ""),
                detail=reason,
                created_at=float(current),
            )
        return int(cursor.rowcount or 0)

    @staticmethod
    def _session_fence_under_tx(
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        session_id: str,
        current: float,
        bootstrap: bool,
    ) -> sqlite3.Row | None:
        row = conn.execute(
            """
            SELECT conversation_id, session_id, lifecycle_bound_at,
                   lifecycle_reset_at,
                   generation, publishable, updated_at
            FROM lcm_compaction_session_fences WHERE conversation_id=?
            """,
            (str(conversation_id),),
        ).fetchone()
        if row is not None or not bootstrap:
            return row
        lifecycle = conn.execute(
            """
            SELECT current_bound_at, last_reset_at FROM lcm_lifecycle_state
            WHERE conversation_id=? AND current_session_id=?
            """,
            (str(conversation_id), str(session_id)),
        ).fetchone()
        if lifecycle is None:
            return None
        conn.execute(
            """
            INSERT OR IGNORE INTO lcm_compaction_session_fences(
                conversation_id, session_id, lifecycle_bound_at,
                lifecycle_reset_at,
                generation, publishable, updated_at
            ) VALUES(?,?,?,?,1,1,?)
            """,
            (
                str(conversation_id),
                str(session_id),
                float(lifecycle["current_bound_at"] or 0.0),
                float(lifecycle["last_reset_at"] or 0.0),
                float(current),
            ),
        )
        return conn.execute(
            """
            SELECT conversation_id, session_id, lifecycle_bound_at,
                   lifecycle_reset_at,
                   generation, publishable, updated_at
            FROM lcm_compaction_session_fences WHERE conversation_id=?
            """,
            (str(conversation_id),),
        ).fetchone()

    def activate_session(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_lifecycle_bound_at: float,
        expected_lifecycle_reset_at: float | None = None,
        now: float | None = None,
        write_timeout_ms: int | None = 100,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> int | None:
        """Activate exactly the lifecycle binding observed by the caller.

        Rechecking ``current_bound_at`` here prevents a stale process from
        overwriting a newer process's bind. A closed fence for the same binding
        remains closed; only a genuinely newer lifecycle bind may reopen it.
        """
        conversation = str(conversation_id or "").strip()
        session = str(session_id or "").strip()
        if not conversation or not session:
            raise ValueError("conversation_id and session_id are required")
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now() if now is None else now)
            lifecycle = conn.execute(
                """
                SELECT current_session_id, current_bound_at, last_reset_at
                FROM lcm_lifecycle_state WHERE conversation_id=?
                """,
                (conversation,),
            ).fetchone()
            if (
                lifecycle is None
                or str(lifecycle["current_session_id"] or "") != session
                or lifecycle["current_bound_at"] is None
                or float(lifecycle["current_bound_at"])
                != float(expected_lifecycle_bound_at)
                or float(lifecycle["last_reset_at"] or 0.0)
                != float(expected_lifecycle_reset_at or 0.0)
            ):
                return None
            # A newer process binding must not reopen a host-switch fence.
            # Only a durable host outcome can release it or adopt the summary.
            switch_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_host_model_switches'"
            ).fetchone()
            if switch_table and conn.execute(
                "SELECT 1 FROM lcm_host_model_switches WHERE conversation_id=? "
                "AND session_id=? AND state='reserved'",
                (conversation, session),
            ).fetchone():
                return None
            lifecycle_bound_at = float(lifecycle["current_bound_at"])
            lifecycle_reset_at = float(lifecycle["last_reset_at"] or 0.0)
            row = self._session_fence_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                current=current,
                bootstrap=False,
            )
            if (
                row is not None
                and str(row["session_id"] or "") == session
                and float(row["lifecycle_reset_at"] or 0.0)
                == lifecycle_reset_at
                and int(row["publishable"] or 0) == 1
            ):
                # ``current_bound_at`` proves that this caller observed the
                # latest lifecycle bind, but it is not itself a semantic
                # invalidation epoch.  Hermes routinely finalizes and rebinds
                # the same conversation/session across ordinary turns.  Move
                # the freshness watermark forward without fencing valid work.
                if float(row["lifecycle_bound_at"] or 0.0) != lifecycle_bound_at:
                    conn.execute(
                        """
                        UPDATE lcm_compaction_session_fences
                        SET lifecycle_bound_at=?, updated_at=?
                        WHERE conversation_id=? AND session_id=?
                          AND generation=? AND publishable=1
                          AND lifecycle_reset_at=?
                        """,
                        (
                            lifecycle_bound_at,
                            current,
                            conversation,
                            session,
                            int(row["generation"] or 0),
                            lifecycle_reset_at,
                        ),
                    )
                return int(row["generation"] or 0)
            if (
                row is not None
                and str(row["session_id"] or "") == session
                and float(row["lifecycle_bound_at"] or 0.0)
                == lifecycle_bound_at
                and float(row["lifecycle_reset_at"] or 0.0)
                == lifecycle_reset_at
            ):
                return None
            generation = int(row["generation"] or 0) + 1 if row is not None else 1
            self._supersede_active_jobs_under_tx(
                conn,
                conversation_id=conversation,
                reason="session_binding_changed",
                current=current,
            )
            conn.execute(
                """
                INSERT INTO lcm_compaction_session_fences(
                    conversation_id, session_id, lifecycle_bound_at,
                    lifecycle_reset_at,
                    generation, publishable, updated_at
                ) VALUES(?,?,?,?,?,1,?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    lifecycle_bound_at=excluded.lifecycle_bound_at,
                    lifecycle_reset_at=excluded.lifecycle_reset_at,
                    generation=excluded.generation,
                    publishable=1,
                    updated_at=excluded.updated_at
                """,
                (
                    conversation,
                    session,
                    lifecycle_bound_at,
                    lifecycle_reset_at,
                    generation,
                    current,
                ),
            )
            return generation

    def begin_session_reset(
        self,
        *,
        conversation_id: str,
        session_id: str,
        reason: str = "session_reset",
        now: float | None = None,
        write_timeout_ms: int | None = 100,
    ) -> int | None:
        """Fence publication and supersede active work before batched DAG deletion."""
        conversation = str(conversation_id or "").strip()
        session = str(session_id or "").strip()
        if not conversation or not session:
            return None
        with self._write_transaction(busy_timeout_ms=write_timeout_ms) as conn:
            current = float(self._now() if now is None else now)
            row = self._session_fence_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                current=current,
                bootstrap=False,
            )
            if row is not None and str(row["session_id"] or "") != session:
                return None
            if row is None:
                lifecycle = conn.execute(
                    """
                    SELECT current_session_id, current_bound_at, last_reset_at
                    FROM lcm_lifecycle_state WHERE conversation_id=?
                    """,
                    (conversation,),
                ).fetchone()
                if (
                    lifecycle is None
                    or str(lifecycle["current_session_id"] or "") != session
                    or lifecycle["current_bound_at"] is None
                ):
                    return None
                lifecycle_bound_at = float(lifecycle["current_bound_at"])
                lifecycle_reset_at = float(lifecycle["last_reset_at"] or 0.0)
            else:
                lifecycle_bound_at = float(row["lifecycle_bound_at"] or 0.0)
                lifecycle_reset_at = float(row["lifecycle_reset_at"] or 0.0)
            if row is not None and int(row["publishable"] or 0) == 0:
                generation = int(row["generation"] or 0)
            else:
                generation = int(row["generation"] or 0) + 1 if row is not None else 1
            conn.execute(
                """
                INSERT INTO lcm_compaction_session_fences(
                    conversation_id, session_id, lifecycle_bound_at,
                    lifecycle_reset_at, generation, publishable, updated_at
                ) VALUES(?,?,?,?,?,0,?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    lifecycle_bound_at=excluded.lifecycle_bound_at,
                    lifecycle_reset_at=excluded.lifecycle_reset_at,
                    generation=excluded.generation,
                    publishable=0,
                    updated_at=excluded.updated_at
                """,
                (
                    conversation,
                    session,
                    lifecycle_bound_at,
                    lifecycle_reset_at,
                    generation,
                    current,
                ),
            )
            fence_reason = _sanitize_error(reason or "session_reset")
            self._supersede_active_jobs_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                reason=fence_reason,
                current=current,
            )
            self._record_event_under_tx(
                conn,
                "session_reset_fence",
                conversation_id=conversation,
                session_id=session,
                detail=f"{fence_reason}:generation={generation}",
                created_at=current,
                deduplicate_seconds=1.0,
            )
            return generation

    def get_publishable_session_fence_generation(
        self,
        *,
        conversation_id: str,
        session_id: str,
    ) -> int | None:
        """Return the active durable publication epoch without changing it."""
        with self._lock:
            conn = self._conn
            if conn is None:
                return None
            row = conn.execute(
                """
                SELECT generation FROM lcm_compaction_session_fences
                WHERE conversation_id=? AND session_id=? AND publishable=1
                """,
                (str(conversation_id), str(session_id)),
            ).fetchone()
            return int(row[0]) if row is not None else None

    def _job_from_row(self, row: sqlite3.Row | None) -> CompactionJob | None:
        if row is None:
            return None
        conn = self._conn
        prepared_ids: list[int] = []
        if conn is not None:
            prepared_ids = [
                int(item[0])
                for item in conn.execute(
                    "SELECT prepared_node_id FROM lcm_compaction_prepared_nodes WHERE job_id=? ORDER BY ordinal",
                    (row["job_id"],),
                ).fetchall()
            ]
        return CompactionJob(
            job_id=str(row["job_id"]),
            conversation_id=str(row["conversation_id"]),
            session_id=str(row["session_id"]),
            session_fence_generation=int(row["session_fence_generation"] or 0),
            expected_frontier_store_id=int(row["expected_frontier_store_id"] or 0),
            captured_watermark_store_id=int(row["captured_watermark_store_id"] or 0),
            source_store_ids=_load_ids(row["source_store_ids"]),
            source_token_count=int(row["source_token_count"] or 0),
            source_identity_hash=str(row["source_identity_hash"] or ""),
            state=str(row["state"]),
            priority=int(row["priority"] or 0),
            created_at=float(row["created_at"] or 0.0),
            updated_at=float(row["updated_at"] or 0.0),
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            lease_generation=int(row["lease_generation"] or 0),
            attempt_count=int(row["attempt_count"] or 0),
            next_retry_at=row["next_retry_at"],
            last_error=row["last_error"],
            policy_fingerprint=str(row["policy_fingerprint"] or ""),
            expected_node_count=int(row["expected_node_count"] or 0),
            prepared_node_count=int(row["prepared_node_count"] or 0),
            completion_reason=row["completion_reason"],
            prepared_at=row["prepared_at"],
            published_at=row["published_at"],
            job_kind=str(row["job_kind"] or "standard"),
            repair_group_store_ids=_load_ids(row["repair_group_store_ids"]),
            repair_group_identity_hash=str(
                row["repair_group_identity_hash"] or ""
            ),
            repair_covered_store_ids=_load_ids(
                row["repair_covered_store_ids"]
            ),
            repair_uncovered_store_ids=_load_ids(
                row["repair_uncovered_store_ids"]
            ),
            repair_owner_node_ids=_load_ids(row["repair_owner_node_ids"]),
            repair_owner_map=str(row["repair_owner_map"] or "{}"),
            repair_capture_dag_epoch=int(row["repair_capture_dag_epoch"] or 0),
            repair_root_node_ids=_load_ids(row["repair_root_node_ids"]),
            repair_ancestry_node_ids=_load_ids(
                row["repair_ancestry_node_ids"]
            ),
            repair_topology_hash=str(row["repair_topology_hash"] or ""),
            repair_root_identity_hash=str(
                row["repair_root_identity_hash"] or ""
            ),
            repair_bound_dag_epoch=int(row["repair_bound_dag_epoch"] or 0),
            prepared_node_ids=prepared_ids,
        )

    def get_job(
        self,
        job_id: str,
        *,
        lock_timeout_seconds: float | None = None,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        busy_timeout_ms: int | None = None,
    ) -> CompactionJob | None:
        with self._bounded_read_connection(
            lock_timeout_seconds=lock_timeout_seconds,
            deadline=deadline,
            deadline_clock=deadline_clock,
            busy_timeout_ms=busy_timeout_ms,
        ) as conn:
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            return self._job_from_row(row)

    def has_active_partial_atomic_repair(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        lock_timeout_seconds: float | None = None,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        busy_timeout_ms: int | None = None,
    ) -> bool:
        """Return whether an exact live-frontier repair is durably active."""
        with self._bounded_read_connection(
            lock_timeout_seconds=lock_timeout_seconds,
            deadline=deadline,
            deadline_clock=deadline_clock,
            busy_timeout_ms=busy_timeout_ms,
        ) as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM lcm_compaction_jobs AS job
                JOIN lcm_lifecycle_state AS lifecycle
                  ON lifecycle.conversation_id=job.conversation_id
                 AND lifecycle.current_session_id=job.session_id
                 AND lifecycle.current_frontier_store_id=
                       job.expected_frontier_store_id
                JOIN lcm_compaction_session_fences AS fence
                  ON fence.conversation_id=job.conversation_id
                 AND fence.session_id=job.session_id
                 AND fence.generation=job.session_fence_generation
                 AND fence.publishable=1
                WHERE job.conversation_id=? AND job.session_id=?
                  AND job.expected_frontier_store_id=?
                  AND job.job_kind=?
                  AND job.state IN ('queued','running','prepared','retry')
                LIMIT 1
                """,
                (
                    str(conversation_id),
                    str(session_id),
                    max(0, int(expected_frontier_store_id or 0)),
                    _PARTIAL_REPAIR_JOB_KIND,
                ),
            ).fetchone()
            return row is not None

    @staticmethod
    def _record_lifecycle_debt_under_tx(
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        debt_tokens: int,
        current: float,
    ) -> None:
        debt = max(0, int(debt_tokens or 0))
        cursor = conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET debt_kind=CASE WHEN ?>0 THEN 'raw_backlog' ELSE NULL END,
                debt_size_estimate=?,
                debt_updated_at=?, updated_at=?
            WHERE conversation_id=? AND current_session_id=?
              AND current_frontier_store_id=?
            """,
            (
                debt,
                debt,
                current,
                current,
                str(conversation_id),
                str(session_id),
                max(0, int(expected_frontier_store_id or 0)),
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("background compaction lifecycle frontier is unavailable")

    @staticmethod
    def _matching_range_disposition_under_tx(
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        session_id: str,
        session_fence_generation: int,
        expected_frontier_store_id: int,
        captured_watermark_store_id: int,
        policy_fingerprint: str,
        source_identity_hash: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM lcm_compaction_range_dispositions
            WHERE conversation_id=? AND session_id=?
              AND session_fence_generation=?
              AND expected_frontier_store_id=?
              AND captured_watermark_store_id=?
              AND policy_fingerprint=? AND source_identity_hash=?
              AND resolved_at IS NULL
            """,
            (
                str(conversation_id),
                str(session_id),
                int(session_fence_generation),
                max(0, int(expected_frontier_store_id or 0)),
                max(0, int(captured_watermark_store_id or 0)),
                str(policy_fingerprint or ""),
                str(source_identity_hash or ""),
            ),
        ).fetchone()

    def _range_disposition_blocks_requeue_under_tx(
        self,
        conn: sqlite3.Connection,
        disposition: sqlite3.Row,
        *,
        current: float,
    ) -> bool:
        """Return whether an exact durable disposition still suppresses work.

        Content verdicts are terminal for an unchanged source identity. Provider
        failures are operational and therefore suppress only for the durable
        circuit-breaker cooldown.  The cooldown is anchored to the terminal
        failure time, not ``last_seen_at``, so frequent foreground captures
        cannot postpone a retry forever.
        """
        reason = str(disposition["reason_code"] or "")
        if reason not in _TRANSIENT_RANGE_DISPOSITION_REASONS:
            return True
        cooldown = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "summary_circuit_breaker_cooldown_seconds",
                    300,
                )
                or 0
            ),
        )
        if float(current) < float(disposition["first_seen_at"] or 0.0) + cooldown:
            return True
        conn.execute(
            """
            UPDATE lcm_compaction_range_dispositions
            SET resolved_at=?, last_seen_at=?
            WHERE conversation_id=? AND session_id=?
              AND session_fence_generation=?
              AND expected_frontier_store_id=?
              AND captured_watermark_store_id=?
              AND policy_fingerprint=? AND source_identity_hash=?
              AND resolved_at IS NULL
            """,
            (
                float(current),
                float(current),
                str(disposition["conversation_id"] or ""),
                str(disposition["session_id"] or ""),
                int(disposition["session_fence_generation"] or 0),
                int(disposition["expected_frontier_store_id"] or 0),
                int(disposition["captured_watermark_store_id"] or 0),
                str(disposition["policy_fingerprint"] or ""),
                str(disposition["source_identity_hash"] or ""),
            ),
        )
        return False

    @staticmethod
    def _range_disposition_capture_started_at(disposition: sqlite3.Row) -> float:
        """Return the schema-normalized durable source-capture boundary."""
        return float(disposition["capture_started_at"])

    @staticmethod
    def _upsert_range_disposition_under_tx(
        conn: sqlite3.Connection,
        *,
        row: Mapping[str, Any],
        reason_code: str,
        current: float,
        prepared_job_id: str = "",
    ) -> None:
        try:
            row_created_at = row["created_at"]
        except (IndexError, KeyError):
            row_created_at = None
        capture_started_at = float(
            current if row_created_at is None else row_created_at
        )
        conn.execute(
            """
            INSERT INTO lcm_compaction_range_dispositions(
                conversation_id, session_id, session_fence_generation,
                expected_frontier_store_id, captured_watermark_store_id,
                policy_fingerprint, source_identity_hash, reason_code,
                source_token_count, prepared_job_id, capture_started_at,
                first_seen_at, last_seen_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(
                conversation_id, session_id, session_fence_generation,
                expected_frontier_store_id, captured_watermark_store_id,
                policy_fingerprint, source_identity_hash
            ) DO UPDATE SET
                reason_code=excluded.reason_code,
                source_token_count=excluded.source_token_count,
                first_seen_at=CASE
                    WHEN lcm_compaction_range_dispositions.resolved_at IS NOT NULL
                    THEN excluded.first_seen_at
                    ELSE lcm_compaction_range_dispositions.first_seen_at
                END,
                capture_started_at=CASE
                    WHEN lcm_compaction_range_dispositions.resolved_at IS NOT NULL
                    THEN excluded.capture_started_at
                    ELSE lcm_compaction_range_dispositions.capture_started_at
                END,
                prepared_job_id=CASE
                    WHEN excluded.prepared_job_id<>''
                    THEN excluded.prepared_job_id
                    WHEN lcm_compaction_range_dispositions.resolved_at IS NOT NULL
                    THEN ''
                    ELSE lcm_compaction_range_dispositions.prepared_job_id
                END,
                last_seen_at=excluded.last_seen_at,
                resolved_at=NULL
            """,
            (
                str(row["conversation_id"] or ""),
                str(row["session_id"] or ""),
                int(row["session_fence_generation"] or 0),
                int(row["expected_frontier_store_id"] or 0),
                int(row["captured_watermark_store_id"] or 0),
                str(row["policy_fingerprint"] or ""),
                str(row["source_identity_hash"] or ""),
                str(reason_code or ""),
                int(row["source_token_count"] or 0),
                str(prepared_job_id or ""),
                capture_started_at,
                float(current),
                float(current),
            ),
        )

    def _prepare_lossless_projection(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        captured_watermark_store_id: int,
        source_store_ids: Sequence[int],
        expected_source_identity_hash: str,
        policy_fingerprint: str,
        projection: str,
        priority: int,
        write_timeout_ms: int | None = 100,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        replace_disposition_reason: str = "",
        new_material_upper_store_id: int = 0,
    ) -> CompactionJob | None:
        """Durably prepare a tiny exact range without provider work.

        The prepared node is a policy-redacted active projection only.  Its
        exact raw lineage remains authoritative and publication still uses the
        normal fixed-reserve CAS before coverage/frontier movement.
        """
        conversation = str(conversation_id or "").strip()
        session = str(session_id or "").strip()
        frontier = max(0, int(expected_frontier_store_id or 0))
        ids = tuple(int(item) for item in source_store_ids)
        watermark = max(frontier, int(captured_watermark_store_id or 0))
        if (
            not conversation
            or not session
            or not ids
            or tuple(sorted(set(ids))) != ids
            or ids[-1] != watermark
        ):
            return None
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now())
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                current=current,
                bootstrap=True,
            )
            if (
                fence is None
                or str(fence["session_id"] or "") != session
                or int(fence["publishable"] or 0) != 1
            ):
                return None
            generation = int(fence["generation"] or 0)
            lifecycle = conn.execute(
                """
                SELECT current_session_id, current_frontier_store_id
                FROM lcm_lifecycle_state WHERE conversation_id=?
                """,
                (conversation,),
            ).fetchone()
            if (
                lifecycle is None
                or str(lifecycle["current_session_id"] or "") != session
                or int(lifecycle["current_frontier_store_id"] or 0) != frontier
            ):
                return None
            raw_messages = self._load_source_range(
                conn,
                session_id=session,
                conversation_id=conversation,
                frontier=frontier,
                watermark=watermark,
            )
            raw_ids = tuple(int(message["store_id"]) for message in raw_messages)
            if raw_ids != ids or self._source_ids_already_canonical(conn, session, ids):
                return None
            identity_hash = self._source_identity_hash(raw_messages)
            if identity_hash != str(expected_source_identity_hash or ""):
                # The projection was serialized from the read-only capture.
                # Never pair that stale representation with rows changed before
                # the short preparation transaction acquired its writer lock.
                return None
            source_tokens = sum(self._message_tokens(item) for item in raw_messages)
            disposition = self._matching_range_disposition_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                session_fence_generation=generation,
                expected_frontier_store_id=frontier,
                captured_watermark_store_id=watermark,
                policy_fingerprint=policy_fingerprint,
                source_identity_hash=identity_hash,
            )
            replacement_reason = str(replace_disposition_reason or "")
            replacement_allowed = bool(
                disposition is not None
                and replacement_reason
                and replacement_reason in _EXTENDABLE_RANGE_DISPOSITION_REASONS
                and str(disposition["reason_code"] or "") == replacement_reason
                and int(new_material_upper_store_id or 0) > watermark
                and conn.execute(
                    f"""
                    SELECT 1 FROM messages
                    WHERE session_id=? AND store_id>? AND store_id<=?
                      AND (conversation_id=? OR conversation_id=''
                           OR conversation_id IS NULL)
                      AND {self._message_arrival_sql}>?
                    ORDER BY store_id
                    LIMIT 1
                    """,
                    (
                        session,
                        watermark,
                        int(new_material_upper_store_id),
                        conversation,
                        self._range_disposition_capture_started_at(disposition),
                    ),
                ).fetchone()
                is not None
            )
            if replacement_reason and not replacement_allowed:
                # A local terminal-prefix replacement must be authorized by
                # the same still-active exact disposition and a still-durable
                # post-capture row.  Fail before touching any concurrent job.
                return None
            active = conn.execute(
                """
                SELECT * FROM lcm_compaction_jobs
                WHERE conversation_id=?
                  AND state IN ('queued','running','prepared','retry')
                LIMIT 1
                """,
                (conversation,),
            ).fetchone()
            if active is not None:
                same_capture_scope = (
                    str(active["session_id"] or "") == session
                    and int(active["session_fence_generation"] or 0) == generation
                    and int(active["expected_frontier_store_id"] or 0) == frontier
                    and str(active["policy_fingerprint"] or "")
                    == str(policy_fingerprint or "")
                )
                if same_capture_scope:
                    active_ids = _load_ids(active["source_store_ids"])
                    active_watermark = int(
                        active["captured_watermark_store_id"] or 0
                    )
                    active_messages = self._load_source_range(
                        conn,
                        session_id=session,
                        conversation_id=conversation,
                        frontier=frontier,
                        watermark=active_watermark,
                    )
                    active_source_is_current = (
                        str(active["job_kind"] or "standard")
                        == _LOSSLESS_RAW_PROJECTION_JOB_KIND
                        and bool(active_ids)
                        and active_ids[-1] == active_watermark
                        and tuple(
                            int(message["store_id"])
                            for message in active_messages
                        )
                        == active_ids
                        and self._source_identity_hash(active_messages)
                        == str(active["source_identity_hash"] or "")
                        and not self._source_ids_already_canonical(
                            conn,
                            session,
                            active_ids,
                        )
                    )
                    if active_source_is_current:
                        # Keep an immutable prepared prefix when later rows were
                        # merely appended above its watermark. Publication of
                        # that valid prefix must not be churned by ordinary turns.
                        return self._job_from_row(active)
                    reason = (
                        "special_job_kind_mismatch"
                        if str(active["job_kind"] or "standard")
                        != _LOSSLESS_RAW_PROJECTION_JOB_KIND
                        else "source_identity_mismatch"
                    )
                else:
                    reason = "session_or_policy_changed_before_claim"
                self._supersede_active_jobs_under_tx(
                    conn,
                    conversation_id=conversation,
                    reason=reason,
                    current=current,
                )
            if disposition is not None:
                if not replacement_allowed and self._range_disposition_blocks_requeue_under_tx(
                    conn,
                    disposition,
                    current=current,
                ):
                    return None

            job_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO lcm_compaction_jobs(
                    job_id, conversation_id, session_id,
                    session_fence_generation, expected_frontier_store_id,
                    captured_watermark_store_id, source_store_ids,
                    source_token_count, source_identity_hash, state, priority,
                    created_at, updated_at, policy_fingerprint,
                    expected_node_count, prepared_node_count,
                    completion_reason, prepared_at, job_kind
                ) VALUES(?,?,?,?,?,?,?,?,?,'prepared',?,?,?,?,1,1,?,?,?)
                """,
                (
                    job_id,
                    conversation,
                    session,
                    generation,
                    frontier,
                    watermark,
                    _json_ids(ids),
                    source_tokens,
                    identity_hash,
                    int(priority or 0),
                    current,
                    current,
                    str(policy_fingerprint or ""),
                    "insufficient_compressible_source",
                    current,
                    _LOSSLESS_RAW_PROJECTION_JOB_KIND,
                ),
            )
            conn.execute(
                """
                INSERT INTO lcm_compaction_prepared_nodes(
                    job_id, ordinal, summary, source_store_ids,
                    source_token_count, summary_token_count,
                    creation_model, creation_provider, route_latency_ms,
                    attempt_number, escalation_level,
                    attribution_available, fallback_used, prepared_at
                ) VALUES(?,?,?,?,?,?,?,?,0,0,0,1,0,?)
                """,
                (
                    job_id,
                    0,
                    str(projection or ""),
                    _json_ids(ids),
                    source_tokens,
                    max(1, int(count_tokens(str(projection or "")))),
                    "deterministic/lossless-raw-v1",
                    "local",
                    current,
                ),
            )
            job_row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            assert job_row is not None
            self._upsert_range_disposition_under_tx(
                conn,
                row=job_row,
                reason_code="insufficient_compressible_source",
                current=current,
                prepared_job_id=job_id,
            )
            self._record_lifecycle_debt_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                expected_frontier_store_id=frontier,
                debt_tokens=source_tokens,
                current=current,
            )
            self._record_event_under_tx(
                conn,
                "insufficient_compressible_source",
                conversation_id=conversation,
                session_id=session,
                job_id=job_id,
                detail="lossless_raw_projection_prepared",
                created_at=current,
            )
            return self._job_from_row(job_row)

    def enqueue_or_coalesce(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        captured_watermark_store_id: int,
        source_store_ids: Sequence[int],
        source_token_count: int,
        policy_fingerprint: str,
        expected_node_count: int = 1,
        priority: int = 0,
        now: float | None = None,
        transaction_hook: Callable[[], None] | None = None,
        write_timeout_ms: int | None = None,
        reject_canonical_coverage: bool = True,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> CompactionJob | None:
        """Insert one active conversation job or extend an unclaimed request."""
        current = float(self._now() if now is None else now)
        conversation_id = str(conversation_id or "").strip()
        session_id = str(session_id or "").strip()
        if not conversation_id or not session_id:
            raise ValueError("conversation_id and session_id are required")
        frontier = max(0, int(expected_frontier_store_id or 0))
        watermark = max(frontier, int(captured_watermark_store_id or 0))
        ids = tuple(sorted({int(value) for value in source_store_ids if frontier < int(value) <= watermark}))
        if not ids:
            raise ValueError("source_store_ids must contain a durable range above the frontier")
        watermark = max(ids)
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            if transaction_hook is not None:
                transaction_hook()
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=conversation_id,
                session_id=session_id,
                current=current,
                bootstrap=True,
            )
            if (
                fence is None
                or str(fence["session_id"] or "") != session_id
                or int(fence["publishable"] or 0) != 1
            ):
                raise RuntimeError("background compaction session is fenced")
            session_fence_generation = int(fence["generation"] or 0)
            durable_rows = self._load_source_range(
                conn,
                session_id=session_id,
                conversation_id=conversation_id,
                frontier=frontier,
                watermark=watermark,
            )
            durable_by_id = {
                int(message["store_id"]): message for message in durable_rows
            }
            if any(store_id not in durable_by_id for store_id in ids):
                raise ValueError("source_store_ids contain missing or cross-session rows")
            selected_rows = [durable_by_id[store_id] for store_id in ids]
            source_identity_hash = self._source_identity_hash(selected_rows)
            durable_source_tokens = sum(
                self._message_tokens(message) for message in selected_rows
            )
            row = conn.execute(
                """
                SELECT * FROM lcm_compaction_jobs
                WHERE conversation_id=? AND state IN ('queued','running','prepared','retry')
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                disposition = self._matching_range_disposition_under_tx(
                    conn,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    session_fence_generation=session_fence_generation,
                    expected_frontier_store_id=frontier,
                    captured_watermark_store_id=watermark,
                    policy_fingerprint=policy_fingerprint,
                    source_identity_hash=source_identity_hash,
                )
                if disposition is not None:
                    if not self._range_disposition_blocks_requeue_under_tx(
                        conn,
                        disposition,
                        current=current,
                    ):
                        disposition = None
                if disposition is not None:
                    conn.execute(
                        """
                        UPDATE lcm_compaction_range_dispositions
                        SET last_seen_at=?
                        WHERE conversation_id=? AND session_id=?
                          AND session_fence_generation=?
                          AND expected_frontier_store_id=?
                          AND captured_watermark_store_id=?
                          AND policy_fingerprint=? AND source_identity_hash=?
                        """,
                        (
                            current,
                            conversation_id,
                            session_id,
                            session_fence_generation,
                            frontier,
                            watermark,
                            str(policy_fingerprint or ""),
                            source_identity_hash,
                        ),
                    )
                    return None
            if reject_canonical_coverage and self._source_ids_already_canonical(
                conn,
                session_id,
                ids,
            ):
                # Capture's read-only cutoff can race a canonical DAG insert.
                # Recheck after BEGIN IMMEDIATE so no provider job can be
                # queued for a range that already has normalized coverage.
                self._supersede_active_jobs_under_tx(
                    conn,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    reason="source_range_already_published",
                    current=current,
                )
                return None
            if row is not None:
                existing_frontier = int(row["expected_frontier_store_id"] or 0)
                same_capture_policy = (
                    row["session_id"] == session_id
                    and int(row["session_fence_generation"] or 0)
                    == session_fence_generation
                    and str(row["policy_fingerprint"] or "")
                    == str(policy_fingerprint or "")
                )
                if (
                    existing_frontier == frontier
                    and same_capture_policy
                    and row["state"] in {"queued", "retry"}
                ):
                    merged_ids = tuple(sorted(set(_load_ids(row["source_store_ids"])) | set(ids)))
                    if (
                        reject_canonical_coverage
                        and self._source_ids_already_canonical(
                            conn,
                            session_id,
                            merged_ids,
                        )
                    ):
                        self._supersede_active_jobs_under_tx(
                            conn,
                            conversation_id=conversation_id,
                            session_id=session_id,
                            reason="source_range_already_published",
                            current=current,
                        )
                        row = None
                    else:
                        merged_watermark = max(merged_ids)
                    if row is not None:
                        merged_rows = self._load_source_range(
                            conn,
                            session_id=session_id,
                            conversation_id=conversation_id,
                            frontier=frontier,
                            watermark=merged_watermark,
                        )
                        merged_by_id = {
                            int(message["store_id"]): message for message in merged_rows
                        }
                    if row is not None and any(
                        store_id not in merged_by_id for store_id in merged_ids
                    ):
                        conn.execute(
                            """
                            UPDATE lcm_compaction_jobs
                            SET state='superseded',
                                completion_reason='source_range_changed_before_claim',
                                updated_at=?, lease_owner=NULL,
                                lease_expires_at=NULL, heartbeat_at=NULL
                            WHERE job_id=?
                            """,
                            (current, row["job_id"]),
                        )
                        self._record_event_under_tx(
                            conn,
                            "prepared_summary_superseded",
                            conversation_id=conversation_id,
                            session_id=str(row["session_id"] or ""),
                            job_id=str(row["job_id"]),
                            detail="source_range_changed_before_claim",
                            created_at=current,
                        )
                        row = None
                    elif row is not None:
                        merged_selected = [merged_by_id[store_id] for store_id in merged_ids]
                        merged_identity_hash = self._source_identity_hash(merged_selected)
                        merged_source_tokens = sum(
                            self._message_tokens(message) for message in merged_selected
                        )
                        source_changed = (
                            merged_identity_hash != str(row["source_identity_hash"] or "")
                        )
                        if source_changed:
                            conn.execute(
                                "DELETE FROM lcm_compaction_prepared_nodes WHERE job_id=?",
                                (row["job_id"],),
                            )
                        conn.execute(
                            """
                            UPDATE lcm_compaction_jobs
                            SET session_id=?, session_fence_generation=?,
                                captured_watermark_store_id=?, source_store_ids=?,
                                source_token_count=?, source_identity_hash=?,
                                state=CASE WHEN state='retry' THEN 'retry' ELSE 'queued' END,
                                priority=MAX(priority, ?), updated_at=?,
                                next_retry_at=CASE WHEN state='retry' THEN next_retry_at ELSE NULL END,
                                last_error=CASE WHEN state='retry' THEN last_error ELSE NULL END,
                                policy_fingerprint=?,
                                expected_node_count=MAX(expected_node_count, ?),
                                prepared_node_count=CASE WHEN ? THEN 0 ELSE prepared_node_count END,
                                prepared_at=CASE WHEN ? THEN NULL ELSE prepared_at END,
                                completion_reason=CASE
                                    WHEN state='retry' THEN completion_reason ELSE NULL END
                            WHERE job_id=?
                            """,
                            (
                                session_id,
                                session_fence_generation,
                                merged_watermark,
                                _json_ids(merged_ids),
                                merged_source_tokens,
                                merged_identity_hash,
                                int(priority or 0),
                                current,
                                str(policy_fingerprint or ""),
                                max(0, int(expected_node_count or 0)),
                                int(source_changed),
                                int(source_changed),
                                row["job_id"],
                            ),
                        )
                        self._record_lifecycle_debt_under_tx(
                            conn,
                            conversation_id=conversation_id,
                            session_id=session_id,
                            expected_frontier_store_id=frontier,
                            debt_tokens=merged_source_tokens,
                            current=current,
                        )
                        updated = conn.execute(
                            "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (row["job_id"],)
                        ).fetchone()
                        job = self._job_from_row(updated)
                        assert job is not None
                        return job
                if row is not None and (
                    existing_frontier != frontier or not same_capture_policy
                ):
                    conn.execute(
                        """
                        UPDATE lcm_compaction_jobs
                        SET state='superseded', completion_reason=?,
                            lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL, updated_at=?
                        WHERE job_id=?
                        """,
                        (
                            "session_or_policy_changed_before_claim"
                            if existing_frontier == frontier
                            else "frontier_changed_before_claim",
                            current,
                            row["job_id"],
                        ),
                    )
                    self._record_event_under_tx(
                        conn,
                        "prepared_summary_superseded",
                        conversation_id=conversation_id,
                        session_id=str(row["session_id"] or ""),
                        job_id=str(row["job_id"]),
                        detail=(
                            "session_or_policy_changed_before_claim"
                            if existing_frontier == frontier
                            else "frontier_changed_before_claim"
                        ),
                        created_at=current,
                    )
                elif row is not None and row["state"] not in {"queued", "retry"}:
                    if not same_capture_policy:
                        conn.execute(
                            """
                            UPDATE lcm_compaction_jobs
                            SET state='superseded', completion_reason='session_or_policy_changed',
                                lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                                lease_generation=lease_generation+1, updated_at=?
                            WHERE job_id=?
                            """,
                            (current, row["job_id"]),
                        )
                        self._record_event_under_tx(
                            conn,
                            "prepared_summary_superseded",
                            conversation_id=conversation_id,
                            session_id=str(row["session_id"] or ""),
                            job_id=str(row["job_id"]),
                            detail="session_or_policy_changed",
                            created_at=current,
                        )
                    else:
                        self._record_lifecycle_debt_under_tx(
                            conn,
                            conversation_id=conversation_id,
                            session_id=session_id,
                            expected_frontier_store_id=frontier,
                            debt_tokens=int(row["source_token_count"] or 0),
                            current=current,
                        )
                        job = self._job_from_row(row)
                        assert job is not None
                        return job

            job_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO lcm_compaction_jobs(
                    job_id, conversation_id, session_id, session_fence_generation,
                    expected_frontier_store_id, captured_watermark_store_id,
                    source_store_ids, source_token_count, source_identity_hash,
                    state, priority, created_at, updated_at, policy_fingerprint,
                    expected_node_count
                ) VALUES(?,?,?,?,?,?,?,?,?,'queued',?,?,?,?,?)
                """,
                (
                    job_id,
                    conversation_id,
                    session_id,
                    session_fence_generation,
                    frontier,
                    watermark,
                    _json_ids(ids),
                    durable_source_tokens,
                    source_identity_hash,
                    int(priority or 0),
                    current,
                    current,
                    str(policy_fingerprint or ""),
                    max(0, int(expected_node_count or 0)),
                ),
            )
            self._record_lifecycle_debt_under_tx(
                conn,
                conversation_id=conversation_id,
                session_id=session_id,
                expected_frontier_store_id=frontier,
                debt_tokens=durable_source_tokens,
                current=current,
            )
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            job = self._job_from_row(row)
            assert job is not None
            return job

    @staticmethod
    def _message_dict(row: sqlite3.Row) -> dict[str, Any]:
        tool_calls = row["tool_calls"]
        if tool_calls:
            try:
                tool_calls = json.loads(tool_calls)
            except (TypeError, json.JSONDecodeError):
                tool_calls = None
        return {
            "store_id": int(row["store_id"]),
            "session_id": str(row["session_id"] or ""),
            "conversation_id": str(row["conversation_id"] or ""),
            "role": row["role"],
            "content": row["content"],
            "tool_call_id": row["tool_call_id"],
            "tool_calls": tool_calls,
            "tool_name": row["tool_name"],
            "timestamp": row["timestamp"],
            "token_estimate": int(row["token_estimate"] or 0),
        }

    def _source_identity_hash(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> str:
        payload = [
            {
                "store_id": int(message.get("store_id") or 0),
                "session_id": str(message.get("session_id") or ""),
                "conversation_id": str(message.get("conversation_id") or ""),
                "role": str(message.get("role") or ""),
                "content": message.get("content"),
                "tool_call_id": message.get("tool_call_id"),
                "tool_calls": message.get("tool_calls"),
                "tool_name": message.get("tool_name"),
                "timestamp": message.get("timestamp"),
            }
            for message in messages
        ]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()

    def capture_foreground_source_snapshot(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        captured_watermark_store_id: int,
        expected_source_store_ids: Sequence[int],
        lock_timeout_seconds: float | None = None,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        busy_timeout_ms: int | None = None,
    ) -> tuple[dict[str, Any], ...] | None:
        """Capture the exact bounded durable interval before provider I/O."""
        expected_ids = tuple(int(item) for item in expected_source_store_ids)
        if not expected_ids:
            return None
        with self._bounded_read_connection(
            lock_timeout_seconds=lock_timeout_seconds,
            deadline=deadline,
            deadline_clock=deadline_clock,
            busy_timeout_ms=busy_timeout_ms,
        ) as conn:
            messages = self._load_source_range(
                conn,
                session_id=str(session_id),
                conversation_id=str(conversation_id),
                frontier=int(expected_frontier_store_id or 0),
                watermark=int(captured_watermark_store_id or 0),
            )
        actual_ids = tuple(int(message["store_id"]) for message in messages)
        if actual_ids != expected_ids:
            return None
        return tuple(dict(message) for message in messages)

    def capture_foreground_interval_snapshot(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        captured_watermark_store_id: int,
        expected_source_store_ids: Sequence[int],
        lock_timeout_seconds: float | None = None,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        busy_timeout_ms: int | None = None,
    ) -> tuple[tuple[dict[str, Any], ...], dict[int, int], dict[int, float]] | None:
        """Capture the whole durable interval a caller chunk lies within.

        Used when the caller's active chunk does not map 1:1 onto the durable
        interval (older builds stored duplicate copies; hosts rewrite or drop
        turns). Returns ``(interval rows, {duplicate store_id: earliest
        store_id})`` only when every caller id lies inside the interval or at or
        below the frontier (directly or through its earliest identical row).
        The caller must then summarize the durable rows, never the caller view.
        """
        from .store import message_dedupe_identity

        frontier = int(expected_frontier_store_id or 0)
        watermark = int(captured_watermark_store_id or 0)
        expected_ids = tuple(int(item) for item in expected_source_store_ids)
        if not expected_ids or watermark <= frontier:
            return None
        with self._bounded_read_connection(
            lock_timeout_seconds=lock_timeout_seconds,
            deadline=deadline,
            deadline_clock=deadline_clock,
            busy_timeout_ms=busy_timeout_ms,
        ) as conn:
            messages = self._load_source_range(
                conn,
                session_id=str(session_id),
                conversation_id=str(conversation_id),
                frontier=frontier,
                watermark=watermark,
            )
            identity_rows = conn.execute(
                """
                SELECT store_id, role, content, tool_call_id, tool_calls, observed_at
                FROM messages
                WHERE session_id=? AND store_id<=? AND observed_at IN (
                    SELECT observed_at FROM messages
                    WHERE session_id=? AND store_id>? AND store_id<=?
                      AND observed_at IS NOT NULL
                )
                ORDER BY store_id
                """,
                (str(session_id), watermark, str(session_id), frontier, watermark),
            ).fetchall()
        if not messages:
            return None
        earliest: dict[tuple[Any, ...], int] = {}
        aliases: dict[int, int] = {}
        observed_by_id: dict[int, float] = {}
        for store_id, role, content, tool_call_id, tool_calls, observed_at in identity_rows:
            if observed_at is not None:
                observed_by_id[int(store_id)] = float(observed_at)
            identity = message_dedupe_identity(
                {
                    "role": role,
                    "content": content,
                    "tool_call_id": tool_call_id,
                    "tool_calls": tool_calls,
                    "observed_at": observed_at,
                },
                stored_row=True,
            )
            if identity is None:
                continue
            kept = earliest.setdefault(identity, int(store_id))
            if kept != int(store_id):
                aliases[int(store_id)] = kept
        interval_ids = {int(message["store_id"]) for message in messages}
        for store_id in expected_ids:
            if store_id <= frontier or store_id in interval_ids:
                continue
            canonical = aliases.get(store_id, store_id)
            if canonical <= frontier or canonical in interval_ids:
                continue
            return None
        return tuple(dict(message) for message in messages), aliases, observed_by_id

    def foreground_source_identity_hash(
        self,
        snapshot: Sequence[Mapping[str, Any]],
        source_store_ids: Sequence[int],
    ) -> str | None:
        """Derive the identity of an exact ordered subset from a prior snapshot."""
        wanted = tuple(int(item) for item in source_store_ids)
        by_id = {int(message.get("store_id") or 0): message for message in snapshot}
        if not wanted or any(store_id not in by_id for store_id in wanted):
            return None
        selected = [by_id[store_id] for store_id in wanted]
        if tuple(int(message.get("store_id") or 0) for message in selected) != wanted:
            return None
        return self._source_identity_hash(selected)

    @classmethod
    def _load_source_range(
        cls,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        conversation_id: str,
        frontier: int,
        watermark: int,
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT store_id, session_id, conversation_id, role, content,
                   tool_call_id, tool_calls, tool_name, timestamp, token_estimate
            FROM messages
            WHERE session_id=? AND store_id>? AND store_id<=?
              AND (conversation_id=? OR conversation_id='' OR conversation_id IS NULL)
            ORDER BY store_id
            """,
            (str(session_id), int(frontier), int(watermark), str(conversation_id)),
        ).fetchall()
        return [cls._message_dict(row) for row in rows]

    @staticmethod
    def _tool_call_ids(message: Mapping[str, Any]) -> set[str]:
        values: set[str] = set()
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, Mapping):
                value = tool_call.get("id") or tool_call.get("tool_call_id")
                if value:
                    values.add(str(value))
        return values

    @classmethod
    def _atomic_groups(
        cls, messages: Sequence[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            call_ids = cls._tool_call_ids(message) if message.get("role") == "assistant" else set()
            if call_ids:
                group = [message]
                cursor = index + 1
                while cursor < len(messages):
                    candidate = messages[cursor]
                    if candidate.get("role") != "tool":
                        break
                    result_id = str(candidate.get("tool_call_id") or "")
                    if result_id and result_id not in call_ids:
                        break
                    group.append(candidate)
                    cursor += 1
                groups.append(group)
                index = cursor
                continue
            groups.append([message])
            index += 1
        return groups

    @classmethod
    def _atomic_group_is_complete(
        cls,
        group: Sequence[Mapping[str, Any]],
    ) -> bool:
        if not group:
            return False
        opening = group[0]
        declared = (
            cls._tool_call_ids(opening)
            if str(opening.get("role") or "") == "assistant"
            else set()
        )
        if not declared:
            return len(group) == 1
        observed = {
            str(message.get("tool_call_id") or "")
            for message in group[1:]
            if str(message.get("role") or "") == "tool"
            and str(message.get("tool_call_id") or "")
        }
        return not (declared - observed)

    @staticmethod
    def _live_owner_map(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        source_ids: Sequence[int],
        owner_limit: int | None = None,
    ) -> tuple[dict[int, tuple[int, ...]], tuple[int, ...]] | None:
        wanted = tuple(dict.fromkeys(int(item) for item in source_ids))
        if not wanted:
            return {}, ()
        owner_cap = max(
            1,
            int(
                _PARTIAL_REPAIR_MAX_OWNERS
                if owner_limit is None
                else owner_limit
            ),
        )
        placeholders = ",".join("?" for _ in wanted)
        owners = tuple(
            int(row[0])
            for row in conn.execute(
                f"""
                SELECT DISTINCT owner.node_id
                FROM lcm_compaction_node_sources AS owner
                JOIN summary_nodes AS live
                  ON live.node_id=owner.node_id
                 AND live.session_id=owner.session_id
                 AND live.source_type='messages'
                WHERE owner.session_id=?
                  AND owner.store_id IN ({placeholders})
                ORDER BY owner.node_id
                LIMIT ?
                """,
                (str(session_id), *wanted, owner_cap + 1),
            ).fetchall()
        )
        if len(owners) > owner_cap:
            return None
        mapping: dict[int, list[int]] = {store_id: [] for store_id in wanted}
        if owners:
            owner_placeholders = ",".join("?" for _ in owners)
            for row in conn.execute(
                f"""
                SELECT owner.store_id, owner.node_id
                FROM lcm_compaction_node_sources AS owner
                JOIN summary_nodes AS live
                  ON live.node_id=owner.node_id
                 AND live.session_id=owner.session_id
                 AND live.source_type='messages'
                WHERE owner.session_id=?
                  AND owner.store_id IN ({placeholders})
                  AND owner.node_id IN ({owner_placeholders})
                ORDER BY owner.store_id, owner.node_id
                """,
                (str(session_id), *wanted, *owners),
            ).fetchall():
                mapping[int(row["store_id"])].append(int(row["node_id"]))
        return (
            {key: tuple(value) for key, value in mapping.items()},
            owners,
        )

    @staticmethod
    def _owner_map_json(owner_map: Mapping[int, Sequence[int]]) -> str:
        return _canonical_json(
            {
                str(int(store_id)): [int(node_id) for node_id in owner_ids]
                for store_id, owner_ids in sorted(owner_map.items())
            }
        )

    @staticmethod
    def _session_dag_epoch(conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            """
            SELECT mutation_epoch FROM lcm_compaction_dag_epochs
            WHERE session_id=?
            """,
            (str(session_id),),
        ).fetchone()
        return int(row[0] or 0) if row is not None else 0

    @staticmethod
    def _repair_root_identity_hash(rows: Sequence[Mapping[str, Any]]) -> str:
        return _canonical_hash(
            [
                {
                    "node_id": int(row["node_id"]),
                    "session_id": str(row["session_id"] or ""),
                    "depth": int(row["depth"] or 0),
                    "summary": str(row["summary"] or ""),
                    "token_count": int(row["token_count"] or 0),
                    "source_token_count": int(row["source_token_count"] or 0),
                    "source_ids": list(_load_ids(row["source_ids"])),
                    "source_type": str(row["source_type"] or ""),
                    "created_at": float(row["created_at"] or 0.0),
                    "earliest_at": row["earliest_at"],
                    "latest_at": row["latest_at"],
                    "expand_hint": str(row["expand_hint"] or ""),
                }
                for row in rows
            ]
        )

    @staticmethod
    def _repair_coverage_elections_valid(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        owner_map: Mapping[int, Sequence[int]],
        covered_ids: Sequence[int],
        uncovered_ids: Sequence[int],
    ) -> bool:
        covered = tuple(int(item) for item in covered_ids)
        uncovered = tuple(int(item) for item in uncovered_ids)
        if covered:
            placeholders = ",".join("?" for _ in covered)
            rows = conn.execute(
                f"""
                SELECT store_id, publication_id
                FROM lcm_compaction_source_coverage
                WHERE session_id=? AND store_id IN ({placeholders})
                ORDER BY store_id
                """,
                (str(session_id), *covered),
            ).fetchall()
            elected = {int(row["store_id"]): str(row["publication_id"] or "") for row in rows}
            for store_id in covered:
                valid = {
                    f"summary-node:{node_id}"
                    for node_id in owner_map.get(store_id, ())
                } | {
                    f"legacy-node:{node_id}"
                    for node_id in owner_map.get(store_id, ())
                }
                if elected.get(store_id) not in valid:
                    return False
        if uncovered:
            placeholders = ",".join("?" for _ in uncovered)
            if conn.execute(
                f"""
                SELECT 1 FROM lcm_compaction_source_coverage
                WHERE session_id=? AND store_id IN ({placeholders}) LIMIT 1
                """,
                (str(session_id), *uncovered),
            ).fetchone() is not None:
                return False
        return True

    @staticmethod
    def _resolve_partial_repair_topology(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        owner_ids: Sequence[int],
        ancestry_limit: int | None = None,
        root_limit: int | None = None,
    ) -> dict[str, Any] | None:
        ancestry_cap = max(
            1,
            int(
                _PARTIAL_REPAIR_MAX_ANCESTRY
                if ancestry_limit is None
                else ancestry_limit
            ),
        )
        root_cap = max(
            1,
            int(_PARTIAL_REPAIR_MAX_ROOTS if root_limit is None else root_limit),
        )
        owners = tuple(sorted({int(item) for item in owner_ids}))
        if not owners or len(owners) > _PARTIAL_REPAIR_MAX_OWNERS:
            return None
        owner_placeholders = ",".join("?" for _ in owners)
        live_owners = tuple(
            int(row[0])
            for row in conn.execute(
                f"""
                SELECT node_id FROM summary_nodes
                WHERE session_id=? AND source_type='messages'
                  AND node_id IN ({owner_placeholders})
                ORDER BY node_id
                """,
                (str(session_id), *owners),
            ).fetchall()
        )
        if live_owners != owners:
            return None

        visited = set(owners)
        queue = list(owners)
        edge_set: set[tuple[int, int]] = set()
        cursor = 0
        while cursor < len(queue):
            child_id = queue[cursor]
            cursor += 1
            remaining_edges = _PARTIAL_REPAIR_MAX_TOPOLOGY_EDGES - len(edge_set)
            if remaining_edges <= 0:
                return None
            rows = conn.execute(
                """
                SELECT edge.parent_node_id, edge.child_node_id
                FROM lcm_compaction_node_edges AS edge
                JOIN summary_nodes AS child
                  ON child.node_id=edge.child_node_id
                 AND child.session_id=edge.session_id
                JOIN summary_nodes AS parent
                  ON parent.node_id=edge.parent_node_id
                 AND parent.session_id=edge.session_id
                 AND parent.depth>child.depth
                WHERE edge.session_id=? AND edge.child_node_id=?
                ORDER BY edge.parent_node_id
                LIMIT ?
                """,
                (
                    str(session_id),
                    child_id,
                    min(remaining_edges, ancestry_cap) + 1,
                ),
            ).fetchall()
            if len(rows) > remaining_edges:
                return None
            for row in rows:
                parent_id = int(row["parent_node_id"])
                edge_set.add((parent_id, int(row["child_node_id"])))
                if len(edge_set) > _PARTIAL_REPAIR_MAX_TOPOLOGY_EDGES:
                    return None
                if parent_id not in visited:
                    visited.add(parent_id)
                    if len(visited) > ancestry_cap:
                        return None
                    queue.append(parent_id)

        ancestry_ids = tuple(sorted(visited))
        placeholders = ",".join("?" for _ in ancestry_ids)
        node_rows = conn.execute(
            f"""
            SELECT node_id, session_id, depth, summary, token_count,
                   source_token_count, source_ids, source_type, created_at,
                   earliest_at, latest_at, expand_hint
            FROM summary_nodes
            WHERE session_id=? AND node_id IN ({placeholders})
            ORDER BY node_id
            """,
            (str(session_id), *ancestry_ids),
        ).fetchall()
        if len(node_rows) != len(ancestry_ids):
            return None
        edges = tuple(sorted(edge_set))
        hidden_children = {child_id for _parent_id, child_id in edges}
        root_ids = tuple(
            node_id for node_id in ancestry_ids if node_id not in hidden_children
        )
        if not root_ids or len(root_ids) > root_cap:
            return None
        structural_nodes = [
            {
                "node_id": int(row["node_id"]),
                "session_id": str(row["session_id"] or ""),
                "depth": int(row["depth"] or 0),
                "source_type": str(row["source_type"] or ""),
                "source_ids": list(_load_ids(row["source_ids"])),
            }
            for row in node_rows
        ]
        roots = [row for row in node_rows if int(row["node_id"]) in set(root_ids)]
        return {
            "ancestry_ids": ancestry_ids,
            "root_ids": root_ids,
            "topology_hash": _canonical_hash(
                {"nodes": structural_nodes, "edges": [list(edge) for edge in edges]}
            ),
            "root_identity_hash": BackgroundCompactionStore._repair_root_identity_hash(
                roots
            ),
            "root_rows": roots,
        }

    def _bind_partial_atomic_repair(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
    ) -> CompactionJob | None:
        """Bind exact live roots/topology to a claimed repair before provider I/O."""
        with self._write_transaction() as conn:
            current = float(self._now())
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?",
                (str(job_id),),
            ).fetchone()

            def supersede(reason: str) -> None:
                if row is None:
                    return
                conn.execute(
                    """
                    UPDATE lcm_compaction_jobs
                    SET state='superseded', completion_reason=?, updated_at=?,
                        lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL
                    WHERE job_id=? AND state='running' AND lease_owner=?
                      AND lease_generation=?
                    """,
                    (reason, current, str(job_id), str(owner), int(generation)),
                )
                conn.execute(
                    """
                    DELETE FROM lcm_compaction_provider_leases
                    WHERE job_id=? AND lease_owner=? AND lease_generation=?
                    """,
                    (str(job_id), str(owner), int(generation)),
                )
                self._record_event_under_tx(
                    conn,
                    "prepared_summary_superseded",
                    conversation_id=str(row["conversation_id"] or ""),
                    session_id=str(row["session_id"] or ""),
                    job_id=str(job_id),
                    detail=reason,
                    created_at=current,
                )

            permit = conn.execute(
                """
                SELECT 1 FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                  AND lease_expires_at>?
                """,
                (str(job_id), str(owner), int(generation), current),
            ).fetchone()
            if (
                row is None
                or str(row["job_kind"] or "") != _PARTIAL_REPAIR_JOB_KIND
                or str(row["state"] or "") != "running"
                or str(row["lease_owner"] or "") != str(owner)
                or int(row["lease_generation"] or 0) != int(generation)
                or float(row["lease_expires_at"] or 0.0) <= current
                or permit is None
            ):
                return None
            session = str(row["session_id"] or "")
            conversation = str(row["conversation_id"] or "")
            frontier = int(row["expected_frontier_store_id"] or 0)
            group_ids = _load_ids(row["repair_group_store_ids"])
            covered_ids = _load_ids(row["repair_covered_store_ids"])
            uncovered_ids = _load_ids(row["repair_uncovered_store_ids"])
            lifecycle = conn.execute(
                """
                SELECT current_session_id, current_frontier_store_id
                FROM lcm_lifecycle_state WHERE conversation_id=?
                """,
                (conversation,),
            ).fetchone()
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                current=current,
                bootstrap=False,
            )
            if (
                lifecycle is None
                or str(lifecycle["current_session_id"] or "") != session
                or int(lifecycle["current_frontier_store_id"] or 0) != frontier
                or fence is None
                or int(fence["publishable"] or 0) != 1
                or int(fence["generation"] or 0)
                != int(row["session_fence_generation"] or 0)
                or not group_ids
                or group_ids != _load_ids(row["source_store_ids"])
            ):
                supersede("partial_repair_frontier_or_session_changed")
                return None
            raw_messages = self._load_source_range(
                conn,
                session_id=session,
                conversation_id=conversation,
                frontier=frontier,
                watermark=group_ids[-1],
            )
            groups = self._atomic_groups(raw_messages)
            raw_ids = tuple(int(message["store_id"]) for message in raw_messages)
            raw_identity = self._source_identity_hash(raw_messages)
            if (
                raw_ids != group_ids
                or len(groups) != 1
                or tuple(int(message["store_id"]) for message in groups[0])
                != group_ids
                or not self._atomic_group_is_complete(groups[0])
                or raw_identity != str(row["source_identity_hash"] or "")
                or raw_identity != str(row["repair_group_identity_hash"] or "")
            ):
                supersede("partial_repair_source_range_changed")
                return None
            ownership = self._live_owner_map(
                conn,
                session_id=session,
                source_ids=group_ids,
            )
            if ownership is None:
                supersede("partial_repair_owner_limit_exceeded")
                return None
            owner_map, owner_ids = ownership
            if (
                tuple(item for item in group_ids if owner_map[item]) != covered_ids
                or tuple(item for item in group_ids if not owner_map[item]) != uncovered_ids
                or owner_ids != _load_ids(row["repair_owner_node_ids"])
                or self._owner_map_json(owner_map)
                != str(row["repair_owner_map"] or "{}")
                or not self._repair_coverage_elections_valid(
                    conn,
                    session_id=session,
                    owner_map=owner_map,
                    covered_ids=covered_ids,
                    uncovered_ids=uncovered_ids,
                )
            ):
                supersede("partial_repair_ownership_changed")
                return None
            topology = self._resolve_partial_repair_topology(
                conn,
                session_id=session,
                owner_ids=owner_ids,
            )
            if topology is None:
                supersede("partial_repair_topology_limit_or_invalid")
                return None
            dag_epoch = self._session_dag_epoch(conn, session)
            old_binding = (
                _load_ids(row["repair_root_node_ids"]),
                _load_ids(row["repair_ancestry_node_ids"]),
                str(row["repair_topology_hash"] or ""),
                str(row["repair_root_identity_hash"] or ""),
                int(row["repair_bound_dag_epoch"] or 0),
            )
            new_binding = (
                tuple(topology["root_ids"]),
                tuple(topology["ancestry_ids"]),
                str(topology["topology_hash"]),
                str(topology["root_identity_hash"]),
                dag_epoch,
            )
            progress_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_compaction_prepared_nodes
                    WHERE job_id=?
                    """,
                    (str(job_id),),
                ).fetchone()[0]
            )
            if progress_count and old_binding != new_binding:
                conn.execute(
                    "DELETE FROM lcm_compaction_prepared_nodes WHERE job_id=?",
                    (str(job_id),),
                )
                conn.execute(
                    """
                    UPDATE lcm_compaction_jobs
                    SET prepared_node_count=0, updated_at=? WHERE job_id=?
                    """,
                    (current, str(job_id)),
                )
            updated = conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET repair_root_node_ids=?, repair_ancestry_node_ids=?,
                    repair_topology_hash=?, repair_root_identity_hash=?,
                    repair_bound_dag_epoch=?, updated_at=?
                WHERE job_id=? AND state='running' AND lease_owner=?
                  AND lease_generation=? AND lease_expires_at>?
                """,
                (
                    _json_ids(topology["root_ids"]),
                    _json_ids(topology["ancestry_ids"]),
                    str(topology["topology_hash"]),
                    str(topology["root_identity_hash"]),
                    dag_epoch,
                    current,
                    str(job_id),
                    str(owner),
                    int(generation),
                    current,
                ),
            )
            if updated.rowcount != 1:
                return None
            return self._job_from_row(
                conn.execute(
                    "SELECT * FROM lcm_compaction_jobs WHERE job_id=?",
                    (str(job_id),),
                ).fetchone()
            )

    def _enqueue_partial_atomic_repair(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        group_store_ids: Sequence[int],
        expected_group_identity_hash: str,
        policy_fingerprint: str,
        priority: int,
        write_timeout_ms: int | None = 100,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> CompactionJob | None:
        """Atomically revalidate and enqueue one mixed-ownership atomic group."""
        conversation = str(conversation_id or "").strip()
        session = str(session_id or "").strip()
        frontier = max(0, int(expected_frontier_store_id or 0))
        captured_ids = tuple(int(item) for item in group_store_ids)
        if (
            not conversation
            or not session
            or not captured_ids
            or len(captured_ids) > _PARTIAL_REPAIR_MAX_GROUP_ROWS
            or tuple(sorted(set(captured_ids))) != captured_ids
        ):
            return None
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now())
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                current=current,
                bootstrap=True,
            )
            if (
                fence is None
                or str(fence["session_id"] or "") != session
                or int(fence["publishable"] or 0) != 1
            ):
                return None
            lifecycle = conn.execute(
                """
                SELECT current_session_id, current_frontier_store_id
                FROM lcm_lifecycle_state WHERE conversation_id=?
                """,
                (conversation,),
            ).fetchone()
            if (
                lifecycle is None
                or str(lifecycle["current_session_id"] or "") != session
                or int(lifecycle["current_frontier_store_id"] or 0) != frontier
            ):
                return None
            raw_messages = self._load_source_range(
                conn,
                session_id=session,
                conversation_id=conversation,
                frontier=frontier,
                watermark=captured_ids[-1],
            )
            raw_ids = tuple(int(message["store_id"]) for message in raw_messages)
            groups = self._atomic_groups(raw_messages)
            if (
                raw_ids != captured_ids
                or len(groups) != 1
                or tuple(int(item["store_id"]) for item in groups[0]) != captured_ids
                or not self._atomic_group_is_complete(groups[0])
            ):
                return None
            identity_hash = self._source_identity_hash(raw_messages)
            if identity_hash != str(expected_group_identity_hash or ""):
                return None
            ownership = self._live_owner_map(
                conn,
                session_id=session,
                source_ids=captured_ids,
            )
            if ownership is None:
                return None
            owner_map, owner_ids = ownership
            covered_ids = tuple(
                store_id for store_id in captured_ids if owner_map[store_id]
            )
            uncovered_ids = tuple(
                store_id for store_id in captured_ids if not owner_map[store_id]
            )
            if not covered_ids or not uncovered_ids:
                return None
            for store_id in covered_ids:
                elected_owner = owner_map[store_id][0]
                owner_row = conn.execute(
                    """
                    SELECT created_at FROM summary_nodes
                    WHERE node_id=? AND session_id=? AND source_type='messages'
                    """,
                    (elected_owner, session),
                ).fetchone()
                if owner_row is None:
                    return None
                conn.execute(
                    """
                    INSERT INTO lcm_compaction_source_coverage(
                        session_id, store_id, publication_id,
                        publication_kind, published_at
                    ) VALUES(?,?,?,?,?)
                    ON CONFLICT(session_id, store_id) DO UPDATE SET
                        publication_id=excluded.publication_id,
                        publication_kind=excluded.publication_kind,
                        published_at=excluded.published_at
                    """,
                    (
                        session,
                        store_id,
                        f"summary-node:{elected_owner}",
                        "canonical",
                        float(owner_row["created_at"] or 0.0),
                    ),
                )
            if uncovered_ids:
                uncovered_placeholders = ",".join("?" for _ in uncovered_ids)
                conn.execute(
                    f"""
                    DELETE FROM lcm_compaction_source_coverage
                    WHERE session_id=?
                      AND store_id IN ({uncovered_placeholders})
                    """,
                    (session, *uncovered_ids),
                )
            existing = conn.execute(
                """
                SELECT * FROM lcm_compaction_jobs
                WHERE conversation_id=?
                  AND state IN ('queued','running','prepared','retry')
                LIMIT 1
                """,
                (conversation,),
            ).fetchone()
            if existing is not None:
                same_capture_scope = (
                    str(existing["session_id"] or "") == session
                    and int(existing["session_fence_generation"] or 0)
                    == int(fence["generation"] or 0)
                    and int(existing["expected_frontier_store_id"] or 0) == frontier
                    and str(existing["policy_fingerprint"] or "")
                    == str(policy_fingerprint or "")
                )
                if not same_capture_scope:
                    self._supersede_active_jobs_under_tx(
                        conn,
                        conversation_id=conversation,
                        reason="session_or_policy_changed_before_claim",
                        current=current,
                    )
                    existing = None
            if existing is not None:
                exact_existing = (
                    str(existing["job_kind"] or "") == _PARTIAL_REPAIR_JOB_KIND
                    and str(existing["session_id"] or "") == session
                    and int(existing["expected_frontier_store_id"] or 0) == frontier
                    and int(existing["captured_watermark_store_id"] or 0)
                    == captured_ids[-1]
                    and _load_ids(existing["source_store_ids"]) == captured_ids
                    and str(existing["source_identity_hash"] or "")
                    == identity_hash
                    and _load_ids(existing["repair_group_store_ids"]) == captured_ids
                    and str(existing["repair_group_identity_hash"] or "")
                    == identity_hash
                    and _load_ids(existing["repair_covered_store_ids"])
                    == covered_ids
                    and _load_ids(existing["repair_uncovered_store_ids"])
                    == uncovered_ids
                    and _load_ids(existing["repair_owner_node_ids"]) == owner_ids
                    and str(existing["repair_owner_map"] or "{}")
                    == self._owner_map_json(owner_map)
                )
                if exact_existing:
                    return self._job_from_row(existing)
                self._supersede_active_jobs_under_tx(
                    conn,
                    conversation_id=conversation,
                    reason="partial_repair_source_range_changed",
                    current=current,
                )
                existing = None

            disposition = self._matching_range_disposition_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                session_fence_generation=int(fence["generation"] or 0),
                expected_frontier_store_id=frontier,
                captured_watermark_store_id=captured_ids[-1],
                policy_fingerprint=policy_fingerprint,
                source_identity_hash=identity_hash,
            )
            if disposition is not None:
                if not self._range_disposition_blocks_requeue_under_tx(
                    conn,
                    disposition,
                    current=current,
                ):
                    disposition = None
            if disposition is not None:
                conn.execute(
                    """
                    UPDATE lcm_compaction_range_dispositions
                    SET last_seen_at=?
                    WHERE conversation_id=? AND session_id=?
                      AND session_fence_generation=?
                      AND expected_frontier_store_id=?
                      AND captured_watermark_store_id=?
                      AND policy_fingerprint=? AND source_identity_hash=?
                    """,
                    (
                        current,
                        conversation,
                        session,
                        int(fence["generation"] or 0),
                        frontier,
                        captured_ids[-1],
                        str(policy_fingerprint or ""),
                        identity_hash,
                    ),
                )
                return None

            source_tokens = sum(
                self._message_tokens(message) for message in raw_messages
            )
            job_id = uuid.uuid4().hex
            owner_map_json = self._owner_map_json(owner_map)
            dag_epoch = self._session_dag_epoch(conn, session)
            conn.execute(
                """
                INSERT INTO lcm_compaction_jobs(
                    job_id, conversation_id, session_id,
                    session_fence_generation, expected_frontier_store_id,
                    captured_watermark_store_id, source_store_ids,
                    source_token_count, source_identity_hash, state, priority,
                    created_at, updated_at, policy_fingerprint,
                    expected_node_count, job_kind,
                    repair_group_store_ids, repair_group_identity_hash,
                    repair_covered_store_ids, repair_uncovered_store_ids,
                    repair_owner_node_ids, repair_owner_map,
                    repair_capture_dag_epoch
                ) VALUES(?,?,?,?,?,?,?,?,?,'queued',?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    conversation,
                    session,
                    int(fence["generation"] or 0),
                    frontier,
                    captured_ids[-1],
                    _json_ids(captured_ids),
                    source_tokens,
                    identity_hash,
                    max(_PARTIAL_REPAIR_PRIORITY, int(priority or 0)),
                    current,
                    current,
                    str(policy_fingerprint or ""),
                    1,
                    _PARTIAL_REPAIR_JOB_KIND,
                    _json_ids(captured_ids),
                    identity_hash,
                    _json_ids(covered_ids),
                    _json_ids(uncovered_ids),
                    _json_ids(owner_ids),
                    owner_map_json,
                    dag_epoch,
                ),
            )
            self._record_lifecycle_debt_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                expected_frontier_store_id=frontier,
                debt_tokens=source_tokens,
                current=current,
            )
            self._record_event_under_tx(
                conn,
                "partial_atomic_repair_pending",
                conversation_id=conversation,
                session_id=session,
                job_id=job_id,
                detail=(
                    f"group={captured_ids[0]}-{captured_ids[-1]} "
                    f"covered={len(covered_ids)} uncovered={len(uncovered_ids)} "
                    f"owners={len(owner_ids)}"
                ),
                created_at=current,
            )
            return self._job_from_row(
                conn.execute(
                    "SELECT * FROM lcm_compaction_jobs WHERE job_id=?",
                    (job_id,),
                ).fetchone()
            )

    @staticmethod
    def _message_tokens(message: Mapping[str, Any]) -> int:
        durable = max(0, int(message.get("token_estimate") or 0))
        return durable or max(1, int(count_message_tokens(dict(message))))

    def _chunk_messages(
        self, messages: Sequence[dict[str, Any]]
    ) -> list[SourceChunk]:
        base_limit = max(
            1, int(getattr(self.config, "leaf_chunk_tokens", 20_000) or 20_000)
        )
        token_limit = base_limit
        if (
            bool(getattr(self.config, "dynamic_leaf_chunk_enabled", False))
            and not bool(
                getattr(self.config, "cache_friendly_condensation_enabled", False)
            )
        ):
            ceiling = max(
                base_limit,
                int(getattr(self.config, "dynamic_leaf_chunk_max", 40_000) or 40_000),
            )
            total_tokens = sum(self._message_tokens(message) for message in messages)
            while token_limit < ceiling and total_tokens > token_limit * 2:
                token_limit = min(ceiling, token_limit * 2)
        chunks: list[SourceChunk] = []
        current: list[dict[str, Any]] = []
        current_tokens = 0
        for group in self._atomic_groups(messages):
            group_tokens = sum(self._message_tokens(message) for message in group)
            if current and current_tokens + group_tokens > token_limit:
                chunks.append(
                    SourceChunk(
                        ordinal=len(chunks),
                        source_store_ids=tuple(int(item["store_id"]) for item in current),
                        source_token_count=current_tokens,
                        messages=tuple(current),
                    )
                )
                current = []
                current_tokens = 0
            current.extend(group)
            current_tokens += group_tokens
        if current:
            chunks.append(
                SourceChunk(
                    ordinal=len(chunks),
                    source_store_ids=tuple(int(item["store_id"]) for item in current),
                    source_token_count=current_tokens,
                    messages=tuple(current),
                )
            )
        return chunks

    def _coalesce_disposition_extension_chunks(
        self,
        chunks: Sequence[SourceChunk],
        *,
        max_chunks: int,
    ) -> list[SourceChunk] | None:
        """Fit one changed terminal range into its durable provider-pass budget.

        ``leaf_chunk_tokens`` remains the ordinary target.  A terminal prefix
        may consume every configured pass, though, so one newly eligible
        adjacent group would otherwise be hidden behind the exact disposition
        forever.  Merge only adjacent chunks, choose the smallest merge first,
        and never create a provider input larger than twice the configured
        target.  Atomic groups and exact source order therefore remain intact.
        """
        budget = max(1, int(max_chunks or 0))
        token_ceiling = max(
            1,
            int(getattr(self.config, "leaf_chunk_tokens", 20_000) or 20_000),
        ) * 2
        merged = list(chunks)
        while len(merged) > budget:
            choices = [
                (
                    int(left.source_token_count) + int(right.source_token_count),
                    index,
                )
                for index, (left, right) in enumerate(zip(merged, merged[1:]))
                if int(left.source_token_count) + int(right.source_token_count)
                <= token_ceiling
            ]
            if not choices:
                return None
            _combined_tokens, index = min(choices)
            left = merged[index]
            right = merged[index + 1]
            merged[index : index + 2] = [
                SourceChunk(
                    ordinal=0,
                    source_store_ids=(
                        *left.source_store_ids,
                        *right.source_store_ids,
                    ),
                    source_token_count=(
                        int(left.source_token_count)
                        + int(right.source_token_count)
                    ),
                    messages=(*left.messages, *right.messages),
                )
            ]
        return [
            SourceChunk(
                ordinal=index,
                source_store_ids=chunk.source_store_ids,
                source_token_count=chunk.source_token_count,
                messages=chunk.messages,
                serialized_text=chunk.serialized_text,
            )
            for index, chunk in enumerate(merged)
        ]

    def _job_extends_terminal_disposition_under_tx(
        self,
        conn: sqlite3.Connection,
        job: CompactionJob,
        messages: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Verify the durable terminal-prefix proof used for bounded replay."""
        extendable_reasons = tuple(sorted(_EXTENDABLE_RANGE_DISPOSITION_REASONS))
        reason_placeholders = ",".join("?" for _ in extendable_reasons)
        dispositions = conn.execute(
            f"""
            SELECT * FROM lcm_compaction_range_dispositions
            WHERE conversation_id=? AND session_id=?
              AND session_fence_generation=?
              AND expected_frontier_store_id=?
              AND captured_watermark_store_id<?
              AND policy_fingerprint=?
              AND resolved_at IS NULL
              AND reason_code IN ({reason_placeholders})
            ORDER BY captured_watermark_store_id DESC
            LIMIT 1
            """,
            (
                job.conversation_id,
                job.session_id,
                job.session_fence_generation,
                job.expected_frontier_store_id,
                job.captured_watermark_store_id,
                job.policy_fingerprint,
                *extendable_reasons,
            ),
        ).fetchall()
        if not dispositions:
            return False
        disposition = dispositions[0]
        watermark = int(disposition["captured_watermark_store_id"] or 0)
        prefix = [
            message
            for message in messages
            if int(message.get("store_id") or 0) <= watermark
        ]
        prefix_matches = (
            bool(prefix)
            and int(prefix[-1].get("store_id") or 0) == watermark
            and self._source_identity_hash(prefix)
            == str(disposition["source_identity_hash"] or "")
        )
        if not prefix_matches:
            return False
        return (
            conn.execute(
                f"""
                SELECT 1 FROM messages
                WHERE session_id=? AND store_id>? AND store_id<=?
                  AND (conversation_id=? OR conversation_id=''
                       OR conversation_id IS NULL)
                  AND {self._message_arrival_sql}>?
                ORDER BY store_id
                LIMIT 1
                """,
                (
                    job.session_id,
                    watermark,
                    job.captured_watermark_store_id,
                    job.conversation_id,
                    self._range_disposition_capture_started_at(disposition),
                ),
            ).fetchone()
            is not None
        )

    def _summary_candidate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        session_id: str,
        preserve_leading_system: bool = False,
        resolve_externalized: bool = True,
    ) -> list[dict[str, Any]]:
        candidate_messages = list(messages)
        if (
            preserve_leading_system
            and candidate_messages
            and str(candidate_messages[0].get("role") or "") == "system"
        ):
            candidate_messages = candidate_messages[1:]
        return filter_summary_messages(
            candidate_messages,
            config=self.config,
            session_id=session_id,
            hermes_home=self._hermes_home,
            resolve_externalized=resolve_externalized,
        )

    def adopt_contiguous_canonical_prefix(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        expected_session_fence_generation: int,
        max_rows: int = _CANONICAL_PREFIX_ADOPTION_ROW_LIMIT,
        now: float | None = None,
        write_timeout_ms: int | None = 100,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> int:
        """Advance across only the oldest already-canonical raw prefix.

        Legacy foreground compaction committed a DAG node and its lifecycle
        frontier in separate transactions.  A process exit between those
        commits can therefore leave canonical message coverage above a stale
        frontier.  New preparation must not summarize that coverage again,
        but it must not jump over an uncovered row either.

        This bounded reconciliation uses only the normalized coverage indexes.
        It validates the exact elected node owner, stops at the first coverage
        gap or partial tool-call group, and advances the lifecycle frontier in
        the same short transaction that supersedes work captured against the
        old frontier.  Raw messages and canonical nodes are never modified.

        The returned value is the frontier adopted by this call, or the
        unchanged expected frontier when any lifecycle/fence CAS precondition
        fails or no safe prefix exists.
        """
        conversation = str(conversation_id or "").strip()
        session = str(session_id or "").strip()
        frontier = max(0, int(expected_frontier_store_id or 0))
        generation = max(0, int(expected_session_fence_generation or 0))
        if not conversation or not session or generation <= 0:
            return frontier
        row_limit = max(
            1,
            min(
                _CANONICAL_PREFIX_ADOPTION_ROW_LIMIT,
                int(max_rows or _CANONICAL_PREFIX_ADOPTION_ROW_LIMIT),
            ),
        )

        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now() if now is None else now)
            lifecycle = conn.execute(
                """
                SELECT current_session_id, current_frontier_store_id
                FROM lcm_lifecycle_state WHERE conversation_id=?
                """,
                (conversation,),
            ).fetchone()
            if (
                lifecycle is None
                or str(lifecycle["current_session_id"] or "") != session
                or int(lifecycle["current_frontier_store_id"] or 0) != frontier
            ):
                return frontier
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                current=current,
                bootstrap=False,
            )
            if (
                fence is None
                or str(fence["session_id"] or "") != session
                or int(fence["publishable"] or 0) != 1
                or int(fence["generation"] or 0) != generation
            ):
                return frontier

            # One sentinel row proves that a tool-call group crossing the
            # advertised bound is not partially adopted.  Canonical ownership
            # itself is resolved exclusively through the two normalized maps;
            # this path never scans summary_nodes or decodes source_ids JSON.
            rows = conn.execute(
                """
                SELECT m.store_id, m.session_id, m.conversation_id, m.role,
                       m.content, m.tool_call_id, m.tool_calls, m.tool_name,
                       m.timestamp, m.token_estimate,
                       CASE
                         WHEN coverage.store_id IS NOT NULL AND EXISTS (
                           SELECT 1
                           FROM lcm_compaction_node_sources AS owner
                           WHERE owner.session_id=coverage.session_id
                             AND owner.store_id=coverage.store_id
                             AND coverage.publication_id IN (
                               'summary-node:' || owner.node_id,
                               'legacy-node:' || owner.node_id
                             )
                         ) THEN 1 ELSE 0
                       END AS canonical_covered
                FROM messages AS m
                LEFT JOIN lcm_compaction_source_coverage AS coverage
                  ON coverage.session_id=? AND coverage.store_id=m.store_id
                WHERE m.session_id=? AND m.store_id>?
                  AND (
                    m.conversation_id=? OR m.conversation_id=''
                    OR m.conversation_id IS NULL
                  )
                ORDER BY m.store_id
                LIMIT ?
                """,
                (
                    session,
                    session,
                    frontier,
                    conversation,
                    row_limit + 1,
                ),
            ).fetchall()
            if not rows:
                return frontier

            candidates: list[dict[str, Any]] = []
            for row in rows:
                message = self._message_dict(row)
                message["_canonical_covered"] = bool(row["canonical_covered"])
                candidates.append(message)

            adopted: list[dict[str, Any]] = []
            for group in self._atomic_groups(candidates):
                if len(adopted) + len(group) > row_limit:
                    break
                if not all(bool(message["_canonical_covered"]) for message in group):
                    break
                opening = group[0]
                declared_call_ids = (
                    self._tool_call_ids(opening)
                    if str(opening.get("role") or "") == "assistant"
                    else set()
                )
                if declared_call_ids:
                    observed_result_ids = {
                        str(message.get("tool_call_id") or "")
                        for message in group[1:]
                        if str(message.get("role") or "") == "tool"
                        and str(message.get("tool_call_id") or "")
                    }
                    if declared_call_ids - observed_result_ids:
                        break
                adopted.extend(group)
            if not adopted:
                return frontier

            adopted_frontier = int(adopted[-1]["store_id"])
            adopted_tokens = sum(self._message_tokens(message) for message in adopted)
            advanced = conn.execute(
                """
                UPDATE lcm_lifecycle_state
                SET current_frontier_store_id=?,
                    debt_size_estimate=MAX(0, debt_size_estimate-?),
                    debt_kind=CASE
                      WHEN MAX(0, debt_size_estimate-?)=0 THEN NULL
                      ELSE debt_kind
                    END,
                    debt_updated_at=?, updated_at=?
                WHERE conversation_id=? AND current_session_id=?
                  AND current_frontier_store_id=?
                  AND EXISTS (
                    SELECT 1 FROM lcm_compaction_session_fences AS fence
                    WHERE fence.conversation_id=? AND fence.session_id=?
                      AND fence.publishable=1 AND fence.generation=?
                  )
                """,
                (
                    adopted_frontier,
                    adopted_tokens,
                    adopted_tokens,
                    current,
                    current,
                    conversation,
                    session,
                    frontier,
                    conversation,
                    session,
                    generation,
                ),
            )
            if advanced.rowcount != 1:
                return frontier
            self._supersede_active_jobs_under_tx(
                conn,
                conversation_id=conversation,
                session_id=session,
                reason="canonical_prefix_adopted",
                current=current,
            )
            self._record_event_under_tx(
                conn,
                "canonical_prefix_adopted",
                conversation_id=conversation,
                session_id=session,
                detail=f"rows={len(adopted)} frontier={adopted_frontier}",
                created_at=current,
            )
            return adopted_frontier

    def capture_and_enqueue(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        active_tokens: int,
        effective_context_window: int,
        policy_fingerprint: str,
        priority: int = 0,
        lock_timeout_seconds: float | None = None,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        busy_timeout_ms: int | None = None,
    ) -> CompactionJob | None:
        """Capture a bounded immutable raw prefix and enqueue it.

        The foreground path reads only a bounded protected suffix plus a
        bounded oldest prefix.  In particular it never materializes or hashes
        the entire post-frontier history.  Externalized payload bytes are not
        resolved here; durable marker rows are the source of truth captured by
        the job.
        """
        clock = deadline_clock or time.monotonic

        def check_deadline() -> None:
            if deadline is not None and float(clock()) >= float(deadline):
                raise TimeoutError("background compaction capture deadline exceeded")

        check_deadline()
        frontier = max(0, int(expected_frontier_store_id or 0))
        fresh_tail_count = max(
            0, int(getattr(self.config, "fresh_tail_count", 0) or 0)
        )
        fresh_tail_max_tokens = max(
            0, int(getattr(self.config, "fresh_tail_max_tokens", 0) or 0)
        )
        effective_tail_count = (
            max(1, fresh_tail_count)
            if fresh_tail_max_tokens > 0
            else fresh_tail_count
        )
        backlog_trigger = max(
            1,
            int(
                getattr(
                    self.config,
                    "background_compaction_backlog_tokens",
                    32_000,
                )
                or 32_000
            ),
        )
        # An unusually large configured protected tail cannot be inspected
        # within the foreground read budget.  Deferring preparation is safer
        # than guessing at a boundary and splitting a tool group.
        if effective_tail_count >= _CAPTURE_TAIL_ROW_LIMIT:
            return None

        scope = (
            "session_id=? AND store_id>? AND store_id<=? "
            "AND (conversation_id=? OR conversation_id='' OR conversation_id IS NULL)"
        )
        aliased_scope = (
            "m.session_id=? AND m.store_id>? AND m.store_id<=? "
            "AND (m.conversation_id=? OR m.conversation_id='' "
            "OR m.conversation_id IS NULL)"
        )
        with self._bounded_read_connection(
            lock_timeout_seconds=lock_timeout_seconds,
            deadline=deadline,
            deadline_clock=clock,
            busy_timeout_ms=busy_timeout_ms,
        ) as conn:
            active = conn.execute(
                """
                SELECT * FROM lcm_compaction_jobs
                WHERE conversation_id=?
                  AND state IN ('queued','running','prepared','retry')
                LIMIT 1
                """,
                (str(conversation_id),),
            ).fetchone()
            fence = conn.execute(
                """
                SELECT session_id, generation, publishable
                FROM lcm_compaction_session_fences WHERE conversation_id=?
                """,
                (str(conversation_id),),
            ).fetchone()
            if fence is not None and (
                str(fence["session_id"] or "") != str(session_id)
                or int(fence["publishable"] or 0) != 1
            ):
                return None
            same_active_scope = (
                active is not None
                and str(active["state"] or "") in {"running", "prepared"}
                and str(active["session_id"] or "") == str(session_id)
                and (
                    fence is None
                    or int(active["session_fence_generation"] or 0)
                    == int(fence["generation"] or 0)
                )
                and int(active["expected_frontier_store_id"] or 0) == frontier
                and str(active["policy_fingerprint"] or "")
                == str(policy_fingerprint or "")
            )
            if same_active_scope:
                active_kind = str(active["job_kind"] or "standard")
                active_ids = _load_ids(active["source_store_ids"])
                reuse_active = False
                if active_kind == _LOSSLESS_RAW_PROJECTION_JOB_KIND:
                    active_watermark = int(
                        active["captured_watermark_store_id"] or 0
                    )
                    active_messages = self._load_source_range(
                        conn,
                        session_id=str(session_id),
                        conversation_id=str(conversation_id),
                        frontier=frontier,
                        watermark=active_watermark,
                    )
                    reuse_active = (
                        bool(active_ids)
                        and active_ids[-1] == active_watermark
                        and tuple(
                            int(message["store_id"])
                            for message in active_messages
                        )
                        == active_ids
                        and self._source_identity_hash(active_messages)
                        == str(active["source_identity_hash"] or "")
                        and not self._source_ids_already_canonical(
                            conn,
                            str(session_id),
                            active_ids,
                        )
                    )
                elif active_kind == "standard":
                    minimum_provider_tokens = max(
                        1,
                        int(
                            getattr(
                                self.config,
                                "background_compaction_min_provider_source_tokens",
                                256,
                            )
                            or 256
                        ),
                    )
                    reuse_active = (
                        bool(active_ids)
                        and int(active["source_token_count"] or 0)
                        >= minimum_provider_tokens
                        and not self._source_ids_already_canonical(
                            conn,
                            str(session_id),
                            active_ids,
                        )
                    )
                # Partial repair jobs are deliberately re-derived from their
                # small atomic group.  Its source identity and canonical-owner
                # topology can change without changing the outer capture scope.
                if reuse_active:
                    return self._job_from_row(active)

            snapshot_row = conn.execute(
                """
                SELECT MAX(store_id) FROM messages
                WHERE session_id=? AND store_id>?
                  AND (conversation_id=? OR conversation_id='' OR conversation_id IS NULL)
                """,
                (str(session_id), frontier, str(conversation_id)),
            ).fetchone()
            snapshot_max_id = int(snapshot_row[0] or 0) if snapshot_row else 0
            if snapshot_max_id <= frontier:
                return None

            eligible_upper = snapshot_max_id
            if effective_tail_count > 0:
                tail_limit = min(
                    _CAPTURE_TAIL_ROW_LIMIT,
                    effective_tail_count + min(64, _CAPTURE_GROUP_LOOKAHEAD),
                )
                tail_rows = conn.execute(
                    f"""
                    SELECT store_id, session_id, conversation_id, role, content,
                           tool_call_id, tool_calls, tool_name, timestamp, token_estimate
                    FROM messages
                    WHERE {scope}
                    ORDER BY store_id DESC
                    LIMIT ?
                    """,
                    (
                        str(session_id),
                        frontier,
                        snapshot_max_id,
                        str(conversation_id),
                        tail_limit,
                    ),
                ).fetchall()
                tail_messages = [
                    self._message_dict(row) for row in reversed(tail_rows)
                ]
                if not tail_messages:
                    return None
                boundary = resolve_fresh_tail_boundary(
                    tail_messages,
                    fresh_tail_count=fresh_tail_count,
                    fresh_tail_max_tokens=fresh_tail_max_tokens,
                )
                if (
                    boundary.start < len(tail_messages)
                    and str(tail_messages[boundary.start].get("role") or "") == "tool"
                    and not boundary.tool_group_extended
                ):
                    # The opening assistant is beyond our bounded look-behind.
                    # Do not risk compacting half of the tool group.
                    return None
                protected_start_id = int(
                    tail_messages[boundary.start]["store_id"]
                )
                eligible_upper = protected_start_id - 1

            if eligible_upper <= frontier:
                return None

            backlog_row = conn.execute(
                f"""
                SELECT COALESCE(SUM(MAX(0, COALESCE(token_estimate, 0))), 0)
                FROM (
                    SELECT token_estimate FROM messages
                    WHERE {scope}
                      AND COALESCE(token_estimate, 0)>0
                    ORDER BY store_id
                    LIMIT ?
                )
                """,
                (
                    str(session_id),
                    frontier,
                    eligible_upper,
                    str(conversation_id),
                    backlog_trigger,
                ),
            ).fetchone()
            source_backlog_tokens = int(backlog_row[0] or 0) if backlog_row else 0

            prepare_ratio = max(
                0.0,
                float(
                    getattr(
                        self.config,
                        "background_compaction_prepare_ratio",
                        0.55,
                    )
                    or 0.0
                ),
            )
            pressure_eligible = (
                int(effective_context_window or 0) > 0
                and int(active_tokens or 0)
                >= int(int(effective_context_window) * prepare_ratio)
            )
            if source_backlog_tokens < backlog_trigger and not pressure_eligible:
                return None

            prefix_query_limit = (
                _CAPTURE_ROW_LIMIT + _CAPTURE_GROUP_LOOKAHEAD + 1
            )
            rows = conn.execute(
                f"""
                SELECT m.store_id, m.session_id, m.conversation_id, m.role, m.content,
                       m.tool_call_id, m.tool_calls, m.tool_name, m.timestamp,
                       m.token_estimate,
                       CASE WHEN EXISTS (
                         SELECT 1
                         FROM lcm_compaction_node_sources AS owner
                         JOIN summary_nodes AS live
                           ON live.node_id=owner.node_id
                          AND live.session_id=owner.session_id
                          AND live.source_type='messages'
                         WHERE owner.session_id=m.session_id
                           AND owner.store_id=m.store_id
                       ) THEN 1 ELSE 0 END AS canonical_covered
                FROM messages AS m
                LEFT JOIN lcm_compaction_source_coverage AS coverage
                  ON coverage.session_id=? AND coverage.store_id=m.store_id
                WHERE {aliased_scope}
                ORDER BY m.store_id
                LIMIT ?
                """,
                (
                    str(session_id),
                    str(session_id),
                    frontier,
                    eligible_upper,
                    str(conversation_id),
                    prefix_query_limit,
                ),
            ).fetchall()
        check_deadline()
        if not rows:
            return None

        has_more = len(rows) > _CAPTURE_ROW_LIMIT + _CAPTURE_GROUP_LOOKAHEAD
        sentinel = self._message_dict(rows[-1]) if has_more else None
        prefix_rows = rows[:-1] if has_more else rows
        bounded = [self._message_dict(row) for row in prefix_rows]
        canonical_covered_ids = {
            int(row["store_id"])
            for row in prefix_rows
            if bool(row["canonical_covered"])
        }
        atomic_groups = self._atomic_groups(bounded)
        first_group = atomic_groups[0] if atomic_groups else []
        first_group_ids = tuple(
            int(message["store_id"]) for message in first_group
        )
        first_group_covered = tuple(
            store_id
            for store_id in first_group_ids
            if store_id in canonical_covered_ids
        )
        first_group_extends_sentinel = bool(
            first_group
            and len(first_group) == len(bounded)
            and sentinel is not None
            and str(first_group[0].get("role") or "") == "assistant"
            and str(sentinel.get("role") or "") == "tool"
            and (
                not str(sentinel.get("tool_call_id") or "")
                or str(sentinel.get("tool_call_id") or "")
                in self._tool_call_ids(first_group[0])
            )
        )
        if (
            first_group_ids
            and len(first_group_ids) <= _PARTIAL_REPAIR_MAX_GROUP_ROWS
            and first_group_covered
            and len(first_group_covered) < len(first_group_ids)
            and self._atomic_group_is_complete(first_group)
            and not first_group_extends_sentinel
        ):
            check_deadline()
            return self._enqueue_partial_atomic_repair(
                conversation_id=conversation_id,
                session_id=session_id,
                expected_frontier_store_id=frontier,
                group_store_ids=first_group_ids,
                expected_group_identity_hash=self._source_identity_hash(first_group),
                policy_fingerprint=policy_fingerprint,
                priority=priority,
                write_timeout_ms=100,
                lock_timeout_seconds=lock_timeout_seconds,
                write_deadline=deadline,
                deadline_clock=clock,
            )
        candidates: list[dict[str, Any]] = []
        offset = 0
        for group in atomic_groups:
            group_start = offset
            offset += len(group)
            if group_start >= _CAPTURE_ROW_LIMIT:
                break
            if any(
                int(message["store_id"]) in canonical_covered_ids
                for message in group
            ):
                # Never cross a later canonical island.  The uncovered prefix
                # can publish first; a subsequent bounded adoption then moves
                # across the island without duplicate provider work.
                break
            if (
                not has_more
                and eligible_upper == snapshot_max_id
                and offset == len(bounded)
                and group
                and str(group[0].get("role") or "") == "assistant"
            ):
                declared_call_ids = self._tool_call_ids(group[0])
                observed_result_ids = {
                    str(message.get("tool_call_id") or "")
                    for message in group[1:]
                    if str(message.get("role") or "") == "tool"
                }
                if declared_call_ids - observed_result_ids:
                    # The frozen snapshot ends while this assistant still has
                    # unresolved tool calls. A result appended immediately
                    # after capture must remain beside its opening assistant.
                    break
            if (
                has_more
                and offset == len(bounded)
                and group
                and str(group[0].get("role") or "") == "assistant"
                and sentinel is not None
                and str(sentinel.get("role") or "") == "tool"
            ):
                call_ids = self._tool_call_ids(group[0])
                result_id = str(sentinel.get("tool_call_id") or "")
                if not result_id or result_id in call_ids:
                    # The group extends beyond the bounded look-ahead.  Leave
                    # the entire group for a later job rather than split it.
                    break
            candidates.extend(group)
        if not candidates:
            return None
        check_deadline()
        summary_candidates = self._summary_candidate_messages(
            candidates,
            session_id=str(session_id),
            preserve_leading_system=frontier == 0,
            resolve_externalized=False,
        )
        available_chunks = self._chunk_messages(summary_candidates)
        max_passes = max(
            1, int(getattr(self.config, "background_compaction_max_passes", 4) or 4)
        )
        combine_prepared_leaves = bool(
            getattr(self.config, "cache_friendly_condensation_enabled", False)
            and len(available_chunks) > 1
            and max_passes >= 2
        )
        leaf_pass_budget = max_passes - 1 if combine_prepared_leaves else max_passes
        chunks = available_chunks[:leaf_pass_budget]
        if not chunks:
            return None
        selected = [message for chunk in chunks for message in chunk.messages]
        selected_lineage_ids = tuple(int(message["store_id"]) for message in selected)
        watermark = selected_lineage_ids[-1]
        selected_ids = tuple(
            int(message["store_id"])
            for message in candidates
            if int(message["store_id"]) <= watermark
        )
        selected_by_id = {
            int(message["store_id"]): message for message in candidates
        }
        selected_identity_hash = self._source_identity_hash(
            [selected_by_id[store_id] for store_id in selected_ids]
        )

        # A bounded prefix can consume every provider pass or end immediately
        # before a canonical island.  If that exact prefix has a terminal
        # disposition, an ordinary slice would keep selecting it forever even
        # after genuinely new eligible material was appended farther ahead.
        # Extend only through the first post-disposition chunk and fit it by a
        # bounded adjacent merge; an unchanged range still reaches
        # enqueue_or_coalesce and is suppressed by its exact durable key.
        terminal_projection: tuple[str, int] | None = None
        if eligible_upper > watermark:
            check_deadline()
            disposition = None
            has_new_durable_material = False
            first_new_store_id = 0
            with self._bounded_read_connection(
                lock_timeout_seconds=lock_timeout_seconds,
                deadline=deadline,
                deadline_clock=clock,
                busy_timeout_ms=busy_timeout_ms,
            ) as disposition_conn:
                current_fence = disposition_conn.execute(
                    """
                    SELECT session_id, generation, publishable
                    FROM lcm_compaction_session_fences
                    WHERE conversation_id=?
                    """,
                    (str(conversation_id),),
                ).fetchone()
                if (
                    current_fence is not None
                    and str(current_fence["session_id"] or "") == str(session_id)
                    and int(current_fence["publishable"] or 0) == 1
                ):
                    disposition = self._matching_range_disposition_under_tx(
                        disposition_conn,
                        conversation_id=str(conversation_id),
                        session_id=str(session_id),
                        session_fence_generation=int(
                            current_fence["generation"] or 0
                        ),
                        expected_frontier_store_id=frontier,
                        captured_watermark_store_id=watermark,
                        policy_fingerprint=policy_fingerprint,
                        source_identity_hash=selected_identity_hash,
                    )
                    if (
                        disposition is not None
                        and str(disposition["reason_code"] or "")
                        in _EXTENDABLE_RANGE_DISPOSITION_REASONS
                    ):
                        new_material = disposition_conn.execute(
                            f"""
                            SELECT store_id FROM messages
                            WHERE session_id=? AND store_id>? AND store_id<=?
                              AND (conversation_id=? OR conversation_id=''
                                   OR conversation_id IS NULL)
                              AND {self._message_arrival_sql}>?
                            ORDER BY store_id
                            LIMIT 1
                            """,
                            (
                                str(session_id),
                                watermark,
                                eligible_upper,
                                str(conversation_id),
                                self._range_disposition_capture_started_at(
                                    disposition
                                ),
                            ),
                        ).fetchone()
                        if new_material is not None:
                            has_new_durable_material = True
                            first_new_store_id = int(new_material["store_id"] or 0)
            if disposition is not None:
                extension_end = next(
                    (
                        index
                        for index in range(
                            leaf_pass_budget,
                            len(available_chunks),
                        )
                        if first_new_store_id
                        in available_chunks[index].source_store_ids
                    ),
                    None,
                )
                coalesced = None
                if extension_end is not None:
                    coalesced = self._coalesce_disposition_extension_chunks(
                        available_chunks[: extension_end + 1],
                        max_chunks=leaf_pass_budget,
                    )
                    if coalesced is not None:
                        chunks = coalesced
                        selected = [
                            message
                            for chunk in chunks
                            for message in chunk.messages
                        ]
                        selected_lineage_ids = tuple(
                            int(message["store_id"]) for message in selected
                        )
                        watermark = selected_lineage_ids[-1]
                        selected_ids = tuple(
                            int(message["store_id"])
                            for message in candidates
                            if int(message["store_id"]) <= watermark
                        )
                        selected_identity_hash = self._source_identity_hash(
                            [selected_by_id[store_id] for store_id in selected_ids]
                        )
                if has_new_durable_material and coalesced is None:
                    # The new row can be beyond the bounded capture or too far
                    # behind pre-existing chunks to fit a provider pass safely.
                    # Publish one exact terminal prefix as a local raw-reference
                    # node; its CAS publication advances contiguously and keeps
                    # every source row available for exact expansion.
                    terminal_projection = (
                        str(disposition["reason_code"] or ""),
                        eligible_upper,
                    )
        if terminal_projection is not None:
            disposition_reason, new_material_upper = terminal_projection
            projection = (
                "[Lossless retained source; provider summarization was skipped "
                "after an exact terminal result for this prefix.]\n"
                f"Raw source rows {selected_ids[0]}-{selected_ids[-1]} remain "
                "verbatim and expandable.\n"
                "Expand for details: exact retained raw source rows"
            )
            check_deadline()
            return self._prepare_lossless_projection(
                conversation_id=conversation_id,
                session_id=session_id,
                expected_frontier_store_id=frontier,
                captured_watermark_store_id=selected_ids[-1],
                source_store_ids=selected_ids,
                expected_source_identity_hash=selected_identity_hash,
                policy_fingerprint=policy_fingerprint,
                projection=projection,
                priority=priority,
                write_timeout_ms=100,
                lock_timeout_seconds=lock_timeout_seconds,
                write_deadline=deadline,
                deadline_clock=clock,
                replace_disposition_reason=disposition_reason,
                new_material_upper_store_id=new_material_upper,
            )
        source_tokens = sum(
            self._message_tokens(selected_by_id[store_id])
            for store_id in selected_ids
        )
        minimum_provider_tokens = max(
            1,
            int(
                getattr(
                    self.config,
                    "background_compaction_min_provider_source_tokens",
                    256,
                )
                or 256
            ),
        )
        if source_tokens < minimum_provider_tokens:
            visible_projection = serialize_summary_messages(
                summary_candidates,
                config=self.config,
                session_id=str(session_id),
                hermes_home=self._hermes_home,
            )
            if not visible_projection:
                visible_projection = (
                    "[Source content is omitted from active replay by policy; "
                    "the exact raw rows remain expandable.]"
                )
            projection = (
                "[Lossless retained source; provider summarization was skipped "
                "because this isolated range is too small to compress safely.]\n"
                f"{visible_projection}\n"
                "Expand for details: exact retained raw source rows"
            )
            check_deadline()
            return self._prepare_lossless_projection(
                conversation_id=conversation_id,
                session_id=session_id,
                expected_frontier_store_id=frontier,
                captured_watermark_store_id=selected_ids[-1],
                source_store_ids=selected_ids,
                expected_source_identity_hash=selected_identity_hash,
                policy_fingerprint=policy_fingerprint,
                projection=projection,
                priority=priority,
                write_timeout_ms=100,
                lock_timeout_seconds=lock_timeout_seconds,
                write_deadline=deadline,
                deadline_clock=clock,
            )
        try:
            check_deadline()
            return self.enqueue_or_coalesce(
                conversation_id=conversation_id,
                session_id=session_id,
                expected_frontier_store_id=frontier,
                captured_watermark_store_id=selected_ids[-1],
                source_store_ids=selected_ids,
                source_token_count=source_tokens,
                policy_fingerprint=policy_fingerprint,
                expected_node_count=1 if combine_prepared_leaves else len(chunks),
                priority=priority,
                write_timeout_ms=100,
                reject_canonical_coverage=True,
                lock_timeout_seconds=lock_timeout_seconds,
                write_deadline=deadline,
                deadline_clock=clock,
            )
        except RuntimeError as exc:
            if str(exc) in {
                "background compaction lifecycle frontier is unavailable",
                "background compaction session is fenced",
            }:
                # A publisher advanced the lifecycle after the read-only source
                # capture.  The enqueue transaction rolled back, so this is an
                # expected stale-capture no-op rather than a foreground error.
                return None
            raise

    def source_chunks(self, job_id: str) -> list[SourceChunk]:
        job = self.get_job(job_id)
        if job is None or not job.source_store_ids:
            return []
        # Partial-atomic repair needs a bound canonical-root snapshot in
        # addition to the raw group.  Never let a generic caller obtain the
        # raw-only provider path for this job kind.
        if job.job_kind == _PARTIAL_REPAIR_JOB_KIND:
            return []
        with self._lock:
            conn = self._conn
            if conn is None:
                return []
            rows = conn.execute(
                """
                SELECT store_id, session_id, conversation_id, role, content,
                       tool_call_id, tool_calls, tool_name, timestamp, token_estimate
                FROM messages
                WHERE session_id=? AND store_id>? AND store_id<=?
                ORDER BY store_id
                """,
                (
                    job.session_id,
                    job.expected_frontier_store_id,
                    job.captured_watermark_store_id,
                ),
            ).fetchall()
        by_id = {int(row["store_id"]): self._message_dict(row) for row in rows}
        if any(store_id not in by_id for store_id in job.source_store_ids):
            return []
        messages = [by_id[store_id] for store_id in job.source_store_ids]
        if self._source_identity_hash(messages) != job.source_identity_hash:
            return []
        summary_messages = self._summary_candidate_messages(
            messages,
            session_id=job.session_id,
            preserve_leading_system=job.expected_frontier_store_id == 0,
            resolve_externalized=False,
        )
        provider_chunks = self._chunk_messages(summary_messages)
        if not provider_chunks:
            return []
        max_passes = max(
            1,
            int(
                getattr(
                    self.config,
                    "background_compaction_max_passes",
                    4,
                )
                or 4
            ),
        )
        combine_prepared_leaves = bool(
            getattr(self.config, "cache_friendly_condensation_enabled", False)
            and len(provider_chunks) > 1
            and max_passes >= 2
        )
        leaf_pass_budget = (
            max_passes - 1 if combine_prepared_leaves else max_passes
        )
        if len(provider_chunks) > leaf_pass_budget:
            with self._lock:
                conn = self._conn
                extends_terminal_range = bool(
                    conn is not None
                    and self._job_extends_terminal_disposition_under_tx(
                        conn,
                        job,
                        messages,
                    )
                )
            if extends_terminal_range:
                coalesced = self._coalesce_disposition_extension_chunks(
                    provider_chunks,
                    max_chunks=leaf_pass_budget,
                )
                if coalesced is None:
                    return []
                provider_chunks = coalesced

        positions = {
            int(message["store_id"]): index for index, message in enumerate(messages)
        }
        covered: list[SourceChunk] = []
        lineage_start = 0
        for index, chunk in enumerate(provider_chunks):
            if index + 1 == len(provider_chunks):
                lineage_end = len(messages)
            else:
                last_provider_id = int(chunk.source_store_ids[-1])
                lineage_end = positions[last_provider_id] + 1
            lineage_messages = messages[lineage_start:lineage_end]
            if not lineage_messages:
                return []
            covered.append(
                SourceChunk(
                    ordinal=index,
                    source_store_ids=tuple(
                        int(message["store_id"]) for message in lineage_messages
                    ),
                    source_token_count=sum(
                        self._message_tokens(message) for message in lineage_messages
                    ),
                    messages=chunk.messages,
                )
            )
            lineage_start = lineage_end
        if tuple(
            store_id for chunk in covered for store_id in chunk.source_store_ids
        ) != job.source_store_ids:
            return []
        return covered

    def _partial_repair_source_chunks(
        self,
        job: CompactionJob,
        *,
        owner: str,
        generation: int,
    ) -> list[SourceChunk]:
        """Build one policy-scrubbed repair prompt from an exact bound snapshot."""
        root_ids = tuple(job.repair_root_node_ids)
        group_ids = tuple(job.repair_group_store_ids)
        if (
            job.job_kind != _PARTIAL_REPAIR_JOB_KIND
            or not root_ids
            or not group_ids
            or group_ids != job.source_store_ids
        ):
            return []

        with self._lock:
            conn = self._conn
            if conn is None or conn.in_transaction:
                return []
            conn.execute("BEGIN")
            try:
                current = float(self._now())
                live_job = conn.execute(
                    """
                    SELECT state, lease_owner, lease_generation, lease_expires_at
                    FROM lcm_compaction_jobs WHERE job_id=?
                    """,
                    (job.job_id,),
                ).fetchone()
                permit = conn.execute(
                    """
                    SELECT 1 FROM lcm_compaction_provider_leases
                    WHERE job_id=? AND lease_owner=? AND lease_generation=?
                      AND lease_expires_at>?
                    """,
                    (job.job_id, str(owner), int(generation), current),
                ).fetchone()
                placeholders = ",".join("?" for _ in root_ids)
                root_rows = conn.execute(
                    f"""
                    SELECT node_id, session_id, depth, summary, token_count,
                           source_token_count, source_ids, source_type, created_at,
                           earliest_at, latest_at, expand_hint
                    FROM summary_nodes
                    WHERE session_id=? AND node_id IN ({placeholders})
                    ORDER BY node_id
                    """,
                    (job.session_id, *root_ids),
                ).fetchall()
                raw_messages = self._load_source_range(
                    conn,
                    session_id=job.session_id,
                    conversation_id=job.conversation_id,
                    frontier=job.expected_frontier_store_id,
                    watermark=job.captured_watermark_store_id,
                )
                dag_epoch = self._session_dag_epoch(conn, job.session_id)
                valid = (
                    live_job is not None
                    and str(live_job["state"] or "") == "running"
                    and str(live_job["lease_owner"] or "") == str(owner)
                    and int(live_job["lease_generation"] or 0) == int(generation)
                    and float(live_job["lease_expires_at"] or 0.0) > current
                    and permit is not None
                    and tuple(int(row["node_id"]) for row in root_rows) == root_ids
                    and dag_epoch == int(job.repair_bound_dag_epoch)
                    and self._repair_root_identity_hash(root_rows)
                    == job.repair_root_identity_hash
                    and tuple(
                        int(message["store_id"]) for message in raw_messages
                    )
                    == group_ids
                    and self._source_identity_hash(raw_messages)
                    == job.repair_group_identity_hash
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        if not valid:
            return []

        root_parts: list[str] = []
        visible_root_tokens = 0
        for row in root_rows:
            candidate = {
                "role": "user",
                "content": str(row["summary"] or ""),
                "session_id": job.session_id,
            }
            filtered_root = self._summary_candidate_messages(
                (candidate,),
                session_id=job.session_id,
                preserve_leading_system=False,
                resolve_externalized=False,
            )
            scrubbed = serialize_summary_messages(
                filtered_root,
                config=self.config,
                session_id=job.session_id,
                hermes_home=self._hermes_home,
            )
            if filtered_root:
                visible_root_tokens += max(0, int(row["token_count"] or 0))
            if not scrubbed:
                scrubbed = "[Provider-visible root content omitted by policy.]"
            root_parts.append(
                f"[CANONICAL ROOT node_id={int(row['node_id'])} "
                f"depth={int(row['depth'] or 0)}]\n{scrubbed}"
            )

        filtered_raw = self._summary_candidate_messages(
            raw_messages,
            session_id=job.session_id,
            preserve_leading_system=False,
            resolve_externalized=False,
        )
        covered = set(job.repair_covered_store_ids)
        annotated_raw: list[dict[str, Any]] = []
        for message in filtered_raw:
            store_id = int(message["store_id"])
            annotated_raw.append(
                {
                    "role": "system",
                    "content": (
                        f"store_id={store_id} ownership="
                        f"{'existing-canonical' if store_id in covered else 'uncovered'}"
                    ),
                    "session_id": job.session_id,
                }
            )
            annotated_raw.append(dict(message))
        raw_text = serialize_summary_messages(
            annotated_raw,
            config=self.config,
            session_id=job.session_id,
            hermes_home=self._hermes_home,
        )
        if not raw_text:
            raw_text = "[No provider-visible raw content remains after policy filtering.]"
        payload = (
            "EXISTING CANONICAL CONTEXT\n"
            + "\n\n".join(root_parts)
            + "\n\nUNCOVERED ATOMIC CONTINUATION\n"
            + "The complete provider-visible atomic group follows; ownership "
            "labels distinguish existing canonical rows from the uncovered "
            "continuation.\n\n"
            + raw_text
        )
        return [
            SourceChunk(
                ordinal=0,
                source_store_ids=group_ids,
                # Job debt remains the raw atomic group.  Provider budgeting
                # and prepared-node metadata additionally account for every
                # provider-visible canonical root summary.
                source_token_count=job.source_token_count + visible_root_tokens,
                serialized_text=payload,
            )
        ]

    def _recover_expired_under_tx(self, conn: sqlite3.Connection, current: float) -> int:
        expired_rows: dict[str, tuple[str, str]] = {}
        for row in conn.execute(
            """
            SELECT job_id, conversation_id, session_id
            FROM lcm_compaction_jobs
            WHERE state IN ('running','prepared')
              AND lease_expires_at IS NOT NULL AND lease_expires_at<=?
            """,
            (current,),
        ).fetchall():
            expired_rows[str(row["job_id"])] = (
                str(row["conversation_id"] or ""),
                str(row["session_id"] or ""),
            )
        expired_permits = conn.execute(
            """
            SELECT permit.job_id, permit.conversation_id,
                   permit.lease_generation, permit.provider_active,
                   permit.owner_pid, permit.owner_start_ticks,
                   permit.owner_boot_id, job.session_id
            FROM lcm_compaction_provider_leases permit
            LEFT JOIN lcm_compaction_jobs job ON job.job_id=permit.job_id
            WHERE permit.lease_expires_at<=?
            """,
            (current,),
        ).fetchall()
        lease_duration = max(
            1.0,
            float(
                getattr(self.config, "background_compaction_lease_seconds", 300)
                or 300
            ),
        )
        for row in expired_permits:
            if _protected_provider_owner_is_live(row):
                # A timestamp heartbeat is only a freshness hint once a
                # physical call has started. Exact boot/PID/start identity is
                # the fail-closed fence when the guardian loses SQLite writes.
                conn.execute(
                    """
                    UPDATE lcm_compaction_provider_leases
                    SET lease_expires_at=?
                    WHERE job_id=? AND lease_generation=?
                      AND lease_expires_at<=?
                    """,
                    (
                        current + lease_duration,
                        str(row["job_id"]),
                        int(row["lease_generation"] or 0),
                        current,
                    ),
                )
                continue
            expired_rows[str(row["job_id"])] = (
                str(row["conversation_id"] or ""),
                str(row["session_id"] or ""),
            )
            conn.execute(
                """
                DELETE FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_generation=?
                  AND lease_expires_at<=?
                """,
                (
                    str(row["job_id"]),
                    int(row["lease_generation"] or 0),
                    current,
                ),
            )
        running = conn.execute(
            """
            UPDATE lcm_compaction_jobs
            SET state='retry', next_retry_at=?, lease_owner=NULL,
                lease_expires_at=NULL, heartbeat_at=NULL,
                lease_generation=lease_generation+1,
                last_error='lease expired', completion_reason='lease_expired', updated_at=?
            WHERE state='running' AND lease_expires_at IS NOT NULL AND lease_expires_at<=?
            """,
            (current, current, current),
        ).rowcount
        prepared = conn.execute(
            """
            UPDATE lcm_compaction_jobs
            SET lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                lease_generation=lease_generation+1, updated_at=?
            WHERE state='prepared' AND lease_expires_at IS NOT NULL AND lease_expires_at<=?
            """,
            (current, current),
        ).rowcount
        for job_id, (conversation_id, session_id) in expired_rows.items():
            self._record_event_under_tx(
                conn,
                "lease_expiration",
                conversation_id=conversation_id,
                session_id=session_id,
                job_id=job_id,
                detail="durable lease expired and was recovered",
                created_at=current,
                deduplicate_seconds=1.0,
            )
        return max(
            len(expired_rows),
            int(running or 0) + int(prepared or 0),
        )

    def recover_expired_leases(self, *, now: float | None = None) -> int:
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            return self._recover_expired_under_tx(conn, current)

    def acquire_emergency_provider_permit(
        self,
        *,
        conversation_id: str,
        owner: str,
        session_id: str | None = None,
        expected_session_fence_generation: int | None = None,
        lease_seconds: float | None = None,
        global_limit: int | None = None,
        now: float | None = None,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        overlap_background: bool = False,
    ) -> ProviderPermit | None:
        """Atomically reserve provider capacity for one foreground emergency.

        Foreground calls use the same durable table as background workers.  The
        serialized check-and-insert enforces both the process-wide provider cap
        and one active provider call per conversation without waiting for a
        competing lease.

        ``overlap_background`` lets an emergency foreground call run beside a
        *background preparation* lease of the same conversation (never beside
        another foreground permit, and never above the global cap). A busy
        backlog otherwise keeps the conversation lease occupied job after job,
        so emergency compaction could never summarize. Publication stays safe:
        both publishers go through the frontier/fence/source-identity CAS, the
        loser reloads, and a winning foreground publication supersedes the
        overlapping background job.
        """
        conversation = str(conversation_id or "").strip()
        lease_owner = str(owner or "").strip()
        if not conversation or not lease_owner:
            raise ValueError("conversation_id and owner are required")
        duration = max(
            1.0,
            float(
                lease_seconds
                if lease_seconds is not None
                else getattr(self.config, "background_compaction_lease_seconds", 300)
            ),
        )
        limit = min(
            _MAX_GLOBAL_PROVIDER_PERMITS,
            max(
                0,
                int(
                    global_limit
                    if global_limit is not None
                    else getattr(
                        self.config,
                        "background_compaction_global_limit",
                        _MAX_GLOBAL_PROVIDER_PERMITS,
                    )
                ),
            ),
        )
        if limit <= 0:
            return None
        process_identity = _current_linux_process_identity()
        if sys.platform.startswith("linux") and process_identity is None:
            # Starting an unfenced physical call on Linux would violate the
            # global limit if its timestamp heartbeat later missed a lock.
            return None
        owner_pid, owner_start_ticks, owner_boot_id = (
            process_identity if process_identity is not None else (None, None, None)
        )
        with self._write_transaction(
            busy_timeout_ms=0,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now() if now is None else now)
            self._recover_expired_under_tx(conn, current)
            session_fence_generation = 0
            if session_id is not None:
                if expected_session_fence_generation is None:
                    raise ValueError(
                        "expected_session_fence_generation is required with session_id"
                    )
                fence = self._session_fence_under_tx(
                    conn,
                    conversation_id=conversation,
                    session_id=str(session_id),
                    current=current,
                    bootstrap=False,
                )
                if (
                    fence is None
                    or str(fence["session_id"] or "") != str(session_id)
                    or int(fence["publishable"] or 0) != 1
                    or int(fence["generation"] or 0)
                    != int(expected_session_fence_generation)
                ):
                    return None
                session_fence_generation = int(fence["generation"] or 0)
            active = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_compaction_provider_leases
                    WHERE lease_expires_at>?
                    """,
                    (current,),
                ).fetchone()[0]
            )
            if active >= limit:
                return None
            conversation_busy = conn.execute(
                """
                SELECT 1 FROM lcm_compaction_provider_leases
                WHERE conversation_id=? AND lease_expires_at>?
                  AND (? = 0 OR job_id LIKE 'foreground-%')
                LIMIT 1
                """,
                (conversation, current, 1 if overlap_background else 0),
            ).fetchone()
            if conversation_busy is not None:
                return None
            permit_id = f"foreground-{uuid.uuid4().hex}"
            generation = 1
            expires_at = current + duration
            conn.execute(
                """
                INSERT INTO lcm_compaction_provider_leases(
                    job_id, conversation_id, lease_generation, lease_owner,
                    started_at, heartbeat_at, lease_expires_at, timed_out_at,
                    provider_active, owner_pid, owner_start_ticks, owner_boot_id
                ) VALUES(?,?,?,?,?,?,?,NULL,?,?,?,?)
                """,
                (
                    permit_id,
                    conversation,
                    generation,
                    lease_owner,
                    current,
                    current,
                    expires_at,
                    1 if process_identity is not None else 0,
                    owner_pid,
                    owner_start_ticks,
                    owner_boot_id,
                ),
            )
            return ProviderPermit(
                job_id=permit_id,
                conversation_id=conversation,
                lease_owner=lease_owner,
                lease_generation=generation,
                lease_expires_at=expires_at,
                session_fence_generation=session_fence_generation,
            )

    def claim_next_job(
        self,
        *,
        owner: str | None = None,
        lease_owner: str | None = None,
        lease_seconds: float | None = None,
        global_limit: int | None = None,
        policy_fingerprint: str | None = None,
        now: float | None = None,
    ) -> CompactionJob | None:
        owner = str(owner or lease_owner or "").strip()
        if not owner:
            raise ValueError("owner is required")
        duration = max(
            1.0,
            float(
                lease_seconds
                if lease_seconds is not None
                else getattr(self.config, "background_compaction_lease_seconds", 300)
            ),
        )
        limit = min(
            _MAX_GLOBAL_PROVIDER_PERMITS,
            max(
                0,
                int(
                    global_limit
                    if global_limit is not None
                    else getattr(
                        self.config,
                        "background_compaction_global_limit",
                        _MAX_GLOBAL_PROVIDER_PERMITS,
                    )
                ),
            ),
        )
        if limit <= 0:
            return None
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            self._recover_expired_under_tx(conn, current)
            active = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_compaction_provider_leases
                    WHERE lease_expires_at>?
                    """,
                    (current,),
                ).fetchone()[0]
            )
            if active >= limit:
                return None
            policy_clause = ""
            args: list[Any] = [current, current]
            if policy_fingerprint is not None:
                policy_clause = " AND policy_fingerprint=?"
                args.append(str(policy_fingerprint))
            row = conn.execute(
                f"""
                SELECT job.* FROM lcm_compaction_jobs AS job
                JOIN lcm_compaction_session_fences AS fence
                  ON fence.conversation_id=job.conversation_id
                 AND fence.session_id=job.session_id
                 AND fence.generation=job.session_fence_generation
                 AND fence.publishable=1
                WHERE (job.state='queued'
                   OR (job.state='retry' AND COALESCE(job.next_retry_at,0)<=?))
                  AND NOT EXISTS (
                      SELECT 1 FROM lcm_compaction_provider_leases permit
                      WHERE permit.conversation_id=job.conversation_id
                        AND permit.lease_expires_at>?
                  )
                  {policy_clause}
                ORDER BY job.priority DESC, job.created_at, job.job_id
                LIMIT 1
                """,
                args,
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='running', lease_owner=?, lease_expires_at=?, heartbeat_at=?,
                    lease_generation=lease_generation+1, attempt_count=attempt_count+1,
                    next_retry_at=NULL, completion_reason=NULL, updated_at=?
                WHERE job_id=? AND state IN ('queued','retry')
                """,
                (owner, current + duration, current, current, row["job_id"]),
            )
            if cursor.rowcount != 1:
                return None
            claimed = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
            if claimed is None:
                return None
            conn.execute(
                """
                INSERT INTO lcm_compaction_provider_leases(
                    job_id, conversation_id, lease_generation, lease_owner,
                    started_at, heartbeat_at, lease_expires_at, timed_out_at
                ) VALUES(?,?,?,?,?,?,?,NULL)
                """,
                (
                    str(claimed["job_id"]),
                    str(claimed["conversation_id"]),
                    int(claimed["lease_generation"] or 0),
                    owner,
                    current,
                    current,
                    current + duration,
                ),
            )
            return self._job_from_row(claimed)

    def heartbeat(
        self,
        job_id: str,
        *,
        owner: str | None = None,
        lease_owner: str | None = None,
        generation: int | None = None,
        lease_generation: int | None = None,
        lease_seconds: float | None = None,
        now: float | None = None,
        provider_active: bool = False,
    ) -> bool:
        owner = str(owner or lease_owner or "")
        generation = int(generation if generation is not None else lease_generation or 0)
        duration = max(
            1.0,
            float(
                lease_seconds
                if lease_seconds is not None
                else getattr(self.config, "background_compaction_lease_seconds", 300)
            ),
        )
        process_identity = _current_linux_process_identity()
        if (
            provider_active
            and sys.platform.startswith("linux")
            and process_identity is None
        ):
            return False
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            job_row = conn.execute(
                """
                SELECT lease_expires_at FROM lcm_compaction_jobs
                WHERE job_id=? AND state='running' AND lease_owner=?
                  AND lease_generation=?
                """,
                (str(job_id), owner, generation),
            ).fetchone()
            permit_row = conn.execute(
                """
                SELECT * FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                """,
                (str(job_id), owner, generation),
            ).fetchone()
            if job_row is None or permit_row is None:
                return False
            current_owner = _provider_permit_owned_by_process(
                permit_row,
                process_identity,
            )
            unexpired = (
                float(job_row["lease_expires_at"] or 0.0) > current
                and float(permit_row["lease_expires_at"] or 0.0) > current
            )
            if not unexpired and not current_owner:
                return False
            conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE job_id=? AND state='running' AND lease_owner=?
                  AND lease_generation=?
                """,
                (current, current + duration, current, str(job_id), owner, generation),
            )
            if provider_active and process_identity is not None:
                owner_pid, owner_start_ticks, owner_boot_id = process_identity
                conn.execute(
                    """
                    UPDATE lcm_compaction_provider_leases
                    SET heartbeat_at=?, lease_expires_at=?, provider_active=1,
                        owner_pid=?, owner_start_ticks=?, owner_boot_id=?
                    WHERE job_id=? AND lease_owner=? AND lease_generation=?
                    """,
                    (
                        current,
                        current + duration,
                        owner_pid,
                        owner_start_ticks,
                        owner_boot_id,
                        str(job_id),
                        owner,
                        generation,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE lcm_compaction_provider_leases
                    SET heartbeat_at=?, lease_expires_at=?
                    WHERE job_id=? AND lease_owner=? AND lease_generation=?
                    """,
                    (current, current + duration, str(job_id), owner, generation),
                )
            return True

    def mark_retry(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        error: BaseException | str,
        keep_provider_permit: bool = False,
        completion_reason: str = "provider_error",
        terminal: bool = False,
        provider_attempted: bool | None = None,
        now: float | None = None,
    ) -> CompactionJob | bool:
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            if (
                row is None
                or row["state"] != "running"
                or row["lease_owner"] != str(owner)
                or int(row["lease_generation"] or 0) != int(generation)
                or float(row["lease_expires_at"] or 0.0) <= current
            ):
                return False
            permit = conn.execute(
                """
                SELECT 1 FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                  AND lease_expires_at>?
                """,
                (str(job_id), owner, generation, current),
            ).fetchone()
            if permit is None:
                return False
            attempts = max(1, int(row["attempt_count"] or 1))
            delay = min(300.0, float(5 * (2 ** min(attempts - 1, 6))))
            guard_deferred = (
                not terminal
                and completion_reason
                in {"provider_circuit_open", "provider_spend_guard"}
                and provider_attempted is not True
            )
            # A guard rejection did not start a physical provider call. Persist
            # the full guard wait and restore the claim counter so repeated
            # circuit/spend deferrals cannot exhaust the finite provider-attempt
            # allowance. Ordinary provider failures still terminate after five.
            if completion_reason == "provider_circuit_open":
                delay = max(
                    delay,
                    float(self.config.summary_circuit_breaker_cooldown_seconds),
                )
            elif completion_reason == "provider_spend_guard":
                delay = max(
                    delay,
                    float(self.config.summary_spend_backoff_seconds),
                    float(self.config.summary_spend_window_seconds),
                )
            sanitized = _sanitize_error(error)
            recorded_attempts = max(0, attempts - 1) if guard_deferred else attempts
            terminal = bool(terminal) or (not guard_deferred and attempts >= 5)
            conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state=?, next_retry_at=?, last_error=?,
                    completion_reason=?, lease_owner=NULL,
                    lease_expires_at=NULL, heartbeat_at=NULL, updated_at=?,
                    attempt_count=?
                WHERE job_id=?
                """,
                (
                    "failed" if terminal else "retry",
                    None if terminal else current + delay,
                    sanitized,
                    str(completion_reason or "provider_error"),
                    current,
                    recorded_attempts,
                    str(job_id),
                ),
            )
            if keep_provider_permit:
                duration = max(
                    1.0,
                    float(
                        getattr(self.config, "background_compaction_lease_seconds", 300)
                        or 300
                    ),
                )
                conn.execute(
                    """
                    UPDATE lcm_compaction_provider_leases
                    SET timed_out_at=COALESCE(timed_out_at, ?), heartbeat_at=?,
                        lease_expires_at=?
                    WHERE job_id=? AND lease_owner=? AND lease_generation=?
                    """,
                    (
                        current,
                        current,
                        current + duration,
                        str(job_id),
                        str(owner),
                        int(generation),
                    ),
                )
            else:
                conn.execute(
                    """
                    DELETE FROM lcm_compaction_provider_leases
                    WHERE job_id=? AND lease_owner=? AND lease_generation=?
                    """,
                    (str(job_id), str(owner), int(generation)),
                )
            if terminal and str(completion_reason or "") in (
                set(_EVENT_REASON_CODES)
                | {"provider_error", "no_eligible_raw_range"}
            ):
                self._upsert_range_disposition_under_tx(
                    conn,
                    row=row,
                    reason_code=str(completion_reason),
                    current=current,
                )
            if str(completion_reason or "") in _EVENT_REASON_CODES:
                self._record_event_under_tx(
                    conn,
                    str(completion_reason),
                    conversation_id=str(row["conversation_id"] or ""),
                    session_id=str(row["session_id"] or ""),
                    job_id=str(job_id),
                    detail=sanitized,
                    created_at=current,
                )
            updated = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            job = self._job_from_row(updated)
            assert job is not None
            return job

    def heartbeat_provider_permit(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        lease_seconds: float | None = None,
        now: float | None = None,
    ) -> bool:
        duration = max(
            1.0,
            float(
                lease_seconds
                if lease_seconds is not None
                else getattr(self.config, "background_compaction_lease_seconds", 300)
            ),
        )
        process_identity = _current_linux_process_identity()
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            row = conn.execute(
                """
                SELECT * FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                """,
                (str(job_id), str(owner), int(generation)),
            ).fetchone()
            if row is None:
                return False
            if int(row["provider_active"] or 0) == 1:
                if process_identity is None and sys.platform.startswith("linux"):
                    raise RuntimeError(
                        "Linux provider owner identity is temporarily unavailable"
                    )
                if not _provider_permit_owned_by_process(row, process_identity):
                    return False
            elif float(row["lease_expires_at"] or 0.0) <= current:
                return False
            cursor = conn.execute(
                """
                UPDATE lcm_compaction_provider_leases
                SET heartbeat_at=?, lease_expires_at=?
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                """,
                (
                    current,
                    current + duration,
                    str(job_id),
                    str(owner),
                    int(generation),
                ),
            )
            return cursor.rowcount == 1

    def release_provider_permit(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        busy_timeout_ms: int | None = None,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> bool:
        with self._write_transaction(
            busy_timeout_ms=busy_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            cursor = conn.execute(
                """
                DELETE FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                """,
                (str(job_id), str(owner), int(generation)),
            )
            return cursor.rowcount == 1

    @staticmethod
    def _coerce_prepared_node(value: PreparedSummaryNode | Mapping[str, Any]) -> PreparedSummaryNode:
        if isinstance(value, PreparedSummaryNode):
            return value
        return PreparedSummaryNode(
            ordinal=int(value.get("ordinal", 0)),
            summary=str(value.get("summary") or ""),
            source_store_ids=tuple(int(item) for item in (value.get("source_store_ids") or [])),
            source_token_count=max(0, int(value.get("source_token_count") or 0)),
            summary_token_count=max(0, int(value.get("summary_token_count") or 0)),
            creation_model=str(value.get("creation_model") or ""),
            creation_provider=str(value.get("creation_provider") or ""),
            route_latency_ms=max(0, int(value.get("route_latency_ms") or 0)),
            attempt_number=max(0, int(value.get("attempt_number") or 0)),
            escalation_level=max(0, int(value.get("escalation_level") or 0)),
            attribution_available=bool(value.get("attribution_available", False)),
            fallback_used=(
                None
                if value.get("fallback_used") is None
                else bool(value.get("fallback_used"))
            ),
        )

    def mark_prepared(
        self,
        job_id: str,
        *,
        owner: str | None = None,
        lease_owner: str | None = None,
        generation: int | None = None,
        lease_generation: int | None = None,
        prepared_nodes: Sequence[PreparedSummaryNode | Mapping[str, Any]] | None = None,
        prepared_node_ids: Sequence[int] | None = None,
        expected_node_count: int | None = None,
        prepared_node_count: int | None = None,
        now: float | None = None,
    ) -> CompactionJob | bool:
        del prepared_node_ids  # legacy API: durable nodes are supplied as records
        owner = str(owner or lease_owner or "")
        generation = int(generation if generation is not None else lease_generation or 0)
        nodes = sorted(
            (self._coerce_prepared_node(item) for item in (prepared_nodes or [])),
            key=lambda item: item.ordinal,
        )
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            if (
                row is None
                or row["state"] != "running"
                or row["lease_owner"] != owner
                or int(row["lease_generation"] or 0) != generation
                or float(row["lease_expires_at"] or 0.0) <= current
            ):
                return False
            permit = conn.execute(
                """
                SELECT 1 FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                  AND lease_expires_at>?
                """,
                (str(job_id), owner, generation, current),
            ).fetchone()
            if permit is None:
                return False
            required = int(
                expected_node_count
                if expected_node_count is not None
                else row["expected_node_count"] or 0
            )
            actual = int(prepared_node_count if prepared_node_count is not None else len(nodes))
            if actual != len(nodes) or (required > 0 and actual != required):
                return False
            consumed = _load_ids(row["source_store_ids"])
            combined = tuple(store_id for node in nodes for store_id in node.source_store_ids)
            if combined != consumed:
                return False
            conn.execute(
                "DELETE FROM lcm_compaction_prepared_nodes WHERE job_id=?", (str(job_id),)
            )
            for node in nodes:
                conn.execute(
                    """
                    INSERT INTO lcm_compaction_prepared_nodes(
                        job_id, ordinal, summary, source_store_ids,
                        source_token_count, summary_token_count,
                        creation_model, creation_provider, route_latency_ms,
                        attempt_number, escalation_level,
                        attribution_available, fallback_used, prepared_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(job_id),
                        node.ordinal,
                        node.summary,
                        _json_ids(node.source_store_ids),
                        node.source_token_count,
                        node.summary_token_count,
                        node.creation_model,
                        node.creation_provider,
                        node.route_latency_ms,
                        node.attempt_number,
                        node.escalation_level,
                        int(node.attribution_available),
                        (
                            None
                            if node.fallback_used is None
                            else int(node.fallback_used)
                        ),
                        current,
                    ),
                )
            conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='prepared', prepared_node_count=?, expected_node_count=?,
                    prepared_at=?, updated_at=?, lease_owner=NULL,
                    lease_expires_at=NULL, heartbeat_at=NULL, next_retry_at=NULL,
                    last_error=NULL, completion_reason='prepared'
                WHERE job_id=?
                """,
                (actual, required or actual, current, current, str(job_id)),
            )
            conn.execute(
                """
                DELETE FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                """,
                (str(job_id), owner, generation),
            )
            self._record_event_under_tx(
                conn,
                "background_summary_prepared",
                conversation_id=str(row["conversation_id"] or ""),
                session_id=str(row["session_id"] or ""),
                job_id=str(job_id),
                detail=f"prepared_nodes={actual}",
                created_at=current,
            )
            updated = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            job = self._job_from_row(updated)
            assert job is not None
            return job

    def save_prepared_progress(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        prepared_node: PreparedSummaryNode | Mapping[str, Any],
        now: float | None = None,
    ) -> bool:
        """Persist one inactive node after a completed provider call."""
        node = self._coerce_prepared_node(prepared_node)
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?",
                (str(job_id),),
            ).fetchone()
            permit = conn.execute(
                """
                SELECT 1 FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                  AND lease_expires_at>?
                """,
                (str(job_id), str(owner), int(generation), current),
            ).fetchone()
            if (
                row is None
                or row["state"] != "running"
                or row["lease_owner"] != str(owner)
                or int(row["lease_generation"] or 0) != int(generation)
                or float(row["lease_expires_at"] or 0.0) <= current
                or permit is None
            ):
                return False
            consumed = _load_ids(row["source_store_ids"])
            consumed_set = set(consumed)
            if any(store_id not in consumed_set for store_id in node.source_store_ids):
                return False
            conn.execute(
                """
                INSERT INTO lcm_compaction_prepared_nodes(
                    job_id, ordinal, summary, source_store_ids,
                    source_token_count, summary_token_count,
                    creation_model, creation_provider, route_latency_ms,
                    attempt_number, escalation_level,
                    attribution_available, fallback_used, prepared_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id, ordinal) DO UPDATE SET
                    summary=excluded.summary,
                    source_store_ids=excluded.source_store_ids,
                    source_token_count=excluded.source_token_count,
                    summary_token_count=excluded.summary_token_count,
                    creation_model=excluded.creation_model,
                    creation_provider=excluded.creation_provider,
                    route_latency_ms=excluded.route_latency_ms,
                    attempt_number=excluded.attempt_number,
                    escalation_level=excluded.escalation_level,
                    attribution_available=excluded.attribution_available,
                    fallback_used=excluded.fallback_used,
                    prepared_at=excluded.prepared_at
                """,
                (
                    str(job_id),
                    node.ordinal,
                    node.summary,
                    _json_ids(node.source_store_ids),
                    node.source_token_count,
                    node.summary_token_count,
                    node.creation_model,
                    node.creation_provider,
                    node.route_latency_ms,
                    node.attempt_number,
                    node.escalation_level,
                    int(node.attribution_available),
                    (
                        None
                        if node.fallback_used is None
                        else int(node.fallback_used)
                    ),
                    current,
                ),
            )
            count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM lcm_compaction_prepared_nodes WHERE job_id=?",
                    (str(job_id),),
                ).fetchone()[0]
            )
            conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET prepared_node_count=?, updated_at=? WHERE job_id=?
                """,
                (count, current, str(job_id)),
            )
            return True

    def supersede_claimed_job(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        reason: str,
        now: float | None = None,
    ) -> CompactionJob | None:
        with self._write_transaction() as conn:
            current = float(self._now() if now is None else now)
            cursor = conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='superseded', completion_reason=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL
                WHERE job_id=? AND state='running' AND lease_owner=?
                  AND lease_generation=? AND lease_expires_at>?
                """,
                (
                    _sanitize_error(reason),
                    current,
                    str(job_id),
                    str(owner),
                    int(generation),
                    current,
                ),
            )
            if cursor.rowcount != 1:
                return self._job_from_row(
                    conn.execute(
                        "SELECT * FROM lcm_compaction_jobs WHERE job_id=?",
                        (str(job_id),),
                    ).fetchone()
                )
            conn.execute(
                """
                DELETE FROM lcm_compaction_provider_leases
                WHERE job_id=? AND lease_owner=? AND lease_generation=?
                """,
                (str(job_id), str(owner), int(generation)),
            )
            row = conn.execute(
                "SELECT conversation_id, session_id FROM lcm_compaction_jobs WHERE job_id=?",
                (str(job_id),),
            ).fetchone()
            if row is not None:
                self._record_event_under_tx(
                    conn,
                    "prepared_summary_superseded",
                    conversation_id=str(row["conversation_id"] or ""),
                    session_id=str(row["session_id"] or ""),
                    job_id=str(job_id),
                    detail=reason,
                    created_at=current,
                )
            return self._job_from_row(
                conn.execute(
                    "SELECT * FROM lcm_compaction_jobs WHERE job_id=?",
                    (str(job_id),),
                ).fetchone()
            )

    def get_prepared_nodes(self, job_id: str) -> list[PreparedSummaryNode]:
        with self._lock:
            conn = self._conn
            if conn is None:
                return []
            rows = conn.execute(
                """
                SELECT * FROM lcm_compaction_prepared_nodes
                WHERE job_id=? ORDER BY ordinal
                """,
                (str(job_id),),
            ).fetchall()
        return [
            PreparedSummaryNode(
                prepared_node_id=int(row["prepared_node_id"]),
                ordinal=int(row["ordinal"]),
                summary=str(row["summary"]),
                source_store_ids=_load_ids(row["source_store_ids"]),
                source_token_count=int(row["source_token_count"] or 0),
                summary_token_count=int(row["summary_token_count"] or 0),
                creation_model=str(row["creation_model"] or ""),
                creation_provider=str(row["creation_provider"] or ""),
                route_latency_ms=int(row["route_latency_ms"] or 0),
                attempt_number=int(row["attempt_number"] or 0),
                escalation_level=int(row["escalation_level"] or 0),
                attribution_available=bool(row["attribution_available"]),
                fallback_used=(
                    None
                    if row["fallback_used"] is None
                    else bool(row["fallback_used"])
                ),
                prepared_at=float(row["prepared_at"] or 0.0),
            )
            for row in rows
        ]

    def acquire_publication_lease(
        self,
        job_id: str,
        *,
        owner: str,
        lease_seconds: float | None = None,
        write_timeout_ms: int | None = 100,
        now: float | None = None,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> CompactionJob | None:
        duration = max(
            1.0,
            float(
                lease_seconds
                if lease_seconds is not None
                else getattr(self.config, "background_compaction_lease_seconds", 300)
            ),
        )
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now() if now is None else now)
            effective_duration = duration
            if write_deadline is not None:
                clock = deadline_clock or time.monotonic
                remaining = float(write_deadline) - float(clock())
                if remaining <= 0:
                    raise TimeoutError(
                        "background compaction publication lease deadline exceeded"
                    )
                # A foreground publisher may fail after this short lease CAS.
                # Bound the durable lease itself to the same wall so timeout
                # leaves no active 300-second blocker and another caller can
                # reclaim it immediately without post-deadline cleanup.
                effective_duration = min(duration, remaining)
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            if row is None or row["state"] != "prepared":
                return None
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=str(row["conversation_id"]),
                session_id=str(row["session_id"]),
                current=current,
                bootstrap=False,
            )
            if (
                fence is None
                or str(fence["session_id"] or "") != str(row["session_id"])
                or int(fence["publishable"] or 0) != 1
                or int(fence["generation"] or 0)
                != int(row["session_fence_generation"] or 0)
            ):
                return None
            held_until = float(row["lease_expires_at"] or 0.0)
            held_by = str(row["lease_owner"] or "")
            if held_until > current and held_by and held_by != str(owner):
                return None
            if held_by and held_until <= current:
                self._record_event_under_tx(
                    conn,
                    "lease_expiration",
                    conversation_id=str(row["conversation_id"] or ""),
                    session_id=str(row["session_id"] or ""),
                    job_id=str(job_id),
                    detail="expired publication lease reclaimed",
                    created_at=current,
                    deduplicate_seconds=1.0,
                )
            if held_until > current and held_by == str(owner):
                conn.execute(
                    """
                    UPDATE lcm_compaction_jobs
                    SET lease_expires_at=?, heartbeat_at=?, updated_at=?
                    WHERE job_id=? AND state='prepared' AND lease_owner=?
                    """,
                    (
                        current + effective_duration,
                        current,
                        current,
                        str(job_id),
                        str(owner),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE lcm_compaction_jobs
                    SET lease_owner=?, lease_expires_at=?, heartbeat_at=?,
                        lease_generation=lease_generation+1, updated_at=?
                    WHERE job_id=? AND state='prepared'
                    """,
                    (
                        str(owner),
                        current + effective_duration,
                        current,
                        current,
                        str(job_id),
                    ),
                )
            updated = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            return self._job_from_row(updated)

    def release_publication_lease(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        write_timeout_ms: int | None = 100,
        now: float | None = None,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> bool:
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now() if now is None else now)
            cursor = conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    lease_generation=lease_generation+1, updated_at=?
                WHERE job_id=? AND state='prepared' AND lease_owner=?
                  AND lease_generation=?
                """,
                (current, str(job_id), str(owner), int(generation)),
            )
            return cursor.rowcount == 1

    @contextmanager
    def _publication_transaction(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        write_timeout_ms: int | None,
        lock_timeout_seconds: float | None,
        write_deadline: float | None,
        deadline_clock: Callable[[], float] | None,
    ) -> Iterator[sqlite3.Connection]:
        try:
            with self._write_transaction(
                busy_timeout_ms=write_timeout_ms,
                lock_timeout_seconds=lock_timeout_seconds,
                write_deadline=write_deadline,
                deadline_clock=deadline_clock,
            ) as conn:
                yield conn
        except BaseException as exc:
            try:
                self.release_publication_lease(
                    job_id,
                    owner=owner,
                    generation=generation,
                    write_timeout_ms=(
                        100
                        if write_timeout_ms is None
                        else min(100, max(0, int(write_timeout_ms)))
                    ),
                    lock_timeout_seconds=lock_timeout_seconds,
                    write_deadline=write_deadline,
                    deadline_clock=deadline_clock,
                )
            except BaseException:
                logger.warning(
                    "LCM could not promptly release a failed publication lease",
                    exc_info=True,
                )
            if isinstance(exc, sqlite3.OperationalError) and any(
                marker in str(exc).lower() for marker in ("locked", "busy")
            ):
                try:
                    job = self.get_job(
                        job_id,
                        lock_timeout_seconds=lock_timeout_seconds,
                        deadline=write_deadline,
                        deadline_clock=deadline_clock,
                        busy_timeout_ms=0,
                    )
                    self.record_diagnostic_event(
                        "sqlite_lock_contention",
                        conversation_id=job.conversation_id if job else "",
                        session_id=job.session_id if job else "",
                        job_id=str(job_id),
                        detail=exc,
                        deduplicate_seconds=60.0,
                        write_timeout_ms=(
                            100
                            if write_timeout_ms is None
                            else min(100, max(0, int(write_timeout_ms)))
                        ),
                        lock_timeout_seconds=lock_timeout_seconds,
                        write_deadline=write_deadline,
                        deadline_clock=deadline_clock,
                    )
                except BaseException:
                    logger.debug(
                        "LCM could not persist publication contention diagnostics",
                        exc_info=True,
                    )
            raise

    @staticmethod
    def _source_ids_already_canonical(
        conn: sqlite3.Connection,
        session_id: str,
        source_ids: Sequence[int],
    ) -> bool:
        wanted = tuple(dict.fromkeys(int(item) for item in source_ids))
        if not wanted:
            return False
        for offset in range(0, len(wanted), 500):
            batch = wanted[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            found = conn.execute(
                f"""
                SELECT 1 FROM lcm_compaction_source_coverage
                WHERE session_id=? AND store_id IN ({placeholders})
                LIMIT 1
                """,
                (str(session_id), *batch),
            ).fetchone()
            if found is not None:
                return True
            owner = conn.execute(
                f"""
                SELECT 1
                FROM lcm_compaction_node_sources AS owner
                JOIN summary_nodes AS live
                  ON live.node_id=owner.node_id
                 AND live.session_id=owner.session_id
                 AND live.source_type='messages'
                WHERE owner.session_id=?
                  AND owner.store_id IN ({placeholders})
                LIMIT 1
                """,
                (str(session_id), *batch),
            ).fetchone()
            if owner is not None:
                return True
        return False

    @staticmethod
    def _insert_canonical_node(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        summary: str,
        token_count: int,
        source_token_count: int,
        source_ids: Sequence[int],
        created_at: float,
        expand_hint: str = "",
    ) -> int:
        source_ids_tuple = tuple(int(item) for item in source_ids)
        timestamps: list[float] = []
        if source_ids_tuple:
            placeholders = ",".join("?" for _ in source_ids_tuple)
            timestamps = [
                float(row[0])
                for row in conn.execute(
                    f"SELECT timestamp FROM messages WHERE store_id IN ({placeholders}) ORDER BY store_id",
                    source_ids_tuple,
                ).fetchall()
                if row[0] is not None
            ]
        cursor = conn.execute(
            """
            INSERT INTO summary_nodes(
                session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at, earliest_at, latest_at, expand_hint
            ) VALUES(?,0,?,?,?,?, 'messages',?,?,?,?)
            """,
            (
                str(session_id),
                str(summary),
                max(0, int(token_count or 0)),
                max(0, int(source_token_count or 0)),
                _json_ids(source_ids_tuple),
                float(created_at),
                min(timestamps) if timestamps else None,
                max(timestamps) if timestamps else None,
                str(expand_hint or ""),
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _insert_canonical_parent_node(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        summary: str,
        token_count: int,
        source_token_count: int,
        child_node_ids: Sequence[int],
        created_at: float,
        expand_hint: str = "",
    ) -> int:
        child_ids = tuple(int(item) for item in child_node_ids)
        if not child_ids or len(set(child_ids)) != len(child_ids):
            raise RuntimeError("repair parent children must be exact and distinct")
        placeholders = ",".join("?" for _ in child_ids)
        children = conn.execute(
            f"""
            SELECT node_id, depth, earliest_at, latest_at, created_at
            FROM summary_nodes
            WHERE session_id=? AND node_id IN ({placeholders})
            ORDER BY node_id
            """,
            (str(session_id), *child_ids),
        ).fetchall()
        if {int(row["node_id"]) for row in children} != set(child_ids):
            raise RuntimeError("repair parent child disappeared")
        earliest_values = [
            float(row["earliest_at"] if row["earliest_at"] is not None else row["created_at"])
            for row in children
        ]
        latest_values = [
            float(row["latest_at"] if row["latest_at"] is not None else row["created_at"])
            for row in children
        ]
        cursor = conn.execute(
            """
            INSERT INTO summary_nodes(
                session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at, earliest_at, latest_at,
                expand_hint
            ) VALUES(?,?,?,?,?,?,'nodes',?,?,?,?)
            """,
            (
                str(session_id),
                max(int(row["depth"] or 0) for row in children) + 1,
                str(summary),
                max(0, int(token_count or 0)),
                max(0, int(source_token_count or 0)),
                _json_ids(child_ids),
                float(created_at),
                min(earliest_values),
                max(latest_values),
                str(expand_hint or ""),
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _summary_expand_hint(summary: str) -> str:
        marker = "Expand for details:"
        position = str(summary or "").rfind(marker)
        if position < 0:
            return ""
        return str(summary)[position + len(marker):].strip().strip("[]")[:500]

    @staticmethod
    def _advance_frontier_cas(
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        captured_watermark_store_id: int,
        debt_tokens: int,
        current: float,
    ) -> bool:
        cursor = conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_frontier_store_id=?,
                debt_size_estimate=MAX(0, debt_size_estimate-?),
                debt_kind=CASE WHEN MAX(0, debt_size_estimate-?)=0 THEN NULL ELSE debt_kind END,
                debt_updated_at=?, updated_at=?
            WHERE conversation_id=? AND current_session_id=?
              AND current_frontier_store_id=?
            """,
            (
                int(captured_watermark_store_id),
                max(0, int(debt_tokens or 0)),
                max(0, int(debt_tokens or 0)),
                current,
                current,
                str(conversation_id),
                str(session_id),
                max(0, int(expected_frontier_store_id or 0)),
            ),
        )
        return cursor.rowcount == 1

    @staticmethod
    def _publication_lifecycle_status_under_tx(
        conn: sqlite3.Connection,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
    ) -> str:
        """Classify an active, ordinary-unbound, or stale publication scope.

        Hermes finalizes the lifecycle row between ordinary API/tool turns.  In
        that narrow gap ``current_session_id`` is NULL, while the same durable
        session and frontier are retained in the last-finalized columns.  That
        state is not an invalidation; publication must wait for the next bind.
        A different session or frontier is still stale and fails closed.
        """
        lifecycle = conn.execute(
            """
            SELECT current_session_id, current_frontier_store_id,
                   last_finalized_session_id,
                   last_finalized_frontier_store_id
            FROM lcm_lifecycle_state WHERE conversation_id=?
            """,
            (str(conversation_id),),
        ).fetchone()
        if lifecycle is None:
            return "stale"
        session = str(session_id or "")
        frontier = max(0, int(expected_frontier_store_id or 0))
        current_session = str(lifecycle["current_session_id"] or "")
        if current_session:
            if (
                current_session == session
                and int(lifecycle["current_frontier_store_id"] or 0) == frontier
            ):
                return "active"
            return "stale"
        if (
            str(lifecycle["last_finalized_session_id"] or "") == session
            and int(lifecycle["last_finalized_frontier_store_id"] or 0) == frontier
        ):
            return "temporarily_unbound"
        return "stale"

    @staticmethod
    def _defer_prepared_publication_under_tx(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        owner: str,
        generation: int,
        current: float,
    ) -> PromotionResult:
        """Keep prepared work intact while atomically yielding its lease."""
        job_id = str(row["job_id"])
        released = conn.execute(
            """
            UPDATE lcm_compaction_jobs
            SET lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                lease_generation=lease_generation+1, updated_at=?
            WHERE job_id=? AND state='prepared' AND lease_owner=?
              AND lease_generation=?
            """,
            (float(current), job_id, str(owner), int(generation)),
        )
        if released.rowcount != 1:
            return PromotionResult(False, "lease_lost", job_id)
        return PromotionResult(
            False,
            "lifecycle_temporarily_unbound",
            job_id,
            int(row["captured_watermark_store_id"] or 0),
        )

    @staticmethod
    def _record_publication_telemetry_under_tx(
        conn: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        prepared_rows: Sequence[sqlite3.Row],
        published_at: float,
        observed_active_tokens: int | None,
        effective_context_window: int | None,
        publication_trigger_tokens: int | None,
        emergency_trigger_tokens: int | None,
        trigger_policy: str,
    ) -> None:
        """Persist the exact successful publication boundary in the same tx."""

        def optional_nonnegative(value: int | None) -> int | None:
            if value is None:
                return None
            try:
                return max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                return None

        observed = optional_nonnegative(observed_active_tokens)
        effective = optional_nonnegative(effective_context_window)
        publication_trigger = optional_nonnegative(publication_trigger_tokens)
        emergency_trigger = optional_nonnegative(emergency_trigger_tokens)
        policy = str(trigger_policy or "")[:64]
        if (
            not policy
            and effective is not None
            and publication_trigger is not None
            and effective - publication_trigger == 100_000
        ):
            policy = "fixed_reserve"
        providers = {
            str(item["creation_provider"] or "")
            for item in prepared_rows
            if str(item["creation_provider"] or "")
        }
        models = {
            str(item["creation_model"] or "")
            for item in prepared_rows
            if str(item["creation_model"] or "")
        }
        provider = next(iter(providers)) if len(providers) == 1 else (
            "mixed" if providers else ""
        )
        model = next(iter(models)) if len(models) == 1 else (
            "mixed" if models else ""
        )
        prepared_at = row["prepared_at"]
        wait_ms = (
            max(0, int((float(published_at) - float(prepared_at)) * 1_000.0))
            if prepared_at is not None
            else None
        )
        conn.execute(
            """
            INSERT INTO lcm_compaction_publications(
                job_id, conversation_id, session_id,
                publication_fence_generation, range_identity,
                observed_active_tokens, effective_context_window,
                publication_trigger_tokens, emergency_trigger_tokens,
                trigger_policy, prepared_to_published_ms,
                source_token_count, summary_token_count,
                provider, model, published_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(row["job_id"] or ""),
                str(row["conversation_id"] or ""),
                str(row["session_id"] or ""),
                int(row["session_fence_generation"] or 0),
                str(row["source_identity_hash"] or ""),
                observed,
                effective,
                publication_trigger,
                emergency_trigger,
                policy,
                wait_ms,
                int(row["source_token_count"] or 0),
                sum(int(item["summary_token_count"] or 0) for item in prepared_rows),
                provider,
                model,
                float(published_at),
            ),
        )
        conn.execute(
            """
            UPDATE lcm_compaction_range_dispositions
            SET resolved_at=?, last_seen_at=?
            WHERE conversation_id=? AND session_id=?
              AND session_fence_generation=?
              AND expected_frontier_store_id=?
              AND captured_watermark_store_id=?
              AND policy_fingerprint=? AND source_identity_hash=?
            """,
            (
                float(published_at),
                float(published_at),
                str(row["conversation_id"] or ""),
                str(row["session_id"] or ""),
                int(row["session_fence_generation"] or 0),
                int(row["expected_frontier_store_id"] or 0),
                int(row["captured_watermark_store_id"] or 0),
                str(row["policy_fingerprint"] or ""),
                str(row["source_identity_hash"] or ""),
            ),
        )

    def _publish_partial_atomic_repair_under_tx(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        owner: str,
        generation: int,
        policy_fingerprint: str,
        failure_hook: str | Callable[[str], None] | None,
        observed_active_tokens: int | None,
        effective_context_window: int | None,
        publication_trigger_tokens: int | None,
        emergency_trigger_tokens: int | None,
        trigger_policy: str,
        now: float | None,
    ) -> PromotionResult:
        """Publish one provenance bridge and its semantic parent atomically."""
        job_id = str(row["job_id"])
        current = float(self._now() if now is None else now)

        def supersede(reason: str) -> PromotionResult:
            conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='superseded', completion_reason=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL
                WHERE job_id=?
                """,
                (reason, current, job_id),
            )
            self._record_event_under_tx(
                conn,
                "prepared_summary_superseded",
                conversation_id=str(row["conversation_id"] or ""),
                session_id=str(row["session_id"] or ""),
                job_id=job_id,
                detail=reason,
                created_at=current,
            )
            return PromotionResult(
                False,
                reason,
                job_id,
                int(row["captured_watermark_store_id"] or 0),
            )

        if str(row["policy_fingerprint"] or "") != str(policy_fingerprint or ""):
            return supersede("policy_mismatch")
        session = str(row["session_id"] or "")
        conversation = str(row["conversation_id"] or "")
        frontier = int(row["expected_frontier_store_id"] or 0)
        group_ids = _load_ids(row["repair_group_store_ids"])
        source_ids = _load_ids(row["source_store_ids"])
        covered_ids = _load_ids(row["repair_covered_store_ids"])
        uncovered_ids = _load_ids(row["repair_uncovered_store_ids"])
        if (
            not group_ids
            or len(group_ids) > _PARTIAL_REPAIR_MAX_GROUP_ROWS
            or group_ids != source_ids
            or tuple(sorted(set(group_ids))) != group_ids
            or int(row["captured_watermark_store_id"] or 0) != group_ids[-1]
            or set(covered_ids) & set(uncovered_ids)
            or tuple(item for item in group_ids if item in set(covered_ids))
            != covered_ids
            or tuple(item for item in group_ids if item in set(uncovered_ids))
            != uncovered_ids
            or set(covered_ids) | set(uncovered_ids) != set(group_ids)
            or not covered_ids
            or not uncovered_ids
        ):
            return supersede("partial_repair_metadata_invalid")
        fence = self._session_fence_under_tx(
            conn,
            conversation_id=conversation,
            session_id=session,
            current=current,
            bootstrap=False,
        )
        if (
            fence is None
            or str(fence["session_id"] or "") != session
            or int(fence["publishable"] or 0) != 1
            or int(fence["generation"] or 0)
            != int(row["session_fence_generation"] or 0)
        ):
            return supersede("session_reset_or_rebound")
        lifecycle_status = self._publication_lifecycle_status_under_tx(
            conn,
            conversation_id=conversation,
            session_id=session,
            expected_frontier_store_id=frontier,
        )
        if lifecycle_status == "temporarily_unbound":
            return self._defer_prepared_publication_under_tx(
                conn,
                row,
                owner=owner,
                generation=generation,
                current=current,
            )
        if lifecycle_status != "active":
            return supersede("frontier_mismatch")
        raw_messages = self._load_source_range(
            conn,
            session_id=session,
            conversation_id=conversation,
            frontier=frontier,
            watermark=group_ids[-1],
        )
        raw_ids = tuple(int(message["store_id"]) for message in raw_messages)
        groups = self._atomic_groups(raw_messages)
        raw_identity = self._source_identity_hash(raw_messages)
        if (
            raw_ids != group_ids
            or len(groups) != 1
            or tuple(int(message["store_id"]) for message in groups[0]) != group_ids
            or not self._atomic_group_is_complete(groups[0])
        ):
            return supersede("partial_repair_source_range_changed")
        if (
            raw_identity != str(row["source_identity_hash"] or "")
            or raw_identity != str(row["repair_group_identity_hash"] or "")
        ):
            return supersede("source_identity_mismatch")
        ownership = self._live_owner_map(
            conn,
            session_id=session,
            source_ids=group_ids,
        )
        if ownership is None:
            return supersede("partial_repair_owner_limit_exceeded")
        owner_map, owner_ids = ownership
        if (
            tuple(item for item in group_ids if owner_map[item]) != covered_ids
            or tuple(item for item in group_ids if not owner_map[item]) != uncovered_ids
            or owner_ids != _load_ids(row["repair_owner_node_ids"])
            or self._owner_map_json(owner_map)
            != str(row["repair_owner_map"] or "{}")
            or not self._repair_coverage_elections_valid(
                conn,
                session_id=session,
                owner_map=owner_map,
                covered_ids=covered_ids,
                uncovered_ids=uncovered_ids,
            )
        ):
            return supersede("partial_repair_ownership_changed")
        if self._session_dag_epoch(conn, session) != int(
            row["repair_bound_dag_epoch"] or 0
        ):
            return supersede("partial_repair_dag_epoch_changed")
        topology = self._resolve_partial_repair_topology(
            conn,
            session_id=session,
            owner_ids=owner_ids,
        )
        if (
            topology is None
            or tuple(topology["root_ids"]) != _load_ids(row["repair_root_node_ids"])
            or tuple(topology["ancestry_ids"])
            != _load_ids(row["repair_ancestry_node_ids"])
            or str(topology["topology_hash"])
            != str(row["repair_topology_hash"] or "")
            or str(topology["root_identity_hash"])
            != str(row["repair_root_identity_hash"] or "")
        ):
            return supersede("partial_repair_topology_changed")
        prepared_rows = conn.execute(
            """
            SELECT * FROM lcm_compaction_prepared_nodes
            WHERE job_id=? ORDER BY ordinal
            """,
            (job_id,),
        ).fetchall()
        if (
            len(prepared_rows) != 1
            or int(row["expected_node_count"] or 0) != 1
            or _load_ids(prepared_rows[0]["source_store_ids"]) != group_ids
        ):
            return supersede("prepared_lineage_mismatch")

        commit_current = float(self._now() if now is None else now)
        lease_row = conn.execute(
            """
            SELECT 1 FROM lcm_compaction_jobs
            WHERE job_id=? AND state='prepared' AND lease_owner=?
              AND lease_generation=? AND lease_expires_at>?
            """,
            (job_id, str(owner), int(generation), commit_current),
        ).fetchone()
        if lease_row is None:
            return PromotionResult(False, "lease_lost", job_id)
        by_id = {int(message["store_id"]): message for message in raw_messages}
        bridge_id = self._insert_canonical_node(
            conn,
            session_id=session,
            summary=(
                "[LCM structural provenance bridge for the uncovered atomic-group "
                "continuation; semantic content is represented by its repair parent.]"
            ),
            token_count=1,
            source_token_count=sum(
                self._message_tokens(by_id[store_id]) for store_id in uncovered_ids
            ),
            source_ids=uncovered_ids,
            created_at=commit_current,
            expand_hint="lcm:hidden-structural-provenance-bridge",
        )
        if failure_hook == "after_repair_bridge_insert":
            raise RuntimeError("injected repair publication failure after bridge")
        if callable(failure_hook):
            failure_hook("after_repair_bridge_insert")
        root_ids = tuple(topology["root_ids"])
        parent_id = self._insert_canonical_parent_node(
            conn,
            session_id=session,
            summary=str(prepared_rows[0]["summary"] or ""),
            token_count=int(prepared_rows[0]["summary_token_count"] or 0),
            source_token_count=int(prepared_rows[0]["source_token_count"] or 0),
            child_node_ids=(*root_ids, bridge_id),
            created_at=commit_current,
            expand_hint=self._summary_expand_hint(
                str(prepared_rows[0]["summary"] or "")
            ),
        )
        if failure_hook in {"after_canonical_insert", "after_repair_parent_insert"}:
            raise RuntimeError("injected repair publication failure after parent")
        if callable(failure_hook):
            failure_hook("after_repair_parent_insert")
        placeholders = ",".join("?" for _ in group_ids)
        covered_after = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM lcm_compaction_source_coverage AS coverage
                WHERE coverage.session_id=?
                  AND coverage.store_id IN ({placeholders})
                  AND EXISTS (
                    SELECT 1
                    FROM lcm_compaction_node_sources AS owner_map
                    JOIN summary_nodes AS owner
                      ON owner.node_id=owner_map.node_id
                     AND owner.session_id=owner_map.session_id
                     AND owner.source_type='messages'
                    WHERE owner_map.session_id=coverage.session_id
                      AND owner_map.store_id=coverage.store_id
                      AND coverage.publication_id IN (
                        'summary-node:' || owner_map.node_id,
                        'legacy-node:' || owner_map.node_id
                      )
                  )
                """,
                (session, *group_ids),
            ).fetchone()[0]
        )
        if covered_after != len(group_ids):
            raise RuntimeError("repair publication did not cover the exact atomic group")
        if not self._advance_frontier_cas(
            conn,
            conversation_id=conversation,
            session_id=session,
            expected_frontier_store_id=frontier,
            captured_watermark_store_id=group_ids[-1],
            debt_tokens=int(row["source_token_count"] or 0),
            current=commit_current,
        ):
            raise RuntimeError("repair publication frontier CAS changed inside transaction")
        completed = conn.execute(
            """
            UPDATE lcm_compaction_jobs
            SET state='published',
                completion_reason='partial_atomic_repair_published',
                published_at=?, updated_at=?, lease_owner=NULL,
                lease_expires_at=NULL, heartbeat_at=NULL
            WHERE job_id=? AND state='prepared' AND lease_owner=?
              AND lease_generation=? AND lease_expires_at>?
            """,
            (
                commit_current,
                commit_current,
                job_id,
                str(owner),
                int(generation),
                commit_current,
            ),
        )
        if completed.rowcount != 1:
            raise RuntimeError("repair publication lease CAS changed inside transaction")
        self._record_publication_telemetry_under_tx(
            conn,
            row=row,
            prepared_rows=prepared_rows,
            published_at=commit_current,
            observed_active_tokens=observed_active_tokens,
            effective_context_window=effective_context_window,
            publication_trigger_tokens=publication_trigger_tokens,
            emergency_trigger_tokens=emergency_trigger_tokens,
            trigger_policy=trigger_policy,
        )
        self._record_event_under_tx(
            conn,
            "partial_atomic_repair_published",
            conversation_id=conversation,
            session_id=session,
            job_id=job_id,
            detail=f"partial_atomic_repair bridge={bridge_id} parent={parent_id}",
            created_at=commit_current,
        )
        return PromotionResult(
            True,
            "partial_atomic_repair_published",
            job_id,
            group_ids[-1],
            [bridge_id, parent_id],
        )

    def publish_prepared(
        self,
        job_id: str,
        *,
        owner: str,
        generation: int,
        policy_fingerprint: str,
        failure_hook: str | Callable[[str], None] | None = None,
        write_timeout_ms: int | None = 100,
        observed_active_tokens: int | None = None,
        effective_context_window: int | None = None,
        publication_trigger_tokens: int | None = None,
        emergency_trigger_tokens: int | None = None,
        trigger_policy: str = "",
        now: float | None = None,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
    ) -> PromotionResult:
        """Promote prepared nodes and the frontier in one strict CAS transaction."""
        with self._publication_transaction(
            str(job_id),
            owner=str(owner),
            generation=int(generation),
            write_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            current = float(self._now() if now is None else now)
            row = conn.execute(
                "SELECT * FROM lcm_compaction_jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
            if row is None:
                return PromotionResult(False, "job_not_found", str(job_id))
            if row["state"] == "published":
                return PromotionResult(
                    False,
                    "already_published",
                    str(job_id),
                    int(row["captured_watermark_store_id"] or 0),
                )
            if row["state"] != "prepared":
                return PromotionResult(False, "job_not_prepared", str(job_id))
            if (
                row["lease_owner"] != str(owner)
                or int(row["lease_generation"] or 0) != int(generation)
                or float(row["lease_expires_at"] or 0.0) <= current
            ):
                return PromotionResult(False, "lease_lost", str(job_id))

            if str(row["job_kind"] or "") == _PARTIAL_REPAIR_JOB_KIND:
                return self._publish_partial_atomic_repair_under_tx(
                    conn,
                    row,
                    owner=str(owner),
                    generation=int(generation),
                    policy_fingerprint=str(policy_fingerprint or ""),
                    failure_hook=failure_hook,
                    observed_active_tokens=observed_active_tokens,
                    effective_context_window=effective_context_window,
                    publication_trigger_tokens=publication_trigger_tokens,
                    emergency_trigger_tokens=emergency_trigger_tokens,
                    trigger_policy=trigger_policy,
                    now=now,
                )

            def supersede(reason: str) -> PromotionResult:
                conn.execute(
                    """
                    UPDATE lcm_compaction_jobs
                    SET state='superseded', completion_reason=?, updated_at=?,
                        lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL
                    WHERE job_id=?
                    """,
                    (reason, current, str(job_id)),
                )
                self._record_event_under_tx(
                    conn,
                    "prepared_summary_superseded",
                    conversation_id=str(row["conversation_id"] or ""),
                    session_id=str(row["session_id"] or ""),
                    job_id=str(job_id),
                    detail=reason,
                    created_at=current,
                )
                return PromotionResult(
                    False,
                    reason,
                    str(job_id),
                    int(row["captured_watermark_store_id"] or 0),
                )

            if str(row["policy_fingerprint"] or "") != str(policy_fingerprint or ""):
                return supersede("policy_mismatch")
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=str(row["conversation_id"]),
                session_id=str(row["session_id"]),
                current=current,
                bootstrap=False,
            )
            if (
                fence is None
                or str(fence["session_id"] or "") != str(row["session_id"])
                or int(fence["publishable"] or 0) != 1
                or int(fence["generation"] or 0)
                != int(row["session_fence_generation"] or 0)
            ):
                return supersede("session_reset_or_rebound")
            lifecycle_status = self._publication_lifecycle_status_under_tx(
                conn,
                conversation_id=str(row["conversation_id"]),
                session_id=str(row["session_id"]),
                expected_frontier_store_id=int(
                    row["expected_frontier_store_id"] or 0
                ),
            )
            if lifecycle_status == "temporarily_unbound":
                return self._defer_prepared_publication_under_tx(
                    conn,
                    row,
                    owner=str(owner),
                    generation=int(generation),
                    current=current,
                )
            if lifecycle_status != "active":
                return supersede("frontier_mismatch")
            source_ids = _load_ids(row["source_store_ids"])
            raw_messages = self._load_source_range(
                conn,
                session_id=str(row["session_id"]),
                conversation_id=str(row["conversation_id"]),
                frontier=int(row["expected_frontier_store_id"] or 0),
                watermark=int(row["captured_watermark_store_id"] or 0),
            )
            raw_ids = tuple(int(message["store_id"]) for message in raw_messages)
            if raw_ids != source_ids:
                return supersede("source_range_changed")
            if self._source_identity_hash(raw_messages) != str(
                row["source_identity_hash"] or ""
            ):
                return supersede("source_identity_mismatch")
            if self._source_ids_already_canonical(conn, row["session_id"], source_ids):
                return supersede("source_range_already_published")
            prepared_rows = conn.execute(
                """
                SELECT * FROM lcm_compaction_prepared_nodes
                WHERE job_id=? ORDER BY ordinal
                """,
                (str(job_id),),
            ).fetchall()
            if len(prepared_rows) != int(row["expected_node_count"] or 0):
                return supersede("prepared_node_count_mismatch")
            combined = tuple(
                store_id
                for prepared_row in prepared_rows
                for store_id in _load_ids(prepared_row["source_store_ids"])
            )
            if combined != source_ids:
                return supersede("prepared_lineage_mismatch")

            # Validation above may include bounded hashing and row decoding.
            # Re-sample time and fence the lease immediately before the first
            # canonical mutation; a worker whose lease expired while validating
            # must leave nodes, frontier, and job state untouched.
            commit_current = float(self._now() if now is None else now)
            lease_row = conn.execute(
                """
                SELECT 1 FROM lcm_compaction_jobs
                WHERE job_id=? AND state='prepared' AND lease_owner=?
                  AND lease_generation=? AND lease_expires_at>?
                """,
                (str(job_id), str(owner), int(generation), commit_current),
            ).fetchone()
            if lease_row is None:
                return PromotionResult(False, "lease_lost", str(job_id))
            node_ids: list[int] = []
            for prepared_row in prepared_rows:
                node_ids.append(
                    self._insert_canonical_node(
                        conn,
                        session_id=row["session_id"],
                        summary=prepared_row["summary"],
                        token_count=int(prepared_row["summary_token_count"] or 0),
                        source_token_count=int(prepared_row["source_token_count"] or 0),
                        source_ids=_load_ids(prepared_row["source_store_ids"]),
                        created_at=commit_current,
                        expand_hint=self._summary_expand_hint(
                            str(prepared_row["summary"] or "")
                        ),
                    )
                )
            if failure_hook == "after_canonical_insert":
                raise RuntimeError("injected background publication failure")
            if callable(failure_hook):
                failure_hook("after_canonical_insert")
            if not self._advance_frontier_cas(
                conn,
                conversation_id=row["conversation_id"],
                session_id=row["session_id"],
                expected_frontier_store_id=int(row["expected_frontier_store_id"] or 0),
                captured_watermark_store_id=int(row["captured_watermark_store_id"] or 0),
                debt_tokens=int(row["source_token_count"] or 0),
                current=commit_current,
            ):
                raise RuntimeError("background publication frontier CAS changed inside transaction")
            publication_reason = (
                "degraded_reference_published"
                if str(row["job_kind"] or "") == _LOSSLESS_RAW_PROJECTION_JOB_KIND
                else "published"
            )
            completed = conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='published', completion_reason=?, published_at=?,
                    updated_at=?, lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL
                WHERE job_id=? AND state='prepared' AND lease_owner=? AND lease_generation=?
                  AND lease_expires_at>?
                """,
                (
                    publication_reason,
                    commit_current,
                    commit_current,
                    str(job_id),
                    str(owner),
                    int(generation),
                    commit_current,
                ),
            )
            if completed.rowcount != 1:
                raise RuntimeError("background publication lease CAS changed inside transaction")
            if str(row["job_kind"] or "") == _LOSSLESS_RAW_PROJECTION_JOB_KIND:
                placeholders = ",".join("?" for _ in source_ids)
                conn.execute(
                    f"""
                    UPDATE lcm_compaction_source_coverage
                    SET publication_kind=?
                    WHERE session_id=? AND store_id IN ({placeholders})
                    """,
                    (
                        _LOSSLESS_RAW_PROJECTION_JOB_KIND,
                        str(row["session_id"] or ""),
                        *source_ids,
                    ),
                )
            self._record_publication_telemetry_under_tx(
                conn,
                row=row,
                prepared_rows=prepared_rows,
                published_at=commit_current,
                observed_active_tokens=observed_active_tokens,
                effective_context_window=effective_context_window,
                publication_trigger_tokens=publication_trigger_tokens,
                emergency_trigger_tokens=emergency_trigger_tokens,
                trigger_policy=trigger_policy,
            )
            self._record_event_under_tx(
                conn,
                (
                    "degraded_reference_published"
                    if publication_reason == "degraded_reference_published"
                    else "prepared_summary_published"
                ),
                conversation_id=str(row["conversation_id"] or ""),
                session_id=str(row["session_id"] or ""),
                job_id=str(job_id),
                detail=f"published_nodes={len(node_ids)}",
                created_at=commit_current,
            )
            return PromotionResult(
                True,
                publication_reason,
                str(job_id),
                int(row["captured_watermark_store_id"] or 0),
                node_ids,
            )

    def find_prepared_job(
        self,
        *,
        conversation_id: str,
        session_id: str | None = None,
        expected_frontier_store_id: int | None = None,
        policy_fingerprint: str | None = None,
        lock_timeout_seconds: float | None = None,
        deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        busy_timeout_ms: int | None = None,
    ) -> CompactionJob | None:
        where = ["job.conversation_id=?", "job.state='prepared'", "fence.publishable=1",
                 "fence.session_id=job.session_id",
                 "fence.generation=job.session_fence_generation"]
        args: list[Any] = [str(conversation_id)]
        if session_id is not None:
            where.append("job.session_id=?")
            args.append(str(session_id))
        if expected_frontier_store_id is not None:
            where.append("job.expected_frontier_store_id=?")
            args.append(int(expected_frontier_store_id))
        if policy_fingerprint is not None:
            where.append("job.policy_fingerprint=?")
            args.append(str(policy_fingerprint))
        with self._bounded_read_connection(
            lock_timeout_seconds=lock_timeout_seconds,
            deadline=deadline,
            deadline_clock=deadline_clock,
            busy_timeout_ms=busy_timeout_ms,
        ) as conn:
            row = conn.execute(
                f"""SELECT job.* FROM lcm_compaction_jobs AS job
                JOIN lcm_compaction_session_fences AS fence
                  ON fence.conversation_id=job.conversation_id
                WHERE {' AND '.join(where)}
                ORDER BY job.prepared_at DESC LIMIT 1""",
                args,
            ).fetchone()
            return self._job_from_row(row)

    def supersede_overlapping(
        self,
        *,
        conversation_id: str,
        new_frontier_store_id: int,
        reason: str = "foreground_emergency_compaction",
        now: float | None = None,
    ) -> int:
        current = float(self._now() if now is None else now)
        with self._write_transaction() as conn:
            affected = conn.execute(
                """
                SELECT job_id, session_id FROM lcm_compaction_jobs
                WHERE conversation_id=?
                  AND state IN ('queued','running','prepared','retry')
                  AND expected_frontier_store_id<?
                """,
                (
                    str(conversation_id),
                    max(0, int(new_frontier_store_id or 0)),
                ),
            ).fetchall()
            cursor = conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='superseded', completion_reason=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    lease_generation=lease_generation+1
                WHERE conversation_id=?
                  AND state IN ('queued','running','prepared','retry')
                  AND expected_frontier_store_id<?
                """,
                (
                    _sanitize_error(reason),
                    current,
                    str(conversation_id),
                    max(0, int(new_frontier_store_id or 0)),
                ),
            )
            for row in affected:
                self._record_event_under_tx(
                    conn,
                    "prepared_summary_superseded",
                    conversation_id=str(conversation_id),
                    session_id=str(row["session_id"] or ""),
                    job_id=str(row["job_id"] or ""),
                    detail=reason,
                    created_at=current,
                )
            return int(cursor.rowcount or 0)

    @staticmethod
    def _node_value(node: Any, name: str, default: Any = None) -> Any:
        if isinstance(node, Mapping):
            return node.get(name, default)
        return getattr(node, name, default)

    def publish_foreground_node(
        self,
        node: Any,
        *,
        conversation_id: str,
        session_id: str,
        expected_session_fence_generation: int,
        expected_frontier_store_id: int,
        captured_watermark_store_id: int,
        expected_source_identity_hash: str,
        debt_tokens: int = 0,
        reason: str = "foreground_emergency_compaction",
        write_timeout_ms: int | None = 100,
        lock_timeout_seconds: float | None = None,
        write_deadline: float | None = None,
        deadline_clock: Callable[[], float] | None = None,
        now: float | None = None,
    ) -> PromotionResult:
        """Publish an emergency foreground node with the same frontier fence."""
        current = float(self._now() if now is None else now)
        source_ids = tuple(
            int(item) for item in (self._node_value(node, "source_ids", ()) or ())
        )
        with self._write_transaction(
            busy_timeout_ms=write_timeout_ms,
            lock_timeout_seconds=lock_timeout_seconds,
            write_deadline=write_deadline,
            deadline_clock=deadline_clock,
        ) as conn:
            fence = self._session_fence_under_tx(
                conn,
                conversation_id=str(conversation_id),
                session_id=str(session_id),
                current=current,
                bootstrap=False,
            )
            if (
                fence is None
                or str(fence["session_id"] or "") != str(session_id)
                or int(fence["publishable"] or 0) != 1
                or int(fence["generation"] or 0)
                != int(expected_session_fence_generation)
            ):
                return PromotionResult(False, "session_reset_or_rebound")
            lifecycle = conn.execute(
                """
                SELECT current_session_id, current_frontier_store_id
                FROM lcm_lifecycle_state WHERE conversation_id=?
                """,
                (str(conversation_id),),
            ).fetchone()
            if (
                lifecycle is None
                or lifecycle["current_session_id"] != str(session_id)
                or int(lifecycle["current_frontier_store_id"] or 0)
                != int(expected_frontier_store_id or 0)
            ):
                return PromotionResult(False, "frontier_mismatch")
            raw_ids = tuple(
                int(row[0])
                for row in conn.execute(
                    """
                    SELECT store_id FROM messages
                    WHERE session_id=? AND store_id>? AND store_id<=?
                      AND (conversation_id=? OR conversation_id='' OR conversation_id IS NULL)
                    ORDER BY store_id
                    """,
                    (
                        str(session_id),
                        int(expected_frontier_store_id or 0),
                        int(captured_watermark_store_id or 0),
                        str(conversation_id),
                    ),
                ).fetchall()
            )
            if not raw_ids or raw_ids != source_ids:
                return PromotionResult(False, "source_range_changed")
            raw_messages = self._load_source_range(
                conn,
                session_id=str(session_id),
                conversation_id=str(conversation_id),
                frontier=int(expected_frontier_store_id or 0),
                watermark=int(captured_watermark_store_id or 0),
            )
            if (
                not str(expected_source_identity_hash or "")
                or self._source_identity_hash(raw_messages)
                != str(expected_source_identity_hash)
            ):
                return PromotionResult(False, "source_identity_mismatch")
            canonical_source_ids = source_ids
            if self._source_ids_already_canonical(
                conn,
                session_id,
                canonical_source_ids,
            ):
                return PromotionResult(False, "source_range_already_published")
            node_id = self._insert_canonical_node(
                conn,
                session_id=session_id,
                summary=str(self._node_value(node, "summary", "") or ""),
                token_count=int(self._node_value(node, "token_count", 0) or 0),
                source_token_count=int(
                    self._node_value(node, "source_token_count", debt_tokens) or 0
                ),
                source_ids=canonical_source_ids,
                created_at=float(self._node_value(node, "created_at", current) or current),
                expand_hint=str(self._node_value(node, "expand_hint", "") or ""),
            )
            if not self._advance_frontier_cas(
                conn,
                conversation_id=conversation_id,
                session_id=session_id,
                expected_frontier_store_id=expected_frontier_store_id,
                captured_watermark_store_id=captured_watermark_store_id,
                debt_tokens=debt_tokens,
                current=current,
            ):
                raise RuntimeError("foreground publication frontier CAS changed inside transaction")
            affected = conn.execute(
                """
                SELECT job_id, session_id FROM lcm_compaction_jobs
                WHERE conversation_id=?
                  AND state IN ('queued','running','prepared','retry')
                  AND expected_frontier_store_id<?
                """,
                (
                    str(conversation_id),
                    int(captured_watermark_store_id or 0),
                ),
            ).fetchall()
            conn.execute(
                """
                UPDATE lcm_compaction_jobs
                SET state='superseded', completion_reason=?, updated_at=?,
                    lease_owner=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                    lease_generation=lease_generation+1
                WHERE conversation_id=?
                  AND state IN ('queued','running','prepared','retry')
                  AND expected_frontier_store_id<?
                """,
                (
                    _sanitize_error(reason),
                    current,
                    str(conversation_id),
                    int(captured_watermark_store_id or 0),
                ),
            )
            for row in affected:
                self._record_event_under_tx(
                    conn,
                    "prepared_summary_superseded",
                    conversation_id=str(conversation_id),
                    session_id=str(row["session_id"] or ""),
                    job_id=str(row["job_id"] or ""),
                    detail=reason,
                    created_at=current,
                )
            self._record_event_under_tx(
                conn,
                "foreground_emergency_compaction",
                conversation_id=str(conversation_id),
                session_id=str(session_id),
                detail=reason,
                created_at=current,
            )
            return PromotionResult(
                True,
                str(reason or "foreground_emergency_compaction"),
                captured_watermark_store_id=int(captured_watermark_store_id or 0),
                node_ids=[node_id],
            )

    def _serialize_chunk(self, chunk: SourceChunk) -> str:
        if chunk.serialized_text is not None:
            return chunk.serialized_text
        return serialize_summary_messages(
            chunk.messages,
            config=self.config,
            session_id=(
                str(chunk.messages[0].get("session_id") or "")
                if chunk.messages
                else ""
            ),
            hermes_home=self._hermes_home,
        )

    @staticmethod
    def _callable_accepts_keyword(function: Callable[..., Any], name: str) -> bool:
        """Return whether an injected callable explicitly accepts a safe kwarg."""
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            return False
        parameter = parameters.get(str(name))
        return bool(
            parameter is not None
            and parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        ) or any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        )

    @staticmethod
    def _provider_failure_reason(error: BaseException | str) -> str:
        """Classify an error without retaining its potentially sensitive text."""
        if isinstance(error, TimeoutError):
            return "provider_timeout"
        status = getattr(error, "status_code", None)
        if status is None:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
        try:
            status_code = int(status) if status is not None else 0
        except (TypeError, ValueError, OverflowError):
            status_code = 0
        if status_code in {401, 403}:
            return "provider_auth_error"
        if status_code == 402:
            return "provider_quota"
        if status_code == 429:
            return "provider_rate_limit"
        lowered_type = type(error).__name__.lower()
        lowered = str(error or "").lower()[:500]
        if "timeout" in lowered_type or "timed out" in lowered or "deadline" in lowered:
            return "provider_timeout"
        if any(marker in lowered_type for marker in ("connection", "network")) or any(
            marker in lowered
            for marker in (
                "connection refused",
                "connection reset",
                "connection error",
                "network unreachable",
                "name resolution",
                "dns",
            )
        ):
            return "provider_connection_error"
        return "provider_error"

    def _record_physical_attempt_receipt(
        self,
        *,
        job: CompactionJob,
        chunk_ordinal: int,
        event: Mapping[str, Any],
        serialized_input_estimate: int,
    ) -> None:
        """Durably upsert one physical send start or settlement event.

        The physical id is allocated before transport.  A committed ``started``
        row therefore remains truthful when the process dies before settlement;
        later settlement for the same id updates that row and never creates a
        second counted attempt.
        """
        physical_id = str(event.get("physical_attempt_id") or "").strip()[:200]
        if not physical_id:
            return

        def nullable_int(name: str) -> int | None:
            value = event.get(name)
            if value is None:
                return None
            try:
                return max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                return None

        started_at = event.get("started_at")
        finished_at = event.get("finished_at")
        try:
            started_at_value = max(0.0, float(started_at or self._now()))
        except (TypeError, ValueError, OverflowError):
            started_at_value = float(self._now())
        try:
            finished_at_value = (
                max(started_at_value, float(finished_at))
                if finished_at is not None and float(finished_at) > 0
                else None
            )
        except (TypeError, ValueError, OverflowError):
            finished_at_value = None
        send_state = (
            "settled" if str(event.get("send_state") or "") == "settled" else "started"
        )
        transport_outcome = str(event.get("outcome") or "started")
        if transport_outcome not in {"started", "success", "failure"}:
            transport_outcome = "failure" if send_state == "settled" else "started"
        transport_reason = str(event.get("failure_reason") or "")[:64]
        requested_provider = str(
            event.get("requested_provider") or event.get("provider") or ""
        )[:64]
        requested_model = str(
            event.get("requested_model") or event.get("model") or ""
        )[:160]
        served_provider = str(event.get("provider") or "")[:64]
        served_model = str(event.get("served_model") or event.get("model") or "")[:160]
        with self._write_transaction() as conn:
            if conn.execute(
                "SELECT 1 FROM lcm_compaction_jobs WHERE job_id=?",
                (str(job.job_id),),
            ).fetchone() is None:
                return
            conn.execute(
                """
                INSERT INTO lcm_compaction_attempt_receipts(
                    physical_attempt_id, job_id, job_attempt, chunk_ordinal,
                    escalation_level, logical_route_ordinal, send_state,
                    transport_outcome, transport_reason, summary_outcome,
                    summary_reason, requested_provider, requested_model,
                    served_provider, served_model, effective_timeout_ms,
                    timeout_source, source_version, config_version,
                    prompt_version, projection_version, provider_input_total,
                    provider_output_total, cache_read_input, cache_write_input,
                    reasoning_output, normalized_input, normalized_output,
                    usage_convention, usage_normalization_version,
                    usage_coverage, source_input_estimate,
                    serialized_input_estimate, framed_input_estimate,
                    estimate_method, finish_reason, latency_ms, started_at,
                    finished_at
                ) VALUES(
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?
                )
                ON CONFLICT(physical_attempt_id) DO UPDATE SET
                    send_state=CASE
                        WHEN lcm_compaction_attempt_receipts.send_state='settled'
                        THEN 'settled' ELSE excluded.send_state END,
                    transport_outcome=CASE
                        WHEN excluded.send_state='settled'
                        THEN excluded.transport_outcome
                        ELSE lcm_compaction_attempt_receipts.transport_outcome END,
                    transport_reason=CASE
                        WHEN excluded.send_state='settled'
                        THEN excluded.transport_reason
                        ELSE lcm_compaction_attempt_receipts.transport_reason END,
                    served_provider=CASE
                        WHEN excluded.send_state='settled'
                        THEN excluded.served_provider
                        ELSE lcm_compaction_attempt_receipts.served_provider END,
                    served_model=CASE
                        WHEN excluded.send_state='settled'
                        THEN excluded.served_model
                        ELSE lcm_compaction_attempt_receipts.served_model END,
                    effective_timeout_ms=COALESCE(
                        excluded.effective_timeout_ms,
                        lcm_compaction_attempt_receipts.effective_timeout_ms
                    ),
                    timeout_source=excluded.timeout_source,
                    provider_input_total=CASE WHEN excluded.send_state='settled'
                        THEN excluded.provider_input_total
                        ELSE lcm_compaction_attempt_receipts.provider_input_total END,
                    provider_output_total=CASE WHEN excluded.send_state='settled'
                        THEN excluded.provider_output_total
                        ELSE lcm_compaction_attempt_receipts.provider_output_total END,
                    cache_read_input=CASE WHEN excluded.send_state='settled'
                        THEN excluded.cache_read_input
                        ELSE lcm_compaction_attempt_receipts.cache_read_input END,
                    cache_write_input=CASE WHEN excluded.send_state='settled'
                        THEN excluded.cache_write_input
                        ELSE lcm_compaction_attempt_receipts.cache_write_input END,
                    reasoning_output=CASE WHEN excluded.send_state='settled'
                        THEN excluded.reasoning_output
                        ELSE lcm_compaction_attempt_receipts.reasoning_output END,
                    normalized_input=CASE WHEN excluded.send_state='settled'
                        THEN excluded.normalized_input
                        ELSE lcm_compaction_attempt_receipts.normalized_input END,
                    normalized_output=CASE WHEN excluded.send_state='settled'
                        THEN excluded.normalized_output
                        ELSE lcm_compaction_attempt_receipts.normalized_output END,
                    usage_convention=CASE WHEN excluded.send_state='settled'
                        THEN excluded.usage_convention
                        ELSE lcm_compaction_attempt_receipts.usage_convention END,
                    usage_normalization_version=CASE WHEN excluded.send_state='settled'
                        THEN excluded.usage_normalization_version
                        ELSE lcm_compaction_attempt_receipts.usage_normalization_version END,
                    usage_coverage=CASE WHEN excluded.send_state='settled'
                        THEN excluded.usage_coverage
                        ELSE lcm_compaction_attempt_receipts.usage_coverage END,
                    framed_input_estimate=COALESCE(
                        excluded.framed_input_estimate,
                        lcm_compaction_attempt_receipts.framed_input_estimate
                    ),
                    estimate_method=excluded.estimate_method,
                    finish_reason=CASE WHEN excluded.send_state='settled'
                        THEN excluded.finish_reason
                        ELSE lcm_compaction_attempt_receipts.finish_reason END,
                    latency_ms=CASE WHEN excluded.send_state='settled'
                        THEN excluded.latency_ms
                        ELSE lcm_compaction_attempt_receipts.latency_ms END,
                    finished_at=CASE WHEN excluded.send_state='settled'
                        THEN excluded.finished_at
                        ELSE lcm_compaction_attempt_receipts.finished_at END
                """,
                (
                    physical_id,
                    str(job.job_id),
                    max(1, int(job.attempt_count or 1)),
                    max(0, int(chunk_ordinal)),
                    nullable_int("escalation_level") or 0,
                    nullable_int("logical_route_ordinal") or 0,
                    send_state,
                    transport_outcome,
                    transport_reason,
                    "pending",
                    "",
                    requested_provider,
                    requested_model,
                    served_provider,
                    served_model,
                    nullable_int("effective_timeout_ms"),
                    str(event.get("timeout_source") or "")[:160],
                    LCM_PHYSICAL_RECEIPT_SCHEMA_VERSION,
                    str(job.policy_fingerprint or "")[:160],
                    LCM_SUMMARY_PROMPT_VERSION,
                    LCM_ACTIVE_REPLAY_PROJECTION_VERSION,
                    nullable_int("provider_input_total"),
                    nullable_int("provider_output_total"),
                    nullable_int("cache_read_input"),
                    nullable_int("cache_write_input"),
                    nullable_int("reasoning_output"),
                    nullable_int("normalized_input"),
                    nullable_int("normalized_output"),
                    str(event.get("usage_convention") or "unknown")[:64],
                    str(event.get("usage_normalization_version") or "")[:64],
                    str(event.get("usage_coverage") or "unavailable")[:32],
                    nullable_int("source_input_estimate"),
                    max(0, int(serialized_input_estimate or 0)),
                    nullable_int("framed_input_estimate"),
                    str(event.get("estimate_method") or "lcm-tokenizer-v1")[:64],
                    str(event.get("finish_reason") or "")[:64],
                    nullable_int("latency_ms") or 0,
                    started_at_value,
                    finished_at_value,
                ),
            )

    def _record_compaction_attempts(
        self,
        *,
        job: CompactionJob,
        chunk_ordinal: int,
        source_token_count: int,
        trace: Mapping[str, Any] | None,
        fallback_reason: str = "",
        fallback_latency_ms: int = 0,
    ) -> None:
        """Persist allowlisted logical route outcomes; never provider content."""
        current = float(self._now())
        allowed_outcomes = {"success", "failure", "rejected", "skipped"}
        allowed_reasons = set(_EVENT_REASON_CODES) | {"provider_error"}
        terminal_reason = str(fallback_reason or "")
        if terminal_reason and terminal_reason not in allowed_reasons:
            terminal_reason = "provider_error"
        raw_attempts = (
            list(trace.get("attempts") or [])
            if isinstance(trace, Mapping)
            else []
        )
        attempts = [dict(item) for item in raw_attempts if isinstance(item, Mapping)]
        guard_only_terminal = bool(
            attempts
            and terminal_reason in {"provider_circuit_open", "provider_spend_guard"}
            and all(str(item.get("outcome") or "") == "skipped" for item in attempts)
        )
        if attempts and terminal_reason and not guard_only_terminal:
            # The runner's terminal observation wins over an incomplete trace
            # snapshot or a provider success that arrived after the absolute
            # deadline / durable lease loss.  Reclassifying the final logical
            # route keeps doctor from reporting that unusable result as the
            # latest success without inventing an extra provider call.
            terminal_attempt = attempts[-1]
            terminal_attempt.update(
                {
                    "outcome": "failure",
                    "failure_reason": terminal_reason,
                    "summary_token_count": 0,
                    "latency_ms": max(0, int(fallback_latency_ms or 0)),
                    "finished_at": current,
                }
            )
            terminal_attempt.setdefault(
                "started_at",
                current - (max(0, int(fallback_latency_ms or 0)) / 1_000.0),
            )
        elif not attempts and terminal_reason:
            attempts = [
                {
                    "escalation_level": 0,
                    "logical_route_ordinal": 0,
                    "actual_provider": "",
                    "actual_model": "",
                    "attribution_available": False,
                    "fallback_used": None,
                    "outcome": "failure",
                    "failure_reason": terminal_reason,
                    "source_token_count": source_token_count,
                    "summary_token_count": 0,
                    "latency_ms": fallback_latency_ms,
                    "started_at": current - (max(0, fallback_latency_ms) / 1_000.0),
                    "finished_at": current,
                }
            ]
        if not attempts:
            return
        with self._write_transaction() as conn:
            if conn.execute(
                "SELECT 1 FROM lcm_compaction_jobs WHERE job_id=?",
                (str(job.job_id),),
            ).fetchone() is None:
                return
            for fallback_ordinal, attempt in enumerate(attempts):
                outcome = str(attempt.get("outcome") or "provider_error")
                failure_reason = str(attempt.get("failure_reason") or "")
                if outcome not in allowed_outcomes:
                    outcome = "failure"
                if failure_reason not in allowed_reasons:
                    failure_reason = "" if outcome == "success" else outcome
                started_at = attempt.get("started_at")
                finished_at = attempt.get("finished_at")
                try:
                    started_at = float(started_at)
                except (TypeError, ValueError, OverflowError):
                    started_at = current
                try:
                    finished_at = float(finished_at)
                except (TypeError, ValueError, OverflowError):
                    finished_at = current
                if started_at <= 0 or started_at > current + 86_400:
                    started_at = current
                if finished_at <= 0 or finished_at > current + 86_400:
                    finished_at = current
                finished_at = max(started_at, finished_at)
                fallback_used = attempt.get("fallback_used")
                conn.execute(
                    """
                    INSERT INTO lcm_compaction_attempts(
                        job_id, job_attempt, chunk_ordinal, escalation_level,
                        logical_route_ordinal, actual_provider, actual_model,
                        attribution_available, fallback_used, outcome,
                        failure_reason, source_token_count, summary_token_count,
                        latency_ms, started_at, finished_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(
                        job_id, job_attempt, chunk_ordinal,
                        escalation_level, logical_route_ordinal
                    ) DO UPDATE SET
                        actual_provider=excluded.actual_provider,
                        actual_model=excluded.actual_model,
                        attribution_available=excluded.attribution_available,
                        fallback_used=excluded.fallback_used,
                        outcome=excluded.outcome,
                        failure_reason=excluded.failure_reason,
                        source_token_count=excluded.source_token_count,
                        summary_token_count=excluded.summary_token_count,
                        latency_ms=excluded.latency_ms,
                        started_at=excluded.started_at,
                        finished_at=excluded.finished_at
                    """,
                    (
                        str(job.job_id),
                        max(1, int(job.attempt_count or 1)),
                        max(0, int(chunk_ordinal)),
                        max(0, int(attempt.get("escalation_level") or 0)),
                        max(
                            0,
                            int(
                                attempt.get("logical_route_ordinal")
                                if attempt.get("logical_route_ordinal") is not None
                                else fallback_ordinal
                            ),
                        ),
                        str(attempt.get("actual_provider") or "")[:64],
                        str(attempt.get("actual_model") or "")[:160],
                        int(bool(attempt.get("attribution_available"))),
                        None if fallback_used is None else int(bool(fallback_used)),
                        outcome,
                        failure_reason,
                        max(
                            0,
                            int(
                                attempt.get("source_token_count")
                                or source_token_count
                                or 0
                            ),
                        ),
                        max(0, int(attempt.get("summary_token_count") or 0)),
                        max(0, int(attempt.get("latency_ms") or 0)),
                        started_at,
                        finished_at,
                    ),
                )
                physical_attempt_id = str(
                    attempt.get("physical_attempt_id") or ""
                ).strip()[:200]
                if physical_attempt_id:
                    conn.execute(
                        """
                        UPDATE lcm_compaction_attempt_receipts
                        SET summary_outcome=?, summary_reason=?
                        WHERE physical_attempt_id=?
                        """,
                        (outcome, failure_reason, physical_attempt_id),
                    )

    def _try_record_compaction_attempts(self, **kwargs: Any) -> None:
        """Persist optional attempt telemetry without owning job correctness."""
        try:
            self._record_compaction_attempts(**kwargs)
        except BaseException as exc:
            logger.warning(
                "LCM compaction attempt telemetry persistence failed; error=%s",
                _sanitize_error(exc),
            )

    def run_next_job(
        self,
        owner: str,
        summarizer: Callable[..., Any] | None = None,
        circuit_breaker: Any = None,
        spend_guard: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> CompactionJob | None:
        """Claim and prepare one job while keeping all model I/O outside SQL.

        ``cancel_event`` is a caller-owned shutdown signal.  This method
        observes it but never sets it; provider calls receive a distinct
        job-local event that also carries deadlines and lease loss.
        """
        current_policy = background_policy_fingerprint(self.config)
        claimed = self.claim_next_job(
            owner=owner,
            policy_fingerprint=current_policy,
        )
        if claimed is None:
            return None
        permit_cleanup: dict[str, Any] = {
            "guardian_owned": False,
            "provider_completed": None,
            "release": None,
            "start_guardian": None,
        }
        try:
            return self._run_claimed_job(
                claimed,
                owner=owner,
                summarizer=summarizer,
                circuit_breaker=circuit_breaker,
                spend_guard=spend_guard,
                cancel_event=cancel_event,
                permit_cleanup=permit_cleanup,
            )
        finally:
            if not bool(permit_cleanup.get("guardian_owned")):
                completed = permit_cleanup.get("provider_completed")
                start_guardian = permit_cleanup.get("start_guardian")
                if (
                    isinstance(completed, threading.Event)
                    and not completed.is_set()
                    and callable(start_guardian)
                ):
                    start_guardian()
                else:
                    release = permit_cleanup.get("release")
                    if callable(release):
                        release()
                    else:
                        try:
                            self.release_provider_permit(
                                claimed.job_id,
                                owner=owner,
                                generation=claimed.lease_generation,
                            )
                        except BaseException as exc:
                            logger.warning(
                                "LCM provider-permit cleanup failed; error=%s",
                                _sanitize_error(exc),
                            )
                            provider_completed = threading.Event()
                            provider_completed.set()
                            self._start_provider_permit_guardian(
                                provider_completed,
                                job_id=claimed.job_id,
                                owner=owner,
                                generation=claimed.lease_generation,
                            )

    def _run_claimed_job(
        self,
        claimed: CompactionJob,
        *,
        owner: str,
        summarizer: Callable[..., Any] | None,
        circuit_breaker: Any,
        spend_guard: Any,
        cancel_event: threading.Event | None,
        permit_cleanup: dict[str, Any],
    ) -> CompactionJob | None:
        """Run one durable claim under ``run_next_job``'s cleanup invariant."""
        deadline_seconds = max(
            1.0,
            float(
                getattr(self.config, "background_compaction_job_timeout_seconds", 120)
                or 120
            ),
        )
        # The durable claim starts the wall-clock budget.  Source validation is
        # part of the job, not free work before the advertised timeout.
        deadline = time.monotonic() + deadline_seconds
        # A job deadline is a cancellation boundary for this provider call,
        # not a shutdown request for the persistent runtime that owns it.
        # Keep the provider-facing event local and only observe the caller's
        # event as an upstream stop signal; job timeout/lease-loss paths must
        # never mutate that caller-owned state.
        shutdown_event = cancel_event
        provider_cancel_event = threading.Event()

        def cancellation_requested() -> bool:
            if provider_cancel_event.is_set():
                return True
            if shutdown_event is not None and shutdown_event.is_set():
                provider_cancel_event.set()
                return True
            return False

        def release_unused_permit() -> None:
            try:
                self.release_provider_permit(
                    claimed.job_id,
                    owner=owner,
                    generation=claimed.lease_generation,
                )
            except BaseException as exc:
                logger.warning(
                    "LCM could not release an unused provider permit; error=%s",
                    _sanitize_error(exc),
                )
                provider_completed = threading.Event()
                provider_completed.set()
                self._start_provider_permit_guardian(
                    provider_completed,
                    job_id=claimed.job_id,
                    owner=owner,
                    generation=claimed.lease_generation,
                )
                permit_cleanup["guardian_owned"] = True

        permit_cleanup["release"] = release_unused_permit

        def heartbeat_before_provider(*, activate_provider: bool = False) -> bool:
            try:
                valid = self.heartbeat(
                    claimed.job_id,
                    owner=owner,
                    generation=claimed.lease_generation,
                    provider_active=activate_provider,
                )
            except BaseException:
                release_unused_permit()
                raise
            if not valid:
                release_unused_permit()
            return valid

        try:
            if claimed.job_kind == _PARTIAL_REPAIR_JOB_KIND:
                bound = self._bind_partial_atomic_repair(
                    claimed.job_id,
                    owner=owner,
                    generation=claimed.lease_generation,
                )
                if bound is None:
                    release_unused_permit()
                    return self.get_job(claimed.job_id)
                claimed = bound
                chunks = self._partial_repair_source_chunks(
                    claimed,
                    owner=owner,
                    generation=claimed.lease_generation,
                )
            else:
                chunks = self.source_chunks(claimed.job_id)
        except BaseException:
            release_unused_permit()
            raise
        if not heartbeat_before_provider():
            return self.get_job(claimed.job_id)
        max_passes = max(
            1, int(getattr(self.config, "background_compaction_max_passes", 4) or 4)
        )
        if not chunks or len(chunks) > max_passes:
            try:
                retry = self.mark_retry(
                    claimed.job_id,
                    owner=owner,
                    generation=claimed.lease_generation,
                    error=RuntimeError("no eligible durable source chunks"),
                    completion_reason="no_eligible_raw_range",
                )
            except BaseException:
                release_unused_permit()
                raise
            if not isinstance(retry, CompactionJob):
                release_unused_permit()
            return retry if isinstance(retry, CompactionJob) else self.get_job(claimed.job_id)
        leaf_chunk_count = len(chunks)
        combine_prepared_leaves = bool(
            getattr(self.config, "cache_friendly_condensation_enabled", False)
            and leaf_chunk_count > 1
            and leaf_chunk_count + 1 <= max_passes
        )

        if summarizer is None:
            from .escalation import summarize_with_escalation

            summarizer = summarize_with_escalation

        lease_seconds = max(
            1.0,
            float(
                getattr(self.config, "background_compaction_lease_seconds", 300)
                or 300
            ),
        )
        heartbeat_interval = max(0.25, min(lease_seconds / 3.0, 30.0))
        model = str(getattr(self.config, "summary_model", "") or "")
        try:
            existing_by_ordinal = {
                node.ordinal: node
                for node in self.get_prepared_nodes(claimed.job_id)
            }
        except BaseException:
            release_unused_permit()
            raise
        prepared_nodes: list[PreparedSummaryNode] = []

        def append_aggregation_chunk_if_ready() -> None:
            if (
                not combine_prepared_leaves
                or len(prepared_nodes) != leaf_chunk_count
                or len(chunks) != leaf_chunk_count
            ):
                return
            aggregation_text = "\n\n---\n\n".join(
                f"[Leaf Summary {index + 1}]\n{prepared.summary}"
                for index, prepared in enumerate(prepared_nodes)
            )
            chunks.append(
                SourceChunk(
                    ordinal=leaf_chunk_count,
                    source_store_ids=tuple(
                        store_id
                        for prepared in prepared_nodes
                        for store_id in prepared.source_store_ids
                    ),
                    source_token_count=sum(
                        prepared.source_token_count for prepared in prepared_nodes
                    ),
                    messages=(
                        {
                            "role": "user",
                            "content": aggregation_text,
                            "session_id": claimed.session_id,
                        },
                    ),
                )
            )

        for chunk in chunks:
            aggregation_pass = bool(
                combine_prepared_leaves and chunk.ordinal == leaf_chunk_count
            )
            existing = existing_by_ordinal.get(chunk.ordinal)
            if existing is not None and existing.source_store_ids == chunk.source_store_ids:
                if not heartbeat_before_provider():
                    return self.get_job(claimed.job_id)
                prepared_nodes.append(existing)
                append_aggregation_chunk_if_ready()
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0 or cancellation_requested():
                provider_cancel_event.set()
                try:
                    retry = self.mark_retry(
                        claimed.job_id,
                        owner=owner,
                        generation=claimed.lease_generation,
                        error=TimeoutError(
                            "background compaction job deadline exhausted"
                        ),
                        completion_reason="provider_timeout",
                    )
                except BaseException:
                    release_unused_permit()
                    raise
                if not isinstance(retry, CompactionJob):
                    release_unused_permit()
                return retry if isinstance(retry, CompactionJob) else self.get_job(claimed.job_id)

            completed = threading.Event()
            result_box: dict[str, Any] = {}
            serialized_chunk = self._serialize_chunk(chunk)
            serialized_input_estimate = max(0, int(count_tokens(serialized_chunk)))

            def receipt_sink(
                event: Mapping[str, Any],
                *,
                _job: CompactionJob = claimed,
                _chunk_ordinal: int = chunk.ordinal,
                _serialized_estimate: int = serialized_input_estimate,
            ) -> None:
                payload = dict(event)
                payload["serialized_input_estimate"] = _serialized_estimate
                payload["timeout_source"] = str(
                    getattr(self.config, "config_sources", {}).get(
                        "summary_timeout_ms",
                        payload.get("timeout_source") or "request.timeout",
                    )
                )
                self._record_physical_attempt_receipt(
                    job=_job,
                    chunk_ordinal=_chunk_ordinal,
                    event=payload,
                    serialized_input_estimate=_serialized_estimate,
                )

            route_trace: dict[str, Any] = {
                "attempts": [],
                "receipt_sink": receipt_sink,
            }
            provider_started = time.monotonic()
            source_tokens = max(1, int(chunk.source_token_count or 0))
            token_budget = min(12_000, max(2_000, int(source_tokens * 0.20)))
            configured_timeout = max(
                0.001,
                float(getattr(self.config, "summary_timeout_ms", 30_000) or 30_000)
                / 1000.0,
            )
            completion_owner_lock = threading.Lock()
            completion_owner: dict[str, Any] = {
                "provider_finished": False,
                "finalizer": None,
                "finalizer_claimed": False,
            }

            def arm_provider_completion_finalizer(
                finalizer: Callable[[], None],
                *,
                _owner: dict[str, Any] = completion_owner,
                _lock: threading.Lock = completion_owner_lock,
            ) -> None:
                """Transfer failed guardian ownership exactly once."""
                run_finalizer: Callable[[], None] | None = None
                with _lock:
                    if _owner["finalizer"] is not None or _owner["finalizer_claimed"]:
                        return
                    _owner["finalizer"] = finalizer
                    if _owner["provider_finished"]:
                        _owner["finalizer_claimed"] = True
                        run_finalizer = finalizer
                if run_finalizer is not None:
                    run_finalizer()

            def notify_provider_finished(
                *,
                _owner: dict[str, Any] = completion_owner,
                _lock: threading.Lock = completion_owner_lock,
            ) -> None:
                run_finalizer: Callable[[], None] | None = None
                with _lock:
                    _owner["provider_finished"] = True
                    if (
                        _owner["finalizer"] is not None
                        and not _owner["finalizer_claimed"]
                    ):
                        _owner["finalizer_claimed"] = True
                        run_finalizer = _owner["finalizer"]
                if run_finalizer is not None:
                    run_finalizer()

            def start_provider_permit_guardian() -> bool:
                if bool(permit_cleanup.get("guardian_owned")):
                    return True
                try:
                    started = self._start_provider_permit_guardian(
                        completed,
                        job_id=claimed.job_id,
                        owner=owner,
                        generation=claimed.lease_generation,
                        on_start_failure=arm_provider_completion_finalizer,
                    )
                except BaseException:
                    permit_cleanup["guardian_owned"] = False
                    raise
                permit_cleanup["guardian_owned"] = True
                return started

            permit_cleanup["provider_completed"] = completed
            permit_cleanup["start_guardian"] = start_provider_permit_guardian

            def summarize_one(
                _notify_provider_finished: Callable[[], None] = (
                    notify_provider_finished
                ),
            ) -> None:
                try:
                    provider_remaining = max(0.001, deadline - time.monotonic())
                    summarize_kwargs: dict[str, Any] = {
                        "text": serialized_chunk,
                        "source_tokens": source_tokens,
                        "token_budget": token_budget,
                        "depth": 1 if aggregation_pass else 0,
                        "model": model,
                        "fallback_models": getattr(
                            self.config, "summary_fallback_models", None
                        ),
                        "circuit_breaker": circuit_breaker,
                        "spend_guard": spend_guard,
                        "timeout": min(configured_timeout, provider_remaining),
                        "l2_budget_ratio": float(
                            getattr(self.config, "l2_budget_ratio", 0.5) or 0.5
                        ),
                        "l3_truncate_tokens": int(
                            getattr(self.config, "l3_truncate_tokens", 512) or 512
                        ),
                        "custom_instructions": str(
                            getattr(self.config, "custom_instructions", "") or ""
                        ),
                    }
                    if self._callable_accepts_keyword(summarizer, "trace"):
                        summarize_kwargs["trace"] = route_trace
                    if self._callable_accepts_keyword(summarizer, "deadline"):
                        summarize_kwargs["deadline"] = deadline
                    if self._callable_accepts_keyword(summarizer, "cancel_event"):
                        summarize_kwargs["cancel_event"] = provider_cancel_event
                    result_box["value"] = summarizer(**summarize_kwargs)
                except BaseException as exc:
                    result_box["error"] = exc
                finally:
                    result_box["completed_at"] = time.monotonic()
                    completed.set()
                    _notify_provider_finished()

            if not heartbeat_before_provider(activate_provider=True):
                return self.get_job(claimed.job_id)
            try:
                start_background_thread(
                    target=summarize_one,
                    name=f"lcm-bg-provider-{claimed.job_id[:8]}-{chunk.ordinal}",
                )
            except BaseException:
                release_unused_permit()
                raise
            lease_valid = True
            next_heartbeat = time.monotonic() + heartbeat_interval
            try:
                while not completed.is_set():
                    monotonic_now = time.monotonic()
                    if (
                        monotonic_now >= deadline
                        or cancellation_requested()
                    ):
                        provider_cancel_event.set()
                        break
                    completed.wait(timeout=min(0.25, deadline - monotonic_now))
                    if completed.is_set():
                        break
                    if time.monotonic() >= next_heartbeat:
                        lease_valid = self.heartbeat(
                            claimed.job_id,
                            owner=owner,
                            generation=claimed.lease_generation,
                        )
                        if not lease_valid:
                            provider_cancel_event.set()
                            break
                        next_heartbeat = time.monotonic() + heartbeat_interval
            except BaseException:
                provider_cancel_event.set()
                if completed.is_set():
                    release_unused_permit()
                else:
                    start_provider_permit_guardian()
                raise

            completed_at = float(result_box.get("completed_at") or 0.0)
            timed_out = (
                cancellation_requested()
                or not completed.is_set()
                or completed_at > deadline
            )
            if not lease_valid:
                provider_cancel_event.set()
                self._try_record_compaction_attempts(
                    job=claimed,
                    chunk_ordinal=chunk.ordinal,
                    source_token_count=source_tokens,
                    trace=route_trace,
                    fallback_reason="stale_lease",
                    fallback_latency_ms=max(
                        0,
                        int((time.monotonic() - provider_started) * 1_000.0),
                    ),
                )
                if not completed.is_set():
                    start_provider_permit_guardian()
                else:
                    release_unused_permit()
                return self.get_job(claimed.job_id)
            if timed_out:
                provider_cancel_event.set()
                provider_still_running = not completed.is_set()
                elapsed_ms = max(
                    0,
                    int((time.monotonic() - provider_started) * 1_000.0),
                )
                self._try_record_compaction_attempts(
                    job=claimed,
                    chunk_ordinal=chunk.ordinal,
                    source_token_count=source_tokens,
                    trace=route_trace,
                    fallback_reason="provider_timeout",
                    fallback_latency_ms=elapsed_ms,
                )
                try:
                    retry = self.mark_retry(
                        claimed.job_id,
                        owner=owner,
                        generation=claimed.lease_generation,
                        error=TimeoutError(
                            "background compaction provider deadline exceeded"
                        ),
                        keep_provider_permit=provider_still_running,
                        completion_reason="provider_timeout",
                    )
                except BaseException:
                    if provider_still_running:
                        start_provider_permit_guardian()
                    else:
                        release_unused_permit()
                    raise
                if provider_still_running:
                    start_provider_permit_guardian()
                return retry if isinstance(retry, CompactionJob) else self.get_job(claimed.job_id)
            if "error" in result_box:
                failure_reason = self._provider_failure_reason(result_box["error"])
                elapsed_ms = max(
                    0,
                    int((time.monotonic() - provider_started) * 1_000.0),
                )
                self._try_record_compaction_attempts(
                    job=claimed,
                    chunk_ordinal=chunk.ordinal,
                    source_token_count=source_tokens,
                    trace=route_trace,
                    fallback_reason=failure_reason,
                    fallback_latency_ms=elapsed_ms,
                )
                try:
                    retry = self.mark_retry(
                        claimed.job_id,
                        owner=owner,
                        generation=claimed.lease_generation,
                        error=result_box["error"],
                        completion_reason=failure_reason,
                    )
                except BaseException:
                    release_unused_permit()
                    raise
                if not isinstance(retry, CompactionJob):
                    release_unused_permit()
                return retry if isinstance(retry, CompactionJob) else self.get_job(claimed.job_id)

            value = result_box.get("value")
            if isinstance(value, tuple):
                summary = str(value[0] or "")
                level = int(value[1] or 0) if len(value) > 1 else 0
            else:
                summary = str(value or "")
                level = 0
            elapsed_ms = max(
                0,
                int((time.monotonic() - provider_started) * 1_000.0),
            )
            terminal_reason = str(route_trace.get("terminal_reason") or "")
            if terminal_reason not in _EVENT_REASON_CODES:
                terminal_reason = ""
            trace_attempts = [
                item
                for item in list(route_trace.get("attempts") or [])
                if isinstance(item, Mapping)
            ]
            last_trace_attempt = trace_attempts[-1] if trace_attempts else None
            deterministic_l3_terminal = bool(
                terminal_reason == "non_compressing_output"
                and last_trace_attempt is not None
                and str(last_trace_attempt.get("outcome") or "") == "rejected"
                and not bool(last_trace_attempt.get("attribution_available"))
                and not any(
                    str(item.get("outcome") or "") == "success"
                    for item in trace_attempts
                )
            )
            if deterministic_l3_terminal:
                transient_failure = next(
                    (
                        str(item.get("failure_reason") or "")
                        for item in reversed(trace_attempts[:-1])
                        if str(item.get("outcome") or "") == "failure"
                        and str(item.get("failure_reason") or "")
                        in _TRANSIENT_RANGE_DISPOSITION_REASONS
                    ),
                    "",
                )
                if transient_failure:
                    terminal_reason = transient_failure
            self._try_record_compaction_attempts(
                job=claimed,
                chunk_ordinal=chunk.ordinal,
                source_token_count=source_tokens,
                trace=route_trace,
                fallback_reason=(
                    terminal_reason
                    if terminal_reason
                    else ("non_compressing_output" if not summary or level == 3 else "")
                ),
                fallback_latency_ms=elapsed_ms,
            )
            if not summary or level == 3:
                error = RuntimeError(
                    "background summarizer returned no publishable provider summary"
                )
                provider_attempted = any(
                    isinstance(item, Mapping)
                    and str(item.get("outcome") or "") != "skipped"
                    and bool(
                        item.get("attribution_available")
                        or item.get("actual_provider")
                    )
                    for item in list(route_trace.get("attempts") or [])
                )
                try:
                    retry = self.mark_retry(
                        claimed.job_id,
                        owner=owner,
                        generation=claimed.lease_generation,
                        error=error,
                        completion_reason=(terminal_reason or "non_compressing_output"),
                        terminal=(
                            not terminal_reason
                            or terminal_reason
                            in {"non_compressing_output", "reasoning_only_output"}
                        ),
                        provider_attempted=provider_attempted,
                    )
                except BaseException:
                    release_unused_permit()
                    raise
                if not isinstance(retry, CompactionJob):
                    release_unused_permit()
                return retry if isinstance(retry, CompactionJob) else self.get_job(claimed.job_id)
            successful_attempt = next(
                (
                    item
                    for item in reversed(list(route_trace.get("attempts") or []))
                    if isinstance(item, Mapping)
                    and str(item.get("outcome") or "") == "success"
                ),
                None,
            )
            creation_provider = (
                str(successful_attempt.get("actual_provider") or "")
                if successful_attempt is not None
                else ""
            )
            creation_model = (
                str(successful_attempt.get("actual_model") or "")
                if successful_attempt is not None
                else ""
            )
            attribution_available = bool(
                successful_attempt is not None
                and successful_attempt.get("attribution_available")
                and creation_provider
                and creation_model
            )
            fallback_used = (
                successful_attempt.get("fallback_used")
                if successful_attempt is not None
                else None
            )
            node = PreparedSummaryNode(
                ordinal=chunk.ordinal,
                summary=summary,
                source_store_ids=chunk.source_store_ids,
                source_token_count=source_tokens,
                summary_token_count=max(1, int(count_tokens(summary))),
                creation_model=creation_model,
                creation_provider=creation_provider,
                route_latency_ms=(
                    max(0, int(successful_attempt.get("latency_ms") or 0))
                    if successful_attempt is not None
                    else elapsed_ms
                ),
                attempt_number=max(1, int(claimed.attempt_count or 1)),
                escalation_level=(
                    max(0, int(successful_attempt.get("escalation_level") or level))
                    if successful_attempt is not None
                    else max(0, level)
                ),
                attribution_available=attribution_available,
                fallback_used=(
                    None if fallback_used is None else bool(fallback_used)
                ),
            )
            try:
                saved = self.save_prepared_progress(
                    claimed.job_id,
                    owner=owner,
                    generation=claimed.lease_generation,
                    prepared_node=node,
                )
            except BaseException:
                release_unused_permit()
                raise
            if not saved:
                release_unused_permit()
                return self.get_job(claimed.job_id)
            prepared_nodes.append(node)
            # Spend the reserved final pass to collapse all prepared leaves
            # into one cache-stable node.  Raw lineage remains the exact
            # ordered union; the intermediate leaves stay inactive progress
            # records and are replaced transactionally by mark_prepared.
            append_aggregation_chunk_if_ready()

        if combine_prepared_leaves:
            aggregate = next(
                (
                    node
                    for node in prepared_nodes
                    if node.ordinal == leaf_chunk_count
                ),
                None,
            )
            if aggregate is None:
                try:
                    retry = self.mark_retry(
                        claimed.job_id,
                        owner=owner,
                        generation=claimed.lease_generation,
                        error=RuntimeError(
                            "cache-friendly aggregation pass was not prepared"
                        ),
                    )
                except BaseException:
                    release_unused_permit()
                    raise
                if not isinstance(retry, CompactionJob):
                    release_unused_permit()
                return retry if isinstance(retry, CompactionJob) else self.get_job(claimed.job_id)
            prepared_nodes = [
                PreparedSummaryNode(
                    ordinal=0,
                    summary=aggregate.summary,
                    source_store_ids=aggregate.source_store_ids,
                    source_token_count=aggregate.source_token_count,
                    summary_token_count=aggregate.summary_token_count,
                    creation_model=aggregate.creation_model,
                    creation_provider=aggregate.creation_provider,
                    route_latency_ms=aggregate.route_latency_ms,
                    attempt_number=aggregate.attempt_number,
                    escalation_level=aggregate.escalation_level,
                    attribution_available=aggregate.attribution_available,
                    fallback_used=aggregate.fallback_used,
                )
            ]

        try:
            prepared = self.mark_prepared(
                claimed.job_id,
                owner=owner,
                generation=claimed.lease_generation,
                prepared_nodes=prepared_nodes,
                expected_node_count=len(prepared_nodes),
            )
        except BaseException:
            release_unused_permit()
            raise
        if not isinstance(prepared, CompactionJob):
            release_unused_permit()
        return prepared if isinstance(prepared, CompactionJob) else self.get_job(claimed.job_id)

    def _start_provider_permit_guardian(
        self,
        completed: threading.Event,
        *,
        job_id: str,
        owner: str,
        generation: int,
        on_start_failure: Callable[[Callable[[], None]], None] | None = None,
    ) -> bool:
        """Fence a timeout-ignoring provider until it physically returns.

        ``on_start_failure`` transfers the one-shot finalizer to the physical
        provider thread when this helper thread cannot start.  Callers that
        own such a thread must arrange to invoke the supplied finalizer only
        after the provider has actually returned.
        """
        db_path = str(self.db_path)
        config = self.config
        lease_seconds = max(
            1.0,
            float(
                getattr(config, "background_compaction_lease_seconds", 300) or 300
            ),
        )
        heartbeat_interval = max(0.25, min(lease_seconds / 3.0, 30.0))
        retry_initial = 0.25
        retry_maximum = max(
            retry_initial,
            min(5.0, max(1.0, heartbeat_interval)),
        )
        log_interval = max(5.0, min(60.0, heartbeat_interval))

        def guard() -> None:
            guardian: BackgroundCompactionStore | None = None
            retry_interval = retry_initial
            last_log_at: float | None = None
            suppressed_logs = 0

            def wait_before_retry() -> float:
                nonlocal retry_interval
                delay = retry_interval
                retry_interval = min(retry_maximum, retry_interval * 2.0)
                if completed.is_set():
                    time.sleep(delay)
                else:
                    completed.wait(timeout=delay)
                return delay

            def reset_retry_interval() -> None:
                nonlocal retry_interval
                retry_interval = retry_initial

            def log_retry(phase: str, error: BaseException, delay: float) -> None:
                nonlocal last_log_at, suppressed_logs
                current = time.monotonic()
                if last_log_at is not None and current - last_log_at < log_interval:
                    suppressed_logs += 1
                    return
                logger.warning(
                    "LCM provider-permit guardian %s failed; retrying in %.2fs; "
                    "error=%s; suppressed=%d",
                    phase,
                    delay,
                    _sanitize_error(error),
                    suppressed_logs,
                )
                last_log_at = current
                suppressed_logs = 0

            def open_guardian() -> BackgroundCompactionStore:
                nonlocal guardian
                if guardian is not None:
                    return guardian
                while guardian is None:
                    try:
                        guardian = BackgroundCompactionStore(db_path, config=config)
                    except BaseException as exc:
                        delay = retry_interval
                        log_retry("open", exc, delay)
                        wait_before_retry()
                reset_retry_interval()
                return guardian

            try:
                while not completed.is_set():
                    current_guardian = open_guardian()
                    if completed.wait(timeout=heartbeat_interval):
                        break
                    try:
                        held = current_guardian.heartbeat_provider_permit(
                            job_id,
                            owner=owner,
                            generation=generation,
                            lease_seconds=lease_seconds,
                        )
                    except BaseException as exc:
                        delay = retry_interval
                        log_retry("heartbeat", exc, delay)
                        wait_before_retry()
                        continue
                    reset_retry_interval()
                    if not held:
                        return

                # Completion is not finished until the durable row is gone.
                # In particular, a set completion event must not turn a locked
                # release into a one-shot best effort.
                while True:
                    current_guardian = open_guardian()
                    try:
                        current_guardian.release_provider_permit(
                            job_id,
                            owner=owner,
                            generation=generation,
                        )
                        return
                    except BaseException as exc:
                        delay = retry_interval
                        log_retry("release", exc, delay)
                        wait_before_retry()
            finally:
                if guardian is not None:
                    guardian.close()
                _notify_background_runtime(db_path)

        try:
            start_background_thread(
                target=guard,
                name=f"lcm-bg-permit-guardian-{str(job_id)[:8]}",
            )
        except BaseException as exc:
            logger.error(
                "LCM provider-permit guardian thread start failed; error=%s",
                _sanitize_error(exc),
            )

            def start_fallback_guardian() -> None:
                """Start the retrying guardian without blocking its caller."""
                try:
                    start_background_thread(
                        target=guard,
                        name=f"lcm-bg-permit-guardian-{str(job_id)[:8]}",
                        low_level=_thread.start_new_thread,
                    )
                except BaseException as fallback_exc:
                    logger.error(
                        "LCM provider-permit fallback guardian start failed; error=%s",
                        _sanitize_error(fallback_exc),
                    )

            if on_start_failure is not None:
                on_start_failure(start_fallback_guardian)
            elif completed.is_set():
                # No live provider owns this permit, so cleanup can begin
                # immediately, but its durable retry loop must still run away
                # from the foreground/provider caller.
                start_fallback_guardian()
            return False
        return True

    def status(
        self,
        *,
        conversation_id: str | None = None,
        publication_trigger_tokens: int = 0,
        emergency_trigger_tokens: int = 0,
    ) -> dict[str, Any]:
        current = float(self._now())
        enabled = bool(getattr(self.config, "background_compaction_enabled", False))
        available = _background_runtime_available(self.db_path) if enabled else False
        if enabled and not available:
            session_id = ""
            if conversation_id:
                with self._lock:
                    if self._conn is not None:
                        lifecycle_session = self._conn.execute(
                            """
                            SELECT current_session_id FROM lcm_lifecycle_state
                            WHERE conversation_id=?
                            """,
                            (str(conversation_id),),
                        ).fetchone()
                        if lifecycle_session is not None:
                            session_id = str(lifecycle_session[0] or "")
            try:
                self.record_diagnostic_event(
                    "worker_unavailable",
                    conversation_id=str(conversation_id or ""),
                    session_id=session_id,
                    detail="background compaction runtime is unavailable",
                    now=current,
                    deduplicate_seconds=60.0,
                )
            except sqlite3.Error:
                logger.debug(
                    "LCM could not persist worker availability diagnostics",
                    exc_info=True,
                )

        with self._lock:
            conn = self._conn
            if conn is None:
                raise RuntimeError("background compaction store is closed")
            counts = {
                str(row["state"]): int(row["count"])
                for row in conn.execute(
                    "SELECT state, COUNT(*) AS count FROM lcm_compaction_jobs GROUP BY state"
                ).fetchall()
            }
            publication_kind_args: tuple[Any, ...] = (
                (str(conversation_id),) if conversation_id else ()
            )
            publication_kind_scope = (
                " AND conversation_id=?" if conversation_id else ""
            )
            publication_kind_row = conn.execute(
                f"""
                SELECT
                  COALESCE(SUM(CASE WHEN job_kind=? THEN 1 ELSE 0 END),0)
                    AS reference_count,
                  COALESCE(SUM(CASE WHEN job_kind<>? THEN 1 ELSE 0 END),0)
                    AS summary_count
                FROM lcm_compaction_jobs
                WHERE state='published'{publication_kind_scope}
                """,
                (
                    _LOSSLESS_RAW_PROJECTION_JOB_KIND,
                    _LOSSLESS_RAW_PROJECTION_JOB_KIND,
                    *publication_kind_args,
                ),
            ).fetchone()
            oldest = conn.execute(
                "SELECT MIN(created_at) FROM lcm_compaction_jobs WHERE state='queued'"
            ).fetchone()[0]
            current_row = None
            if conversation_id:
                current_row = conn.execute(
                    """
                    SELECT * FROM lcm_compaction_jobs
                    WHERE conversation_id=?
                    ORDER BY CASE WHEN state IN ('queued','running','prepared','retry') THEN 0 ELSE 1 END,
                             updated_at DESC
                    LIMIT 1
                    """,
                    (str(conversation_id),),
                ).fetchone()
            lifecycle = None
            session_fence = None
            if conversation_id:
                lifecycle = conn.execute(
                    "SELECT * FROM lcm_lifecycle_state WHERE conversation_id=?",
                    (str(conversation_id),),
                ).fetchone()
                session_fence = conn.execute(
                    """
                    SELECT session_id, generation, publishable, updated_at
                    FROM lcm_compaction_session_fences
                    WHERE conversation_id=?
                    """,
                    (str(conversation_id),),
                ).fetchone()
            scope_clause = " AND conversation_id=?" if conversation_id else ""
            scope_args: tuple[Any, ...] = (
                (str(conversation_id),) if conversation_id else ()
            )
            last_prepared = conn.execute(
                f"""
                SELECT * FROM lcm_compaction_jobs
                WHERE prepared_at IS NOT NULL{scope_clause}
                ORDER BY prepared_at DESC LIMIT 1
                """,
                scope_args,
            ).fetchone()
            last_published = conn.execute(
                f"""
                SELECT * FROM lcm_compaction_jobs
                WHERE state IN ('published','superseded'){scope_clause}
                ORDER BY updated_at DESC LIMIT 1
                """,
                scope_args,
            ).fetchone()
            event_where = "WHERE conversation_id=?" if conversation_id else ""
            event_args: tuple[Any, ...] = (
                (str(conversation_id),) if conversation_id else ()
            )
            event_counts = {
                str(row["reason_code"]): int(row["count"])
                for row in conn.execute(
                    f"""
                    SELECT reason_code, COUNT(*) AS count
                    FROM lcm_compaction_events {event_where}
                    GROUP BY reason_code
                    """,
                    event_args,
                ).fetchall()
            }
            permit_where = "lease_expires_at>?"
            permit_args: list[Any] = [current]
            if conversation_id:
                permit_where += " AND conversation_id=?"
                permit_args.append(str(conversation_id))
            permit_row = conn.execute(
                f"""
                SELECT COUNT(*) AS active_count,
                       COALESCE(SUM(CASE WHEN timed_out_at IS NOT NULL THEN 1 ELSE 0 END),0)
                           AS timed_out_count,
                       MIN(started_at) AS oldest_started_at,
                       MIN(lease_expires_at) AS minimum_lease_expires_at
                FROM lcm_compaction_provider_leases
                WHERE {permit_where}
                """,
                permit_args,
            ).fetchone()
            publication_where = ""
            publication_args: tuple[Any, ...] = ()
            if conversation_id:
                publication_where = "WHERE conversation_id=?"
                publication_args = (str(conversation_id),)
            publication_telemetry_row = conn.execute(
                f"""
                SELECT * FROM lcm_compaction_publications
                {publication_where}
                ORDER BY published_at DESC LIMIT 1
                """,
                publication_args,
            ).fetchone()
            attempt_where = ""
            attempt_args: list[Any] = []
            if conversation_id:
                attempt_where = "WHERE job.conversation_id=?"
                attempt_args.append(str(conversation_id))
            last_attempt_row = conn.execute(
                f"""
                SELECT attempt.*
                FROM lcm_compaction_attempts AS attempt
                JOIN lcm_compaction_jobs AS job ON job.job_id=attempt.job_id
                {attempt_where}
                ORDER BY attempt.finished_at DESC, attempt.attempt_id DESC
                LIMIT 1
                """,
                attempt_args,
            ).fetchone()
            recent_where = "attempt.finished_at>=?"
            recent_args: list[Any] = [current - 86_400.0]
            if conversation_id:
                recent_where += " AND job.conversation_id=?"
                recent_args.append(str(conversation_id))
            recent_provider_outcomes = {
                str(result["reason"]): int(result["count"] or 0)
                for result in conn.execute(
                    f"""
                    SELECT CASE
                             WHEN attempt.outcome='success' THEN 'success'
                             ELSE COALESCE(
                                 NULLIF(attempt.failure_reason,''),
                                 attempt.outcome
                             )
                           END AS reason,
                           COUNT(*) AS count
                    FROM lcm_compaction_attempts AS attempt
                    JOIN lcm_compaction_jobs AS job
                      ON job.job_id=attempt.job_id
                    WHERE {recent_where}
                    GROUP BY reason
                    """,
                    recent_args,
                ).fetchall()
            }
            attribution_where = ""
            attribution_args: tuple[Any, ...] = ()
            if conversation_id:
                attribution_where = "WHERE job.conversation_id=?"
                attribution_args = (str(conversation_id),)
            attribution_row = conn.execute(
                f"""
                SELECT COUNT(*) AS prepared_nodes,
                       COALESCE(SUM(node.attribution_available),0)
                           AS attributed_nodes
                FROM lcm_compaction_prepared_nodes AS node
                JOIN lcm_compaction_jobs AS job ON job.job_id=node.job_id
                {attribution_where}
                """,
                attribution_args,
            ).fetchone()
            disposition_where = "WHERE resolved_at IS NULL"
            disposition_args: list[Any] = []
            if conversation_id:
                disposition_where += " AND conversation_id=?"
                disposition_args.append(str(conversation_id))
            range_dispositions = {
                str(result["reason_code"]): int(result["count"] or 0)
                for result in conn.execute(
                    f"""
                    SELECT reason_code, COUNT(*) AS count
                    FROM lcm_compaction_range_dispositions
                    {disposition_where}
                    GROUP BY reason_code
                    """,
                    disposition_args,
                ).fetchall()
            }

        current_job = None
        if current_row is not None:
            lease_active = bool(
                current_row["lease_owner"]
                and current_row["lease_expires_at"] is not None
                and float(current_row["lease_expires_at"]) > current
            )
            remaining = None
            if lease_active:
                remaining = max(0.0, float(current_row["lease_expires_at"]) - current)
            current_job = {
                "job_id": str(current_row["job_id"]),
                "state": str(current_row["state"]),
                "expected_frontier_store_id": int(current_row["expected_frontier_store_id"] or 0),
                "captured_watermark_store_id": int(current_row["captured_watermark_store_id"] or 0),
                "debt_tokens": int(current_row["source_token_count"] or 0),
                "attempt_count": int(current_row["attempt_count"] or 0),
                "lease_owner_hint": (
                    _owner_hint(current_row["lease_owner"]) if lease_active else None
                ),
                "lease_remaining_seconds": remaining,
                "next_retry_at": current_row["next_retry_at"],
                "last_error": _sanitize_error(current_row["last_error"]),
            }

        def result_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
            if row is None:
                return None
            return {
                "job_id": str(row["job_id"]),
                "status": str(row["state"]),
                "reason": str(row["completion_reason"] or row["state"]),
                "at": float(row["updated_at"] or 0.0),
                "expected_frontier_store_id": int(row["expected_frontier_store_id"] or 0),
                "captured_watermark_store_id": int(row["captured_watermark_store_id"] or 0),
                "debt_tokens": int(row["source_token_count"] or 0),
            }

        reasons: list[str] = list(event_counts)
        if not conversation_id and counts.get("prepared", 0):
            reasons.append("background_summary_prepared")
        if not conversation_id and counts.get("published", 0):
            reasons.append("prepared_summary_published")
        if not conversation_id and counts.get("superseded", 0):
            reasons.append("prepared_summary_superseded")
        if any(
            row_reason == "lease_expired"
            for row_reason in (
                current_row["completion_reason"] if current_row is not None else None,
                last_published["completion_reason"] if last_published is not None else None,
            )
        ):
            reasons.append("lease_expiration")
        if current_row is not None and "timeout" in str(current_row["last_error"] or "").lower():
            reasons.append("provider_timeout")
        if enabled and not available:
            reasons.append("worker_unavailable")
        frontier = int(lifecycle["current_frontier_store_id"] or 0) if lifecycle else 0
        if current_row is not None and str(current_row["state"] or "") in _ACTIVE_STATES:
            debt = int(current_row["source_token_count"] or 0)
        elif lifecycle is not None:
            debt = int(lifecycle["debt_size_estimate"] or 0)
        else:
            debt = int(current_row["source_token_count"] or 0) if current_row is not None else 0
        active_provider_permits = int(permit_row["active_count"] or 0)
        timed_out_provider_permits = int(permit_row["timed_out_count"] or 0)
        lease_seconds = max(
            1.0,
            float(
                getattr(self.config, "background_compaction_lease_seconds", 300)
                or 300
            ),
        )
        oldest_permit_age = None
        if permit_row["oldest_started_at"] is not None:
            oldest_permit_age = min(
                max(lease_seconds, 86_400.0),
                max(0.0, current - float(permit_row["oldest_started_at"])),
            )
        minimum_permit_remaining = None
        if permit_row["minimum_lease_expires_at"] is not None:
            minimum_permit_remaining = min(
                lease_seconds,
                max(0.0, float(permit_row["minimum_lease_expires_at"]) - current),
            )

        last_publication_telemetry = None
        if publication_telemetry_row is not None:
            wait_ms = publication_telemetry_row["prepared_to_published_ms"]
            last_publication_telemetry = {
                "job_id": str(publication_telemetry_row["job_id"] or ""),
                "range_identity": str(
                    publication_telemetry_row["range_identity"] or ""
                ),
                "publication_fence_generation": int(
                    publication_telemetry_row["publication_fence_generation"] or 0
                ),
                "observed_active_tokens": publication_telemetry_row[
                    "observed_active_tokens"
                ],
                "effective_context_window": publication_telemetry_row[
                    "effective_context_window"
                ],
                "publication_trigger_tokens": publication_telemetry_row[
                    "publication_trigger_tokens"
                ],
                "emergency_trigger_tokens": publication_telemetry_row[
                    "emergency_trigger_tokens"
                ],
                "trigger_policy": str(
                    publication_telemetry_row["trigger_policy"] or ""
                ),
                "prepared_to_published_seconds": (
                    max(0.0, float(wait_ms) / 1_000.0)
                    if wait_ms is not None
                    else None
                ),
                "source_token_count": int(
                    publication_telemetry_row["source_token_count"] or 0
                ),
                "summary_token_count": int(
                    publication_telemetry_row["summary_token_count"] or 0
                ),
                "provider": str(publication_telemetry_row["provider"] or ""),
                "model": str(publication_telemetry_row["model"] or ""),
                "published_at": float(
                    publication_telemetry_row["published_at"] or 0.0
                ),
            }
        last_provider_attempt = None
        if last_attempt_row is not None:
            last_provider_attempt = {
                "job_id": str(last_attempt_row["job_id"] or ""),
                "attempt_number": int(last_attempt_row["job_attempt"] or 0),
                "chunk_ordinal": int(last_attempt_row["chunk_ordinal"] or 0),
                "escalation_level": int(
                    last_attempt_row["escalation_level"] or 0
                ),
                "logical_route_ordinal": int(
                    last_attempt_row["logical_route_ordinal"] or 0
                ),
                "provider": str(last_attempt_row["actual_provider"] or ""),
                "model": str(last_attempt_row["actual_model"] or ""),
                "attribution_available": bool(
                    last_attempt_row["attribution_available"]
                ),
                "fallback_used": (
                    None
                    if last_attempt_row["fallback_used"] is None
                    else bool(last_attempt_row["fallback_used"])
                ),
                "outcome": str(last_attempt_row["outcome"] or ""),
                "failure_reason": str(last_attempt_row["failure_reason"] or ""),
                "source_token_count": int(
                    last_attempt_row["source_token_count"] or 0
                ),
                "summary_token_count": int(
                    last_attempt_row["summary_token_count"] or 0
                ),
                "latency_ms": int(last_attempt_row["latency_ms"] or 0),
                "finished_at": float(last_attempt_row["finished_at"] or 0.0),
            }
        return {
            "enabled": enabled,
            "worker_count": max(
                0, int(getattr(self.config, "background_compaction_workers", 1) or 0)
            ),
            "global_limit": min(
                _MAX_GLOBAL_PROVIDER_PERMITS,
                max(
                    0,
                    int(
                        getattr(
                            self.config,
                            "background_compaction_global_limit",
                            _MAX_GLOBAL_PROVIDER_PERMITS,
                        )
                        or 0
                    ),
                ),
            ),
            "worker_available": available,
            "queued_jobs": counts.get("queued", 0),
            "running_jobs": counts.get("running", 0),
            "prepared_jobs": counts.get("prepared", 0),
            "retry_jobs": counts.get("retry", 0),
            "published_jobs": counts.get("published", 0),
            "published_summary_jobs": int(
                publication_kind_row["summary_count"] or 0
            ),
            "published_reference_jobs": int(
                publication_kind_row["reference_count"] or 0
            ),
            "superseded_jobs": counts.get("superseded", 0),
            "failed_jobs": counts.get("failed", 0),
            "active_provider_permits": active_provider_permits,
            "active_non_timed_out_provider_permits": max(
                0, active_provider_permits - timed_out_provider_permits
            ),
            "timed_out_provider_permits": timed_out_provider_permits,
            "provider_permit_oldest_age_seconds": oldest_permit_age,
            "provider_permit_min_lease_remaining_seconds": minimum_permit_remaining,
            "oldest_queued_job_age_seconds": (
                max(0.0, current - float(oldest)) if oldest is not None else None
            ),
            "current_job": current_job,
            "publication_fence": (
                {
                    "generation": int(session_fence["generation"] or 0),
                    "publishable": bool(session_fence["publishable"]),
                    "updated_at": float(session_fence["updated_at"] or 0.0),
                }
                if session_fence is not None
                else None
            ),
            "current_debt_tokens": max(0, debt),
            "last_preparation_result": result_payload(last_prepared),
            "last_publication_result": result_payload(last_published),
            "last_publication_telemetry": last_publication_telemetry,
            "last_provider_attempt": last_provider_attempt,
            "recent_provider_outcomes": recent_provider_outcomes,
            "provider_attribution": {
                "prepared_nodes": int(attribution_row["prepared_nodes"] or 0),
                "attributed_nodes": int(attribution_row["attributed_nodes"] or 0),
            },
            "range_dispositions": range_dispositions,
            "durable_event_counts": event_counts,
            "durable_event_reasons": sorted(event_counts),
            "backlog_trigger_tokens": max(
                0,
                int(
                    getattr(
                        self.config,
                        "background_compaction_backlog_tokens",
                        32_000,
                    )
                    or 0
                ),
            ),
            "minimum_provider_source_tokens": max(
                1,
                int(
                    getattr(
                        self.config,
                        "background_compaction_min_provider_source_tokens",
                        256,
                    )
                    or 256
                ),
            ),
            "prepare_ratio": max(
                0.0,
                min(
                    1.0,
                    float(
                        getattr(
                            self.config,
                            "background_compaction_prepare_ratio",
                            0.55,
                        )
                        or 0.0
                    ),
                ),
            ),
            "publication_trigger_tokens": max(0, int(publication_trigger_tokens or 0)),
            "emergency_trigger_tokens": max(0, int(emergency_trigger_tokens or 0)),
            "durable_frontier_store_id": frontier,
            "active_coverage": "durable_published" if frontier > 0 else "raw",
            "diagnostic_reasons": sorted(set(reasons)),
        }

    def _next_maintenance_delay(self) -> float | None:
        current = float(self._now())
        with self._lock:
            if self._conn is None:
                return None
            row = self._conn.execute(
                """
                SELECT MIN(wake_at) FROM (
                    SELECT ? AS wake_at
                    WHERE EXISTS (
                        SELECT 1 FROM lcm_compaction_jobs WHERE state='queued'
                    )
                    UNION ALL
                    SELECT next_retry_at AS wake_at
                    FROM lcm_compaction_jobs
                    WHERE state='retry' AND next_retry_at IS NOT NULL
                    UNION ALL
                    SELECT lease_expires_at AS wake_at
                    FROM lcm_compaction_jobs
                    WHERE state IN ('running','prepared') AND lease_expires_at IS NOT NULL
                    UNION ALL
                    SELECT lease_expires_at AS wake_at
                    FROM lcm_compaction_provider_leases
                )
                """,
                (current,),
            ).fetchone()
        if row is None or row[0] is None:
            return 30.0
        delay = max(0.0, float(row[0]) - current)
        return 5.0 if delay <= 0 else min(30.0, delay)


class _BackgroundRuntime:
    def __init__(self, key: str, db_path: str | Path, config: LCMConfig) -> None:
        self.key = key
        self.db_path = str(db_path)
        self.config = config
        self._condition = threading.Condition()
        self._stopping = False
        self._stop_event = threading.Event()
        self._wake_generation = 0
        from .escalation import SummaryCircuitBreaker, SummarySpendGuard

        self._circuit_breaker = SummaryCircuitBreaker(
            failure_threshold=int(
                getattr(config, "summary_circuit_breaker_failure_threshold", 2) or 2
            ),
            cooldown_seconds=float(
                getattr(config, "summary_circuit_breaker_cooldown_seconds", 300)
                or 300
            ),
        )
        self._spend_guard = SummarySpendGuard(
            max_calls=int(getattr(config, "summary_spend_max_calls", 0) or 0),
            window_seconds=float(
                getattr(config, "summary_spend_window_seconds", 3600) or 3600
            ),
            backoff_seconds=float(
                getattr(config, "summary_spend_backoff_seconds", 300) or 300
            ),
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"lcm-background-{hashlib.sha256(key.encode()).hexdigest()[:8]}",
            daemon=True,
        )
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive() and not self._stopping

    def notify(self) -> None:
        with self._condition:
            self._wake_generation += 1
            self._condition.notify_all()

    def request_stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._stop_event.set()
            self._wake_generation += 1
            self._condition.notify_all()

    def wait_stopped(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def stop(self) -> bool:
        self.request_stop()
        return self.wait_stopped(timeout=_RUNTIME_STOP_TIMEOUT_SECONDS)

    def _run(self) -> None:
        owner = f"process-{os.getpid()}-worker-{threading.get_ident()}"
        seen_generation = 0
        while True:
            with self._condition:
                if self._stopping:
                    return
            admission = acquire_background_admission()
            delay = 1.0
            if admission is not None:
                try:
                    delay = self._run_admitted_cycle(owner)
                finally:
                    # Include store.close()/WAL cleanup before declaring idle.
                    admission.release()
                if delay is None:
                    return
            with self._condition:
                if self._stopping:
                    return
                if self._wake_generation != seen_generation:
                    seen_generation = self._wake_generation
                    continue
                self._condition.wait(timeout=delay)
                seen_generation = self._wake_generation

    def _run_admitted_cycle(self, owner: str) -> float | None:
        """Drain available jobs; close the store before waiting without a lease."""
        try:
            store = BackgroundCompactionStore(self.db_path, config=self.config)
        except BaseException:
            logger.exception("LCM background worker could not open its private store")
            return None
        try:
            store.recover_expired_leases()
            while True:
                with self._condition:
                    if self._stopping:
                        return None
                try:
                    job = store.run_next_job(
                        owner=owner,
                        circuit_breaker=self._circuit_breaker,
                        spend_guard=self._spend_guard,
                        cancel_event=self._stop_event,
                    )
                except sqlite3.OperationalError as exc:
                    logger.warning("LCM background worker hit SQLite contention", exc_info=True)
                    try:
                        store.record_diagnostic_event(
                            "sqlite_lock_contention",
                            detail=exc,
                            deduplicate_seconds=60.0,
                        )
                    except sqlite3.Error:
                        logger.debug(
                            "LCM could not persist SQLite contention diagnostics",
                            exc_info=True,
                        )
                    job = None
                except BaseException:
                    logger.exception("LCM background worker failed preparing a job")
                    job = None
                if job is not None and job.state == "prepared":
                    continue
                return store._next_maintenance_delay()
        finally:
            store.close()


@dataclass
class _RuntimeEntry:
    runtime: _BackgroundRuntime
    references: int = 0
    replacement_config: LCMConfig | None = None
    retirement_watcher: threading.Thread | None = None


_RUNTIME_LOCK = threading.RLock()
_RUNTIME_REGISTRY: dict[str, _RuntimeEntry] = {}


def _start_replacement_runtime_locked(
    key: str,
    entry: _RuntimeEntry,
    *,
    fallback_config: LCMConfig,
) -> None:
    """Replace a fully stopped runtime while holding ``_RUNTIME_LOCK``."""
    old_runtime = entry.runtime
    if old_runtime._thread.is_alive():
        raise RuntimeError("cannot replace a live background compaction runtime")
    replacement_config = entry.replacement_config or fallback_config
    replacement = _BackgroundRuntime(key, old_runtime.db_path, replacement_config)
    entry.runtime = replacement
    entry.replacement_config = None
    entry.retirement_watcher = None


def _retire_runtime_and_maybe_replace(
    key: str,
    entry: _RuntimeEntry,
    runtime: _BackgroundRuntime,
) -> None:
    """Wait outside the registry lock, then retire or replace one runtime."""
    runtime.wait_stopped(timeout=None)
    with _RUNTIME_LOCK:
        registered = _RUNTIME_REGISTRY.get(key)
        if registered is not entry or entry.runtime is not runtime:
            return
        entry.retirement_watcher = None
        if entry.references <= 0:
            _RUNTIME_REGISTRY.pop(key, None)
            return
        try:
            _start_replacement_runtime_locked(
                key,
                entry,
                fallback_config=runtime.config,
            )
        except BaseException:
            # A constructor failure must not strand live references forever.
            # Keep one watcher responsible for retrying after a short backoff.
            logger.exception("LCM could not replace a retired background runtime")
            entry.retirement_watcher = threading.Thread(
                target=_retry_runtime_replacement,
                args=(key, entry, runtime),
                name=f"lcm-background-retry-{hashlib.sha256(key.encode()).hexdigest()[:8]}",
                daemon=True,
            )
            entry.retirement_watcher.start()


def _retry_runtime_replacement(
    key: str,
    entry: _RuntimeEntry,
    retired_runtime: _BackgroundRuntime,
) -> None:
    """Retry a rare replacement-constructor failure without duplicating workers."""
    while True:
        time.sleep(1.0)
        with _RUNTIME_LOCK:
            registered = _RUNTIME_REGISTRY.get(key)
            if registered is not entry or entry.runtime is not retired_runtime:
                return
            if entry.references <= 0:
                _RUNTIME_REGISTRY.pop(key, None)
                return
            try:
                _start_replacement_runtime_locked(
                    key,
                    entry,
                    fallback_config=retired_runtime.config,
                )
            except BaseException:
                logger.exception("LCM background runtime replacement retry failed")
                continue
            return


def _ensure_retirement_watcher_locked(
    key: str,
    entry: _RuntimeEntry,
    runtime: _BackgroundRuntime,
) -> None:
    watcher = entry.retirement_watcher
    if watcher is not None and watcher.is_alive():
        return
    watcher = threading.Thread(
        target=_retire_runtime_and_maybe_replace,
        args=(key, entry, runtime),
        name=f"lcm-background-retire-{hashlib.sha256(key.encode()).hexdigest()[:8]}",
        daemon=True,
    )
    entry.retirement_watcher = watcher
    watcher.start()


def _release_runtime_reference(key: str, entry: _RuntimeEntry) -> None:
    """Release one handle without holding the registry during a bounded join."""
    runtime: _BackgroundRuntime | None = None
    wait_for_stop = False
    with _RUNTIME_LOCK:
        if _RUNTIME_REGISTRY.get(key) is not entry:
            return
        entry.references = max(0, entry.references - 1)
        if entry.references > 0:
            return
        runtime = entry.runtime
        if not runtime._stopping:
            runtime.request_stop()
            wait_for_stop = True
        else:
            _ensure_retirement_watcher_locked(key, entry, runtime)
    if not wait_for_stop or runtime is None:
        return
    stopped = runtime.wait_stopped(timeout=_RUNTIME_STOP_TIMEOUT_SECONDS)
    with _RUNTIME_LOCK:
        if _RUNTIME_REGISTRY.get(key) is not entry or entry.runtime is not runtime:
            return
        if stopped:
            if entry.references <= 0:
                _RUNTIME_REGISTRY.pop(key, None)
            else:
                _start_replacement_runtime_locked(
                    key,
                    entry,
                    fallback_config=runtime.config,
                )
            return
        _ensure_retirement_watcher_locked(key, entry, runtime)


def _canonical_runtime_key(db_path: str | Path) -> str:
    raw = str(db_path)
    if raw == ":memory:":
        return raw
    return str(Path(raw).expanduser().resolve())


def _background_runtime_available(db_path: str | Path) -> bool:
    key = _canonical_runtime_key(db_path)
    with _RUNTIME_LOCK:
        entry = _RUNTIME_REGISTRY.get(key)
        return bool(entry and entry.runtime.alive)


def _notify_background_runtime(db_path: str | Path) -> None:
    key = _canonical_runtime_key(db_path)
    with _RUNTIME_LOCK:
        entry = _RUNTIME_REGISTRY.get(key)
        runtime = entry.runtime if entry is not None else None
    if runtime is not None:
        runtime.notify()


class BackgroundRuntimeHandle:
    def __init__(
        self,
        key: str | None,
        runtime: _BackgroundRuntime | None,
        *,
        entry: _RuntimeEntry | None = None,
    ) -> None:
        self._key = key
        self._runtime = runtime
        self._entry = entry
        self._released = False
        self._lock = threading.Lock()

    def notify(self) -> None:
        with self._lock:
            if self._released:
                return
            key = self._key
            entry = self._entry
            runtime = self._runtime
        if key is not None and entry is not None:
            with _RUNTIME_LOCK:
                if _RUNTIME_REGISTRY.get(key) is entry:
                    runtime = entry.runtime
        if runtime is not None and runtime.alive:
            runtime.notify()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            key = self._key
            runtime = self._runtime
            entry = self._entry
            self._runtime = None
            self._entry = None
        if key is None:
            return
        if entry is not None:
            _release_runtime_reference(key, entry)
            return
        if runtime is None:
            return
        with _RUNTIME_LOCK:
            entry = _RUNTIME_REGISTRY.get(key)
            if entry is not None and entry.runtime is runtime:
                entry.references -= 1
                if entry.references <= 0:
                    stopped = runtime.stop()
                    if stopped:
                        _RUNTIME_REGISTRY.pop(key, None)


def acquire_background_runtime(
    db_path: str | Path,
    config: LCMConfig | None = None,
) -> BackgroundRuntimeHandle:
    """Acquire the process-global worker for one canonical database path."""
    resolved_config = config or LCMConfig(database_path=str(db_path))
    enabled = bool(getattr(resolved_config, "background_compaction_enabled", False))
    workers = max(
        0, int(getattr(resolved_config, "background_compaction_workers", 1) or 0)
    )
    key = _canonical_runtime_key(db_path)
    if not enabled or workers <= 0 or key == ":memory:":
        return BackgroundRuntimeHandle(None, None)
    with _RUNTIME_LOCK:
        entry = _RUNTIME_REGISTRY.get(key)
        if entry is not None and not entry.runtime._thread.is_alive():
            if entry.references <= 0:
                _RUNTIME_REGISTRY.pop(key, None)
                entry = None
            else:
                _start_replacement_runtime_locked(
                    key,
                    entry,
                    fallback_config=resolved_config,
                )
        if entry is None:
            runtime = _BackgroundRuntime(key, db_path, resolved_config)
            entry = _RuntimeEntry(runtime=runtime, references=0)
            _RUNTIME_REGISTRY[key] = entry
        elif entry.runtime._stopping:
            entry.replacement_config = resolved_config
            _ensure_retirement_watcher_locked(key, entry, entry.runtime)
        entry.references += 1
        return BackgroundRuntimeHandle(key, entry.runtime, entry=entry)


__all__ = [
    "BACKGROUND_COMPACTION_MIGRATION_STEP",
    "BackgroundCompactionStore",
    "BackgroundRuntimeHandle",
    "CompactionJob",
    "PreparedSummaryNode",
    "PromotionResult",
    "PublicationResult",
    "SourceChunk",
    "acquire_background_runtime",
    "background_policy_fingerprint",
]
