"""Read-only, bounded LCM compaction relationship auditor."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from typing import Any, Mapping, Sequence


AUDIT_VERSION = "lcm-compaction-invariants-v1"
CLASSIFICATIONS = (
    "valid_awaiting_pressure",
    "provably_obsolete",
    "inconsistent_stranded",
    "unknown",
)
SUPPORTED_JOB_KINDS = {
    "standard",
    "lossless_raw_projection",
    "partial_atomic_repair",
}


def _ids(value: Any) -> tuple[int, ...] | None:
    try:
        parsed = json.loads(str(value or "[]"))
        if not isinstance(parsed, list):
            return None
        result = tuple(int(item) for item in parsed)
    except (TypeError, ValueError, json.JSONDecodeError, OverflowError):
        return None
    return result


def _bounded(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return list(items[: max(0, int(limit))])


def _source_identity(messages: Sequence[Mapping[str, Any]]) -> str:
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


def _tool_pair_status(messages: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    pending: list[str] = []
    seen: set[str] = set()
    for message in messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, str):
                try:
                    raw_calls = json.loads(raw_calls)
                except json.JSONDecodeError:
                    return "inconsistent", "tool_calls_invalid_json"
            if raw_calls:
                if not isinstance(raw_calls, list):
                    return "inconsistent", "tool_calls_not_list"
                for call in raw_calls:
                    if not isinstance(call, Mapping):
                        return "inconsistent", "tool_call_not_object"
                    call_id = str(call.get("id") or call.get("tool_call_id") or "")
                    if not call_id or call_id in seen:
                        return "inconsistent", "tool_call_id_missing_or_duplicate"
                    seen.add(call_id)
                    pending.append(call_id)
        elif role == "tool":
            result_id = str(message.get("tool_call_id") or "")
            if not pending or result_id != pending[0]:
                return "inconsistent", "tool_result_id_or_order_mismatch"
            pending.pop(0)
    if pending:
        return "inconsistent", "tool_call_without_result"
    return "valid", ""


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def audit_compaction_invariants(
    conn: sqlite3.Connection,
    *,
    current_policy_fingerprint: str,
    scope: str = "session",
    session_id: str = "",
    job_id: str = "",
    fence_generation: int | None = None,
    cursor: str = "",
    limit: int = 100,
    example_limit: int = 10,
    snapshot_kind: str = "bounded_live_snapshot",
) -> dict[str, Any]:
    """Audit a declared snapshot without mutating it.

    Pagination is by stable ``job_id``. A limited whole-store page is explicitly
    incomplete and therefore cannot report a passing status.
    """
    scope = str(scope or "session")
    if scope not in {"session", "job", "epoch", "whole_store"}:
        return {
            "audit_version": AUDIT_VERSION,
            "status": "unknown",
            "complete": False,
            "error": "invalid_scope",
        }
    limit = min(500, max(1, int(limit or 100)))
    example_limit = min(50, max(1, int(example_limit or 10)))
    required = {
        "lcm_compaction_jobs",
        "lcm_compaction_prepared_nodes",
        "lcm_compaction_publications",
        "lcm_compaction_source_coverage",
        "lcm_compaction_node_sources",
        "lcm_compaction_node_edges",
        "lcm_compaction_session_fences",
        "lcm_compaction_provider_leases",
        "lcm_compaction_attempts",
        "lcm_compaction_attempt_receipts",
        "lcm_lifecycle_state",
        "summary_nodes",
        "messages",
    }
    missing_tables = sorted(required - _table_names(conn))
    if missing_tables:
        return {
            "audit_version": AUDIT_VERSION,
            "status": "unknown",
            "complete": False,
            "scope": scope,
            "snapshot": {"kind": snapshot_kind, "consistent": True},
            "error": "missing_tables",
            "missing_tables": missing_tables,
        }

    where = ["job.job_id>?"]
    params: list[Any] = [str(cursor or "")]
    if scope == "session":
        if not session_id:
            return {
                "audit_version": AUDIT_VERSION,
                "status": "unknown",
                "complete": False,
                "scope": scope,
                "error": "session_id_required",
            }
        where.append("job.session_id=?")
        params.append(str(session_id))
    elif scope == "job":
        if not job_id:
            return {
                "audit_version": AUDIT_VERSION,
                "status": "unknown",
                "complete": False,
                "scope": scope,
                "error": "job_id_required",
            }
        where.append("job.job_id=?")
        params.append(str(job_id))
    elif scope == "epoch":
        if not session_id or fence_generation is None:
            return {
                "audit_version": AUDIT_VERSION,
                "status": "unknown",
                "complete": False,
                "scope": scope,
                "error": "session_id_and_fence_generation_required",
            }
        where.extend(["job.session_id=?", "job.session_fence_generation=?"])
        params.extend([str(session_id), int(fence_generation)])

    base_where = " AND ".join(where)
    rows = conn.execute(
        f"""
        SELECT job.*
        FROM lcm_compaction_jobs AS job
        WHERE {base_where}
        ORDER BY job.job_id
        LIMIT ?
        """,
        (*params, limit + 1),
    ).fetchall()
    has_more = len(rows) > limit
    rows = rows[:limit]
    classifications: list[dict[str, Any]] = []
    relation_issues: list[dict[str, Any]] = []

    scoped_job_ids = [str(row["job_id"]) for row in rows]
    for row in rows:
        job = dict(row)
        jid = str(job["job_id"])
        if str(job.get("state") or "") != "prepared":
            continue
        reasons: list[str] = []
        category = "valid_awaiting_pressure"
        source_ids = _ids(job.get("source_store_ids"))
        prepared = conn.execute(
            """
            SELECT * FROM lcm_compaction_prepared_nodes
            WHERE job_id=? ORDER BY ordinal
            """,
            (jid,),
        ).fetchall()

        if str(job.get("job_kind") or "standard") not in SUPPORTED_JOB_KINDS:
            category = "unknown"
            reasons.append("unsupported_job_kind")
        elif source_ids is None or not source_ids or len(set(source_ids)) != len(source_ids):
            category = "inconsistent_stranded"
            reasons.append("source_ids_invalid_or_duplicate")
        elif tuple(sorted(source_ids)) != source_ids:
            category = "inconsistent_stranded"
            reasons.append("source_ids_not_ordered")
        elif int(job.get("captured_watermark_store_id") or 0) != source_ids[-1]:
            category = "inconsistent_stranded"
            reasons.append("watermark_source_mismatch")
        elif (
            len(prepared) != int(job.get("expected_node_count") or 0)
            or len(prepared) != int(job.get("prepared_node_count") or 0)
        ):
            category = "inconsistent_stranded"
            reasons.append("prepared_node_count_mismatch")
        else:
            artifact_ids: list[int] = []
            artifact_invalid = False
            for artifact in prepared:
                decoded = _ids(artifact["source_store_ids"])
                if decoded is None:
                    artifact_invalid = True
                    break
                artifact_ids.extend(decoded)
            if artifact_invalid or tuple(artifact_ids) != source_ids:
                category = "inconsistent_stranded"
                reasons.append("prepared_lineage_mismatch")

        fence = conn.execute(
            """
            SELECT * FROM lcm_compaction_session_fences
            WHERE conversation_id=?
            """,
            (str(job.get("conversation_id") or ""),),
        ).fetchone()
        lifecycle = conn.execute(
            """
            SELECT * FROM lcm_lifecycle_state WHERE conversation_id=?
            """,
            (str(job.get("conversation_id") or ""),),
        ).fetchone()
        if category == "valid_awaiting_pressure":
            if fence is None or lifecycle is None:
                category = "unknown"
                reasons.append("fence_or_lifecycle_evidence_missing")
            elif (
                str(fence["session_id"] or "") != str(job.get("session_id") or "")
                or int(fence["generation"] or 0)
                != int(job.get("session_fence_generation") or 0)
                or int(fence["publishable"] or 0) != 1
            ):
                category = "provably_obsolete"
                reasons.append("session_fence_changed")
            elif (
                str(lifecycle["current_session_id"] or "")
                != str(job.get("session_id") or "")
            ):
                category = "provably_obsolete"
                reasons.append("lifecycle_session_changed")
            elif int(lifecycle["current_frontier_store_id"] or 0) != int(
                job.get("expected_frontier_store_id") or 0
            ):
                category = "provably_obsolete"
                reasons.append("frontier_changed")
            elif str(job.get("policy_fingerprint") or "") != str(
                current_policy_fingerprint or ""
            ):
                category = "provably_obsolete"
                reasons.append("policy_changed")

        messages: list[dict[str, Any]] = []
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            message_rows = conn.execute(
                f"""
                SELECT store_id, session_id, conversation_id, role, content,
                       tool_call_id, tool_calls, tool_name, timestamp
                FROM messages
                WHERE session_id=? AND store_id IN ({placeholders})
                ORDER BY store_id
                """,
                (str(job.get("session_id") or ""), *source_ids),
            ).fetchall()
            messages = []
            for item in message_rows:
                message = dict(item)
                raw_tool_calls = message.get("tool_calls")
                if raw_tool_calls:
                    try:
                        message["tool_calls"] = json.loads(str(raw_tool_calls))
                    except json.JSONDecodeError:
                        message["tool_calls"] = None
                messages.append(message)
            actual_ids = tuple(int(item["store_id"]) for item in messages)
            if actual_ids != source_ids and category == "valid_awaiting_pressure":
                category = "inconsistent_stranded"
                reasons.append("captured_source_missing_or_changed")
            elif actual_ids == source_ids:
                if _source_identity(messages) != str(job.get("source_identity_hash") or ""):
                    if category == "valid_awaiting_pressure":
                        category = "inconsistent_stranded"
                    reasons.append("source_identity_mismatch")
                tool_status, tool_reason = _tool_pair_status(messages)
                if tool_status != "valid":
                    if category == "valid_awaiting_pressure":
                        category = "inconsistent_stranded"
                    reasons.append(tool_reason)

        if str(job.get("job_kind") or "") == "partial_atomic_repair":
            group_ids = _ids(job.get("repair_group_store_ids"))
            covered = _ids(job.get("repair_covered_store_ids"))
            uncovered = _ids(job.get("repair_uncovered_store_ids"))
            if (
                source_ids is None
                or group_ids != source_ids
                or covered is None
                or uncovered is None
                or set(covered) & set(uncovered)
                or set(covered) | set(uncovered) != set(source_ids)
                or not covered
                or not uncovered
            ):
                category = "inconsistent_stranded"
                reasons.append("partial_repair_partition_invalid")
            epoch = conn.execute(
                "SELECT mutation_epoch FROM lcm_compaction_dag_epochs WHERE session_id=?",
                (str(job.get("session_id") or ""),),
            ).fetchone()
            if epoch is None:
                if category == "valid_awaiting_pressure":
                    category = "unknown"
                reasons.append("dag_epoch_missing")
            elif int(epoch[0] or 0) != int(job.get("repair_bound_dag_epoch") or 0):
                if category == "valid_awaiting_pressure":
                    category = "provably_obsolete"
                reasons.append("partial_repair_dag_epoch_changed")

        classifications.append(
            {
                "job_id": jid,
                "session_id": str(job.get("session_id") or ""),
                "fence_generation": int(job.get("session_fence_generation") or 0),
                "job_kind": str(job.get("job_kind") or "standard"),
                "classification": category,
                "reasons": sorted(set(reasons)),
                "next_action": {
                    "valid_awaiting_pressure": "none_wait_for_publication_pressure",
                    "provably_obsolete": "operator_review_no_mutation_performed",
                    "inconsistent_stranded": "investigate_relationship_violation",
                    "unknown": "resume_or_supply_missing_snapshot_evidence",
                }[category],
            }
        )

    if scoped_job_ids:
        placeholders = ",".join("?" for _ in scoped_job_ids)
        checks = (
            (
                "publication_without_published_job",
                f"""SELECT publication.job_id FROM lcm_compaction_publications publication
                    JOIN lcm_compaction_jobs job ON job.job_id=publication.job_id
                    WHERE publication.job_id IN ({placeholders})
                      AND job.state<>'published' LIMIT ?""",
            ),
            (
                "publication_identity_or_epoch_mismatch",
                f"""SELECT publication.job_id FROM lcm_compaction_publications publication
                    JOIN lcm_compaction_jobs job ON job.job_id=publication.job_id
                    WHERE publication.job_id IN ({placeholders})
                      AND (publication.range_identity<>job.source_identity_hash
                           OR publication.publication_fence_generation<>
                              job.session_fence_generation) LIMIT ?""",
            ),
            (
                "provider_permit_without_active_claim",
                f"""SELECT permit.job_id FROM lcm_compaction_provider_leases permit
                    JOIN lcm_compaction_jobs job ON job.job_id=permit.job_id
                    WHERE permit.job_id IN ({placeholders})
                      AND job.state NOT IN ('running','retry') LIMIT ?""",
            ),
            (
                "attempt_claim_exceeds_job_claim",
                f"""SELECT attempt.job_id FROM lcm_compaction_attempts attempt
                    JOIN lcm_compaction_jobs job ON job.job_id=attempt.job_id
                    WHERE attempt.job_id IN ({placeholders})
                      AND attempt.job_attempt>job.attempt_count LIMIT ?""",
            ),
            (
                "physical_receipt_claim_exceeds_job_claim",
                f"""SELECT receipt.job_id FROM lcm_compaction_attempt_receipts receipt
                    JOIN lcm_compaction_jobs job ON job.job_id=receipt.job_id
                    WHERE receipt.job_id IN ({placeholders})
                      AND receipt.job_attempt>job.attempt_count LIMIT ?""",
            ),
        )
        for check, sql in checks:
            found = conn.execute(sql, (*scoped_job_ids, example_limit)).fetchall()
            if found:
                relation_issues.append(
                    {"check": check, "count_lower_bound": len(found),
                     "example_job_ids": [str(item[0]) for item in found]}
                )

        scoped_sessions = sorted({str(row["session_id"] or "") for row in rows})
        if scoped_sessions:
            session_placeholders = ",".join("?" for _ in scoped_sessions)
            graph_checks = (
                (
                    "node_source_owner_missing",
                    f"""SELECT owner.node_id
                        FROM lcm_compaction_node_sources owner
                        LEFT JOIN summary_nodes node ON node.node_id=owner.node_id
                        WHERE owner.session_id IN ({session_placeholders})
                          AND (node.node_id IS NULL OR node.session_id<>owner.session_id
                               OR node.source_type<>'messages') LIMIT ?""",
                ),
                (
                    "dag_edge_endpoint_missing",
                    f"""SELECT edge.parent_node_id
                        FROM lcm_compaction_node_edges edge
                        LEFT JOIN summary_nodes parent
                          ON parent.node_id=edge.parent_node_id
                        LEFT JOIN summary_nodes child
                          ON child.node_id=edge.child_node_id
                        WHERE edge.session_id IN ({session_placeholders})
                          AND (parent.node_id IS NULL OR child.node_id IS NULL
                               OR parent.session_id<>edge.session_id
                               OR child.session_id<>edge.session_id) LIMIT ?""",
                ),
                (
                    "canonical_coverage_owner_missing",
                    f"""SELECT coverage.store_id
                        FROM lcm_compaction_source_coverage coverage
                        LEFT JOIN summary_nodes node
                          ON node.node_id=CAST(
                             substr(coverage.publication_id, 14) AS INTEGER)
                         AND coverage.publication_id LIKE 'summary-node:%'
                        LEFT JOIN lcm_compaction_node_sources owner
                          ON owner.node_id=node.node_id
                         AND owner.session_id=coverage.session_id
                         AND owner.store_id=coverage.store_id
                        WHERE coverage.session_id IN ({session_placeholders})
                          AND coverage.publication_kind='canonical'
                          AND coverage.publication_id LIKE 'summary-node:%'
                          AND (node.node_id IS NULL OR owner.node_id IS NULL)
                        LIMIT ?""",
                ),
            )
            for check, sql in graph_checks:
                found = conn.execute(
                    sql, (*scoped_sessions, example_limit)
                ).fetchall()
                if found:
                    relation_issues.append(
                        {
                            "check": check,
                            "count_lower_bound": len(found),
                            "example_ids": [int(item[0]) for item in found],
                        }
                    )

    counts = Counter(item["classification"] for item in classifications)
    for label in CLASSIFICATIONS:
        counts.setdefault(label, 0)
    complete = not has_more
    if not complete:
        counts["unknown"] += 1
    failing = counts["inconsistent_stranded"] > 0 or bool(relation_issues)
    unknown = counts["unknown"] > 0 or not complete
    status = "fail" if failing else ("unknown" if unknown else "pass")
    examples = {
        label: _bounded(
            [item for item in classifications if item["classification"] == label],
            example_limit,
        )
        for label in CLASSIFICATIONS
    }
    next_cursor = str(rows[-1]["job_id"]) if has_more and rows else None
    return {
        "audit_version": AUDIT_VERSION,
        "status": status,
        "complete": complete,
        "scope": {
            "kind": scope,
            "session_id": str(session_id or ""),
            "job_id": str(job_id or ""),
            "fence_generation": fence_generation,
            "cursor": str(cursor or ""),
            "limit": limit,
        },
        "snapshot": {"kind": str(snapshot_kind), "consistent": True},
        "coverage": {
            "job_rows_scanned": len(rows),
            "prepared_jobs_scanned": len(classifications),
            "classification_counts": dict(counts),
            "all_scanned_prepared_jobs_classified": (
                len(classifications) == sum(
                    1 for item in classifications if item.get("classification")
                )
            ),
            "has_more": has_more,
            "next_cursor": next_cursor,
        },
        "examples": examples,
        "relationship_issues": relation_issues,
        "read_only": True,
        "age_used_as_corruption_signal": False,
    }


__all__ = ["AUDIT_VERSION", "audit_compaction_invariants"]
