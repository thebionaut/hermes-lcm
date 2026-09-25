#!/usr/bin/env python3
"""Report (and optionally remove) duplicate raw message rows in an LCM store.

Older builds could re-insert host messages that were already stored after a
process restart or an active prefix rewrite. Replay/assembly now skips those
copies (see ``MessageStore.duplicate_store_id_aliases``), so removing them is
optional housekeeping, not a correctness requirement.

A row is a duplicate when an earlier row of the same session has the same
replay identity (``hermes_lcm.store.message_dedupe_identity``: host timestamp
+ role + tool_call_id [+ content/tool_calls digests for non-tool rows]). Rows
without a host timestamp are never touched.

Default is a dry run. ``--apply`` requires Hermes to be stopped and a backup:
  * lcm_compaction_node_sources / lcm_compaction_source_coverage /
    summary_nodes.source_ids are re-pointed to the kept (earliest) row;
  * duplicates referenced from tables whose references are bound to content
    or identity hashes (assertions, query views, chunks, compaction jobs,
    prepared nodes, model-switch records) are kept and reported;
  * everything else is deleted (FTS/corpus/assertion triggers fire normally).

Usage:
  dedupe_session_messages.py --db /path/lcm.db [--session ID ...] [--apply]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import types
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if "hermes_lcm" not in sys.modules:
    _pkg = types.ModuleType("hermes_lcm")
    _pkg.__path__ = [str(_REPO)]
    sys.modules["hermes_lcm"] = _pkg
from hermes_lcm.store import message_dedupe_identity  # noqa: E402

# (table, column) holding a single store id that --apply keeps as-is: these
# rows are bound to the source row's content/metadata, so a duplicate they
# reference is left in place.
_PINNING_REFERENCES = (
    ("lcm_assertion_sources", "source_store_id"),
    ("lcm_query_view_sources", "source_store_id"),
    ("lcm_chunk_meta", "store_id"),
)
# Tables that carry store-id lists inside JSON/identity-hashed columns.
_PINNING_JSON_REFERENCES = (
    ("lcm_compaction_jobs", "source_store_ids"),
    ("lcm_compaction_jobs", "repair_group_store_ids"),
    ("lcm_compaction_jobs", "repair_covered_store_ids"),
    ("lcm_compaction_jobs", "repair_uncovered_store_ids"),
    ("lcm_compaction_prepared_nodes", "source_store_ids"),
    ("lcm_host_model_switches", "source_ids"),
)

_COLUMNS = (
    "store_id, session_id, role, content, tool_call_id, tool_calls, observed_at"
)


def _table_columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _json_ids(value):
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    out = []

    def walk(item):
        if isinstance(item, bool):
            return
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and item.isdigit():
            out.append(int(item))
        elif isinstance(item, list):
            for sub in item:
                walk(sub)
        elif isinstance(item, dict):
            for sub in item.values():
                walk(sub)

    walk(parsed)
    return out


def find_duplicates(conn, sessions=None):
    """Return {dup_store_id: (kept_store_id, session_id, role)}."""
    if sessions:
        marks = ",".join("?" for _ in sessions)
        cursor = conn.execute(
            f"SELECT {_COLUMNS} FROM messages WHERE session_id IN ({marks}) "
            "AND observed_at IS NOT NULL ORDER BY session_id, store_id",
            list(sessions),
        )
    else:
        cursor = conn.execute(
            f"SELECT {_COLUMNS} FROM messages WHERE observed_at IS NOT NULL "
            "ORDER BY session_id, store_id"
        )
    dups = {}
    current_session = None
    seen = {}
    for store_id, session_id, role, content, tool_call_id, tool_calls, observed_at in cursor:
        if session_id != current_session:
            current_session = session_id
            seen = {}
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
        kept = seen.get(identity)
        if kept is None:
            seen[identity] = store_id
        else:
            dups[store_id] = (kept, session_id, role)
    return dups


def pinned_duplicates(conn, dups):
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    pinned = Counter()
    pinned_ids = set()
    dup_ids = set(dups)
    for table, column in _PINNING_REFERENCES:
        if table not in tables or column not in _table_columns(conn, table):
            continue
        for (store_id,) in conn.execute(f"SELECT DISTINCT {column} FROM {table}"):
            if store_id in dup_ids:
                pinned[f"{table}.{column}"] += 1
                pinned_ids.add(store_id)
    for table, column in _PINNING_JSON_REFERENCES:
        if table not in tables or column not in _table_columns(conn, table):
            continue
        for (value,) in conn.execute(f"SELECT {column} FROM {table}"):
            for store_id in _json_ids(value):
                if store_id in dup_ids and store_id not in pinned_ids:
                    pinned[f"{table}.{column}"] += 1
                    pinned_ids.add(store_id)
    return pinned_ids, pinned


def repointable_counts(conn, dups):
    dup_ids = set(dups)
    counts = Counter()
    for (store_id,) in conn.execute("SELECT store_id FROM lcm_compaction_node_sources"):
        if store_id in dup_ids:
            counts["lcm_compaction_node_sources"] += 1
    for (store_id,) in conn.execute("SELECT store_id FROM lcm_compaction_source_coverage"):
        if store_id in dup_ids:
            counts["lcm_compaction_source_coverage"] += 1
    for (source_ids,) in conn.execute(
        "SELECT source_ids FROM summary_nodes WHERE source_type = 'messages'"
    ):
        hits = sum(1 for store_id in _json_ids(source_ids) if store_id in dup_ids)
        if hits:
            counts["summary_nodes (nodes)"] += 1
            counts["summary_nodes (source ids)"] += hits
    return counts


def apply(conn, dups, pinned_ids):
    deletable = {dup: meta for dup, meta in dups.items() if dup not in pinned_ids}
    with conn:
        for dup, (kept, session_id, _role) in deletable.items():
            conn.execute(
                "INSERT OR IGNORE INTO lcm_compaction_node_sources(node_id, session_id, store_id) "
                "SELECT node_id, session_id, ? FROM lcm_compaction_node_sources WHERE store_id = ?",
                (kept, dup),
            )
            conn.execute("DELETE FROM lcm_compaction_node_sources WHERE store_id = ?", (dup,))
            conn.execute(
                "INSERT OR IGNORE INTO lcm_compaction_source_coverage("
                "session_id, store_id, publication_id, publication_kind, published_at) "
                "SELECT session_id, ?, publication_id, publication_kind, published_at "
                "FROM lcm_compaction_source_coverage WHERE session_id = ? AND store_id = ?",
                (kept, session_id, dup),
            )
            conn.execute(
                "DELETE FROM lcm_compaction_source_coverage WHERE session_id = ? AND store_id = ?",
                (session_id, dup),
            )
        by_session = {}
        for dup, (kept, session_id, _role) in deletable.items():
            by_session.setdefault(session_id, {})[dup] = kept
        for session_id, aliases in by_session.items():
            rows = conn.execute(
                "SELECT node_id, source_ids FROM summary_nodes "
                "WHERE session_id = ? AND source_type = 'messages'",
                (session_id,),
            ).fetchall()
            for node_id, source_ids in rows:
                try:
                    ids = json.loads(source_ids or "[]")
                except (TypeError, ValueError):
                    continue
                if not isinstance(ids, list) or not any(i in aliases for i in ids):
                    continue
                rewritten = []
                for item in ids:
                    item = aliases.get(item, item)
                    if item not in rewritten:
                        rewritten.append(item)
                conn.execute(
                    "UPDATE summary_nodes SET source_ids = ? WHERE node_id = ?",
                    (json.dumps(rewritten), node_id),
                )
        ids = sorted(deletable)
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" for _ in chunk)
            conn.execute(f"DELETE FROM messages WHERE store_id IN ({marks})", chunk)
    return len(deletable)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--session", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    uri = f"file:{args.db}" + ("" if args.apply else "?mode=ro")
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        total = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        dups = find_duplicates(conn, args.session or None)
        pinned_ids, pinned = pinned_duplicates(conn, dups)
        per_session = Counter(meta[1] for meta in dups.values())
        report = {
            "mode": "apply" if args.apply else "dry-run",
            "scope": args.session or "all sessions",
            "messages_total": total,
            "duplicate_rows": len(dups),
            "sessions_with_duplicates": len(per_session),
            "duplicates_by_role": dict(Counter(meta[2] for meta in dups.values())),
            "top_sessions": per_session.most_common(10),
            "repointable_references": dict(repointable_counts(conn, dups)),
            "kept_because_pinned": len(pinned_ids),
            "pinned_by": dict(pinned),
            "would_delete": len(dups) - len(pinned_ids),
        }
        if args.apply:
            report["deleted"] = apply(conn, dups, pinned_ids)
        print(json.dumps(report, indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
