"""Recoverable LCM publication boundary for an authoritative host switch.

Reserve before the host commit; adopt only after its durable receipt is read.
The caller holds the host session lock throughout final publication/recovery.
No history or DAG nodes are deleted, including when preparation is cancelled.
"""

import hashlib
import json
import time


def _roots(conn, session):
    return conn.execute("""
        SELECT n.* FROM summary_nodes AS n WHERE n.session_id=?
        AND NOT EXISTS (
            SELECT 1 FROM summary_nodes AS p, json_each(p.source_ids) AS child
            WHERE p.session_id=n.session_id AND p.source_type='nodes'
            AND p.depth>n.depth AND CAST(child.value AS INTEGER)=n.node_id
        ) ORDER BY n.node_id
    """, (session,)).fetchall()


def _fingerprint(rows):
    return hashlib.sha256(json.dumps([dict(row) for row in rows], sort_keys=True,
                                    default=str).encode()).hexdigest()


def _source_rows(store, conn, *, session_id, source_ids):
    """Load the exact host-projected lineage, including prior conversations.

    One durable Hermes session can span more than one LCM conversation after a
    lifecycle rollover.  A model switch summarizes the host session history,
    so its source proof must follow the exact store IDs rather than filtering
    them to only the currently bound conversation.
    """
    wanted = [int(value) for value in source_ids]
    if not wanted:
        return []
    rows = conn.execute("""
        SELECT m.store_id, m.session_id, m.conversation_id, m.role, m.content,
               m.tool_call_id, m.tool_calls, m.tool_name, m.timestamp,
               m.token_estimate
        FROM json_each(?) AS wanted
        JOIN messages AS m ON m.store_id=CAST(wanted.value AS INTEGER)
        WHERE m.session_id=?
        ORDER BY m.store_id
    """, (json.dumps(wanted), str(session_id))).fetchall()
    return [store._message_dict(row) for row in rows]


class ModelSwitchPublication:
    def __init__(self, store):
        self.store = store
        with store._write_transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lcm_host_model_switches (
                    job_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, host_revision TEXT NOT NULL,
                    generation INTEGER NOT NULL, frontier INTEGER NOT NULL,
                    watermark INTEGER NOT NULL, source_ids TEXT NOT NULL,
                    source_hash TEXT NOT NULL, roots_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('reserved','adopted','cancelled')),
                    deadline REAL NOT NULL, node_id INTEGER, updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS lcm_host_model_switch_one_reserved
                ON lcm_host_model_switches(conversation_id) WHERE state='reserved'
            """)

    def reserve(self, *, job_id, conversation_id, session_id, host_revision,
                expected_frontier, watermark, source_ids, deadline):
        """Freeze publication of old work while leaving its source intact."""
        store = self.store
        with store._write_transaction() as conn:
            now = time.time()
            if not job_id or not host_revision or not now < deadline:
                raise ValueError('invalid or expired host switch reservation')
            existing = conn.execute('SELECT * FROM lcm_host_model_switches WHERE job_id=?', (job_id,)).fetchone()
            if existing is not None:
                if (existing['state'] == 'reserved' and existing['session_id'] == session_id
                        and existing['conversation_id'] == conversation_id
                        and existing['host_revision'] == host_revision
                        and existing['frontier'] == expected_frontier
                        and existing['watermark'] == watermark
                        and json.loads(existing['source_ids']) == list(source_ids)):
                    return dict(existing)
                raise ValueError('host switch reservation identity changed')
            fence = store._session_fence_under_tx(conn, conversation_id=conversation_id,
                                                   session_id=session_id, current=now, bootstrap=True)
            lifecycle = conn.execute('SELECT * FROM lcm_lifecycle_state WHERE conversation_id=?',
                                     (conversation_id,)).fetchone()
            if (fence is None or fence['session_id'] != session_id or not fence['publishable']
                    or lifecycle is None or lifecycle['current_session_id'] != session_id
                    or lifecycle['current_frontier_store_id'] != expected_frontier
                    or watermark < expected_frontier):
                raise ValueError('LCM session frontier changed or is already reserved')
            source_ids = [int(value) for value in source_ids]
            if (source_ids != sorted(set(source_ids))
                    or any(value <= expected_frontier or value > watermark for value in source_ids)):
                raise ValueError('LCM switch source range is not ordered within the reserved frontier')
            raw = _source_rows(store, conn, session_id=session_id, source_ids=source_ids)
            if [int(row['store_id']) for row in raw] != source_ids:
                raise ValueError('LCM switch source range changed')
            current_raw = store._load_source_range(
                conn, session_id=session_id, conversation_id=conversation_id,
                frontier=expected_frontier, watermark=watermark,
            )
            represented_current = [
                int(row['store_id']) for row in raw
                if str(row.get('conversation_id') or '') in ('', str(conversation_id))
            ]
            if [int(row['store_id']) for row in current_raw] != represented_current:
                raise ValueError('LCM switch source range changed')
            roots = _roots(conn, session_id)
            if not raw and not roots:
                raise ValueError('LCM has no source lineage for the host summary')
            generation = int(fence['generation']) + 1
            conn.execute("""UPDATE lcm_compaction_session_fences
                            SET generation=?,publishable=0,updated_at=? WHERE conversation_id=?""",
                         (generation, now, conversation_id))
            store._supersede_active_jobs_under_tx(conn, conversation_id=conversation_id,
                                                 session_id=session_id, reason='host_model_switch', current=now)
            conn.execute("""INSERT INTO lcm_host_model_switches
                (job_id,conversation_id,session_id,host_revision,generation,frontier,watermark,
                 source_ids,source_hash,roots_hash,state,deadline,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,'reserved',?,?)""",
                         (job_id, conversation_id, session_id, host_revision, generation,
                          expected_frontier, watermark, json.dumps(list(source_ids)),
                          store._source_identity_hash(raw), _fingerprint(roots), deadline, now))
            return dict(conn.execute('SELECT * FROM lcm_host_model_switches WHERE job_id=?', (job_id,)).fetchone())

    def finish(self, *, job_id, host_receipt, summary='', summary_tokens=0):
        """Adopt or release using a freshly read durable host job receipt.

        A missing/ambiguous host receipt NEVER releases the publication fence.
        After restart, a committed host job still gets adopted even if its
        original preparation deadline has passed; committing it again is not
        attempted. Failure/cancellation may release only a noncommitted job.
        """
        state = host_receipt.get('state')
        if state not in ('committed', 'cancelled', 'failed', 'expired'):
            raise ValueError('host switch outcome is not terminal; recovery must wait')
        store = self.store
        with store._write_transaction() as conn:
            row = conn.execute('SELECT * FROM lcm_host_model_switches WHERE job_id=?', (job_id,)).fetchone()
            if (row is None or host_receipt.get('id') != job_id
                    or host_receipt.get('session_id') != row['session_id']
                    or host_receipt.get('revision') != row['host_revision']):
                raise ValueError('host switch receipt does not match reservation')
            expected_state = 'adopted' if state == 'committed' else 'cancelled'
            if row['state'] == expected_state:
                return row['node_id']
            if row['state'] != 'reserved':
                raise ValueError('host and LCM switch outcomes disagree')
            conversation, session = row['conversation_id'], row['session_id']
            fence = conn.execute('SELECT * FROM lcm_compaction_session_fences WHERE conversation_id=?',
                                 (conversation,)).fetchone()
            if (fence is None or fence['session_id'] != session or fence['generation'] != row['generation']
                    or fence['publishable']):
                raise ValueError('LCM switch ownership changed')
            now = time.time()
            node_id = None
            if state == 'committed':
                if not summary.strip() or summary_tokens <= 0:
                    raise ValueError('committed switch summary is missing')
                roots = _roots(conn, session)
                raw = _source_rows(
                    store, conn, session_id=session,
                    source_ids=json.loads(row['source_ids']),
                )
                if (_fingerprint(roots) != row['roots_hash']
                        or store._source_identity_hash(raw) != row['source_hash']):
                    raise ValueError('LCM switch lineage changed; keep inference gated')
                children = [int(root['node_id']) for root in roots]
                source_tokens = sum(int(root['source_token_count'] or 0) for root in roots)
                if raw:
                    raw_tokens = sum(store._message_tokens(message) for message in raw)
                    source_tokens += raw_tokens
                    bridge = store._insert_canonical_node(
                        conn, session_id=session,
                        summary='[Source lineage for confirmed host context switch; content represented by parent.]',
                        token_count=1, source_token_count=raw_tokens,
                        source_ids=json.loads(row['source_ids']), created_at=now,
                        expand_hint='lcm:hidden-structural-provenance-bridge')
                    children.append(bridge)
                node_id = store._insert_canonical_parent_node(
                    conn, session_id=session, summary=summary, token_count=summary_tokens,
                    source_token_count=source_tokens, child_node_ids=children, created_at=now)
                if not store._advance_frontier_cas(
                    conn, conversation_id=conversation, session_id=session,
                    expected_frontier_store_id=row['frontier'], captured_watermark_store_id=row['watermark'],
                    debt_tokens=sum(store._message_tokens(message) for message in raw), current=now):
                    raise ValueError('LCM switch frontier changed; keep inference gated')
            conn.execute("""UPDATE lcm_host_model_switches SET state=?,node_id=?,updated_at=? WHERE job_id=?""",
                         (expected_state, node_id, now, job_id))
            conn.execute("""UPDATE lcm_compaction_session_fences SET publishable=1,updated_at=?
                            WHERE conversation_id=? AND generation=?""", (now, conversation, row['generation']))
            return node_id
