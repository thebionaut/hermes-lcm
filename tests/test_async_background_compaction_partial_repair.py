"""First-class provenance repair for partially covered atomic tool groups."""

from __future__ import annotations

import json
import threading

import pytest

import hermes_lcm.background_compaction as background_module
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    CompactionJob,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def _config(db_path, **overrides):
    values = dict(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1,
        background_compaction_prepare_ratio=0.0,
        background_compaction_max_passes=4,
        background_compaction_lease_seconds=300,
        fresh_tail_count=1,
        leaf_chunk_tokens=10_000,
    )
    values.update(overrides)
    return LCMConfig(**values)


def _partial_fixture(
    db_path,
    *,
    duplicate_owner: bool = False,
    root_depths: tuple[int, ...] = (),
    owner_spill: bool = False,
):
    messages = MessageStore(db_path)
    source_ids = messages.append_batch(
        "session",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-a",
                        "type": "function",
                        "function": {"name": "first", "arguments": "{}"},
                    },
                    {
                        "id": "call-b",
                        "type": "function",
                        "function": {"name": "second", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "first result"},
            {"role": "tool", "tool_call_id": "call-b", "content": "second result"},
            {"role": "user", "content": "protected tail"},
        ],
        token_estimates=[10, 10, 10, 10],
        conversation_id="conversation",
    )
    messages.close()
    group_ids = tuple(source_ids[:3])

    dag = SummaryDAG(db_path)
    owners = [
        dag.add_node(
            SummaryNode(
                session_id="session",
                summary="legacy owner one",
                source_ids=list(
                    (*group_ids[:2], source_ids[-1])
                    if owner_spill
                    else group_ids[:2]
                ),
                source_type="messages",
                token_count=4,
                source_token_count=20,
            )
        )
    ]
    if duplicate_owner:
        owners.append(
            dag.add_node(
                SummaryNode(
                    session_id="session",
                    summary="legacy duplicate owner",
                    source_ids=[group_ids[0]],
                    source_type="messages",
                    token_count=4,
                    source_token_count=10,
                )
            )
        )
    roots = list(owners)
    for depth in root_depths:
        roots = [
            dag.add_node(
                SummaryNode(
                    session_id="session",
                    depth=depth,
                    summary=f"root for owner {owner}",
                    source_ids=[owner],
                    source_type="nodes",
                    token_count=3,
                    source_token_count=20,
                )
            )
            for owner in roots
        ]
    dag.close()

    lifecycle = LifecycleStateStore(db_path)
    state = lifecycle.bind_session("session", conversation_id="conversation")
    lifecycle.close()
    config = _config(db_path)
    background = BackgroundCompactionStore(db_path, config=config)
    generation = background.activate_session(
        conversation_id="conversation",
        session_id="session",
        expected_lifecycle_bound_at=state.current_bound_at,
    )
    assert generation is not None
    return background, config, source_ids, group_ids, tuple(owners), tuple(roots)


def _capture_repair(background, config):
    return background.capture_and_enqueue(
        conversation_id="conversation",
        session_id="session",
        expected_frontier_store_id=0,
        active_tokens=300_000,
        effective_context_window=372_000,
        policy_fingerprint=background_policy_fingerprint(config),
    )


def _prepare_repair(background, config, *, seen_text=None):
    def summarizer(**kwargs):
        if seen_text is not None:
            seen_text.append(kwargs["text"])
        return "provider repair parent", 0

    prepared = background.run_next_job("worker", summarizer=summarizer)
    assert isinstance(prepared, CompactionJob)
    assert prepared.state == "prepared"
    lease = background.acquire_publication_lease(
        prepared.job_id,
        owner="publisher",
    )
    assert lease is not None
    return prepared, lease


def _bind_and_prepare_without_provider(background, config):
    """Exercise the durable repair protocol without external provider I/O."""
    claimed = background.claim_next_job(
        owner="worker",
        policy_fingerprint=background_policy_fingerprint(config),
    )
    assert claimed is not None
    bound = background._bind_partial_atomic_repair(
        claimed.job_id,
        owner="worker",
        generation=claimed.lease_generation,
    )
    assert bound is not None
    assert bound.repair_root_node_ids
    prepared = background.mark_prepared(
        claimed.job_id,
        owner="worker",
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "provider repair parent",
                "source_store_ids": list(bound.source_store_ids),
                "source_token_count": bound.source_token_count,
                "summary_token_count": 3,
                "creation_model": "test-model",
                "creation_provider": "test-provider",
            }
        ],
        expected_node_count=1,
    )
    assert isinstance(prepared, CompactionJob)
    lease = background.acquire_publication_lease(
        prepared.job_id,
        owner="publisher",
    )
    assert lease is not None
    return prepared, lease


def test_partial_group_capture_queues_high_priority_repair_without_foreground_provider(
    tmp_path,
):
    background, config, _source_ids, group_ids, owners, _roots = _partial_fixture(
        tmp_path / "capture.db"
    )
    try:
        job = _capture_repair(background, config)
        assert job is not None
        assert job.job_kind == "partial_atomic_repair"
        assert job.priority >= background_module._PARTIAL_REPAIR_PRIORITY
        assert job.source_store_ids == group_ids
        assert job.repair_group_store_ids == group_ids
        assert job.repair_covered_store_ids == group_ids[:2]
        assert job.repair_uncovered_store_ids == group_ids[2:]
        assert job.repair_owner_node_ids == owners
        assert background.has_active_partial_atomic_repair(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
        )
        event = background.connection.execute(
            """
            SELECT reason_code FROM lcm_compaction_events
            WHERE job_id=? ORDER BY event_id DESC LIMIT 1
            """,
            (job.job_id,),
        ).fetchone()
        assert event[0] == "partial_atomic_repair_pending"
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_prepared_nodes"
        ).fetchone()[0] == 0
    finally:
        background.close()


def test_worker_binds_all_duplicate_owner_roots_and_marks_provider_context(tmp_path):
    background, config, _source_ids, group_ids, owners, roots = _partial_fixture(
        tmp_path / "roots.db",
        duplicate_owner=True,
        root_depths=(1,),
    )
    seen_text: list[str] = []
    try:
        job = _capture_repair(background, config)
        assert job is not None
        prepared, _lease = _prepare_repair(background, config, seen_text=seen_text)
        assert prepared.repair_owner_node_ids == tuple(sorted(owners))
        assert prepared.repair_root_node_ids == tuple(sorted(roots))
        assert prepared.repair_ancestry_node_ids == tuple(sorted((*owners, *roots)))
        assert prepared.repair_topology_hash
        assert prepared.repair_root_identity_hash
        assert prepared.repair_bound_dag_epoch > 0
        assert len(seen_text) == 1
        assert "EXISTING CANONICAL CONTEXT" in seen_text[0]
        assert "UNCOVERED ATOMIC CONTINUATION" in seen_text[0]
        for root in roots:
            assert f"node_id={root}" in seen_text[0]
        for store_id in group_ids:
            assert f"store_id={store_id}" in seen_text[0]
    finally:
        background.close()


def test_repair_publication_adds_uncovered_bridge_and_parent_atomically(tmp_path):
    db_path = tmp_path / "publish.db"
    background, config, _source_ids, group_ids, owners, roots = _partial_fixture(
        db_path,
        duplicate_owner=True,
        root_depths=(1,),
    )
    try:
        job = _capture_repair(background, config)
        assert job is not None
        covered_before = [
            tuple(row)
            for row in background.connection.execute(
                """
                SELECT node_id, store_id FROM lcm_compaction_node_sources
                WHERE session_id='session' AND store_id IN (?,?)
                ORDER BY node_id, store_id
                """,
                group_ids[:2],
            ).fetchall()
        ]
        _prepared, lease = _bind_and_prepare_without_provider(background, config)
        result = background.publish_prepared(
            lease.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert result.promoted is True
        assert len(result.node_ids) == 2
        bridge_id, parent_id = result.node_ids
        bridge = background.connection.execute(
            "SELECT * FROM summary_nodes WHERE node_id=?", (bridge_id,)
        ).fetchone()
        parent = background.connection.execute(
            "SELECT * FROM summary_nodes WHERE node_id=?", (parent_id,)
        ).fetchone()
        assert bridge["source_type"] == "messages"
        assert tuple(json.loads(bridge["source_ids"])) == group_ids[2:]
        assert "structural provenance bridge" in bridge["summary"].lower()
        assert parent["source_type"] == "nodes"
        assert tuple(json.loads(parent["source_ids"])) == tuple((*sorted(roots), bridge_id))
        assert int(parent["depth"]) > max(
            int(
                background.connection.execute(
                    "SELECT depth FROM summary_nodes WHERE node_id=?", (root,)
                ).fetchone()[0]
            )
            for root in roots
        )
        covered_after = [
            tuple(row)
            for row in background.connection.execute(
                """
                SELECT node_id, store_id FROM lcm_compaction_node_sources
                WHERE session_id='session' AND store_id IN (?,?)
                ORDER BY node_id, store_id
                """,
                group_ids[:2],
            ).fetchall()
        ]
        assert covered_after == covered_before
        assert background.connection.execute(
            """
            SELECT node_id FROM lcm_compaction_node_sources
            WHERE session_id='session' AND store_id=?
            """,
            (group_ids[2],),
        ).fetchone()[0] == bridge_id
        dag = SummaryDAG(db_path)
        try:
            assert dag.source_message_ids(parent_id, limit=100) == list(group_ids)
            active_ids = {
                node.node_id
                for depth in range(int(parent["depth"]) + 1)
                for node in dag.get_uncondensed_at_depth("session", depth, limit=100)
            }
            assert parent_id in active_ids
            assert bridge_id not in active_ids
            assert not set(roots) & active_ids
        finally:
            dag.close()
        lifecycle = LifecycleStateStore(db_path)
        try:
            assert (
                lifecycle.get_by_conversation("conversation").current_frontier_store_id
                == group_ids[-1]
            )
        finally:
            lifecycle.close()
    finally:
        background.close()


@pytest.mark.parametrize(
    "failure_hook",
    ["after_repair_bridge_insert", "after_repair_parent_insert"],
)
def test_repair_publication_failure_rolls_back_bridge_parent_and_frontier(
    tmp_path,
    failure_hook,
):
    db_path = tmp_path / f"rollback-{failure_hook}.db"
    background, config, _source_ids, _group_ids, _owners, _roots = _partial_fixture(
        db_path,
        duplicate_owner=True,
        root_depths=(1,),
    )
    try:
        job = _capture_repair(background, config)
        assert job is not None
        _prepared, lease = _bind_and_prepare_without_provider(background, config)
        nodes_before = background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0]
        coverage_before = [
            tuple(row)
            for row in background.connection.execute(
                """
                SELECT session_id, store_id, publication_id
                FROM lcm_compaction_source_coverage
                ORDER BY session_id, store_id
                """
            ).fetchall()
        ]
        with pytest.raises(RuntimeError, match="injected repair publication failure"):
            background.publish_prepared(
                lease.job_id,
                owner="publisher",
                generation=lease.lease_generation,
                policy_fingerprint=background_policy_fingerprint(config),
                failure_hook=failure_hook,
            )
        assert background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == nodes_before
        assert [
            tuple(row)
            for row in background.connection.execute(
                """
                SELECT session_id, store_id, publication_id
                FROM lcm_compaction_source_coverage
                ORDER BY session_id, store_id
                """
            ).fetchall()
        ] == coverage_before
        assert background.connection.execute(
            """
            SELECT current_frontier_store_id FROM lcm_lifecycle_state
            WHERE conversation_id='conversation'
            """
        ).fetchone()[0] == 0
        assert background.get_job(job.job_id).state == "prepared"
    finally:
        background.close()


def test_repair_bind_fails_closed_on_raw_mutation(tmp_path):
    background, config, _source_ids, group_ids, _owners, _roots = _partial_fixture(
        tmp_path / "raw-race.db"
    )
    try:
        job = _capture_repair(background, config)
        assert job is not None
        claimed = background.claim_next_job(
            owner="worker",
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert claimed is not None
        background.connection.execute(
            "UPDATE messages SET content='mutated raw result' WHERE store_id=?",
            (group_ids[-1],),
        )
        assert background._bind_partial_atomic_repair(
            claimed.job_id,
            owner="worker",
            generation=claimed.lease_generation,
        ) is None
        stale = background.get_job(job.job_id)
        assert stale.state == "superseded"
        assert stale.completion_reason == "partial_repair_source_range_changed"
    finally:
        background.close()


def test_repair_publication_supersedes_on_bound_root_mutation(tmp_path):
    background, config, _source_ids, _group_ids, _owners, roots = _partial_fixture(
        tmp_path / "root-race.db",
        duplicate_owner=True,
        root_depths=(1,),
    )
    try:
        job = _capture_repair(background, config)
        assert job is not None
        _prepared, lease = _bind_and_prepare_without_provider(background, config)
        nodes_before = background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0]
        background.connection.execute(
            "UPDATE summary_nodes SET summary='changed root' WHERE node_id=?",
            (roots[0],),
        )
        result = background.publish_prepared(
            lease.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert result.promoted is False
        assert result.reason == "partial_repair_dag_epoch_changed"
        assert background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == nodes_before
        assert background.connection.execute(
            """
            SELECT current_frontier_store_id FROM lcm_lifecycle_state
            WHERE conversation_id='conversation'
            """
        ).fetchone()[0] == 0
    finally:
        background.close()


def test_repair_enqueue_repairs_missing_election_and_ignores_orphan_coverage(
    tmp_path,
):
    background, config, _source_ids, group_ids, owners, _roots = _partial_fixture(
        tmp_path / "election.db"
    )
    try:
        background.connection.execute(
            """
            DELETE FROM lcm_compaction_source_coverage
            WHERE session_id='session' AND store_id=?
            """,
            (group_ids[0],),
        )
        background.connection.execute(
            """
            INSERT OR REPLACE INTO lcm_compaction_source_coverage(
                session_id, store_id, publication_id,
                publication_kind, published_at
            ) VALUES('session', ?, 'summary-node:999999', 'canonical', 0)
            """,
            (group_ids[-1],),
        )
        job = _capture_repair(background, config)
        assert job is not None
        assert job.repair_covered_store_ids == group_ids[:2]
        assert job.repair_uncovered_store_ids == group_ids[2:]
        elected = background.connection.execute(
            """
            SELECT publication_id FROM lcm_compaction_source_coverage
            WHERE session_id='session' AND store_id=?
            """,
            (group_ids[0],),
        ).fetchone()[0]
        assert elected in {
            f"summary-node:{owners[0]}",
            f"legacy-node:{owners[0]}",
        }
        assert background.connection.execute(
            """
            SELECT 1 FROM lcm_compaction_source_coverage
            WHERE session_id='session' AND store_id=?
            """,
            (group_ids[-1],),
        ).fetchone() is None
    finally:
        background.close()


def test_two_connections_publish_one_repair_exactly_once(tmp_path):
    db_path = tmp_path / "publish-race.db"
    background, config, _source_ids, group_ids, _owners, _roots = _partial_fixture(
        db_path,
        duplicate_owner=True,
        root_depths=(1,),
    )
    peer = None
    try:
        assert _capture_repair(background, config) is not None
        _prepared, lease = _bind_and_prepare_without_provider(background, config)
        peer = BackgroundCompactionStore(db_path, config=config)
        barrier = threading.Barrier(2)
        outcomes = []
        errors = []

        def publish(store):
            try:
                barrier.wait()
                outcomes.append(
                    store.publish_prepared(
                        lease.job_id,
                        owner="publisher",
                        generation=lease.lease_generation,
                        policy_fingerprint=background_policy_fingerprint(config),
                        write_timeout_ms=5_000,
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [
            threading.Thread(target=publish, args=(background,)),
            threading.Thread(target=publish, args=(peer,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert len(outcomes) == 2
        assert sum(result.promoted for result in outcomes) == 1
        assert {result.reason for result in outcomes} == {
            "partial_atomic_repair_published",
            "already_published",
        }
        assert background.connection.execute(
            """
            SELECT current_frontier_store_id FROM lcm_lifecycle_state
            WHERE conversation_id='conversation'
            """
        ).fetchone()[0] == group_ids[-1]
        assert background.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_jobs
            WHERE job_kind='partial_atomic_repair' AND state='published'
            """
        ).fetchone()[0] == 1
    finally:
        if peer is not None:
            peer.close()
        background.close()


def test_repair_root_spill_expands_distinct_without_changing_covered_ownership(
    tmp_path,
):
    db_path = tmp_path / "root-spill.db"
    background, config, source_ids, group_ids, _owners, _roots = _partial_fixture(
        db_path,
        root_depths=(1,),
        owner_spill=True,
    )
    try:
        assert _capture_repair(background, config) is not None
        covered_before = [
            tuple(row)
            for row in background.connection.execute(
                """
                SELECT node_id, store_id FROM lcm_compaction_node_sources
                WHERE session_id='session' AND store_id IN (?,?)
                ORDER BY node_id, store_id
                """,
                group_ids[:2],
            ).fetchall()
        ]
        _prepared, lease = _bind_and_prepare_without_provider(background, config)
        result = background.publish_prepared(
            lease.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert result.promoted
        parent_id = result.node_ids[-1]
        dag = SummaryDAG(db_path)
        try:
            assert dag.source_message_ids(parent_id, limit=100) == list(source_ids)
        finally:
            dag.close()
        assert [
            tuple(row)
            for row in background.connection.execute(
                """
                SELECT node_id, store_id FROM lcm_compaction_node_sources
                WHERE session_id='session' AND store_id IN (?,?)
                ORDER BY node_id, store_id
                """,
                group_ids[:2],
            ).fetchall()
        ] == covered_before
    finally:
        background.close()


def test_restart_rebind_discards_progress_bound_to_changed_roots(tmp_path):
    db_path = tmp_path / "restart-rebind.db"
    background, config, _source_ids, _group_ids, _owners, roots = _partial_fixture(
        db_path,
        duplicate_owner=True,
        root_depths=(1,),
    )
    try:
        job = _capture_repair(background, config)
        assert job is not None
        claimed = background.claim_next_job(
            owner="worker-one",
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert claimed is not None
        bound = background._bind_partial_atomic_repair(
            claimed.job_id,
            owner="worker-one",
            generation=claimed.lease_generation,
        )
        assert bound is not None
        assert background.save_prepared_progress(
            bound.job_id,
            owner="worker-one",
            generation=bound.lease_generation,
            prepared_node={
                "ordinal": 0,
                "summary": "stale provider result",
                "source_store_ids": list(bound.source_store_ids),
                "source_token_count": bound.source_token_count,
                "summary_token_count": 3,
            },
        )
        background.connection.execute(
            "UPDATE lcm_compaction_jobs SET lease_expires_at=0 WHERE job_id=?",
            (bound.job_id,),
        )
        background.connection.execute(
            "UPDATE lcm_compaction_provider_leases SET lease_expires_at=0 WHERE job_id=?",
            (bound.job_id,),
        )
        assert background.recover_expired_leases() >= 1
    finally:
        background.close()

    restarted = BackgroundCompactionStore(db_path, config=config)
    try:
        restarted.connection.execute(
            "UPDATE summary_nodes SET summary='new root identity' WHERE node_id=?",
            (roots[0],),
        )
        claimed = restarted.claim_next_job(
            owner="worker-two",
            policy_fingerprint=background_policy_fingerprint(config),
            now=10**12,
        )
        assert claimed is not None
        rebound = restarted._bind_partial_atomic_repair(
            claimed.job_id,
            owner="worker-two",
            generation=claimed.lease_generation,
        )
        assert rebound is not None
        assert restarted.get_prepared_nodes(rebound.job_id) == []
        assert rebound.prepared_node_count == 0
        assert rebound.repair_root_identity_hash != bound.repair_root_identity_hash
    finally:
        restarted.close()


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "duplicate_owner", "root_depths"),
    [
        ("_PARTIAL_REPAIR_MAX_OWNERS", 1, True, ()),
        ("_PARTIAL_REPAIR_MAX_ROOTS", 1, True, (1,)),
        ("_PARTIAL_REPAIR_MAX_ANCESTRY", 1, False, (1,)),
    ],
)
def test_repair_owner_root_and_ancestry_caps_fail_closed(
    tmp_path,
    monkeypatch,
    limit_name,
    limit_value,
    duplicate_owner,
    root_depths,
):
    background, config, _source_ids, _group_ids, _owners, _roots = _partial_fixture(
        tmp_path / f"cap-{limit_name}.db",
        duplicate_owner=duplicate_owner,
        root_depths=root_depths,
    )
    monkeypatch.setattr(background_module, limit_name, limit_value)
    try:
        job = _capture_repair(background, config)
        if limit_name == "_PARTIAL_REPAIR_MAX_OWNERS":
            assert job is None
            assert not background.has_active_partial_atomic_repair(
                conversation_id="conversation",
                session_id="session",
                expected_frontier_store_id=0,
            )
            return
        assert job is not None
        claimed = background.claim_next_job(
            owner="worker",
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert claimed is not None
        assert background._bind_partial_atomic_repair(
            claimed.job_id,
            owner="worker",
            generation=claimed.lease_generation,
        ) is None
        assert background.get_job(job.job_id).state == "superseded"
    finally:
        background.close()


def test_more_than_one_hundred_levels_of_sequential_repair_parents_remain_live(
    tmp_path,
):
    db_path = tmp_path / "deep-repair-chain.db"
    messages = MessageStore(db_path)
    raw_messages = []
    repair_count = 51
    for index in range(repair_count):
        call_id = f"deep-call-{index}"
        raw_messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "deep", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"result {index}",
                },
            ]
        )
    raw_messages.append({"role": "user", "content": "protected tail"})
    source_ids = messages.append_batch(
        "session",
        raw_messages,
        token_estimates=[10] * len(raw_messages),
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    state = lifecycle.bind_session("session", conversation_id="conversation")
    lifecycle.close()
    config = _config(db_path)
    background = BackgroundCompactionStore(db_path, config=config)
    generation = background.activate_session(
        conversation_id="conversation",
        session_id="session",
        expected_lifecycle_bound_at=state.current_bound_at,
    )
    assert generation is not None
    previous_parent = None
    frontier = 0
    try:
        for index in range(repair_count):
            group_ids = tuple(source_ids[index * 2:index * 2 + 2])
            dag = SummaryDAG(db_path)
            try:
                owner = dag.add_node(
                    SummaryNode(
                        session_id="session",
                        summary=f"covered owner {index}",
                        source_ids=[group_ids[0]],
                        source_type="messages",
                        token_count=2,
                        source_token_count=10,
                    )
                )
                if previous_parent is not None:
                    previous_depth = int(dag.get_node(previous_parent).depth)
                    dag.add_node(
                        SummaryNode(
                            session_id="session",
                            depth=previous_depth + 1,
                            summary=f"active root {index}",
                            source_ids=[previous_parent, owner],
                            source_type="nodes",
                            token_count=2,
                            source_token_count=20,
                        )
                    )
            finally:
                dag.close()
            job = background.capture_and_enqueue(
                conversation_id="conversation",
                session_id="session",
                expected_frontier_store_id=frontier,
                active_tokens=300_000,
                effective_context_window=372_000,
                policy_fingerprint=background_policy_fingerprint(config),
            )
            assert job is not None
            assert job.job_kind == "partial_atomic_repair"
            assert job.source_store_ids == group_ids
            _prepared, lease = _bind_and_prepare_without_provider(background, config)
            result = background.publish_prepared(
                lease.job_id,
                owner="publisher",
                generation=lease.lease_generation,
                policy_fingerprint=background_policy_fingerprint(config),
            )
            assert result.promoted
            previous_parent = result.node_ids[-1]
            frontier = group_ids[-1]
        assert previous_parent is not None
        final = background.connection.execute(
            "SELECT depth FROM summary_nodes WHERE node_id=?",
            (previous_parent,),
        ).fetchone()
        assert int(final[0]) > 100
        assert background.connection.execute(
            """
            SELECT current_frontier_store_id FROM lcm_lifecycle_state
            WHERE conversation_id='conversation'
            """
        ).fetchone()[0] == frontier
    finally:
        background.close()


def test_edge_epoch_migration_backfill_and_mutation_triggers(tmp_path):
    db_path = tmp_path / "edge-migration.db"
    dag = SummaryDAG(db_path)
    leaf = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="leaf",
            source_ids=[],
            source_type="messages",
        )
    )
    parent = dag.add_node(
        SummaryNode(
            session_id="session",
            depth=1,
            summary="parent",
            source_ids=[leaf],
            source_type="nodes",
        )
    )
    dag.close()

    stores: list[BackgroundCompactionStore] = []
    errors: list[BaseException] = []

    def open_store():
        try:
            stores.append(BackgroundCompactionStore(db_path, config=_config(db_path)))
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(stores) == 2
    first, second = stores
    try:
        assert first.connection.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name=?",
            (background_module.BACKGROUND_COMPACTION_NODE_EDGES_MIGRATION_STEP,),
        ).fetchone() is not None
        assert [
            tuple(row)
            for row in first.connection.execute(
                "SELECT parent_node_id, child_node_id, session_id FROM lcm_compaction_node_edges"
            ).fetchall()
        ] == [(parent, leaf, "session")]
        epoch_before = first.connection.execute(
            "SELECT mutation_epoch FROM lcm_compaction_dag_epochs WHERE session_id='session'"
        ).fetchone()[0]

        dag = SummaryDAG(db_path)
        other = dag.add_node(
            SummaryNode(
                session_id="session",
                summary="other",
                source_ids=[],
                source_type="messages",
            )
        )
        dag.connection.execute(
            "UPDATE summary_nodes SET source_ids=? WHERE node_id=?",
            (json.dumps([other]), parent),
        )
        dag.connection.commit()
        assert [
            tuple(row)
            for row in first.connection.execute(
                "SELECT parent_node_id, child_node_id, session_id FROM lcm_compaction_node_edges"
            ).fetchall()
        ] == [(parent, other, "session")]
        dag.connection.execute(
            "UPDATE summary_nodes SET session_id='moved' WHERE node_id=?", (parent,)
        )
        dag.connection.commit()
        assert tuple(
            first.connection.execute(
                """
                SELECT parent_node_id, child_node_id, session_id
                FROM lcm_compaction_node_edges
                """
            ).fetchone()
        ) == (parent, other, "moved")
        assert first.connection.execute(
            """
            SELECT COUNT(*)
            FROM lcm_compaction_node_edges AS edge
            JOIN summary_nodes AS parent_node
              ON parent_node.node_id=edge.parent_node_id
             AND parent_node.session_id=edge.session_id
            JOIN summary_nodes AS child_node
              ON child_node.node_id=edge.child_node_id
             AND child_node.session_id=edge.session_id
            """
        ).fetchone()[0] == 0
        dag.connection.execute(
            "UPDATE summary_nodes SET session_id='moved' WHERE node_id=?", (other,)
        )
        dag.connection.commit()
        assert first.connection.execute(
            """
            SELECT COUNT(*)
            FROM lcm_compaction_node_edges AS edge
            JOIN summary_nodes AS parent_node
              ON parent_node.node_id=edge.parent_node_id
             AND parent_node.session_id=edge.session_id
            JOIN summary_nodes AS child_node
              ON child_node.node_id=edge.child_node_id
             AND child_node.session_id=edge.session_id
            """
        ).fetchone()[0] == 1
        dag.connection.execute("DELETE FROM summary_nodes WHERE node_id=?", (parent,))
        dag.connection.commit()
        dag.close()
        assert first.connection.execute(
            "SELECT mutation_epoch FROM lcm_compaction_dag_epochs WHERE session_id='session'"
        ).fetchone()[0] > epoch_before
        assert first.connection.execute(
            "SELECT mutation_epoch FROM lcm_compaction_dag_epochs WHERE session_id='moved'"
        ).fetchone()[0] >= 2
        reverse_plan = " ".join(
            str(row[3])
            for row in first.connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT parent_node_id FROM lcm_compaction_node_edges
                WHERE session_id='session' AND child_node_id=?
                """,
                (leaf,),
            ).fetchall()
        )
        assert "idx_lcm_compaction_node_edges_reverse" in reverse_plan
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("parent_first", [False, True])
def test_whole_session_reassign_preserves_declared_edges_in_either_row_order(
    tmp_path,
    parent_first,
):
    db_path = tmp_path / f"reassign-{parent_first}.db"
    dag = SummaryDAG(db_path)
    dag.close()
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    background.close()
    dag = SummaryDAG(db_path)
    try:
        if parent_first:
            next_id = int(
                dag.connection.execute(
                    "SELECT COALESCE(MAX(node_id),0)+1 FROM summary_nodes"
                ).fetchone()[0]
            )
            parent = dag.add_node(
                SummaryNode(
                    session_id="old",
                    depth=1,
                    summary="parent inserted before child",
                    source_ids=[next_id + 1],
                    source_type="nodes",
                )
            )
            child = dag.add_node(
                SummaryNode(
                    session_id="old",
                    summary="child inserted second",
                    source_ids=[],
                    source_type="messages",
                )
            )
            assert parent < child
        else:
            child = dag.add_node(
                SummaryNode(
                    session_id="old",
                    summary="child inserted first",
                    source_ids=[],
                    source_type="messages",
                )
            )
            parent = dag.add_node(
                SummaryNode(
                    session_id="old",
                    depth=1,
                    summary="parent inserted second",
                    source_ids=[child],
                    source_type="nodes",
                )
            )
            assert child < parent
        assert dag.reassign_session_nodes("old", "new") == 2
        assert tuple(
            dag.connection.execute(
                """
                SELECT parent_node_id, child_node_id, session_id
                FROM lcm_compaction_node_edges
                """
            ).fetchone()
        ) == (parent, child, "new")
        assert dag.connection.execute(
            """
            SELECT COUNT(*)
            FROM lcm_compaction_node_edges AS edge
            JOIN summary_nodes AS parent_node
              ON parent_node.node_id=edge.parent_node_id
             AND parent_node.session_id=edge.session_id
            JOIN summary_nodes AS child_node
              ON child_node.node_id=edge.child_node_id
             AND child_node.session_id=edge.session_id
            WHERE edge.session_id='new'
            """
        ).fetchone()[0] == 1
    finally:
        dag.close()
