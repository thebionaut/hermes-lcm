"""Publication-fence races at ordinary Hermes turn boundaries."""

from __future__ import annotations

import pytest

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    CompactionJob,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


_POLICY = "publication-fence-sprint-v1"


def _config(db_path):
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=2,
        background_compaction_backlog_tokens=1,
        background_compaction_prepare_ratio=0.0,
        background_compaction_max_passes=4,
        background_compaction_lease_seconds=300,
        fresh_tail_count=1,
        leaf_chunk_tokens=10_000,
    )


def _prepare_standard(db_path):
    config = _config(db_path)
    messages = MessageStore(db_path)
    source_ids = messages.append_batch(
        "session",
        [
            {"role": "user", "content": "already committed prefix"},
            {"role": "assistant", "content": "captured raw one"},
            {"role": "user", "content": "captured raw two"},
        ],
        token_estimates=[10, 20, 20],
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    dag.close()

    lifecycle = LifecycleStateStore(db_path)
    state = lifecycle.bind_session("session", conversation_id="conversation")
    state = lifecycle.advance_frontier(
        "conversation", "session", source_ids[0]
    )
    assert state is not None
    assert state.current_frontier_store_id == source_ids[0]

    background = BackgroundCompactionStore(db_path, config=config)
    fence_generation = background.activate_session(
        conversation_id="conversation",
        session_id="session",
        expected_lifecycle_bound_at=state.current_bound_at,
        expected_lifecycle_reset_at=state.last_reset_at,
    )
    assert fence_generation is not None
    job = background.enqueue_or_coalesce(
        conversation_id="conversation",
        session_id="session",
        expected_frontier_store_id=source_ids[0],
        captured_watermark_store_id=source_ids[-1],
        source_store_ids=source_ids[1:],
        source_token_count=40,
        policy_fingerprint=_POLICY,
        expected_node_count=1,
    )
    assert job is not None
    claimed = background.claim_next_job(
        owner="worker", policy_fingerprint=_POLICY
    )
    assert claimed is not None
    prepared = background.mark_prepared(
        job.job_id,
        owner="worker",
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "prepared provider summary",
                "source_store_ids": list(job.source_store_ids),
                "source_token_count": job.source_token_count,
                "summary_token_count": 3,
                "creation_model": "test-model",
                "creation_provider": "test-provider",
            }
        ],
    )
    assert isinstance(prepared, CompactionJob)
    lease = background.acquire_publication_lease(
        job.job_id, owner="publisher-before-finalize"
    )
    assert lease is not None
    return background, lifecycle, job, lease, source_ids, fence_generation


def _prepare_partial_repair(db_path):
    config = _config(db_path)
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
        token_estimates=[100, 100, 100, 10],
        conversation_id="conversation",
    )
    messages.close()
    group_ids = tuple(source_ids[:3])

    dag = SummaryDAG(db_path)
    dag.add_node(
        SummaryNode(
            session_id="session",
            summary="existing canonical owner",
            source_ids=list(group_ids[:2]),
            source_type="messages",
            token_count=4,
            source_token_count=200,
        )
    )
    dag.close()

    lifecycle = LifecycleStateStore(db_path)
    state = lifecycle.bind_session("session", conversation_id="conversation")
    background = BackgroundCompactionStore(db_path, config=config)
    fence_generation = background.activate_session(
        conversation_id="conversation",
        session_id="session",
        expected_lifecycle_bound_at=state.current_bound_at,
        expected_lifecycle_reset_at=state.last_reset_at,
    )
    assert fence_generation is not None
    policy = background_policy_fingerprint(config)
    job = background.capture_and_enqueue(
        conversation_id="conversation",
        session_id="session",
        expected_frontier_store_id=0,
        active_tokens=300_000,
        effective_context_window=372_000,
        policy_fingerprint=policy,
    )
    assert job is not None
    assert job.job_kind == "partial_atomic_repair"
    claimed = background.claim_next_job(owner="worker", policy_fingerprint=policy)
    assert claimed is not None
    bound = background._bind_partial_atomic_repair(
        claimed.job_id,
        owner="worker",
        generation=claimed.lease_generation,
    )
    assert bound is not None
    prepared = background.mark_prepared(
        claimed.job_id,
        owner="worker",
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "prepared repair parent",
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
        job.job_id, owner="publisher-before-finalize"
    )
    assert lease is not None
    return background, lifecycle, job, lease, policy, fence_generation


def _assert_deferred_and_released(background, job, lease, *, policy):
    result = background.publish_prepared(
        job.job_id,
        owner="publisher-before-finalize",
        generation=lease.lease_generation,
        policy_fingerprint=policy,
    )
    assert result.promoted is False
    assert result.reason == "lifecycle_temporarily_unbound"
    deferred = background.get_job(job.job_id)
    assert deferred is not None
    assert deferred.state == "prepared"
    assert deferred.completion_reason == "prepared"
    assert deferred.lease_owner is None
    assert deferred.lease_expires_at is None
    assert deferred.heartbeat_at is None
    assert deferred.lease_generation == lease.lease_generation + 1
    assert background.connection.execute(
        """
        SELECT COUNT(*) FROM lcm_compaction_events
        WHERE job_id=? AND reason_code='prepared_summary_superseded'
        """,
        (job.job_id,),
    ).fetchone()[0] == 0


def test_standard_publication_defers_during_finalize_gap_then_publishes_once(tmp_path):
    db_path = tmp_path / "standard.db"
    (
        background,
        lifecycle,
        job,
        lease,
        source_ids,
        fence_generation,
    ) = _prepare_standard(db_path)
    rebound_background = None
    try:
        finalized = lifecycle.finalize_session(
            "conversation", "session", frontier_store_id=source_ids[0]
        )
        assert finalized is not None
        assert finalized.current_session_id is None
        assert finalized.last_finalized_session_id == "session"
        assert finalized.last_finalized_frontier_store_id == source_ids[0]

        _assert_deferred_and_released(background, job, lease, policy=_POLICY)
        assert background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == 0

        rebound = lifecycle.bind_session("session", conversation_id="conversation")
        assert rebound.current_frontier_store_id == source_ids[0]
        # A separate store connection models the other Hermes consumer taking
        # over publication after the ordinary turn boundary.
        rebound_background = BackgroundCompactionStore(
            db_path, config=_config(db_path)
        )
        rebound_generation = rebound_background.activate_session(
            conversation_id="conversation",
            session_id="session",
            expected_lifecycle_bound_at=rebound.current_bound_at,
            expected_lifecycle_reset_at=rebound.last_reset_at,
        )
        assert rebound_generation == fence_generation
        rebound_lease = rebound_background.acquire_publication_lease(
            job.job_id, owner="publisher-after-rebind"
        )
        assert rebound_lease is not None
        published = rebound_background.publish_prepared(
            job.job_id,
            owner="publisher-after-rebind",
            generation=rebound_lease.lease_generation,
            policy_fingerprint=_POLICY,
        )
        assert published.promoted is True
        assert rebound_background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == 1

        duplicate = background.publish_prepared(
            job.job_id,
            owner="publisher-after-rebind",
            generation=rebound_lease.lease_generation,
            policy_fingerprint=_POLICY,
        )
        assert duplicate.promoted is False
        assert duplicate.reason == "already_published"
        assert background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == 1
    finally:
        if rebound_background is not None:
            rebound_background.close()
        background.close()
        lifecycle.close()


def test_partial_repair_publication_defers_during_finalize_gap_then_publishes(tmp_path):
    db_path = tmp_path / "partial.db"
    (
        background,
        lifecycle,
        job,
        lease,
        policy,
        fence_generation,
    ) = _prepare_partial_repair(db_path)
    rebound_background = None
    try:
        node_count = background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0]
        finalized = lifecycle.finalize_session(
            "conversation", "session", frontier_store_id=0
        )
        assert finalized is not None
        assert finalized.current_session_id is None

        _assert_deferred_and_released(background, job, lease, policy=policy)
        assert background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == node_count

        rebound = lifecycle.bind_session("session", conversation_id="conversation")
        rebound_background = BackgroundCompactionStore(
            db_path, config=_config(db_path)
        )
        rebound_generation = rebound_background.activate_session(
            conversation_id="conversation",
            session_id="session",
            expected_lifecycle_bound_at=rebound.current_bound_at,
            expected_lifecycle_reset_at=rebound.last_reset_at,
        )
        assert rebound_generation == fence_generation
        rebound_lease = rebound_background.acquire_publication_lease(
            job.job_id, owner="publisher-after-rebind"
        )
        assert rebound_lease is not None
        published = rebound_background.publish_prepared(
            job.job_id,
            owner="publisher-after-rebind",
            generation=rebound_lease.lease_generation,
            policy_fingerprint=policy,
        )
        assert published.promoted is True
        assert published.reason == "partial_atomic_repair_published"
        assert rebound_background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == node_count + 2
    finally:
        if rebound_background is not None:
            rebound_background.close()
        background.close()
        lifecycle.close()


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("reset", "job_not_prepared"),
        ("rollover", "frontier_mismatch"),
        ("policy", "policy_mismatch"),
        ("frontier", "frontier_mismatch"),
    ],
)
def test_real_publication_invalidations_still_fail_closed(
    tmp_path, mutation, expected_reason
):
    db_path = tmp_path / f"{mutation}.db"
    background, lifecycle, job, lease, source_ids, _generation = _prepare_standard(
        db_path
    )
    try:
        policy = _POLICY
        if mutation == "reset":
            reset_generation = background.begin_session_reset(
                conversation_id="conversation", session_id="session"
            )
            assert reset_generation is not None
        elif mutation == "rollover":
            lifecycle.bind_session("next-session", conversation_id="conversation")
        elif mutation == "policy":
            policy = "different-policy"
        elif mutation == "frontier":
            advanced = lifecycle.advance_frontier(
                "conversation", "session", source_ids[1]
            )
            assert advanced is not None
            assert advanced.current_frontier_store_id == source_ids[1]

        result = background.publish_prepared(
            job.job_id,
            owner="publisher-before-finalize",
            generation=lease.lease_generation,
            policy_fingerprint=policy,
        )
        assert result.promoted is False
        assert result.reason == expected_reason
        persisted = background.get_job(job.job_id)
        assert persisted is not None
        assert persisted.state == "superseded"
        assert persisted.lease_owner is None
        assert background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes"
        ).fetchone()[0] == 0
    finally:
        background.close()
        lifecycle.close()
