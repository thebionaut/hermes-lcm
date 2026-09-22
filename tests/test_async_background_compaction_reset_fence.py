"""Durable publication fences around reset and compression rollover."""

from __future__ import annotations

import sqlite3
import time

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    PreparedSummaryNode,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore


def _config(db_path):
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=2,
        background_compaction_backlog_tokens=1,
        background_compaction_min_provider_source_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=1,
        new_session_retain_depth=0,
    )


def _prepared_job(engine: LCMEngine):
    ids = engine._store.append_batch(
        engine._session_id,
        [
            {"role": "user", "content": "old raw one"},
            {"role": "assistant", "content": "old raw two"},
        ],
        token_estimates=[10, 10],
        conversation_id=engine._conversation_id,
    )
    job = engine._background_compaction.enqueue_or_coalesce(
        conversation_id=engine._conversation_id,
        session_id=engine._session_id,
        expected_frontier_store_id=0,
        captured_watermark_store_id=ids[-1],
        source_store_ids=ids,
        source_token_count=20,
        policy_fingerprint=engine._background_policy_fingerprint(),
        expected_node_count=1,
    )
    claimed = engine._background_compaction.claim_next_job(owner="worker")
    assert claimed is not None and claimed.job_id == job.job_id
    prepared = engine._background_compaction.mark_prepared(
        job.job_id,
        owner="worker",
        generation=claimed.lease_generation,
        prepared_nodes=[
            PreparedSummaryNode(
                ordinal=0,
                summary="prepared summary",
                source_store_ids=tuple(ids),
                source_token_count=20,
                summary_token_count=2,
            )
        ],
        expected_node_count=1,
    )
    assert prepared and prepared.state == "prepared"
    return prepared, ids


def test_reset_fences_prepared_publisher_between_delete_batches(tmp_path, monkeypatch):
    engine = LCMEngine(_config(tmp_path / "prepared-reset.db"))
    engine.on_session_start("session", conversation_id="conversation")
    prepared, raw_ids = _prepared_job(engine)
    lease = engine._background_compaction.acquire_publication_lease(
        prepared.job_id,
        owner="stale-publisher",
    )
    assert lease is not None

    for index in range(300):
        engine._dag.add_node(
            SummaryNode(
                session_id="session",
                depth=1,
                summary=f"derived-{index}",
                source_ids=[],
                source_type="nodes",
                created_at=time.time(),
            )
        )

    attempts = []
    original_purge = engine._purge_embeddings_for_nodes

    def race_after_first_batch(node_ids):
        original_purge(node_ids)
        if not attempts:
            attempts.append(
                engine._background_compaction.publish_prepared(
                    prepared.job_id,
                    owner="stale-publisher",
                    generation=lease.lease_generation,
                    policy_fingerprint=engine._background_policy_fingerprint(),
                )
            )

    monkeypatch.setattr(engine, "_purge_embeddings_for_nodes", race_after_first_batch)
    try:
        engine.on_session_reset()
        assert len(attempts) == 1
        assert attempts[0].promoted is False
        assert attempts[0].reason in {"job_not_prepared", "lease_lost"}
        assert engine._dag.get_session_node_count("session") == 0
        assert engine._store.get_session_count("session") == len(raw_ids)
        assert engine._lifecycle.get_by_conversation(
            "conversation"
        ).current_frontier_store_id == 0
        assert engine._background_compaction.get_job(prepared.job_id).state == "superseded"
    finally:
        engine.shutdown()


def test_reset_epoch_rejects_stale_emergency_and_reactivates_on_bind(tmp_path):
    engine = LCMEngine(_config(tmp_path / "foreground-reset.db"))
    engine.on_session_start("session", conversation_id="conversation")
    ids = engine._store.append_batch(
        "session",
        [
            {"role": "user", "content": "raw one"},
            {"role": "assistant", "content": "raw two"},
        ],
        token_estimates=[10, 10],
        conversation_id="conversation",
    )
    stale_generation = (
        engine._background_compaction.get_publishable_session_fence_generation(
            conversation_id="conversation",
            session_id="session",
        )
    )
    assert stale_generation is not None
    node = SummaryNode(
        session_id="session",
        summary="emergency summary",
        source_ids=ids,
        source_type="messages",
        source_token_count=20,
        token_count=2,
    )
    source_snapshot = (
        engine._background_compaction.capture_foreground_source_snapshot(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            captured_watermark_store_id=ids[-1],
            expected_source_store_ids=ids,
        )
    )
    assert source_snapshot is not None
    source_identity = engine._background_compaction.foreground_source_identity_hash(
        source_snapshot,
        ids,
    )
    assert source_identity
    try:
        engine.on_session_reset()
        stale = engine._background_compaction.publish_foreground_node(
            node,
            conversation_id="conversation",
            session_id="session",
            expected_session_fence_generation=stale_generation,
            expected_frontier_store_id=0,
            captured_watermark_store_id=ids[-1],
            expected_source_identity_hash=source_identity,
            debt_tokens=20,
        )
        assert stale.promoted is False
        assert stale.reason == "session_reset_or_rebound"
        assert engine._dag.get_session_node_count("session") == 0

        engine.on_session_start("session", conversation_id="conversation")
        active_generation = (
            engine._background_compaction.get_publishable_session_fence_generation(
                conversation_id="conversation",
                session_id="session",
            )
        )
        assert active_generation is not None and active_generation > stale_generation
        current = engine._background_compaction.publish_foreground_node(
            node,
            conversation_id="conversation",
            session_id="session",
            expected_session_fence_generation=active_generation,
            expected_frontier_store_id=0,
            captured_watermark_store_id=ids[-1],
            expected_source_identity_hash=source_identity,
            debt_tokens=20,
        )
        assert current.promoted is True
    finally:
        engine.shutdown()


def test_compression_rollover_fences_old_session_before_reassign(tmp_path):
    engine = LCMEngine(_config(tmp_path / "compression-fence.db"))
    engine.on_session_start("old", conversation_id="conversation")
    raw_id = engine._store.append_batch(
        "old",
        [{"role": "user", "content": "old raw"}],
        token_estimates=[10],
        conversation_id="conversation",
    )[0]
    engine._dag.add_node(
        SummaryNode(
            session_id="old",
            summary="carried summary",
            source_ids=[raw_id],
            source_type="messages",
            source_token_count=10,
            token_count=2,
        )
    )
    old_generation = (
        engine._background_compaction.get_publishable_session_fence_generation(
            conversation_id="conversation",
            session_id="old",
        )
    )
    assert old_generation is not None
    old_snapshot = engine._background_compaction.capture_foreground_source_snapshot(
        conversation_id="conversation",
        session_id="old",
        expected_frontier_store_id=0,
        captured_watermark_store_id=raw_id,
        expected_source_store_ids=[raw_id],
    )
    assert old_snapshot is not None
    old_identity = engine._background_compaction.foreground_source_identity_hash(
        old_snapshot,
        [raw_id],
    )
    assert old_identity
    try:
        engine.on_session_start(
            "new",
            conversation_id="conversation",
            boundary_reason="compression",
            old_session_id="old",
        )
        assert engine._dag.get_session_nodes("old") == []
        assert len(engine._dag.get_session_nodes("new")) == 1
        stale = engine._background_compaction.publish_foreground_node(
            SummaryNode(
                session_id="old",
                summary="stale emergency",
                source_ids=[raw_id],
                source_type="messages",
            ),
            conversation_id="conversation",
            session_id="old",
            expected_session_fence_generation=old_generation,
            expected_frontier_store_id=0,
            captured_watermark_store_id=raw_id,
            expected_source_identity_hash=old_identity,
        )
        assert stale.promoted is False
        assert stale.reason == "session_reset_or_rebound"
    finally:
        engine.shutdown()


def test_transient_bind_activation_lock_recovers_on_later_foreground_enqueue(
    tmp_path,
    monkeypatch,
):
    engine = LCMEngine(_config(tmp_path / "activation-retry.db"))
    original_activate = engine._background_compaction.activate_session
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_activate(**kwargs)

    monkeypatch.setattr(
        engine._background_compaction,
        "activate_session",
        fail_once,
    )
    try:
        engine.on_session_start("session", conversation_id="conversation")
        assert (
            engine._background_compaction.get_publishable_session_fence_generation(
                conversation_id="conversation",
                session_id="session",
            )
            is None
        )
        engine._store.append_batch(
            "session",
            [
                {"role": "user", "content": "old raw"},
                {"role": "assistant", "content": "older reply"},
                {"role": "user", "content": "protected tail"},
            ],
            token_estimates=[10, 10, 10],
            conversation_id="conversation",
        )
        job = engine._maybe_enqueue_background_compaction(
            [
                {"role": "user", "content": "old raw"},
                {"role": "assistant", "content": "older reply"},
                {"role": "user", "content": "protected tail"},
            ],
            observed_tokens=30,
        )
        assert attempts >= 2
        assert job is not None
        assert (
            engine._background_compaction.get_publishable_session_fence_generation(
                conversation_id="conversation",
                session_id="session",
            )
            is not None
        )
    finally:
        engine.shutdown()


def test_stale_process_cannot_activate_after_newer_lifecycle_bind(tmp_path):
    db_path = tmp_path / "stale-bind-cas.db"
    lifecycle = LifecycleStateStore(db_path)
    first = lifecycle.bind_session("session-a", conversation_id="conversation")
    second = lifecycle.bind_session("session-b", conversation_id="conversation")
    lifecycle.close()
    assert first.current_bound_at is not None
    assert second.current_bound_at is not None
    assert first.current_bound_at != second.current_bound_at

    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        current_generation = background.activate_session(
            conversation_id="conversation",
            session_id="session-b",
            expected_lifecycle_bound_at=second.current_bound_at,
        )
        assert current_generation is not None
        stale_generation = background.activate_session(
            conversation_id="conversation",
            session_id="session-a",
            expected_lifecycle_bound_at=first.current_bound_at,
        )
        assert stale_generation is None
        fence = background.connection.execute(
            """
            SELECT session_id, lifecycle_bound_at, generation, publishable
            FROM lcm_compaction_session_fences WHERE conversation_id=?
            """,
            ("conversation",),
        ).fetchone()
        assert fence is not None
        assert fence["session_id"] == "session-b"
        assert float(fence["lifecycle_bound_at"]) == second.current_bound_at
        assert int(fence["generation"]) == current_generation
        assert int(fence["publishable"]) == 1
    finally:
        background.close()
