"""Deterministic concurrency stress for durable background compaction.

These tests use independent SQLite connections and event/barrier coordination.
They deliberately avoid sleep-based ordering so failures identify a contract
violation rather than scheduler timing.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


class Clock:
    def __init__(self, value: float = 10_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _config(db_path, **overrides) -> LCMConfig:
    values = dict(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=2,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=5,
        background_compaction_lease_seconds=30,
        background_compaction_backlog_tokens=1,
        background_compaction_min_provider_source_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=100,
        context_reserve_tokens=100_000,
    )
    values.update(overrides)
    return LCMConfig(**values)


def _seed(
    db_path,
    *,
    session: str,
    conversation: str,
    count: int = 6,
) -> tuple[list[int], list[tuple[int, str]]]:
    messages = [
        {"role": "user", "content": f"{conversation}-raw-{index}"}
        for index in range(count)
    ]
    store = MessageStore(db_path)
    ids = store.append_batch(
        session,
        messages,
        token_estimates=[10] * count,
        conversation_id=conversation,
    )
    store.close()
    dag = SummaryDAG(db_path)
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session(session, conversation_id=conversation)
    lifecycle.close()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT store_id, content FROM messages WHERE session_id=? ORDER BY store_id",
            (session,),
        ).fetchall()
    finally:
        conn.close()
    return ids, rows


def _enqueue(
    store: BackgroundCompactionStore,
    ids: list[int],
    *,
    session: str,
    conversation: str,
    policy: str,
    priority: int = 0,
):
    return store.enqueue_or_coalesce(
        conversation_id=conversation,
        session_id=session,
        expected_frontier_store_id=0,
        captured_watermark_store_id=ids[-1],
        source_store_ids=ids,
        source_token_count=len(ids) * 10,
        policy_fingerprint=policy,
        expected_node_count=1,
        priority=priority,
    )


def _prepare_one(
    store: BackgroundCompactionStore,
    job,
    *,
    owner: str,
):
    claimed = store.claim_next_job(owner=owner)
    assert claimed is not None
    assert claimed.job_id == job.job_id
    prepared = store.mark_prepared(
        job.job_id,
        owner=owner,
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "prepared once",
                "source_store_ids": list(job.source_store_ids),
                "source_token_count": job.source_token_count,
                "summary_token_count": 2,
                "creation_model": "test-model",
                "creation_provider": "test-provider",
            }
        ],
    )
    assert prepared is not False
    return prepared


def test_two_independent_publishers_race_same_lease_and_only_one_commits(tmp_path):
    """Two real publication transactions cannot duplicate a canonical range."""
    db_path = tmp_path / "two-publisher-cas.db"
    ids, raw_before = _seed(db_path, session="s", conversation="c", count=4)
    config = _config(db_path)
    first = BackgroundCompactionStore(db_path, config=config)
    second = BackgroundCompactionStore(db_path, config=config)
    policy = background_policy_fingerprint(config)
    job = _enqueue(
        first,
        ids,
        session="s",
        conversation="c",
        policy=policy,
    )
    _prepare_one(first, job, owner="worker")
    leased = first.acquire_publication_lease(job.job_id, owner="publisher")
    assert leased is not None

    start = threading.Barrier(3)
    results = []
    errors = []
    result_lock = threading.Lock()

    def publish(store: BackgroundCompactionStore) -> None:
        try:
            start.wait(timeout=5)
            result = store.publish_prepared(
                job.job_id,
                owner="publisher",
                generation=leased.lease_generation,
                policy_fingerprint=policy,
            )
            with result_lock:
                results.append(result)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=publish, args=(first,)),
        threading.Thread(target=publish, args=(second,)),
    ]
    try:
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert len(results) == 2
        assert sum(result.promoted for result in results) == 1
        assert {result.reason for result in results} == {"published", "already_published"}

        conn = sqlite3.connect(db_path)
        try:
            canonical = conn.execute(
                "SELECT source_ids FROM summary_nodes WHERE session_id='s'"
            ).fetchall()
            frontier = conn.execute(
                "SELECT current_frontier_store_id FROM lcm_lifecycle_state "
                "WHERE conversation_id='c'"
            ).fetchone()[0]
            raw_after = conn.execute(
                "SELECT store_id, content FROM messages WHERE session_id='s' ORDER BY store_id"
            ).fetchall()
            state = conn.execute(
                "SELECT state FROM lcm_compaction_jobs WHERE job_id=?",
                (job.job_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        assert canonical == [(json.dumps(ids, separators=(",", ":")),)]
        assert frontier == ids[-1]
        assert raw_after == raw_before
        assert state == "published"
        assert first.status()["active_provider_permits"] == 0
    finally:
        for thread in threads:
            thread.join(timeout=1)
        first.close()
        second.close()


def test_foreground_ingest_stays_bounded_when_enqueue_writer_is_locked(
    tmp_path,
    monkeypatch,
):
    """A lower-priority enqueue lock cannot turn committed raw ingest into a stall."""
    db_path = tmp_path / "foreground-writer-lock.db"
    engine = LCMEngine(_config(db_path, background_compaction_workers=0))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"raw-{index} " + "x " * 20}
        for index in range(6)
    ]
    writer = sqlite3.connect(db_path, timeout=0.0, isolation_level=None)
    original_ingest = engine._ingest_messages
    lock_acquired = threading.Event()

    def ingest_then_hold_writer_lock(batch):
        original_ingest(batch)
        writer.execute("PRAGMA busy_timeout=0")
        writer.execute("BEGIN IMMEDIATE")
        lock_acquired.set()

    monkeypatch.setattr(engine, "_ingest_messages", ingest_then_hold_writer_lock)
    try:
        started = time.monotonic()
        engine.ingest(messages)
        elapsed = time.monotonic() - started
        assert lock_acquired.is_set()
        assert elapsed < 0.75

        writer.execute("ROLLBACK")
        count = engine._store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='s'"
        ).fetchone()[0]
        assert count == len(messages)
        assert engine.get_background_compaction_status()["queued_jobs"] == 0

        # Prove this capture was eligible: it queues immediately after the
        # independent lock is gone, rather than the first call being a no-op.
        queued = engine._maybe_enqueue_background_compaction(
            messages,
            observed_tokens=10_000,
        )
        assert queued is not None
        assert engine.get_background_compaction_status()["queued_jobs"] == 1
    finally:
        if writer.in_transaction:
            writer.execute("ROLLBACK")
        writer.close()
        engine.shutdown()


def test_expired_worker_restart_publishes_once_and_stale_owner_cannot_duplicate(
    tmp_path,
):
    """Crash recovery fences stale generations and publication stays idempotent."""
    clock = Clock()
    db_path = tmp_path / "restart-idempotence.db"
    ids, raw_before = _seed(db_path, session="s", conversation="c", count=4)
    config = _config(db_path, background_compaction_lease_seconds=10)
    first = BackgroundCompactionStore(db_path, config=config, now=clock)
    policy = background_policy_fingerprint(config)
    job = _enqueue(
        first,
        ids,
        session="s",
        conversation="c",
        policy=policy,
    )
    stale = first.claim_next_job(owner="crashed-worker")
    assert stale is not None
    assert first.status()["active_provider_permits"] == 1
    first.close()

    clock.advance(11)
    restarted = BackgroundCompactionStore(db_path, config=config, now=clock)
    try:
        assert restarted.recover_expired_leases() == 1
        assert restarted.mark_prepared(
            job.job_id,
            owner="crashed-worker",
            generation=stale.lease_generation,
            prepared_nodes=[],
        ) is False
        replacement = restarted.claim_next_job(owner="replacement-worker")
        assert replacement is not None
        assert replacement.job_id == job.job_id
        assert replacement.lease_generation > stale.lease_generation
        prepared = restarted.mark_prepared(
            job.job_id,
            owner="replacement-worker",
            generation=replacement.lease_generation,
            prepared_nodes=[
                {
                    "ordinal": 0,
                    "summary": "recovered summary",
                    "source_store_ids": ids,
                    "source_token_count": 40,
                    "summary_token_count": 2,
                    "creation_model": "test-model",
                    "creation_provider": "test-provider",
                }
            ],
        )
        assert prepared is not False
        publication_lease = restarted.acquire_publication_lease(
            job.job_id,
            owner="publisher",
        )
        assert publication_lease is not None
        published = restarted.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=publication_lease.lease_generation,
            policy_fingerprint=policy,
        )
        assert published.promoted
    finally:
        restarted.close()

    after_restart = BackgroundCompactionStore(db_path, config=config, now=clock)
    try:
        assert after_restart.recover_expired_leases() == 0
        assert after_restart.claim_next_job(owner="third-worker") is None
        duplicate = after_restart.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=publication_lease.lease_generation,
            policy_fingerprint=policy,
        )
        assert duplicate.promoted is False
        assert duplicate.reason == "already_published"
        assert after_restart.status()["active_provider_permits"] == 0

        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 1
            assert conn.execute(
                "SELECT state FROM lcm_compaction_jobs WHERE job_id=?",
                (job.job_id,),
            ).fetchone()[0] == "published"
            raw_after = conn.execute(
                "SELECT store_id, content FROM messages WHERE session_id='s' ORDER BY store_id"
            ).fetchall()
        finally:
            conn.close()
        assert raw_after == raw_before
    finally:
        after_restart.close()


def test_superseded_inflight_provider_holds_permit_until_physical_return(tmp_path):
    """Supersession fences output but cannot hide still-running provider spend."""
    db_path = tmp_path / "supersede-provider-permit.db"
    first_ids, raw_before = _seed(db_path, session="s1", conversation="c1", count=4)
    second_ids, _ = _seed(db_path, session="s2", conversation="c2", count=4)
    config = _config(
        db_path,
        background_compaction_global_limit=1,
        background_compaction_lease_seconds=30,
    )
    worker_store = BackgroundCompactionStore(db_path, config=config)
    competing_store = BackgroundCompactionStore(db_path, config=config)
    policy = background_policy_fingerprint(config)
    first_job = _enqueue(
        worker_store,
        first_ids,
        session="s1",
        conversation="c1",
        policy=policy,
        priority=10,
    )
    second_job = _enqueue(
        worker_store,
        second_ids,
        session="s2",
        conversation="c2",
        policy=policy,
    )
    provider_entered = threading.Event()
    provider_release = threading.Event()
    provider_returned = threading.Event()
    result_box = {}

    def blocked_provider(**_kwargs):
        provider_entered.set()
        try:
            assert provider_release.wait(timeout=5)
            return "late superseded summary", 1
        finally:
            provider_returned.set()

    def run_worker() -> None:
        result_box["job"] = worker_store.run_next_job(
            owner="worker",
            summarizer=blocked_provider,
        )

    thread = threading.Thread(target=run_worker)
    try:
        thread.start()
        assert provider_entered.wait(timeout=5)
        assert worker_store.get_job(first_job.job_id).state == "running"
        assert competing_store.status()["active_provider_permits"] == 1

        assert competing_store.supersede_overlapping(
            conversation_id="c1",
            new_frontier_store_id=first_ids[0],
            reason="foreground_emergency_compaction",
        ) == 1
        assert competing_store.get_job(first_job.job_id).state == "superseded"

        # The logical job is fenced, but the physical provider call still
        # exists. The global permit must remain occupied until it returns.
        assert competing_store.status()["active_provider_permits"] == 1
        assert competing_store.claim_next_job(owner="competing-worker") is None

        provider_release.set()
        assert provider_returned.wait(timeout=5)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result_box["job"].state == "superseded"
        assert competing_store.status()["active_provider_permits"] == 0
        assert competing_store.get_prepared_nodes(first_job.job_id) == []

        replacement = competing_store.claim_next_job(owner="competing-worker")
        assert replacement is not None
        assert replacement.job_id == second_job.job_id
        cleaned = competing_store.mark_retry(
            replacement.job_id,
            owner="competing-worker",
            generation=replacement.lease_generation,
            error=RuntimeError("test cleanup"),
        )
        assert cleaned is not False
        assert competing_store.status()["active_provider_permits"] == 0

        conn = sqlite3.connect(db_path)
        try:
            raw_after = conn.execute(
                "SELECT store_id, content FROM messages WHERE session_id='s1' ORDER BY store_id"
            ).fetchall()
        finally:
            conn.close()
        assert raw_after == raw_before
    finally:
        provider_release.set()
        thread.join(timeout=5)
        worker_store.close()
        competing_store.close()
