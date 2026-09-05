"""Release-blocking latency, lease, and permit contracts."""

from __future__ import annotations

import sqlite3
import threading
import time

import hermes_lcm.background_compaction as background_compaction_module
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


class Clock:
    def __init__(self, value: float = 1_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def _config(db_path, **overrides) -> LCMConfig:
    values = dict(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=1,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=5,
        background_compaction_lease_seconds=30,
        background_compaction_backlog_tokens=1,
        background_compaction_min_provider_source_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=20,
        summary_model="test/provider-model",
    )
    values.update(overrides)
    return LCMConfig(**values)


def _seed(db_path, count: int = 8, *, token_estimate: int = 10) -> list[int]:
    store = MessageStore(db_path)
    try:
        ids = store.append_batch(
            "s",
            [{"role": "user", "content": f"raw-{index}"} for index in range(count)],
            token_estimates=[token_estimate] * count,
            conversation_id="c",
        )
    finally:
        store.close()
    SummaryDAG(db_path).close()
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session("s", conversation_id="c")
    finally:
        lifecycle.close()
    return ids


def _enqueue(background: BackgroundCompactionStore, ids: list[int]):
    return background.enqueue_or_coalesce(
        conversation_id="c",
        session_id="s",
        expected_frontier_store_id=0,
        captured_watermark_store_id=ids[-1],
        source_store_ids=ids,
        source_token_count=len(ids) * 10,
        policy_fingerprint=background_policy_fingerprint(background.config),
        expected_node_count=1,
    )


def _prepare(background: BackgroundCompactionStore, job):
    claimed = background.claim_next_job(owner="worker")
    assert claimed is not None
    prepared = background.mark_prepared(
        job.job_id,
        owner="worker",
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "prepared",
                "source_store_ids": list(job.source_store_ids),
                "source_token_count": job.source_token_count,
                "summary_token_count": 2,
            }
        ],
    )
    assert prepared is not False
    return prepared


def test_capture_reads_only_a_bounded_window_not_the_full_durable_history(
    tmp_path,
    monkeypatch,
):
    """Foreground eligibility work scales with one job, not total history."""
    db_path = tmp_path / "bounded-capture.db"
    _seed(db_path, count=5_000)
    config = _config(db_path)
    background = BackgroundCompactionStore(db_path, config=config)
    converted_rows = 0
    original = background._message_dict

    def counted(row):
        nonlocal converted_rows
        converted_rows += 1
        return original(row)

    monkeypatch.setattr(background, "_message_dict", counted)
    try:
        job = background.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert job is not None
        assert converted_rows <= 1_024
        assert len(job.source_store_ids) < 5_000
    finally:
        background.close()


def test_default_backlog_trigger_detects_many_one_token_rows_without_full_scan(
    tmp_path,
):
    """The configured B-row bound still proves D >= B for tiny messages."""
    db_path = tmp_path / "one-token-backlog.db"
    _seed(db_path, count=32_001, token_estimate=1)
    config = _config(
        db_path,
        background_compaction_backlog_tokens=32_000,
        fresh_tail_count=1,
    )
    background = BackgroundCompactionStore(db_path, config=config)
    try:
        job = background.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert job is not None
        assert len(job.source_store_ids) <= 512
    finally:
        background.close()


def test_capture_does_not_freeze_an_assistant_before_its_late_tool_result(tmp_path):
    """A snapshot ending on unresolved tool calls leaves the opener raw."""
    db_path = tmp_path / "late-tool-result.db"
    store = MessageStore(db_path)
    try:
        ids = store.append_batch(
            "s",
            [
                {"role": "user", "content": "older compactable turn"},
                {
                    "role": "assistant",
                    "content": "calling tool",
                    "tool_calls": [
                        {
                            "id": "call-late",
                            "type": "function",
                            "function": {"name": "late", "arguments": "{}"},
                        }
                    ],
                },
            ],
            token_estimates=[10, 10],
            conversation_id="c",
        )
    finally:
        store.close()
    SummaryDAG(db_path).close()
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session("s", conversation_id="c")
    lifecycle.close()

    config = _config(db_path, fresh_tail_count=0)
    background = BackgroundCompactionStore(db_path, config=config)
    try:
        job = background.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert job is not None
        assert job.source_store_ids == (ids[0],)
        assert job.captured_watermark_store_id == ids[0]

        store = MessageStore(db_path)
        try:
            tool_id = store.append(
                "s",
                {
                    "role": "tool",
                    "tool_call_id": "call-late",
                    "content": "late result",
                },
                token_estimate=10,
                conversation_id="c",
            )
        finally:
            store.close()
        assert tool_id > job.captured_watermark_store_id
        assert ids[1] > job.captured_watermark_store_id
    finally:
        background.close()


def test_publication_rechecks_lease_after_validation_before_any_canonical_write(
    tmp_path,
    monkeypatch,
):
    """A lease that expires during validation cannot commit a frontier."""
    clock = Clock()
    db_path = tmp_path / "publication-expiry.db"
    ids = _seed(db_path, count=4)
    config = _config(db_path, background_compaction_lease_seconds=10)
    background = BackgroundCompactionStore(db_path, config=config, now=clock)
    try:
        job = _enqueue(background, ids)
        _prepare(background, job)
        leased = background.acquire_publication_lease(
            job.job_id,
            owner="publisher",
            lease_seconds=10,
        )
        assert leased is not None
        original = background._source_identity_hash

        def expire_while_validating(messages):
            clock.value = 1_011.0
            return original(messages)

        monkeypatch.setattr(background, "_source_identity_hash", expire_while_validating)
        result = background.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=leased.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert result.promoted is False
        assert result.reason == "lease_lost"
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
            assert conn.execute(
                "SELECT current_frontier_store_id FROM lcm_lifecycle_state "
                "WHERE conversation_id='c'"
            ).fetchone()[0] == 0
        finally:
            conn.close()
    finally:
        background.close()


def test_lease_loss_before_provider_start_releases_unused_global_permit(
    tmp_path,
    monkeypatch,
):
    """A fenced worker that made no request cannot strand shared capacity."""
    db_path = tmp_path / "pre-provider-permit.db"
    ids = _seed(db_path, count=4)
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    try:
        _enqueue(background, ids)
        monkeypatch.setattr(background, "heartbeat", lambda *_args, **_kwargs: False)
        background.run_next_job(
            owner="worker",
            summarizer=lambda **_kwargs: ("must not run", 1),
        )
        assert background.status()["active_provider_permits"] == 0
        assert sqlite3.connect(db_path).execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases"
        ).fetchone()[0] == 0
    finally:
        background.close()


def test_post_start_heartbeat_error_keeps_permit_until_provider_physically_returns(
    tmp_path,
    monkeypatch,
):
    """SQLite heartbeat failure cannot make still-running spend invisible."""
    db_path = tmp_path / "heartbeat-error-permit.db"
    ids = _seed(db_path, count=4)
    config = _config(db_path, background_compaction_lease_seconds=1)
    background = BackgroundCompactionStore(db_path, config=config)
    observer = BackgroundCompactionStore(db_path, config=config)
    _enqueue(background, ids)
    provider_entered = threading.Event()
    provider_release = threading.Event()
    worker_done = threading.Event()
    heartbeat_calls = 0
    original_heartbeat = background.heartbeat

    def flaky_heartbeat(*args, **kwargs):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls <= 2:
            return original_heartbeat(*args, **kwargs)
        raise sqlite3.OperationalError("injected heartbeat lock fault")

    def provider(**_kwargs):
        provider_entered.set()
        assert provider_release.wait(timeout=5)
        return "late result", 1

    def worker():
        try:
            background.run_next_job(owner="worker", summarizer=provider)
        except sqlite3.OperationalError:
            pass
        finally:
            worker_done.set()

    monkeypatch.setattr(background, "heartbeat", flaky_heartbeat)
    thread = threading.Thread(target=worker)
    try:
        thread.start()
        assert provider_entered.wait(timeout=5)
        assert worker_done.wait(timeout=5)
        time.sleep(1.2)
        assert observer.status()["active_provider_permits"] == 1

        provider_release.set()
        thread.join(timeout=5)
        deadline = time.monotonic() + 3
        while observer.status()["active_provider_permits"] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert observer.status()["active_provider_permits"] == 0
    finally:
        provider_release.set()
        thread.join(timeout=5)
        background.close()
        observer.close()


def test_guardian_lock_past_expiry_keeps_global_limit_until_provider_returns(
    tmp_path,
    monkeypatch,
):
    """A missed heartbeat cannot expose live physical provider capacity."""
    db_path = tmp_path / "guardian-lock-owner-liveness.db"
    config = _config(
        db_path,
        background_compaction_global_limit=1,
        background_compaction_lease_seconds=1,
    )
    process_identity = (4321, "987654", "test-boot-id")
    monkeypatch.setattr(
        background_compaction_module,
        "_current_linux_process_identity",
        lambda: process_identity,
        raising=False,
    )
    monkeypatch.setattr(
        background_compaction_module,
        "_linux_process_identity_is_live",
        lambda pid, start_ticks, boot_id: (
            (pid, start_ticks, boot_id) == process_identity
        ),
        raising=False,
    )
    owner = BackgroundCompactionStore(db_path, config=config)
    contender = BackgroundCompactionStore(db_path, config=config)
    permit = owner.acquire_emergency_provider_permit(
        conversation_id="physical-call",
        owner="process-a",
        global_limit=1,
    )
    assert permit is not None
    completed = threading.Event()
    heartbeat_locked = threading.Event()
    release_locked = threading.Event()
    allow_heartbeat = threading.Event()
    allow_release = threading.Event()
    original_heartbeat = BackgroundCompactionStore.heartbeat_provider_permit
    original_release = BackgroundCompactionStore.release_provider_permit

    def locked_heartbeat(self, *args, **kwargs):
        if self is not owner and not allow_heartbeat.is_set():
            heartbeat_locked.set()
            raise sqlite3.OperationalError("database is locked")
        return original_heartbeat(self, *args, **kwargs)

    def locked_release(self, *args, **kwargs):
        if self is not owner and not allow_release.is_set():
            release_locked.set()
            raise sqlite3.OperationalError("database is locked")
        return original_release(self, *args, **kwargs)

    monkeypatch.setattr(
        BackgroundCompactionStore,
        "heartbeat_provider_permit",
        locked_heartbeat,
    )
    monkeypatch.setattr(
        BackgroundCompactionStore,
        "release_provider_permit",
        locked_release,
    )
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT provider_active, owner_pid, owner_start_ticks, owner_boot_id
                FROM lcm_compaction_provider_leases WHERE job_id=?
                """,
                (permit.job_id,),
            ).fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] > 0
        assert row[2]
        assert row[3]

        owner._start_provider_permit_guardian(
            completed,
            job_id=permit.job_id,
            owner=permit.lease_owner,
            generation=permit.lease_generation,
        )
        assert heartbeat_locked.wait(timeout=3)

        # Force the durable timestamp past expiry while the guardian is unable
        # to write. Recovery must prove the process identity is still live and
        # renew the reservation instead of admitting a second physical call.
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE lcm_compaction_provider_leases SET lease_expires_at=0 "
                "WHERE job_id=?",
                (permit.job_id,),
            )
            conn.commit()
        finally:
            conn.close()
        assert contender.acquire_emergency_provider_permit(
            conversation_id="competing-call",
            owner="process-b",
            global_limit=1,
        ) is None
        assert contender.status()["active_provider_permits"] == 1

        completed.set()
        assert release_locked.wait(timeout=3)
        allow_heartbeat.set()
        allow_release.set()
        deadline = time.monotonic() + 3
        while contender.status()["active_provider_permits"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert contender.status()["active_provider_permits"] == 0
    finally:
        completed.set()
        allow_heartbeat.set()
        allow_release.set()
        owner.close()
        contender.close()


def test_provider_owner_schema_upgrades_legacy_permit_table(tmp_path):
    """An existing background marker cannot skip the owner-fence upgrade."""
    db_path = tmp_path / "provider-owner-schema-upgrade.db"
    _seed(db_path, count=2)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE lcm_compaction_provider_leases (
                job_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                lease_generation INTEGER NOT NULL,
                lease_owner TEXT NOT NULL,
                started_at REAL NOT NULL,
                heartbeat_at REAL NOT NULL,
                lease_expires_at REAL NOT NULL,
                timed_out_at REAL
            );
            INSERT INTO lcm_compaction_provider_leases VALUES(
                'legacy-permit', 'legacy-conversation', 1, 'legacy-owner',
                1, 1, 1, NULL
            );
            INSERT OR REPLACE INTO lcm_migration_state(step_name, completed_at)
            VALUES('background_compaction_v1', 1);
            """
        )
        conn.commit()
    finally:
        conn.close()

    background = BackgroundCompactionStore(
        db_path,
        config=_config(db_path),
        now=lambda: 10.0,
    )
    try:
        columns = {
            row[1]
            for row in background.connection.execute(
                "PRAGMA table_info(lcm_compaction_provider_leases)"
            )
        }
        assert {
            "provider_active",
            "owner_pid",
            "owner_start_ticks",
            "owner_boot_id",
        } <= columns
        migration_step = getattr(
            background_compaction_module,
            "BACKGROUND_COMPACTION_PROVIDER_OWNER_MIGRATION_STEP",
        )
        assert background.connection.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name=?",
            (migration_step,),
        ).fetchone() is not None
        assert background.recover_expired_leases() == 1
        assert background.connection.execute(
            "SELECT 1 FROM lcm_compaction_provider_leases"
        ).fetchone() is None
    finally:
        background.close()


def test_expired_provider_permit_is_reaped_after_exact_owner_dies(
    tmp_path,
    monkeypatch,
):
    """The fail-closed fence does not strand capacity after owner death."""
    db_path = tmp_path / "provider-owner-death.db"
    config = _config(db_path, background_compaction_global_limit=1)
    process_identity = (4321, "987654", "test-boot-id")
    monkeypatch.setattr(
        background_compaction_module,
        "_current_linux_process_identity",
        lambda: process_identity,
    )
    monkeypatch.setattr(
        background_compaction_module,
        "_linux_process_identity_is_live",
        lambda *_args: False,
    )
    owner = BackgroundCompactionStore(db_path, config=config)
    contender = BackgroundCompactionStore(db_path, config=config)
    try:
        stale = owner.acquire_emergency_provider_permit(
            conversation_id="stale-call",
            owner="dead-process",
            global_limit=1,
        )
        assert stale is not None
        owner.connection.execute(
            "UPDATE lcm_compaction_provider_leases SET lease_expires_at=0 "
            "WHERE job_id=?",
            (stale.job_id,),
        )
        replacement = contender.acquire_emergency_provider_permit(
            conversation_id="replacement-call",
            owner="live-process",
            global_limit=1,
        )
        assert replacement is not None
        assert replacement.job_id != stale.job_id
        assert contender.connection.execute(
            "SELECT 1 FROM lcm_compaction_provider_leases WHERE job_id=?",
            (stale.job_id,),
        ).fetchone() is None
    finally:
        owner.close()
        contender.close()
