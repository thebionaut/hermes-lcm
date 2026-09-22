"""Deadline contracts for automatic-foreground LCM store operations.

These tests deliberately contend on the real process-local locks and SQLite
writer lock.  Events are used only as synchronization barriers; none of the
tests polls or sleeps to guess when a competing operation reached its lock.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import pytest

from hermes_lcm.background_compaction import (
    CompactionJob,
    PromotionResult,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.reconcile import ForegroundDeadlineExceeded
from hermes_lcm.store import MessageStore


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> int:
        self.value += seconds
        return 0


def _config(db_path, **overrides) -> LCMConfig:
    values = {
        "database_path": str(db_path),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "background_compaction_backlog_tokens": 1,
        "background_compaction_min_provider_source_tokens": 1,
        "background_compaction_prepare_ratio": 0.01,
        "fresh_tail_count": 1,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 120_000,
        "context_reserve_tokens": 100_000,
    }
    values.update(overrides)
    return LCMConfig(**values)


def _engine(tmp_path, **overrides) -> LCMEngine:
    engine = LCMEngine(_config(tmp_path / "foreground-store-deadline.db", **overrides))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    return engine


def _seed_messages(engine: LCMEngine, count: int = 4) -> list[dict[str, str]]:
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"raw-{index} x x"}
        for index in range(count)
    ]
    engine._store.append_batch(
        "s",
        messages,
        token_estimates=[10] * len(messages),
        conversation_id="c",
    )
    return messages


def _prepare_job(engine: LCMEngine) -> tuple[list[dict[str, str]], CompactionJob]:
    messages = _seed_messages(engine)
    background = engine._background_compaction
    assert background is not None
    job = background.capture_and_enqueue(
        conversation_id="c",
        session_id="s",
        expected_frontier_store_id=0,
        active_tokens=300_000,
        effective_context_window=372_000,
        policy_fingerprint=background_policy_fingerprint(engine._config),
    )
    assert job is not None
    claimed = background.claim_next_job(owner="worker")
    assert claimed is not None and claimed.job_id == job.job_id
    prepared = background.mark_prepared(
        job.job_id,
        owner="worker",
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "prepared summary",
                "source_store_ids": list(job.source_store_ids),
                "source_token_count": job.source_token_count,
                "summary_token_count": 2,
            }
        ],
    )
    assert prepared is not False
    persisted = background.get_job(job.job_id)
    assert persisted is not None and persisted.state == "prepared"
    return messages, persisted


@contextmanager
def _held_lock(lock: threading.RLock) -> Iterator[None]:
    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with lock:
            entered.set()
            assert release.wait(timeout=3)

    holder = threading.Thread(target=hold)
    holder.start()
    assert entered.wait(timeout=2)
    try:
        yield
    finally:
        release.set()
        holder.join(timeout=2)
        assert not holder.is_alive()


def _call_before_releasing_lock(
    lock: threading.RLock,
    operation: Callable[[], object],
) -> tuple[bool, dict[str, object]]:
    finished = threading.Event()
    result: dict[str, object] = {}

    def invoke() -> None:
        try:
            result["value"] = operation()
        except BaseException as exc:
            result["error"] = exc
        finally:
            finished.set()

    with _held_lock(lock):
        caller = threading.Thread(target=invoke)
        caller.start()
        completed_while_contended = finished.wait(timeout=0.25)
    caller.join(timeout=2)
    assert not caller.is_alive()
    return completed_while_contended, result


def _job_count(db_path) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM lcm_compaction_jobs").fetchone()[0])


def test_automatic_enqueue_bounds_lifecycle_python_lock_and_does_not_capture(tmp_path):
    engine = _engine(tmp_path)
    messages = _seed_messages(engine)
    try:
        completed, result = _call_before_releasing_lock(
            engine._lifecycle._lock,
            lambda: engine._maybe_enqueue_background_compaction(
                messages,
                observed_tokens=300_000,
                deadline=time.monotonic() + 0.05,
            ),
        )
        assert completed
        assert result.get("error") is None
        assert result.get("value") is None
        assert _job_count(engine._store.db_path) == 0
    finally:
        engine.shutdown()


def test_automatic_enqueue_bounds_background_python_lock_and_does_not_capture(tmp_path):
    engine = _engine(tmp_path)
    messages = _seed_messages(engine)
    background = engine._background_compaction
    assert background is not None
    try:
        completed, result = _call_before_releasing_lock(
            background._lock,
            lambda: engine._maybe_enqueue_background_compaction(
                messages,
                observed_tokens=300_000,
                deadline=time.monotonic() + 0.05,
            ),
        )
        assert completed
        assert result.get("error") is None
        assert result.get("value") is None
        assert _job_count(engine._store.db_path) == 0
    finally:
        engine.shutdown()


def test_capture_enqueue_sqlite_wait_uses_deadline_and_never_commits_late(tmp_path):
    engine = _engine(tmp_path)
    _seed_messages(engine)
    background = engine._background_compaction
    assert background is not None
    blocker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    blocker.execute("PRAGMA busy_timeout=0")
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
            background.capture_and_enqueue(
                conversation_id="c",
                session_id="s",
                expected_frontier_store_id=0,
                active_tokens=300_000,
                effective_context_window=372_000,
                policy_fingerprint=background_policy_fingerprint(engine._config),
                deadline=started + 0.05,
                deadline_clock=time.monotonic,
            )
        assert time.monotonic() - started < 0.25
        assert _job_count(engine._store.db_path) == 0
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
        engine.shutdown()


def test_capture_enqueue_rolls_back_when_deadline_expires_before_commit(tmp_path):
    engine = _engine(tmp_path)
    _seed_messages(engine)
    background = engine._background_compaction
    assert background is not None
    clock = _FakeClock()
    conn = background.connection
    assert conn is not None
    conn.create_function("expire_foreground_deadline", 0, lambda: clock.advance(1.0))
    conn.execute(
        """
        CREATE TEMP TRIGGER expire_enqueue_after_insert
        AFTER INSERT ON lcm_compaction_jobs
        BEGIN
          SELECT expire_foreground_deadline();
        END
        """
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            background.capture_and_enqueue(
                conversation_id="c",
                session_id="s",
                expected_frontier_store_id=0,
                active_tokens=300_000,
                effective_context_window=372_000,
                policy_fingerprint=background_policy_fingerprint(engine._config),
                deadline=100.5,
                deadline_clock=clock,
            )
        assert _job_count(engine._store.db_path) == 0
        state = engine._lifecycle.get_by_conversation("c")
        assert state is not None and state.debt_size_estimate == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "operation_name",
    ["get_job", "find_prepared_job", "snapshot", "partial_repair"],
)
def test_prepared_and_snapshot_reads_bound_background_python_lock(
    tmp_path,
    operation_name,
):
    engine = _engine(tmp_path)
    messages, job = _prepare_job(engine)
    background = engine._background_compaction
    assert background is not None
    source_ids = tuple(job.source_store_ids)

    def operation():
        deadline = time.monotonic() + 0.05
        if operation_name == "get_job":
            return background.get_job(
                job.job_id,
                lock_timeout_seconds=0.05,
                deadline=deadline,
                deadline_clock=time.monotonic,
            )
        if operation_name == "find_prepared_job":
            return background.find_prepared_job(
                conversation_id="c",
                session_id="s",
                expected_frontier_store_id=0,
                policy_fingerprint=background_policy_fingerprint(engine._config),
                lock_timeout_seconds=0.05,
                deadline=deadline,
                deadline_clock=time.monotonic,
            )
        if operation_name == "snapshot":
            return background.capture_foreground_source_snapshot(
                conversation_id="c",
                session_id="s",
                expected_frontier_store_id=0,
                captured_watermark_store_id=source_ids[-1],
                expected_source_store_ids=source_ids,
                lock_timeout_seconds=0.05,
                deadline=deadline,
                deadline_clock=time.monotonic,
            )
        return background.has_active_partial_atomic_repair(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            lock_timeout_seconds=0.05,
            deadline=deadline,
            deadline_clock=time.monotonic,
        )

    try:
        completed, result = _call_before_releasing_lock(background._lock, operation)
        assert completed
        assert isinstance(result.get("error"), TimeoutError)
        assert result.get("value") is None
        persisted = background.get_job(job.job_id)
        assert persisted is not None and persisted.state == "prepared"
        assert len(messages) == 4
    finally:
        engine.shutdown()


def test_prepared_promotion_python_lock_contention_returns_without_publication(tmp_path):
    engine = _engine(tmp_path)
    messages, job = _prepare_job(engine)
    background = engine._background_compaction
    assert background is not None
    try:
        completed, result = _call_before_releasing_lock(
            background._lock,
            lambda: engine.promote_prepared_compaction(
                job.job_id,
                messages,
                current_tokens=300_000,
                deadline=time.monotonic() + 0.05,
            ),
        )
        assert completed
        assert result.get("error") is None
        promotion = result.get("value")
        assert isinstance(promotion, PromotionResult)
        assert promotion.promoted is False
        assert promotion.messages == messages
        persisted = background.get_job(job.job_id)
        assert persisted is not None and persisted.state == "prepared"
        assert engine._lifecycle.get_by_conversation("c").current_frontier_store_id == 0
    finally:
        engine.shutdown()


def test_prepared_promotion_sqlite_lock_returns_without_lease_or_publication(tmp_path):
    engine = _engine(tmp_path)
    messages, job = _prepare_job(engine)
    background = engine._background_compaction
    assert background is not None
    blocker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    blocker.execute("PRAGMA busy_timeout=0")
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        promotion = engine.promote_prepared_compaction(
            job.job_id,
            messages,
            current_tokens=300_000,
            deadline=started + 0.05,
        )
        assert time.monotonic() - started < 0.25
        assert promotion.promoted is False
        assert promotion.messages == messages
        with sqlite3.connect(engine._store.db_path) as observer:
            row = observer.execute(
                "SELECT state, lease_owner, lease_expires_at FROM lcm_compaction_jobs WHERE job_id=?",
                (job.job_id,),
            ).fetchone()
            assert row == ("prepared", None, None)
            assert observer.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
        engine.shutdown()


def test_prepared_publication_rolls_back_if_deadline_expires_inside_transaction(tmp_path):
    engine = _engine(tmp_path)
    messages, job = _prepare_job(engine)
    background = engine._background_compaction
    assert background is not None
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    background._now = clock
    conn = background.connection
    assert conn is not None
    conn.create_function("expire_publication_deadline", 0, lambda: clock.advance(1.0))
    conn.execute(
        """
        CREATE TEMP TRIGGER expire_publication_after_frontier
        AFTER UPDATE OF current_frontier_store_id ON lcm_lifecycle_state
        BEGIN
          SELECT expire_publication_deadline();
        END
        """
    )
    try:
        promotion = engine.promote_prepared_compaction(
            job.job_id,
            messages,
            current_tokens=300_000,
            deadline=100.5,
        )
        assert promotion.promoted is False
        assert promotion.messages == messages
        with sqlite3.connect(engine._store.db_path) as observer:
            assert observer.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
            assert observer.execute(
                "SELECT current_frontier_store_id FROM lcm_lifecycle_state WHERE conversation_id='c'"
            ).fetchone()[0] == 0
            lease_row = observer.execute(
                "SELECT state, lease_owner, lease_expires_at "
                "FROM lcm_compaction_jobs WHERE job_id=?",
                (job.job_id,),
            ).fetchone()
            assert lease_row[0] == "prepared"
            assert not (
                lease_row[1]
                and float(lease_row[2] or 0.0) > float(clock())
            ), "a timed-out foreground publisher must not retain an active lease"

        status = background.status(conversation_id="c")
        assert status["current_job"]["lease_owner_hint"] is None
        assert status["current_job"]["lease_remaining_seconds"] is None

        recovered = background.acquire_publication_lease(
            job.job_id,
            owner="immediate-recovery",
        )
        assert recovered is not None
        assert recovered.lease_owner == "immediate-recovery"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("operation_name", ["activate", "adopt"])
def test_fence_activation_and_canonical_adoption_bound_python_and_sqlite_waits(
    tmp_path,
    operation_name,
):
    engine = _engine(tmp_path)
    background = engine._background_compaction
    assert background is not None
    lifecycle = engine._lifecycle.get_by_conversation("c")
    assert lifecycle is not None
    generation = background.get_publishable_session_fence_generation(
        conversation_id="c",
        session_id="s",
    )
    assert generation is not None

    def operation():
        deadline = time.monotonic() + 0.05
        if operation_name == "activate":
            return background.activate_session(
                conversation_id="c",
                session_id="s",
                expected_lifecycle_bound_at=lifecycle.current_bound_at,
                expected_lifecycle_reset_at=lifecycle.last_reset_at,
                lock_timeout_seconds=0.05,
                write_deadline=deadline,
                deadline_clock=time.monotonic,
            )
        return background.adopt_contiguous_canonical_prefix(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
            lock_timeout_seconds=0.05,
            write_deadline=deadline,
            deadline_clock=time.monotonic,
        )

    try:
        completed, result = _call_before_releasing_lock(background._lock, operation)
        assert completed
        assert isinstance(result.get("error"), TimeoutError)

        blocker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
        blocker.execute("PRAGMA busy_timeout=0")
        blocker.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
                if operation_name == "activate":
                    background.activate_session(
                        conversation_id="c",
                        session_id="s",
                        expected_lifecycle_bound_at=lifecycle.current_bound_at,
                        expected_lifecycle_reset_at=lifecycle.last_reset_at,
                        write_deadline=started + 0.05,
                        deadline_clock=time.monotonic,
                    )
                else:
                    background.adopt_contiguous_canonical_prefix(
                        conversation_id="c",
                        session_id="s",
                        expected_frontier_store_id=0,
                        expected_session_fence_generation=generation,
                        write_deadline=started + 0.05,
                        deadline_clock=time.monotonic,
                    )
            assert time.monotonic() - started < 0.25
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
    finally:
        engine.shutdown()


def test_foreground_snapshot_contention_never_starts_emergency_provider(tmp_path, monkeypatch):
    engine = _engine(tmp_path, fresh_tail_count=1, leaf_chunk_tokens=1)
    messages = [
        {"role": "user", "content": f"emergency-source-{index} x x x"}
        for index in range(5)
    ]
    provider_calls: list[str] = []
    background = engine._background_compaction
    assert background is not None
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        engine,
        "_maybe_enqueue_background_compaction",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine,
        "_maybe_adopt_existing_canonical_prefix",
        lambda **_kwargs: 0,
    )
    monkeypatch.setattr(engine, "_has_publishable_background_job", lambda **_kwargs: False)
    monkeypatch.setattr(
        background,
        "has_active_partial_atomic_repair",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: provider_calls.append("provider") or ("forbidden", 1),
    )

    try:
        completed, result = _call_before_releasing_lock(
            background._lock,
            lambda: engine.compress(messages, current_tokens=360_000),
        )
        assert completed
        assert result.get("error") is None
        assert result.get("value") is messages
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        engine.shutdown()


def test_store_deadline_options_are_optional_for_background_and_manual_callers(tmp_path):
    engine = _engine(tmp_path)
    messages, job = _prepare_job(engine)
    background = engine._background_compaction
    assert background is not None
    try:
        assert background.get_job(job.job_id) is not None
        assert background.find_prepared_job(conversation_id="c") is not None
        snapshot = background.capture_foreground_source_snapshot(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            captured_watermark_store_id=job.captured_watermark_store_id,
            expected_source_store_ids=job.source_store_ids,
        )
        assert snapshot is not None
        lease = background.acquire_publication_lease(job.job_id, owner="manual")
        assert lease is not None
        published = background.publish_prepared(
            job.job_id,
            owner="manual",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(engine._config),
        )
        assert published.promoted is True
        assert len(messages) == 4
    finally:
        engine.shutdown()


def test_lifecycle_read_deadline_is_optional_and_bounds_its_python_lock(tmp_path):
    db_path = tmp_path / "lifecycle-read-deadline.db"
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session("s", conversation_id="c")
    try:
        assert lifecycle.get_by_conversation("c") is not None
        completed, result = _call_before_releasing_lock(
            lifecycle._lock,
            lambda: lifecycle.get_by_conversation(
                "c",
                lock_timeout_seconds=0.05,
                deadline=time.monotonic() + 0.05,
                deadline_clock=time.monotonic,
            ),
        )
        assert completed
        assert isinstance(result.get("error"), TimeoutError)
    finally:
        lifecycle.close()


def test_failed_foreground_publication_lifecycle_lookup_respects_shared_deadline(
    tmp_path,
    monkeypatch,
):
    """A lost publication race cannot block on the follow-up lifecycle read."""
    engine = _engine(tmp_path, fresh_tail_count=1, leaf_chunk_tokens=1)
    messages = [
        {"role": "user", "content": f"foreground-source-{index} x x x"}
        for index in range(6)
    ]
    engine.ingest(messages)
    publication_reached = threading.Event()
    lifecycle_lock_held = threading.Event()
    release_lifecycle_lock = threading.Event()
    caller_finished = threading.Event()
    result: dict[str, object] = {}

    def hold_lifecycle_after_publication() -> None:
        assert publication_reached.wait(timeout=2)
        with engine._lifecycle._lock:
            lifecycle_lock_held.set()
            assert release_lifecycle_lock.wait(timeout=3)

    def lose_publication(*_args, **_kwargs):
        publication_reached.set()
        assert lifecycle_lock_held.wait(timeout=2)
        return PromotionResult(False, "frontier_mismatch")

    def invoke() -> None:
        try:
            result["value"] = engine.compress(messages, current_tokens=340_000)
        except BaseException as exc:
            result["error"] = exc
        finally:
            caller_finished.set()

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: ("bounded foreground summary", 1),
    )
    monkeypatch.setattr(
        engine,
        "_publish_foreground_background_node",
        lose_publication,
    )
    holder = threading.Thread(target=hold_lifecycle_after_publication)
    caller = threading.Thread(target=invoke)
    holder.start()
    caller.start()
    node_count = -1
    try:
        completed_while_contended = caller_finished.wait(timeout=0.25)
    finally:
        release_lifecycle_lock.set()
        holder.join(timeout=2)
        caller.join(timeout=2)
        node_count = engine._dag.get_session_node_count("s")
        engine.shutdown()
    assert completed_while_contended
    assert not holder.is_alive()
    assert not caller.is_alive()
    assert result.get("error") is None
    assert result.get("value") is messages
    assert node_count == 0


def test_durable_tail_projection_bounds_message_store_python_lock(tmp_path):
    engine = _engine(tmp_path)
    _seed_messages(engine)
    try:
        completed, result = _call_before_releasing_lock(
            engine._store._write_lock,
            lambda: engine._project_durable_active_tail(
                0,
                deadline=time.monotonic() + 0.05,
            ),
        )
        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert result.get("value") is None
    finally:
        engine.shutdown()


def test_protected_batch_bounds_real_python_lock_and_keeps_raw_batch_atomic(tmp_path):
    store = MessageStore(tmp_path / "append-python-lock.db")
    try:
        completed, result = _call_before_releasing_lock(
            store._write_lock,
            lambda: store._append_protected_batch(
                "s",
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
                [1, 1],
                deadline=time.monotonic() + 0.05,
                deadline_clock=time.monotonic,
            ),
        )
        assert completed
        assert isinstance(result.get("error"), TimeoutError)
        assert store.get_session_count("s") == 0
    finally:
        store.close()


def test_protected_batch_bounds_real_sqlite_contention_and_keeps_raw_batch_atomic(
    tmp_path,
):
    store = MessageStore(tmp_path / "append-sqlite-lock.db")
    blocker = sqlite3.connect(str(store.db_path), isolation_level=None)
    blocker.execute("PRAGMA busy_timeout=0")
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
            store._append_protected_batch(
                "s",
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
                [1, 1],
                deadline=started + 0.05,
                deadline_clock=time.monotonic,
            )
        assert time.monotonic() - started < 0.25
        assert store.get_session_count("s") == 0
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
        store.close()


def test_protected_batch_checks_deadline_between_rows_and_rolls_back(tmp_path):
    store = MessageStore(tmp_path / "append-row-deadline.db")
    clock = _FakeClock()
    conn = store.connection
    assert conn is not None
    conn.create_function("expire_append_row_deadline", 0, lambda: clock.advance(1.0))
    conn.execute(
        """
        CREATE TEMP TRIGGER expire_after_first_append_row
        AFTER INSERT ON messages
        WHEN NEW.content = 'first'
        BEGIN
          SELECT expire_append_row_deadline();
        END
        """
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            store._append_protected_batch(
                "s",
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ],
                [1, 1],
                deadline=100.5,
                deadline_clock=clock,
            )
        assert store.get_session_count("s") == 0
    finally:
        store.close()


def test_protected_batch_checks_deadline_before_commit_and_rolls_back(tmp_path):
    store = MessageStore(tmp_path / "append-commit-deadline.db")
    clock = _FakeClock()
    conn = store.connection
    assert conn is not None
    conn.create_function(
        "expire_append_commit_deadline",
        0,
        lambda: clock.advance(1.0),
    )
    conn.execute(
        """
        CREATE TEMP TRIGGER expire_before_append_commit
        AFTER INSERT ON messages
        BEGIN
          SELECT expire_append_commit_deadline();
        END
        """
    )
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            store._append_protected_batch(
                "s",
                [{"role": "user", "content": "only row"}],
                [1],
                deadline=100.5,
                deadline_clock=clock,
            )
        assert store.get_session_count("s") == 0
    finally:
        store.close()


def test_automatic_compress_rolls_back_raw_ingest_when_deadline_expires_before_commit(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    conn = engine._store.connection
    assert conn is not None
    conn.create_function(
        "expire_automatic_ingest_deadline",
        0,
        lambda: clock.advance(1.0),
    )
    conn.execute(
        """
        CREATE TEMP TRIGGER expire_automatic_ingest_before_commit
        AFTER INSERT ON messages
        BEGIN
          SELECT expire_automatic_ingest_deadline();
        END
        """
    )
    messages = [{"role": "user", "content": "preserve this raw turn"}]
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.5,
    )
    try:
        result = engine.compress(messages, current_tokens=1)
        assert result is messages
        assert engine._store.get_session_count("s") == 0
        assert engine._ingest_cursor == 0
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle is not None
        assert lifecycle.current_frontier_store_id == 0
    finally:
        engine.shutdown()
