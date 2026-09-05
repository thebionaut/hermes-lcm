"""Attempt telemetry must never own job state or provider-permit cleanup."""

from __future__ import annotations

import logging
import threading

import pytest

import hermes_lcm.background_compaction as background_compaction_module
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


def _config(db_path, **overrides) -> LCMConfig:
    values = {
        "database_path": str(db_path),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "background_compaction_global_limit": 1,
        "background_compaction_max_passes": 4,
        "background_compaction_job_timeout_seconds": 1,
        "background_compaction_lease_seconds": 30,
        "background_compaction_backlog_tokens": 1,
        "background_compaction_min_provider_source_tokens": 1,
        "background_compaction_prepare_ratio": 0.55,
        "fresh_tail_count": 0,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 120_000,
        "summary_model": "primary/model-a",
        "summary_fallback_models": ["fallback/model-b"],
    }
    values.update(overrides)
    return LCMConfig(**values)


def _seed_and_enqueue(background: BackgroundCompactionStore):
    db_path = str(background.db_path)
    messages = MessageStore(db_path)
    try:
        source_ids = messages.append_batch(
            "session",
            [{"role": "user", "content": "provider source " + "x " * 100}],
            token_estimates=[1_000],
            conversation_id="conversation",
        )
    finally:
        messages.close()
    SummaryDAG(db_path).close()
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session("session", conversation_id="conversation")
    finally:
        lifecycle.close()
    job = background.enqueue_or_coalesce(
        conversation_id="conversation",
        session_id="session",
        expected_frontier_store_id=0,
        captured_watermark_store_id=source_ids[-1],
        source_store_ids=source_ids,
        source_token_count=1_000,
        policy_fingerprint=background_policy_fingerprint(background.config),
        expected_node_count=1,
    )
    assert job is not None
    return job


def _provider_permit_count(background: BackgroundCompactionStore) -> int:
    return int(
        background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases"
        ).fetchone()[0]
    )


def _install_failing_attempt_telemetry(
    background: BackgroundCompactionStore,
    monkeypatch,
) -> list[str]:
    reasons: list[str] = []

    def fail_attempt_telemetry(*, fallback_reason="", **_kwargs):
        reasons.append(str(fallback_reason or ""))
        raise RuntimeError("authorization=secret-attempt-telemetry-value")

    monkeypatch.setattr(
        background,
        "_record_compaction_attempts",
        fail_attempt_telemetry,
    )
    return reasons


def test_attempt_telemetry_failure_after_provider_success_still_prepares_job(
    tmp_path,
    monkeypatch,
    caplog,
):
    db_path = tmp_path / "success-telemetry-failure.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    job = _seed_and_enqueue(background)
    reasons = _install_failing_attempt_telemetry(background, monkeypatch)
    caplog.set_level(logging.WARNING, logger=background_compaction_module.__name__)

    def successful_provider(*, trace, **_kwargs):
        trace["attempts"].append(
            {
                "actual_provider": "fixture",
                "actual_model": "model",
                "attribution_available": True,
                "outcome": "success",
                "summary_token_count": 4,
            }
        )
        return "small durable summary", 1

    try:
        result = background.run_next_job(
            owner="success-worker",
            summarizer=successful_provider,
        )

        assert result is not None and result.job_id == job.job_id
        assert result.state == "prepared"
        assert result.completion_reason == "prepared"
        assert reasons == [""]
        assert len(background.get_prepared_nodes(job.job_id)) == 1
        assert _provider_permit_count(background) == 0
        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert "attempt telemetry persistence failed" in logs
        assert "secret-attempt-telemetry-value" not in logs
    finally:
        background.close()


@pytest.mark.parametrize(
    ("provider", "expected_state", "expected_reason"),
    [
        (
            lambda **_kwargs: (_ for _ in ()).throw(
                ConnectionError("authorization=secret-provider-error")
            ),
            "retry",
            "provider_connection_error",
        ),
        (
            lambda **_kwargs: ("not publishable", 3),
            "failed",
            "non_compressing_output",
        ),
    ],
    ids=["provider-error", "noncompressing-output"],
)
def test_attempt_telemetry_failure_still_persists_authoritative_failure(
    tmp_path,
    monkeypatch,
    provider,
    expected_state,
    expected_reason,
):
    db_path = tmp_path / f"{expected_reason}-telemetry-failure.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    job = _seed_and_enqueue(background)
    reasons = _install_failing_attempt_telemetry(background, monkeypatch)
    try:
        result = background.run_next_job(
            owner=f"{expected_reason}-worker",
            summarizer=provider,
        )

        assert result is not None and result.job_id == job.job_id
        assert result.state == expected_state
        assert result.completion_reason == expected_reason
        assert reasons == [expected_reason]
        assert _provider_permit_count(background) == 0
    finally:
        background.close()


def test_attempt_telemetry_failure_after_timeout_still_persists_retry(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "timeout-telemetry-failure.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    job = _seed_and_enqueue(background)
    reasons = _install_failing_attempt_telemetry(background, monkeypatch)
    clock = _Clock()
    provider_returned = threading.Event()

    def late_provider(**_kwargs):
        clock.advance(2.0)
        provider_returned.set()
        return "late summary", 1

    monkeypatch.setattr(background_compaction_module.time, "monotonic", clock)
    try:
        result = background.run_next_job(
            owner="timeout-worker",
            summarizer=late_provider,
        )

        assert provider_returned.is_set()
        assert result is not None and result.job_id == job.job_id
        assert result.state == "retry"
        assert result.completion_reason == "provider_timeout"
        assert reasons == ["provider_timeout"]
        assert _provider_permit_count(background) == 0
    finally:
        background.close()


def test_outer_cleanup_releases_permit_when_post_provider_bookkeeping_raises(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "post-provider-bookkeeping-failure.db"
    background = BackgroundCompactionStore(db_path, config=_config(db_path))
    _seed_and_enqueue(background)

    def fail_token_count(_summary):
        raise RuntimeError("post-provider bookkeeping failed")

    monkeypatch.setattr(background_compaction_module, "count_tokens", fail_token_count)
    try:
        with pytest.raises(RuntimeError, match="post-provider bookkeeping failed"):
            background.run_next_job(
                owner="bookkeeping-worker",
                summarizer=lambda **_kwargs: ("valid provider summary", 1),
            )

        assert _provider_permit_count(background) == 0
    finally:
        background.close()
