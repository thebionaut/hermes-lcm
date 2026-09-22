"""Operator-facing contracts for durable background-compaction diagnostics."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from hermes_lcm.config import LCMConfig
from hermes_lcm.command import _background_compaction_status_lines
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tools import (
    _BACKGROUND_DIAGNOSTIC_REASONS,
    _background_compaction_status_payload,
    lcm_doctor,
    lcm_inspect,
    lcm_status,
)


class _StatusEngine:
    def __init__(self, status: dict):
        self._status = status
        self.calls = 0
        self._config = SimpleNamespace(
            background_compaction_enabled=True,
            background_compaction_workers=1,
            background_compaction_global_limit=2,
        )

    def get_background_compaction_status(self) -> dict:
        self.calls += 1
        return dict(self._status)


def _complete_status() -> dict:
    event_counts = {
        reason: index
        for index, reason in enumerate(sorted(_BACKGROUND_DIAGNOSTIC_REASONS), 1)
    }
    return {
        "configured_enabled": True,
        "enabled": False,
        "compatibility_blocked_reason": (
            "assertion_extraction_requires_foreground_compaction"
        ),
        "worker_count": 0,
        "global_limit": 2,
        "worker_available": False,
        "queued_jobs": 11,
        "running_jobs": 12,
        "prepared_jobs": 13,
        "retry_jobs": 14,
        "published_jobs": 15,
        "superseded_jobs": 16,
        "failed_jobs": 17,
        "active_provider_permits": 3,
        "active_non_timed_out_provider_permits": 1,
        "timed_out_provider_permits": 2,
        "provider_permit_oldest_age_seconds": 12.3456,
        "provider_permit_min_lease_remaining_seconds": 8.7654,
        "oldest_queued_job_age_seconds": 4.5,
        "current_job": None,
        "publication_fence": {
            "generation": 7,
            "publishable": False,
            "updated_at": 123.4567,
        },
        "current_debt_tokens": 32_123,
        "last_preparation_result": None,
        "last_publication_result": None,
        "last_publication_telemetry": {
            "job_id": "private-job-id",
            "range_identity": "private-source-identity",
            "publication_fence_generation": 7,
            "observed_active_tokens": 900_321,
            "effective_context_window": 1_000_000,
            "publication_trigger_tokens": 900_000,
            "emergency_trigger_tokens": 950_000,
            "trigger_policy": "fixed_reserve",
            "prepared_to_published_seconds": 12.3456,
            "source_token_count": 32_000,
            "summary_token_count": 400,
            "provider": "openai-codex",
            "model": "gpt-5.6-luna",
            "published_at": 124.0,
        },
        "last_provider_attempt": {
            "job_id": "private-job-id",
            "attempt_number": 2,
            "chunk_ordinal": 1,
            "escalation_level": 1,
            "logical_route_ordinal": 1,
            "provider": "openai-codex",
            "model": "gpt-5.6-luna",
            "attribution_available": True,
            "fallback_used": True,
            "outcome": "success",
            "failure_reason": "",
            "source_token_count": 32_000,
            "summary_token_count": 400,
            "latency_ms": 1_234,
            "finished_at": 123.0,
        },
        "recent_provider_outcomes": {
            "success": 2,
            "provider_timeout": 1,
            "not_a_stable_reason": 999,
        },
        "provider_attribution": {
            "prepared_nodes": 4,
            "attributed_nodes": 3,
        },
        "range_dispositions": {
            "insufficient_compressible_source": 2,
            "not_a_stable_reason": 999,
        },
        "durable_event_counts": {
            **event_counts,
            "not_a_stable_reason": 999,
        },
        "durable_event_reasons": [
            *reversed(tuple(event_counts)),
            "not_a_stable_reason",
        ],
        "backlog_trigger_tokens": 32_000,
        "minimum_provider_source_tokens": 256,
        "prepare_ratio": 0.61,
        "publication_trigger_tokens": 90_000,
        "emergency_trigger_tokens": 100_000,
        "durable_frontier_store_id": 456,
        "active_coverage": "raw",
        "diagnostic_reasons": [
            "provider_timeout",
            "not_a_stable_reason",
        ],
    }


def test_json_status_preserves_scoped_durable_diagnostics_without_inference():
    raw = _complete_status()
    engine = _StatusEngine(raw)

    payload = _background_compaction_status_payload(engine)

    assert engine.calls == 1
    assert payload["configured_enabled"] is True
    assert payload["enabled"] is False
    assert payload["compatibility_blocked_reason"] == (
        "assertion_extraction_requires_foreground_compaction"
    )
    assert payload["worker_available"] is False
    assert payload["active_provider_permits"] == 3
    assert payload["active_non_timed_out_provider_permits"] == 1
    assert payload["timed_out_provider_permits"] == 2
    assert payload["provider_permit_oldest_age_seconds"] == 12.346
    assert payload["provider_permit_min_lease_remaining_seconds"] == 8.765
    assert payload["current_debt_tokens"] == 32_123
    assert payload["publication_fence"] == {
        "generation": 7,
        "publishable": False,
        "updated_at": 123.457,
    }
    assert payload["backlog_trigger_tokens"] == 32_000
    assert payload["minimum_provider_source_tokens"] == 256
    assert payload["prepare_ratio"] == 0.61
    assert payload["publication_trigger_tokens"] == 90_000
    assert payload["emergency_trigger_tokens"] == 100_000
    assert payload["durable_frontier_store_id"] == 456
    assert payload["active_coverage"] == "raw"
    publication = payload["last_publication_telemetry"]
    assert publication["job_hint"] != "private-job-id"
    assert publication["range_identity_hint"] != "private-source-identity"
    assert publication["observed_active_tokens"] == 900_321
    assert publication["publication_trigger_tokens"] == 900_000
    assert publication["trigger_policy"] == "fixed_reserve"
    assert publication["prepared_to_published_seconds"] == 12.346
    attempt = payload["last_provider_attempt"]
    assert attempt["job_hint"] == publication["job_hint"]
    assert attempt["provider"] == "openai-codex"
    assert attempt["model"] == "gpt-5.6-luna"
    assert attempt["fallback_used"] is True
    assert payload["recent_provider_outcomes"] == {
        "success": 2,
        "provider_timeout": 1,
    }
    assert payload["provider_attribution"] == {
        "prepared_nodes": 4,
        "attributed_nodes": 3,
        "coverage_ratio": 0.75,
    }
    assert payload["range_dispositions"] == {
        "insufficient_compressible_source": 2,
    }
    serialized = json.dumps(payload)
    assert "private-job-id" not in serialized
    assert "private-source-identity" not in serialized

    expected_counts = {
        reason: raw["durable_event_counts"][reason]
        for reason in sorted(_BACKGROUND_DIAGNOSTIC_REASONS)
    }
    assert payload["durable_event_counts"] == expected_counts
    assert payload["durable_event_reasons"] == sorted(
        _BACKGROUND_DIAGNOSTIC_REASONS
    )
    assert "not_a_stable_reason" not in payload["durable_event_counts"]
    assert "not_a_stable_reason" not in payload["durable_event_reasons"]
    # Durable counts are the already-scoped event query. Global state totals
    # must never be reinterpreted as per-conversation durable event counts.
    assert payload["durable_event_counts"]["prepared_summary_published"] != (
        payload["published_jobs"]
    )


def test_text_status_surfaces_policy_frontier_debt_permits_and_exact_events():
    engine = _StatusEngine(_complete_status())

    text = "\n".join(_background_compaction_status_lines(engine))

    assert (
        "background_compaction_enabled: configured=yes effective=no "
        "blocker=assertion_extraction_requires_foreground_compaction"
    ) in text
    assert "background_compaction_limits: workers=0 global=2 available=no" in text
    assert (
        "background_compaction_jobs: queued=11 running=12 prepared=13 retry=14 "
        "published=15 superseded=16 failed=17"
    ) in text
    assert (
        "background_compaction_provider_permits: active=3 non_timed_out=1 "
        "timed_out=2 oldest_age_seconds=12.346 "
        "min_lease_remaining_seconds=8.765"
    ) in text
    assert (
        "background_compaction_policy: backlog=32000 "
        "min_provider_source=256 prepare_ratio=0.61"
    ) in text
    assert "background_compaction_triggers: publication=90000 emergency=100000" in text
    assert (
        "background_compaction_frontier: durable=456 debt=32123 coverage=raw"
    ) in text
    assert (
        "background_compaction_publication_fence: generation=7 "
        "publishable=no updated_at=123.457"
    ) in text
    assert "background_compaction_durable_event_counts:" in text
    assert "background_compaction_publication_telemetry:" in text
    assert "background_compaction_last_provider_attempt:" in text
    assert "background_compaction_recent_provider_outcomes_24h:" in text
    assert "background_compaction_provider_attribution:" in text
    assert "background_compaction_range_dispositions:" in text
    for reason in _BACKGROUND_DIAGNOSTIC_REASONS:
        assert f'"{reason}":' in text
    assert "not_a_stable_reason" not in text


def test_status_failure_keeps_configured_and_effective_state_distinct():
    class _BrokenEngine(_StatusEngine):
        def get_background_compaction_status(self) -> dict:
            raise RuntimeError("token=top-secret")

    payload = _background_compaction_status_payload(_BrokenEngine({}))

    assert payload["configured_enabled"] is True
    assert payload["enabled"] is False
    assert payload["worker_available"] is False
    assert payload["active_provider_permits"] == 0
    assert payload["timed_out_provider_permits"] == 0
    assert payload["durable_event_counts"] == {}
    assert payload["durable_event_reasons"] == []
    assert "top-secret" not in payload["status_error"]


def test_doctor_warns_when_background_status_collection_fails(
    tmp_path,
    monkeypatch,
):
    config = LCMConfig(
        database_path=str(tmp_path / "broken-status-doctor.db"),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))
    engine.on_session_start("scope-a", platform="cli", context_length=200_000)
    try:
        def broken_status() -> dict:
            raise RuntimeError("token=top-secret")

        monkeypatch.setattr(engine, "get_background_compaction_status", broken_status)

        check = _doctor_background_check(engine)

        assert check["detail"]["configured_enabled"] is True
        assert check["detail"]["enabled"] is False
        assert "top-secret" not in check["detail"]["status_error"]
        assert check["status"] == "warn"
    finally:
        engine.shutdown()


def _doctor_background_check(engine: LCMEngine) -> dict:
    doctor = json.loads(lcm_doctor({}, engine=engine))
    return next(
        check
        for check in doctor["checks"]
        if check["check"] == "background_compaction"
    )


def test_doctor_warns_when_enabled_background_worker_is_unavailable(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "worker-unavailable-doctor.db"),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))
    engine.on_session_start("scope-a", platform="cli", context_length=200_000)
    try:
        check = _doctor_background_check(engine)

        assert check["detail"]["enabled"] is True
        assert check["detail"]["worker_available"] is False
        assert "worker_unavailable" in check["detail"]["diagnostic_reasons"]
        assert check["status"] == "warn"
    finally:
        engine.shutdown()


def test_doctor_warns_while_timed_out_provider_permit_is_active(
    tmp_path,
    monkeypatch,
):
    config = LCMConfig(
        database_path=str(tmp_path / "timed-out-permit-doctor.db"),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_lease_seconds=30,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))
    engine.on_session_start("scope-a", platform="cli", context_length=200_000)
    background = engine._background_compaction
    assert background is not None
    try:
        generation = background.get_publishable_session_fence_generation(
            conversation_id=engine.current_conversation_id,
            session_id="scope-a",
        )
        assert generation is not None
        permit = background.acquire_emergency_provider_permit(
            conversation_id=engine.current_conversation_id,
            session_id="scope-a",
            expected_session_fence_generation=generation,
            owner="timed-out-doctor-test",
            lease_seconds=30,
        )
        assert permit is not None
        background.connection.execute(
            """
            UPDATE lcm_compaction_provider_leases
            SET timed_out_at=?
            WHERE job_id=?
            """,
            (time.time(), permit.job_id),
        )
        real_status = engine.get_background_compaction_status

        def available_worker_status() -> dict:
            status = real_status()
            status["worker_available"] = True
            status["worker_count"] = 1
            return status

        monkeypatch.setattr(
            engine,
            "get_background_compaction_status",
            available_worker_status,
        )

        check = _doctor_background_check(engine)

        assert check["detail"]["enabled"] is True
        assert check["detail"]["worker_available"] is True
        assert check["detail"]["timed_out_provider_permits"] == 1
        assert check["detail"]["retry_jobs"] == 0
        assert check["detail"]["recent_provider_outcomes"] == {}
        assert check["status"] == "warn"
    finally:
        engine.shutdown()


def test_lcm_status_keeps_durable_events_scoped_to_active_conversation(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "scoped-status.db"),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))
    engine.on_session_start("scope-a", platform="cli", context_length=200_000)
    active_conversation = engine.current_conversation_id
    store = engine._background_compaction
    assert store is not None
    try:
        store.record_diagnostic_event(
            "provider_timeout",
            conversation_id=active_conversation,
            session_id="scope-a",
        )
        store.record_diagnostic_event(
            "sqlite_lock_contention",
            conversation_id="scope-b",
            session_id="scope-b",
        )
        store.record_diagnostic_event(
            "sqlite_lock_contention",
            conversation_id="scope-b",
            session_id="scope-b",
        )

        status = json.loads(lcm_status({}, engine=engine))["background_compaction"]
        inspect = json.loads(lcm_inspect({"limit": 1}, engine=engine))[
            "background_compaction"
        ]
        doctor = json.loads(lcm_doctor({}, engine=engine))
        doctor_status = next(
            check["detail"]
            for check in doctor["checks"]
            if check["check"] == "background_compaction"
        )

        for surface in (status, inspect, doctor_status):
            assert surface["durable_event_counts"]["provider_timeout"] == 1
            assert "provider_timeout" in surface["durable_event_reasons"]
            assert "sqlite_lock_contention" not in surface["durable_event_counts"]
            assert "sqlite_lock_contention" not in surface["durable_event_reasons"]
    finally:
        engine.shutdown()
