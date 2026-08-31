"""Operator-facing contracts for durable background-compaction diagnostics."""

from __future__ import annotations

import json
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
        "durable_event_counts": {
            **event_counts,
            "not_a_stable_reason": 999,
        },
        "durable_event_reasons": [
            *reversed(tuple(event_counts)),
            "not_a_stable_reason",
        ],
        "backlog_trigger_tokens": 32_000,
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
    assert payload["prepare_ratio"] == 0.61
    assert payload["publication_trigger_tokens"] == 90_000
    assert payload["emergency_trigger_tokens"] == 100_000
    assert payload["durable_frontier_store_id"] == 456
    assert payload["active_coverage"] == "raw"

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
    assert "background_compaction_policy: backlog=32000 prepare_ratio=0.61" in text
    assert "background_compaction_triggers: publication=90000 emergency=100000" in text
    assert (
        "background_compaction_frontier: durable=456 debt=32123 coverage=raw"
    ) in text
    assert (
        "background_compaction_publication_fence: generation=7 "
        "publishable=no updated_at=123.457"
    ) in text
    assert "background_compaction_durable_event_counts:" in text
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
