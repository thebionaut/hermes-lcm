import importlib.util
from pathlib import Path

import pytest

from hermes_lcm.background_compaction import BackgroundCompactionStore
from hermes_lcm.escalation import SummaryCircuitBreaker

spec = importlib.util.spec_from_file_location(
    "bg_fixture",
    str(Path(__file__).with_name("test_async_background_compaction_design.py")),
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.mark.parametrize(
    "reason,minimum",
    [
        ("provider_circuit_open", 300),
        ("provider_spend_guard", 1800),
        ("provider_timeout", 5),
    ],
)
def test_retry_waits_for_relevant_guard(tmp_path, reason, minimum):
    path = tmp_path / "lcm.db"
    clock = fixture.Clock()
    ids = fixture._seed(path)
    config = fixture._config(path)
    bg = BackgroundCompactionStore(path, config=config, now=clock)
    try:
        job = fixture._explicit_job(bg, ids)
        claim = bg.claim_next_job(owner="worker")
        result = bg.mark_retry(
            job.job_id,
            owner="worker",
            generation=claim.lease_generation,
            error=RuntimeError("blocked"),
            completion_reason=reason,
        )
        assert result.state == "retry"
        assert result.next_retry_at == clock() + minimum
        clock.advance(minimum - 0.01)
        assert bg.claim_next_job(owner="worker") is None
        clock.advance(0.01)
        assert bg.claim_next_job(owner="worker") is not None
    finally:
        bg.close()


def test_retry_has_a_real_opportunity_after_circuit_reopens(tmp_path):
    path = tmp_path / "lcm.db"
    clock = fixture.Clock()
    ids = fixture._seed(path)
    bg = BackgroundCompactionStore(path, config=fixture._config(path), now=clock)
    breaker = SummaryCircuitBreaker()
    breaker.record_failure("", now=clock())
    breaker.record_failure("", now=clock())
    try:
        job = fixture._explicit_job(bg, ids)
        claim = bg.claim_next_job(owner="worker")
        result = bg.mark_retry(
            job.job_id,
            owner="worker",
            generation=claim.lease_generation,
            error=RuntimeError("blocked"),
            completion_reason="provider_circuit_open",
        )
        clock.advance(result.next_retry_at - clock())
        assert breaker.allows("", now=clock())
        # Guard deferral did not launch a provider call, so the recovered claim
        # is still provider attempt one rather than an exhausted second try.
        assert bg.claim_next_job(owner="worker").attempt_count == 1
    finally:
        bg.close()
