import importlib.util
from pathlib import Path

import pytest

from hermes_lcm import escalation
from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)


def _module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


design = _module("test_async_background_compaction_design.py")
terminal = _module("test_background_compaction_terminal_disposition_progress.py")


@pytest.mark.parametrize(
    "reason,wait_seconds",
    [("provider_circuit_open", 300), ("provider_spend_guard", 1800)],
)
def test_guard_deferrals_do_not_consume_provider_attempts(tmp_path, reason, wait_seconds):
    path = tmp_path / f"{reason}.db"
    clock = design.Clock()
    ids = design._seed(path)
    bg = BackgroundCompactionStore(path, config=design._config(path), now=clock)
    try:
        job = design._explicit_job(bg, ids)
        for _ in range(7):
            claim = bg.claim_next_job(owner="worker")
            assert claim is not None and claim.attempt_count == 1
            deferred = bg.mark_retry(
                job.job_id,
                owner="worker",
                generation=claim.lease_generation,
                error=RuntimeError("guard closed"),
                completion_reason=reason,
            )
            assert deferred.state == "retry"
            assert deferred.attempt_count == 0
            assert deferred.next_retry_at == clock() + wait_seconds
            clock.advance(wait_seconds)
    finally:
        bg.close()


def test_real_provider_failures_retain_finite_attempt_limit(tmp_path):
    path = tmp_path / "provider-timeout.db"
    clock = design.Clock()
    ids = design._seed(path)
    bg = BackgroundCompactionStore(path, config=design._config(path), now=clock)
    try:
        job = design._explicit_job(bg, ids)
        result = None
        for expected in range(1, 6):
            claim = bg.claim_next_job(owner="worker")
            assert claim is not None and claim.attempt_count == expected
            result = bg.mark_retry(
                job.job_id,
                owner="worker",
                generation=claim.lease_generation,
                error=TimeoutError("timed out"),
                completion_reason="provider_timeout",
            )
            if result.next_retry_at:
                clock.advance(result.next_retry_at - clock())
        assert result is not None
        assert result.state == "failed"
        assert result.attempt_count == 5
    finally:
        bg.close()


def test_cache_friendly_background_chunks_stay_at_configured_leaf_size(tmp_path):
    path = tmp_path / "chunks.db"
    config = design._config(path)
    config.leaf_chunk_tokens = 8_000
    config.dynamic_leaf_chunk_enabled = True
    config.dynamic_leaf_chunk_max = 40_000
    config.cache_friendly_condensation_enabled = True
    bg = BackgroundCompactionStore(path, config=config)
    messages = [
        {
            "store_id": index + 1,
            "role": "user",
            "content": f"message-{index}",
            "token_count": 2_000,
        }
        for index in range(12)
    ]
    try:
        chunks = bg._chunk_messages(messages)
        assert len(chunks) == 3
        assert max(chunk.source_token_count for chunk in chunks) <= 8_000
        config.cache_friendly_condensation_enabled = False
        dynamic = bg._chunk_messages(messages)
        assert len(dynamic) == 1
        assert dynamic[0].source_token_count == 24_000
    finally:
        bg.close()


def test_summary_host_receives_absolute_deadline(monkeypatch):
    observed = {}

    def host_call(*, deadline=None, **_kwargs):
        observed["deadline"] = deadline
        return type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "bounded summary"})()})()]},
        )()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", host_call)
    result = escalation._call_llm_for_summary(
        "fixture", 100, timeout=10.0, deadline=123.0, route_info={}, trace={}
    )
    assert result == "bounded summary"
    assert observed["deadline"] == 123.0


def test_host_physical_attempts_are_flattened_into_durable_trace(monkeypatch):
    def host_summary(_prompt, _max_tokens, **kwargs):
        kwargs["route_info"].update(
            {
                "provider": "openai-codex",
                "model": "luna",
                "physical_attempts": [
                    {
                        "ordinal": 0,
                        "provider": "zai",
                        "model": "glm",
                        "outcome": "failure",
                        "failure_reason": "provider_timeout",
                        "started_at": 10.0,
                        "finished_at": 40.0,
                        "latency_ms": 30_000,
                    },
                    {
                        "ordinal": 1,
                        "provider": "openai-codex",
                        "model": "luna",
                        "outcome": "success",
                        "failure_reason": "",
                        "started_at": 40.0,
                        "finished_at": 50.0,
                        "latency_ms": 10_000,
                    },
                ],
            }
        )
        kwargs["trace"].update(outcome="success", failure_reason="")
        return "retained factual summary"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", host_summary)
    trace = {}
    result = escalation.summarize_with_escalation(
        "source " * 200,
        source_tokens=500,
        token_budget=100,
        model="",
        timeout=60.0,
        deadline=escalation.time.monotonic() + 60.0,
        trace=trace,
    )
    assert result[0] == "retained factual summary"
    assert [(item["actual_provider"], item["outcome"]) for item in trace["attempts"]] == [
        ("zai", "failure"),
        ("openai-codex", "success"),
    ]
    assert [item["logical_route_ordinal"] for item in trace["attempts"]] == [0, 1]


def test_reference_publication_is_reported_separately_from_summary_success(tmp_path):
    path = tmp_path / "reference-publication.db"
    source_ids = terminal._seed(path, 13)
    clock = terminal.Clock()
    config = terminal._config(path)
    bg = BackgroundCompactionStore(path, config=config, now=clock)
    try:
        initial = terminal._capture(bg, config)
        exact_prefix = tuple(source_ids[: config.background_compaction_max_passes])
        failed = bg.run_next_job(
            owner="terminal-worker",
            summarizer=lambda **_kwargs: ("still not compressing", 3),
        )
        assert failed.state == "failed"
        terminal._append_new(path, ingested_at=clock() + 1)
        progressed = terminal._capture(bg, config)
        assert progressed.job_kind == "lossless_raw_projection"
        lease = bg.acquire_publication_lease(progressed.job_id, owner="publisher")
        published = bg.publish_prepared(
            progressed.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert published.promoted
        job = bg.get_job(progressed.job_id)
        assert job.completion_reason == "degraded_reference_published"
        status = bg.status()
        assert status["published_summary_jobs"] == 0
        assert status["published_reference_jobs"] == 1
        assert status["published_jobs"] == 1
        assert exact_prefix[-1] == published.captured_watermark_store_id
    finally:
        bg.close()
