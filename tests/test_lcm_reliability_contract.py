import hashlib
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
def test_guard_deferrals_do_not_consume_provider_attempts(
    tmp_path, reason, wait_seconds
):
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


def test_mixed_physical_failure_then_guard_skip_consumes_attempt(tmp_path):
    path = tmp_path / "mixed-physical-and-guard.db"
    clock = design.Clock()
    ids = design._seed(path)
    bg = BackgroundCompactionStore(path, config=design._config(path), now=clock)
    try:
        job = design._explicit_job(bg, ids)
        claim = bg.claim_next_job(owner="worker")
        result = bg.mark_retry(
            job.job_id,
            owner="worker",
            generation=claim.lease_generation,
            error=RuntimeError("later route was circuit blocked"),
            completion_reason="provider_circuit_open",
            provider_attempted=True,
        )
        assert result.state == "retry"
        assert result.attempt_count == 1
        clock.advance(result.next_retry_at - clock())
        assert bg.claim_next_job(owner="worker").attempt_count == 2
    finally:
        bg.close()


def test_transient_provider_failure_before_l3_fallback_remains_retryable(tmp_path):
    path = tmp_path / "transient-before-l3.db"
    design._seed(path)
    config = design._config(
        path,
        background_compaction_workers=0,
        leaf_chunk_tokens=1_000,
    )
    bg = BackgroundCompactionStore(path, config=config)

    def transient_then_l3(*, trace=None, **_kwargs):
        trace.update(
            {
                "terminal_reason": "non_compressing_output",
                "attempts": [
                    {
                        "outcome": "failure",
                        "failure_reason": "provider_timeout",
                        "actual_provider": "zai",
                        "actual_model": "glm-test",
                        "attribution_available": True,
                    },
                    {
                        "outcome": "rejected",
                        "failure_reason": "non_compressing_output",
                        "attribution_available": False,
                    },
                ],
            }
        )
        return ("lossless deterministic fallback", 3)

    try:
        job = bg.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=1,
            effective_context_window=372_000,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert job is not None
        result = bg.run_next_job(owner="worker", summarizer=transient_then_l3)
        assert result.state == "retry"
        assert result.completion_reason == "provider_timeout"
        assert result.attempt_count == 1
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
            "token_estimate": 2_000,
        }
        for index in range(12)
    ]
    try:
        chunks = bg._chunk_messages(messages)
        assert len(chunks) == 3
        assert max(chunk.source_token_count for chunk in chunks) <= 8_000
        config.cache_friendly_condensation_enabled = False
        dynamic = bg._chunk_messages(messages)
        assert len(dynamic) == 2
        assert dynamic[0].source_token_count == 16_000
    finally:
        bg.close()


def test_full_historical_sized_range_progresses_across_jobs_and_restart(tmp_path):
    path = tmp_path / "historical-sized-range.db"
    messages = [
        {"role": "user", "content": f"historical-row-{index}"} for index in range(57)
    ]
    source_ids = design._seed(path, messages=messages, tokens=[2_000] * len(messages))
    config = design._config(
        path,
        background_compaction_workers=0,
        fresh_tail_count=1,
        leaf_chunk_tokens=8_000,
        dynamic_leaf_chunk_enabled=True,
        dynamic_leaf_chunk_max=40_000,
        cache_friendly_condensation_enabled=True,
    )
    policy = background_policy_fingerprint(config)
    calls = []

    def summarize(**kwargs):
        calls.append(
            {
                "source_tokens": kwargs["source_tokens"],
                "depth": kwargs["depth"],
            }
        )
        return (f"summary-{len(calls)}", 0)

    bg = BackgroundCompactionStore(path, config=config)
    raw_before = list(
        bg.connection.execute(
            "SELECT store_id, role, content, tool_calls, tool_call_id, tool_name "
            "FROM messages ORDER BY store_id"
        )
    )
    raw_digest = hashlib.sha256(
        repr([tuple(row) for row in raw_before]).encode()
    ).hexdigest()
    frontier = 0
    jobs = []
    try:
        while True:
            job = bg.capture_and_enqueue(
                conversation_id="c",
                session_id="s",
                expected_frontier_store_id=frontier,
                active_tokens=900_000,
                effective_context_window=1_000_000,
                policy_fingerprint=policy,
            )
            if job is None:
                break
            prepared = bg.run_next_job(owner="offline-worker", summarizer=summarize)
            assert prepared is not None and prepared.state == "prepared"
            lease = bg.acquire_publication_lease(
                prepared.job_id, owner="offline-publisher"
            )
            assert lease is not None
            published = bg.publish_prepared(
                prepared.job_id,
                owner="offline-publisher",
                generation=lease.lease_generation,
                policy_fingerprint=policy,
                observed_active_tokens=900_000,
                effective_context_window=1_000_000,
                publication_trigger_tokens=900_000,
                emergency_trigger_tokens=950_000,
                trigger_policy="fixed_reserve",
            )
            assert published.promoted
            frontier = published.captured_watermark_store_id
            jobs.append(prepared.job_id)
            if len(jobs) == 1:
                bg.close()
                bg = BackgroundCompactionStore(path, config=config)
            assert len(jobs) < 10

        raw_after = list(
            bg.connection.execute(
                "SELECT store_id, role, content, tool_calls, tool_call_id, tool_name "
                "FROM messages ORDER BY store_id"
            )
        )
        after_digest = hashlib.sha256(
            repr([tuple(row) for row in raw_after]).encode()
        ).hexdigest()
        covered = [
            int(row[0])
            for row in bg.connection.execute(
                "SELECT store_id FROM lcm_compaction_source_coverage "
                "WHERE session_id='s' ORDER BY store_id"
            )
        ]
        assert raw_digest == after_digest
        assert covered == source_ids[:-1]
        assert source_ids[-1] not in covered
        assert frontier == source_ids[-2]
        assert len(jobs) == 5
        assert len(calls) == 19
        assert (
            max(call["source_tokens"] for call in calls if call["depth"] == 0) <= 8_000
        )
        assert sum(call["depth"] == 1 for call in calls) == len(jobs)
        assert bg.status(conversation_id="c")["failed_jobs"] == 0
    finally:
        bg.close()


def test_summary_host_receives_absolute_deadline(monkeypatch):
    observed = {}

    def host_call(*, deadline=None, **_kwargs):
        observed["deadline"] = deadline
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message", (), {"content": "bounded summary"}
                            )()
                        },
                    )()
                ]
            },
        )()

    monkeypatch.setattr("agent.auxiliary_client.call_llm", host_call)
    result = escalation._call_llm_for_summary(
        "fixture", 100, timeout=10.0, deadline=123.0, route_info={}, trace={}
    )
    assert result == "bounded summary"
    assert observed["deadline"] == 123.0


def test_host_physical_attempts_are_flattened_into_durable_trace(monkeypatch):
    receipt_events = []

    def host_summary(
        _prompt,
        _max_tokens,
        model="",
        timeout=None,
        route_info=None,
        trace=None,
        deadline=None,
    ):
        del model, timeout, deadline
        route_info.update(
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
        trace.update(outcome="success", failure_reason="")
        return "retained factual summary"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", host_summary)
    trace = {"receipt_sink": lambda event: receipt_events.append(dict(event))}
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
    assert [
        (item["actual_provider"], item["outcome"]) for item in trace["attempts"]
    ] == [
        ("zai", "failure"),
        ("openai-codex", "success"),
    ]
    assert [item["logical_route_ordinal"] for item in trace["attempts"]] == [0, 1]
    # The per-route bridge preserves the durable sink while adding stable
    # logical and estimate metadata to each physical event.
    route_info = {}

    def sink_probe(
        _prompt, _max_tokens, *, route_info=None, trace=None, **_kwargs
    ):
        del trace
        info = route_info
        info["receipt_sink"](
            {
                "physical_attempt_id": "probe:0",
                "send_state": "started",
                "provider": "zai",
                "model": "glm",
            }
        )
        info.update(provider="zai", model="glm", physical_attempts=[])
        return "short"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", sink_probe)
    probe_trace = {"receipt_sink": lambda event: route_info.update(event)}
    escalation.summarize_with_escalation(
        "source " * 200,
        source_tokens=500,
        token_budget=100,
        model="glm",
        timeout=60.0,
        deadline=escalation.time.monotonic() + 60.0,
        trace=probe_trace,
    )
    assert route_info["physical_attempt_id"] == "probe:0"
    assert route_info["escalation_level"] == 1
    assert route_info["logical_route_ordinal"] == 0
    assert route_info["requested_model"] == "glm"
    assert route_info["source_input_estimate"] == 500
    assert route_info["framed_input_estimate"] > 0
    assert route_info["estimate_method"] == "lcm-tokenizer-v1"


def test_reference_publication_is_reported_separately_from_summary_success(tmp_path):
    path = tmp_path / "reference-publication.db"
    source_ids = terminal._seed(path, 13)
    clock = terminal.Clock()
    config = terminal._config(path)
    bg = BackgroundCompactionStore(path, config=config, now=clock)
    try:
        assert terminal._capture(bg, config) is not None
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
