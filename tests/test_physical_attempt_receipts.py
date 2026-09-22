import importlib.util
from pathlib import Path

from hermes_lcm.background_compaction import BackgroundCompactionStore


def _module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


design = _module("test_async_background_compaction_design.py")


def test_physical_receipt_start_is_durable_idempotent_and_later_settles(tmp_path):
    path = tmp_path / "physical-receipts.db"
    ids = design._seed(path)
    background = BackgroundCompactionStore(path, config=design._config(path))
    try:
        job = design._explicit_job(background, ids)
        claimed = background.claim_next_job(owner="receipt-worker")
        assert claimed is not None and claimed.job_id == job.job_id
        started = {
            "physical_attempt_id": "request-1:0",
            "send_state": "started",
            "outcome": "started",
            "provider": "zai",
            "model": "glm-fixture",
            "effective_timeout_ms": 45_000,
            "timeout_source": "config_yaml:auxiliary.compression.timeout",
            "escalation_level": 1,
            "logical_route_ordinal": 0,
            "source_input_estimate": 900,
            "framed_input_estimate": 1_050,
            "estimate_method": "lcm-tokenizer-v1",
            "started_at": 1_000.0,
        }
        background._record_physical_attempt_receipt(
            job=claimed,
            chunk_ordinal=0,
            event=started,
            serialized_input_estimate=920,
        )
        # A repeated delivery of the start event does not count another send.
        background._record_physical_attempt_receipt(
            job=claimed,
            chunk_ordinal=0,
            event=started,
            serialized_input_estimate=920,
        )
        row = background.connection.execute(
            """
            SELECT send_state, transport_outcome, finished_at,
                   provider_input_total, cache_read_input, usage_coverage,
                   source_input_estimate, serialized_input_estimate,
                   framed_input_estimate
            FROM lcm_compaction_attempt_receipts
            WHERE physical_attempt_id=?
            """,
            ("request-1:0",),
        ).fetchone()
        assert tuple(row) == (
            "started", "started", None, None, None, "unavailable", 900, 920, 1_050
        )
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_attempt_receipts"
        ).fetchone()[0] == 1

        settled = dict(started)
        settled.update(
            {
                "send_state": "settled",
                "outcome": "success",
                "served_model": "glm-served",
                "provider_input_total": 1_100,
                "provider_output_total": 90,
                "cache_read_input": 200,
                "cache_write_input": None,
                "reasoning_output": 15,
                "normalized_input": 900,
                "normalized_output": 90,
                "usage_convention": "input_includes_cache_subsets",
                "usage_normalization_version": "hermes-canonical-v1",
                "usage_coverage": "reported",
                "finish_reason": "stop",
                "latency_ms": 12_345,
                "finished_at": 1_012.345,
            }
        )
        background._record_physical_attempt_receipt(
            job=claimed,
            chunk_ordinal=0,
            event=settled,
            serialized_input_estimate=920,
        )
        # Out-of-order duplicate start delivery cannot downgrade a settlement.
        background._record_physical_attempt_receipt(
            job=claimed,
            chunk_ordinal=0,
            event=started,
            serialized_input_estimate=920,
        )
        row = background.connection.execute(
            """
            SELECT send_state, transport_outcome, served_model,
                   provider_input_total, provider_output_total,
                   cache_read_input, cache_write_input, reasoning_output,
                   normalized_input, normalized_output, usage_coverage,
                   summary_outcome
            FROM lcm_compaction_attempt_receipts
            WHERE physical_attempt_id=?
            """,
            ("request-1:0",),
        ).fetchone()
        assert tuple(row) == (
            "settled", "success", "glm-served", 1_100, 90, 200, None, 15,
            900, 90, "reported", "pending"
        )

        background._record_compaction_attempts(
            job=claimed,
            chunk_ordinal=0,
            source_token_count=900,
            trace={
                "attempts": [
                    {
                        "physical_attempt_id": "request-1:0",
                        "escalation_level": 1,
                        "logical_route_ordinal": 0,
                        "outcome": "rejected",
                        "failure_reason": "non_compressing_output",
                    }
                ]
            },
        )
        summary = background.connection.execute(
            """
            SELECT transport_outcome, summary_outcome, summary_reason
            FROM lcm_compaction_attempt_receipts
            WHERE physical_attempt_id=?
            """,
            ("request-1:0",),
        ).fetchone()
        assert tuple(summary) == (
            "success", "rejected", "non_compressing_output"
        )
    finally:
        background.close()


def test_physical_receipt_started_row_survives_reopen_as_unknown_settlement(tmp_path):
    path = tmp_path / "physical-start-only.db"
    ids = design._seed(path)
    config = design._config(path)
    background = BackgroundCompactionStore(path, config=config)
    job = design._explicit_job(background, ids)
    claimed = background.claim_next_job(owner="receipt-worker")
    background._record_physical_attempt_receipt(
        job=claimed,
        chunk_ordinal=0,
        event={
            "physical_attempt_id": "request-crash:0",
            "send_state": "started",
            "outcome": "started",
            "provider": "openai-codex",
            "model": "luna",
            "started_at": 1_001.0,
        },
        serialized_input_estimate=500,
    )
    background.close()

    reopened = BackgroundCompactionStore(path, config=config)
    try:
        row = reopened.connection.execute(
            """
            SELECT send_state, transport_outcome, finished_at, usage_coverage
            FROM lcm_compaction_attempt_receipts
            WHERE physical_attempt_id=?
            """,
            ("request-crash:0",),
        ).fetchone()
        assert tuple(row) == ("started", "started", None, "unavailable")
    finally:
        reopened.close()
