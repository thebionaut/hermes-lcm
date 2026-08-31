"""Focused cache-friendly contracts for asynchronous compaction."""

from __future__ import annotations

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def test_cache_friendly_preparation_combines_leaves_within_pass_budget(tmp_path):
    """Production cache-friendly mode publishes one stable prepared prefix."""
    db_path = tmp_path / "cache-friendly-background.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=1,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=120,
        background_compaction_lease_seconds=300,
        background_compaction_backlog_tokens=1,
        background_compaction_prepare_ratio=0.55,
        cache_friendly_condensation_enabled=True,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=20,
        summary_model="test/provider-model",
    )
    messages = [
        {"role": "user", "content": f"raw-{index}"}
        for index in range(7)
    ]
    store = MessageStore(db_path)
    store.append_batch(
        "s",
        messages,
        token_estimates=[10] * len(messages),
        conversation_id="c",
    )
    store.close()
    SummaryDAG(db_path).close()
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session("s", conversation_id="c")
    lifecycle.close()

    background = BackgroundCompactionStore(db_path, config=config)
    calls: list[dict] = []

    def summarize(**kwargs):
        calls.append(kwargs)
        if kwargs["depth"] == 1:
            return "combined prepared summary", 1
        return f"leaf prepared summary {len(calls)}", 0

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
        prepared = background.run_next_job(owner="worker", summarizer=summarize)
        assert prepared is not None and prepared.state == "prepared"
        nodes = background.get_prepared_nodes(job.job_id)
        assert [call["depth"] for call in calls] == [0, 0, 0, 1]
        assert len(calls) <= config.background_compaction_max_passes
        assert len(nodes) == 1
        assert nodes[0].summary == "combined prepared summary"
        assert nodes[0].source_store_ids == job.source_store_ids
        assert prepared.expected_node_count == 1
    finally:
        background.close()


def test_cache_friendly_retry_reuses_durable_leaves_and_only_retries_aggregation(tmp_path):
    """A failed final pass resumes from persisted leaf progress after restart."""
    db_path = tmp_path / "cache-friendly-retry.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=1,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=120,
        background_compaction_lease_seconds=300,
        background_compaction_backlog_tokens=1,
        cache_friendly_condensation_enabled=True,
        fresh_tail_count=1,
        leaf_chunk_tokens=20,
        summary_model="test/provider-model",
    )
    store = MessageStore(db_path)
    store.append_batch(
        "s",
        [{"role": "user", "content": f"raw-{index}"} for index in range(7)],
        token_estimates=[10] * 7,
        conversation_id="c",
    )
    store.close()
    SummaryDAG(db_path).close()
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session("s", conversation_id="c")
    lifecycle.close()

    now = [1_000.0]
    first = BackgroundCompactionStore(db_path, config=config, now=lambda: now[0])
    first_calls: list[int] = []

    def fail_aggregation(**kwargs):
        first_calls.append(kwargs["depth"])
        if kwargs["depth"] == 1:
            raise RuntimeError("aggregation unavailable")
        return f"leaf-{len(first_calls)}", 0

    try:
        job = first.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert job is not None
        retry = first.run_next_job(owner="first-worker", summarizer=fail_aggregation)
        assert retry is not None and retry.state == "retry"
        assert first_calls == [0, 0, 0, 1]
        assert [node.ordinal for node in first.get_prepared_nodes(job.job_id)] == [0, 1, 2]
    finally:
        first.close()

    now[0] = 1_006.0
    restarted = BackgroundCompactionStore(db_path, config=config, now=lambda: now[0])
    retry_calls: list[int] = []
    try:
        prepared = restarted.run_next_job(
            owner="restart-worker",
            summarizer=lambda **kwargs: (
                retry_calls.append(kwargs["depth"]) or "combined-after-retry",
                1,
            ),
        )
        assert prepared is not None and prepared.state == "prepared"
        assert retry_calls == [1]
        nodes = restarted.get_prepared_nodes(job.job_id)
        assert len(nodes) == 1
        assert nodes[0].summary == "combined-after-retry"
        assert nodes[0].source_store_ids == job.source_store_ids
    finally:
        restarted.close()
