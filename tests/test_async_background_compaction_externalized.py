"""Privacy and lineage contracts for externalized background source rows.

The durable SQLite marker row is the captured source of truth.  Background
capture, provider preparation, and publication never hydrate or hash external
payload bytes, so those filesystem resources cannot enter a provider request
or lengthen a foreground writer transaction.  Exact expansion remains anchored
to the original marker row and the externalization subsystem owns its payload
lifecycle independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.externalize import (
    externalize_ingest_payload,
    maybe_externalize_tool_output,
)
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


SESSION_ID = "externalized-session"
CONVERSATION_ID = "externalized-conversation"


def _config(tmp_path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=2,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=10,
        background_compaction_lease_seconds=30,
        background_compaction_backlog_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=0,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=100_000,
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key", "password_assignment"],
        large_output_externalization_enabled=True,
        # Keep the normal threshold below the fixtures.  Once durable content
        # has been hydrated for a summary provider it must proceed through the
        # canonical redaction/truncation path, not be replaced by another ref.
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(tmp_path / "externalized"),
        summary_model="test-provider/test-summary-model",
    )


def _initialize_source_rows(
    config: LCMConfig,
    messages: list[dict],
    *,
    token_estimates: list[int] | None = None,
) -> list[int]:
    db_path = config.database_path
    store = MessageStore(db_path)
    try:
        ids = store.append_batch(
            SESSION_ID,
            messages,
            token_estimates=token_estimates or [100] * len(messages),
            conversation_id=CONVERSATION_ID,
        )
    finally:
        store.close()

    # Initialize the two schemas publication relies on and bind the expected
    # zero frontier for this logical conversation.
    dag = SummaryDAG(db_path)
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session(SESSION_ID, conversation_id=CONVERSATION_ID)
    finally:
        lifecycle.close()
    return ids


def _capture(store: BackgroundCompactionStore, config: LCMConfig):
    policy = background_policy_fingerprint(config)
    job = store.capture_and_enqueue(
        conversation_id=CONVERSATION_ID,
        session_id=SESSION_ID,
        expected_frontier_store_id=0,
        active_tokens=1,
        effective_context_window=372_000,
        policy_fingerprint=policy,
    )
    assert job is not None
    return job, policy


def _single_externalized_raw_fixture(tmp_path: Path):
    config = _config(tmp_path)
    original = (
        "PAYLOAD_HEAD api_key=synthetic-sensitive-value\n"
        + ("immutable externalized source body\n" * 180)
        + "PAYLOAD_TAIL"
    )
    externalized = externalize_ingest_payload(
        original,
        role="user",
        session_id=SESSION_ID,
        field_path="content",
        config=config,
        kind="ingest_payload",
    )
    assert externalized is not None
    placeholder = externalized["placeholder"]
    payload_path = Path(externalized["path"])
    source_ids = _initialize_source_rows(
        config,
        [{"role": "user", "content": placeholder}],
    )
    return config, original, placeholder, payload_path, source_ids


def _damage_payload(path: Path, mode: str) -> None:
    if mode == "deleted":
        path.unlink()
        return
    assert mode == "mutated"
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = str(payload["content"])
    # Preserve byte/character length and all persisted metadata.  Validation
    # must bind actual content, not trust self-reported size fields.
    replacement_prefix = "X" if not original.startswith("X") else "Y"
    payload["content"] = replacement_prefix + original[1:]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stored_source_rows(config: LCMConfig) -> list[dict]:
    store = MessageStore(config.database_path)
    try:
        return store.get_session_messages(SESSION_ID)
    finally:
        store.close()


def test_provider_keeps_externalized_payloads_local_and_summarizes_only_refs(
    tmp_path,
    monkeypatch,
):
    """External payload bytes stay local while their durable refs remain visible."""
    config = _config(tmp_path)
    config.ignore_message_patterns = [r"RAW_HEAD"]

    class PayloadOnlyPattern:
        pattern = r"RAW_HEAD"

        def search(self, text, timeout=None):
            del timeout
            return "RAW_HEAD" in text

    monkeypatch.setattr(
        "hermes_lcm.summary_serialization.compile_message_patterns",
        lambda _patterns: [PayloadOnlyPattern()],
    )
    monkeypatch.setattr(
        "hermes_lcm.summary_serialization.load_externalized_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("background preparation must not hydrate payload bytes")
        ),
    )
    raw_secret = "synthetic-background-sensitive-value"
    tool_secret = "background-tool-secret-1234567890abcdef"
    raw_text = (
        f"RAW_HEAD api_key={raw_secret}\n"
        + ("raw externalized body " * 240)
        + " RAW_TAIL"
    )
    tool_text = (
        f"TOOL_HEAD password={tool_secret}\n"
        + ("tool externalized body " * 240)
        + " TOOL_TAIL"
    )
    raw_payload = externalize_ingest_payload(
        raw_text,
        role="user",
        session_id=SESSION_ID,
        field_path="content",
        config=config,
        kind="ingest_payload",
    )
    tool_payload = maybe_externalize_tool_output(
        tool_text,
        tool_call_id="call-externalized",
        session_id=SESSION_ID,
        config=config,
        force=True,
    )
    assert raw_payload is not None
    assert tool_payload is not None
    source_ids = _initialize_source_rows(
        config,
        [
            {"role": "user", "content": raw_payload["placeholder"]},
            {
                "role": "assistant",
                "content": "Running the durable tool call.",
                "tool_calls": [
                    {
                        "id": "call-externalized",
                        "type": "function",
                        "function": {"name": "large_result", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-externalized",
                "content": tool_payload["placeholder"],
            },
        ],
        token_estimates=[1_500, 20, 1_500],
    )
    bg = BackgroundCompactionStore(config.database_path, config=config)
    captured_text: list[str] = []

    def summarize(**kwargs):
        captured_text.append(kwargs["text"])
        return "prepared externalized summary", 0

    try:
        job, _policy = _capture(bg, config)
        assert job.source_store_ids == tuple(source_ids)
        result = bg.run_next_job(owner="externalized-worker", summarizer=summarize)
        assert result is not None and result.state == "prepared"
        assert len(captured_text) == 1
        provider_text = captured_text[0]

        assert "RAW_HEAD" not in provider_text
        assert "RAW_TAIL" not in provider_text
        assert "TOOL_HEAD" not in provider_text
        assert "TOOL_TAIL" not in provider_text
        assert raw_secret not in provider_text
        assert tool_secret not in provider_text
        assert "[Externalized LCM ingest payload:" in provider_text
        assert "[Externalized tool output:" in provider_text
    finally:
        bg.close()


@pytest.mark.parametrize("damage_mode", ["mutated", "deleted"])
def test_payload_drift_after_capture_does_not_hydrate_bytes_or_change_row_lineage(
    tmp_path,
    damage_mode,
):
    """Payload mutation cannot make local bytes enter the background provider."""
    config, original, placeholder, payload_path, source_ids = (
        _single_externalized_raw_fixture(tmp_path)
    )
    bg = BackgroundCompactionStore(config.database_path, config=config)
    provider_calls: list[str] = []
    try:
        job, _policy = _capture(bg, config)
        _damage_payload(payload_path, damage_mode)

        result = bg.run_next_job(
            owner="externalized-worker",
            summarizer=lambda **kwargs: provider_calls.append(kwargs["text"])
            or ("marker-only prepared summary", 0),
        )

        rows = _stored_source_rows(config)
        assert [row["store_id"] for row in rows] == source_ids
        assert [row["content"] for row in rows] == [placeholder]
        assert len(provider_calls) == 1
        assert placeholder in provider_calls[0]
        assert original not in provider_calls[0]
        assert "PAYLOAD_HEAD" not in provider_calls[0]
        assert result is not None and result.state == "prepared"
        prepared_nodes = bg.get_prepared_nodes(job.job_id)
        assert len(prepared_nodes) == 1
        assert prepared_nodes[0].source_store_ids == tuple(source_ids)
    finally:
        bg.close()


@pytest.mark.parametrize("damage_mode", ["mutated", "deleted"])
def test_payload_drift_after_preparation_cannot_change_marker_based_publication(
    tmp_path,
    damage_mode,
):
    """Publication validates immutable rows and preserves exact marker lineage."""
    config, _original, placeholder, payload_path, source_ids = (
        _single_externalized_raw_fixture(tmp_path)
    )
    bg = BackgroundCompactionStore(config.database_path, config=config)
    try:
        job, policy = _capture(bg, config)
        prepared = bg.run_next_job(
            owner="externalized-worker",
            summarizer=lambda **_kwargs: ("prepared before dependency drift", 0),
        )
        assert prepared is not None and prepared.state == "prepared"
        _damage_payload(payload_path, damage_mode)

        publication_lease = bg.acquire_publication_lease(
            job.job_id,
            owner="externalized-publisher",
        )
        assert publication_lease is not None
        result = bg.publish_prepared(
            job.job_id,
            owner="externalized-publisher",
            generation=publication_lease.lease_generation,
            policy_fingerprint=policy,
        )

        rows = _stored_source_rows(config)
        assert [row["store_id"] for row in rows] == source_ids
        assert [row["content"] for row in rows] == [placeholder]
        assert result.promoted is True
        assert result.reason == "published"
        assert bg.get_job(job.job_id).state == "published"

        lifecycle = LifecycleStateStore(config.database_path)
        try:
            state = lifecycle.get_by_conversation(CONVERSATION_ID)
            assert state is not None
            assert state.current_frontier_store_id == source_ids[-1]
        finally:
            lifecycle.close()
        dag = SummaryDAG(config.database_path)
        try:
            assert dag.get_session_node_count(SESSION_ID) == 1
            assert dag.source_message_ids(result.node_ids[0], limit=100) == source_ids
        finally:
            dag.close()
    finally:
        bg.close()
