"""Race and long-chain proofs not covered by the core partial-repair tests."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Callable

import pytest

from hermes_lcm.background_compaction import (
    BackgroundCompactionStore,
    CompactionJob,
    background_policy_fingerprint,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


class _TimeoutPattern:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    def search(self, text: str, *, timeout: float | None = None):
        del timeout
        return self._compiled.search(text)


class _TimeoutRegexEngine:
    error = re.error

    @staticmethod
    def compile(pattern: str):
        return _TimeoutPattern(pattern)


@dataclass(frozen=True)
class PartialFixture:
    db_path: object
    config: LCMConfig
    background: BackgroundCompactionStore
    source_ids: tuple[int, ...]
    group_ids: tuple[int, ...]
    owner_id: int
    root_id: int


def _config(db_path, **overrides) -> LCMConfig:
    values = dict(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=2,
        background_compaction_backlog_tokens=1,
        background_compaction_prepare_ratio=0.0,
        background_compaction_max_passes=4,
        background_compaction_lease_seconds=300,
        background_compaction_job_timeout_seconds=30,
        fresh_tail_count=1,
        leaf_chunk_tokens=10_000,
    )
    values.update(overrides)
    return LCMConfig(**values)


def _activate(db_path, config: LCMConfig) -> BackgroundCompactionStore:
    lifecycle = LifecycleStateStore(db_path)
    state = lifecycle.bind_session("session", conversation_id="conversation")
    lifecycle.close()
    background = BackgroundCompactionStore(db_path, config=config)
    generation = background.activate_session(
        conversation_id="conversation",
        session_id="session",
        expected_lifecycle_bound_at=state.current_bound_at,
    )
    assert generation is not None
    return background


def _single_group_fixture(db_path) -> PartialFixture:
    messages = MessageStore(db_path)
    source_ids = tuple(
        messages.append_batch(
            "session",
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "first", "arguments": "{}"},
                        },
                        {
                            "id": "call-b",
                            "type": "function",
                            "function": {"name": "second", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-a", "content": "first result"},
                {"role": "tool", "tool_call_id": "call-b", "content": "second result"},
                {"role": "user", "content": "protected tail"},
            ],
            token_estimates=[10, 10, 10, 10],
            conversation_id="conversation",
        )
    )
    messages.close()
    group_ids = source_ids[:3]
    dag = SummaryDAG(db_path)
    owner_id = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy owner",
            source_ids=list(group_ids[:2]),
            source_type="messages",
            token_count=4,
            source_token_count=20,
        )
    )
    root_id = dag.add_node(
        SummaryNode(
            session_id="session",
            depth=1,
            summary="legacy root",
            source_ids=[owner_id],
            source_type="nodes",
            token_count=3,
            source_token_count=20,
        )
    )
    dag.close()
    config = _config(db_path)
    return PartialFixture(
        db_path=db_path,
        config=config,
        background=_activate(db_path, config),
        source_ids=source_ids,
        group_ids=group_ids,
        owner_id=owner_id,
        root_id=root_id,
    )


def _capture(
    background: BackgroundCompactionStore,
    config: LCMConfig,
    *,
    frontier: int,
) -> CompactionJob:
    job = background.capture_and_enqueue(
        conversation_id="conversation",
        session_id="session",
        expected_frontier_store_id=frontier,
        active_tokens=300_000,
        effective_context_window=372_000,
        policy_fingerprint=background_policy_fingerprint(config),
    )
    assert isinstance(job, CompactionJob)
    assert job.job_kind == "partial_atomic_repair"
    return job


def _bind_prepare_publish(
    background: BackgroundCompactionStore,
    config: LCMConfig,
    *,
    worker: str,
    publisher: str,
):
    claimed = background.claim_next_job(
        owner=worker,
        policy_fingerprint=background_policy_fingerprint(config),
    )
    assert isinstance(claimed, CompactionJob)
    bound = background._bind_partial_atomic_repair(
        claimed.job_id,
        owner=worker,
        generation=claimed.lease_generation,
    )
    assert isinstance(bound, CompactionJob)
    prepared = background.mark_prepared(
        bound.job_id,
        owner=worker,
        generation=bound.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": f"repair parent {bound.captured_watermark_store_id}",
                "source_store_ids": list(bound.source_store_ids),
                "source_token_count": bound.source_token_count,
                "summary_token_count": 3,
                "creation_model": "test-model",
                "creation_provider": "test-provider",
            }
        ],
        expected_node_count=1,
    )
    assert isinstance(prepared, CompactionJob)
    lease = background.acquire_publication_lease(prepared.job_id, owner=publisher)
    assert isinstance(lease, CompactionJob)
    result = background.publish_prepared(
        lease.job_id,
        owner=publisher,
        generation=lease.lease_generation,
        policy_fingerprint=background_policy_fingerprint(config),
    )
    return bound, result


def _canonical_snapshot(background: BackgroundCompactionStore) -> tuple:
    nodes = tuple(
        tuple(row)
        for row in background.connection.execute(
            """
            SELECT node_id, session_id, depth, summary, source_ids, source_type
            FROM summary_nodes ORDER BY node_id
            """
        ).fetchall()
    )
    coverage = tuple(
        tuple(row)
        for row in background.connection.execute(
            """
            SELECT session_id, store_id, publication_id
            FROM lcm_compaction_source_coverage ORDER BY session_id, store_id
            """
        ).fetchall()
    )
    lifecycle = tuple(
        background.connection.execute(
            """
            SELECT current_frontier_store_id, debt_size_estimate
            FROM lcm_lifecycle_state WHERE conversation_id='conversation'
            """
        ).fetchone()
    )
    return nodes, coverage, lifecycle


def _mutate_owner(fixture: PartialFixture) -> None:
    conn = sqlite3.connect(str(fixture.db_path), timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE summary_nodes SET source_ids=? WHERE node_id=?",
            (json.dumps([fixture.group_ids[0]]), fixture.owner_id),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()


def _mutate_root_identity(fixture: PartialFixture) -> None:
    conn = sqlite3.connect(str(fixture.db_path), timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE summary_nodes SET summary=summary || ' changed' WHERE node_id=?",
            (fixture.root_id,),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()


def _mutate_topology(fixture: PartialFixture) -> None:
    dag = SummaryDAG(fixture.db_path)
    try:
        dag.add_node(
            SummaryNode(
                session_id="session",
                depth=2,
                summary="new root introduced during provider work",
                source_ids=[fixture.root_id],
                source_type="nodes",
                token_count=3,
                source_token_count=20,
            )
        )
    finally:
        dag.close()


def _mutate_raw(fixture: PartialFixture) -> None:
    conn = sqlite3.connect(str(fixture.db_path), timeout=30, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE messages SET content=content || ' changed' WHERE store_id=?",
            (fixture.group_ids[-1],),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()


@pytest.mark.parametrize(
    "mutator",
    [_mutate_owner, _mutate_root_identity, _mutate_topology, _mutate_raw],
    ids=["owner", "root-identity", "topology", "raw"],
)
def test_mutation_while_provider_is_blocked_cannot_publish_stale_repair(
    tmp_path,
    mutator: Callable[[PartialFixture], None],
):
    """The provider wait holds no SQL lock and stale output never publishes."""
    fixture = _single_group_fixture(
        tmp_path / f"provider-race-{mutator.__name__}.db"
    )
    background = fixture.background
    provider_started = threading.Event()
    provider_release = threading.Event()
    worker_result: list[CompactionJob | None] = []
    worker_errors: list[BaseException] = []

    def summarizer(**_kwargs):
        provider_started.set()
        assert provider_release.wait(timeout=10), "test did not release provider"
        return "provider repair parent", 0

    def prepare() -> None:
        try:
            worker_result.append(
                background.run_next_job("worker", summarizer=summarizer)
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            worker_errors.append(exc)

    try:
        job = _capture(background, fixture.config, frontier=0)
        thread = threading.Thread(target=prepare, name="partial-repair-provider-race")
        thread.start()
        assert provider_started.wait(timeout=10), "provider did not start"
        mutator(fixture)
        post_mutation = _canonical_snapshot(background)
        provider_release.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker did not finish after provider release"
        assert not worker_errors
        assert len(worker_result) == 1

        current = background.get_job(job.job_id)
        assert isinstance(current, CompactionJob)
        if current.state == "prepared":
            lease = background.acquire_publication_lease(
                current.job_id, owner="publisher"
            )
            assert isinstance(lease, CompactionJob)
            result = background.publish_prepared(
                current.job_id,
                owner="publisher",
                generation=lease.lease_generation,
                policy_fingerprint=background_policy_fingerprint(fixture.config),
            )
            assert result.promoted is False
        current = background.get_job(job.job_id)
        assert isinstance(current, CompactionJob)
        assert current.state == "superseded"
        assert current.completion_reason in {
            "partial_repair_ownership_changed",
            "partial_repair_dag_epoch_changed",
            "partial_repair_topology_changed",
            "partial_repair_source_range_changed",
            "source_identity_mismatch",
        }
        assert _canonical_snapshot(background) == post_mutation
    finally:
        provider_release.set()
        background.close()


def test_partial_repair_provider_payload_applies_policy_without_losing_lineage(
    tmp_path,
    monkeypatch,
):
    import hermes_lcm.message_patterns as message_patterns

    monkeypatch.setattr(message_patterns, "_regex_engine", _TimeoutRegexEngine)
    db_path = tmp_path / "partial-repair-provider-policy.db"
    raw_argument_secret = "rawargumentsecret12345"
    raw_tool_secret = "rawtoolsecret12345"
    ignored_raw_secret = "ignoredrawsecret12345"
    visible_root_secret = "canonicalrootsecret12345"
    ignored_root_secret = "ignoredrootsecret12345"

    messages = MessageStore(db_path)
    source_ids = tuple(
        messages.append_batch(
            "session",
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {
                                "name": "first",
                                "arguments": json.dumps(
                                    {"api_key": raw_argument_secret}
                                ),
                            },
                        },
                        {
                            "id": "call-b",
                            "type": "function",
                            "function": {"name": "second", "arguments": "{}"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-a",
                    "content": f"Bearer {raw_tool_secret}",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-b",
                    "content": (
                        f"DROP_RAW_MESSAGE Bearer {ignored_raw_secret}"
                    ),
                },
                {"role": "user", "content": "protected tail"},
            ],
            token_estimates=[10, 10, 10, 10],
            conversation_id="conversation",
        )
    )
    messages.close()
    group_ids = source_ids[:3]

    dag = SummaryDAG(db_path)
    owner_one = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy owner one",
            source_ids=list(group_ids[:2]),
            source_type="messages",
            token_count=4,
            source_token_count=20,
        )
    )
    owner_two = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy duplicate owner",
            source_ids=[group_ids[0]],
            source_type="messages",
            token_count=4,
            source_token_count=10,
        )
    )
    ignored_root = dag.add_node(
        SummaryNode(
            session_id="session",
            depth=1,
            summary=(
                f"DROP_ROOT_MESSAGE api_key={ignored_root_secret}"
            ),
            source_ids=[owner_one],
            source_type="nodes",
            token_count=7,
            source_token_count=20,
        )
    )
    visible_root = dag.add_node(
        SummaryNode(
            session_id="session",
            depth=1,
            summary=f"visible canonical api_key={visible_root_secret}",
            source_ids=[owner_two],
            source_type="nodes",
            token_count=8,
            source_token_count=10,
        )
    )
    dag.close()

    config = _config(
        db_path,
        ignore_message_patterns=[r"DROP_(?:ROOT|RAW)_MESSAGE"],
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key", "bearer_token"],
    )
    background = _activate(db_path, config)
    seen_text: list[str] = []

    def summarizer(**kwargs):
        seen_text.append(kwargs["text"])
        return "policy-safe repair parent", 0

    try:
        captured = _capture(background, config, frontier=0)
        assert captured.source_store_ids == group_ids
        assert captured.repair_group_store_ids == group_ids
        assert captured.repair_covered_store_ids == group_ids[:2]
        assert captured.repair_uncovered_store_ids == group_ids[2:]
        assert captured.repair_owner_node_ids == (owner_one, owner_two)
        captured_identity = captured.source_identity_hash

        prepared = background.run_next_job("policy-worker", summarizer=summarizer)
        assert isinstance(prepared, CompactionJob)
        assert prepared.state == "prepared"
        assert prepared.source_store_ids == group_ids
        assert prepared.repair_group_store_ids == group_ids
        assert prepared.repair_owner_node_ids == (owner_one, owner_two)
        assert prepared.repair_root_node_ids == (ignored_root, visible_root)
        assert prepared.source_identity_hash == captured_identity
        assert prepared.repair_group_identity_hash == captured_identity
        assert prepared.repair_root_identity_hash
        durable_prepared = background.get_prepared_nodes(prepared.job_id)
        assert len(durable_prepared) == 1
        assert durable_prepared[0].source_store_ids == group_ids

        assert len(seen_text) == 1
        provider_text = seen_text[0]
        assert f"[CANONICAL ROOT node_id={ignored_root}" in provider_text
        assert f"[CANONICAL ROOT node_id={visible_root}" in provider_text
        assert "Provider-visible root content omitted by policy" in provider_text
        assert "visible canonical api_key=" in provider_text
        assert "DROP_ROOT_MESSAGE" not in provider_text
        assert ignored_root_secret not in provider_text
        assert visible_root_secret not in provider_text
        assert f"store_id={group_ids[0]} ownership=existing-canonical" in provider_text
        assert f"store_id={group_ids[1]} ownership=existing-canonical" in provider_text
        assert f"store_id={group_ids[2]} ownership=" not in provider_text
        assert "DROP_RAW_MESSAGE" not in provider_text
        assert raw_argument_secret not in provider_text
        assert raw_tool_secret not in provider_text
        assert ignored_raw_secret not in provider_text
        assert "[LCM sensitive redaction: name=api_key" in provider_text
        assert "[LCM sensitive redaction: name=bearer_token" in provider_text

        durable_rows = background.connection.execute(
            """
            SELECT store_id, content, tool_calls FROM messages
            WHERE store_id IN (?,?,?) ORDER BY store_id
            """,
            group_ids,
        ).fetchall()
        assert tuple(int(row["store_id"]) for row in durable_rows) == group_ids
        assert raw_argument_secret in str(durable_rows[0]["tool_calls"])
        assert raw_tool_secret in str(durable_rows[1]["content"])
        assert ignored_raw_secret in str(durable_rows[2]["content"])
        root_summaries = {
            int(row["node_id"]): str(row["summary"])
            for row in background.connection.execute(
                """
                SELECT node_id, summary FROM summary_nodes
                WHERE node_id IN (?,?)
                """,
                (ignored_root, visible_root),
            ).fetchall()
        }
        assert ignored_root_secret in root_summaries[ignored_root]
        assert visible_root_secret in root_summaries[visible_root]

        lease = background.acquire_publication_lease(
            prepared.job_id, owner="policy-publisher"
        )
        assert isinstance(lease, CompactionJob)
        published = background.publish_prepared(
            prepared.job_id,
            owner="policy-publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        assert published.promoted is True
        bridge_id, parent_id = published.node_ids
        bridge_sources = background.connection.execute(
            "SELECT source_ids FROM summary_nodes WHERE node_id=?",
            (bridge_id,),
        ).fetchone()[0]
        assert tuple(json.loads(bridge_sources)) == group_ids[2:]
        dag = SummaryDAG(db_path)
        try:
            assert dag.source_message_ids(parent_id, limit=100) == list(group_ids)
        finally:
            dag.close()
    finally:
        background.close()


def test_102_sequential_partial_repairs_remain_bounded_live_and_exact(tmp_path):
    """Production's observed worst chain stays below the topology bound."""
    db_path = tmp_path / "partial-repair-chain-102.db"
    group_count = 102
    payload: list[dict] = []
    token_estimates: list[int] = []
    for index in range(group_count):
        call_a = f"call-{index}-a"
        call_b = f"call-{index}-b"
        payload.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_a,
                            "type": "function",
                            "function": {"name": "first", "arguments": "{}"},
                        },
                        {
                            "id": call_b,
                            "type": "function",
                            "function": {"name": "second", "arguments": "{}"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_a,
                    "content": f"first result {index}",
                },
                {
                    "role": "tool",
                    "tool_call_id": call_b,
                    "content": f"second result {index}",
                },
            ]
        )
        token_estimates.extend([1, 1, 1])
    payload.append({"role": "user", "content": "protected tail"})
    token_estimates.append(1)

    messages = MessageStore(db_path)
    source_ids = tuple(
        messages.append_batch(
            "session",
            payload,
            token_estimates=token_estimates,
            conversation_id="conversation",
        )
    )
    messages.close()
    group_ids = tuple(
        source_ids[offset : offset + 3]
        for offset in range(0, group_count * 3, 3)
    )

    dag = SummaryDAG(db_path)
    owner_ids = tuple(
        dag.add_node(
            SummaryNode(
                session_id="session",
                summary=f"legacy owner {index}",
                source_ids=list(ids[:2]),
                source_type="messages",
                token_count=1,
                source_token_count=2,
            )
        )
        for index, ids in enumerate(group_ids)
    )
    legacy_root = dag.add_node(
        SummaryNode(
            session_id="session",
            depth=1,
            summary="legacy root over all partial groups",
            source_ids=list(owner_ids),
            source_type="nodes",
            token_count=10,
            source_token_count=group_count * 2,
        )
    )
    dag.close()

    config = _config(db_path)
    background = _activate(db_path, config)
    frontier = 0
    latest_parent = legacy_root
    max_bound_ancestry = 0
    try:
        for index, expected_group in enumerate(group_ids):
            job = _capture(background, config, frontier=frontier)
            assert job.repair_group_store_ids == expected_group
            bound, result = _bind_prepare_publish(
                background,
                config,
                worker=f"worker-{index}",
                publisher=f"publisher-{index}",
            )
            assert result.promoted is True
            assert result.reason == "partial_atomic_repair_published"
            assert result.captured_watermark_store_id == expected_group[-1]
            assert len(result.node_ids) == 2
            latest_parent = result.node_ids[-1]
            frontier = expected_group[-1]
            max_bound_ancestry = max(
                max_bound_ancestry, len(bound.repair_ancestry_node_ids)
            )
            assert len(bound.repair_root_node_ids) == 1
            assert len(bound.repair_ancestry_node_ids) <= 512

        lifecycle = background.connection.execute(
            """
            SELECT current_frontier_store_id, debt_size_estimate
            FROM lcm_lifecycle_state WHERE conversation_id='conversation'
            """
        ).fetchone()
        assert tuple(lifecycle) == (group_ids[-1][-1], 0)
        assert max_bound_ancestry == group_count + 1
        assert background.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_jobs
            WHERE job_kind='partial_atomic_repair' AND state='published'
            """
        ).fetchone()[0] == group_count
        assert background.connection.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0] == group_count * 3 + 1

        dag = SummaryDAG(db_path)
        try:
            active = dag.get_active_frontier_nodes("session")
            assert [node.node_id for node in active] == [latest_parent]
            expected_sources = [
                store_id for ids in group_ids for store_id in ids
            ]
            assert dag.source_message_ids(latest_parent, limit=1_000) == expected_sources
            assert active[0].depth == group_count + 1
        finally:
            dag.close()
    finally:
        background.close()
