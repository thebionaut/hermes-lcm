"""Regression coverage for special-job capture scope and dispositions."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.background_compaction import BackgroundCompactionStore, CompactionJob
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


_CONVERSATION = "conversation"
_SESSION = "session"
_POLICY_A = "policy-a"
_POLICY_B = "policy-b"


def _config(db_path, **overrides) -> LCMConfig:
    values = {
        "database_path": str(db_path),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "background_compaction_global_limit": 2,
        "background_compaction_backlog_tokens": 1,
        "background_compaction_min_provider_source_tokens": 256,
        "background_compaction_prepare_ratio": 0.0,
        "background_compaction_max_passes": 4,
        "background_compaction_job_timeout_seconds": 30,
        "background_compaction_lease_seconds": 300,
        "fresh_tail_count": 0,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 10_000,
    }
    values.update(overrides)
    return LCMConfig(**values)


def _activate(db_path, config: LCMConfig) -> BackgroundCompactionStore:
    lifecycle = LifecycleStateStore(str(db_path))
    try:
        state = lifecycle.bind_session(_SESSION, conversation_id=_CONVERSATION)
    finally:
        lifecycle.close()
    background = BackgroundCompactionStore(str(db_path), config=config)
    generation = background.activate_session(
        conversation_id=_CONVERSATION,
        session_id=_SESSION,
        expected_lifecycle_bound_at=state.current_bound_at,
        expected_lifecycle_reset_at=state.last_reset_at,
    )
    assert generation is not None
    return background


def _capture(
    background: BackgroundCompactionStore,
    *,
    policy: str,
    frontier: int = 0,
) -> CompactionJob | None:
    return background.capture_and_enqueue(
        conversation_id=_CONVERSATION,
        session_id=_SESSION,
        expected_frontier_store_id=frontier,
        active_tokens=300_000,
        effective_context_window=372_000,
        policy_fingerprint=policy,
    )


def _lossless_fixture(db_path):
    messages = MessageStore(str(db_path))
    try:
        source_ids = tuple(
            messages.append_batch(
                _SESSION,
                [{"role": "user", "content": "small retained correction"}],
                token_estimates=[32],
                conversation_id=_CONVERSATION,
            )
        )
    finally:
        messages.close()
    dag = SummaryDAG(str(db_path))
    dag.close()
    config = _config(db_path)
    return _activate(db_path, config), config, source_ids


def _partial_fixture(db_path, *, preseed_global_gap: bool = False):
    messages = MessageStore(str(db_path))
    try:
        gap_store_id = 0
        if preseed_global_gap:
            gap_store_id = int(
                messages.append_batch(
                    "other-session",
                    [{"role": "user", "content": "global store-id gap"}],
                    token_estimates=[1],
                    conversation_id="other-conversation",
                )[0]
            )
        source_ids = tuple(
            messages.append_batch(
                _SESSION,
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
                    {"role": "tool", "tool_call_id": "call-a", "content": "first"},
                    {"role": "tool", "tool_call_id": "call-b", "content": "second"},
                    {"role": "user", "content": "protected tail"},
                ],
                token_estimates=[10, 10, 10, 10],
                conversation_id=_CONVERSATION,
            )
        )
    finally:
        messages.close()

    group_ids = source_ids[:3]
    dag = SummaryDAG(str(db_path))
    try:
        dag.add_node(
            SummaryNode(
                session_id=_SESSION,
                summary="legacy partial owner",
                source_ids=list(group_ids[:2]),
                source_type="messages",
                token_count=4,
                source_token_count=20,
            )
        )
    finally:
        dag.close()
    config = _config(
        db_path,
        background_compaction_min_provider_source_tokens=1,
        fresh_tail_count=1,
    )
    return _activate(db_path, config), config, source_ids, group_ids, gap_store_id


def _advance_fence_without_superseding(background: BackgroundCompactionStore) -> int:
    row = background.connection.execute(
        "SELECT generation FROM lcm_compaction_session_fences WHERE conversation_id=?",
        (_CONVERSATION,),
    ).fetchone()
    assert row is not None
    generation = int(row[0]) + 1
    background.connection.execute(
        """
        UPDATE lcm_compaction_session_fences
        SET generation=?, publishable=1
        WHERE conversation_id=? AND session_id=?
        """,
        (generation, _CONVERSATION, _SESSION),
    )
    background.connection.commit()
    return generation


@pytest.mark.parametrize("stale_dimension", ["policy", "fence"])
def test_stale_lossless_job_is_superseded_for_current_capture_scope(
    tmp_path,
    stale_dimension,
):
    background, _config_value, _source_ids = _lossless_fixture(
        tmp_path / f"lossless-{stale_dimension}.db"
    )
    try:
        stale = _capture(background, policy=_POLICY_A)
        assert isinstance(stale, CompactionJob)
        assert stale.job_kind == "lossless_raw_projection"
        assert stale.state == "prepared"

        current_policy = _POLICY_A
        expected_generation = stale.session_fence_generation
        if stale_dimension == "policy":
            current_policy = _POLICY_B
        else:
            expected_generation = _advance_fence_without_superseding(background)

        replacement = _capture(background, policy=current_policy)
        assert isinstance(replacement, CompactionJob)
        assert replacement.job_id != stale.job_id
        assert replacement.job_kind == "lossless_raw_projection"
        assert replacement.policy_fingerprint == current_policy
        assert replacement.session_fence_generation == expected_generation
        assert background.get_job(stale.job_id).state == "superseded"
    finally:
        background.close()


@pytest.mark.parametrize("stale_dimension", ["policy", "fence"])
def test_stale_partial_repair_is_superseded_for_current_capture_scope(
    tmp_path,
    stale_dimension,
):
    background, _config_value, _source_ids, group_ids, _gap = _partial_fixture(
        tmp_path / f"partial-{stale_dimension}.db"
    )
    try:
        stale = _capture(background, policy=_POLICY_A)
        assert isinstance(stale, CompactionJob)
        assert stale.job_kind == "partial_atomic_repair"
        assert stale.repair_group_store_ids == group_ids

        current_policy = _POLICY_A
        expected_generation = stale.session_fence_generation
        if stale_dimension == "policy":
            current_policy = _POLICY_B
        else:
            expected_generation = _advance_fence_without_superseding(background)

        replacement = _capture(background, policy=current_policy)
        assert isinstance(replacement, CompactionJob)
        assert replacement.job_id != stale.job_id
        assert replacement.job_kind == "partial_atomic_repair"
        assert replacement.policy_fingerprint == current_policy
        assert replacement.session_fence_generation == expected_generation
        assert background.get_job(stale.job_id).state == "superseded"
    finally:
        background.close()


def _terminally_dispose_partial(background: BackgroundCompactionStore) -> CompactionJob:
    job = _capture(background, policy=_POLICY_A)
    assert isinstance(job, CompactionJob)
    assert job.job_kind == "partial_atomic_repair"
    claimed = background.claim_next_job(
        owner="worker",
        policy_fingerprint=_POLICY_A,
    )
    assert isinstance(claimed, CompactionJob)
    failed = background.mark_retry(
        claimed.job_id,
        owner="worker",
        generation=claimed.lease_generation,
        error=RuntimeError("non-compressing test result"),
        completion_reason="non_compressing_output",
        terminal=True,
    )
    assert isinstance(failed, CompactionJob)
    assert failed.state == "failed"
    return failed


def _mark_prepared_without_provider(
    background: BackgroundCompactionStore,
    job: CompactionJob,
    *,
    owner: str,
    bind_partial: bool = False,
) -> CompactionJob:
    claimed = background.claim_next_job(
        owner=owner,
        policy_fingerprint=job.policy_fingerprint,
    )
    assert isinstance(claimed, CompactionJob)
    assert claimed.job_id == job.job_id
    if bind_partial:
        bound = background._bind_partial_atomic_repair(
            claimed.job_id,
            owner=owner,
            generation=claimed.lease_generation,
        )
        assert isinstance(bound, CompactionJob)
        claimed = bound
    prepared = background.mark_prepared(
        claimed.job_id,
        owner=owner,
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "prepared fixture summary",
                "source_store_ids": list(claimed.source_store_ids),
                "source_token_count": claimed.source_token_count,
                "summary_token_count": 3,
                "creation_model": "fixture-model",
                "creation_provider": "fixture-provider",
            }
        ],
        expected_node_count=1,
    )
    assert isinstance(prepared, CompactionJob)
    assert prepared.state == "prepared"
    return prepared


def _enqueue_standard(
    background: BackgroundCompactionStore,
    source_ids: tuple[int, ...],
    *,
    policy: str,
) -> CompactionJob:
    messages = MessageStore(str(background.db_path))
    try:
        rows = messages.get_range(_SESSION, source_ids[0], source_ids[-1])
    finally:
        messages.close()
    job = background.enqueue_or_coalesce(
        conversation_id=_CONVERSATION,
        session_id=_SESSION,
        expected_frontier_store_id=0,
        captured_watermark_store_id=source_ids[-1],
        source_store_ids=source_ids,
        source_token_count=sum(int(row.get("token_estimate") or 0) for row in rows),
        policy_fingerprint=policy,
        expected_node_count=1,
        # Cross-kind fixtures deliberately emulate a pre-existing standard job
        # captured before canonical ownership changed underneath it.
        reject_canonical_coverage=False,
    )
    assert isinstance(job, CompactionJob)
    assert job.job_kind == "standard"
    return job


def test_terminal_partial_disposition_suppresses_exact_recreation_after_reopen(
    tmp_path,
):
    db_path = tmp_path / "partial-terminal-exact.db"
    background, config, _source_ids, _group_ids, _gap = _partial_fixture(db_path)
    failed = _terminally_dispose_partial(background)
    background.close()

    reopened = BackgroundCompactionStore(str(db_path), config=config)
    try:
        assert _capture(reopened, policy=_POLICY_A) is None
        assert reopened.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_jobs
            WHERE conversation_id=? AND state IN ('queued','running','prepared','retry')
            """,
            (_CONVERSATION,),
        ).fetchone()[0] == 0
        assert reopened.get_job(failed.job_id).state == "failed"
    finally:
        reopened.close()


@pytest.mark.parametrize("changed_dimension", ["policy", "source", "frontier"])
def test_terminal_partial_disposition_does_not_suppress_changed_capture_scope(
    tmp_path,
    changed_dimension,
):
    db_path = tmp_path / f"partial-terminal-{changed_dimension}.db"
    background, config, source_ids, group_ids, gap_store_id = _partial_fixture(
        db_path,
        preseed_global_gap=changed_dimension == "frontier",
    )
    failed = _terminally_dispose_partial(background)
    background.close()

    current_policy = _POLICY_A
    current_frontier = 0
    if changed_dimension == "policy":
        current_policy = _POLICY_B
    elif changed_dimension == "source":
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE messages SET content=? WHERE store_id=?",
                ("changed second result", group_ids[-1]),
            )
    else:
        assert gap_store_id > 0 and gap_store_id < source_ids[0]
        lifecycle = LifecycleStateStore(str(db_path))
        try:
            advanced = lifecycle.advance_frontier(
                _CONVERSATION,
                _SESSION,
                gap_store_id,
            )
            assert advanced is not None
            assert advanced.current_frontier_store_id == gap_store_id
        finally:
            lifecycle.close()
        current_frontier = gap_store_id

    reopened = BackgroundCompactionStore(str(db_path), config=config)
    try:
        replacement = _capture(
            reopened,
            policy=current_policy,
            frontier=current_frontier,
        )
        assert isinstance(replacement, CompactionJob)
        assert replacement.job_id != failed.job_id
        assert replacement.job_kind == "partial_atomic_repair"
        assert replacement.policy_fingerprint == current_policy
        assert replacement.expected_frontier_store_id == current_frontier
    finally:
        reopened.close()


def test_lossless_capture_replaces_prepared_job_after_captured_source_changes(
    tmp_path,
):
    db_path = tmp_path / "lossless-active-source-change.db"
    background, _config_value, source_ids = _lossless_fixture(db_path)
    try:
        stale = _capture(background, policy=_POLICY_A)
        assert isinstance(stale, CompactionJob)
        assert stale.job_kind == "lossless_raw_projection"
        assert stale.state == "prepared"

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE messages SET content=? WHERE store_id=?",
                ("changed retained correction", source_ids[-1]),
            )

        replacement = _capture(background, policy=_POLICY_A)
        assert isinstance(replacement, CompactionJob)
        assert replacement.job_id != stale.job_id
        assert replacement.job_kind == "lossless_raw_projection"
        assert replacement.state == "prepared"
        assert replacement.source_identity_hash != stale.source_identity_hash
        superseded = background.get_job(stale.job_id)
        assert isinstance(superseded, CompactionJob)
        assert superseded.state == "superseded"
        assert superseded.completion_reason == "source_identity_mismatch"
    finally:
        background.close()


def test_lossless_capture_replaces_same_scope_prepared_standard_job(tmp_path):
    db_path = tmp_path / "lossless-cross-kind.db"
    background, _config_value, source_ids = _lossless_fixture(db_path)
    try:
        stale = _enqueue_standard(background, source_ids, policy=_POLICY_A)
        stale = _mark_prepared_without_provider(
            background,
            stale,
            owner="standard-worker",
        )

        replacement = _capture(background, policy=_POLICY_A)
        assert isinstance(replacement, CompactionJob)
        assert replacement.job_id != stale.job_id
        assert replacement.job_kind == "lossless_raw_projection"
        assert replacement.state == "prepared"
        superseded = background.get_job(stale.job_id)
        assert isinstance(superseded, CompactionJob)
        assert superseded.state == "superseded"
        assert superseded.completion_reason == "special_job_kind_mismatch"
    finally:
        background.close()


def test_lossless_capture_preserves_valid_prepared_job_when_only_newer_rows_arrive(
    tmp_path,
):
    db_path = tmp_path / "lossless-newer-tail.db"
    background, _config_value, _source_ids = _lossless_fixture(db_path)
    try:
        prepared = _capture(background, policy=_POLICY_A)
        assert isinstance(prepared, CompactionJob)
        assert prepared.job_kind == "lossless_raw_projection"
        assert prepared.state == "prepared"

        messages = MessageStore(str(db_path))
        try:
            newer_id = messages.append_batch(
                _SESSION,
                [{"role": "assistant", "content": "newer retained tail"}],
                token_estimates=[32],
                conversation_id=_CONVERSATION,
            )[0]
        finally:
            messages.close()
        assert newer_id > prepared.captured_watermark_store_id

        still_prepared = _capture(background, policy=_POLICY_A)
        assert isinstance(still_prepared, CompactionJob)
        assert still_prepared.job_id == prepared.job_id
        assert still_prepared.state == "prepared"
        assert still_prepared.captured_watermark_store_id < newer_id
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs WHERE conversation_id=?",
            (_CONVERSATION,),
        ).fetchone()[0] == 1
    finally:
        background.close()


def test_partial_capture_replaces_prepared_job_after_captured_source_changes(
    tmp_path,
):
    db_path = tmp_path / "partial-active-source-change.db"
    background, _config_value, _source_ids, group_ids, _gap = _partial_fixture(
        db_path
    )
    try:
        stale = _capture(background, policy=_POLICY_A)
        assert isinstance(stale, CompactionJob)
        assert stale.job_kind == "partial_atomic_repair"
        stale = _mark_prepared_without_provider(
            background,
            stale,
            owner="partial-worker",
            bind_partial=True,
        )

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE messages SET content=? WHERE store_id=?",
                ("changed second tool result", group_ids[-1]),
            )

        replacement = _capture(background, policy=_POLICY_A)
        assert isinstance(replacement, CompactionJob)
        assert replacement.job_id != stale.job_id
        assert replacement.job_kind == "partial_atomic_repair"
        assert replacement.source_identity_hash != stale.source_identity_hash
        superseded = background.get_job(stale.job_id)
        assert isinstance(superseded, CompactionJob)
        assert superseded.state == "superseded"
        assert superseded.completion_reason == "partial_repair_source_range_changed"
    finally:
        background.close()


def test_partial_capture_replaces_same_scope_prepared_standard_job(tmp_path):
    db_path = tmp_path / "partial-cross-kind.db"
    background, _config_value, _source_ids, group_ids, _gap = _partial_fixture(
        db_path
    )
    try:
        stale = _enqueue_standard(background, group_ids, policy=_POLICY_A)
        stale = _mark_prepared_without_provider(
            background,
            stale,
            owner="standard-worker",
        )

        replacement = _capture(background, policy=_POLICY_A)
        assert isinstance(replacement, CompactionJob)
        assert replacement.job_id != stale.job_id
        assert replacement.job_kind == "partial_atomic_repair"
        superseded = background.get_job(stale.job_id)
        assert isinstance(superseded, CompactionJob)
        assert superseded.state == "superseded"
        assert superseded.completion_reason == "partial_repair_source_range_changed"
    finally:
        background.close()
