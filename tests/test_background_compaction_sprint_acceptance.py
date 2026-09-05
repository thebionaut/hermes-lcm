"""Acceptance coverage for the September 2026 compaction reliability sprint.

Cross-process cases use the ``spawn`` context and explicit barriers/events so
they exercise SQLite durability without depending on scheduler sleeps.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import sqlite3
import sys
from pathlib import Path


def _bootstrap_plugin_package(plugin_root: str) -> None:
    """Make ``hermes_lcm`` importable in a freshly spawned test process."""
    if "hermes_lcm" in sys.modules:
        return
    root = Path(plugin_root)
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm",
        str(root / "__init__.py"),
        submodule_search_locations=[str(root)],
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(root)]
    module.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = module


def _config(db_path: str | Path, **overrides):
    from hermes_lcm.config import LCMConfig

    values = {
        "database_path": str(db_path),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "background_compaction_global_limit": 2,
        "background_compaction_max_passes": 4,
        "background_compaction_job_timeout_seconds": 5,
        "background_compaction_lease_seconds": 30,
        "background_compaction_backlog_tokens": 1,
        "background_compaction_prepare_ratio": 0.55,
        "background_compaction_min_provider_source_tokens": 256,
        "fresh_tail_count": 0,
        "fresh_tail_max_tokens": 0,
        "leaf_chunk_tokens": 120_000,
        "summary_model": "primary/model-a",
        "summary_fallback_models": ["fallback/model-b"],
        "context_reserve_tokens": 100_000,
    }
    values.update(overrides)
    return LCMConfig(**values)


def _seed(
    db_path: str | Path,
    messages: list[dict],
    token_estimates: list[int],
    *,
    session: str = "s",
    conversation: str = "c",
) -> list[int]:
    from hermes_lcm.dag import SummaryDAG
    from hermes_lcm.lifecycle_state import LifecycleStateStore
    from hermes_lcm.store import MessageStore

    store = MessageStore(str(db_path))
    try:
        source_ids = store.append_batch(
            session,
            messages,
            token_estimates=token_estimates,
            conversation_id=conversation,
        )
    finally:
        store.close()
    dag = SummaryDAG(str(db_path))
    dag.close()
    lifecycle = LifecycleStateStore(str(db_path))
    try:
        lifecycle.bind_session(session, conversation_id=conversation)
    finally:
        lifecycle.close()
    return source_ids


def _enqueue_explicit(
    background,
    source_ids: list[int],
    *,
    session: str,
    conversation: str,
    policy: str,
    frontier: int = 0,
    priority: int = 0,
):
    return background.enqueue_or_coalesce(
        conversation_id=conversation,
        session_id=session,
        expected_frontier_store_id=frontier,
        captured_watermark_store_id=source_ids[-1],
        source_store_ids=source_ids,
        source_token_count=1_000 * len(source_ids),
        policy_fingerprint=policy,
        expected_node_count=1,
        priority=priority,
    )


def _prepare(background, job, *, owner: str = "worker"):
    claimed = background.claim_next_job(owner=owner)
    assert claimed is not None and claimed.job_id == job.job_id
    prepared = background.mark_prepared(
        job.job_id,
        owner=owner,
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": "prepared acceptance summary",
                "source_store_ids": list(job.source_store_ids),
                "source_token_count": job.source_token_count,
                "summary_token_count": 3,
                "creation_model": "fixture-model",
                "creation_provider": "fixture-provider",
            }
        ],
    )
    assert prepared is not False
    return prepared


def _capture(background, config, *, session="s", conversation="c", frontier=0):
    from hermes_lcm.background_compaction import background_policy_fingerprint

    return background.capture_and_enqueue(
        conversation_id=conversation,
        session_id=session,
        expected_frontier_store_id=frontier,
        active_tokens=900_000,
        effective_context_window=1_000_000,
        policy_fingerprint=background_policy_fingerprint(config),
    )


def _publish_process(
    plugin_root,
    db_path,
    job_id,
    owner,
    generation,
    policy,
    start_barrier,
    results,
):
    _bootstrap_plugin_package(plugin_root)
    from hermes_lcm.background_compaction import BackgroundCompactionStore

    background = None
    try:
        background = BackgroundCompactionStore(
            db_path,
            config=_config(db_path),
        )
        start_barrier.wait(timeout=10)
        result = background.publish_prepared(
            job_id,
            owner=owner,
            generation=generation,
            policy_fingerprint=policy,
        )
        results.put(("ok", bool(result.promoted), str(result.reason)))
    except BaseException as exc:  # pragma: no cover - asserted in parent
        results.put(("error", type(exc).__name__, str(exc)[:200]))
    finally:
        if background is not None:
            background.close()


def _publish_after_event_process(
    plugin_root,
    db_path,
    job_id,
    owner,
    generation,
    policy,
    ready,
    start,
    results,
):
    _bootstrap_plugin_package(plugin_root)
    from hermes_lcm.background_compaction import BackgroundCompactionStore

    background = None
    try:
        background = BackgroundCompactionStore(
            db_path,
            config=_config(db_path),
        )
        ready.set()
        assert start.wait(timeout=10)
        result = background.publish_prepared(
            job_id,
            owner=owner,
            generation=generation,
            policy_fingerprint=policy,
        )
        results.put(("ok", bool(result.promoted), str(result.reason)))
    except BaseException as exc:  # pragma: no cover - asserted in parent
        results.put(("error", type(exc).__name__, str(exc)[:200]))
    finally:
        if background is not None:
            background.close()


def _capture_after_restart_process(
    plugin_root,
    db_path,
    ready,
    start,
    results,
):
    _bootstrap_plugin_package(plugin_root)
    from hermes_lcm.background_compaction import (
        BackgroundCompactionStore,
        background_policy_fingerprint,
    )

    background = None
    try:
        config = _config(db_path)
        background = BackgroundCompactionStore(db_path, config=config)
        ready.set()
        assert start.wait(timeout=10)
        captured = background.capture_and_enqueue(
            conversation_id="c",
            session_id="s",
            expected_frontier_store_id=0,
            active_tokens=900_000,
            effective_context_window=1_000_000,
            policy_fingerprint=background_policy_fingerprint(config),
        )
        active_count = background.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_jobs
            WHERE conversation_id='c'
              AND state IN ('queued','running','prepared','retry')
            """
        ).fetchone()[0]
        total_job_count = background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs WHERE conversation_id='c'"
        ).fetchone()[0]
        disposition_rows = background.connection.execute(
            """
            SELECT conversation_id, session_id, session_fence_generation,
                   expected_frontier_store_id, captured_watermark_store_id,
                   policy_fingerprint, source_identity_hash, reason_code,
                   source_token_count
            FROM lcm_compaction_range_dispositions
            ORDER BY conversation_id, session_id, expected_frontier_store_id,
                     captured_watermark_store_id
            """
        ).fetchall()
        raw_rows = background.connection.execute(
            """
            SELECT store_id, role, content, token_estimate
            FROM messages WHERE session_id='s' ORDER BY store_id
            """
        ).fetchall()
        results.put(
            (
                "ok",
                None if captured is None else captured.job_id,
                active_count,
                total_job_count,
                [tuple(row) for row in disposition_rows],
                [tuple(row) for row in raw_rows],
            )
        )
    except BaseException as exc:  # pragma: no cover - asserted in parent
        results.put(("error", type(exc).__name__, str(exc)[:200]))
    finally:
        if background is not None:
            background.close()


def _schema_start_process(plugin_root, db_path, start_barrier, results):
    _bootstrap_plugin_package(plugin_root)
    from hermes_lcm.background_compaction import BackgroundCompactionStore

    background = None
    try:
        start_barrier.wait(timeout=10)
        background = BackgroundCompactionStore(
            db_path,
            config=_config(db_path),
        )
        results.put(("ok",))
    except BaseException as exc:  # pragma: no cover - asserted in parent
        results.put(("error", type(exc).__name__, str(exc)[:200]))
    finally:
        if background is not None:
            background.close()


def _join_process(process) -> None:
    process.join(timeout=15)
    if process.is_alive():  # pragma: no cover - cleanup for a failed contract
        process.terminate()
        process.join(timeout=5)
    assert process.exitcode == 0


def test_prepared_job_survives_ordinary_rebinds_and_publishes_only_at_fixed_reserve_boundary(
    tmp_path,
):
    from hermes_lcm.background_compaction import background_policy_fingerprint
    from hermes_lcm.engine import LCMEngine

    db_path = tmp_path / "combined-rebind-threshold.db"
    config = _config(db_path, fresh_tail_count=1, leaf_chunk_tokens=1)
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=1_000_000)
    messages = [
        {"role": "user", "content": f"message-{index} " + "x " * 200}
        for index in range(6)
    ]
    try:
        engine.ingest(messages)
        job = engine.run_background_compaction_once(
            summarizer=lambda **_kwargs: ("prepared summary", 1)
        )
        assert job is not None and job.state == "prepared"
        generation = job.session_fence_generation

        for _ in range(3):
            finalized = engine._lifecycle.finalize_session(
                "c",
                "s",
                frontier_store_id=0,
            )
            assert finalized is not None and finalized.current_session_id is None
            rebound = engine._lifecycle.bind_session("s", conversation_id="c")
            assert engine._background_compaction.activate_session(
                conversation_id="c",
                session_id="s",
                expected_lifecycle_bound_at=rebound.current_bound_at,
                expected_lifecycle_reset_at=rebound.last_reset_at,
            ) == generation

        below = engine.promote_prepared_compaction(
            job.job_id,
            messages,
            current_tokens=899_999,
        )
        assert below.promoted is False
        assert below.reason == "publication_trigger_not_reached"
        assert engine._background_compaction.get_job(job.job_id).state == "prepared"

        published = engine.promote_prepared_compaction(
            job.job_id,
            messages,
            current_tokens=900_000,
        )
        assert published.promoted is True
        persisted = engine._background_compaction.get_job(job.job_id)
        assert persisted is not None and persisted.state == "published"
        assert persisted.policy_fingerprint == background_policy_fingerprint(config)
    finally:
        engine.shutdown()


def test_two_process_publish_race_fences_stale_and_duplicate_work(tmp_path):
    from hermes_lcm.background_compaction import (
        BackgroundCompactionStore,
        background_policy_fingerprint,
    )
    from hermes_lcm.lifecycle_state import LifecycleStateStore

    plugin_root = str(Path(__file__).resolve().parent.parent)
    context = multiprocessing.get_context("spawn")

    race_db = tmp_path / "two-process-race.db"
    race_ids = _seed(
        race_db,
        [{"role": "user", "content": f"race-{index}"} for index in range(4)],
        [1_000] * 4,
    )
    race_config = _config(race_db)
    race_background = BackgroundCompactionStore(str(race_db), config=race_config)
    try:
        race_policy = background_policy_fingerprint(race_config)
        race_job = _enqueue_explicit(
            race_background,
            race_ids,
            session="s",
            conversation="c",
            policy=race_policy,
        )
        assert race_job is not None
        _prepare(race_background, race_job)
        lease = race_background.acquire_publication_lease(
            race_job.job_id,
            owner="multiprocess-publisher",
        )
        assert lease is not None

        barrier = context.Barrier(3)
        results = context.Queue()
        processes = [
            context.Process(
                target=_publish_process,
                args=(
                    plugin_root,
                    str(race_db),
                    race_job.job_id,
                    "multiprocess-publisher",
                    lease.lease_generation,
                    race_policy,
                    barrier,
                    results,
                ),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        barrier.wait(timeout=10)
        outcomes = [results.get(timeout=10) for _ in processes]
        for process in processes:
            _join_process(process)
        assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
        assert sum(bool(outcome[1]) for outcome in outcomes) == 1
        assert {outcome[2] for outcome in outcomes} == {
            "published",
            "already_published",
        }
        assert race_background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id='s'"
        ).fetchone()[0] == 1
    finally:
        race_background.close()

    stale_db = tmp_path / "two-process-stale.db"
    stale_ids = _seed(
        stale_db,
        [{"role": "user", "content": f"stale-{index}"} for index in range(4)],
        [1_000] * 4,
    )
    stale_config = _config(stale_db)
    stale_background = BackgroundCompactionStore(str(stale_db), config=stale_config)
    lifecycle = LifecycleStateStore(str(stale_db))
    try:
        stale_policy = background_policy_fingerprint(stale_config)
        stale_job = _enqueue_explicit(
            stale_background,
            stale_ids,
            session="s",
            conversation="c",
            policy=stale_policy,
        )
        assert stale_job is not None
        _prepare(stale_background, stale_job)
        stale_lease = stale_background.acquire_publication_lease(
            stale_job.job_id,
            owner="stale-publisher",
        )
        assert stale_lease is not None

        ready = context.Event()
        start = context.Event()
        results = context.Queue()
        process = context.Process(
            target=_publish_after_event_process,
            args=(
                plugin_root,
                str(stale_db),
                stale_job.job_id,
                "stale-publisher",
                stale_lease.lease_generation,
                stale_policy,
                ready,
                start,
                results,
            ),
        )
        process.start()
        assert ready.wait(timeout=10)
        advanced = lifecycle.advance_frontier("c", "s", stale_ids[0])
        assert advanced is not None
        start.set()
        outcome = results.get(timeout=10)
        _join_process(process)
        assert outcome == ("ok", False, "frontier_mismatch")
        assert stale_background.connection.execute(
            "SELECT COUNT(*) FROM summary_nodes WHERE session_id='s'"
        ).fetchone()[0] == 0
    finally:
        lifecycle.close()
        stale_background.close()


def test_78_token_gap_publishes_once_and_never_requeues(tmp_path):
    from hermes_lcm.background_compaction import (
        BackgroundCompactionStore,
        background_policy_fingerprint,
    )
    from hermes_lcm.dag import SummaryDAG, SummaryNode
    from hermes_lcm.lifecycle_state import LifecycleStateStore

    db_path = tmp_path / "seventy-eight-token-gap.db"
    config = _config(db_path)
    source_ids = _seed(
        db_path,
        [
            {"role": "assistant", "content": "canonical island before gap"},
            {"role": "user", "content": "isolated user correction"},
            {"role": "assistant", "content": "canonical island after gap"},
        ],
        [50, 78, 50],
    )
    dag = SummaryDAG(str(db_path))
    for label, source_id in (
        ("before", source_ids[0]),
        ("after", source_ids[2]),
    ):
        dag.add_node(
            SummaryNode(
                session_id="s",
                depth=0,
                summary=f"existing canonical island {label}",
                source_ids=[source_id],
                source_type="messages",
                source_token_count=50,
                token_count=4,
            )
        )
    dag.close()

    lifecycle = LifecycleStateStore(str(db_path))
    try:
        advanced = lifecycle.advance_frontier("c", "s", source_ids[0])
        assert advanced is not None
        assert advanced.current_frontier_store_id == source_ids[0]
    finally:
        lifecycle.close()

    with sqlite3.connect(db_path) as connection:
        raw_before = connection.execute(
            """
            SELECT store_id, role, content, token_estimate
            FROM messages WHERE session_id='s' ORDER BY store_id
            """
        ).fetchall()

    background = BackgroundCompactionStore(str(db_path), config=config)
    try:
        job = _capture(background, config, frontier=source_ids[0])
        assert job is not None and job.state == "prepared"
        assert job.source_store_ids == (source_ids[1],)
        assert job.source_token_count == 78
        assert job.completion_reason == "insufficient_compressible_source"
        for _ in range(20):
            repeated = _capture(background, config, frontier=source_ids[0])
            assert repeated is not None and repeated.job_id == job.job_id
        assert background.run_next_job(
            owner="provider-must-not-run",
            summarizer=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("tiny gap reached provider")
            ),
        ) is None

        lease = background.acquire_publication_lease(job.job_id, owner="publisher")
        assert lease is not None
        published = background.publish_prepared(
            job.job_id,
            owner="publisher",
            generation=lease.lease_generation,
            policy_fingerprint=background_policy_fingerprint(config),
            observed_active_tokens=900_000,
            effective_context_window=1_000_000,
            publication_trigger_tokens=900_000,
            emergency_trigger_tokens=950_000,
        )
        assert published.promoted is True

        dag = SummaryDAG(str(db_path))
        try:
            assert dag.source_message_ids(published.node_ids[0], limit=10) == [
                source_ids[1]
            ]
        finally:
            dag.close()

        for _ in range(20):
            lifecycle = LifecycleStateStore(str(db_path))
            try:
                current_frontier = lifecycle.get_by_conversation(
                    "c"
                ).current_frontier_store_id
            finally:
                lifecycle.close()
            assert _capture(background, config, frontier=current_frontier) is None
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs WHERE conversation_id='c'"
        ).fetchone()[0] == 1
        assert background.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_source_coverage
            WHERE session_id='s' AND store_id=?
            """,
            (source_ids[1],),
        ).fetchone()[0] == 1
        raw_after = [
            tuple(row)
            for row in background.connection.execute(
                """
                SELECT store_id, role, content, token_estimate
                FROM messages WHERE session_id='s' ORDER BY store_id
                """
            ).fetchall()
        ]
        assert raw_after == raw_before
    finally:
        background.close()

    from hermes_lcm.engine import LCMEngine

    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    engine._session_id = "s"
    engine._conversation_id = "c"
    try:
        expanded = json.loads(
            engine.handle_tool_call(
                "lcm_expand",
                {"node_id": published.node_ids[0], "max_tokens": 100_000},
            )
        )
        assert expanded["pagination"]["total_sources"] == 1
        assert expanded["pagination"]["returned_sources"] == 1
        expanded_row = expanded["expanded"][0]
        assert (
            expanded_row["store_id"],
            expanded_row["session_id"],
            expanded_row["role"],
            expanded_row["content"],
        ) == (source_ids[1], "s", "user", "isolated user correction")
        assert expanded_row["content_truncated"] is False
    finally:
        engine.shutdown()


def test_terminal_range_suppression_survives_restart_in_second_process(tmp_path):
    from hermes_lcm.background_compaction import BackgroundCompactionStore

    plugin_root = str(Path(__file__).resolve().parent.parent)
    context = multiprocessing.get_context("spawn")
    db_path = tmp_path / "second-process-suppression.db"
    config = _config(db_path)
    source_ids = _seed(
        db_path,
        [{"role": "user", "content": "large incompressible source"}],
        [1_000],
    )
    first = BackgroundCompactionStore(str(db_path), config=config)
    try:
        job = _capture(first, config)
        assert job is not None
        failed = first.run_next_job(
            owner="first-process-worker",
            summarizer=lambda **_kwargs: ("not useful", 3),
        )
        assert failed is not None and failed.state == "failed"
        assert failed.completion_reason == "non_compressing_output"
        initial_job_count = first.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_jobs WHERE conversation_id='c'"
        ).fetchone()[0]
        assert initial_job_count == 1
        disposition = first.connection.execute(
            """
            SELECT conversation_id, session_id, session_fence_generation,
                   expected_frontier_store_id, captured_watermark_store_id,
                   policy_fingerprint, source_identity_hash, reason_code,
                   source_token_count
            FROM lcm_compaction_range_dispositions
            """
        ).fetchone()
        assert disposition is not None
        expected_disposition = (
            "c",
            "s",
            job.session_fence_generation,
            job.expected_frontier_store_id,
            job.captured_watermark_store_id,
            job.policy_fingerprint,
            job.source_identity_hash,
            "non_compressing_output",
            job.source_token_count,
        )
        assert tuple(disposition) == expected_disposition
        raw_before = [
            tuple(row)
            for row in first.connection.execute(
                """
                SELECT store_id, role, content, token_estimate
                FROM messages WHERE session_id='s' ORDER BY store_id
                """
            ).fetchall()
        ]
        assert raw_before == [
            (source_ids[0], "user", "large incompressible source", 1_000)
        ]
    finally:
        first.close()

    ready = context.Event()
    start = context.Event()
    results = context.Queue()
    process = context.Process(
        target=_capture_after_restart_process,
        args=(plugin_root, str(db_path), ready, start, results),
    )
    process.start()
    assert ready.wait(timeout=10)
    start.set()
    outcome = results.get(timeout=10)
    _join_process(process)
    assert outcome == (
        "ok",
        None,
        0,
        initial_job_count,
        [expected_disposition],
        raw_before,
    )


def test_changed_frontier_reenables_preparation_after_terminal_disposition(tmp_path):
    from hermes_lcm.background_compaction import BackgroundCompactionStore
    from hermes_lcm.lifecycle_state import LifecycleStateStore

    db_path = tmp_path / "changed-frontier-reenables.db"
    config = _config(db_path)
    source_ids = _seed(
        db_path,
        [{"role": "user", "content": f"source-{index}"} for index in range(3)],
        [1_000, 1_000, 1_000],
    )
    background = BackgroundCompactionStore(str(db_path), config=config)
    lifecycle = LifecycleStateStore(str(db_path))
    try:
        first = _capture(background, config)
        assert first is not None
        failed = background.run_next_job(
            owner="first-worker",
            summarizer=lambda **_kwargs: ("not useful", 3),
        )
        assert failed is not None and failed.state == "failed"
        assert _capture(background, config) is None

        advanced = lifecycle.advance_frontier("c", "s", source_ids[0])
        assert advanced is not None
        replacement = _capture(
            background,
            config,
            frontier=source_ids[0],
        )
        assert replacement is not None
        assert replacement.job_id != first.job_id
        assert replacement.expected_frontier_store_id == source_ids[0]
        assert replacement.source_store_ids == tuple(source_ids[1:])
        prepared = background.run_next_job(
            owner="replacement-worker",
            summarizer=lambda **_kwargs: ("prepared changed-frontier summary", 1),
        )
        assert prepared is not None
        assert prepared.job_id == replacement.job_id
        assert prepared.state == "prepared"
        assert prepared.completion_reason == "prepared"
        assert background.get_prepared_nodes(replacement.job_id)[
            0
        ].source_store_ids == tuple(source_ids[1:])
    finally:
        lifecycle.close()
        background.close()


def test_run_next_job_persists_distinct_timeout_and_non_compressing_reasons(
    tmp_path,
):
    from hermes_lcm.background_compaction import (
        BackgroundCompactionStore,
        background_policy_fingerprint,
    )

    db_path = tmp_path / "persisted-reasons.db"
    timeout_ids = _seed(
        db_path,
        [{"role": "user", "content": "timeout source"}],
        [1_000],
        session="timeout-session",
        conversation="timeout-conversation",
    )
    noncompressing_ids = _seed(
        db_path,
        [{"role": "user", "content": "noncompressing source"}],
        [1_000],
        session="noncompressing-session",
        conversation="noncompressing-conversation",
    )
    config = _config(db_path)
    background = BackgroundCompactionStore(str(db_path), config=config)
    policy = background_policy_fingerprint(config)
    try:
        timeout_job = _enqueue_explicit(
            background,
            timeout_ids,
            session="timeout-session",
            conversation="timeout-conversation",
            policy=policy,
            priority=20,
        )
        noncompressing_job = _enqueue_explicit(
            background,
            noncompressing_ids,
            session="noncompressing-session",
            conversation="noncompressing-conversation",
            policy=policy,
            priority=10,
        )
        assert timeout_job is not None and noncompressing_job is not None

        def timeout_provider(**_kwargs):
            raise TimeoutError("sensitive provider timeout detail")

        timed_out = background.run_next_job(
            owner="timeout-worker",
            summarizer=timeout_provider,
        )
        assert timed_out is not None and timed_out.job_id == timeout_job.job_id
        assert timed_out.state == "retry"
        assert timed_out.completion_reason == "provider_timeout"

        noncompressing = background.run_next_job(
            owner="noncompressing-worker",
            summarizer=lambda **_kwargs: ("not useful", 3),
        )
        assert noncompressing is not None
        assert noncompressing.job_id == noncompressing_job.job_id
        assert noncompressing.state == "failed"
        assert noncompressing.completion_reason == "non_compressing_output"

        persisted = dict(
            background.connection.execute(
                "SELECT job_id, completion_reason FROM lcm_compaction_jobs"
            ).fetchall()
        )
        assert persisted[timeout_job.job_id] == "provider_timeout"
        assert persisted[noncompressing_job.job_id] == "non_compressing_output"
        attempts = {
            row[0]: row[1]
            for row in background.connection.execute(
                """
                SELECT job_id, failure_reason FROM lcm_compaction_attempts
                WHERE job_id IN (?,?)
                """,
                (timeout_job.job_id, noncompressing_job.job_id),
            ).fetchall()
        }
        assert attempts == {
            timeout_job.job_id: "provider_timeout",
            noncompressing_job.job_id: "non_compressing_output",
        }
        assert background.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_provider_leases"
        ).fetchone()[0] == 0
    finally:
        background.close()


def test_range_disposition_and_telemetry_migration_is_idempotent_across_two_process_startup(
    tmp_path,
):
    plugin_root = str(Path(__file__).resolve().parent.parent)
    context = multiprocessing.get_context("spawn")
    db_path = tmp_path / "two-process-schema.db"
    _seed(
        db_path,
        [{"role": "user", "content": "core schema only"}],
        [10],
    )

    barrier = context.Barrier(3)
    results = context.Queue()
    processes = [
        context.Process(
            target=_schema_start_process,
            args=(plugin_root, str(db_path), barrier, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    barrier.wait(timeout=10)
    outcomes = [results.get(timeout=10) for _ in processes]
    for process in processes:
        _join_process(process)
    assert outcomes == [("ok",), ("ok",)]

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "lcm_compaction_range_dispositions",
            "lcm_compaction_attempts",
            "lcm_compaction_publications",
            "lcm_compaction_prepared_nodes",
        } <= tables
        prepared_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(lcm_compaction_prepared_nodes)"
            )
        }
        assert {
            "route_latency_ms",
            "attempt_number",
            "escalation_level",
            "attribution_available",
            "fallback_used",
        } <= prepared_columns
        migration_steps = {
            row[0]
            for row in connection.execute(
                """
                SELECT step_name FROM lcm_migration_state
                WHERE step_name IN (
                    'background_compaction_range_dispositions_v1',
                    'background_compaction_telemetry_v1'
                )
                """
            )
        }
        assert migration_steps == {
            "background_compaction_range_dispositions_v1",
            "background_compaction_telemetry_v1",
        }
