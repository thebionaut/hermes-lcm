"""Database-enforced source-coverage contracts for background publication."""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

import hermes_lcm.background_compaction as background_compaction_module
from hermes_lcm.background_compaction import BackgroundCompactionStore
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def test_lazy_migration_backfills_existing_canonical_message_coverage(tmp_path):
    db_path = tmp_path / "coverage-backfill.db"
    messages = MessageStore(db_path)
    source_ids = messages.append_batch(
        "session",
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ],
        token_estimates=[1, 1],
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    node_id = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy canonical summary",
            source_ids=source_ids,
            source_type="messages",
            token_count=3,
            source_token_count=2,
        )
    )
    dag.close()

    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_min_provider_source_tokens=1,
    )
    background = BackgroundCompactionStore(db_path, config=config)
    try:
        rows = background.connection.execute(
            """
            SELECT store_id, publication_id
            FROM lcm_compaction_source_coverage
            WHERE session_id='session'
            ORDER BY store_id
            """
        ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [
            (source_ids[0], f"legacy-node:{node_id}"),
            (source_ids[1], f"legacy-node:{node_id}"),
        ]
        mapped = background.connection.execute(
            """
            SELECT node_id, session_id, store_id
            FROM lcm_compaction_node_sources
            ORDER BY store_id
            """
        ).fetchall()
        assert [tuple(row) for row in mapped] == [
            (node_id, "session", source_ids[0]),
            (node_id, "session", source_ids[1]),
        ]
        assert background.connection.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name=?",
            (background_compaction_module.BACKGROUND_COMPACTION_NODE_SOURCES_MIGRATION_STEP,),
        ).fetchone() is not None
        assert background._source_ids_already_canonical(
            background.connection,
            "session",
            source_ids,
        ) is True
    finally:
        background.close()

def test_coverage_trigger_rejects_any_future_overlapping_message_node(tmp_path):
    db_path = tmp_path / "coverage-trigger.db"
    messages = MessageStore(db_path)
    source_id = messages.append_batch(
        "session",
        [{"role": "user", "content": "raw"}],
        token_estimates=[1],
        conversation_id="conversation",
    )[0]
    messages.close()
    dag = SummaryDAG(db_path)
    dag.close()
    background = BackgroundCompactionStore(
        db_path,
        config=LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        ),
    )
    background.close()

    dag = SummaryDAG(db_path)
    try:
        first = SummaryNode(
            session_id="session",
            summary="first",
            source_ids=[source_id],
            source_type="messages",
        )
        dag.add_node(first)
        with pytest.raises(sqlite3.IntegrityError):
            dag.add_node(
                SummaryNode(
                    session_id="session",
                    summary="duplicate",
                    source_ids=[source_id],
                    source_type="messages",
                )
            )
    finally:
        dag.close()


def test_coverage_delete_trigger_releases_source_after_reassigned_node_reset(tmp_path):
    db_path = tmp_path / "coverage-delete.db"
    messages = MessageStore(db_path)
    source_id = messages.append_batch(
        "source-session",
        [{"role": "user", "content": "raw"}],
        token_estimates=[1],
        conversation_id="conversation",
    )[0]
    messages.close()
    dag = SummaryDAG(db_path)
    dag.close()
    background = BackgroundCompactionStore(
        db_path,
        config=LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        ),
    )
    background.close()

    dag = SummaryDAG(db_path)
    try:
        node_id = dag.add_node(
            SummaryNode(
                session_id="source-session",
                summary="first",
                source_ids=[source_id],
                source_type="messages",
            )
        )
        assert dag.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_source_coverage
            WHERE session_id='source-session' AND store_id=?
            """,
            (source_id,),
        ).fetchone()[0] == 1
        assert dag.reassign_session_nodes("source-session", "carried-session") == 1
        moved_coverage = dag.connection.execute(
            """
            SELECT session_id, store_id, publication_id
            FROM lcm_compaction_source_coverage
            """
        ).fetchall()
        assert [tuple(row) for row in moved_coverage] == [
            ("carried-session", source_id, f"summary-node:{node_id}")
        ]
        assert dag.delete_session_nodes("carried-session") == 1
        coverage_count = dag.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_source_coverage
            WHERE publication_id=?
            """,
            (f"summary-node:{node_id}",),
        ).fetchone()[0]
        assert coverage_count == 0

        replacement_id = dag.add_node(
            SummaryNode(
                session_id="source-session",
                summary="replacement after reset",
                source_ids=[source_id],
                source_type="messages",
            )
        )
        assert replacement_id != node_id
    finally:
        dag.close()


def test_deleting_legacy_duplicate_rehomes_coverage_to_surviving_node(tmp_path):
    db_path = tmp_path / "coverage-duplicate-delete.db"
    messages = MessageStore(db_path)
    source_id = messages.append_batch(
        "session",
        [{"role": "user", "content": "raw"}],
        token_estimates=[1],
        conversation_id="conversation",
    )[0]
    messages.close()
    dag = SummaryDAG(db_path)
    first_id = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy duplicate one",
            source_ids=[source_id],
            source_type="messages",
        )
    )
    second_id = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy duplicate two",
            source_ids=[source_id],
            source_type="messages",
        )
    )
    dag.close()

    background = BackgroundCompactionStore(
        db_path,
        config=LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        ),
    )
    background.close()

    dag = SummaryDAG(db_path)
    try:
        elected = dag.connection.execute(
            """
            SELECT publication_id FROM lcm_compaction_source_coverage
            WHERE session_id='session' AND store_id=?
            """,
            (source_id,),
        ).fetchone()[0]
        assert elected == f"legacy-node:{first_id}"

        assert dag.reassign_session_nodes("session", "carried-session") == 2
        moved = dag.connection.execute(
            """
            SELECT session_id, publication_id
            FROM lcm_compaction_source_coverage WHERE store_id=?
            """,
            (source_id,),
        ).fetchone()
        assert tuple(moved) == (
            "carried-session",
            f"summary-node:{first_id}",
        )

        dag.connection.execute(
            "DELETE FROM summary_nodes WHERE node_id=?",
            (first_id,),
        )
        dag.connection.commit()
        survivor = dag.connection.execute(
            """
            SELECT publication_id FROM lcm_compaction_source_coverage
            WHERE session_id='carried-session' AND store_id=?
            """,
            (source_id,),
        ).fetchone()[0]
        assert survivor == f"summary-node:{second_id}"
        with pytest.raises(sqlite3.IntegrityError):
            dag.add_node(
                SummaryNode(
                    session_id="carried-session",
                    summary="overlap remains fenced",
                    source_ids=[source_id],
                    source_type="messages",
                )
            )

        dag.connection.execute(
            "DELETE FROM summary_nodes WHERE node_id=?",
            (second_id,),
        )
        dag.connection.commit()
        assert dag.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_source_coverage"
        ).fetchone()[0] == 0
        dag.add_node(
            SummaryNode(
                session_id="carried-session",
                summary="valid after every duplicate is gone",
                source_ids=[source_id],
                source_type="messages",
            )
        )
    finally:
        dag.close()


def test_coverage_update_tracks_source_ids_and_source_type(tmp_path):
    db_path = tmp_path / "coverage-update.db"
    messages = MessageStore(db_path)
    first_id, second_id = messages.append_batch(
        "session",
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
        token_estimates=[1, 1],
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    dag.close()
    background = BackgroundCompactionStore(
        db_path,
        config=LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        ),
    )
    background.close()

    dag = SummaryDAG(db_path)
    try:
        node_id = dag.add_node(
            SummaryNode(
                session_id="session",
                summary="mutable lineage",
                source_ids=[first_id],
                source_type="messages",
            )
        )
        dag.connection.execute(
            "UPDATE summary_nodes SET source_ids=? WHERE node_id=?",
            (json.dumps([second_id]), node_id),
        )
        dag.connection.commit()
        assert [
            tuple(row)
            for row in dag.connection.execute(
                """
                SELECT session_id, store_id FROM lcm_compaction_source_coverage
                ORDER BY store_id
                """
            ).fetchall()
        ] == [("session", second_id)]
        assert [
            tuple(row)
            for row in dag.connection.execute(
                """
                SELECT session_id, store_id FROM lcm_compaction_node_sources
                WHERE node_id=?
                """,
                (node_id,),
            ).fetchall()
        ] == [("session", second_id)]

        dag.connection.execute(
            "UPDATE summary_nodes SET source_type='nodes' WHERE node_id=?",
            (node_id,),
        )
        dag.connection.commit()
        assert dag.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_source_coverage"
        ).fetchone()[0] == 0
        assert dag.connection.execute(
            "SELECT COUNT(*) FROM lcm_compaction_node_sources WHERE node_id=?",
            (node_id,),
        ).fetchone()[0] == 0

        dag.connection.execute(
            "UPDATE summary_nodes SET source_type='messages' WHERE node_id=?",
            (node_id,),
        )
        dag.connection.commit()
        assert dag.connection.execute(
            """
            SELECT store_id FROM lcm_compaction_source_coverage
            WHERE publication_id=?
            """,
            (f"summary-node:{node_id}",),
        ).fetchone()[0] == second_id
        assert dag.connection.execute(
            """
            SELECT store_id FROM lcm_compaction_node_sources WHERE node_id=?
            """,
            (node_id,),
        ).fetchone()[0] == second_id
    finally:
        dag.close()


def test_coverage_update_rejects_new_overlap_and_rolls_back_old_coverage(tmp_path):
    db_path = tmp_path / "coverage-update-overlap.db"
    messages = MessageStore(db_path)
    first_id, second_id = messages.append_batch(
        "session",
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
        token_estimates=[1, 1],
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    dag.close()
    background = BackgroundCompactionStore(
        db_path,
        config=LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        ),
    )
    background.close()

    dag = SummaryDAG(db_path)
    try:
        dag.add_node(
            SummaryNode(
                session_id="session",
                summary="first owner",
                source_ids=[first_id],
                source_type="messages",
            )
        )
        second_node_id = dag.add_node(
            SummaryNode(
                session_id="session",
                summary="second owner",
                source_ids=[second_id],
                source_type="messages",
            )
        )
        with pytest.raises(sqlite3.IntegrityError):
            dag.connection.execute(
                "UPDATE summary_nodes SET source_ids=? WHERE node_id=?",
                (json.dumps([first_id]), second_node_id),
            )
        dag.connection.rollback()
        assert json.loads(
            dag.connection.execute(
                "SELECT source_ids FROM summary_nodes WHERE node_id=?",
                (second_node_id,),
            ).fetchone()[0]
        ) == [second_id]
        assert {
            int(row[0])
            for row in dag.connection.execute(
                "SELECT store_id FROM lcm_compaction_source_coverage"
            ).fetchall()
        } == {first_id, second_id}
    finally:
        dag.close()


def test_backfill_skips_stale_metadata_after_concurrent_reassign(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "coverage-backfill-reassign-race.db"
    messages = MessageStore(db_path)
    old_source_id, new_source_id = messages.append_batch(
        "old-session",
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "new"},
        ],
        token_estimates=[1, 1],
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    node_id = dag.add_node(
        SummaryNode(
            session_id="old-session",
            summary="moves during backfill",
            source_ids=[old_source_id],
            source_type="messages",
        )
    )
    dag.close()

    original_load_ids = background_compaction_module._load_ids
    changed = False

    def reassign_after_snapshot(value):
        nonlocal changed
        result = original_load_ids(value)
        if not changed:
            changed = True
            concurrent = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                concurrent.execute(
                    """
                    UPDATE summary_nodes
                    SET session_id='new-session', source_ids=?
                    WHERE node_id=?
                    """,
                    (json.dumps([new_source_id]), node_id),
                )
                concurrent.commit()
            finally:
                concurrent.close()
        return result

    monkeypatch.setattr(
        background_compaction_module,
        "_load_ids",
        reassign_after_snapshot,
    )
    background = BackgroundCompactionStore(
        db_path,
        config=LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        ),
    )
    try:
        assert changed is True
        rows = background.connection.execute(
            """
            SELECT session_id, store_id FROM lcm_compaction_source_coverage
            ORDER BY session_id, store_id
            """
        ).fetchall()
        assert [tuple(row) for row in rows] == [("new-session", new_source_id)]
    finally:
        background.close()


def test_existing_coverage_marker_upgrades_old_delete_only_triggers(tmp_path):
    db_path = tmp_path / "coverage-trigger-upgrade.db"
    messages = MessageStore(db_path)
    source_id = messages.append_batch(
        "session",
        [{"role": "user", "content": "legacy"}],
        token_estimates=[1],
        conversation_id="conversation",
    )[0]
    messages.close()
    dag = SummaryDAG(db_path)
    node_id = dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy node",
            source_ids=[source_id],
            source_type="messages",
        )
    )
    dag.close()
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    background = BackgroundCompactionStore(db_path, config=config)
    background.close()

    connection = sqlite3.connect(str(db_path))
    try:
        connection.executescript(
            """
            DELETE FROM lcm_migration_state
            WHERE step_name='background_compaction_node_sources_v1';
            DELETE FROM lcm_compaction_node_sources;
            DROP TRIGGER lcm_compaction_summary_coverage_delete;
            DROP TRIGGER lcm_compaction_summary_coverage_update_before;
            DROP TRIGGER lcm_compaction_summary_coverage_update_after;
            CREATE TRIGGER lcm_compaction_summary_coverage_delete
            AFTER DELETE ON summary_nodes
            WHEN OLD.source_type='messages'
            BEGIN
                DELETE FROM lcm_compaction_source_coverage
                WHERE publication_id IN (
                    'summary-node:' || OLD.node_id,
                    'legacy-node:' || OLD.node_id
                );
            END;
            """
        )
        connection.commit()
    finally:
        connection.close()

    upgraded = BackgroundCompactionStore(db_path, config=config)
    try:
        trigger_rows = upgraded.connection.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type='trigger'
              AND name LIKE 'lcm_compaction_summary_coverage_%'
            """
        ).fetchall()
        trigger_sql = {str(row[0]): str(row[1]) for row in trigger_rows}
        assert set(trigger_sql) == {
            "lcm_compaction_summary_coverage_insert",
            "lcm_compaction_summary_coverage_delete",
            "lcm_compaction_summary_coverage_update_before",
            "lcm_compaction_summary_coverage_update_after",
        }
        lifecycle_sql = "\n".join(
            (
                trigger_sql["lcm_compaction_summary_coverage_delete"],
                trigger_sql["lcm_compaction_summary_coverage_update_before"],
            )
        )
        assert "lcm_compaction_node_sources AS survivor" in lifecycle_sql
        assert "summary_nodes AS candidate" not in lifecycle_sql
        assert "json_each" not in lifecycle_sql
        mapped = upgraded.connection.execute(
            """
            SELECT session_id, store_id FROM lcm_compaction_node_sources
            WHERE node_id=?
            """,
            (node_id,),
        ).fetchone()
        assert tuple(mapped) == ("session", source_id)
        assert upgraded.connection.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name=?",
            (background_compaction_module.BACKGROUND_COMPACTION_NODE_SOURCES_MIGRATION_STEP,),
        ).fetchone() is not None
    finally:
        upgraded.close()


def test_indexed_rehome_plan_never_scans_summary_nodes(tmp_path):
    db_path = tmp_path / "coverage-indexed-plan.db"
    dag = SummaryDAG(db_path)
    dag.close()
    background = BackgroundCompactionStore(
        db_path,
        config=LCMConfig(
            database_path=str(db_path),
            background_compaction_enabled=True,
            background_compaction_workers=0,
        ),
    )
    try:
        lifecycle_trigger_sql = "\n".join(
            str(row[0])
            for row in background.connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type='trigger' AND name IN (
                    'lcm_compaction_summary_coverage_delete',
                    'lcm_compaction_summary_coverage_update_before'
                )
                ORDER BY name
                """
            ).fetchall()
        )
        assert "lcm_compaction_node_sources AS survivor" in lifecycle_trigger_sql
        assert "FROM summary_nodes" not in lifecycle_trigger_sql
        assert "json_each" not in lifecycle_trigger_sql

        plan = background.connection.execute(
            """
            EXPLAIN QUERY PLAN
            UPDATE lcm_compaction_source_coverage
            SET publication_id=COALESCE(
                    (
                        SELECT 'summary-node:' || survivor.node_id
                        FROM lcm_compaction_node_sources AS survivor
                        WHERE survivor.session_id=
                                  lcm_compaction_source_coverage.session_id
                          AND survivor.store_id=
                                  lcm_compaction_source_coverage.store_id
                        ORDER BY survivor.node_id LIMIT 1
                    ),
                    publication_id
                )
            WHERE publication_id IN (?,?)
            """,
            ("summary-node:1", "legacy-node:1"),
        ).fetchall()
        detail = "\n".join(str(row[3]) for row in plan)
        assert "idx_lcm_compaction_coverage_publication" in detail
        assert "idx_lcm_compaction_node_sources_lookup" in detail
        assert "summary_nodes" not in detail
        map_delete_plan = background.connection.execute(
            """
            EXPLAIN QUERY PLAN
            DELETE FROM lcm_compaction_node_sources WHERE node_id=?
            """,
            (1,),
        ).fetchall()
        map_delete_detail = "\n".join(str(row[3]) for row in map_delete_plan)
        assert "PRIMARY KEY (node_id=?)" in map_delete_detail
        assert "SCAN" not in map_delete_detail
    finally:
        background.close()


def test_foreground_publication_does_not_absorb_reused_session_other_conversation(tmp_path):
    db_path = tmp_path / "conversation-scope.db"
    messages = MessageStore(db_path)
    first = messages.append_batch(
        "shared-session",
        [{"role": "user", "content": "first-a"}],
        token_estimates=[1],
        conversation_id="conversation-a",
    )[0]
    foreign = messages.append_batch(
        "shared-session",
        [{"role": "user", "content": "foreign-b"}],
        token_estimates=[1],
        conversation_id="conversation-b",
    )[0]
    last = messages.append_batch(
        "shared-session",
        [{"role": "assistant", "content": "last-a"}],
        token_estimates=[1],
        conversation_id="conversation-a",
    )[0]
    messages.close()
    dag = SummaryDAG(db_path)
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    bound_state = lifecycle.bind_session(
        "shared-session", conversation_id="conversation-a"
    )
    lifecycle.close()

    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    background = BackgroundCompactionStore(db_path, config=config)
    try:
        session_fence_generation = background.activate_session(
            conversation_id="conversation-a",
            session_id="shared-session",
            expected_lifecycle_bound_at=bound_state.current_bound_at,
        )
        assert session_fence_generation is not None
        source_snapshot = background.capture_foreground_source_snapshot(
            conversation_id="conversation-a",
            session_id="shared-session",
            expected_frontier_store_id=0,
            captured_watermark_store_id=last,
            expected_source_store_ids=[first, last],
        )
        assert source_snapshot is not None
        source_identity = background.foreground_source_identity_hash(
            source_snapshot,
            [first, last],
        )
        assert source_identity
        result = background.publish_foreground_node(
            SummaryNode(
                session_id="shared-session",
                summary="only conversation a",
                source_ids=[first, last],
                source_type="messages",
                token_count=3,
                source_token_count=2,
            ),
            conversation_id="conversation-a",
            session_id="shared-session",
            expected_session_fence_generation=session_fence_generation,
            expected_frontier_store_id=0,
            captured_watermark_store_id=last,
            expected_source_identity_hash=source_identity,
            debt_tokens=2,
        )
        assert result.promoted is True
        canonical = background.connection.execute(
            "SELECT source_ids FROM summary_nodes WHERE node_id=?",
            (result.node_ids[0],),
        ).fetchone()[0]
        assert json.loads(canonical) == [first, last]
        assert foreign not in json.loads(canonical)
        assert background.connection.execute(
            "SELECT content FROM messages WHERE store_id=?",
            (foreign,),
        ).fetchone()[0] == "foreign-b"
    finally:
        background.close()


def _legacy_prefix_fixture(
    db_path,
    *,
    message_count: int,
    covered_ranges: list[list[int]],
):
    messages = MessageStore(db_path)
    source_ids = messages.append_batch(
        "session",
        [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"raw-{index}",
            }
            for index in range(message_count)
        ],
        token_estimates=[10] * message_count,
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    node_ids = []
    for offsets in covered_ranges:
        node_ids.append(
            dag.add_node(
                SummaryNode(
                    session_id="session",
                    summary=f"legacy-{len(node_ids)}",
                    source_ids=[source_ids[index] for index in offsets],
                    source_type="messages",
                    source_token_count=10 * len(offsets),
                    token_count=2,
                )
            )
        )
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    state = lifecycle.bind_session("session", conversation_id="conversation")
    lifecycle.close()
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_min_provider_source_tokens=1,
    )
    background = BackgroundCompactionStore(db_path, config=config)
    generation = background.activate_session(
        conversation_id="conversation",
        session_id="session",
        expected_lifecycle_bound_at=state.current_bound_at,
        expected_lifecycle_reset_at=state.last_reset_at,
    )
    assert generation is not None
    return background, config, source_ids, node_ids, generation


def test_adopts_only_full_canonical_prefix_and_supersedes_stale_job(tmp_path):
    db_path = tmp_path / "adopt-legacy-prefix.db"
    background, config, source_ids, node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=5,
        covered_ranges=[[0, 1, 2]],
    )
    try:
        job = background.enqueue_or_coalesce(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            captured_watermark_store_id=source_ids[3],
            source_store_ids=source_ids[:4],
            source_token_count=40,
            policy_fingerprint=background_compaction_module.background_policy_fingerprint(
                config
            ),
            # Simulate a job captured by the pre-reconciliation release.
            reject_canonical_coverage=False,
        )
        adopted = background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
        )
        assert adopted == source_ids[2]
        lifecycle = LifecycleStateStore(db_path)
        try:
            state = lifecycle.get_by_conversation("conversation")
            assert state is not None
            assert state.current_frontier_store_id == source_ids[2]
            assert state.debt_size_estimate == 10
        finally:
            lifecycle.close()
        assert background.get_job(job.job_id).state == "superseded"
        assert background.get_job(job.job_id).completion_reason == "canonical_prefix_adopted"
        assert MessageStore(db_path).get_session_count("session") == 5
        dag = SummaryDAG(db_path)
        try:
            assert [node.node_id for node in dag.get_session_nodes("session")] == node_ids
        finally:
            dag.close()
    finally:
        background.close()


def test_canonical_prefix_adoption_stops_at_first_uncovered_gap(tmp_path):
    db_path = tmp_path / "adopt-gap.db"
    background, _config, source_ids, _node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=5,
        covered_ranges=[[0, 1], [3]],
    )
    try:
        first = background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
        )
        assert first == source_ids[1]
        second = background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=source_ids[1],
            expected_session_fence_generation=generation,
        )
        assert second == source_ids[1]
        lifecycle = LifecycleStateStore(db_path)
        try:
            assert (
                lifecycle.get_by_conversation(
                    "conversation"
                ).current_frontier_store_id
                == source_ids[1]
            )
        finally:
            lifecycle.close()
    finally:
        background.close()


def test_canonical_prefix_adoption_is_noop_without_coverage(tmp_path):
    db_path = tmp_path / "adopt-no-coverage.db"
    background, _config, source_ids, _node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=3,
        covered_ranges=[],
    )
    try:
        assert background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
        ) == 0
        assert MessageStore(db_path).get(source_ids[0])["content"] == "raw-0"
    finally:
        background.close()


def test_canonical_prefix_adoption_requires_exact_fence_and_session_cas(tmp_path):
    db_path = tmp_path / "adopt-fence-cas.db"
    background, _config, source_ids, _node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=3,
        covered_ranges=[[0, 1]],
    )
    try:
        assert background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation + 1,
        ) == 0
        lifecycle = LifecycleStateStore(db_path)
        try:
            lifecycle.bind_session("replacement", conversation_id="conversation")
        finally:
            lifecycle.close()
        assert background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
        ) == 0
        lifecycle = LifecycleStateStore(db_path)
        try:
            state = lifecycle.get_by_conversation("conversation")
            assert state.current_session_id == "replacement"
            assert state.current_frontier_store_id == 0
        finally:
            lifecycle.close()
        assert MessageStore(db_path).get(source_ids[0])["content"] == "raw-0"
    finally:
        background.close()


def test_two_connections_cannot_double_adopt_same_canonical_prefix(tmp_path):
    db_path = tmp_path / "adopt-two-connections.db"
    first, config, source_ids, _node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=4,
        covered_ranges=[[0, 1]],
    )
    second = BackgroundCompactionStore(db_path, config=config)
    barrier = threading.Barrier(3)
    results: list[int] = []
    errors: list[BaseException] = []

    def adopt(store: BackgroundCompactionStore) -> None:
        try:
            barrier.wait()
            results.append(
                store.adopt_contiguous_canonical_prefix(
                    conversation_id="conversation",
                    session_id="session",
                    expected_frontier_store_id=0,
                    expected_session_fence_generation=generation,
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    threads = [
        threading.Thread(target=adopt, args=(first,)),
        threading.Thread(target=adopt, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
    try:
        assert errors == []
        assert sorted(results) == [0, source_ids[1]]
        lifecycle = LifecycleStateStore(db_path)
        try:
            assert (
                lifecycle.get_by_conversation(
                    "conversation"
                ).current_frontier_store_id
                == source_ids[1]
            )
        finally:
            lifecycle.close()
    finally:
        first.close()
        second.close()


def test_canonical_prefix_adoption_accepts_elected_duplicate_owner(tmp_path):
    db_path = tmp_path / "adopt-duplicate-owner.db"
    background, _config, source_ids, node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=3,
        covered_ranges=[[0, 1], [0, 1]],
    )
    try:
        owner = background.connection.execute(
            """
            SELECT publication_id FROM lcm_compaction_source_coverage
            WHERE session_id='session' AND store_id=?
            """,
            (source_ids[0],),
        ).fetchone()[0]
        assert owner == f"legacy-node:{node_ids[0]}"
        assert background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
        ) == source_ids[1]
    finally:
        background.close()


def test_canonical_prefix_adoption_is_bounded_and_uses_normalized_maps(tmp_path):
    db_path = tmp_path / "adopt-bounded.db"
    message_count = 520
    background, _config, source_ids, _node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=message_count,
        covered_ranges=[list(range(message_count))],
    )
    statements: list[str] = []
    background.connection.set_trace_callback(statements.append)
    try:
        adopted = background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
            max_rows=10_000,
        )
    finally:
        background.connection.set_trace_callback(None)
    try:
        assert adopted == source_ids[511]
        coverage_reads = [
            statement
            for statement in statements
            if "LEFT JOIN lcm_compaction_source_coverage AS coverage" in statement
        ]
        assert len(coverage_reads) == 1
        normalized_read = coverage_reads[0]
        assert "lcm_compaction_node_sources AS owner" in normalized_read
        assert "summary_nodes" not in normalized_read
        assert "json_each" not in normalized_read
        assert "LIMIT 513" in normalized_read
        assert MessageStore(db_path).get_session_count("session") == message_count
    finally:
        background.close()


def test_canonical_prefix_adoption_does_not_split_partial_tool_group(tmp_path):
    db_path = tmp_path / "adopt-tool-gap.db"
    messages = MessageStore(db_path)
    source_ids = messages.append_batch(
        "session",
        [
            {"role": "user", "content": "covered user"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "uncovered"},
            {"role": "user", "content": "tail"},
        ],
        token_estimates=[10, 10, 10, 10],
        conversation_id="conversation",
    )
    messages.close()
    dag = SummaryDAG(db_path)
    dag.add_node(
        SummaryNode(
            session_id="session",
            summary="legacy partial tool group",
            source_ids=source_ids[:2],
            source_type="messages",
        )
    )
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    state = lifecycle.bind_session("session", conversation_id="conversation")
    lifecycle.close()
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    background = BackgroundCompactionStore(db_path, config=config)
    generation = background.activate_session(
        conversation_id="conversation",
        session_id="session",
        expected_lifecycle_bound_at=state.current_bound_at,
    )
    try:
        assert generation is not None
        assert background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
        ) == source_ids[0]
    finally:
        background.close()


def _prepare_and_publish_captured_job(
    background: BackgroundCompactionStore,
    job,
    *,
    policy_fingerprint: str,
    suffix: str,
):
    worker = f"worker-{suffix}"
    claimed = background.claim_next_job(owner=worker)
    assert claimed is not None
    assert claimed.job_id == job.job_id
    prepared = background.mark_prepared(
        job.job_id,
        owner=worker,
        generation=claimed.lease_generation,
        prepared_nodes=[
            {
                "ordinal": 0,
                "summary": f"prepared-{suffix}",
                "source_store_ids": list(job.source_store_ids),
                "source_token_count": job.source_token_count,
                "summary_token_count": 2,
            }
        ],
    )
    assert prepared is not False
    publisher = f"publisher-{suffix}"
    leased = background.acquire_publication_lease(job.job_id, owner=publisher)
    assert leased is not None
    result = background.publish_prepared(
        job.job_id,
        owner=publisher,
        generation=leased.lease_generation,
        policy_fingerprint=policy_fingerprint,
    )
    assert result.promoted is True, result.reason
    return result


def test_mixed_legacy_coverage_island_eventually_publishes_and_advances(tmp_path):
    """[covered 1,2][gap 3][covered 4][gap 5] cannot reject forever."""
    db_path = tmp_path / "mixed-coverage-liveness.db"
    background, config, source_ids, _node_ids, generation = _legacy_prefix_fixture(
        db_path,
        message_count=5,
        covered_ranges=[[0, 1], [3]],
    )
    config.fresh_tail_count = 0
    policy = background_compaction_module.background_policy_fingerprint(config)
    try:
        frontier = background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            expected_session_fence_generation=generation,
        )
        assert frontier == source_ids[1]

        first = background.capture_and_enqueue(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=frontier,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )
        assert first is not None
        assert first.source_store_ids == (source_ids[2],)
        _prepare_and_publish_captured_job(
            background,
            first,
            policy_fingerprint=policy,
            suffix="first-gap",
        )

        frontier = background.adopt_contiguous_canonical_prefix(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=source_ids[2],
            expected_session_fence_generation=generation,
        )
        assert frontier == source_ids[3]
        second = background.capture_and_enqueue(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=frontier,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )
        assert second is not None
        assert second.source_store_ids == (source_ids[4],)
        _prepare_and_publish_captured_job(
            background,
            second,
            policy_fingerprint=policy,
            suffix="second-gap",
        )

        lifecycle = LifecycleStateStore(db_path)
        try:
            assert (
                lifecycle.get_by_conversation(
                    "conversation"
                ).current_frontier_store_id
                == source_ids[-1]
            )
        finally:
            lifecycle.close()
        assert background.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_jobs
            WHERE completion_reason='source_range_already_published'
            """
        ).fetchone()[0] == 0
    finally:
        background.close()


def test_enqueue_rechecks_coverage_after_capture_cutoff(tmp_path, monkeypatch):
    """A DAG insert between capture read and enqueue cannot queue overlap."""
    db_path = tmp_path / "coverage-capture-enqueue-race.db"
    background, config, source_ids, _node_ids, _generation = _legacy_prefix_fixture(
        db_path,
        message_count=5,
        covered_ranges=[],
    )
    config.fresh_tail_count = 0
    policy = background_compaction_module.background_policy_fingerprint(config)
    original_enqueue = background.enqueue_or_coalesce
    inserted = False

    def insert_canonical_then_enqueue(**kwargs):
        nonlocal inserted
        if not inserted:
            inserted = True
            dag = SummaryDAG(db_path)
            try:
                dag.add_node(
                    SummaryNode(
                        session_id="session",
                        summary="concurrent canonical island",
                        source_ids=[source_ids[-1]],
                        source_type="messages",
                    )
                )
            finally:
                dag.close()
        return original_enqueue(**kwargs)

    monkeypatch.setattr(
        background,
        "enqueue_or_coalesce",
        insert_canonical_then_enqueue,
    )
    try:
        raced = background.capture_and_enqueue(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )
        assert inserted is True
        assert raced is None
        assert background.claim_next_job(owner="must-not-run") is None

        monkeypatch.setattr(background, "enqueue_or_coalesce", original_enqueue)
        retry = background.capture_and_enqueue(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )
        assert retry is not None
        assert retry.source_store_ids == tuple(source_ids[:-1])
    finally:
        background.close()


def test_capture_supersedes_stale_queued_job_that_crosses_covered_island(tmp_path):
    db_path = tmp_path / "stale-queued-covered-island.db"
    background, config, source_ids, _node_ids, _generation = _legacy_prefix_fixture(
        db_path,
        message_count=5,
        covered_ranges=[],
    )
    config.fresh_tail_count = 0
    policy = background_compaction_module.background_policy_fingerprint(config)
    try:
        stale = background.enqueue_or_coalesce(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            captured_watermark_store_id=source_ids[-1],
            source_store_ids=source_ids,
            source_token_count=50,
            policy_fingerprint=policy,
        )
        assert stale is not None
        dag = SummaryDAG(db_path)
        try:
            dag.add_node(
                SummaryNode(
                    session_id="session",
                    summary="canonical island after old capture",
                    source_ids=[source_ids[-1]],
                    source_type="messages",
                )
            )
        finally:
            dag.close()

        replacement = background.capture_and_enqueue(
            conversation_id="conversation",
            session_id="session",
            expected_frontier_store_id=0,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=policy,
        )
        assert replacement is not None
        assert replacement.job_id != stale.job_id
        assert replacement.source_store_ids == tuple(source_ids[:-1])
        stale_after = background.get_job(stale.job_id)
        assert stale_after is not None
        assert stale_after.state == "superseded"
        assert stale_after.completion_reason == "source_range_already_published"
    finally:
        background.close()
