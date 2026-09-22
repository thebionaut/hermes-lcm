"""Exact active-summary frontier selection regressions."""

import json

from hermes_lcm.dag import SummaryDAG, SummaryNode


def _insert_nodes(dag: SummaryDAG, rows: list[tuple]) -> None:
    dag.connection.executemany(
        """
        INSERT INTO summary_nodes(
            session_id, depth, summary, token_count, source_token_count,
            source_ids, source_type, created_at, earliest_at, latest_at,
            expand_hint
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    dag.connection.commit()


def _row(
    session_id: str,
    depth: int,
    summary: str,
    *,
    source_ids: list[int] | None = None,
    source_type: str = "messages",
    created_at: float = 1.0,
) -> tuple:
    return (
        session_id,
        depth,
        summary,
        1,
        1,
        json.dumps(source_ids or []),
        source_type,
        created_at,
        created_at,
        created_at,
        "",
    )


def test_active_frontier_json_fallback_is_uncapped_and_finds_late_parent(tmp_path):
    dag = SummaryDAG(tmp_path / "frontier-fallback.db")
    try:
        _insert_nodes(
            dag,
            [
                _row("session-a", 0, f"leaf-{index}", created_at=float(index))
                for index in range(1_001)
            ],
        )
        last_leaf_id = dag.connection.execute(
            "SELECT node_id FROM summary_nodes WHERE summary='leaf-1000'"
        ).fetchone()[0]
        parent_id = dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=1,
                summary="late parent",
                token_count=1,
                source_token_count=1,
                source_ids=[last_leaf_id],
                source_type="nodes",
                created_at=2_000.0,
            )
        )

        frontier = dag.get_active_frontier_nodes("session-a")

        assert len(frontier) == 1_001
        assert frontier[0].node_id == parent_id
        assert last_leaf_id not in {node.node_id for node in frontier}
        assert [node.depth for node in frontier] == [1] + ([0] * 1_000)
    finally:
        dag.close()


def test_active_frontier_json_fallback_returns_more_than_100_roots(tmp_path):
    dag = SummaryDAG(tmp_path / "frontier-many-roots.db")
    try:
        _insert_nodes(
            dag,
            [
                _row("session-a", 0, f"root-{index}", created_at=float(index))
                for index in range(137)
            ],
        )

        frontier = dag.get_active_frontier_nodes("session-a")

        assert len(frontier) == 137
        assert [node.summary for node in frontier] == [
            f"root-{index}" for index in range(137)
        ]
    finally:
        dag.close()


def test_active_frontier_falls_back_when_optional_edge_schema_is_absent(tmp_path):
    dag = SummaryDAG(tmp_path / "frontier-no-edges.db")
    try:
        leaf_id = dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=0,
                summary="covered leaf",
                source_ids=[],
                source_type="messages",
                created_at=1.0,
            )
        )
        parent_id = dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=1,
                summary="active parent",
                source_ids=[leaf_id],
                source_type="nodes",
                created_at=2.0,
            )
        )

        assert dag.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='lcm_compaction_node_edges'"
        ).fetchone() is None
        assert [node.node_id for node in dag.get_active_frontier_nodes("session-a")] == [
            parent_id
        ]
    finally:
        dag.close()


def test_active_frontier_uses_completed_normalized_reverse_edges(tmp_path):
    dag = SummaryDAG(tmp_path / "frontier-indexed.db")
    try:
        _insert_nodes(
            dag,
            [
                _row("session-a", 0, "child", created_at=1.0),
                _row("session-a", 0, "root same time a", created_at=1.0),
                _row("session-a", 0, "root same time b", created_at=1.0),
                _row("session-a", 0, "later root", created_at=2.0),
            ],
        )
        child_id, same_time_root_a_id, same_time_root_b_id, later_root_id = [
            row[0]
            for row in dag.connection.execute(
                "SELECT node_id FROM summary_nodes ORDER BY node_id"
            ).fetchall()
        ]
        parent_id = dag.add_node(
            SummaryNode(
                session_id="session-a",
                depth=2,
                summary="indexed parent",
                source_ids=[child_id],
                source_type="nodes",
                created_at=3.0,
            )
        )
        other_parent_id = dag.add_node(
            SummaryNode(
                session_id="session-b",
                depth=3,
                summary="other-session parent",
                source_ids=[same_time_root_a_id],
                source_type="nodes",
                created_at=4.0,
            )
        )
        dag.connection.executescript(
            """
            CREATE TABLE lcm_compaction_node_edges(
                parent_node_id INTEGER NOT NULL,
                child_node_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                PRIMARY KEY(parent_node_id, child_node_id)
            ) WITHOUT ROWID;
            CREATE INDEX idx_lcm_compaction_node_edges_reverse
                ON lcm_compaction_node_edges(
                    session_id, child_node_id, parent_node_id
                );
            """
        )
        dag.connection.executemany(
            """
            INSERT INTO lcm_compaction_node_edges(
                parent_node_id, child_node_id, session_id
            ) VALUES (?, ?, ?)
            """,
            [
                (parent_id, child_id, "session-a"),
                (other_parent_id, same_time_root_a_id, "session-b"),
            ],
        )
        # Make the normalized-map path observable rather than allowing the
        # legacy JSON fallback to produce the same answer accidentally.
        dag.connection.execute(
            "UPDATE summary_nodes SET source_ids='[]' WHERE node_id=?",
            (parent_id,),
        )
        dag.connection.execute(
            """
            INSERT INTO lcm_migration_state(step_name, completed_at)
            VALUES ('background_compaction_node_edges_v1', 1)
            """
        )
        dag.connection.commit()

        frontier = dag.get_active_frontier_nodes("session-a")

        assert [node.node_id for node in frontier] == [
            parent_id,
            same_time_root_a_id,
            same_time_root_b_id,
            later_root_id,
        ]
    finally:
        dag.close()
