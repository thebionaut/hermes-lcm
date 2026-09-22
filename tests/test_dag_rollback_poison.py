"""Regression coverage for failed SummaryDAG transaction compensation."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.dag import SummaryDAG, SummaryNode


class _RollbackFailOnceConnection:
    """Connection proxy that models an indeterminate failed rollback."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.rollback_attempted = False

    def rollback(self) -> None:
        if not self.rollback_attempted:
            self.rollback_attempted = True
            raise sqlite3.OperationalError("synthetic rollback failure")
        self._connection.rollback()

    def __getattr__(self, name: str):
        return getattr(self._connection, name)


def test_summary_dag_rollback_failure_poison_closes_transaction(tmp_path):
    db_path = tmp_path / "dag-rollback-poison.db"
    dag = SummaryDAG(db_path)
    real_connection = dag.connection
    assert real_connection is not None
    poisoned_connection = _RollbackFailOnceConnection(real_connection)
    dag._conn = poisoned_connection

    def fail_before_commit(_connection: sqlite3.Connection) -> None:
        raise TimeoutError("primary DAG deadline")

    try:
        with pytest.raises(TimeoutError, match="primary DAG deadline"):
            dag.add_node(
                SummaryNode(
                    session_id="s",
                    depth=0,
                    summary="should roll back",
                    source_ids=[1],
                ),
                before_commit=fail_before_commit,
            )

        assert poisoned_connection.rollback_attempted
        assert dag.connection is None
        with pytest.raises(RuntimeError, match="closed"):
            dag.add_node(
                SummaryNode(
                    session_id="s",
                    depth=0,
                    summary="later",
                    source_ids=[2],
                )
            )

        with sqlite3.connect(str(db_path)) as verifier:
            summaries = verifier.execute(
                "SELECT summary FROM summary_nodes ORDER BY node_id"
            ).fetchall()
        assert summaries == []
    finally:
        dag.close()
