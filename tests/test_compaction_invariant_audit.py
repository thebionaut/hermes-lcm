import importlib.util
import json
from pathlib import Path

import pytest

from hermes_lcm.background_compaction import BackgroundCompactionStore
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.invariant_audit import audit_compaction_invariants
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore
from hermes_lcm.tools import lcm_doctor


def _module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


design = _module("test_async_background_compaction_design.py")


def _prepared(background, ids, *, conversation="c", session="s", policy="p1"):
    job = design._explicit_job(
        background,
        ids,
        conversation=conversation,
        session=session,
        policy=policy,
    )
    return design._prepare(background, job)


@pytest.mark.parametrize(
    "job_kind",
    ["standard", "lossless_raw_projection", "partial_atomic_repair"],
)
def test_auditor_classifies_valid_job_kinds_and_legitimate_source_gaps(
    tmp_path, job_kind
):
    path = tmp_path / f"valid-{job_kind}.db"
    store = MessageStore(path)
    first = store.append_batch(
        "s", [{"role": "user", "content": "one"}],
        token_estimates=[10], conversation_id="c"
    )
    store.append_batch(
        "other", [{"role": "user", "content": "interleaved"}],
        token_estimates=[10], conversation_id="other"
    )
    rest = store.append_batch(
        "s", [{"role": "assistant", "content": "two"}],
        token_estimates=[10], conversation_id="c"
    )
    store.close()
    dag = SummaryDAG(path)
    dag.close()
    lifecycle = LifecycleStateStore(path)
    lifecycle.bind_session("s", conversation_id="c")
    lifecycle.close()
    background = BackgroundCompactionStore(path, config=design._config(path))
    try:
        prepared = _prepared(background, first + rest)
        if job_kind != "standard":
            values = [job_kind, prepared.job_id]
            background.connection.execute(
                "UPDATE lcm_compaction_jobs SET job_kind=? WHERE job_id=?", values
            )
        if job_kind == "partial_atomic_repair":
            ids = tuple(first + rest)
            background.connection.execute(
                """
                INSERT INTO lcm_compaction_dag_epochs(session_id, mutation_epoch)
                VALUES('s', 7)
                ON CONFLICT(session_id) DO UPDATE SET mutation_epoch=7
                """
            )
            background.connection.execute(
                """
                UPDATE lcm_compaction_jobs
                SET repair_group_store_ids=?, repair_covered_store_ids=?,
                    repair_uncovered_store_ids=?, repair_bound_dag_epoch=7
                WHERE job_id=?
                """,
                (str(list(ids)), str([ids[0]]), str([ids[1]]), prepared.job_id),
            )
        background.connection.commit()
        before = background.connection.total_changes
        result = audit_compaction_invariants(
            background.connection,
            current_policy_fingerprint="p1",
            scope="job",
            job_id=prepared.job_id,
        )
        assert result["status"] == "pass"
        assert result["complete"] is True
        assert result["coverage"]["classification_counts"][
            "valid_awaiting_pressure"
        ] == 1
        assert result["age_used_as_corruption_signal"] is False
        assert background.connection.total_changes == before
    finally:
        background.close()


@pytest.mark.parametrize(
    "mutation,classification,reason",
    [
        (
            "UPDATE lcm_compaction_jobs SET policy_fingerprint='old' WHERE job_id=?",
            "provably_obsolete",
            "policy_changed",
        ),
        (
            "UPDATE lcm_compaction_jobs SET prepared_node_count=99 WHERE job_id=?",
            "inconsistent_stranded",
            "prepared_node_count_mismatch",
        ),
    ],
)
def test_auditor_distinguishes_obsolete_from_inconsistent(
    tmp_path, mutation, classification, reason
):
    path = tmp_path / f"{classification}.db"
    ids = design._seed(path)
    background = BackgroundCompactionStore(path, config=design._config(path))
    try:
        prepared = _prepared(background, ids)
        background.connection.execute(mutation, (prepared.job_id,))
        background.connection.commit()
        result = audit_compaction_invariants(
            background.connection,
            current_policy_fingerprint="p1",
            scope="job",
            job_id=prepared.job_id,
        )
        assert result["status"] == (
            "fail" if classification == "inconsistent_stranded" else "pass"
        )
        item = result["examples"][classification][0]
        assert reason in item["reasons"]
    finally:
        background.close()


def test_auditor_exact_tool_ids_and_order_not_simple_counts(tmp_path):
    path = tmp_path / "tool-pairs.db"
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call-b", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "b first", "tool_call_id": "call-b"},
        {"role": "tool", "content": "a second", "tool_call_id": "call-a"},
    ]
    ids = design._seed(path, messages=messages, tokens=[10, 10, 10])
    background = BackgroundCompactionStore(path, config=design._config(path))
    try:
        prepared = _prepared(background, ids)
        result = audit_compaction_invariants(
            background.connection,
            current_policy_fingerprint="p1",
            scope="job",
            job_id=prepared.job_id,
        )
        assert result["status"] == "fail"
        item = result["examples"]["inconsistent_stranded"][0]
        assert "tool_result_id_or_order_mismatch" in item["reasons"]
    finally:
        background.close()


def test_limited_whole_store_page_is_unknown_and_resumable(tmp_path):
    path = tmp_path / "paged.db"
    first_ids = design._seed(path, session="s1", conversation="c1")
    store = MessageStore(path)
    second_ids = store.append_batch(
        "s2",
        [{"role": "user", "content": f"second-{index}"} for index in range(4)],
        token_estimates=[10] * 4,
        conversation_id="c2",
    )
    store.close()
    lifecycle = LifecycleStateStore(path)
    lifecycle.bind_session("s2", conversation_id="c2")
    lifecycle.close()
    background = BackgroundCompactionStore(path, config=design._config(path))
    try:
        _prepared(background, first_ids, conversation="c1", session="s1")
        _prepared(background, second_ids, conversation="c2", session="s2")
        page = audit_compaction_invariants(
            background.connection,
            current_policy_fingerprint="p1",
            scope="whole_store",
            limit=1,
        )
        assert page["status"] == "unknown"
        assert page["complete"] is False
        assert page["coverage"]["has_more"] is True
        assert page["coverage"]["next_cursor"]
        second_page = audit_compaction_invariants(
            background.connection,
            current_policy_fingerprint="p1",
            scope="whole_store",
            cursor=page["coverage"]["next_cursor"],
            limit=1,
        )
        assert second_page["complete"] is True
        assert second_page["coverage"]["prepared_jobs_scanned"] == 1
    finally:
        background.close()


def test_historical_epoch_change_is_obsolete_not_corruption(tmp_path):
    path = tmp_path / "historical-epoch.db"
    ids = design._seed(path)
    background = BackgroundCompactionStore(path, config=design._config(path))
    try:
        prepared = _prepared(background, ids)
        background.connection.execute(
            """
            UPDATE lcm_compaction_session_fences
            SET generation=generation+1
            WHERE conversation_id='c'
            """
        )
        background.connection.commit()
        result = audit_compaction_invariants(
            background.connection,
            current_policy_fingerprint="p1",
            scope="epoch",
            session_id="s",
            fence_generation=prepared.session_fence_generation,
        )
        assert result["status"] == "pass"
        item = result["examples"]["provably_obsolete"][0]
        assert item["reasons"] == ["session_fence_changed"]
    finally:
        background.close()


def test_graph_relationship_violation_fails_with_bounded_example(tmp_path):
    path = tmp_path / "graph-violation.db"
    ids = design._seed(path)
    background = BackgroundCompactionStore(path, config=design._config(path))
    try:
        _prepared(background, ids)
        background.connection.execute(
            """
            INSERT INTO lcm_compaction_node_sources(node_id, session_id, store_id)
            VALUES(999999, 's', ?)
            """,
            (ids[0],),
        )
        background.connection.commit()
        result = audit_compaction_invariants(
            background.connection,
            current_policy_fingerprint="p1",
            scope="session",
            session_id="s",
            example_limit=1,
        )
        assert result["status"] == "fail"
        issue = next(
            item for item in result["relationship_issues"]
            if item["check"] == "node_source_owner_missing"
        )
        assert issue["example_ids"] == [999999]
    finally:
        background.close()


def test_lcm_doctor_exposes_structured_read_only_audit(tmp_path):
    path = tmp_path / "doctor-audit.db"
    design._seed(path)
    engine = LCMEngine(design._config(path, background_compaction_workers=0))
    try:
        engine.on_session_start("s", conversation_id="c")
        result = json.loads(
            lcm_doctor(
                {
                    "integrity_audit": True,
                    "audit_scope": "session",
                    "session_id": "s",
                    "limit": 10,
                },
                engine=engine,
            )
        )
        audit = result["compaction_invariant_audit"]
        assert audit["read_only"] is True
        assert audit["complete"] is True
        assert audit["snapshot"] == {
            "kind": "bounded_live_read_transaction",
            "consistent": True,
        }
        assert any(
            check["check"] == "compaction_invariants"
            for check in result["checks"]
        )
    finally:
        engine.shutdown()
