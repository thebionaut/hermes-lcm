"""Engine integration for exact active-summary frontier selection."""

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def test_summary_frontier_nodes_delegates_to_exact_reader(tmp_path, monkeypatch):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "frontier-delegate.db"))
    )
    engine._session_id = "session-a"
    expected = [SummaryNode(node_id=7, session_id="session-a", depth=2)]
    calls = []

    def exact_frontier(session_id):
        calls.append(session_id)
        return expected

    monkeypatch.setattr(engine._dag, "get_active_frontier_nodes", exact_frontier)
    try:
        assert engine._summary_frontier_nodes() is expected
    finally:
        engine.shutdown()

    assert calls == ["session-a"]


def test_assemble_context_reads_exact_frontier_once_and_keeps_budget_trimming(
    tmp_path,
    monkeypatch,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "frontier-engine.db"))
    )
    engine._session_id = "session-a"
    calls = 0
    frontier = [
        SummaryNode(
            node_id=index + 1,
            session_id="session-a",
            depth=0,
            summary=f"root-{index} " + ("x" * 40),
            token_count=20,
            source_token_count=20,
            created_at=float(index),
        )
        for index in range(137)
    ]

    def exact_frontier(session_id):
        nonlocal calls
        calls += 1
        assert session_id == "session-a"
        return frontier

    monkeypatch.setattr(engine._dag, "get_active_frontier_nodes", exact_frontier)
    try:
        assembled = engine._assemble_context(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "tail"}],
            assembly_cap_override=180,
        )
    finally:
        engine.shutdown()

    assert calls == 1
    rendered = "\n".join(str(message.get("content", "")) for message in assembled)
    assert "root-0" in rendered
    assert "root-136" not in rendered
    assert assembled[-1]["content"] == "tail"
