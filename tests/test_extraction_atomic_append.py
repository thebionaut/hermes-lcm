"""Atomic persistence contracts for prepared pre-compaction extractions."""

from __future__ import annotations

from datetime import datetime
import threading

import hermes_lcm.extraction as extraction


class _FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _daily_file(output_dir):
    return output_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"


def test_concurrent_extractions_each_commit_one_complete_record(tmp_path):
    ready = threading.Barrier(2)
    finished = {name: threading.Event() for name in ("alpha", "beta")}
    results: dict[str, bool] = {}

    def persist(name: str) -> None:
        try:
            ready.wait(timeout=2)
            results[name] = extraction.persist_pre_compaction_extraction(
                extraction.PreparedExtraction(content=f"- decision-{name}"),
                output_path=str(tmp_path),
                session_id=name,
            )
        finally:
            finished[name].set()

    workers = [
        threading.Thread(target=persist, args=(name,), name=f"writer-{name}")
        for name in finished
    ]
    for worker in workers:
        worker.start()
    try:
        assert all(done.wait(timeout=3) for done in finished.values())
    finally:
        for worker in workers:
            worker.join(timeout=2)

    assert results == {"alpha": True, "beta": True}
    content = _daily_file(tmp_path).read_text(encoding="utf-8")
    assert content.count("## Extraction") == 2
    assert content.count("decision-alpha") == 1
    assert content.count("decision-beta") == 1
    assert content.count("(alpha)") == 1
    assert content.count("(beta)") == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_expired_deadline_never_creates_a_daily_note(tmp_path):
    clock = _FakeClock(101.0)

    assert extraction.persist_pre_compaction_extraction(
        extraction.PreparedExtraction(content="- must-not-be-appended"),
        output_path=str(tmp_path),
        deadline=101.0,
        now=clock,
    ) is False
    assert not _daily_file(tmp_path).exists()


def test_cancelled_result_never_creates_a_daily_note(tmp_path):
    cancelled = threading.Event()
    cancelled.set()

    assert extraction.persist_pre_compaction_extraction(
        extraction.PreparedExtraction(content="- must-not-be-appended"),
        output_path=str(tmp_path),
        cancel_event=cancelled,
    ) is False
    assert not _daily_file(tmp_path).exists()
