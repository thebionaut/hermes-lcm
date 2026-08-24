"""Regression tests for bounded restart-replay reconciliation."""

import json
import time
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.externalize import (
    ExternalizedPayloadLookupBudgetExceeded,
    _payload_index_cache,
    _payload_index_lock,
    find_externalized_tool_result_content_for_call,
)
from hermes_lcm.reconcile import ReconcileBudgetExceeded, ReconcileMixin


def _write_payload(path: Path, *, tool_call_id: str, session_id: str, content: str) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "tool_result",
                "tool_call_id": tool_call_id,
                "session_id": session_id,
                "role": "tool",
                "content": content,
            }
        ),
        encoding="utf-8",
    )


def test_repeated_lookup_does_not_rescan_unrelated_payloads(tmp_path, monkeypatch):
    storage = tmp_path / "payloads"
    storage.mkdir()
    _write_payload(storage / "wanted.json", tool_call_id="call-wanted", session_id="s1", content="wanted")
    for index in range(12):
        _write_payload(
            storage / f"other-{index}.json",
            tool_call_id=f"call-other-{index}",
            session_id="s1",
            content="other",
        )

    config = LCMConfig(large_output_externalization_path=str(storage))
    _payload_index_cache.clear()
    reads = 0
    original_read_text = Path.read_text

    def counted_read_text(path, *args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)

    first = find_externalized_tool_result_content_for_call(
        tool_call_id="call-wanted", session_id="s1", config=config
    )
    reads_after_first = reads
    second = find_externalized_tool_result_content_for_call(
        tool_call_id="call-wanted", session_id="s1", config=config
    )

    assert first == second == "wanted"
    assert reads_after_first == 14  # 13 index reads + the matching payload
    assert reads == reads_after_first + 1  # cached index; only matching payload reread


def test_payload_index_lock_wait_respects_deadline(tmp_path):
    storage = tmp_path / "payloads"
    storage.mkdir()
    config = LCMConfig(large_output_externalization_path=str(storage))

    _payload_index_lock.acquire()
    try:
        started = time.monotonic()
        with pytest.raises(ExternalizedPayloadLookupBudgetExceeded, match="lock"):
            find_externalized_tool_result_content_for_call(
                tool_call_id="missing",
                config=config,
                deadline=time.monotonic() + 0.02,
            )
        assert time.monotonic() - started < 0.2
    finally:
        _payload_index_lock.release()


def test_reconcile_budget_is_configurable_from_environment(monkeypatch):
    monkeypatch.setenv("LCM_RECONCILE_BUDGET_SECONDS", "2.5")
    assert LCMConfig.from_env().reconcile_budget_seconds == 2.5


def test_reconcile_translates_payload_lookup_deadline_to_engine_budget_error():
    class Harness(ReconcileMixin):
        _config = LCMConfig(reconcile_budget_seconds=1.0)

        def _reconcile_ingest_cursor_from_store_impl(self, messages):
            raise ExternalizedPayloadLookupBudgetExceeded("expired")

    with pytest.raises(ReconcileBudgetExceeded, match="payload lookup"):
        Harness()._reconcile_ingest_cursor_from_store([{"role": "user", "content": "x"}])


def test_reconcile_deadline_is_initialized_and_cleared():
    class Harness(ReconcileMixin):
        _config = LCMConfig(reconcile_budget_seconds=1.0)

        def _reconcile_ingest_cursor_from_store_impl(self, messages):
            assert self._reconcile_deadline is not None
            assert self._reconcile_deadline > time.monotonic()
            assert self._identity_memo == {}
            return 7

    harness = Harness()
    assert harness._reconcile_ingest_cursor_from_store([{"role": "user", "content": "x"}]) == 7
    assert harness._reconcile_deadline is None
    assert harness._identity_memo is None
