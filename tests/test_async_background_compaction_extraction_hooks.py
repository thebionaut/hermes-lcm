"""Compatibility gates for opt-in provider-backed extraction paths."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        (
            {"extraction_enabled": True},
            "file_extraction_requires_foreground_compaction",
        ),
        (
            {
                "assertions_enabled": True,
                "assertion_extraction_enabled": True,
            },
            "assertion_extraction_requires_foreground_compaction",
        ),
    ],
)
def test_provider_backed_derivations_keep_existing_foreground_mode(
    tmp_path,
    overrides,
    expected_reason,
):
    """A new background egress path is never enabled implicitly."""
    db_path = tmp_path / "compatibility-gate.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=1,
        background_compaction_backlog_tokens=1,
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        **overrides,
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    try:
        assert engine._background_compaction_enabled() is False
        assert engine._background_compaction is None
        status = engine.get_background_compaction_status()
        assert status["configured_enabled"] is True
        assert status["enabled"] is False
        assert status["worker_count"] == 0
        assert status["compatibility_blocked_reason"] == expected_reason

        engine.ingest(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ]
        )
        tables = {
            row[0]
            for row in sqlite3.connect(db_path).execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "lcm_compaction_jobs" not in tables
    finally:
        engine.shutdown()
