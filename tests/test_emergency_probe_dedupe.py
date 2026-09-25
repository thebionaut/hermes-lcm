"""Emergency archive probe must not count stored duplicate rows."""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "probe-session"


def test_archive_probe_ignores_stored_duplicates(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            sensitive_patterns_enabled=False,
            proactive_recall_enabled=False,
        ),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine._session_id = SESSION
    engine._conversation_id = SESSION
    messages = [
        {"role": "user", "content": f"m{index}", "timestamp": 1_790_000_000.0 + index}
        for index in range(10)
    ]
    try:
        for _copy in range(4):
            engine._store.append_batch(SESSION, messages, conversation_id=SESSION)
        probe = engine._store.get_session_messages_after(SESSION, 0, limit=21)
        assert len(probe) == 21
        assert engine._distinct_durable_probe_rows(probe, replay_limit=20) == 10
        assert engine._distinct_durable_probe_rows(probe[:5], replay_limit=20) == 5
    finally:
        engine.shutdown()
