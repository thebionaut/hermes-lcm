"""Host-boundary regressions for background compaction.

Hermes may re-announce the same session on ordinary turns and after an
in-place transcript rewrite.  Neither notification is a new session.  Losing
the process-local ingest cursor at either boundary makes the next observation
append the whole active transcript again and can turn a recoverable context
admission stop into an ever-growing retry loop.
"""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _config(db_path):
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1_000_000,
        context_reserve_tokens=100_000,
    )


def test_global_host_pin_aligns_codex56_cap_with_fixed_reserve(tmp_path):
    engine = LCMEngine(_config(tmp_path / "codex56-global-pin.db"))
    try:
        engine.update_model(
            model="gpt-5.6-sol",
            provider="openai-codex",
            context_length=1_000_000,
        )

        # Stock Hermes admits 80% of the configured 1M window.  LCM must cap
        # that raw override back to Codex OAuth's real 900K total window, then
        # publish at the same 800K boundary under its fixed 100K reserve.
        assert engine.raw_context_length == 1_000_000
        assert engine.context_length == 900_000
        assert engine.effective_context_length_cap == 900_000
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"
        assert engine.threshold_tokens == 800_000
        assert engine._background_publication_trigger_tokens() == 800_000

        status = engine.get_status()
        assert status["threshold_policy"] == "fixed_reserve"
        assert status["context_reserve_tokens"] == 100_000
        assert status["background_compaction"]["publication_trigger_tokens"] == 800_000
    finally:
        engine.shutdown()


def test_same_session_rebinds_preserve_cursor_and_do_not_reingest(tmp_path):
    engine = LCMEngine(_config(tmp_path / "same-session-rebind.db"))
    original = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    try:
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        engine.ingest(original)
        first_store_id = int(
            engine._store.connection.execute(
                "SELECT MIN(store_id) FROM messages WHERE session_id = ?",
                ("session-a",),
            ).fetchone()[0]
        )
        engine._lifecycle.advance_frontier(
            "conversation-a",
            "session-a",
            first_store_id,
        )
        engine._last_compacted_store_id = first_store_id

        # Ordinary host turn lifecycle: finalize then bind the same durable
        # conversation/session again.  This is freshness, not invalidation.
        engine._lifecycle.finalize_session(
            "conversation-a",
            "session-a",
            frontier_store_id=first_store_id,
        )
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )

        assert engine._ingest_cursor == len(original)
        assert engine._last_compacted_store_id == first_store_id
        assert engine._ingest_cursor_needs_reconcile is False

        # Hermes also emits a same-ID compression notification after an
        # in-place rewrite.  It must preserve the cursor for the list the
        # compressor already made durable.
        engine.on_session_start(
            "session-a",
            boundary_reason="compression",
            old_session_id="session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )

        assert engine._ingest_cursor == len(original)
        assert engine._last_compacted_store_id == first_store_id
        assert engine._ingest_cursor_needs_reconcile is False

        extended = [*original, {"role": "user", "content": "new prompt"}]
        engine.ingest(extended)
        assert engine._store.get_session_count("session-a") == len(extended)
    finally:
        engine.shutdown()


def test_same_session_rebind_reconciles_a_divergent_active_prefix(tmp_path):
    engine = LCMEngine(_config(tmp_path / "same-session-divergent-prefix.db"))
    original = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    rebound = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "C"},
        {"role": "user", "content": "D"},
    ]
    try:
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        engine.ingest(original)

        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        engine.ingest(rebound)

        contents = [
            row["content"]
            for row in engine._store.get_session_messages("session-a")
        ]
        assert contents.count("A") == 1
        assert contents.count("B") == 1
        assert contents.count("C") == 1
        assert contents.count("D") == 1
    finally:
        engine.shutdown()


def test_conversation_reassignment_still_resets_same_session_runtime(tmp_path):
    engine = LCMEngine(_config(tmp_path / "conversation-reassignment.db"))
    try:
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        engine.ingest([{"role": "user", "content": "first"}])
        assert engine._ingest_cursor == 1

        engine.on_session_start(
            "session-a",
            conversation_id="conversation-b",
            context_length=900_000,
        )

        # A real conversation reassignment is an invalidating event.  It must
        # not take the ordinary same-session fast path.
        assert engine._ingest_cursor == 0
        assert engine._ingest_cursor_needs_reconcile is True
    finally:
        engine.shutdown()


def test_repeated_no_progress_foreground_calls_do_not_amplify_or_requeue(tmp_path):
    config = _config(tmp_path / "repeated-no-progress.db")
    config.fresh_tail_count = 0
    config.fresh_tail_max_tokens = 0
    config.background_compaction_backlog_tokens = 1
    config.background_compaction_min_provider_source_tokens = 1
    engine = LCMEngine(config)
    messages = [
        {"role": "user", "content": "durable backlog " * 200},
        {"role": "assistant", "content": "still waiting for background work"},
    ]
    try:
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        for _ in range(20):
            assert engine.compress(messages, current_tokens=810_000) == messages

        assert engine._store.get_session_count("session-a") == len(messages)
        with engine._background_compaction._lock:
            job_count = engine._background_compaction.connection.execute(
                "SELECT COUNT(*) FROM lcm_compaction_jobs "
                "WHERE conversation_id='conversation-a'"
            ).fetchone()[0]
        assert job_count == 1

        extended = [*messages, {"role": "user", "content": "new material"}]
        assert engine.compress(extended, current_tokens=810_100) == extended
        assert engine._store.get_session_count("session-a") == len(extended)
        with engine._background_compaction._lock:
            job_count = engine._background_compaction.connection.execute(
                "SELECT COUNT(*) FROM lcm_compaction_jobs "
                "WHERE conversation_id='conversation-a'"
            ).fetchone()[0]
        assert job_count == 1
    finally:
        engine.shutdown()
