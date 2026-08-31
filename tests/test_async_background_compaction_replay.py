"""Active-tail merge contracts for asynchronous compaction."""

from __future__ import annotations

import json

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


class _IgnorePrefixPattern:
    pattern = r"^IGNORE:"

    def search(self, text, timeout=None):
        del timeout
        return text.startswith("IGNORE:")


def test_unmapped_pre_frontier_placeholder_drops_but_newest_unmapped_suffix_survives(
    tmp_path,
):
    """Only caller objects demonstrably newer than durable mapped rows survive."""
    db_path = tmp_path / "unmapped-boundary.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    raw_one = {"role": "user", "content": "raw-one"}
    raw_two = {"role": "assistant", "content": "raw-two"}
    durable_tail = {"role": "user", "content": "durable-tail"}
    engine._store.append_batch(
        "s",
        [raw_one, raw_two, durable_tail],
        token_estimates=[10, 10, 10],
        conversation_id="c",
    )
    ids = [
        int(row[0])
        for row in engine._store.connection.execute(
            "SELECT store_id FROM messages WHERE session_id='s' ORDER BY store_id"
        ).fetchall()
    ]
    ignored_placeholder = {
        "role": "user",
        "content": (
            "[LCM active replay placeholder: message ignored; "
            "digest=deadbeef; recover with lcm_expand]"
        ),
    }
    newest_unmapped = {
        "role": "user",
        "content": "newest-not-yet-durable",
        "client_turn_metadata": {"nonce": "newest-unmapped"},
    }
    preserve = [
        raw_one,
        ignored_placeholder,
        raw_two,
        durable_tail,
        newest_unmapped,
    ]
    try:
        projected = engine._project_durable_active_tail(
            ids[1],
            preserve_messages=preserve,
        )
        assert all("message ignored" not in str(message.get("content")) for message in projected)
        assert sum(message.get("content") == "durable-tail" for message in projected) == 1
        assert [
            message
            for message in projected
            if message.get("client_turn_metadata", {}).get("nonce")
            == "newest-unmapped"
        ] == [newest_unmapped]
    finally:
        engine.shutdown()


def test_projection_does_not_resurrect_policy_dropped_durable_rows(
    tmp_path,
    monkeypatch,
):
    """Only a truly unmapped caller suffix may overlay the durable projection."""
    db_path = tmp_path / "policy-dropped-replay.db"
    raw_messages = [
        {"role": "user", "content": "IGNORE: private instruction"},
        {"role": "assistant", "content": "dependent secret"},
        {"role": "user", "content": "durable keep"},
    ]
    store = MessageStore(db_path)
    try:
        store.append_batch(
            "s",
            raw_messages,
            token_estimates=[10, 10, 10],
            conversation_id="c",
        )
    finally:
        store.close()

    monkeypatch.setattr(
        "hermes_lcm.engine.compile_message_patterns",
        lambda _patterns: [_IgnorePrefixPattern()],
    )
    monkeypatch.setattr(
        "hermes_lcm.summary_serialization.compile_message_patterns",
        lambda _patterns: [_IgnorePrefixPattern()],
    )
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        ignore_message_patterns=[r"^IGNORE:"],
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    unmapped_suffix = {
        "role": "user",
        "content": [{"type": "text", "text": "not durable yet"}],
        "client_turn_metadata": {"nonce": "exact-unmapped-suffix"},
    }
    try:
        projected = engine._project_durable_active_tail(
            0,
            preserve_messages=raw_messages + [unmapped_suffix],
        )
        assert [message.get("content") for message in projected[:-1]] == [
            "durable keep"
        ]
        assert projected[-1] == unmapped_suffix
        assert all(
            "dependent secret" not in str(message.get("content"))
            and "private instruction" not in str(message.get("content"))
            for message in projected
        )
    finally:
        engine.shutdown()


def test_committed_publication_defers_local_rebuild_failure_without_raising(
    tmp_path, monkeypatch
):
    """A post-commit projection fault cannot undo or obscure the durable CAS."""
    db_path = tmp_path / "postcommit-rebuild.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1,
        fresh_tail_count=1,
        leaf_chunk_tokens=10,
        context_reserve_tokens=100_000,
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"raw-{index} x x"}
        for index in range(6)
    ]
    try:
        engine.ingest(messages)
        job = engine.run_background_compaction_once(
            summarizer=lambda **_kwargs: ("prepared summary", 1)
        )
        assert job is not None and job.state == "prepared"
        original_marker = engine._last_compacted_store_id
        monkeypatch.setattr(
            engine,
            "_project_durable_active_tail",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected postcommit projection failure")
            ),
        )
        result = engine.promote_prepared_compaction(
            job.job_id,
            messages,
            current_tokens=272_000,
        )
        assert result.promoted is True
        assert result.reason == "published_context_rebuild_deferred"
        assert result.messages == messages
        lifecycle = engine._lifecycle.get_by_conversation("c")
        assert lifecycle.current_frontier_store_id == job.captured_watermark_store_id
        assert engine._last_compacted_store_id == original_marker
        assert engine._store.get_session_count("s") == len(messages)
        assert engine._dag.get_session_node_count("s") >= 1
    finally:
        engine.shutdown()


def test_background_enqueue_failure_does_not_relabel_committed_raw_ingest(
    tmp_path, monkeypatch
):
    """Maintenance failure is best-effort after immutable raw commit succeeds."""
    db_path = tmp_path / "enqueue-after-ingest.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1,
        fresh_tail_count=1,
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"raw-{index}"}
        for index in range(4)
    ]
    monkeypatch.setattr(
        engine,
        "_maybe_enqueue_background_compaction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected maintenance queue failure")
        ),
    )
    try:
        engine.ingest(messages)
        status = engine.get_status()
        assert engine._store.get_session_count("s") == len(messages)
        assert status["ingest_failure_count"] == 0
        assert status["consecutive_ingest_failures"] == 0
        assert status["last_ingest_error"] == ""
    finally:
        engine.shutdown()


def test_public_promotion_refuses_job_from_another_bound_conversation(tmp_path):
    """An engine can publish only work belonging to its active session scope."""
    db_path = tmp_path / "promotion-scope.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1,
        fresh_tail_count=0,
        leaf_chunk_tokens=100,
        context_reserve_tokens=100_000,
    )
    engine = LCMEngine(config)
    engine.on_session_start("session-a", conversation_id="conversation-a", context_length=372_000)
    foreign_messages = [
        {"role": "user", "content": f"foreign-{index}"}
        for index in range(4)
    ]
    store = MessageStore(db_path)
    try:
        store.append_batch(
            "session-b",
            foreign_messages,
            token_estimates=[10] * len(foreign_messages),
            conversation_id="conversation-b",
        )
    finally:
        store.close()
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session("session-b", conversation_id="conversation-b")
    finally:
        lifecycle.close()
    SummaryDAG(db_path).close()
    try:
        job = engine._background_compaction.capture_and_enqueue(
            conversation_id="conversation-b",
            session_id="session-b",
            expected_frontier_store_id=0,
            active_tokens=300_000,
            effective_context_window=372_000,
            policy_fingerprint=engine._background_policy_fingerprint(),
        )
        assert job is not None
        prepared = engine._background_compaction.run_next_job(
            owner="foreign-worker",
            summarizer=lambda **_kwargs: ("foreign prepared summary", 1),
        )
        assert prepared is not None and prepared.state == "prepared"
        result = engine.promote_prepared_compaction(
            job.job_id,
            [{"role": "user", "content": "active-a"}],
            current_tokens=272_000,
        )
        assert result.promoted is False
        assert result.reason == "job_scope_mismatch"
        foreign_state = engine._lifecycle.get_by_conversation("conversation-b")
        assert foreign_state.current_frontier_store_id == 0
    finally:
        engine.shutdown()


def test_engine_status_distinguishes_durable_frontier_from_local_active_coverage(
    tmp_path,
):
    """A sibling/rotate frontier is not active summary coverage until adopted."""
    db_path = tmp_path / "coverage-adoption.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    engine._store.append_batch(
        "s",
        [{"role": "user", "content": "raw"}],
        token_estimates=[10],
        conversation_id="c",
    )
    try:
        engine._lifecycle.advance_frontier("c", "s", 1)
        pending = engine.get_background_compaction_status()
        assert pending["durable_frontier_store_id"] == 1
        assert pending["active_coverage"] == "raw"

        engine._last_compacted_store_id = 1
        adopted = engine.get_background_compaction_status()
        assert adopted["active_coverage"] == "published_summary"
    finally:
        engine.shutdown()


def test_low_pressure_preflight_requests_zero_provider_sibling_frontier_adoption(
    tmp_path,
    monkeypatch,
):
    """A sibling commit is adopted on the next turn even below all triggers."""
    db_path = tmp_path / "low-pressure-sibling-adoption.db"
    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1,
        fresh_tail_count=1,
        leaf_chunk_tokens=20,
        context_reserve_tokens=100_000,
    )
    publisher = LCMEngine(config)
    sibling = LCMEngine(config)
    publisher.on_session_start("s", conversation_id="c", context_length=372_000)
    sibling.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"raw-{index} " + "x " * 20}
        for index in range(6)
    ]
    try:
        publisher.ingest(messages)
        prepared = publisher.run_background_compaction_once(
            summarizer=lambda **_kwargs: ("sibling-visible summary", 1),
        )
        assert prepared is not None and prepared.state == "prepared"
        published = publisher.promote_prepared_compaction(
            prepared.job_id,
            messages,
            current_tokens=272_000,
        )
        assert published.promoted
        assert sibling._last_compacted_store_id == 0

        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("sibling adoption must not call a provider")
            ),
        )
        next_turn = [{"role": "user", "content": "small next turn"}]
        assert sibling.should_compress_preflight(next_turn) is True
        adopted = sibling.compress(next_turn, current_tokens=10)
        assert sibling._last_compacted_store_id == published.captured_watermark_store_id
        assert any("sibling-visible summary" in str(message.get("content")) for message in adopted)
    finally:
        publisher.shutdown()
        sibling.shutdown()


def test_legacy_canonical_prefix_advances_without_provider_then_queues_only_suffix(
    tmp_path,
    monkeypatch,
):
    """A migrated lagging frontier adopts canonical work before preparation."""
    db_path = tmp_path / "legacy-prefix-adoption.db"
    raw_messages = [
        {"role": "user", "content": f"legacy-raw-{index} " + "x " * 8}
        for index in range(4)
    ]
    store = MessageStore(db_path)
    source_ids = store.append_batch(
        "s",
        raw_messages,
        token_estimates=[10] * len(raw_messages),
        conversation_id="c",
    )
    store.close()
    dag = SummaryDAG(db_path)
    dag.add_node(
        SummaryNode(
            session_id="s",
            summary="legacy already-canonical prefix",
            source_ids=source_ids[:2],
            source_type="messages",
            token_count=5,
            source_token_count=20,
        )
    )
    dag.close()
    lifecycle = LifecycleStateStore(db_path)
    lifecycle.bind_session("s", conversation_id="c")
    lifecycle.close()

    config = LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1,
        fresh_tail_count=1,
        leaf_chunk_tokens=100,
        context_reserve_tokens=100_000,
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("canonical-prefix adoption must not call a provider")
        ),
    )
    try:
        assert engine.should_compress_preflight(raw_messages) is True
        adopted_state = engine._lifecycle.get_by_conversation("c")
        assert adopted_state.current_frontier_store_id == source_ids[1]
        assert engine._background_compaction.connection.execute(
            """
            SELECT COUNT(*) FROM lcm_compaction_jobs
            WHERE state IN ('queued','running','prepared','retry')
            """
        ).fetchone()[0] == 0

        rebuilt = engine.compress(raw_messages, current_tokens=10)
        assert engine._last_compacted_store_id == source_ids[1]
        rebuilt_text = "\n".join(str(message.get("content")) for message in rebuilt)
        assert "legacy already-canonical prefix" in rebuilt_text
        assert "legacy-raw-0" not in rebuilt_text
        assert "legacy-raw-1" not in rebuilt_text
        assert "legacy-raw-2" in rebuilt_text
        assert "legacy-raw-3" in rebuilt_text

        active_job = engine._background_compaction.connection.execute(
            """
            SELECT expected_frontier_store_id, source_store_ids
            FROM lcm_compaction_jobs
            WHERE state IN ('queued','running','prepared','retry')
            """
        ).fetchone()
        assert active_job is not None
        assert int(active_job["expected_frontier_store_id"]) == source_ids[1]
        queued_ids = set(json.loads(active_job["source_store_ids"]))
        assert source_ids[0] not in queued_ids
        assert source_ids[1] not in queued_ids
    finally:
        engine.shutdown()
