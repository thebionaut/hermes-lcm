"""Foreground emergency contracts for asynchronous compaction."""

from __future__ import annotations

import sqlite3
import threading
import time

from hermes_lcm.background_compaction import BackgroundCompactionStore, PromotionResult
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


class _IgnorePrefixPattern:
    pattern = r"^IGNORE:"

    def search(self, text, timeout=None):
        del timeout
        return text.startswith("IGNORE:")


def _config(db_path):
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_global_limit=2,
        background_compaction_max_passes=4,
        background_compaction_job_timeout_seconds=120,
        background_compaction_lease_seconds=300,
        background_compaction_backlog_tokens=1,
        # These tests exercise provider-backed emergency mechanics with tiny
        # fixtures; opt out of the production micro-range projection policy.
        background_compaction_min_provider_source_tokens=1,
        background_compaction_prepare_ratio=0.55,
        fresh_tail_count=1,
        fresh_tail_max_tokens=0,
        leaf_chunk_tokens=1,
        context_reserve_tokens=100_000,
    )


def test_prepared_publisher_winning_emergency_race_preserves_live_tail(
    tmp_path, monkeypatch
):
    """The CAS loser overlays the caller's exact post-watermark messages."""
    db_path = tmp_path / "emergency-race.db"
    engine = LCMEngine(_config(db_path))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    live_tail = {
        "role": "user",
        "content": [{"type": "text", "text": "exact emergency live tail"}],
        "provider_metadata": {"cache": {"mode": "ephemeral"}},
        "client_turn_metadata": {"nonce": "emergency-race"},
    }
    messages = [
        {"role": "user", "content": "old-0 x x x"},
        {"role": "assistant", "content": "old-1 x x x"},
        {"role": "user", "content": "old-2 x x x"},
        live_tail,
    ]
    engine.ingest(messages)
    ids = [
        int(row[0])
        for row in engine._store.connection.execute(
            "SELECT store_id FROM messages WHERE session_id='s' ORDER BY store_id"
        ).fetchall()
    ]
    winning_frontier = ids[-2]
    observed: dict[str, object] = {}

    def lose_foreground(*_args, **_kwargs):
        observed["permits_during_publication"] = (
            engine.get_background_compaction_status()["active_provider_permits"]
        )
        engine._lifecycle.advance_frontier("c", "s", winning_frontier)
        return PromotionResult(False, "frontier_mismatch")

    def project(store_id, *, preserve_messages=None):
        observed["store_id"] = store_id
        observed["preserve_messages"] = preserve_messages
        return [dict(live_tail)]

    monkeypatch.setattr(engine, "_publish_foreground_background_node", lose_foreground)
    monkeypatch.setattr(engine, "_project_durable_active_tail", project)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: ("foreground emergency summary", 1),
    )
    try:
        engine.compress(messages, current_tokens=340_000)
        preserved = observed.get("preserve_messages")
        assert isinstance(preserved, list)
        assert any(
            message.get("client_turn_metadata", {}).get("nonce")
            == "emergency-race"
            for message in preserved
        )
        assert observed["permits_during_publication"] == 1
        assert engine.get_background_compaction_status()[
            "active_provider_permits"
        ] == 0
    finally:
        engine.shutdown()


def test_automatic_emergency_has_a_hard_wall_when_provider_ignores_timeout(
    tmp_path, monkeypatch
):
    """An uncooperative provider cannot block the foreground past its budget."""
    db_path = tmp_path / "emergency-hard-wall.db"
    engine = LCMEngine(_config(db_path))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"raw-{index} x x x"}
        for index in range(6)
    ]
    release = threading.Event()
    returned = threading.Event()

    def blocks(**_kwargs):
        try:
            release.wait(timeout=5)
            return "too late", 1
        finally:
            returned.set()

    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS", 0.05
    )
    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocks)
    try:
        started = time.monotonic()
        try:
            engine.compress(messages, current_tokens=340_000)
        except TimeoutError:
            pass
        assert time.monotonic() - started < 1.0
        assert engine._dag.get_session_node_count("s") == 0
        assert engine._store.get_session_count("s") == len(messages)
        assert engine.get_background_compaction_status()[
            "active_provider_permits"
        ] == 1
    finally:
        release.set()
        returned.wait(timeout=2)
        deadline = time.monotonic() + 2
        while (
            engine.get_background_compaction_status()["active_provider_permits"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert engine.get_background_compaction_status()[
            "active_provider_permits"
        ] == 0
        engine.shutdown()


def test_emergency_global_provider_limit_falls_back_without_provider_io(
    tmp_path,
    monkeypatch,
):
    """An occupied global slot never makes the bounded foreground wait."""
    db_path = tmp_path / "emergency-global-limit.db"
    config = _config(db_path)
    config.background_compaction_global_limit = 1
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    blocker = BackgroundCompactionStore(db_path, config=config)
    permit = blocker.acquire_emergency_provider_permit(
        conversation_id="other-conversation",
        owner="other-process",
        global_limit=1,
    )
    assert permit is not None
    messages = [
        {"role": "user", "content": f"raw-{index} x x x"}
        for index in range(6)
    ]
    provider_calls: list[dict] = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: provider_calls.append(kwargs) or ("must not run", 1),
    )
    try:
        started = time.monotonic()
        result = engine.compress(messages, current_tokens=340_000)
        assert time.monotonic() - started < 1.0
        assert provider_calls == []
        assert result
        assert engine._dag.get_session_node_count("s") == 0
        assert blocker.status()["active_provider_permits"] == 1
    finally:
        blocker.release_provider_permit(
            permit.job_id,
            owner=permit.lease_owner,
            generation=permit.lease_generation,
        )
        blocker.close()
        engine.shutdown()


def test_emergency_provider_permit_excludes_same_conversation_across_stores(
    tmp_path,
):
    """Independent processes cannot reserve the same conversation concurrently."""
    db_path = tmp_path / "emergency-conversation-exclusion.db"
    config = _config(db_path)
    first = BackgroundCompactionStore(db_path, config=config)
    second = BackgroundCompactionStore(db_path, config=config)
    try:
        permit = first.acquire_emergency_provider_permit(
            conversation_id="same-conversation",
            owner="process-a",
            global_limit=2,
        )
        assert permit is not None
        assert second.acquire_emergency_provider_permit(
            conversation_id="same-conversation",
            owner="process-b",
            global_limit=2,
        ) is None
        other = second.acquire_emergency_provider_permit(
            conversation_id="different-conversation",
            owner="process-b",
            global_limit=2,
        )
        assert other is not None
    finally:
        first.close()
        second.close()


def test_emergency_does_not_overlap_active_background_provider(
    tmp_path,
    monkeypatch,
):
    """A background lease for the conversation forces local overflow recovery."""
    db_path = tmp_path / "emergency-background-exclusion.db"
    config = _config(db_path)
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"raw-{index} x x x"}
        for index in range(6)
    ]
    provider_calls: list[dict] = []
    try:
        engine.ingest(messages)
        claimed = engine._background_compaction.claim_next_job(
            owner="background-process",
            global_limit=2,
        )
        assert claimed is not None
        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: provider_calls.append(kwargs) or ("must not run", 1),
        )
        result = engine.compress(messages, current_tokens=340_000)
        assert result
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
        assert engine.get_background_compaction_status()[
            "active_provider_permits"
        ] == 1
    finally:
        if "claimed" in locals() and claimed is not None:
            engine._background_compaction.release_provider_permit(
                claimed.job_id,
                owner="background-process",
                generation=claimed.lease_generation,
            )
        engine.shutdown()


def test_emergency_never_retires_hidden_concurrent_raw_not_seen_by_provider(
    tmp_path,
    monkeypatch,
):
    """Caller and durable intervals must match before any provider request."""
    db_path = tmp_path / "emergency-hidden-raw.db"
    engine = LCMEngine(_config(db_path))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    engine._store.append(
        "s",
        {"role": "user", "content": "hidden concurrent durable raw"},
        token_estimate=10,
        conversation_id="c",
    )
    visible = [
        {"role": "user", "content": f"visible-{index} x x x"}
        for index in range(6)
    ]
    engine.ingest(visible)
    provider_calls: list[dict] = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: provider_calls.append(kwargs) or ("must not publish", 1),
    )
    try:
        result = engine.compress(visible, current_tokens=340_000)
        assert result
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
        assert engine._lifecycle.get_by_conversation(
            "c"
        ).current_frontier_store_id == 0
        assert engine._store.get_session_count("s") == 7
    finally:
        engine.shutdown()


def test_emergency_rejects_source_content_mutation_after_provider_snapshot(
    tmp_path,
    monkeypatch,
):
    """Publication revalidates durable content, not only ordered IDs."""
    db_path = tmp_path / "emergency-source-mutation.db"
    engine = LCMEngine(_config(db_path))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"ORIGINAL-{index} x x x"}
        for index in range(6)
    ]
    engine.ingest(messages)
    first_store_id = int(
        engine._store.connection.execute(
            "SELECT MIN(store_id) FROM messages WHERE session_id='s'"
        ).fetchone()[0]
    )
    provider_inputs: list[str] = []

    def mutate_after_snapshot(**kwargs):
        provider_inputs.append(str(kwargs.get("text") or ""))
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE messages SET content=? WHERE store_id=?",
                ("MUTATED AFTER PROVIDER SNAPSHOT", first_store_id),
            )
            conn.commit()
        finally:
            conn.close()
        return "summary of original snapshot", 1

    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        mutate_after_snapshot,
    )
    try:
        result = engine.compress(messages, current_tokens=340_000)
        assert result
        assert len(provider_inputs) == 1
        assert "ORIGINAL-0" in provider_inputs[0]
        assert "MUTATED AFTER PROVIDER SNAPSHOT" not in provider_inputs[0]
        assert engine._dag.get_session_node_count("s") == 0
        assert engine._lifecycle.get_by_conversation(
            "c"
        ).current_frontier_store_id == 0
        assert engine._store.connection.execute(
            "SELECT content FROM messages WHERE store_id=?",
            (first_store_id,),
        ).fetchone()[0] == "MUTATED AFTER PROVIDER SNAPSHOT"
    finally:
        engine.shutdown()


def test_preflight_uses_last_full_prompt_pressure_to_publish_prepared_work(tmp_path):
    """System/tool-schema overhead cannot hide the fixed publication trigger."""
    db_path = tmp_path / "preflight-overhead.db"
    engine = LCMEngine(_config(db_path))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": f"small-{index}"}
        for index in range(6)
    ]
    engine.last_prompt_tokens = 280_000
    try:
        engine.ingest(messages)
        prepared = engine.run_background_compaction_once(
            summarizer=lambda **_kwargs: ("prepared summary", 1)
        )
        assert prepared is not None and prepared.state == "prepared"
        assert engine.should_compress_preflight(messages) is True
    finally:
        engine.shutdown()


def test_prepared_publication_keeps_prompt_overhead_in_emergency_recheck(
    tmp_path, monkeypatch
):
    """Publishing a small batch cannot erase unchanged provider overhead."""
    db_path = tmp_path / "emergency-overhead.db"
    config = _config(db_path)
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=1_000_000)
    messages = [
        {"role": "user", "content": f"small-{index} x x x"}
        for index in range(10)
    ]
    engine.last_prompt_tokens = 950_000
    foreground_calls: list[dict] = []
    try:
        engine.ingest(messages)
        prepared = engine.run_background_compaction_once(
            summarizer=lambda **_kwargs: ("prepared summary", 1)
        )
        assert prepared is not None and prepared.state == "prepared"
        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: foreground_calls.append(kwargs)
            or ("emergency fallback summary", 1),
        )
        engine.compress(messages, current_tokens=950_000)
        assert len(foreground_calls) == 1
    finally:
        engine.shutdown()


def test_sibling_frontier_adoption_that_clears_emergency_avoids_provider_call(
    tmp_path, monkeypatch
):
    """Emergency eligibility is recalculated after adopting a winning sibling."""
    db_path = tmp_path / "adoption-clears-emergency.db"
    engine = LCMEngine(_config(db_path))
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    original = [
        {"role": "user", "content": f"large-old-{index}"}
        for index in range(6)
    ]
    adopted = [
        {"role": "user", "content": f"small-tail-{index}"}
        for index in range(4)
    ]
    provider_calls: list[dict] = []

    def token_count(messages):
        return 340_000 if len(messages) >= 6 else 100

    monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", token_count)
    monkeypatch.setattr(
        engine,
        "_adopt_committed_background_frontier",
        lambda _messages: [dict(message) for message in adopted],
    )
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: provider_calls.append(kwargs) or ("should not run", 1),
    )
    try:
        engine.compress(original, current_tokens=340_000)
        assert provider_calls == []
    finally:
        engine.shutdown()


def test_emergency_summary_lineage_covers_filtered_rows_that_frontier_consumes(
    tmp_path,
    monkeypatch,
):
    """Ignored provider input remains exactly recoverable through node lineage."""
    db_path = tmp_path / "emergency-filtered-lineage.db"
    config = _config(db_path)
    config.ignore_message_patterns = [r"^IGNORE:"]
    monkeypatch.setattr(
        "hermes_lcm.engine.compile_message_patterns",
        lambda _patterns: [_IgnorePrefixPattern()],
    )
    engine = LCMEngine(config)
    engine.on_session_start("s", conversation_id="c", context_length=372_000)
    messages = [
        {"role": "user", "content": "IGNORE: private instruction"},
        {"role": "assistant", "content": "dependent answer"},
        {"role": "user", "content": "summarize this"},
        {"role": "user", "content": "fresh tail"},
    ]
    provider_inputs: list[str] = []
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: provider_inputs.append(kwargs["text"])
        or ("safe emergency summary", 1),
    )
    try:
        engine.compress(messages, current_tokens=340_000)
        assert len(provider_inputs) == 1
        assert "private instruction" not in provider_inputs[0]
        assert "dependent answer" not in provider_inputs[0]
        nodes = engine._dag.get_session_nodes("s")
        assert len(nodes) == 1
        durable_ids = [
            int(row[0])
            for row in engine._store.connection.execute(
                "SELECT store_id FROM messages WHERE session_id='s' "
                "ORDER BY store_id"
            ).fetchall()
        ]
        consumed_ids = durable_ids[:-1]
        assert nodes[0].source_ids == consumed_ids
        assert engine._dag.source_message_ids(nodes[0].node_id, limit=100) == consumed_ids
        assert (nodes[0].earliest_at, nodes[0].latest_at) == (
            engine._store.get_time_bounds(consumed_ids)
        )
    finally:
        engine.shutdown()
