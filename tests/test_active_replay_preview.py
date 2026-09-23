from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

from hermes_lcm.active_replay_preview import (
    ActiveReplayPreviewPolicy,
    PREVIEW_PREFIX,
    project_externalized_tool_results,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import (
    load_externalized_payload,
    maybe_externalize_tool_output,
)
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens, count_tokens


def _config(tmp_path, **overrides):
    values = dict(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=100,
        active_replay_preview_enabled=True,
        active_replay_preview_min_tokens=100,
        active_replay_preview_head_tokens=40,
        active_replay_preview_tail_tokens=40,
        active_replay_preview_max_tokens=160,
        fresh_tail_count=1,
        focused_working_set_enabled=False,
    )
    values.update(overrides)
    return LCMConfig(**values)


def _policy(**overrides):
    values = dict(
        enabled=True,
        version="externalized-tool-preview-v1",
        min_tokens=100,
        head_tokens=40,
        tail_tokens=40,
        max_tokens=160,
        fresh_tail_count=1,
    )
    values.update(overrides)
    return ActiveReplayPreviewPolicy(**values)


def _payload() -> str:
    return (
        "HEAD_REQUIRED\n"
        + ("middle bulk unicode αβγ {\"k\":1}\n" * 400)
        + "MIDDLE_CRUCIAL_FACT=retrieve-exact-payload\n"
        + ("middle bulk unicode αβγ {\"k\":1}\n" * 400)
        + "TAIL_REQUIRED"
    )


def _messages(content, *, tool_name="read_file", call_id="call-1"):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
        {"role": "user", "content": "current turn protected by fresh tail"},
    ]


def _externalize(config, tmp_path, content, *, call_id="call-1", session="session"):
    result = maybe_externalize_tool_output(
        content,
        tool_call_id=call_id,
        session_id=session,
        config=config,
        hermes_home=str(tmp_path),
        force=True,
    )
    assert result is not None
    return result


def test_preview_preserves_head_tail_reference_and_exact_recovery(tmp_path):
    config = _config(tmp_path)
    content = _payload()
    externalized = _externalize(config, tmp_path, content)
    path = Path(externalized["path"])
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    messages = _messages(content)

    decision = project_externalized_tool_results(
        messages,
        session_id="session",
        config=config,
        hermes_home=str(tmp_path),
        policy=_policy(),
    )

    assert decision.reason == "projected"
    assert decision.projected_results == 1
    assert decision.messages is not None
    preview = normalize_content_value(decision.messages[1]["content"])
    assert preview.startswith(PREVIEW_PREFIX)
    assert "HEAD_REQUIRED" in preview
    assert "TAIL_REQUIRED" in preview
    assert "MIDDLE_CRUCIAL_FACT" not in preview
    assert externalized["path"].name in preview
    assert count_tokens(preview) <= 160
    assert decision.projected_tokens < decision.source_tokens * 0.20
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    recovered = load_externalized_payload(
        path.name, config=config, hermes_home=str(tmp_path)
    )
    assert recovered["content"] == content
    assert "MIDDLE_CRUCIAL_FACT=retrieve-exact-payload" in recovered["content"]

    repeated = project_externalized_tool_results(
        decision.messages,
        session_id="session",
        config=config,
        hermes_home=str(tmp_path),
        policy=_policy(),
    )
    assert repeated.messages is None
    assert repeated.reason == "no_qualified_payload"


def test_missing_payload_pair_mismatch_recovery_and_media_fail_open(tmp_path):
    config = _config(tmp_path)
    content = _payload()
    cases = [
        _messages(content),
        _messages(content, tool_name="lcm_expand"),
        [
            _messages(content)[0],
            {"role": "tool", "tool_call_id": "wrong", "content": content},
            {"role": "user", "content": "tail"},
        ],
        _messages([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]),
    ]
    for messages in cases:
        decision = project_externalized_tool_results(
            messages,
            session_id="session",
            config=config,
            hermes_home=str(tmp_path),
            policy=_policy(),
        )
        assert decision.messages is None


def test_structured_text_shape_and_fresh_tail_are_preserved(tmp_path):
    config = _config(tmp_path)
    content = _payload()
    structured_content = [{"type": "text", "text": content}]
    _externalize(
        config,
        tmp_path,
        normalize_content_value(structured_content),
    )
    structured = _messages(structured_content)
    projected = project_externalized_tool_results(
        structured,
        session_id="session",
        config=config,
        hermes_home=str(tmp_path),
        policy=_policy(),
    )
    # Durable identity is over the normalized text, so supported structured
    # text receives the same provider-compatible block type.
    assert projected.messages is not None
    assert projected.messages[1]["content"][0]["type"] == "text"
    assert PREVIEW_PREFIX in projected.messages[1]["content"][0]["text"]

    protected = project_externalized_tool_results(
        _messages(content)[:2],
        session_id="session",
        config=config,
        hermes_home=str(tmp_path),
        policy=_policy(fresh_tail_count=1),
    )
    assert protected.messages is None


def test_existing_redaction_and_ignore_policy_excludes_preview(tmp_path):
    sensitive = "HEAD api_key=synthetic-secret\n" + ("bulk\n" * 800) + "TAIL"
    ignored = "NEVER_PREVIEW\n" + ("bulk\n" * 800) + "TAIL"
    config = _config(
        tmp_path,
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key"],
        ignore_message_patterns=[r"NEVER_PREVIEW"],
    )
    home = tmp_path / "policy-home"
    _externalize(config, home, sensitive, call_id="sensitive-call")
    _externalize(
        config,
        home,
        ignored,
        call_id="ignored-call",
        session="ignored-session",
    )
    engine = LCMEngine(config, hermes_home=str(home))
    try:
        engine.on_session_start(
            "session", conversation_id="conversation", context_length=200_000
        )
        sensitive_messages = _messages(sensitive, call_id="sensitive-call")
        ignored_messages = _messages(ignored, call_id="ignored-call")
        assert engine.select_context(
            sensitive_messages,
            conversation_messages=sensitive_messages,
            incoming_message=sensitive_messages[-1],
            budget_tokens=200_000,
        ) is None
        engine.on_session_start(
            "ignored-session",
            conversation_id="ignored-conversation",
            context_length=200_000,
        )
        assert engine.select_context(
            ignored_messages,
            conversation_messages=ignored_messages,
            incoming_message=ignored_messages[-1],
            budget_tokens=200_000,
        ) is None
        assert engine._last_active_replay_preview["reason"] == "disabled"
    finally:
        engine.shutdown()


def test_engine_flag_off_equivalence_and_pre_measurement_projection(tmp_path):
    content = _payload()
    off = _config(
        tmp_path,
        database_path=str(tmp_path / "off.db"),
        active_replay_preview_enabled=False,
    )
    off_engine = LCMEngine(off, hermes_home=str(tmp_path / "off-home"))
    off_engine.on_session_start("off", conversation_id="off", context_length=200_000)
    assert off_engine.select_context(_messages(content)) is None
    off_engine.shutdown()

    config = _config(tmp_path, database_path=str(tmp_path / "on.db"))
    on_home = tmp_path / "on-home"
    _externalize(config, on_home, content)
    engine = LCMEngine(config, hermes_home=str(on_home))
    engine.on_session_start("session", conversation_id="conversation", context_length=200_000)
    messages = _messages(content)
    selected = engine.select_context(
        messages,
        conversation_messages=messages,
        incoming_message=messages[-1],
        budget_tokens=200_000,
    )
    assert selected is not None
    assert PREVIEW_PREFIX in normalize_content_value(selected[1]["content"])
    assert count_messages_tokens(selected) < count_messages_tokens(messages) * 0.20
    assert engine._store.get_session_count("session") == len(messages)
    stored = engine._store.get_session_messages("session")
    stored_tool = next(row for row in stored if row.get("role") == "tool")
    assert content not in normalize_content_value(stored_tool.get("content"))
    assert int(stored_tool.get("token_estimate") or 0) == count_message_tokens(messages[1])
    assert sum(int(row.get("token_estimate") or 0) for row in stored) > (
        count_messages_tokens(selected) * 5
    )
    selected_again = engine.select_context(
        messages,
        conversation_messages=messages,
        incoming_message=messages[-1],
        budget_tokens=200_000,
    )
    assert selected_again == selected
    assert content not in normalize_content_value(selected_again[1]["content"])
    assert engine.get_status()["active_replay_preview"]["last_decision"][
        "reason"
    ] == "projected"
    engine.shutdown()


def test_preview_allowlist_and_policy_version_load_from_environment(monkeypatch):
    monkeypatch.setenv(
        "LCM_ACTIVE_REPLAY_PREVIEW_SESSION_ALLOWLIST",
        "session-a, session-b, session-a,  ",
    )
    monkeypatch.setenv(
        "LCM_ACTIVE_REPLAY_PREVIEW_POLICY_VERSION",
        "externalized-tool-preview-v1-canary",
    )

    config = LCMConfig.from_env()

    assert config.active_replay_preview_session_allowlist == [
        "session-a",
        "session-b",
    ]
    assert (
        config.active_replay_preview_policy_version
        == "externalized-tool-preview-v1-canary"
    )


def test_background_prepare_ratio_loads_from_config_yaml(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "lcm:\n  background_compaction_prepare_ratio: 0.75\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LCM_BACKGROUND_COMPACTION_PREPARE_RATIO", "0.55")

    config = LCMConfig.from_env()

    assert config.background_compaction_prepare_ratio == 0.75
    assert (
        config.config_sources["background_compaction_prepare_ratio"]
        == "config_yaml:lcm.background_compaction_prepare_ratio"
    )
    assert config.ignored_config_yaml_lcm_keys == []


def test_preview_matcher_supports_exact_and_date_prefix_and_fails_closed(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home-prefix"
    hermes_home.mkdir()
    date_prefix = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_"
    (hermes_home / "config.yaml").write_text(
        "\n".join(
            [
                "lcm:",
                "  active_replay_preview_enabled: true",
                "  active_replay_preview_session_allowlist:",
                "    - exact-session",
                "  active_replay_preview_session_prefix_allowlist:",
                f"    - '{date_prefix}'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("LCM_ACTIVE_REPLAY_PREVIEW_ENABLED", raising=False)
    monkeypatch.delenv(
        "LCM_ACTIVE_REPLAY_PREVIEW_SESSION_ALLOWLIST", raising=False
    )
    monkeypatch.delenv(
        "LCM_ACTIVE_REPLAY_PREVIEW_SESSION_PREFIX_ALLOWLIST", raising=False
    )

    config = LCMConfig.from_env()
    engine = LCMEngine(config, hermes_home=str(hermes_home))
    try:
        engine.on_session_start(
            "exact-session", conversation_id="exact", context_length=200_000
        )
        assert engine._active_replay_preview_policy().enabled is True
        engine.on_session_start(
            "20260923_120000_abcdef",
            conversation_id="dated",
            context_length=200_000,
        )
        assert engine._active_replay_preview_policy().enabled is True
        engine.on_session_start(
            "api_1790134213_7b0e41c3",
            conversation_id="api",
            context_length=200_000,
        )
        assert engine._active_replay_preview_policy().enabled is False
    finally:
        engine.shutdown()

    (hermes_home / "config.yaml").write_text("lcm: [\n", encoding="utf-8")
    fail_closed = LCMConfig.from_env()
    denied = LCMEngine(fail_closed, hermes_home=str(hermes_home))
    try:
        denied.on_session_start(
            "exact-session", conversation_id="denied", context_length=200_000
        )
        assert denied._active_replay_preview_policy().enabled is False
        denied.on_session_start(
            "20260923_120000_abcdef",
            conversation_id="denied-dated",
            context_length=200_000,
        )
        assert denied._active_replay_preview_policy().enabled is False
    finally:
        denied.shutdown()


def test_preview_allowlist_loads_from_config_yaml_and_fails_closed(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
lcm:
  active_replay_preview_enabled: true
  active_replay_preview_session_allowlist:
    - build-session
    - long-session
    - build-session
""".strip()
        + "\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("LCM_ACTIVE_REPLAY_PREVIEW_ENABLED", raising=False)
    monkeypatch.delenv(
        "LCM_ACTIVE_REPLAY_PREVIEW_SESSION_ALLOWLIST", raising=False
    )

    config = LCMConfig.from_env()

    assert config.active_replay_preview_enabled is True
    assert config.active_replay_preview_session_allowlist == [
        "build-session",
        "long-session",
    ]
    assert config.ignored_config_yaml_lcm_keys == []

    (hermes_home / "config.yaml").write_text(
        """
lcm:
  active_replay_preview_enabled: true
  active_replay_preview_session_allowlist: []
""".strip()
        + "\n"
    )

    fail_closed = LCMConfig.from_env()
    assert fail_closed.active_replay_preview_enabled is False
    assert fail_closed.active_replay_preview_session_allowlist == []


def test_preview_does_not_cancel_raw_backlog_job_or_change_captured_source(tmp_path):
    content = _payload()
    config = _config(
        tmp_path,
        database_path=str(tmp_path / "backlog.db"),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=100,
        background_compaction_min_provider_source_tokens=1,
        background_compaction_job_timeout_seconds=10,
        background_compaction_lease_seconds=30,
    )
    home = tmp_path / "backlog-home"
    externalized = _externalize(config, home, content)
    payload_hash = hashlib.sha256(Path(externalized["path"]).read_bytes()).hexdigest()
    engine = LCMEngine(config, hermes_home=str(home))
    try:
        engine.on_session_start(
            "session", conversation_id="conversation", context_length=200_000
        )
        messages = _messages(content)
        selected = engine.select_context(
            messages,
            conversation_messages=messages,
            incoming_message=messages[-1],
            budget_tokens=200_000,
        )
        assert selected is not None
        assert count_messages_tokens(selected) < count_messages_tokens(messages) * 0.20

        # The normal post-turn ingest path must still capture immutable raw
        # backlog even though visible request pressure was reduced.
        engine.ingest(messages)
        row = engine._store.connection.execute(
            """
            SELECT source_token_count, source_store_ids, state
            FROM lcm_compaction_jobs
            WHERE session_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            ("session",),
        ).fetchone()
        assert row is not None
        assert int(row[0]) >= count_message_tokens(messages[1])
        assert str(row[2]) in {"queued", "prepared"}
        assert hashlib.sha256(Path(externalized["path"]).read_bytes()).hexdigest() == payload_hash
    finally:
        engine.shutdown()


def test_host_selection_hook_projects_before_complete_request_measurement(tmp_path):
    # Plugin tests run with the plugin directory as the import root, where its
    # legacy ``tools.py`` would otherwise shadow the host's ``tools`` package.
    # Put the selected host checkout first only for the integration import.
    import agent

    core_root = str(Path(agent.__file__).resolve().parent.parent)
    sys.path.insert(0, core_root)
    try:
        from agent.conversation_loop import _apply_context_engine_selection
    finally:
        sys.path.remove(core_root)
    from agent.model_metadata import estimate_messages_tokens_rough

    content = _payload()
    config = _config(tmp_path)
    home = tmp_path / "host-home"
    _externalize(config, home, content)
    engine = LCMEngine(config, hermes_home=str(home))
    engine.on_session_start("session", conversation_id="conversation", context_length=200_000)
    messages = _messages(content)
    agent = SimpleNamespace(context_compressor=engine, session_id="session")

    selected = _apply_context_engine_selection(
        agent,
        [dict(message) for message in messages],
        messages,
        messages[-1],
        logger=logging.getLogger("preview-host-fixture"),
    )

    control_tokens = estimate_messages_tokens_rough(messages)
    selected_tokens = estimate_messages_tokens_rough(selected)
    assert selected_tokens < control_tokens * 0.20
    assert PREVIEW_PREFIX in normalize_content_value(selected[1]["content"])
    engine.shutdown()


def test_preview_policy_is_snapshotted_per_session(tmp_path):
    config = _config(tmp_path)
    engine = LCMEngine(config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("session-a", conversation_id="a", context_length=200_000)
    assert engine._active_replay_preview_policy().enabled is True
    config.active_replay_preview_enabled = False
    assert engine._active_replay_preview_policy().enabled is True

    engine.on_session_start("session-b", conversation_id="b", context_length=200_000)
    assert engine._active_replay_preview_policy().enabled is False
    engine.on_session_start("session-a", conversation_id="a", context_length=200_000)
    assert engine._active_replay_preview_policy().enabled is True
    engine.shutdown()


def test_preview_can_be_enabled_for_one_exact_canary_session(tmp_path):
    config = _config(
        tmp_path,
        active_replay_preview_session_allowlist=["session-a"],
    )
    engine = LCMEngine(config, hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("session-a", conversation_id="a", context_length=200_000)
        assert engine._active_replay_preview_policy().enabled is True
        engine.on_session_start("session-b", conversation_id="b", context_length=200_000)
        assert engine._active_replay_preview_policy().enabled is False
    finally:
        engine.shutdown()
