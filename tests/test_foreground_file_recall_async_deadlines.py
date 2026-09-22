"""Hard deadline contracts for file projections, recall, and derived workers."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

import hermes_lcm.externalize as externalize_module
import hermes_lcm.ingest_protection as ingest_protection_module
import hermes_lcm.retrieval_core as retrieval_core_module
import hermes_lcm.summary_serialization as summary_serialization_module
from hermes_lcm import assertion_extraction, tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.reconcile import ForegroundDeadlineExceeded


def _import_engine():
    try:
        from hermes_lcm.engine import LCMEngine as engine_class

        return sys.modules["hermes_lcm.engine"], engine_class
    except (ImportError, ModuleNotFoundError):
        agent_module = sys.modules.get("agent")
        if agent_module is None:
            agent_module = ModuleType("agent")
            agent_module.__path__ = []
            sys.modules["agent"] = agent_module
        context_engine_module = ModuleType("agent.context_engine")

        class ContextEngine:
            def on_session_reset(self):
                return None

            def get_status(self):
                return {}

        context_engine_module.ContextEngine = ContextEngine
        sys.modules["agent.context_engine"] = context_engine_module
        agent_module.context_engine = context_engine_module
        sys.modules.pop("hermes_lcm.engine", None)
        from hermes_lcm.engine import LCMEngine as engine_class

        return sys.modules["hermes_lcm.engine"], engine_class


engine_module, LCMEngine = _import_engine()


class _FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _engine(tmp_path: Path, **overrides: object) -> LCMEngine:
    values = {
        "database_path": str(tmp_path / "deadline-derived.db"),
        "background_compaction_enabled": True,
        "background_compaction_workers": 0,
        "fresh_tail_count": 0,
        "large_output_externalization_enabled": True,
        "large_output_externalization_threshold_chars": 1,
        "large_output_externalization_path": str(tmp_path / "externalized"),
    }
    values.update(overrides)
    Path(str(values["large_output_externalization_path"])).mkdir(
        parents=True,
        exist_ok=True,
    )
    config = LCMConfig(**values)
    engine = LCMEngine(config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("s", conversation_id="c", context_length=100_000)
    return engine


def test_engine_bootstrap_resolves_externalization_path_without_creating_it(
    tmp_path,
):
    storage = tmp_path / "externalized"
    engine = LCMEngine(
        LCMConfig(
            database_path=str(tmp_path / "bootstrap.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_path=str(storage),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        assert engine._foreground_externalization_storage_dir == storage.resolve()
        assert not storage.exists()
    finally:
        engine.shutdown()


def test_deadline_ingest_does_not_create_missing_externalization_directory(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    engine = _engine(tmp_path)
    storage.rmdir()
    mkdir_entered = threading.Event()
    release_mkdir = threading.Event()
    caller_finished = threading.Event()
    result: dict[str, object] = {}
    real_mkdir = Path.mkdir

    def blocked_storage_mkdir(path: Path, *args, **kwargs):
        if path == storage:
            mkdir_entered.set()
            assert release_mkdir.wait(timeout=2)
        return real_mkdir(path, *args, **kwargs)

    def ingest() -> None:
        try:
            result["value"] = engine._ingest_messages(
                [{"role": "user", "content": "deadline payload stays inline"}],
                deadline=time.monotonic() + 0.1,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            result["error"] = exc
        finally:
            caller_finished.set()

    monkeypatch.setattr(Path, "mkdir", blocked_storage_mkdir)
    caller = threading.Thread(target=ingest, name="missing-storage-deadline")
    caller.start()
    try:
        assert caller_finished.wait(timeout=0.3)
        assert not mkdir_entered.is_set()
        assert result.get("error") is None
        rows = engine._store.get_session_messages("s")
        assert rows[-1]["content"] == "deadline payload stays inline"
        assert not storage.exists()
    finally:
        release_mkdir.set()
        caller.join(timeout=2)
        engine.shutdown()


def test_deadline_ingest_externalizes_when_storage_directory_already_exists(
    tmp_path,
):
    storage = tmp_path / "externalized"
    storage.mkdir()
    engine = _engine(tmp_path)
    try:
        engine._ingest_messages(
            [{"role": "user", "content": "deadline payload is externalized"}],
            deadline=time.monotonic() + 1.0,
        )

        rows = engine._store.get_session_messages("s")
        ref = externalize_module.extract_externalized_ref(rows[-1]["content"])
        assert ref
        assert (storage / ref).is_file()
    finally:
        engine.shutdown()


def _blocking_payload_write(monkeypatch: pytest.MonkeyPatch):
    entered = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}

    def blocked_write(_path, _payload, **kwargs):
        observed.update(kwargs)
        entered.set()
        assert release.wait(timeout=2)
        if kwargs.get("cancel_event") is not None and kwargs["cancel_event"].is_set():
            raise TimeoutError("cancelled synthetic payload write")

    monkeypatch.setattr(
        externalize_module,
        "_write_externalized_payload",
        blocked_write,
    )
    return entered, release, observed


def _block_after_real_externalizer_returns(
    monkeypatch: pytest.MonkeyPatch,
    module,
    name: str,
):
    entered = threading.Event()
    release = threading.Event()
    observed: dict[str, object] = {}
    real_externalizer = getattr(module, name)

    def blocked_externalizer(*args, **kwargs):
        result = real_externalizer(*args, **kwargs)
        observed["result"] = result
        observed["ownership"] = kwargs.get("ownership")
        entered.set()
        assert release.wait(timeout=2)
        return result

    monkeypatch.setattr(module, name, blocked_externalizer)
    return entered, release, observed


def _block_after_externalization_intent_registration(
    monkeypatch: pytest.MonkeyPatch,
):
    entered = threading.Event()
    release = threading.Event()
    real_register = externalize_module.ExternalizationOwnership.register_pending

    def blocked_register(ownership, path, serialized_payload):
        accepted = real_register(ownership, path, serialized_payload)
        entered.set()
        assert release.wait(timeout=2)
        return accepted

    monkeypatch.setattr(
        externalize_module.ExternalizationOwnership,
        "register_pending",
        blocked_register,
    )
    return entered, release


def _wait_for_foreground_projection_worker(engine: LCMEngine) -> None:
    assert engine._foreground_projection_call_slot.acquire(timeout=2)
    engine._foreground_projection_call_slot.release()


def _call_while_operation_is_blocked(
    operation,
    *,
    entered: threading.Event,
    release: threading.Event,
):
    finished = threading.Event()
    result: dict[str, object] = {}

    def invoke() -> None:
        try:
            result["value"] = operation()
        except BaseException as exc:  # noqa: BLE001 - asserted by the caller
            result["error"] = exc
        finally:
            finished.set()

    caller = threading.Thread(target=invoke, name="deadline-file-caller")
    caller.start()
    assert entered.wait(timeout=2)
    try:
        completed_while_write_blocked = finished.wait(timeout=0.25)
    finally:
        release.set()
        caller.join(timeout=2)
    assert not caller.is_alive()
    return completed_while_write_blocked, result


def test_guarded_externalized_payload_is_not_published_when_fsync_expires_deadline(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    storage.mkdir()
    clock = _FakeClock()
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_path=str(storage),
    )

    def expire_on_file_fsync(_fd: int) -> None:
        clock.value = 101.0

    monkeypatch.setattr(externalize_module.os, "fsync", expire_on_file_fsync)
    monkeypatch.setattr(externalize_module, "_fsync_directory", lambda _path: None)

    result = externalize_module.externalize_ingest_payload(
        "lossless payload",
        role="user",
        session_id="s",
        field_path="content",
        config=config,
        deadline=101.0,
        now=clock,
    )

    assert result is None
    assert list(storage.iterdir()) == []


def test_automatic_ingest_returns_at_wall_while_payload_fsync_is_blocked(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    entered, release, observed = _blocking_payload_write(monkeypatch)
    message = {"role": "user", "content": "payload that must remain lossless"}
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine.compress([message], current_tokens=10_000),
            entered=entered,
            release=release,
        )

        assert completed
        assert result.get("error") is None
        assert result.get("value") == [message]
        assert engine._store.get_session_count("s") == 0
        assert engine._dag.get_session_node_count("s") == 0
        assert observed["deadline"] is not None
        assert observed["cancel_event"].is_set()
    finally:
        release.set()
        engine.shutdown()


def test_automatic_ingest_active_stub_returns_at_wall_while_write_is_blocked(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        large_output_externalization_threshold_chars=1_000_000,
        large_output_active_replay_stubbing_enabled=True,
        large_output_active_replay_stub_threshold_tokens=1,
    )
    entered, release, observed = _blocking_payload_write(monkeypatch)
    message = {"role": "tool", "tool_call_id": "call-1", "content": "tool payload"}
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine.compress([message], current_tokens=10_000),
            entered=entered,
            release=release,
        )

        assert completed
        assert result.get("error") is None
        assert result.get("value") == [message]
        assert engine._store.get_session_count("s") == 0
        assert engine._dag.get_session_node_count("s") == 0
        assert observed["deadline"] is not None
        assert observed["cancel_event"].is_set()
    finally:
        release.set()
        engine.shutdown()


@pytest.mark.parametrize(
    ("path_kind", "message", "overrides"),
    [
        (
            "protection",
            {"role": "user", "content": "data:image/png;base64," + ("A" * 4096)},
            {},
        ),
        (
            "quarantine",
            {"role": "assistant", "content": "A" * 65_536},
            {},
        ),
        (
            "active-stub",
            {"role": "tool", "tool_call_id": "call-stub", "content": "tool payload"},
            {
                "large_output_externalization_threshold_chars": 1_000_000,
                "large_output_active_replay_stubbing_enabled": True,
                "large_output_active_replay_stub_threshold_tokens": 1,
            },
        ),
    ],
)
def test_abandoned_ingest_workers_never_attempt_final_payload_rename(
    tmp_path,
    monkeypatch,
    path_kind,
    message,
    overrides,
):
    engine = _engine(tmp_path, **overrides)
    entered, release = _block_after_externalization_intent_registration(
        monkeypatch
    )
    visibility_attempted = threading.Event()
    real_replace = Path.replace

    def observe_final_rename(source: Path, target: Path) -> Path:
        if Path(target).suffix == ".json":
            visibility_attempted.set()
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_final_rename)
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine.compress([message], current_tokens=10_000),
            entered=entered,
            release=release,
        )

        assert completed, path_kind
        assert result.get("error") is None
        assert result.get("value") == [message]
        _wait_for_foreground_projection_worker(engine)
        assert not visibility_attempted.is_set(), path_kind
        assert list((tmp_path / "externalized").glob("*.json")) == []
        assert engine._store.get_session_count("s") == 0
    finally:
        release.set()
        engine.shutdown()


def test_ingest_publish_is_rolled_back_when_raw_append_fails(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    storage = tmp_path / "externalized"
    message = {"role": "user", "content": "data:image/png;base64," + ("A" * 4096)}

    def fail_append(*_args, **_kwargs):
        assert list(storage.glob("*.json"))
        raise ForegroundDeadlineExceeded("ingest", "synthetic precommit failure")

    monkeypatch.setattr(engine, "_append_ingest_batch_with_deadline", fail_append)
    try:
        with pytest.raises(ForegroundDeadlineExceeded, match="precommit"):
            engine._ingest_messages(
                [message],
                deadline=time.monotonic() + 1.0,
            )
        assert engine._store.get_session_count("s") == 0
        assert list(storage.glob("*.json")) == []
        assert list(storage.glob(".*.prepared")) == []
    finally:
        engine.shutdown()


def test_ingest_keeps_published_file_when_deadline_expires_after_raw_commit(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    storage = tmp_path / "externalized"
    message = {"role": "user", "content": "data:image/png;base64," + ("A" * 4096)}
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    real_append = engine._append_ingest_batch_with_deadline

    def append_then_expire(*args, **kwargs):
        store_ids = real_append(*args, **kwargs)
        assert list(storage.glob("*.json"))
        clock.value = 101.0
        return store_ids

    monkeypatch.setattr(
        engine,
        "_append_ingest_batch_with_deadline",
        append_then_expire,
    )
    try:
        with pytest.raises(ForegroundDeadlineExceeded):
            engine._ingest_messages([message], deadline=101.0)
        stored = engine._store.get_session_messages("s")
        assert len(stored) == 1
        ref = externalize_module.extract_externalized_ref(
            str(stored[0]["content"])
        )
        assert ref
        assert (storage / ref).is_file()
    finally:
        engine.shutdown()


def test_automatic_ingest_skips_optional_metadata_writes_after_raw_commit(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        large_output_externalization_threshold_chars=1_000_000,
    )
    message = {"role": "user", "content": "raw row must not await replay metadata"}
    raw_committed = threading.Event()
    metadata_entered = threading.Event()
    metadata_release = threading.Event()
    caller_finished = threading.Event()
    result: dict[str, object] = {}
    metadata_calls: list[tuple[str, ...]] = []
    real_append = engine._append_ingest_batch_with_deadline
    real_write_metadata = engine._store.write_metadata_json

    def append_then_signal(*args, **kwargs):
        store_ids = real_append(*args, **kwargs)
        raw_committed.set()
        return store_ids

    def blocked_metadata_write(keys, serialized, **kwargs):
        metadata_calls.append(tuple(keys))
        metadata_entered.set()
        assert metadata_release.wait(timeout=2)
        return real_write_metadata(keys, serialized, **kwargs)

    def invoke() -> None:
        try:
            result["value"] = engine.compress([message], current_tokens=10_000)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            result["error"] = exc
        finally:
            caller_finished.set()

    monkeypatch.setattr(
        engine,
        "_append_ingest_batch_with_deadline",
        append_then_signal,
    )
    monkeypatch.setattr(
        engine._store,
        "write_metadata_json",
        blocked_metadata_write,
    )
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    caller = threading.Thread(target=invoke, name="deadline-metadata-caller")
    caller.start()
    try:
        assert raw_committed.wait(timeout=2)
        assert caller_finished.wait(timeout=0.2)
        assert not metadata_entered.is_set()
    finally:
        metadata_release.set()
        caller.join(timeout=2)

    try:
        assert not caller.is_alive()
        assert result.get("error") is None
        assert result.get("value") == [message]
        assert engine._store.get_session_count("s") == 1
        assert metadata_calls == []
    finally:
        engine.shutdown()


def test_multi_message_ingest_protection_rolls_back_files_when_later_item_expires(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(storage),
    )
    clock = _FakeClock()
    real_externalize = ingest_protection_module.maybe_externalize_payload
    calls = 0

    def expire_after_first(*args, **kwargs):
        nonlocal calls
        result = real_externalize(*args, **kwargs)
        calls += 1
        if calls == 1:
            clock.value = 101.0
        return result

    monkeypatch.setattr(
        ingest_protection_module,
        "maybe_externalize_payload",
        expire_after_first,
    )

    with pytest.raises(TimeoutError):
        ingest_protection_module.protect_messages_for_ingest(
            [
                {"role": "user", "content": "first payload"},
                {"role": "user", "content": "second payload"},
            ],
            config=config,
            session_id="s",
            hermes_home=str(tmp_path / "home"),
            deadline=101.0,
            now=clock,
        )

    assert list(storage.glob("*.json")) == []


def test_assistant_quarantine_batch_rolls_back_owned_file_when_deadline_expires(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(storage),
    )
    clock = _FakeClock()
    real_externalize = ingest_protection_module.externalize_ingest_payload
    calls = 0

    def expire_after_first(*args, **kwargs):
        nonlocal calls
        result = real_externalize(*args, **kwargs)
        calls += 1
        if calls == 1:
            clock.value = 101.0
        return result

    monkeypatch.setattr(
        ingest_protection_module,
        "externalize_ingest_payload",
        expire_after_first,
    )

    with pytest.raises(TimeoutError):
        ingest_protection_module.quarantine_suspicious_assistant_messages(
            [
                {"role": "assistant", "content": "A" * 65_536},
                {"role": "assistant", "content": "B" * 65_536},
            ],
            config=config,
            session_id="s",
            hermes_home=str(tmp_path / "home"),
            deadline=101.0,
            now=clock,
        )

    assert list(storage.glob("*.json")) == []


def test_aborted_ingest_batch_preserves_reused_ref_and_original_metadata(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(storage),
    )
    content = "same recovered result"
    existing = externalize_module.maybe_externalize_payload(
        content,
        kind="tool_result",
        tool_call_id="call-meta",
        session_id="s",
        role="tool",
        config=config,
        force=True,
    )
    assert existing is not None
    existing_path = Path(existing["path"])
    original_bytes = existing_path.read_bytes()
    host_result = tmp_path / "call-meta.txt"
    host_result.write_text(content, encoding="utf-8")
    preview = content[:10]
    marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(content):,} characters, 0.0 KB).\n"
        f"Full output saved to: {host_result}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        f"Preview (first {len(preview)} chars):\n{preview}\n"
        "</persisted-output>"
    )
    clock = _FakeClock()
    real_externalize = ingest_protection_module.maybe_externalize_payload
    calls = 0

    def expire_after_metadata_projection(*args, **kwargs):
        nonlocal calls
        result = real_externalize(*args, **kwargs)
        calls += 1
        if calls == 1:
            clock.value = 101.0
        return result

    monkeypatch.setattr(
        ingest_protection_module,
        "maybe_externalize_payload",
        expire_after_metadata_projection,
    )

    with pytest.raises(TimeoutError):
        ingest_protection_module.protect_messages_for_ingest(
            [
                {"role": "tool", "tool_call_id": "call-meta", "content": marker},
                {"role": "user", "content": "later payload"},
            ],
            config=config,
            session_id="s",
            hermes_home=str(tmp_path / "home"),
            deadline=101.0,
            now=clock,
        )

    assert existing_path.read_bytes() == original_bytes
    assert list(storage.glob("*.json")) == [existing_path]


def test_externalizer_distinguishes_new_payload_from_reused_ref(tmp_path):
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    kwargs = {
        "kind": "raw_payload",
        "session_id": "s",
        "role": "user",
        "config": config,
    }
    created = externalize_module.maybe_externalize_payload("payload", **kwargs)
    reused = externalize_module.maybe_externalize_payload("payload", **kwargs)

    assert created is not None and created["created"] is True
    assert reused is not None and reused["created"] is False
    assert created["path"] == reused["path"]


def test_ownership_rollback_retries_unlink_and_fsyncs_parent_without_index_lock(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    storage.mkdir()
    payload = storage / "owned.json"
    payload.write_text("owned", encoding="utf-8")
    ownership = externalize_module.ExternalizationOwnership()
    assert ownership.register_created(payload)
    real_unlink = Path.unlink
    unlink_calls = 0
    fsynced: list[Path] = []

    def fail_first_unlink(path: Path, *args, **kwargs):
        nonlocal unlink_calls
        if path == payload:
            unlink_calls += 1
            if unlink_calls == 1:
                raise OSError("synthetic transient unlink failure")
        return real_unlink(path, *args, **kwargs)

    def reject_index_invalidation(_storage_dir: Path) -> None:
        raise AssertionError("deadline cleanup must not acquire the index lock")

    monkeypatch.setattr(Path, "unlink", fail_first_unlink)
    monkeypatch.setattr(
        externalize_module,
        "_fsync_directory",
        lambda path: fsynced.append(path),
    )
    monkeypatch.setattr(
        externalize_module,
        "_invalidate_payload_lookup_index",
        reject_index_invalidation,
    )

    ownership.rollback()

    assert unlink_calls == 2
    assert not payload.exists()
    assert fsynced == [storage]


def test_ownership_rollback_surfaces_persistent_unlink_and_retains_recovery_state(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    storage.mkdir()
    payload = storage / "owned.json"
    payload.write_text("owned", encoding="utf-8")
    ownership = externalize_module.ExternalizationOwnership()
    assert ownership.register_created(payload)
    real_unlink = Path.unlink
    unlink_calls = 0

    def reject_owned_unlink(path: Path, *args, **kwargs):
        nonlocal unlink_calls
        if path == payload:
            unlink_calls += 1
            raise OSError("synthetic persistent unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject_owned_unlink)

    with pytest.raises(RuntimeError, match="cleanup"):
        ownership.rollback()

    assert unlink_calls == 2
    assert payload.is_file()

    # A loud failure retains ownership so an operator/retry can recover after
    # the underlying filesystem refusal is repaired.
    monkeypatch.setattr(Path, "unlink", real_unlink)
    ownership.rollback()
    assert not payload.exists()


def test_existing_payload_metadata_replace_restores_original_when_publish_expires(
    tmp_path,
    monkeypatch,
):
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    content = "existing payload"
    existing = externalize_module.maybe_externalize_payload(
        content,
        kind="tool_result",
        tool_call_id="call-existing",
        session_id="s",
        role="tool",
        config=config,
        force=True,
    )
    assert existing is not None
    existing_path = Path(existing["path"])
    original_bytes = existing_path.read_bytes()
    clock = _FakeClock()
    real_replace = Path.replace

    def expire_after_visible_replace(path, target):
        result = real_replace(path, target)
        if Path(target) == existing_path:
            clock.value = 101.0
        return result

    monkeypatch.setattr(Path, "replace", expire_after_visible_replace)

    updated = externalize_module.maybe_externalize_payload(
        content,
        kind="tool_result",
        tool_call_id="call-existing",
        session_id="s",
        role="tool",
        config=config,
        force=True,
        metadata={
            "persisted_output_source_path": str(tmp_path / "source.txt"),
            "persisted_output_expected_chars": len(content),
        },
        deadline=101.0,
        now=clock,
    )

    assert updated is None
    assert existing_path.read_bytes() == original_bytes


def test_existing_payload_replacement_durably_records_backup_and_cleanup(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    storage.mkdir()
    path = storage / "existing.json"
    path.write_text('{"value":"old"}', encoding="utf-8")
    events: list[str] = []
    real_link = externalize_module.os.link
    real_replace = Path.replace
    real_unlink = Path.unlink
    real_fsync_directory = externalize_module._fsync_directory

    def observe_link(source, target, *args, **kwargs):
        result = real_link(source, target, *args, **kwargs)
        if Path(target).name.endswith(".rollback"):
            events.append("backup_link")
        return result

    def observe_replace(source: Path, target: Path) -> Path:
        result = real_replace(source, target)
        if Path(target) == path:
            events.append("publish_replace")
        return result

    def observe_unlink(source: Path, *args, **kwargs):
        result = real_unlink(source, *args, **kwargs)
        if source.name.endswith(".rollback"):
            events.append("backup_unlink")
        return result

    def observe_fsync(directory: Path) -> None:
        real_fsync_directory(directory)
        if directory == storage:
            events.append("directory_fsync")

    monkeypatch.setattr(externalize_module.os, "link", observe_link)
    monkeypatch.setattr(Path, "replace", observe_replace)
    monkeypatch.setattr(Path, "unlink", observe_unlink)
    monkeypatch.setattr(externalize_module, "_fsync_directory", observe_fsync)

    externalize_module._replace_externalized_payload(
        path,
        {"value": "new"},
        deadline=time.monotonic() + 1.0,
        now=time.monotonic,
    )

    link_index = events.index("backup_link")
    publish_index = events.index("publish_replace")
    unlink_index = events.index("backup_unlink")
    assert "directory_fsync" in events[link_index + 1 : publish_index]
    assert "directory_fsync" in events[unlink_index + 1 :]
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "new"}


def test_existing_payload_failed_restore_retains_original_recovery_hardlink(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    config = LCMConfig(
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(storage),
    )
    content = "existing payload requiring recovery"
    existing = externalize_module.maybe_externalize_payload(
        content,
        kind="tool_result",
        tool_call_id="call-restore-failure",
        session_id="s",
        role="tool",
        config=config,
        force=True,
    )
    assert existing is not None
    existing_path = Path(existing["path"])
    original_bytes = existing_path.read_bytes()
    clock = _FakeClock()
    real_replace = Path.replace

    def expire_publish_but_fail_restore(path, target):
        source = Path(path)
        if source.name.endswith(".rollback"):
            raise OSError("synthetic rollback restore failure")
        result = real_replace(path, target)
        if Path(target) == existing_path:
            clock.value = 101.0
        return result

    monkeypatch.setattr(Path, "replace", expire_publish_but_fail_restore)

    with pytest.raises(RuntimeError, match="rollback"):
        externalize_module.maybe_externalize_payload(
            content,
            kind="tool_result",
            tool_call_id="call-restore-failure",
            session_id="s",
            role="tool",
            config=config,
            force=True,
            metadata={
                "persisted_output_source_path": str(tmp_path / "source.txt"),
                "persisted_output_expected_chars": len(content),
            },
            deadline=101.0,
            now=clock,
        )

    recovery_paths = list(storage.glob(".*.rollback"))
    assert len(recovery_paths) == 1
    assert recovery_paths[0].read_bytes() == original_bytes


def test_automatic_preflight_returns_at_wall_while_payload_fsync_is_blocked(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    entered, release, observed = _blocking_payload_write(monkeypatch)
    message = {"role": "user", "content": "preflight payload must remain lossless"}
    monkeypatch.setattr(
        "hermes_lcm.compaction._AUTOMATIC_FOREGROUND_MAX_SECONDS",
        0.05,
    )
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine.should_compress_preflight([message]),
            entered=entered,
            release=release,
        )

        assert completed
        assert result.get("error") is None
        assert result.get("value") is False
        assert engine._store.get_session_count("s") == 0
        assert engine._dag.get_session_node_count("s") == 0
        assert observed["deadline"] is not None
        assert observed["cancel_event"].is_set()
    finally:
        release.set()
        engine.shutdown()


def test_foreground_serialization_returns_at_wall_without_provider_or_publication(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    entered, release, observed = _blocking_payload_write(monkeypatch)
    provider_calls: list[str] = []
    monkeypatch.setattr(
        engine_module,
        "summarize_with_escalation",
        lambda **_kwargs: provider_calls.append("provider") or ("summary", 1),
    )
    messages = [{"role": "tool", "tool_call_id": "call-1", "content": "tool payload"}]
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._summarize_leaf_chunk_with_rescue(
                messages,
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert provider_calls == []
        assert engine._dag.get_session_node_count("s") == 0
        assert observed["cancel_event"].is_set()
    finally:
        release.set()
        engine.shutdown()


def test_cancelled_summary_serialization_removes_late_created_payload(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    entered, release, observed = _block_after_real_externalizer_returns(
        monkeypatch,
        summary_serialization_module,
        "maybe_externalize_tool_output",
    )
    messages = [{"role": "tool", "tool_call_id": "call-summary", "content": "summary payload"}]
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._summarize_leaf_chunk_with_rescue(
                messages,
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert observed.get("result") is not None
        _wait_for_foreground_projection_worker(engine)
        assert list((tmp_path / "externalized").glob("*.json")) == []
    finally:
        release.set()
        engine.shutdown()


def test_abandoned_projection_never_opens_a_payload_file_after_caller_returns(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    storage = tmp_path / "externalized"
    entered = threading.Event()
    release = threading.Event()
    caller_returned = threading.Event()
    late_write_open = threading.Event()
    real_open = externalize_module.os.open
    real_externalizer = summary_serialization_module.maybe_externalize_tool_output

    def block_write_open(path, flags, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent == storage and flags & externalize_module.os.O_CREAT:
            entered.set()
            assert release.wait(timeout=2)
            if caller_returned.is_set():
                late_write_open.set()
        return real_open(path, flags, *args, **kwargs)

    def block_after_intent(*args, **kwargs):
        result = real_externalizer(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=2)
        return result

    def summarize():
        try:
            return engine._summarize_leaf_chunk_with_rescue(
                [
                    {
                        "role": "tool",
                        "tool_call_id": "call-no-late-private-write",
                        "content": "payload intent",
                    }
                ],
                deadline=time.monotonic() + 0.05,
            )
        finally:
            caller_returned.set()

    monkeypatch.setattr(externalize_module.os, "open", block_write_open)
    monkeypatch.setattr(
        summary_serialization_module,
        "maybe_externalize_tool_output",
        block_after_intent,
    )
    try:
        completed, result = _call_while_operation_is_blocked(
            summarize,
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        _wait_for_foreground_projection_worker(engine)
        assert not late_write_open.is_set()
        assert list(storage.glob("*.json")) == []
        assert list(storage.glob(".*.prepared")) == []
    finally:
        release.set()
        engine.shutdown()


def test_foreground_payload_publication_stops_between_bounded_write_chunks(
    tmp_path,
    monkeypatch,
):
    storage = tmp_path / "externalized"
    storage.mkdir()
    path = storage / "bounded.json"
    ownership = externalize_module.ExternalizationOwnership(
        defer_publication=True,
        storage_dir=storage,
    )
    clock = _FakeClock()
    real_write = externalize_module.os.write
    write_calls = 0

    def expire_after_first_chunk(fd: int, data: bytes) -> int:
        nonlocal write_calls
        written = real_write(fd, data)
        write_calls += 1
        if write_calls == 1:
            clock.value = 101.0
        return written

    monkeypatch.setattr(externalize_module.os, "write", expire_after_first_chunk)
    externalize_module._write_externalized_payload(
        path,
        {"content": "x" * (256 * 1024)},
        deadline=101.0,
        now=clock,
        ownership=ownership,
    )

    with pytest.raises(externalize_module.ExternalizationDeadlineExceeded):
        ownership.publish(deadline=101.0, now=clock)
    ownership.rollback()

    assert write_calls == 1
    assert not path.exists()
    assert list(storage.iterdir()) == []


def test_cancelled_summary_serialization_preserves_reused_payload(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    content = "reused summary payload"
    existing = externalize_module.maybe_externalize_tool_output(
        content,
        tool_call_id="call-reused-summary",
        session_id="s",
        config=engine._config,
        hermes_home=engine._hermes_home,
        force=True,
    )
    assert existing is not None
    existing_path = Path(existing["path"])
    original_bytes = existing_path.read_bytes()
    entered, release, observed = _block_after_real_externalizer_returns(
        monkeypatch,
        summary_serialization_module,
        "maybe_externalize_tool_output",
    )
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call-reused-summary",
            "content": content,
        }
    ]
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._summarize_leaf_chunk_with_rescue(
                messages,
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert observed.get("result") is not None
        _wait_for_foreground_projection_worker(engine)
        assert existing_path.read_bytes() == original_bytes
        assert list(existing_path.parent.glob("*.json")) == [existing_path]
    finally:
        release.set()
        engine.shutdown()


def test_foreground_publication_does_not_acquire_payload_index_lock(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    storage = tmp_path / "externalized"

    def reject_invalidation(_storage_dir: Path) -> None:
        raise AssertionError("foreground publication must not wait on index state")

    monkeypatch.setattr(
        externalize_module,
        "_invalidate_payload_lookup_index",
        reject_invalidation,
    )
    try:
        serialized = engine._serialize_messages(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call-no-index-lock",
                    "content": "payload publication independent of index lock",
                }
            ],
            deadline=time.monotonic() + 1.0,
        )
        assert "Externalized tool output" in serialized
        assert len(list(storage.glob("*.json"))) == 1
    finally:
        engine.shutdown()


def test_abandoned_projection_never_attempts_final_visibility_rename(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    storage = tmp_path / "externalized"
    entered, release = _block_after_externalization_intent_registration(
        monkeypatch
    )
    caller_finished = threading.Event()
    result: dict[str, object] = {}
    visibility_attempted = threading.Event()
    real_replace = Path.replace

    def observe_final_rename(source: Path, target: Path) -> Path:
        if Path(target).suffix == ".json":
            visibility_attempted.set()
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_final_rename)

    def invoke() -> None:
        try:
            result["value"] = engine._serialize_messages(
                [
                    {
                        "role": "tool",
                        "tool_call_id": "call-paused-rename",
                        "content": "payload whose final rename pauses",
                    }
                ],
                deadline=time.monotonic() + 0.05,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            result["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=invoke, name="deadline-no-late-rename")
    caller.start()
    try:
        assert entered.wait(timeout=2)
        assert caller_finished.wait(timeout=0.25)
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert list(storage.glob("*.json")) == []
    finally:
        release.set()
        caller.join(timeout=2)
        _wait_for_foreground_projection_worker(engine)
        assert not visibility_attempted.is_set()
        assert list(storage.glob("*.json")) == []
        engine.shutdown()


def test_projection_workers_are_process_bounded_across_engine_clones(tmp_path):
    engines = [_engine(tmp_path / f"engine-{index}") for index in range(6)]
    start = threading.Event()
    release = threading.Event()
    entered_lock = threading.Lock()
    entered: set[int] = set()
    errors: list[BaseException] = []

    def invoke(index: int) -> None:
        assert start.wait(timeout=2)

        def blocked_projection() -> None:
            with entered_lock:
                entered.add(index)
            assert release.wait(timeout=2)

        try:
            engines[index]._call_foreground_projection(
                blocked_projection,
                deadline=time.monotonic() + 0.15,
                stage="cross-engine saturation",
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    callers = [
        threading.Thread(
            target=invoke,
            args=(index,),
            name=f"deadline-engine-clone-{index}",
        )
        for index in range(len(engines))
    ]
    for caller in callers:
        caller.start()
    start.set()
    try:
        for caller in callers:
            caller.join(timeout=0.5)
        assert all(not caller.is_alive() for caller in callers)
        assert len(entered) == 4
        assert len(errors) == len(engines)
        assert all(isinstance(exc, ForegroundDeadlineExceeded) for exc in errors)
    finally:
        release.set()
        for caller in callers:
            caller.join(timeout=2)
        for engine in engines:
            _wait_for_foreground_projection_worker(engine)
            engine.shutdown()


def test_deadline_none_projection_bypasses_the_process_worker_cap(tmp_path):
    engine = _engine(tmp_path)
    process_slot = engine_module._FOREGROUND_PROJECTION_PROCESS_SLOT
    acquired = 0
    try:
        for _ in range(engine_module._FOREGROUND_PROJECTION_MAX_WORKERS):
            assert process_slot.acquire(blocking=False)
            acquired += 1
        assert (
            engine._call_foreground_projection(
                lambda: "manual-result",
                deadline=None,
                stage="manual projection",
            )
            == "manual-result"
        )
    finally:
        for _ in range(acquired):
            process_slot.release()
        engine.shutdown()


def test_context_replay_stubbing_returns_at_wall_without_mutating_raw_or_dag(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        large_output_active_replay_stubbing_enabled=True,
        large_output_active_replay_stub_threshold_tokens=1,
    )
    entered, release, observed = _blocking_payload_write(monkeypatch)
    messages = [{"role": "tool", "tool_call_id": "call-1", "content": "tool payload"}]
    raw_before = engine._store.get_session_messages("s")
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._assemble_context(
                None,
                messages,
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert engine._store.get_session_messages("s") == raw_before
        assert engine._dag.get_session_node_count("s") == 0
        assert observed["cancel_event"].is_set()
    finally:
        release.set()
        engine.shutdown()


def test_cancelled_context_replay_stubbing_removes_late_created_payload(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        large_output_externalization_threshold_chars=1_000_000,
        large_output_active_replay_stubbing_enabled=True,
        large_output_active_replay_stub_threshold_tokens=1,
    )
    entered, release, observed = _block_after_real_externalizer_returns(
        monkeypatch,
        engine_module,
        "maybe_externalize_tool_output",
    )
    messages = [
        {"role": "tool", "tool_call_id": "call-replay", "content": "replay payload"}
    ]
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._assemble_context(
                None,
                messages,
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert observed.get("result") is not None
        _wait_for_foreground_projection_worker(engine)
        assert list((tmp_path / "externalized").glob("*.json")) == []
    finally:
        release.set()
        engine.shutdown()


def test_durable_replay_externalized_read_returns_at_the_callers_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        ignore_message_patterns=["NEEDLE"],
    )
    externalized = externalize_module.maybe_externalize_payload(
        "externalized NEEDLE",
        kind="raw_payload",
        session_id="s",
        role="user",
        config=engine._config,
        hermes_home=engine._hermes_home,
        force=True,
    )
    assert externalized is not None
    engine._store.append(
        "s",
        {"role": "user", "content": externalized["placeholder"]},
    )
    raw_before = engine._store.get_session_messages("s")
    entered = threading.Event()
    release = threading.Event()
    real_load = summary_serialization_module.load_externalized_payload

    def blocked_load(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(
        summary_serialization_module,
        "load_externalized_payload",
        blocked_load,
    )
    monkeypatch.setattr(
        summary_serialization_module,
        "compile_message_patterns",
        lambda _patterns: [object()],
    )
    monkeypatch.setattr(
        summary_serialization_module,
        "matches_message_pattern",
        lambda text, _patterns: "NEEDLE" in text,
    )
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._project_durable_active_tail(
                0,
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert engine._store.get_session_messages("s") == raw_before
        assert engine._dag.get_session_node_count("s") == 0
    finally:
        release.set()
        engine.shutdown()


def test_context_objective_externalization_returns_at_the_callers_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    entered, release, observed = _blocking_payload_write(monkeypatch)
    inline_payload = "data:image/png;base64," + ("A" * 4096)
    engine._pending_context_anchor_messages = [
        {"role": "user", "content": f"Preserve this objective: {inline_payload}"}
    ]
    raw_before = engine._store.get_session_messages("s")
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._assemble_context(
                None,
                [{"role": "assistant", "content": "working"}],
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert engine._store.get_session_messages("s") == raw_before
        assert engine._dag.get_session_node_count("s") == 0
        assert observed["cancel_event"].is_set()
    finally:
        release.set()
        engine.shutdown()


def test_cancelled_context_objective_projection_removes_late_created_payload(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path)
    entered, release, observed = _block_after_real_externalizer_returns(
        monkeypatch,
        ingest_protection_module,
        "externalize_ingest_payload",
    )
    inline_payload = "data:image/png;base64," + ("A" * 4096)
    engine._pending_context_anchor_messages = [
        {"role": "user", "content": f"Preserve this objective: {inline_payload}"}
    ]
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._assemble_context(
                None,
                [{"role": "assistant", "content": "working"}],
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert isinstance(result.get("error"), ForegroundDeadlineExceeded)
        assert observed.get("result") is not None
        _wait_for_foreground_projection_worker(engine)
        assert list((tmp_path / "externalized").glob("*.json")) == []
    finally:
        release.set()
        engine.shutdown()


def test_context_recall_receives_the_callers_absolute_deadline(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        proactive_recall_enabled=True,
        embeddings_enabled=True,
    )
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    observed: dict[str, float] = {}

    def recall(_args, **kwargs):
        observed["deadline"] = float(kwargs["deadline"])
        clock.value = observed["deadline"]
        return json.dumps(
            {
                "hits": [
                    {
                        "score": 1.0,
                        "snippet": "must not be injected late",
                        "session_id": "other",
                    }
                ]
            }
        )

    monkeypatch.setattr(lcm_tools, "lcm_recall", recall)
    try:
        assert engine._build_proactive_recall_message(
            [{"role": "user", "content": "query"}],
            "user",
            set(),
            deadline=100.25,
        ) is None
        assert observed["deadline"] == pytest.approx(100.25)
        assert engine._proactive_recall_timeout_count == 1
    finally:
        engine.shutdown()


def test_context_recall_returns_at_wall_when_recall_worker_is_blocked(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        proactive_recall_enabled=True,
        embeddings_enabled=True,
    )
    entered = threading.Event()
    release = threading.Event()
    observed: dict[str, float] = {}

    def blocked_recall(_args, **kwargs):
        observed["deadline"] = float(kwargs["deadline"])
        entered.set()
        assert release.wait(timeout=2)
        return json.dumps({"hits": []})

    monkeypatch.setattr(lcm_tools, "lcm_recall", blocked_recall)
    deadline = time.monotonic() + 0.05
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: engine._build_proactive_recall_message(
                [{"role": "user", "content": "query"}],
                "user",
                set(),
                deadline=deadline,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert result.get("error") is None
        assert result.get("value") is None
        assert observed["deadline"] == pytest.approx(deadline)
        assert engine._proactive_recall_timeout_count == 1
    finally:
        release.set()
        engine.shutdown()


def test_recall_tool_caps_its_internal_budget_to_an_outer_deadline(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, embeddings_enabled=False, recall_query_timeout_s=8.0)
    observed: dict[str, float] = {}
    monkeypatch.setattr(lcm_tools.time, "monotonic", lambda: 100.0)

    def fts_arm(_engine, _query, *, candidate_limit, deadline):
        assert candidate_limit > 0
        observed["deadline"] = float(deadline)
        return [], None

    monkeypatch.setattr(lcm_tools, "_lcm_recall_fts_arm", fts_arm)
    try:
        lcm_tools.lcm_recall(
            {"query": "bounded query"},
            engine=engine,
            deadline=100.25,
        )
        assert observed["deadline"] == pytest.approx(100.25)
    finally:
        engine.shutdown()


def test_recall_post_arm_scope_read_returns_a_timeout_at_the_callers_wall(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path, embeddings_enabled=False)
    entered = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_fts_arm",
        lambda *_args, **_kwargs: (
            [
                {
                    "kind": "message_excerpt",
                    "store_id": 1,
                    "session_id": "s",
                    "source": "",
                    "role": "user",
                    "timestamp": 0,
                    "content_offset": 0,
                    "snippet": "post arm hit",
                    "expand_hint": "lcm_expand lcm:1",
                }
            ],
            None,
        ),
    )

    def blocked_scope(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return ["s"]

    monkeypatch.setattr(
        lcm_tools,
        "_recent_conversation_scope_session_ids",
        blocked_scope,
    )
    try:
        completed, result = _call_while_operation_is_blocked(
            lambda: lcm_tools.lcm_recall(
                {"query": "post arm"},
                engine=engine,
                deadline=time.monotonic() + 0.05,
            ),
            entered=entered,
            release=release,
        )

        assert completed
        assert result.get("error") is None
        payload = json.loads(str(result["value"]))
        assert payload["timeout"] is True
        assert payload["degraded"] is True
    finally:
        release.set()
        engine.shutdown()


def test_summary_dag_node_hydration_honors_the_recall_read_deadline(tmp_path):
    engine = _engine(tmp_path)
    lock_held = threading.Event()
    release = threading.Event()

    def hold_dag_lock() -> None:
        with engine._dag._db_lock:
            lock_held.set()
            assert release.wait(timeout=2)

    holder = threading.Thread(target=hold_dag_lock, name="deadline-dag-lock-holder")
    holder.start()
    assert lock_held.wait(timeout=2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            engine._dag.get_node(
                1,
                deadline=started + 0.05,
                deadline_clock=time.monotonic,
                lock_timeout_seconds=0.05,
                busy_timeout_ms=50,
            )
        assert time.monotonic() - started < 0.25
    finally:
        release.set()
        holder.join(timeout=2)
        engine.shutdown()


def test_bounded_store_read_does_not_wait_on_concurrent_legacy_connection_read(
    tmp_path,
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "s",
        {"role": "user", "content": "legacy store read"},
    )
    conn = engine._store.connection
    assert conn is not None
    legacy_entered = threading.Event()
    legacy_release = threading.Event()
    legacy_finished = threading.Event()
    bounded_finished = threading.Event()
    results: dict[str, object] = {}

    def block_legacy_progress() -> int:
        if threading.current_thread().name == "legacy-store-read":
            legacy_entered.set()
            assert legacy_release.wait(timeout=2)
        return 0

    def legacy_read() -> None:
        try:
            results["legacy"] = engine._store.get(store_id)
        finally:
            legacy_finished.set()

    def bounded_read() -> None:
        try:
            results["bounded"] = engine._store.get(
                store_id,
                deadline=time.monotonic() + 0.05,
                deadline_clock=time.monotonic,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            results["bounded_error"] = exc
        finally:
            bounded_finished.set()

    conn.set_progress_handler(block_legacy_progress, 1)
    legacy = threading.Thread(target=legacy_read, name="legacy-store-read")
    bounded = threading.Thread(target=bounded_read, name="bounded-store-read")
    legacy.start()
    assert legacy_entered.wait(timeout=2)
    bounded.start()
    try:
        assert bounded_finished.wait(timeout=0.2)
        assert isinstance(results.get("bounded_error"), TimeoutError)
        assert not legacy_finished.is_set()
    finally:
        legacy_release.set()
        legacy.join(timeout=2)
        bounded.join(timeout=2)
        conn.set_progress_handler(None, 0)
        engine.shutdown()

    assert results["legacy"]["store_id"] == store_id


def test_bounded_dag_read_does_not_wait_on_concurrent_legacy_connection_read(
    tmp_path,
):
    engine = _engine(tmp_path)
    node = engine_module.SummaryNode(
        session_id="s",
        depth=0,
        summary="legacy DAG read",
        token_count=4,
        source_token_count=8,
        source_ids=[],
        source_type="messages",
    )
    node_id = engine._dag.add_node(node)
    conn = engine._dag.connection
    assert conn is not None
    legacy_entered = threading.Event()
    legacy_release = threading.Event()
    legacy_finished = threading.Event()
    bounded_finished = threading.Event()
    results: dict[str, object] = {}

    def block_legacy_progress() -> int:
        if threading.current_thread().name == "legacy-dag-read":
            legacy_entered.set()
            assert legacy_release.wait(timeout=2)
        return 0

    def legacy_read() -> None:
        try:
            results["legacy"] = engine._dag.get_node(node_id)
        finally:
            legacy_finished.set()

    def bounded_read() -> None:
        try:
            results["bounded"] = engine._dag.get_node(
                node_id,
                deadline=time.monotonic() + 0.05,
                deadline_clock=time.monotonic,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            results["bounded_error"] = exc
        finally:
            bounded_finished.set()

    conn.set_progress_handler(block_legacy_progress, 1)
    legacy = threading.Thread(target=legacy_read, name="legacy-dag-read")
    bounded = threading.Thread(target=bounded_read, name="bounded-dag-read")
    legacy.start()
    assert legacy_entered.wait(timeout=2)
    bounded.start()
    try:
        assert bounded_finished.wait(timeout=0.2)
        assert isinstance(results.get("bounded_error"), TimeoutError)
        assert not legacy_finished.is_set()
    finally:
        legacy_release.set()
        legacy.join(timeout=2)
        bounded.join(timeout=2)
        conn.set_progress_handler(None, 0)
        engine.shutdown()

    assert results["legacy"].node_id == node_id


def test_bounded_store_read_uses_connection_isolated_from_retrieval_helper(
    tmp_path,
):
    engine = _engine(tmp_path)
    store_id = engine._store.append(
        "s",
        {"role": "user", "content": "operator helper store read"},
        conversation_id="c",
    )
    conn = engine._store.connection
    assert conn is not None
    helper_entered = threading.Event()
    helper_release = threading.Event()
    helper_finished = threading.Event()
    bounded_finished = threading.Event()
    results: dict[str, object] = {}

    def block_helper_progress() -> int:
        if threading.current_thread().name == "retrieval-helper-read":
            helper_entered.set()
            assert helper_release.wait(timeout=2)
        return 0

    def helper_read() -> None:
        try:
            results["helper"] = (
                retrieval_core_module._resolve_semantic_conversation_scope(
                    engine,
                    search_session_id=None,
                    conversation_id="c",
                )
            )
        finally:
            helper_finished.set()

    def bounded_read() -> None:
        try:
            results["bounded"] = engine._store.get(
                store_id,
                deadline=time.monotonic() + 0.05,
                deadline_clock=time.monotonic,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            results["bounded_error"] = exc
        finally:
            bounded_finished.set()

    conn.set_progress_handler(block_helper_progress, 1)
    helper = threading.Thread(target=helper_read, name="retrieval-helper-read")
    bounded = threading.Thread(target=bounded_read, name="bounded-helper-store-read")
    helper.start()
    assert helper_entered.wait(timeout=2)
    bounded.start()
    try:
        assert bounded_finished.wait(timeout=0.2)
        assert isinstance(results.get("bounded_error"), TimeoutError)
        assert not helper_finished.is_set()
    finally:
        helper_release.set()
        helper.join(timeout=2)
        bounded.join(timeout=2)
        conn.set_progress_handler(None, 0)
        engine.shutdown()

    assert results["helper"] == ["s"]


def test_bounded_dag_read_uses_connection_isolated_from_inspect_helper(tmp_path):
    engine = _engine(tmp_path)
    node = engine_module.SummaryNode(
        session_id="s",
        depth=0,
        summary="operator helper DAG read",
        token_count=4,
        source_token_count=8,
        source_ids=[7],
        source_type="messages",
    )
    node_id = engine._dag.add_node(node)
    conn = engine._dag.connection
    assert conn is not None
    helper_entered = threading.Event()
    helper_release = threading.Event()
    helper_finished = threading.Event()
    bounded_finished = threading.Event()
    results: dict[str, object] = {}

    def block_helper_progress() -> int:
        if threading.current_thread().name == "inspect-helper-read":
            helper_entered.set()
            assert helper_release.wait(timeout=2)
        return 0

    def helper_read() -> None:
        try:
            results["helper"] = lcm_tools._inspect_highest_compacted_source_store_id(
                engine,
                "s",
            )
        finally:
            helper_finished.set()

    def bounded_read() -> None:
        try:
            results["bounded"] = engine._dag.get_node(
                node_id,
                deadline=time.monotonic() + 0.05,
                deadline_clock=time.monotonic,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            results["bounded_error"] = exc
        finally:
            bounded_finished.set()

    conn.set_progress_handler(block_helper_progress, 1)
    helper = threading.Thread(target=helper_read, name="inspect-helper-read")
    bounded = threading.Thread(target=bounded_read, name="bounded-helper-dag-read")
    helper.start()
    assert helper_entered.wait(timeout=2)
    bounded.start()
    try:
        assert bounded_finished.wait(timeout=0.2)
        assert isinstance(results.get("bounded_error"), TimeoutError)
        assert not helper_finished.is_set()
    finally:
        helper_release.set()
        helper.join(timeout=2)
        bounded.join(timeout=2)
        conn.set_progress_handler(None, 0)
        engine.shutdown()

    assert results["helper"] == 7


@pytest.mark.parametrize("backend", ["store", "dag"])
def test_busy_timeout_restore_failure_does_not_mask_primary_deadline_error(
    tmp_path,
    backend,
):
    engine = _engine(tmp_path)
    target = engine._store if backend == "store" else engine._dag
    conn = target.connection
    assert conn is not None
    pragma_calls = 0

    def deny_restore(
        action_code: int,
        first_arg: str | None,
        _second_arg: str | None,
        _database_name: str | None,
        _trigger_name: str | None,
    ) -> int:
        nonlocal pragma_calls
        if (
            action_code == sqlite3.SQLITE_PRAGMA
            and str(first_arg or "").lower() == "busy_timeout"
        ):
            pragma_calls += 1
            if pragma_calls == 3:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(deny_restore)
    access = (
        target._bounded_connection_access
        if backend == "store"
        else target._bounded_read_connection
    )
    try:
        with pytest.raises(TimeoutError, match="primary deadline"):
            with access(
                deadline=time.monotonic() + 1.0,
                deadline_clock=time.monotonic,
                busy_timeout_ms=50,
            ):
                raise TimeoutError("primary deadline")
        assert pragma_calls == 3
    finally:
        conn.set_authorizer(None)
        engine.shutdown()


def test_deadline_bound_transcript_gc_returns_at_wall_while_optional_read_blocks(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        large_output_transcript_gc_enabled=True,
    )
    message = {"role": "tool", "tool_call_id": "call-gc", "content": "raw tool"}
    store_id = engine._store.append("s", message)
    raw_before = engine._store.get(store_id)
    entered = threading.Event()
    release = threading.Event()
    caller_finished = threading.Event()
    result: dict[str, object] = {}

    def blocked_get_batch(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return {store_id: raw_before}

    monkeypatch.setattr(engine._store, "get_batch", blocked_get_batch)

    def invoke() -> None:
        try:
            result["value"] = engine._maybe_gc_compacted_tool_results(
                [message],
                [store_id],
                deadline=time.monotonic() + 0.15,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            result["error"] = exc
        finally:
            caller_finished.set()

    caller = threading.Thread(target=invoke, name="deadline-transcript-gc")
    caller.start()
    try:
        assert entered.wait(timeout=2)
        completed = caller_finished.wait(timeout=0.35)
        assert completed
        assert result.get("error") is None
        assert engine._store.get(store_id) == raw_before
    finally:
        release.set()
        caller.join(timeout=2)
        _wait_for_foreground_projection_worker(engine)
        assert engine._store.get(store_id) == raw_before
        engine.shutdown()


def test_transcript_gc_rolls_back_raw_rewrite_when_chunk_archive_fails(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        embeddings_enabled=True,
        large_output_transcript_gc_enabled=True,
    )
    message = {"role": "tool", "tool_call_id": "call-gc", "content": "raw tool"}
    store_id = engine._store.append("s", message)
    raw_before = engine._store.get(store_id)
    archive_entered = threading.Event()

    monkeypatch.setattr(
        engine_module,
        "find_externalized_payload_for_message",
        lambda *_args, **_kwargs: {
            "kind": "tool_result",
            "tool_call_id": "call-gc",
            "content_chars": len(message["content"]),
            "content_bytes": len(message["content"]),
            "ref": "payload.json",
        },
    )

    def fail_archive(*_args, **_kwargs):
        archive_entered.set()
        raise sqlite3.OperationalError("synthetic chunk archive failure")

    monkeypatch.setattr(
        "hermes_lcm.vector_store.VectorStore.archive_chunks_for_messages_on_connection",
        fail_archive,
    )

    try:
        with pytest.raises(sqlite3.OperationalError, match="chunk archive failure"):
            engine._maybe_gc_compacted_tool_results(
                [message],
                [store_id],
                deadline=time.monotonic() + 1.0,
            )
        assert archive_entered.is_set()
        assert engine._store.get(store_id) == raw_before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("locked_phase", ["plan", "write"])
def test_deadline_bound_transcript_gc_treats_lock_contention_as_optional(
    tmp_path,
    monkeypatch,
    locked_phase,
):
    engine = _engine(
        tmp_path,
        large_output_transcript_gc_enabled=True,
    )
    message = {"role": "tool", "tool_call_id": "call-gc", "content": "raw tool"}
    store_id = engine._store.append("s", message)
    raw_before = engine._store.get(store_id)

    if locked_phase == "plan":
        monkeypatch.setattr(
            engine._store,
            "get_batch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                sqlite3.OperationalError("database is locked")
            ),
        )
    else:
        monkeypatch.setattr(
            engine_module,
            "find_externalized_payload_for_message",
            lambda *_args, **_kwargs: {
                "kind": "tool_result",
                "tool_call_id": "call-gc",
                "content_chars": len(message["content"]),
                "content_bytes": len(message["content"]),
                "ref": "payload.json",
            },
        )
        monkeypatch.setattr(
            engine._store,
            "gc_externalized_tool_result",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                sqlite3.OperationalError("database is locked")
            ),
        )

    try:
        engine._maybe_gc_compacted_tool_results(
            [message],
            [store_id],
            deadline=time.monotonic() + 1.0,
        )
        assert engine._store.get(store_id) == raw_before
    finally:
        engine.shutdown()


def test_assertion_provider_result_finishing_after_origin_deadline_is_not_published(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        assertions_enabled=True,
        assertion_extraction_enabled=True,
        assertion_extraction_max_sources_per_pass=1,
    )
    assert engine._assertions is not None
    message = {"role": "user", "content": "I prefer tea."}
    store_id = engine._store.append("s", message)
    source_snapshot = engine._assertions.snapshot_source(store_id)
    clock = _FakeClock()
    engine._foreground_monotonic = clock
    provider_entered = threading.Event()
    release_provider = threading.Event()
    observed_timeout: list[float] = []

    def provider_call(prompt: str, _model: str, timeout_seconds: float):
        observed_timeout.append(float(timeout_seconds))
        provider_entered.set()
        assert release_provider.wait(timeout=2)
        payload = {
            "schema_version": 1,
            "source_store_id": source_snapshot.store_id,
            "source_content_sha256": source_snapshot.content_sha256,
            "assertions": [],
            "relations": [],
        }
        return json.dumps(payload), 1, 1

    monkeypatch.setattr(
        assertion_extraction,
        "_call_structured_assertion_llm",
        provider_call,
    )
    try:
        assert engine._schedule_pre_compaction_assertions(
            [message],
            deadline=101.0,
        ) is True
        assert provider_entered.wait(timeout=2)
        clock.value = 101.0
        release_provider.set()
        assert engine._assertion_extraction_idle.wait(timeout=2)

        snapshot = engine._assertions.snapshot_source(store_id)
        assert not engine._assertions.has_current_receipt(snapshot)
        assert observed_timeout and 0 < observed_timeout[0] <= 1.0
    finally:
        release_provider.set()
        assert engine._assertion_extraction_idle.wait(timeout=2)
        engine.shutdown()


def test_assertion_publication_sqlite_wait_is_capped_by_origin_deadline(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        tmp_path,
        assertions_enabled=True,
        assertion_extraction_enabled=True,
        assertion_extraction_max_sources_per_pass=1,
    )
    assert engine._assertions is not None
    message = {"role": "user", "content": "I prefer tea."}
    store_id = engine._store.append("s", message)
    source_snapshot = engine._assertions.snapshot_source(store_id)
    provider_called = threading.Event()

    def provider_call(prompt: str, _model: str, _timeout_seconds: float):
        provider_called.set()
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "source_store_id": source_snapshot.store_id,
                    "source_content_sha256": source_snapshot.content_sha256,
                    "assertions": [],
                    "relations": [],
                }
            ),
            1,
            1,
        )

    monkeypatch.setattr(
        assertion_extraction,
        "_call_structured_assertion_llm",
        provider_call,
    )

    class ExistingWriter:
        def __getattr__(self, name):
            return getattr(engine._assertions, name)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        engine_module,
        "AssertionStore",
        lambda *_args, **_kwargs: ExistingWriter(),
    )
    blocker = sqlite3.connect(str(engine._store.db_path), isolation_level=None)
    blocker.execute("PRAGMA busy_timeout=0")
    blocker.execute("BEGIN IMMEDIATE")
    completed_while_locked = False
    try:
        assert engine._schedule_pre_compaction_assertions(
            [message],
            deadline=time.monotonic() + 0.05,
        ) is True
        completed_while_locked = engine._assertion_extraction_idle.wait(timeout=0.25)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
        assert engine._assertion_extraction_idle.wait(timeout=2)

    try:
        assert completed_while_locked
        assert provider_called.is_set()
        snapshot = engine._assertions.snapshot_source(store_id)
        assert not engine._assertions.has_current_receipt(snapshot)
    finally:
        engine.shutdown()
