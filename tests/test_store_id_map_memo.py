"""Per-compress memo of the durable store-id map."""

from __future__ import annotations

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import ReconcileMixin


def _engine(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            sensitive_patterns_enabled=False,
            proactive_recall_enabled=False,
        ),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine._session_id = "s"
    engine._conversation_id = "s"
    return engine


def test_memo_reuses_identical_lists_and_invalidates_on_new_rows(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    messages = [
        {"role": "user", "content": f"m{index}", "timestamp": 1.7e9 + index}
        for index in range(5)
    ]
    computed: list[int] = []
    original = ReconcileMixin._compute_store_id_map_for_messages

    def counting(self, msgs, **kwargs):
        computed.append(len(msgs))
        return original(self, msgs, **kwargs)

    monkeypatch.setattr(ReconcileMixin, "_compute_store_id_map_for_messages", counting)
    try:
        engine._store.append_batch("s", messages, conversation_id="s")
        uncached = ReconcileMixin._get_store_id_map_for_messages(engine, messages)
        engine._store_id_map_memo = {}
        first = ReconcileMixin._get_store_id_map_for_messages(engine, messages)
        first[id(messages[0])] = -1  # callers may mutate their copy
        second = ReconcileMixin._get_store_id_map_for_messages(engine, messages)
        assert second == uncached
        assert computed == [5, 5]
        # A different list object with equal content is a miss.
        ReconcileMixin._get_store_id_map_for_messages(engine, [dict(m) for m in messages])
        assert computed == [5, 5, 5]
        # New durable rows change the session version: recompute.
        engine._store.append_batch(
            "s", [{"role": "user", "content": "new", "timestamp": 1.8e9}], conversation_id="s"
        )
        ReconcileMixin._get_store_id_map_for_messages(engine, messages)
        assert computed == [5, 5, 5, 5]
    finally:
        engine._store_id_map_memo = None
        engine.shutdown()


def test_memo_is_scoped_to_one_compress_call(tmp_path):
    engine = _engine(tmp_path)
    try:
        assert getattr(engine, "_store_id_map_memo", None) is None
        engine.compress([{"role": "user", "content": "hi", "timestamp": 1.7e9}])
        assert getattr(engine, "_store_id_map_memo", None) is None
    finally:
        engine.shutdown()
