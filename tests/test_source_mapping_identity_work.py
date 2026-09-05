from collections import Counter

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_active_identity_is_built_once_per_mapping_and_not_cached_across_calls(tmp_path, monkeypatch):
    engine = LCMEngine(LCMConfig(database_path=str(tmp_path / 'mapping.db'),
                                 background_compaction_enabled=False))
    try:
        engine.on_session_start('mapping', conversation_id='mapping', context_length=900000)
        messages = [{'role': 'user', 'content': 'unique %s %s' % (i, 'payload ' * 100)}
                    for i in range(40)]
        expected = {id(m): engine._store.append('mapping', m) for m in messages}
        original = engine._message_replay_identity
        calls = Counter()

        def count(message, *args, **kwargs):
            if not kwargs.get('stored_row'):
                calls[id(message)] += 1
            return original(message, *args, **kwargs)

        monkeypatch.setattr(engine, '_message_replay_identity', count)
        assert engine._get_store_id_map_for_messages(messages) == expected
        assert max(calls.values()) == 1
        calls.clear()
        messages[0]['content'] = 'changed since previous invocation'
        changed = engine._get_store_id_map_for_messages(messages)
        assert id(messages[0]) not in changed
        assert changed == {key: value for key, value in expected.items() if key != id(messages[0])}
        assert max(calls.values()) == 1
    finally:
        engine.shutdown()
