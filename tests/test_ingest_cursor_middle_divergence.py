"""Same-session middle rewrites reuse the unchanged durable suffix."""

from __future__ import annotations

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _config(db_path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db_path),
        background_compaction_enabled=True,
        background_compaction_workers=0,
        background_compaction_backlog_tokens=1_000_000,
        context_reserve_tokens=100_000,
    )


_ORIGINAL = [
    {"role": "user", "content": "A"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content": "C"},
    {"role": "assistant", "content": "D"},
]


@pytest.mark.parametrize(
    ("rebound", "new_durable_contents"),
    [
        pytest.param(
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "X"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
                {"role": "user", "content": "E"},
            ],
            ["X", "E"],
            id="replacement",
        ),
        pytest.param(
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "assistant", "content": "X"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
                {"role": "user", "content": "E"},
            ],
            ["X", "E"],
            id="insertion",
        ),
        pytest.param(
            [
                {"role": "user", "content": "A"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
                {"role": "user", "content": "E"},
            ],
            ["E"],
            id="deletion",
        ),
    ],
)
def test_middle_divergence_reuses_unchanged_suffix_rows(
    tmp_path,
    rebound,
    new_durable_contents,
):
    engine = LCMEngine(_config(tmp_path / "middle-divergence.db"))
    try:
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        engine.ingest(list(_ORIGINAL))
        original_rows = engine._store.get_session_messages("session-a")
        original_suffix_ids = {
            row["content"]: int(row["store_id"])
            for row in original_rows
            if row["content"] in {"C", "D"}
        }

        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        engine.ingest(rebound)

        durable_rows = engine._store.get_session_messages("session-a")
        assert [row["content"] for row in durable_rows] == [
            "A",
            "B",
            "C",
            "D",
            *new_durable_contents,
        ]
        for content, store_id in original_suffix_ids.items():
            matching_rows = [
                row for row in durable_rows if row["content"] == content
            ]
            assert [int(row["store_id"]) for row in matching_rows] == [store_id]

        active_store_ids = engine._get_store_id_map_for_messages(rebound)
        assert {
            message["content"]: active_store_ids.get(id(message))
            for message in rebound
        } == {
            row["content"]: int(row["store_id"])
            for row in durable_rows
            if row["content"] in {message["content"] for message in rebound}
        }

        # Once aligned to the rebound view, an ordinary repeated host bind is
        # still a no-op and cannot re-append either changed or reused rows.
        engine.on_session_start(
            "session-a",
            conversation_id="conversation-a",
            context_length=900_000,
        )
        engine.ingest(rebound)
        assert engine._store.get_session_count("session-a") == len(durable_rows)
        assert engine._ingest_cursor == len(rebound)
    finally:
        engine.shutdown()
