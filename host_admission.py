"""Reserve existing Hermes process admission for plugin background work.

Standalone LCM and older hosts without the retirement module remain supported.
Other import/runtime failures must propagate; they cannot silently admit work.
"""
from __future__ import annotations

import threading
from typing import Any


class BackgroundAdmission:
    """An exactly-once release that may transfer to a daemon thread."""

    def __init__(self, fence: Any = None) -> None:
        self._fence = fence
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            fence, self._fence = self._fence, None
        if fence is not None:
            fence.release()


def acquire_background_admission() -> BackgroundAdmission | None:
    try:
        from hermes_cli.backend_retirement import retirement
    except ModuleNotFoundError as exc:
        if exc.name not in {"hermes_cli", "hermes_cli.backend_retirement"}:
            raise
        return BackgroundAdmission()
    if not retirement.acquire():
        return None
    return BackgroundAdmission(retirement)
