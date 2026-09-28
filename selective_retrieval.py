"""Narrow session pilot for the existing retrieval controller, without stores."""
from __future__ import annotations

from threading import Lock
from weakref import WeakKeyDictionary

from .adaptive_retrieval import AdaptiveRetrievalRegistry

def session_allowed(csv: str, session_id: str) -> bool:
    """Reject wildcards and substrings; require an exact nonempty session ID."""
    return bool(session_id) and session_id in {
        value.strip() for value in str(csv or "").split(",") if value.strip()
    }

_lock = Lock()
_controllers = WeakKeyDictionary()


def controller_for(engine):
    """Keep the global legacy path unchanged; pilots never bind query views.

    Engine identity isolates profiles. The existing controller also binds each
    episode to its originating session. Weak keys avoid retaining shut-down
    engines. Recheck admission on every call, including finish/status.
    """
    session_id = str(getattr(engine, "current_session_id", "") or "")
    allowed = session_allowed(
        getattr(engine._config, "selective_retrieval_session_ids", ""), session_id
    )
    if not allowed:
        return getattr(engine, "_adaptive_retrieval", None)
    with _lock:
        controller = _controllers.get(engine)
        if controller is None:
            controller = AdaptiveRetrievalRegistry(query_views=None)
            _controllers[engine] = controller
        return controller
