"""Regression tests for provider-error log redaction.

Provider exception messages can contain authorization headers, request bodies,
or upstream response content.  Operational logs may retain the failure outcome
and exception category, but never the raw exception text.
"""

import logging
import sys
from types import ModuleType

from hermes_lcm import extraction
from hermes_lcm.bypass import BypassMixin


_SECRET_MARKER = "sensitive-provider-marker-must-not-appear"


def _install_failing_auxiliary_client(monkeypatch) -> None:
    agent_module = sys.modules.get("agent") or ModuleType("agent")
    if not hasattr(agent_module, "__path__"):
        agent_module.__path__ = []
    auxiliary_module = ModuleType("agent.auxiliary_client")

    def fail_call(**_kwargs):
        raise RuntimeError(f"provider rejected Authorization: {_SECRET_MARKER}")

    auxiliary_module.call_llm = fail_call
    monkeypatch.setattr(agent_module, "auxiliary_client", auxiliary_module, raising=False)
    monkeypatch.setitem(sys.modules, "agent", agent_module)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary_module)


def test_extraction_provider_failure_log_omits_raw_exception_text(monkeypatch, caplog):
    _install_failing_auxiliary_client(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.extraction"):
        result = extraction._call_extraction_llm("extract this")

    assert result is None
    assert _SECRET_MARKER not in caplog.text
    assert "Authorization" not in caplog.text
    assert "Extraction LLM call failed" in caplog.text
    assert "RuntimeError" in caplog.text


def test_extraction_wrapper_failure_log_omits_raw_exception_text(monkeypatch, caplog, tmp_path):
    def fail_extraction(*_args, **_kwargs):
        raise RuntimeError(f"provider payload contained {_SECRET_MARKER}")

    monkeypatch.setattr(extraction, "_call_extraction_llm", fail_extraction)

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.extraction"):
        result = extraction.extract_before_compaction(
            "serialized messages",
            str(tmp_path / "extractions"),
        )

    assert result is False
    assert _SECRET_MARKER not in caplog.text
    assert "provider payload" not in caplog.text
    assert "Pre-compaction extraction failed (non-blocking)" in caplog.text
    assert "RuntimeError" in caplog.text


class _FailingHostCompressor:
    compression_count = 0

    def compress(self, *_args, **_kwargs):
        raise RuntimeError(f"upstream response exposed {_SECRET_MARKER}")


class _BypassHarness(BypassMixin):
    _session_ignored = True
    _session_stateless = False
    _session_id = "ignored:test"
    threshold_tokens = 1
    context_length = 100
    protect_first_n = 1
    protect_last_n = 1
    compression_count = 0

    def __init__(self):
        self._compressor = _FailingHostCompressor()
        self._last_compression_status = ""
        self._last_compression_noop_reason = ""
        self._host_fallback_compressor = self._compressor
        self._host_fallback_session_id = self._session_id

    @staticmethod
    def _thread_context_stateless():
        return False

    @staticmethod
    def _thread_context_session_id():
        return ""

    @staticmethod
    def _remember_lcm_bypass_message_prefix(*_args, **_kwargs):
        return None

    @staticmethod
    def _should_force_overflow_recovery(**_kwargs):
        return False

    @staticmethod
    def _redact_active_replay_messages(messages):
        return list(messages)

    @staticmethod
    def _overflow_recovery_assembly_cap(**_kwargs):
        return None

    def _get_host_fallback_compressor(self):
        return self._compressor

    @staticmethod
    def _sync_host_fallback_compressor(_compressor):
        return None

    @staticmethod
    def _call_with_hard_foreground_deadline(operation, **_kwargs):
        return operation()

    @staticmethod
    def _fallback_tail_compaction(_messages, *, target_tokens=None):
        del target_tokens
        return [{"role": "system", "content": "deterministic fallback"}]


def test_bypass_provider_failure_log_omits_raw_exception_text(caplog):
    harness = _BypassHarness()
    messages = [{"role": "user", "content": "oversized message"}]

    with caplog.at_level(logging.WARNING, logger="hermes_lcm.bypass"):
        result = harness._compress_lcm_bypassed_session(messages, current_tokens=10)

    assert result == [{"role": "system", "content": "deterministic fallback"}]
    assert _SECRET_MARKER not in caplog.text
    assert "upstream response" not in caplog.text
    assert "using deterministic trim" in caplog.text
    assert "RuntimeError" in caplog.text
