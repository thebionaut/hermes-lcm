"""Security contracts for durable and operator-facing diagnostics."""

from hermes_lcm.background_compaction import _sanitize_error
from hermes_lcm.tools import (
    _sanitize_background_error,
    _sanitize_persisted_background_error,
)


def test_provider_exception_body_is_never_persisted_or_displayed():
    credential = "sk" + "-ant-" + ("x" * 40)
    prompt_fragment = "private customer prompt fragment"
    persisted = _sanitize_error(
        RuntimeError(f"{prompt_fragment} credential={credential}")
    )
    displayed, _truncated = _sanitize_background_error(
        f"provider_error {prompt_fragment} {credential}"
    )
    legacy_displayed, legacy_truncated = _sanitize_persisted_background_error(
        f"RuntimeError: {prompt_fragment} credential={credential}"
    )

    assert persisted == "RuntimeError:provider_error"
    assert prompt_fragment not in persisted
    assert credential not in persisted
    assert credential not in displayed
    assert legacy_displayed == "background_error:redacted"
    assert legacy_truncated is True
    assert prompt_fragment not in legacy_displayed
