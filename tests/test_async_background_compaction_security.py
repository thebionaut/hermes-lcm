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


def test_local_type_error_persists_bounded_message_excerpt():
    persisted = _sanitize_error(
        TypeError("mapping object is not subscriptable " + "x " * 240)
    )

    prefix, excerpt = persisted.split(":", 2)[:2], persisted.split(":", 2)[2]
    assert prefix == ["TypeError", "provider_error"]
    assert excerpt.startswith("mapping object is not subscriptable")
    assert len(excerpt) == 200


def test_local_error_excerpt_redacts_credential_shapes():
    api_key = "sk-secret-value-that-must-not-survive"
    opaque = "A" * 24

    persisted = _sanitize_error(
        TypeError(f"request api_key={api_key} authorization bearer {opaque} failed")
    )

    assert persisted.startswith("TypeError:provider_error:")
    assert persisted.count("<redacted>") >= 2
    assert api_key not in persisted
    assert opaque not in persisted


def test_genuine_provider_exception_retains_short_form_byte_for_byte():
    class ProviderAPIError(RuntimeError):
        status_code = 429

    assert (
        _sanitize_error(ProviderAPIError("provider response body must not persist"))
        == "ProviderAPIError:provider_error:http_429"
    )
