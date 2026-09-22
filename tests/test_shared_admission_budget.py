from types import SimpleNamespace
import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from agent.context_admission import resolve_context_budget

@pytest.mark.parametrize("window", [128000,200000,272000,500000,900000,1000000,1048576])
def test_every_compression_boundary_precedes_admission(tmp_path, window):
    engine = LCMEngine(LCMConfig(database_path=str(tmp_path / "lcm.db"), context_reserve_tokens=100000))
    try:
        engine.update_model("test-model", window, provider="test-provider")
        budget = resolve_context_budget(provider="test-provider", model="test-model", config_context_length=window)
        engine.bind_context_budget(budget)
        pub = engine._background_publication_trigger_tokens()
        emergency = engine._background_emergency_trigger_tokens()
        cap = engine._effective_assembly_token_cap()
        assert 0 < pub <= emergency <= cap < budget.safe_input_tokens
        assert engine.threshold_tokens <= pub
        assert engine._overflow_recovery_assembly_cap(observed_tokens=50000, messages=[]) == max(1, cap - 50000)
    finally:
        engine.shutdown()

def test_switch_replaces_prior_budget_and_respects_lower_explicit_cap(tmp_path):
    engine = LCMEngine(LCMConfig(database_path=str(tmp_path / "lcm.db"), max_assembly_tokens=150000))
    try:
        engine.update_model("large", 1000000)
        engine.bind_context_budget(SimpleNamespace(context_window=1000000, safe_input_tokens=800000))
        assert engine._effective_assembly_token_cap() == 150000
        engine.update_model("small", 200000)
        engine.bind_context_budget(SimpleNamespace(context_window=200000, safe_input_tokens=100000))
        assert engine._effective_assembly_token_cap() < 100000
        assert engine._background_publication_trigger_tokens() <= engine._background_emergency_trigger_tokens() < 100000
    finally:
        engine.shutdown()


def test_registry_budget_rebinds_unknown_window_up_down_and_up_again(tmp_path):
    engine = LCMEngine(LCMConfig(database_path=str(tmp_path / "lcm.db")))
    try:
        engine.update_model("unknown-model", 256000, provider="test-provider")
        assert engine.raw_context_length == engine.context_length == 256000

        for window, safe in ((1000000, 800000), (180000, 144000), (640000, 512000)):
            engine.bind_context_budget(
                SimpleNamespace(context_window=window, safe_input_tokens=safe)
            )
            assert engine.raw_context_length == window
            assert engine.context_length == window
            assert engine._context_length_source == "host_context_budget"
            assert (
                engine._background_publication_trigger_tokens()
                <= engine._background_emergency_trigger_tokens()
                <= engine._effective_assembly_token_cap()
                < safe
            )
            assert engine.threshold_tokens <= engine._background_publication_trigger_tokens()
    finally:
        engine.shutdown()


def test_registry_budget_preserves_provider_cap_and_raw_window(tmp_path):
    engine = LCMEngine(LCMConfig(database_path=str(tmp_path / "lcm.db")))
    try:
        engine.update_model("gpt-5.6-sol", 256000, provider="openai-codex")
        engine.bind_context_budget(
            SimpleNamespace(context_window=1000000, safe_input_tokens=800000)
        )
        assert engine.raw_context_length == 1000000
        assert engine.context_length == 900000
        assert engine.effective_context_length_cap == 900000
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"
    finally:
        engine.shutdown()
