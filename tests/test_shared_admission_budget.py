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
        engine.bind_context_budget(SimpleNamespace(safe_input_tokens=800000))
        assert engine._effective_assembly_token_cap() == 150000
        engine.update_model("small", 200000)
        engine.bind_context_budget(SimpleNamespace(safe_input_tokens=100000))
        assert engine._effective_assembly_token_cap() < 100000
        assert engine._background_publication_trigger_tokens() <= engine._background_emergency_trigger_tokens() < 100000
    finally:
        engine.shutdown()
