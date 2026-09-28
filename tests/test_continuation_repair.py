"""Regression gates use frozen failures, never synthetic provider receipts."""
import json
from pathlib import Path

from hermes_lcm.continuation import safe_continuation_contract
from hermes_lcm import escalation

FIXTURES = Path(__file__).resolve().parents[1] / 'benchmarks/fixtures'


def test_unsafe_contract_falls_back_for_every_frozen_checkpoint():
    checkpoints = json.loads((FIXTURES / 'continuation_v1_checkpoints.json').read_text())
    for row in checkpoints:
        assert safe_continuation_contract('continuation-v1', '\n'.join(row['records'])) == ''


def test_critical_failures_cannot_enter_versioned_prompt(monkeypatch):
    prompts = []
    monkeypatch.setattr(escalation, '_invoke_summary_llm_chain',
                        lambda prompt, *a, **kw: prompts.append(prompt) or 'baseline summary')
    rows = json.loads((FIXTURES / 'continuation_v1_checkpoints.json').read_text())
    for row in rows[:2]:
        source = '\n'.join(row['records'])
        escalation.summarize_with_escalation(source, 2000, row['budget'],
                                             continuation_contract='continuation-v1')
        assert prompts[-1] == escalation._build_l1_prompt(source, row['budget'], row['depth'])


def test_all_frozen_checkpoints_use_identical_baseline_attempts(monkeypatch):
    import subprocess
    import types
    baseline_source = subprocess.check_output(
        ['git', 'show', '29fbffb31122dbdcac7d32dd0d87fc8bb5f5dc6e:escalation.py'],
        cwd=FIXTURES.parents[1], text=True)
    baseline_module = types.ModuleType('hermes_lcm._frozen_baseline_escalation')
    baseline_module.__package__ = 'hermes_lcm'
    import sys
    monkeypatch.setitem(sys.modules, baseline_module.__name__, baseline_module)
    exec(compile(baseline_source, '<frozen-baseline>', 'exec'), baseline_module.__dict__)
    rows = json.loads((FIXTURES / 'continuation_v1_checkpoints.json').read_text())
    for row in rows:
        captured = []
        capture = lambda prompt, *a, **kw: captured.append(prompt) or None
        monkeypatch.setattr(escalation, '_invoke_summary_llm_chain', capture)
        monkeypatch.setattr(baseline_module, '_invoke_summary_llm_chain', capture)
        source = '\n'.join(row['records'])
        baseline = baseline_module.summarize_with_escalation(source, 2000, row['budget'], depth=row['depth'])
        before = list(captured)
        captured.clear()
        candidate = escalation.summarize_with_escalation(source, 2000, row['budget'], depth=row['depth'],
                                                         continuation_contract='continuation-v1')
        assert captured == before
        assert candidate == baseline


def test_unknown_contract_is_not_silently_accepted():
    import pytest
    with pytest.raises(ValueError):
        safe_continuation_contract('unknown', 'hello')
