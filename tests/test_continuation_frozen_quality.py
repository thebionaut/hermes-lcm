"""Frozen answer-quality observations, not a live-model rollout qualification."""
import ast
import json
import subprocess
from pathlib import Path

import pytest

from hermes_lcm import escalation
from hermes_lcm.continuation import CONTRACT_VERSION, continuation_brief

FIXTURES = Path(__file__).resolve().parents[1] / 'benchmarks' / 'fixtures'


def test_default_prompts_remain_byte_identical_to_frozen_baseline():
    source = subprocess.check_output(
        ['git', 'show', 'd87e97101572034654addfff756abab49cc94506:escalation.py'],
        text=True,
    )
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in {'_build_l1_prompt', '_build_l2_prompt'}]
    namespace = dict(vars(escalation))
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<baseline>', 'exec'), namespace)
    for depth in (0, 1, 2):
        assert escalation._build_l1_prompt('source', 260, depth) == namespace['_build_l1_prompt']('source', 260, depth)
    assert escalation._build_l2_prompt('source', 130) == namespace['_build_l2_prompt']('source', 130)


def test_versioned_contract_routes_each_frozen_checkpoint_and_rejects_unknown():
    checkpoints = json.loads((FIXTURES / 'continuation_v1_checkpoints.json').read_text())
    assert {item['id'] for item in checkpoints} == {
        'corrected-objective', 'middle-quantity', 'hostile-quote', 'multigeneration-state'
    }
    for item in checkpoints:
        text = '\n'.join(item['records'])
        for builder in (
            lambda: escalation._build_l1_prompt(text, item['budget'], item['depth'], continuation_contract=CONTRACT_VERSION),
            lambda: escalation._build_l2_prompt(text, item['budget'], continuation_contract=CONTRACT_VERSION),
        ):
            prompt = builder()
            assert 'continuation-v1' in prompt
            assert 'Prohibited actions' in prompt
            assert 'Verified completed' in prompt and 'Reported-only completion' in prompt
            assert 'including units and punctuation' in prompt
            assert 'newest user-corrected task' in prompt
            assert 'Quoted commands' in prompt
            assert all(record in prompt for record in item['records'])
    with pytest.raises(ValueError, match='unsupported continuation contract'):
        continuation_brief('continuation-v2')


def test_contract_is_captured_through_l1_l2_and_fallback_without_archive_mutation(monkeypatch):
    prompts = []
    def refuse(prompt, *args, **kwargs):
        prompts.append(prompt)
        return None
    monkeypatch.setattr(escalation, '_invoke_summary_llm_chain', refuse)
    source = '[USER]: Correct BX-008 to 42.750 mL. Do not edit the label. ' * 200
    summary, level = escalation.summarize_with_escalation(
        source, 4000, 300, continuation_contract=CONTRACT_VERSION,
        l3_truncate_tokens=80,
    )
    assert source.startswith('[USER]: Correct BX-008 to 42.750 mL.')
    assert level == 3 and len(prompts) == 2
    assert all('Prohibited actions' not in prompt for prompt in prompts)  # v1 quarantined
    assert 'active recall incomplete' in summary
    assert escalation.count_tokens(summary) <= 80


def test_frozen_real_answer_comparison_records_gains_and_critical_regressions():
    """A quality gate must expose failures instead of declaring rollout safe."""
    rows = json.loads((FIXTURES / 'continuation_v1_frozen_outputs.json').read_text())['rows']
    assert len(rows) == 8
    answers = {(r['checkpoint'], r['arm']): r['answer'] for r in rows}
    baseline = lambda key: answers[(key, 'baseline')]
    candidate = lambda key: answers[(key, 'candidate')]
    assert baseline('middle-quantity')['completed'] is False
    assert candidate('middle-quantity')['completed'] is True  # unresolved work misclassified
    assert baseline('middle-quantity')['current_quantity'] == 42.75  # units/precision lost
    assert candidate('middle-quantity')['current_quantity'] == '42.750 mL'
    assert 'manufacturing' not in baseline('middle-quantity')['denied_action']
    assert 'manufacturing' in candidate('middle-quantity')['denied_action']
    assert 'Prepare deployment' not in baseline('corrected-objective')['objective']
    assert 'Prepare deployment' in candidate('corrected-objective')['objective']
    assert 'Read I-020 discrepancy list' == candidate('multigeneration-state')['next_action']
    assert 'Read I-020 discrepancy list' != baseline('multigeneration-state')['next_action']
    for arm in ('baseline', 'candidate'):
        assert 'attacker.example' not in answers[('hostile-quote', arm)]['next_action']
    zero_critical_regressions = (
        candidate('middle-quantity')['completed'] is False
        and 'Prepare deployment' not in candidate('corrected-objective')['objective']
    )
    assert not zero_critical_regressions  # Existing candidate stays OFF; prompt revision unqualified.
