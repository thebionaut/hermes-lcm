# Full compression redesign — acceptance evidence

Date: 2026-09-23

## Acceptance matrix

| Requirement | Evidence |
|---|---|
| Isolated implementation | Worktree `/root/hermes-projects/worktrees/hermes-lcm-compression`, branch `feat/full-compression-redesign-20260923`; live checkout remains untouched. |
| No direct live `state.db` access | Production inspection used code, non-secret config selection, logs/journals, and parent-supplied product status only. |
| Bounded incremental chunks | Existing background capture retains bounded tail/prefix reads, 8k configured leaf chunks, dynamic sizing, max 4 passes, atomic tool groups. |
| Exact source coverage | Existing publication CAS still requires exact ordered source identity/lineage and normalized non-overlap coverage. |
| No uncovered overflow eviction | New fail-before-provider durable mapping gate in `engine.py`; unarchived tails are never silently dropped. |
| Explicit retained-raw policy | Bounded `retained_raw_reference` archive locator says active recall is incomplete and semantic coverage is false; full final request is cap-checked. |
| Recent tail / decisions | Existing 24-message/24k-token tail remains; role-aware prompts prioritize corrections without claiming deterministic completeness. |
| Repeated compaction quality | Isolated-DB integration forces two real background prepare/publish generations and verifies exact raw source lineage. |
| Smaller summarizers | Route/source-specific output reservation accepts a 12k fallback by reducing chunks; normal and background paths are also exercised with bounded source chunks while main context remains 1,000,000 tokens. |
| Stale/superseded work | Existing fence/frontier/policy validation retained; no bypass or early publish added. |
| No restart/deploy | None performed. |

## Tests

Focused acceptance command:

```bash
/root/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/test_compaction_coverage_quality.py \
  tests/test_model_switch_preparation.py \
  tests/test_escalation_provider_trace.py \
  tests/test_escalation_deadline_cancellation.py \
  tests/test_async_background_compaction_source_coverage.py \
  tests/test_background_compaction_sprint_acceptance.py \
  tests/test_background_compaction_acceptance_hardening.py \
  -o 'addopts=' -q
```

Historical pre-review runs returned 72 and 77 passes. After independent-review repairs, the expanded focused gate (including honest coverage, hard-cap overflow, route-specific small-summarizer budgeting, normal/background source bounds, and affected guardrails) returned **99 passed, 737 deselected in 2.57s**.

A credential-dependent placeholder test failed identically on pristine `4e80e41`; it attempts the configured `openai-codex`/fallback auxiliary route in this non-interactive test environment and exhausts the foreground deadline.

Full-suite collection blocker:

```text
ImportError in tests/test_model_switch_host_integration.py:
hermes_state -> agent.memory_manager -> tools.registry resolves repository tools.py
as top-level tools, then fails its relative import.
```

This reproduces before test execution and is a repository/host import-path collision, not a candidate assertion failure.

Full run excluding the repository-root import collision: **3393 passed, 23 failed, 1 skipped in 1639.86s** after the hardening pass. The prior baseline already carried 20 unrelated failures. The three candidate-adjacent failures were stale expectations: the L3 marker now honestly says `active recall incomplete`, and two impossible one-token overflow fixtures expected synthetic tool-pair repair where the new contract deliberately fails closed rather than evict unarchived raw. Those expectations were updated in `d464fdd` and their three focused tests pass. The host-integration collection blocker is avoidable with `--import-mode=importlib`; its two tests pass. Representative remaining failures reproduce on pristine `4e80e41`: Codex 5.6 host-cap drift, externalized import payload expectation drift, placeholder rollover state, pre-answer hook config leakage, and shared admission-cap drift. A second 27-minute full run was not repeated after changing tests only; all changed production paths remain covered by the final focused gate.

Lint/diff checks:

- `git diff --check`: pass.
- Ruff on every changed Python file: **All checks passed**.

## Quality fixture semantics

The fixture is not a repeated-paragraph speed benchmark. It includes:

- beginning fact: project code and owner;
- middle decision: exact path and date;
- middle correction: old dosage superseded by final dosage;
- end requirement/blocker: provider constraint and exact ticket;
- second-generation deadline correction.

The deterministic regression includes more than 24 decision lines, a correction longer than 320 characters with a late negation, quoted tool instructions, and repeated generations. Product code does not regex-repair model output. Prompts require role attribution, newer-source precedence, and honest incomplete coverage; deterministic fallback explicitly labels active recall incomplete. The isolated-DB integration separately proves that two background generations retain exact raw source lineage.

## Integration verification required by parent

After cherry-pick (before deployment):

1. Run the focused command above and the repository suite in the integration environment with the host import path fixed.
2. Completed deterministically in the candidate: an isolated DB forces two background prepare/publish generations with varied synthetic facts and a late long correction, then verifies both summaries and each node's exact raw source IDs. A subscription-model quality probe remains optional and was not needed for correctness; no PAYG or private corpus was used.
3. Verify the active main-model budget is unchanged when selecting a smaller auxiliary summary route.
4. Through live `lcm_status`/`lcm_inspect`/`lcm_doctor`, record queue→prepared→published latency, supersession reason, raw backlog, active/published counts, and provider-attempt attribution.
5. Change only YAML `lcm.background_compaction_prepare_ratio` from 0.75 to 0.55 for first rollout; verify prepared work exists materially before publication/emergency pressure.
