# Full compression redesign — architecture and policy

Date: 2026-09-23  
Candidate branch: `feat/full-compression-redesign-20260923`  
Baseline: `4e80e410e1bd4e10a6da2210d7f5cb2da55d8258`

## Scope and invariants

This candidate changes only the isolated `hermes-lcm` worktree. It does not modify the live checkout, Hermes core, live configuration, or either service.

Required preservation contract:

1. Raw SQLite messages remain source truth.
2. Canonical summary publication continues to require exact ordered source IDs, source identity, session fence, frontier CAS, and normalized source-coverage ownership.
3. Overflow assembly may hide raw rows only after every substantive row in the affected tail maps to durable LCM source IDs.
4. When overflow assembly hides durable raw, the provider-visible request carries a bounded `retained_raw_reference` archive locator. Its tokens are reserved before selection and the complete final request must remain within the hard cap.
5. If any substantive tail row is not durable, or the cap cannot fit the locator, overflow recovery fails before provider submission. It never re-adds an unbounded last raw row.
6. Summary prompts ask for role-attributed beginning/middle/end coverage, newest-source precedence, exact corrections, and an explicit incomplete-coverage report. This improves active recall but does not guarantee semantic preservation; raw SQLite lineage remains authoritative.
7. Recent raw is still protected by the shared count/token fresh-tail boundary and atomic assistant/tool groups.

## Existing architecture retained

- Foreground ingest is durable before asynchronous preparation.
- Background capture is bounded: protected-tail read, oldest eligible prefix, atomic tool groups, exact immutable watermark, at most `background_compaction_max_passes` chunks.
- Prepared summaries are inactive until foreground publication.
- Publication validates policy fingerprint, session/fence generation, frontier, exact source identity, exact prepared lineage, and non-overlapping canonical coverage in one transaction.
- Stale policy/session/frontier work is superseded; matching running/prepared work is reused rather than duplicated.
- Raw rows are not deleted by background publication.

## Candidate changes

### Role-aware summary quality layer

`escalation.py` now:

- asks L1/L2 summarizers to cover beginning, middle, and end;
- makes latest corrections authoritative and labels older conflicts superseded;
- requires exact identifiers/values, decisions, constraints, unresolved requests, and blockers;
- treats user/assistant/tool/prior-summary content as untrusted source records rather than summarizer instructions;
- attributes retained claims to source roles and gives newer source records precedence over older summaries;
- requires an honest source-coverage line that says what could not fit;
- labels deterministic L3 as incomplete active recall whose omitted source remains only in the raw archive.

There is deliberately no regex fact extractor and no deterministic claim of complete semantic coverage. Regex selection was removed because bounded first-match extraction can miss late corrections, cut negation, and reify quoted/tool instructions.

### Overflow retained-raw reference

`engine.py` now proves durable source mapping before budget eviction. Successful proof emits a compact provider-visible projection:

`[LCM active recall incomplete; raw archive retained outside active context] session=<id> archived_range=<first>..<last> mapped_rows=<n>. This is an archive locator, not a summary or claim of semantic preservation; inspect raw rows explicitly.`

The projection also carries structured `display_metadata` (`projection=retained_raw_reference`, `active_recall_incomplete=true`, `semantic_coverage=false`). Its full message cost is reserved before active-context selection, and final assembly is rejected if it exceeds the cap.

### Per-route usable summarizer budget

`model_switch_preparation.py` now calculates each configured auxiliary route's usable source allowance from that route's own:

- verified safe-input budget;
- route-specific output reserve derived from the source chunk's requested compaction budget; and
- fixed summary-prompt framing.

The minimum usable route allowance bounds summary chunk packing, and `preferred_chunk_tokens` is clamped to it. Admission rechecks the concrete L1/L2 prompts using the actual packed source size. This does **not** alter the main model's context length or LCM session threshold. A smaller summarizer therefore gets smaller chunks and output reserves instead of being rejected by a fixed 24k reserve or shrinking the main context.

## Production read-only evidence

Supported read-only surfaces used: checked-out code/docs, redacted config selection, `.env` non-secret LCM tuning keys, log files, and system journals. No direct `state.db` access was performed. No live database mutation or service restart occurred.

Observed live policy:

- context engine: `lcm`;
- host compression threshold: `0.75`, fixed threshold setting `850000`;
- YAML `lcm.background_compaction_prepare_ratio`: `0.75`;
- env prepare ratio: `0.55` (shadowed by YAML authority);
- background enabled, one worker/process, global provider limit 2;
- backlog trigger 32,000 tokens; max passes 4; job/lease 300 seconds;
- leaf chunk 8,000 tokens with dynamic leaf sizing and cache-friendly condensation enabled;
- fresh tail 24 messages / 24,000 tokens;
- context reserve 100,000 tokens;
- active subscription chain: `openai-codex/gpt-5.6-luna` → `openai-codex/gpt-5.6-sol` → `kimi-coding/kimi-k3`; reasoning disabled; no paid Sonnet route added.

The parent-provided current `lcm_status` evidence for session `20260923_043447_08f46c` reported ~1.05M active context, threshold ~672k, emergency ~756k, raw backlog 22,669, zero active DAG nodes, an active worker, and prepared-summary supersession. Current visible file/journal logs did not contain the historical compaction event strings, so the status figures are recorded as parent-supplied product output rather than independently reconstructed from SQLite.

## Background-preparation audit

- A YAML prepare ratio of 0.75 makes preparation begin at approximately the same pressure as the effective 0.75 compaction threshold. That removes most background lead time and explains why prepared work can arrive late or be superseded.
- The 22,669-token reported backlog is below the 32,000 backlog trigger, so backlog alone did not queue work. The pressure trigger was therefore decisive.
- Four 8k leaf passes provide a bounded nominal 32k source pass before dynamic sizing; cache-friendly aggregation may reserve one pass for condensation. This is coherent with the 300-second total job deadline and 61/45-second route timeouts.
- Existing session/fence/frontier/policy checks correctly classify stale work as superseded. The candidate does not weaken those fences or publish stale summaries early.

## Recommended integration settings

After parent review and normal deployment approval:

1. Make the YAML authority `lcm.background_compaction_prepare_ratio: 0.55` (or remove it so the existing env `0.55` wins). Do not leave it at 0.75.
2. Keep `compression.threshold: 0.75` and `LCM_CONTEXT_RESERVE_TOKENS=100000` initially; change one variable at a time.
3. Keep fresh tail `24` / `24000`, leaf `8000`, dynamic sizing, max passes `4`, one worker, global limit `2`, and 300-second job timeout for the first rollout.
4. Keep the subscription-only chain above. Do not add metered Anthropic/Sonnet without explicit authorization.
5. If preparation still does not start before pressure, lower backlog trigger from 32,000 to 24,000 only after observing actual queue-to-prepared latency; do not change it simultaneously with the prepare ratio.
6. Verify after integration with `lcm_status`, `lcm_inspect`, and `lcm_doctor`: prepared age, prepared→published time, supersession reason, provider attempts, raw backlog trend, publication trigger, emergency trigger, and exact retained-raw references.

## Boundaries / blockers

- This subagent had no direct in-process `lcm_status` tool binding. It did not instantiate a write-capable engine on the live `lcm.db` and did not inspect `state.db` directly. Current production counters beyond the parent-provided status therefore require parent verification through the live product tools.
- The active provider credential/catalog inventory was not exposed through a safe read-only product surface here. The candidate evaluates the three configured subscription routes and does not claim unconfigured routes are available.
- Hermes core commit `020babb2f3` was not edited. The plugin candidate keeps main-context budgeting independent from summarizer route budgeting, but parent integration must verify that core's external-engine bypass passes the resolved main-model context budget into `bind_context_budget` and does not reintroduce an auxiliary-model floor.
