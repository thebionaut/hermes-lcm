# Async background compaction

This is the production contract for opt-in Hermes-LCM background compaction. The feature moves summary-provider work off the turn-critical path, but it does not make prepared summaries active in the background. Foreground code remains the sole publisher of canonical summary nodes and advances the conversation frontier with a SQLite compare-and-swap (CAS).

The design has five non-negotiable invariants:

1. Raw rows in `messages` remain authoritative and are not deleted by this feature.
2. With the feature disabled, no background store, worker, or optional table is created.
3. Background preparation provider calls run outside SQLite transactions and outside the foreground turn. The bounded emergency path is the explicit foreground exception.
4. Prepared nodes are inactive: normal DAG, search, expand, recall, GC, and active-context readers cannot see them.
5. Canonical nodes, lifecycle-frontier movement, and job publication are one durable transaction. Active-context assembly happens only after that transaction commits.

## Configuration

All nine settings are explicit environment overrides. Defaults preserve existing foreground-only behavior because the master enable flag is false.

| Config field | Environment variable | Default | Meaning |
| --- | --- | ---: | --- |
| `background_compaction_enabled` | `LCM_BACKGROUND_COMPACTION_ENABLED` | `false` | Enables the optional store, preparation, and foreground publication paths. |
| `background_compaction_workers` | `LCM_BACKGROUND_COMPACTION_WORKERS` | `1` | Process-local worker request. Normalization clamps this to `0` or `1`; `0` keeps operator/test one-shot preparation available without an automatic worker. |
| `background_compaction_global_limit` | `LCM_BACKGROUND_COMPACTION_GLOBAL_LIMIT` | `2` | Maximum live provider permits across all processes sharing the database; normalized to the safe range `1..2`. |
| `background_compaction_max_passes` | `LCM_BACKGROUND_COMPACTION_MAX_PASSES` | `4` | Maximum source chunks/provider passes captured in one job. |
| `background_compaction_job_timeout_seconds` | `LCM_BACKGROUND_COMPACTION_JOB_TIMEOUT_SECONDS` | `120` | Total wall-clock deadline shared by every provider call in one job. |
| `background_compaction_lease_seconds` | `LCM_BACKGROUND_COMPACTION_LEASE_SECONDS` | `300` | Job, publication, and provider-permit lease duration; live work heartbeats it. |
| `background_compaction_backlog_tokens` | `LCM_BACKGROUND_COMPACTION_BACKLOG_TOKENS` | `32000` | Durable old-raw backlog that independently makes preparation eligible. |
| `background_compaction_min_provider_source_tokens` | `LCM_BACKGROUND_COMPACTION_MIN_PROVIDER_SOURCE_TOKENS` | `256` | Provider-input floor; smaller isolated ranges use an exact lossless local projection and make no provider call. |
| `background_compaction_prepare_ratio` | `LCM_BACKGROUND_COMPACTION_PREPARE_RATIO` | `0.55` | Active-token pressure fraction that independently makes preparation eligible. |

Integer limits except worker count normalize to at least 1. The global provider limit additionally clamps to at most 2. The preparation ratio clamps to `[0.0, 1.0]`; an invalid or non-finite value returns to `0.55`.

`background_compaction_global_limit` is not a thread-pool size. Every enabled Hermes process may request its one process-local worker, while durable provider permits enforce the shared cross-process limit.

## Thresholds

Let:

- `E` be the engine's effective context window (`context_length`);
- `R` be `context_reserve_tokens`;
- `B` be `background_compaction_backlog_tokens`;
- `r` be `background_compaction_prepare_ratio`;
- `A` be observed active prompt tokens;
- `D` be the token estimate of durable raw rows after the current frontier and outside the protected fresh tail.

### Preparation

Preparation is eligible when a non-empty, summarizable durable prefix exists and either:

```text
D >= B
```

or:

```text
E > 0 and A >= floor(E * r)
```

This is only a queue/preparation trigger. It does not change active context.

### Prepared publication

The fixed-reserve foreground publication boundary is:

```text
P = max(1, E - R)                  when E > 0 and R > 0
P = threshold_tokens              otherwise
```

Outside emergency/overflow recovery, foreground preflight at `A >= P` requests compaction only if a prepared job matches the current conversation, session, frontier, and live policy fingerprint. Below `P`, a prepared job stays invisible.

For example, an effective window of 1,000,000 tokens with a 100,000-token reserve publishes no earlier than 900,000 tokens.

With the production 100,000-token reserve, the required mappings are:

| Effective window | Publication trigger |
| ---: | ---: |
| 372,000 | 272,000 |
| 900,000 | 800,000 |
| 1,000,000 | 900,000 |

### Emergency foreground fallback

The automatic emergency boundary is:

```text
Q = max(1, E - max(32000, floor(0.05 * E)))    when E > 0
Q = 0                                          when E <= 0
```

At or above `Q`, or when the existing overflow-recovery guard fires, the engine enters the foreground fallback when preparation is absent, stale, blocked, or too small. One 15-second monotonic deadline begins at public `compress()` entry and covers token counting, reconciliation, ingest and SQLite waits, canonical-frontier adoption, publication, context reconstruction, and at most one automatic provider leaf pass. Expiry returns the exact caller message list, records the stage, and leaves durable rows/background work intact. Immediately before starting its provider thread, the engine atomically acquires a permit from the same durable global/per-conversation pool as background workers. If global capacity is unavailable, or another foreground emergency call already holds the conversation permit, it does not wait or call the provider; it uses deterministic overflow recovery instead. A background preparation lease on the same conversation does not block the emergency call (a large backlog would otherwise hold that lease job after job and starve emergency summarization): the two calls may overlap, the global cap still counts both, and the frontier/fence/source-identity CAS picks one canonical publisher; a winning foreground publication supersedes the background job. A successful permit is held through canonical publication. If the daemon provider thread ignores its timeout, the foreground stops waiting at the deadline and never publishes that late result, while an independent guardian heartbeats the permit until the physical call returns and then releases it.

If a prepared batch publishes at emergency pressure but the rebuilt prompt remains at or above `Q`, the engine continues into the same one-pass/shared-deadline emergency fallback. Manual `force=True` compaction retains the existing synchronous/manual behavior rather than entering background mode.

## Optional additive schema

The optional schema is created lazily only when an enabled engine opens `BackgroundCompactionStore`. Startup runs normal versioned migrations first, then creates the following tables inside an idempotent `BEGIN IMMEDIATE` transaction and records migration step `background_compaction_v1`:

- `lcm_compaction_jobs`: captured source range, source-identity hash, durable session-fence generation, policy fingerprint, state, retry data, job/publication lease, progress counts, and terminal outcome.
- `lcm_compaction_session_fences`: one durable publishable generation for each conversation/session binding, tied to the lifecycle bind and completed-reset epochs. Reset closes and advances this fence before DAG pruning; only a current lifecycle epoch can open a newer generation.
- `lcm_compaction_prepared_nodes`: inactive summaries keyed by job and ordinal, with exact summarized source lineage and provider/model metadata.
- `lcm_compaction_provider_leases`: the durable cross-process provider-call permits used for global concurrency and timeout fencing. Once a physical call starts, its row also stores the owning Linux boot ID, PID, and procfs start ticks so PID reuse cannot impersonate a live owner.
- `lcm_compaction_events`: a bounded, identifier-safe historical ledger for normal lifecycle outcomes and transient faults that cannot be reconstructed from the current job row alone.
- `lcm_compaction_source_coverage`: one row per canonically owned raw store ID, with a `(session_id, store_id)` primary key that makes overlapping publication a database constraint rather than a full-history JSON scan.
- `lcm_compaction_node_sources`: normalized `(node_id, session_id, store_id)` lineage used only to elect a surviving owner when legacy duplicate leaves are reassigned or deleted. It is derived from `summary_nodes` and backfilled in bounded batches.
- `lcm_compaction_node_edges`: normalized parent-declared DAG edges used by active-frontier and repair-topology checks. An edge is active only while its parent and child are both live in the edge's parent session.
- `lcm_compaction_dag_epochs`: a monotonic per-session mutation generation. Repair jobs bind this generation before provider I/O and require the identical generation at publication.
- `lcm_compaction_range_dispositions`: durable terminal or locally prepared range outcomes keyed by conversation, session, fence, frontier, watermark, policy, and exact source identity. Identical terminal work stays suppressed across restarts and processes until one of those inputs changes.
- `lcm_compaction_attempts`: bounded, content-free route-attempt telemetry including actual provider/model attribution when the host supplies it, latency, outcome class, token counts, and fallback selection.
- `lcm_compaction_publications`: one content-free audit row per successful publication CAS, including the fence, observed active tokens, effective window, calculated publication/emergency triggers, fixed-reserve policy, provider/model, and prepared-to-published wait.

Indexes provide:

- one active job per conversation for states `queued`, `running`, `prepared`, and `retry`;
- ordered claim selection by readiness, priority, and age;
- expired-job-lease lookup;
- ordered prepared-node lookup;
- provider-permit expiry and per-conversation lookup; and
- scoped event lookup and bounded retention; the source-coverage primary key is
  the publication overlap fence;
- coverage-owner lookup by `publication_id`; and
- surviving-node lookup by `(session_id, store_id, node_id)` without decoding
  DAG JSON.

Schema creation is safe when two processes enable the feature simultaneously. The migration is additive, backfills missing job columns from earlier development schemas, installs triggers for all future message-source DAG inserts, deletes, and lineage/session updates, and imports existing DAG lineage in write batches of at most 500 source IDs. Every batch re-reads and validates the node's current `session_id`, `source_ids`, and `source_type` under the same write transaction, so a concurrent carry-over cannot leave stale coverage in the old session.

Coverage and normalized lineage have independent completion markers: `background_compaction_source_coverage_v1` and `background_compaction_node_sources_v1`. This lets a database whose coverage migration already completed upgrade without discarding or rescanning coverage. Until both markers are durable, migration-only triggers preserve coverage during concurrent DAG writes; the final marker transaction atomically replaces them with the indexed definitions. The installed DELETE and relevant UPDATE triggers touch only coverage rows owned by the changing node and indexed surviving map rows. They never scan `summary_nodes` or invoke `json_each` to find another owner.

Normalized DAG edges have their own completion marker, `background_compaction_node_edges_v1`. Edge triggers are installed before the bounded backfill, so inserts, lineage updates, deletes, and session moves remain visible while two processes race to open the migrated database. The stored `session_id` is the current parent session; active-frontier and repair-topology readers additionally join the live parent and child in that session. Existing expand and recall APIs continue to follow canonical node IDs, preserving their preexisting lineage behavior.

Provider-owner fencing has its own additive marker, `background_compaction_provider_owner_v1`. Databases that already contain the original provider-leases table receive nullable owner-identity columns plus a default-inactive physical-call flag. Those legacy rows retain timestamp-expiry recovery; only calls that durably record a complete Linux process identity receive the stronger liveness fence.

Existing production DAGs may contain duplicate legacy leaves: backfill elects one owner per `(session_id, store_id)`, and deletion, reset, or reassignment of that owner atomically re-homes coverage to the lowest surviving duplicate. Coverage is released only when no canonical message-source node remains. Newly inserted nodes and newly added source IDs still use strict inserts, so lifecycle handling cannot become an overlap bypass. Core bootstrap recognizes the `lcm_compaction` table family, but does not create it while the feature is disabled.

The migration itself never moves a lifecycle frontier. On a later foreground boundary, an enabled engine may discover that an older canonical leaf already covers raw rows immediately above a lagging frontier (for example, an old process exited after committing the DAG node but before its separate frontier write). Before enqueueing any provider work, the engine scans at most 513 conversation-scoped rows and advances across at most 512 already-canonical rows in one short transaction. It proves ownership through the normalized coverage tables, stops at the first gap or incomplete assistant/tool-result group, CAS-checks the live session frontier and publication-fence generation, and supersedes jobs captured against the old frontier. The normal foreground adoption path then rebuilds active context from the canonical DAG. No provider is called, no raw or canonical row is rewritten, and longer legacy prefixes are amortized at one bounded batch per foreground boundary.

### Legacy partial atomic-group repair

A legacy database can contain an assistant/tool-result group for which only some raw rows already have live canonical owners. Neither ordinary prefix adoption nor ordinary publication may cross that group. Capture instead enqueues one high-priority `partial_atomic_repair` job for the exact oldest complete group.

The job partitions the group by live normalized ownership, not by a possibly stale coverage election. Under the enqueue transaction it re-elects a valid live owner for every covered row and removes orphaned coverage for rows with no live owner. It never deletes a canonical node, valid ownership, or raw message.

Before provider I/O, the worker binds all live message-source owners, their complete bounded ancestor closure, every active maximal root, root identities, topology hash, and the session DAG epoch. Provider input contains the filtered exact atomic group plus the implicated root summaries, clearly separated as existing canonical context and uncovered continuation. The normal ignore, dependent-message, sensitive-redaction, and externalized-payload rules still apply. A retry may reuse prepared output only while the bound roots and topology are identical.

Foreground publication validates the immutable raw identity, ownership partition, coverage elections, lifecycle frontier, session fence, lease generation, DAG epoch, root identities, and topology in one short transaction. It then:

1. inserts a provenance-neutral message-source bridge owning only the previously uncovered rows;
2. inserts the provider-produced node-source parent over every bound root plus that bridge;
3. advances the frontier across the complete atomic group; and
4. marks the repair published.

The bridge and parent are committed or rolled back together. Covered rows acquire no new message-source owner, old roots remain durable but become inactive children, raw rows remain unchanged, and DISTINCT expansion through the parent returns the complete group plus any legitimate root spill exactly once.

If emergency pressure arrives while a matching repair is queued, running, retrying, or prepared but not publishable, foreground compaction fails closed without a provider permit or provider call. It returns the existing deterministic, provider-valid overflow view, records `partial_atomic_repair_pending`, and reports `partial atomic repair pending`. If a small prepared repair publishes but pressure remains critical, the engine first enqueues against the new frontier; a subsequent legacy partial group is therefore detected before any ordinary emergency provider path can overlap it.

## Capture and source identity

Foreground ingest remains synchronous and durable. After ingest, capture performs no provider call and uses a 100 ms SQLite write timeout so background queue contention cannot hold the turn.

The caller's message list is never mutated in place. Ingest, publication, sibling-frontier adoption, and emergency assembly work on new lists; the background runtime has no reference to a turn's in-memory list at all.

Capture uses bounded reads rather than materializing the whole durable suffix.
It inspects at most 255 newest rows to establish the protected tail, scans at
most `B` oldest positive token estimates for backlog eligibility (`32,000` rows
at the default trigger), then materializes at most 769 oldest eligible rows (512
normal rows, 256 tool-group look-ahead rows, and one sentinel) to propose a job.
Because each scanned estimate is positive, inspecting at most `B` rows is
sufficient to decide whether their sum reached `B` without work growing with
the complete transcript. If an unusually large configured tail or tool group
cannot be proven safe within those bounds, capture defers instead of splitting
it.

Capture:

1. Reads raw rows for the current session/conversation strictly above the lifecycle frontier.
2. Resolves the protected suffix with the same `fresh_tail_count` and `fresh_tail_max_tokens` policy used by foreground compaction.
3. Selects only the oldest prefix before that suffix and stops before the first atomic group containing normalized canonical coverage. This lets an uncovered gap publish before a later legacy canonical island; the next foreground boundary can then adopt that island instead of repeatedly preparing an overlapping range.
4. Excludes a leading system anchor from provider input at frontier zero while allowing the consumed watermark to advance across it; the caller preserves the active system message.
5. Applies the shared stored-row ignore/dependent-reply filter and summary serializer. Sensitive data, tool arguments, injected context, and large outputs receive the same redaction/externalization treatment as foreground summaries.
6. Keeps an assistant tool call and its contiguous tool results in one atomic chunk.
7. Uses the configured leaf size, including dynamic doubling up to `dynamic_leaf_chunk_max`, and caps the job at `background_compaction_max_passes` chunks.

If the exact provider-visible source is below `background_compaction_min_provider_source_tokens`, capture prepares a deterministic `lossless_raw_projection` locally. The projection contains the policy-redacted serialized rows verbatim, owns the exact raw IDs, and is attributed to `local` / `deterministic/lossless-raw-v1`; it never calls an LLM. This lets an isolated externalized marker or other tiny canonical gap advance without pretending it was summarized, while exact expansion still reads the authoritative raw rows.

Once claimed, a job's watermark is immutable; later foreground rows remain above it. An explicit coalesce may extend a still-queued or retrying request before its next claim, but merely appending raw rows does not move the stored watermark.

The job stores two related lineages:

- `source_store_ids` on the job are the consumed durable raw range through the immutable captured watermark, including rows filtered from provider input.
- `source_store_ids` across the prepared nodes are an exact ordered partition of the complete consumed range. Rows excluded from provider input (a leading system anchor, ignored messages, and dependent replies) remain assigned to the adjacent node's expansion lineage, so every consumed raw row is recoverable exactly once.

The job hashes canonical identity fields for each consumed durable row, including store/session/conversation identity, role, stored content, tool metadata, and timestamp. For an externalized payload reference, that stored reference marker is the source identity and the only representation sent through background summary preparation. Capture, preparation, and publication do not hydrate or hash sidecar payload bytes. This prevents a new provider-data path and keeps filesystem I/O out of the foreground and publication transactions; payload recovery and lifecycle remain the responsibility of the existing externalization subsystem. Publication fails closed if any captured SQLite row changes or disappears.

The bounded foreground emergency path uses the same rule. Before provider I/O it maps the selected active rows to one exact durable interval, captures those rows, and immediately repeats the active-to-store mapping to close a concurrent mutation gap. Publication requires the exact consumed ordered IDs and the identity hash derived from that pre-provider snapshot. A hidden durable row, a changed active mapping, or any SQLite content/tool/timestamp mutation before commit makes the attempt a no-op; it never broadens canonical lineage to rows the provider did not summarize or removes more caller messages than it published.

Queued or retrying work for the same conversation/frontier/policy/session-fence generation may coalesce monotonically. Retry coalescing preserves its backoff and error state. A reset, session, policy, frontier, or fence-generation change supersedes stale work instead of letting the one-active-job index block the new generation.

Capture rechecks the complete selected ID set for canonical coverage after acquiring its short `BEGIN IMMEDIATE` enqueue transaction. A concurrent canonical DAG insert between the bounded read and enqueue therefore supersedes overlapping active work and returns without creating a provider job.

The policy fingerprint includes preparation protocol version plus every configured input that changes selection, filtering, serialization, chunking, summary routing, or summary output. Workers claim only jobs matching their live fingerprint. A changed deployment therefore leaves old work unclaimed until the foreground enqueue path supersedes it; stale work cannot silently publish under a new policy.

This feature does not repurpose `LCM_DEFERRED_MAINTENANCE_ENABLED`. Background mode disables the foreground deferred-maintenance path; durable jobs and the background runtime own opportunistic maintenance instead.

## Job lifecycle

Active states are `queued`, `running`, `prepared`, and `retry`. Terminal states are `published`, `superseded`, and `failed`.

```text
queued ──claim──> running ──all chunks durable──> prepared ──foreground CAS──> published
  │                 │                                  │
  │                 ├──provider/error/deadline──> retry┤
  │                 ├──five attempts────────────> failed
  │                 └──stale source/policy/frontier──> superseded
  └──session/policy/frontier replacement─────────────> superseded
```

### Claims, heartbeats, and permits

Claim uses a short `BEGIN IMMEDIATE` transaction. In that transaction it:

1. recovers expired leases;
2. counts unexpired provider permits against the global limit;
3. chooses a due `queued` or `retry` job whose policy and publishable session-fence generation match and whose conversation has no live provider permit;
4. changes the job to `running`, increments attempt count and lease generation, and assigns a generation-fenced job lease;
5. creates the matching durable provider permit.

The worker heartbeats both records. Every worker progress/completion write validates owner, generation, and unexpired lease, so a late worker from an expired generation cannot save progress or publish.

Reset and compression carry-over first close the old session's durable publication fence and supersede active jobs in the same transaction, then prune or reassign DAG nodes. A reset that cannot acquire this short fence fails closed by preserving derived nodes. The lifecycle reset epoch is recorded only after batched DAG mutation completes. Fence activation rechecks the lifecycle session plus bind/reset epochs inside its own writer transaction, so a stale process cannot overwrite a newer process's bind; a transient lock can be retried on a later foreground turn without reopening an in-progress reset. Rebinding the same conversation and session after an ordinary finalized tool/API turn restores its valid frontier and keeps the current fence generation; only a reset, rollover, reassignment, stale frontier/source, policy change, or explicit administrative invalidation advances the fence. Prepared publication, foreground emergency permit acquisition, and final emergency publication all require the originally captured fence generation.

### Partial progress and retry

Each completed provider chunk is upserted immediately into the inactive prepared-node table, outside any provider call. On retry, matching completed ordinals are reused and only missing chunks are sent again. The job becomes `prepared` only when the expected node count is present and the concatenated prepared lineage exactly equals the consumed range.

When `cache_friendly_condensation_enabled=true` and more than one leaf fits, capture reserves one pass from `background_compaction_max_passes` for aggregation. The worker durably prepares up to three leaf chunks, then uses the fourth pass to condense those leaves into one cache-stable prepared node with the exact ordered union of their raw lineage. If aggregation fails, the leaves remain inactive durable progress and a retry invokes only the missing aggregation pass; successful finalization transactionally replaces the intermediate leaves with the single publishable node.

Retry delay is exponential and bounded:

```text
delay = min(300 seconds, 5 * 2^(attempt_count - 1))
```

The fifth failed attempt is terminal `failed`. A terminal range disposition records the exact conversation/session/fence/frontier/watermark/policy/source identity, so another turn or process cannot recreate identical work. Changed source material, frontier, fence, or policy naturally produces a different key and can be prepared. A non-compressing final escalation is terminal immediately rather than consuming five identical job attempts. Provider exception bodies are never persisted. Durable errors contain only an allowlisted exception class, failure category, and optional numeric HTTP status; legacy free-form values are replaced with an opaque redacted marker on operator surfaces.

Expired `running` work returns to due `retry` with a new lease generation. An expired publication lease on `prepared` work is cleared without discarding the prepared nodes.

### Timeout-ignoring providers

`background_compaction_job_timeout_seconds` is one deadline for the whole job, starting immediately after the durable claim rather than after source loading. Each provider receives the smaller of the normal summary timeout and the job's remaining time.

If a provider call outlives the deadline, the job moves to retry immediately, but its provider permit remains active. A daemon guardian with a private SQLite connection heartbeats that permit until the physical call returns, then releases it and wakes the local runtime. Guardian connection-open, heartbeat, and completed-call release failures are retried through SQLite lock contention; completion is not considered fenced until the durable row is deleted. This prevents a timeout-ignoring client from hiding real concurrent spend behind a retry. The guardian is a concurrency fence, not transport cancellation.

One missed timestamp heartbeat is not proof that provider capacity became free. On Linux, expiry recovery checks the recorded boot ID, PID, and process start ticks. A matching live owner—or an indeterminate procfs read—renews the permit and fails closed; a missing process, boot mismatch, or PID-reuse mismatch permits normal recovery. If the entire process exits while native/client code is still blocked, the identity becomes dead and the durable permit is reclaimable after its configured lease. Status exposes active and timed-out permits, their age, and remaining lease so an operator can identify a permanently wedged call; restart that live process only after confirming no other work depends on it.

### Runtime and shutdown

Within one process, engines using the same canonical database path share one reference-counted runtime. The runtime owns one worker thread and one private store connection, installs the normal summary circuit breaker and spend guard, and uses notifications plus a 5-to-30-second durable polling backstop so a missed cross-process notification cannot strand work.

Shutdown signals cancellation, wakes the worker, waits up to five seconds, and closes stores only after their current transaction boundary. If a hot rebind acquires the same database while an old runtime is still stopping, a registry-bound deferred handle starts exactly one successor only after the old thread exits; it neither duplicates a worker nor strands queued work.

### Compatibility with opt-in derivation hooks

File extraction and assertion extraction are separate provider-backed derivation paths whose established timing and receipt semantics are foreground-only. If `extraction_enabled=true`, or both `assertions_enabled=true` and `assertion_extraction_enabled=true`, async background compaction is therefore compatibility-blocked for that engine. Status reports the configured master flag, effective disabled state, zero workers, and the exact blocker reason. Existing foreground compaction and its hooks continue unchanged. Production may enable the background path only while those opt-in derivations are off; no new background data-egress route is implied by this feature.

## Preparation is not publication

Preparation writes only `lcm_compaction_prepared_nodes` and job metadata. It does not write `summary_nodes`, move `lcm_lifecycle_state.current_frontier_store_id`, increment foreground compaction count, change active replay, or make prepared IDs available to search/expand/recall.

Foreground publication first acquires a short generation-fenced lease on a `prepared` job, then runs a single transaction with a 100 ms SQLite busy timeout. Inside that transaction it validates:

1. job state and publication lease;
2. the live policy fingerprint;
3. current publishable session-fence generation;
4. current session and expected lifecycle frontier;
5. exact consumed store-ID range and source-identity hash;
6. absence of canonical overlap through the bounded source-coverage primary-key fence;
7. exact prepared-node count; and
8. ordered prepared lineage exactly equal to the consumed raw range.

It inserts canonical depth-zero `summary_nodes`; the coverage trigger reserves every exact raw store ID in the same statement, then the transaction advances the lifecycle frontier with a CAS and marks the job `published`. Any exception rolls coverage, nodes, frontier, and job state back together. Existing DAG lineage is backfilled once during migration, so publication never scans and decodes the full historical DAG while holding the writer lock. A stale validation outcome marks the job `superseded` without adding canonical nodes or advancing the frontier.

Active-context assembly occurs after commit. The caller keeps the leading system anchor and the exact in-memory suffix above the committed watermark; durable rows are used to fill only concurrent suffix rows from the same session and conversation that are not represented by the caller. Reusing a session ID for another conversation cannot inject that conversation's rows into publication replay or sibling adoption. The in-process frontier marker moves only after assembly succeeds. A sibling process or a later call can adopt a durable frontier that committed before its local marker changed.

Emergency foreground summaries use the same short transaction and lifecycle, session-epoch, exact-ID, and source-identity CAS. Whichever publisher wins the frontier fence is canonical; the loser reloads the committed frontier instead of creating overlapping nodes. Successful foreground emergency publication supersedes overlapping queued/running/prepared/retry jobs.

## Diagnostics

Background state is included in `lcm_status`, `lcm_inspect`, `lcm_doctor`, and the `/lcm` text report. The bounded operator payload exposes:

- enabled flag, requested worker count, durable global limit, and worker availability;
- configured versus effective enablement and any extraction/assertion compatibility blocker;
- counts for every job state and age of the oldest queued job;
- current job state, frontier/watermark, debt, attempts, lease time remaining, retry time, and scrubbed error;
- current publication-fence generation, publishability, and last update time;
- last preparation and publication result;
- last foreground-deadline outcome and precise exhausted stage;
- last publication proof: observed active tokens, effective window, calculated publication/emergency triggers, fixed-reserve policy, fence, and prepared wait;
- recent content-free provider outcomes and the last actual route attempt, including latency, token counts, fallback, and whether attribution was available;
- prepared-node attribution coverage and unresolved durable range dispositions;
- backlog, prepare-ratio, publication, and emergency thresholds;
- durable frontier and process-local active coverage;
- active/timed-out provider-permit counts, oldest age, and minimum remaining lease; and
- scoped durable event counts plus normalized diagnostic reason codes.

Raw job and owner IDs are not exposed; operator-facing hints are one-way hashes. Text fields are bounded and credential-like values are redacted.

Recognized reason codes are:

| Reason | Meaning |
| --- | --- |
| `background_summary_prepared` | At least one inactive prepared job is available. |
| `prepared_summary_published` | Prepared work became canonical through the foreground CAS. |
| `prepared_summary_superseded` | Prepared or in-flight work lost to a policy/session/frontier change or another publisher. |
| `foreground_emergency_compaction` | The bounded synchronous fallback published progress. |
| `provider_timeout` | A background provider exceeded the shared job deadline. |
| `provider_connection_error` | The selected provider could not be reached. |
| `provider_auth_error` | The provider rejected authentication (HTTP 401/403). |
| `provider_quota` | The provider reported exhausted payment/quota capacity. |
| `provider_rate_limit` | The provider rate-limited the route. |
| `invalid_provider_response` | The provider returned an unusable response shape. |
| `reasoning_only_output` | The provider returned reasoning but no usable summary. |
| `non_compressing_output` | The provider output did not reduce the selected source. |
| `insufficient_compressible_source` | The isolated provider-visible source was too small; it was represented losslessly without a provider call. |
| `provider_circuit_open` | The summary circuit breaker rejected the attempt. |
| `provider_spend_guard` | The configured spend guard rejected the attempt. |
| `stale_lease` | The worker lost its generation-fenced job or provider permit. |
| `foreground_deadline_exhausted` | Foreground work stopped at its shared 15-second deadline and preserved the caller view. |
| `lease_expiration` | A job/publication lease expired or a generation lost ownership. |
| `sqlite_lock_contention` | A bounded queue/publication write could not obtain SQLite's writer lock. |
| `worker_unavailable` | Background mode is enabled but this process has no live runtime for the database. |
| `no_eligible_raw_range` | No non-empty summarizable raw prefix existed outside the fresh tail. |
| `session_reset_fence` | Reset or compression rollover closed the old session's publication generation before DAG mutation. |
| `canonical_prefix_adopted` | A foreground boundary advanced a lagging lifecycle frontier across an already-canonical contiguous legacy prefix without provider work. |
| `partial_atomic_repair_pending` | A mixed-ownership legacy assistant/tool group is waiting for its exact bridge-and-parent repair; emergency foreground compaction is failing closed. |
| `partial_atomic_repair_published` | The exact legacy atomic-group bridge and parent committed with their frontier CAS. |

`lcm_doctor` treats durable historical events and terminal jobs as observations. It warns for current retry work, status-read faults, or failed provider attempts in the bounded recent telemetry window; an old recovered timeout does not leave health permanently yellow.

The event ledger recognizes the documented codes and prunes oldest rows to a bounded retention limit. Conversation-scoped status never leaks results, events, attempts, dispositions, publications, or permits from another conversation; aggregate status is available only when no conversation scope is requested.

## Operational rollout

Do not use the production database as a migration test fixture.

1. **Verify the candidate while disabled.** Run focused background tests, affected LCM tests, the complete repository suite, concurrency stress, syntax/diff checks, and a staged secret scan. Confirm the production checkout is committed and clean.
2. **Inspect live state.** Record the deployed commit, service PIDs, current eight settings, database path, raw-message counts, lifecycle frontiers, and `PRAGMA integrity_check` result.
3. **Take a WAL-consistent backup.** Stop no services yet; use SQLite's backup API against the live connection path so the main database and WAL are captured coherently. Open the backup independently, run `PRAGMA integrity_check`, compare critical counts/frontiers, and record its path and hash.
4. **Set all eight variables explicitly.** Do not rely on mixed defaults across multiple Hermes processes. Start with the production values approved for that host; use `LCM_BACKGROUND_COMPACTION_WORKERS=0` for migration-only observation or `1` for automatic preparation.
5. **Restart every Hermes process that opens the database.** This installs one runtime per process/database and performs the lazy additive migration. Confirm new PIDs and successful service activation.
6. **Verify migration and disabled-reader invariants.** Check migration markers `background_compaction_v1`, `background_compaction_source_coverage_v1`, `background_compaction_node_sources_v1`, `background_compaction_node_edges_v1`, and `background_compaction_provider_owner_v1`; verify the optional tables/indexes/triggers including session fences, normalized edges/epochs, and provider-owner columns, SQLite integrity, unchanged raw counts, and zero unexpected canonical nodes/frontier changes.
7. **Exercise a normal turn.** Bind a real session, send a non-destructive message, and inspect status. Enqueue/preparation may occur, but active context must remain raw below the publication threshold. If migration found legacy canonical coverage above a lagging frontier, a bounded `canonical_prefix_adopted` transition is expected here (not during migration), must rebuild active context without a provider call, and must stop before the first uncovered row.
8. **Exercise preparation on an isolated database.** Use a controlled fixture to verify queue, claim, partial progress, retry, restart recovery, provider-permit fencing, prepared invisibility, atomic publication, and suffix preservation. Do not force artificial million-token debt into the live transcript.
9. **Observe production.** Review status and logs for duplicate provider calls, retry churn, expired leases, SQLite locks, worker duplication, prepared backlog, and unexpected frontier movement. Recheck raw counts and lineage after the first real publication.

## Rollback

The normal rollback is configuration-only:

1. Set `LCM_BACKGROUND_COMPACTION_ENABLED=false` in every process.
2. Restart every Hermes process that opens the database.
3. Confirm no worker is available, no job state changes, normal foreground compaction still works, SQLite integrity passes, and raw counts/frontiers are unchanged.

Disabling does not delete optional rows and does not undo canonical summary nodes that already published; those are ordinary valid DAG nodes. Leaving optional tables in place is safe because disabled engines do not instantiate the background store and canonical readers never consult prepared rows.

Do not drop optional tables during an incident. A binary older than the bootstrap recognition for the `lcm_compaction` table family may safely refuse the migrated database. Before such a downgrade, either keep the feature-capable binary with the flag disabled or restore a separately verified pre-migration backup, explicitly accepting loss of post-backup writes. Preserve the failed/live database and WAL for diagnosis.

## Release acceptance

Production enablement is complete only when all of the following are independently true:

- disabled mode creates no optional tables or workers;
- enabled migration is additive, idempotent, and leaves raw rows unchanged;
- migrated canonical coverage above a lagging frontier is adopted only on a foreground boundary, without provider work or skipping an uncovered gap;
- background provider work does not run on the foreground turn;
- prepared nodes remain absent from canonical readers;
- global permits, retries, restarts, and late-worker fences behave across independent SQLite connections;
- publication and concurrent foreground fallback produce exactly one canonical frontier transition;
- reset and compression carry-over reject stale prepared and foreground publishers before DAG deletion or reassignment;
- the exact in-memory suffix above the watermark survives publication;
- post-publication durable replay remains scoped to the active conversation when a session ID is reused;
- the automatic emergency path returns within its one-pass/15-second wall even when the provider ignores timeout;
- diagnostics distinguish normal preparation/publication/supersession from provider, lease, SQLite, worker, and eligibility faults; and
- production services run the committed candidate, status is healthy, and the checkout remains clean.
