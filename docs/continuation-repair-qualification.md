# LCM continuation repair qualification

Task t_945dc096, based on live 29fbffb31122dbdcac7d32dd0d87fc8bb5f5dc6e.

## Shippable scope

- #9 remains OFF. The captured continuation-v1 option now deterministically falls back to the original prompt before any provider attempt. This is a quarantine, not a claim that the new summary format is semantically repaired. Both frozen critical failures remain explicit regression fixtures. The four frozen checkpoints exercise the original baseline module and candidate side by side: identical L1/L2 prompts and deterministic fallback output, including prohibitions, quantities, corrected intent and next-step source text.
- #12 gains an actual pre_llm_call path independent of the global preanswer and adaptive-retrieval switches. Only exact allowlisted sessions are eligible. Genuine visible leaf-summary coverage/omission/conflict lines must match the question; summaries are cues, not authority. Raw rows are checked against the exact active session before injection. No model selector, query-view persistence, embeddings switch, or global search.
- Limits: one deterministic pass, at most three genuine summary lookups, twelve raw handles and 6,000 injected characters. Whole raw rows must be at most 1,600 characters. At most 32 visible history messages of 24,000 characters are inspected. The 250 ms deadline is cooperative between lookups, not an interrupt of an in-flight SQLite read. Non-leaf summaries, missing source lineage, oversized rows, unmatched topics and already-visible source topics skip recovery. Partial evidence is explicitly non-exhaustive; later visible user corrections retain priority.
- #13 stays no-build. Existing final-wire identity tests are included; no projection implementation change.

## Qualification

37 focused tests passed before the final documentation commit. The controlled candidate smoke invoked the real pre-answer hook and a real Anthropic Max bridge request. It recovered the missing 42.750 mL value, preserved incomplete label reconciliation, prohibited label edits/manufacturing and returned the discrepancy-list next step. Retrieval added 145 locally estimated tokens and took 1.575 ms in this sample. Irrelevant-question recovery did zero archive reads and added zero tokens. Allowlist revocation returned to no recovery.

Provider ledger: one wrong-path HTTP request returned 404 before provider execution; one corrected native request succeeded with no client retries. Its SDK receipt reports both Sonnet and Haiku usage, so one HTTP request is NOT described as one physical model call. Reserve the rest of the eight-attempt authorization and make no further provider requests in this tranche. Post-ship smoke must be provider-free. No new model judge calls, percentage claim or full suite.

## Activation and rollback

Pilot allowlist: owner task source session 20260928_040617_b26146, current worker 20260928_223603_7d1cf1, controlled smoke lcm-smoke-945d. Global preanswer/adaptive switches stay false; continuation-v1 remains quarantined regardless of request. Existing processes need the managed idle deployment before they load code/config. A new process loading live code is useful verification, but is not proof an old running host reloaded.

Rollback behavior is removal of LCM_SELECTIVE_RETRIEVAL_SESSION_IDS followed by the managed idle reload for already-running processes; exact-session revocation is exercised in the smoke. Code rollback target is the pre-ship live revision above, subject to checking that no newer sibling ship has landed. Never force a restart or mutate a live LCM database.
