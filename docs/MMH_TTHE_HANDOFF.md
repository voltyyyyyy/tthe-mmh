# Implementation handoff: MMH for TTHE

Prepared 2026-09-17. This is an implementation specification, not a claim that the work below is already complete.

## 1. Objective and agreed decisions

Extend the existing LiveCodeBench MMH integration so TTHE learns reusable lessons from executable harness changes, especially lessons that prevent repeating failed changes.

The user agreed to these decisions:

1. Primary objective: avoid repeating failed harness changes.
2. Supply memory to the proposer that edits harness code. Do not automatically inject memory into solver prompts.
3. Trust lessons only after successful evidence on later tasks, across multiple rounds.
4. Equal parent/candidate results are inconclusive, not successes or failures.
5. One regression is recorded and followed by more testing; it does not immediately discard the lesson.
6. Memory validation uses a small fixed share of the total inference budget, rather than unbounded additional calls.

Start with LiveCodeBench. Preserve the frozen solver and public-evidence-only adaptation protocol. Implement and verify the changes; do not launch a paid benchmark study as part of this handoff unless separately requested.

The numeric settings below are proposed configurable defaults, not empirically established thresholds or additional user decisions. Do not tune them against hidden scores.

## 2. Read the existing implementation first

Inspect applicable AGENTS.md instructions and current git changes before editing. Preserve unrelated work.

Read:

- `README.md`, `PROGRESS.md`, `docs/MMH_IMPLEMENTATION.md`.
- `meta_memory/types.py`, `engine.py`, `store.py`, `adapters.py`, `embeddings.py`.
- `livecodebench/mmh_adapter.py`, `lcb_optimize.py`, `lcb_proposer.py`.
- `livecodebench/lcb_bridge.py`, `harness_base.py`, and `text_to_sql/claude_wrapper.py` for execution and call accounting.
- `tests/test_meta_memory.py`, `tests/test_lcb_mmh_adapter.py`.

Existing building blocks include SQLite rule/patch persistence, five atomic patch operations, confidence/promotion, public-only DTOs, proposal cards, candidate associations, proposer retrieval, and later-task paired execution.

Specific issues observed in the current code:

- `validate_pending()` computes a strict improvement boolean; ties become failures.
- `MetaMemoryEngine.validate_patch()` resolves a pending patch on the first applicable observation. This is incompatible with the agreed online policy.
- Replay traverses eligible associations without a cost cap.
- Retrieval uses task descriptions, without structured failure evidence.
- Failed patches are not explicitly returned to the proposer as warnings against repeating interventions.
- Registration hashes the child, but replay does not enforce immutable parent and child artifacts.
- The proposal origin is supplied as `batch[0]`, even if the actual motivation came from another task.
- Adapter JSON and SQLite writes are separate, creating a crash-consistency problem.
- `--fresh` removes generated candidate files globally; persisted validation must not depend on those files remaining in place.

These are code inspection findings. Re-run existing checks to establish the implementation baseline.

## 3. Scope and architecture

Keep the existing TTHE fixed-branch search and agentic batch judge. Add a memory layer around it:

`public observations -> retrieve advice -> propose one intervention -> register frozen artifacts -> ordinary TTHE selection -> later paired validation -> update memory`

Winning the current batch is not independent validation. Losing the batch does not automatically make an intervention harmful. Nonwinning candidates may still provide useful validation evidence.

Separate four identities:

- A reusable rule describing when and how to intervene.
- An application of that rule to a particular parent harness.
- A memory patch that adds, edits, or removes rules.
- A paired execution observation on a later task.

Allow a proposer to apply an existing rule without inventing an ADD/REFINE patch. The current mandatory memory-patch card should be extended accordingly. Applying a rule and editing memory are different operations.

Keep domain-independent evidence policy and scheduling small and testable. Keep LiveCodeBench task conversion and public execution in its adapter. Do not refactor all domains during this work.

## 4. Evidence contract

Represent these outcomes explicitly:

| Outcome | Meaning | Effect on confidence |
| --- | --- | --- |
| improvement | Child passes more of the same public tests | Add one success |
| regression | Child passes fewer of the same public tests | Add one failure |
| inconclusive | Equal results, including both passing or both failing | No update |
| inapplicable | Rule prerequisites do not match | No update |
| execution_error | Infrastructure failure, invalid comparison, or missing artifact | No update |

Compare the same test set with matching test identities, model configuration, executor settings, and rollout allowance. A task contributes one observation, not one observation per public test. Zero tests or mismatched test sets cannot yield decisive evidence.

Distinguish generated-program failures from infrastructure failures: a validly executed program that crashes or times out may be a real failed public test; an unavailable model endpoint, killed worker, or missing artifact is not evidence against a rule.

Record outcome, applicability reason, parent/child artifact hashes, task ID, test fingerprint, batch/round, execution configuration fingerprint, cost, and public summaries. Use a persistent uniqueness key for an application/task/configuration comparison. Repeated tasks and resumed runs must not inflate evidence. Distinct tasks are not proof of statistical independence; promotion also requires distinct later batches.

Keep neutral/error observations in the audit log. They must not increment successes, failures, successful lifespan, decisive subset counts, or precedent outcomes.

## 5. Online lifecycle policy

Preserve the existing paper/demo behavior as an explicit legacy policy if other callers rely on it. Add an explicitly configured TTHE online policy; document the difference. Do not silently change the standalone demo's semantics.

Suggested online defaults:

- Beta prior: alpha=1, beta=1, retaining the existing confidence formula.
- Resolve a pending patch as supported after at least 3 decisive observations from 3 distinct later tasks spanning at least 2 later batches, with posterior confidence >= 0.75.
- Resolve as harmful after at least 3 decisive observations from 3 distinct later tasks spanning at least 2 later batches, with posterior confidence <= 0.40.
- Otherwise keep gathering evidence. A single regression never resolves the patch as harmful.
- Promotion additionally requires confidence >= 0.80, successes in at least 3 distinct later batches, elapsed volatile age >= 3 batches, and any explicitly required recovery condition.

For example, three successes yield confidence 0.80; two successes and one failure yield 0.60 and remain unresolved. Inconclusive observations do not change either result.

Define recovery precisely. Replaying the original failure may establish recovery, but does not count as independent later-task evidence. Fixing an unrelated later task must not automatically mark the original failure recovered.

Unresolved lessons remain tentative and may be retrieved with that label. Budget starvation or age alone must not turn them into failed lessons or trusted rules.

Continue monitoring supported rules. Later contrary evidence changes confidence and retrieval priority. Stable-rule destructive edits retain conservative cross-batch failure protection. Do not overwrite historical precedents silently or repeatedly resolve the same patch.

Avoid rollback overwriting later legitimate changes: unresolved patches that share target rule identities must be serialized, deferred, or version-checked. Test inter-round conflicts, not only conflicts inside one proposal call.

## 6. Proposer memory, especially failures

Return a bounded memory package with:

- Trusted applicable advice.
- Tentative advice with evidence counts and caveats.
- Failed interventions with their context, mechanism, and supporting public evidence.

Suggested defaults: at most 5 positive/tentative lessons combined and 3 failed interventions, within a configurable prompt budget. Deduplicate rules shared across a batch. Store evidence references rather than copying entire traces into every prompt.

Use structured task/execution features such as stdin versus functional interface, syntax/runtime/formatting error, observed timeout, truncated output, or exhausted repair attempts. Semantic similarity ranks candidates; it does not establish applicability. Check explicit prerequisites and report unknown applicability as such.

Use observations of the relevant parent to match failure conditions. Do not use the child's improvement to retrospectively decide that the rule was applicable. Share/cache parent observations to avoid redundant calls.

Do not classify every rejected patch as a failed intervention: malformed cards, influence-gate rejection, conflicts, and lack of budget are not empirical failures.

A negative lesson should say, for example, "This intervention regressed in this context; avoid repeating it unless a material condition changes." Do not encode task-specific answers. Allow an override with a stated changed condition and a new application record. Track exact repeated intervention/context fingerprints as a conservative repeat metric; do not claim semantic repeat detection is perfect.

## 7. Proposal and artifact contract

Extend cards to contain:

- Exact parent and child identities and source hashes.
- Origin task IDs and trace references, not an assumed first task.
- One declared behavioral intervention, its rationale, and expected effect.
- Applied rule ID(s), or an explicit new hypothesis.
- Applicability prerequisites and failure signature.
- Optional typed memory patch.
- Reason for overriding a relevant failed intervention, if applicable.

Persist the source diff and changed symbols for review. Structural validation and diff inspection can flag compound changes; they cannot prove causal isolation. Mark ambiguous interventions as non-attributable and exclude them from rule-level causal evidence rather than assigning false credit. Existing TTHE eligibility checks still govern whether code can participate in ordinary search.

Freeze both parent and child in a run-scoped, content-addressed artifact store. Verify hashes before replay and include relevant shared executor/harness configuration fingerprints. Missing or mutated artifacts are execution errors. Existing generated-module naming must not cause cross-run overwrites or stale imports.

Memory rollback does not revert the currently selected executable harness. If harmful behavior may already exist in the incumbent, give the next proposer a targeted repair warning. Do not automatically replace the incumbent with an old parent, which could discard unrelated improvements.

## 8. Budget and execution control

Default memory validation allocation: 10% of an explicitly configured total inference budget. Make the fraction configurable. It is a cap, not a spending target. Unused allocation may return to ordinary evolution; memory may not exceed its allocation.

Inventory all inference paths before selecting the accounting unit: solver, proposer, judge, back-translation, embedding, and any model-assisted applicability calls. Prefer token usage with conservative reservations for in-flight calls. If a provider cannot report usage, record that limitation and use documented conservative bounds. Do not present harness rollout counts as actual inference cost.

A hard budget needs enforcement at call boundaries, including calls in proposer/judge subprocesses. Use a process-safe shared ledger or enforceable worker reservations. Account for retries, cache hits, concurrent calls, and calls still running after timeout. Thread timeouts alone do not stop continued spending; use cancellable/process-isolated execution or an equivalent enforceable call guard.

Reserve enough budget for a complete parent/child comparison before scheduling it. An incomplete pair supplies no decisive evidence. Prioritize applicable unresolved lessons near a decision threshold, with deterministic aging/tie-breaking to avoid starvation. Report queued and skipped work.

Cache with artifact hash, task/test fingerprint, solver/executor configuration, and sampling identity. A cached result does not become fresh independent evidence when reused. Assign shared observations to a cost category once, without double charging.

Keep ordinary legacy `--memory-mode none` usable without new budgeting requirements. For budget-controlled comparisons, use the same explicit total budget mechanism in every arm. Add clear CLI options for total budget, memory fraction, retrieval limit, and online policy/config. A 10% setting without a finite denominator is not a budget implementation.

## 9. Persistence and integration order

Prefer SQLite as the source of truth for associations, evidence, budget ledger, and policy metadata; export JSON for inspection. Alternatively implement a tested recoverable transaction journal. Do not keep an unrepairable two-file commit protocol.

Version schemas and store policy configuration, embedding identity, run identity, and immutable artifact references. Migrate legacy state without inventing evidence; old booleans retain their historical meaning and must not be reinterpreted as newly verified outcomes. Resume must reject incompatible run/configuration combinations clearly.

Batch flow:

1. Load persisted state and committed harness.
2. On newly arrived tasks, schedule bounded validation of earlier applications before generating new candidates.
3. Persist observations and update eligible memory transitions transactionally.
4. Observe current harness branches and retrieve task/failure-relevant advice.
5. Run ordinary proposal rounds; validate cards, freeze artifacts, register applications and optional patches.
6. Run the existing public-evidence judge and commit the selected harness.
7. Advance memory age/promotions once, and checkpoint committed harness, stream position, and budget.
8. Perform hidden scoring as measurement only, outside any memory/proposer inputs.

Use typed public-only views throughout memory APIs. Check actual tool/file visibility of proposer subprocesses: rejecting forbidden field names is not sufficient if a proposer can read hidden result files. Keep measurement artifacts outside accessible adaptation inputs; prefer a separate scoring step for the experimental runner.

## 10. Experiment support

Implement reproducible configurations for three arms:

1. Ordinary TTHE (`none`).
2. TTHE with flat accumulated advice (`flat`).
3. TTHE with the structured lifecycle (`mmh`).

For flat advice, define the control precisely: append public-trace-derived intervention summaries chronologically and truncate to the same proposer memory budget; no confidence lifecycle, semantic retrieval, or promotion. Reuse proposal-card summaries to avoid an unaccounted extra reflection model. Leave flat mode's saved validation budget available to ordinary search.

Hold task stream/order, model, seed harness, execution limits, and total inference budget fixed. Start each independent arm/order with clean memory. Record differences in actual calls/tokens and search work; equal budgets need not mean equal completed rounds.

Report:

- Hidden-test accuracy, separately from public proxy outcomes.
- Total inference usage and its allocation, wall time, and cache statistics.
- Repeated failed-intervention rate using the documented fingerprint metric.
- Improvement/regression/inconclusive/error counts.
- Pending, supported, harmful, and promoted lesson counts and time to resolution.
- Later-task transfer results and memory retrieval/application rates.

Preserve current-batch transductive accuracy as such. If reporting performance before adaptation to each batch, measure and label that separately. Do not confuse later-task memory validation with an untouched final test set.

## 11. Implementation sequence and acceptance tests

Deliver in these increments:

1. Evidence types, versioned persistence, and explicit online lifecycle policy.
2. Immutable applications/artifacts, applicability, and failed-intervention retrieval.
3. Budget ledger, bounded scheduling, call enforcement, and replay caching.
4. Optimizer/proposer integration, checkpoint/resume, and flat control.
5. Offline integration fixture, experiment configurations, and documentation.

Required behavioral tests:

- Both pass, both fail, zero tests, or equal scores cannot increase/decrease confidence.
- One regression stays unresolved and remains eligible for later testing.
- Multiple decisive observations resolve according to configured thresholds; promotion needs separate later batches.
- Transport errors are neutral; valid program failures remain usable public evidence.
- Wrong interface/failure condition is inapplicable.
- Repeated task IDs, retries, cache reuse, and resume do not duplicate confidence or budget accounting.
- Parent/child mutation and mismatched test fingerprints cannot create evidence.
- Failed interventions reach the proposer; neither hidden results nor automatic memory injection reach the solver.
- Reusing an existing rule creates an application without forcing a duplicate rule patch.
- A harmful memory transition does not silently replace the current harness.
- Conflicting pending edits and rollback cannot erase newer state.
- Validation cannot exceed its configured allocation, including concurrent/subprocess calls; an incomplete pair is neutral.
- Crash/reopen preserves association, evidence, artifact identity, stream position, and spend consistently.
- Legacy standalone behavior and `memory-mode none` remain functional.

Use a credential-free synthetic stream to demonstrate: an intervention, a neutral tie, one regression with continued testing, eventual support or rejection, failure retrieval, and promotion only after enough later rounds. Test behavior, not private helper implementation details.

Run existing core and adapter tests, the new focused tests, offline demo, CLI help, and syntax/whitespace checks. Update `README.md`, `docs/MMH_IMPLEMENTATION.md`, and `PROGRESS.md` to distinguish implemented behavior from planned or unmeasured claims. No accuracy improvement claim is justified by offline fixtures.

## 12. Completion report

Return changed files, a concise architecture explanation, exact commands/results for verification, example run commands, configuration defaults, migration/resume behavior, and remaining limitations. Explicitly state whether a live model benchmark was run. Do not mark the integration complete if the budget cap is only estimated, failed interventions are never retrieved, or online ties still count as failures.
