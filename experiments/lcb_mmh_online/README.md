# LiveCodeBench online-MMH experiment

This is a **separate experiment** implementing the `docs/MMH_TTHE_HANDOFF.md`
specification without changing the legacy `--memory-mode none` path or the
standalone paper/demo semantics.

Primary objective: avoid repeating failed executable-harness changes.  TTHE
memory is supplied only to the proposer that edits harness code; it is never
automatically injected into frozen solver prompts.

## Why a separate experiment?

The existing `livecodebench/mmh_adapter.py` integration resolves a pending patch
on the first applicable observation and treats ties as failures.  The handoff
requires a different online policy:

- equal parent/candidate results are inconclusive;
- one regression is recorded and tested further, not discarded;
- validation consumes a bounded share of a finite inference budget;
- applications and artifacts are immutable;
- failed interventions are explicitly returned to the proposer as warnings.

This package keeps those semantics separate and testable.  The legacy engine is
reused for rule/patch storage, but it is never silently reinterpreted.

## Architecture

| Module | Responsibility |
|---|---|
| `types.py` | Evidence contract, online policy defaults, application/proposal-card/memory-package types, stable task/test fingerprints. |
| `store.py` | Versioned SQLite source of truth sharing the existing rule/patch tables and adding applications, evidence, recoveries, and stream checkpoints. |
| `memory.py` | Online lifecycle: neutral ties, multi-observation support/harmful resolution, recovery replay, promotion, bounded retrieval, failed-intervention warnings. |
| `artifacts.py` | Run-scoped SHA-256 content-addressed artifact store; missing/mutated artifacts become execution errors. |
| `budget.py` | Process-safe SQLite budget ledger with a hard total cap and memory-allocation cap. |
| `cache.py` | Artifact/config/task-aware replay cache; reused results are marked non-fresh. |
| `scheduler.py` | Deterministic near-threshold scheduling that reserves a complete parent/child pair before running it. |
| `cards.py` | Extended proposal-card validation; applying an existing rule no longer forces a duplicate ADD/REFINE patch. |
| `adapter.py` | Public-only LiveCodeBench adapter: registers frozen candidates and validates them on later public tasks. |
| `flat.py` | Flat chronological-summary control with no lifecycle, semantic retrieval, or promotion. |
| `fixture.py` | Credential-free synthetic stream demonstrating the required lifecycle. |
| `runner.py` | CLI for reproducible `none`, `flat`, and `mmh` arm configurations. |

## Quick start

Run the deterministic offline fixture:

```bash
PYTHONPATH=. python -m experiments.lcb_mmh_online --arm mmh \
  --total-budget 1000000 --memory-budget-fraction 0.10 \
  --run-dir runs/lcb_online_mmh
```

Run the behavior tests:

```bash
PYTHONPATH=. python -m experiments.lcb_mmh_online.tests.test_online_mmh
```

Run the three reproducible arm configurations (offline fixtures; no paid
benchmark is launched):

```bash
PYTHONPATH=. python -m experiments.lcb_mmh_online --config experiments/lcb_mmh_online/configs/none.json
PYTHONPATH=. python -m experiments.lcb_mmh_online --config experiments/lcb_mmh_online/configs/flat.json
PYTHONPATH=. python -m experiments.lcb_mmh_online --config experiments/lcb_mmh_online/configs/mmh.json
```

A live-loop entry point is provided as a shim around the existing optimizer:

```bash
PYTHONPATH=. python -m experiments.lcb_mmh_online.live_runner \
  --arm mmh --total-budget 1000000 --memory-budget-fraction 0.10 \
  --pilot livecodebench/logs/pilot50.json --batch-size 5 --group 2 --max-rounds 3
```

Use `--arm none` to invoke the ordinary loop unchanged, or `--arm flat` for the
chronological-summary control.  The shim reuses `livecodebench.lcb_optimize`
without changing its default path.  It still requires the normal solver endpoint,
config, and public credentials; this repository does **not** launch a paid live
benchmark as part of the handoff.

## Defaults

Online policy (configurable under `online_policy`):

- Beta(1, 1) posterior; `confidence = (1 + successes) / (2 + successes + failures)`.
- Support: at least 3 decisive observations, 3 distinct later tasks, 2 later
  batches, confidence >= 0.75.
- Harm: same count/task/batch gates, confidence <= 0.40; one regression never
  resolves as harmful.
- Promotion: confidence >= 0.80, successes in >= 3 distinct later batches,
  elapsed age >= 3 batches, recovery condition satisfied when required.
- Retrieval: at most 5 positive/tentative lessons combined and 3 failed
  interventions, within a configurable prompt character budget.

Budget defaults:

- `total_budget` is required for `flat` and `mmh` arms.
- `memory_budget_fraction = 0.10` of the finite total.
- The memory allocation is a hard cap, not a spending target; unused allocation
  can return to ordinary evolution.

## Evidence contract

Each application/task/configuration comparison produces exactly one of:

| Outcome | Effect |
|---|---|
| `improvement` | one success |
| `regression` | one failure |
| `inconclusive` | no update |
| `inapplicable` | no update |
| `execution_error` | no update |

Only decisions are counted.  Neutral/error observations remain in the audit log.
A uniqueness key prevents repeated task IDs, retries, cache reuse, and resumed
runs from inflating confidence or spend.

## Persistence and resume

SQLite is the source of truth for applications, evidence, rules, patches,
stream/batch checkpoints, and the budget ledger.  JSON is used only for
inspection exports such as `offline_report.json`.  Reopening a store rejects
incompatible schema/configuration combinations instead of inventing evidence.

Artifacts are addressed by SHA-256.  Missing or mutated artifacts are execution
errors; they never create causal evidence.  `--fresh` candidate cleanup cannot
invalidate persisted validation because persisted applications refer to artifact
hashes, not generated module filenames.

## Limitations / not claimed

- No live model benchmark was run, and no accuracy improvement is claimed.
- The shipped embedding is the dependency-free deterministic hash embedding;
  production semantic retrieval can use `meta_memory.EmbeddingEngine`.
- The live LCB loop wiring is an integration point, not a completed paid study.
  The offline fixture and adapter tests exercise behavior, not hidden scores.
- Structural card validation can flag compound changes but cannot prove causal
  isolation; ambiguous cards are marked non-attributable.
