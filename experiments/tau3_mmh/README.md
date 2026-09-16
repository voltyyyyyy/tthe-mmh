# tau3 × MMH experiment harness

MMH's two-tier causal-rule memory applied to **behavioral guidelines** on the tau3
customer-service benchmark, under a **non-stationary** (regime-shifting) task stream.

## Why this exists

The MMH paper's premise is a non-stationary stream — `P_i ≠ P_{i+1}` — and its dual tier
exists to survive concept drift. But every experiment the draft reports (LawBench, S2D,
USPTO-50k, IMO math, TerminalBench) is **stationary**. Its largest ablation drop is
−Stable Tier (52.3% → 49.2%), yet that tier is never exercised under the condition it was
built for. In a stationary stream the volatile and stable tiers are indistinguishable in
expectation, so the temporal-lifespan `τ` mechanism is untestable.

This harness runs that missing experiment.

## Rules are behavioral guidelines, not code

A rule is the paper's `r = ⟨φ, ψ, ω, c, τ⟩` interpreted as natural language:

- `φ` — applicability condition ("when the customer asks to cancel a booking")
- `ψ` — the instruction ("look up the booking and quote the exact policy before acting")

This is forced by the paper's own Eq. 11: influence is a similarity-weighted success rate
over the embedding of a patch's **applicability context**, `z = f(φ)`. That is only
coherent for comparable natural language. A code diff has no applicability context, which
is why the LiveCodeBench `--memory-mode mmh` code-edit interpretation makes Eq. 11
degenerate.

## Layout

| Module | Responsibility |
|---|---|
| `guidelines.py` | `Guideline`, `TaskObservation`, `GuidelineMemory` (two-tier, round-scoped evidence, promotion), gate reporting |
| `proposer.py` | `OfflineProposer` (deterministic, credential-free) and `LLMProposer` (validated through `meta_memory.adapters.parse_patch_response`) |
| `schedule.py` | `RegimeSpec`, `Schedule`, validation subsets, and `assess_schedule` viability checks |
| `runner.py` | the round-stepped loop, per-round metrics, per-guideline ledger, change-point detection |
| `demo.py` | runnable end-to-end drift demonstration |
| `traits.py` | volatile traits frozen at staging, plus AUC / Cliff's δ / Spearman helpers |
| `sweep.py` | seeded multi-run sweep, tiered vs flat arms, aggregate reporting |
| `integrity.py` | guards that separate infrastructure faults from model failures |
| `logging_utils.py` | per-run structured artifacts, written incrementally |
| `tests/` | lifecycle + adversarial integrity tests, no model or network required |

## Quick start

```bash
python -m experiments.tau3_mmh.tests.test_guidelines   # 11 lifecycle tests
python -m experiments.tau3_mmh.demo                    # drift demo + artifacts
```

## The constraint that silently breaks everything

`MMHConfig` defaults require `successful_lifespan >= promotion_age (3)` **and**
`elapsed_age >= promotion_age (3)` **and** `independent_subsets >= promotion_subsets (2)`.

Two consequences that are easy to get wrong:

1. **A regime shorter than 3 rounds can never promote anything.** A drift schedule that
   switches domain every round turns the entire two-tier mechanism into a no-op. The
   schedule is checked for this up front (`assess_schedule`), and `ExperimentRunner`
   refuses to run a non-viable schedule unless explicitly told otherwise.
2. **Effective promotion latency is `promotion_age + 1` rounds**, not `promotion_age`. A
   rule staged in round N first ages at N+1, so both freshness conditions cannot be met
   before N+4. Budget round counts against this.

## Design invariants

- **A pending guideline cannot steer the round that judges it.** Staging happens after
  evaluation, so the evidence resolving a patch is not produced by the patch's own effect.
- **`subset_id` is the validation data group, not the round.** Encoding the round into it
  (e.g. `"3:val-a"`) would make every round a fresh "independent subset" and dissolve the
  `>= promotion_subsets` gate. Reusing a subset label across rounds is legal — it advances
  `successful_lifespan` without advancing `independent_subsets`.
- **Errored tasks are not failures.** Timeouts and crashes (`error` set) are excluded from
  scoring. The archived tau3 run folded 12 terminal `ReadTimeout`s into the score and
  inflated variance; that must not recur.
- **Mixed validation rounds are indecisive.** A round that neither fully passes nor fully
  fails leaves the patch pending rather than resolving it on noise.
- **Validated rules keep collecting evidence.** A rule cannot promote if evidence stops the
  moment it validates, because `successful_lifespan` counts distinct successful rounds.

## Integrity: infrastructure faults must never look like model failures

The invariant:

> **A scored zero means the model or harness genuinely failed the task.**
> **An infrastructure fault is fatal and loudly reported.**
> No third category exists.

Without guards the dangerous failure mode is silent — a truncated candidate list, a stalled
model, an unparsed proposal, or a saturated task set all present as ordinary low scores,
and analysis then reports a *mechanism* result that is really a *plumbing* result.

| Guard | Kind | Protects against |
|---|---|---|
| `error_rate_too_high` | **fatal** | timeouts/crashes reported as a model failure rate |
| `candidate_budget_exceeded` | defect | truncation stranding patches as PENDING, misread as "failed to promote" |
| `no_guidelines_injected` | defect | mechanism never switched on, so a null result means nothing |
| `degenerate_score_series` | defect | saturation, where "both arms at 1.0" is not equivalence |
| `proposals_all_rejected_at_stage` | defect | proposals vanishing without a record |
| `traits_missing_for_staged_rules` | defect | trait statistics silently omitting rules |
| `no_validation_decisions` | defect | the gate never resolving anything |
| `validation_call_failed` | warn | a validation crash recorded as evidence about a rule |
| unfinished run | **fatal** | an unverified run defaulting to "healthy" |

- **fatal** → raises `InfrastructureFault`; the run stops rather than emitting a plausible
  number.
- **defect** → the run completes but is marked `analysable: false`; the sweep **excludes it
  from aggregates** and prints why.
- **warn** → recorded and carried into the report; the result stays usable.

Every run writes `integrity.json` beside its artifacts, and `ExperimentRunner.finish()`
must be called (also automatically by `run()`). Driving rounds manually without calling
`finish()` is caught by `require_finished()` — a missing report is never read as "clean".

## Tests

```bash
python -m experiments.tau3_mmh.tests.test_guidelines   # 11 lifecycle tests
python -m experiments.tau3_mmh.tests.test_integrity    # 13 adversarial integrity tests
python -m experiments.tau3_mmh.demo                    # drift demo + artifacts
python -m experiments.tau3_mmh.sweep --seeds 0 1 2 3   # seeded sweep with full logging
```

The integrity suite is deliberately adversarial: each test injects a realistic benchmarking
fault and asserts it surfaces as an integrity issue rather than as a low score. Runs are
deterministic given a seed (asserted: 14 scientific fields identical across repeats; only
`wall_time_s` may vary).

## Not yet built

The tau3 binding: functions that turn a tau3 task run into `TaskObservation`, and the
harness hook that injects active guidelines into the agent prompt. Until those exist the
harness runs only against injected callbacks (which is what the tests and demo do).
