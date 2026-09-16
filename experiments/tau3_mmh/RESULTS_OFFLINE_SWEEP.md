# Offline sweep results — 20 seeds × 2 arms

Command:

```bash
python -m experiments.tau3_mmh.sweep --seeds 0..19 --out experiments/tau3_mmh/_runs/full
```

**These are synthetic-world results.** The world is seeded and its ground truth is planted,
so nothing here is evidence about real agent behaviour. What it establishes is what the
*design* can detect and whether the analysis path works at volume — the prerequisite for
spending GPU hours on tau3.

## What ran

- 40 runs (20 seeds × 2 arms: `tiered` vs `flat`), 20 rounds each = 800 rounds
- 1,510 staged rule observations with trait vectors frozen at staging
- 361 artifact files (≈2.7 MB): per run `manifest.json`, `command.txt`, `schedule.json`,
  `rounds.json`, `lifetimes.json`, `patches.json`, `promotion_report.json`, `traits.csv`,
  `exit.json`
- World: 4 regimes (airline → retail → telecom → banking), planted shift at round 11,
  valid candidates vs planted-spurious candidates, agent prompt capped at 4 guidelines

## Result 1 — the gate earns its keep (planted, so mechanism-only)

| arm | promoted | pre-shift | post-shift | stable at end |
|---|---:|---:|---:|---:|
| **tiered** | 4.95 | **0.705** | 0.056 | 4.95 |
| **flat** (no gate) | 9.15 | **0.483** | 0.055 | 9.15 |

Promoting everything indiscriminately builds a **larger** stable tier (9.15 vs 4.95 rules)
that performs **worse** — 22 points of pre-shift success lost. The mechanism is the prompt
cap: spurious rules are load-bearing only in the sense that they displace useful ones.

This is a property of the constructed world, not a discovery about MMH. It confirms the
harness can register the difference the design cares about.

## Result 2 — change-point detection depends on evidence quality

The planted shift was detected in **20/20 tiered runs but only 5/20 flat runs.** The flat
arm's pre-shift rate is already depressed (0.483), so the post-shift level (0.055) is not a
sharp-enough departure to register.

Methodological consequence for the real experiment: **a degraded run is harder to diagnose
than a healthy one**, and change-point detection on a contaminated baseline will silently
under-report drift. Detection should be evaluated on the arm whose baseline is intact, and
a missed change point is not by itself evidence that no shift occurred.

## Result 3 — trait signal is weak, and two traits are dead

Trait vectors were frozen at staging against the pre-staging tier snapshot, so a rule's own
promotion cannot retroactively define its novelty.

**Label = promoted by the gate** (n=1510):

| trait | AUC | Cliff's δ |
|---|---:|---:|
| novelty | 0.461 | −0.079 |
| overlap | 0.539 | +0.079 |
| uniqueness | 0.574 | +0.148 |
| sparsity | 0.584 | +0.168 |
| **surprise** | **0.500** | **0.000** |
| **lifespan** | **0.500** | **0.000** |

Two of the six carry **exactly zero** information as operationalised here:

- **`lifespan` is a gate variable, not a finding.** The promotion gate requires
  `successful_lifespan >= promotion_age`, so volatile and stable rules are separated by
  construction. AUC 0.500 is the correct answer, and reporting "volatile is younger" would
  restate `engine.py:482-491`.
- **`surprise` is dead as defined.** The posterior-shift form is computed from the
  validation outcome, so it *encodes* what it would predict — the circularity identified
  during design. `traits.ex_ante_surprise` implements the non-circular version (precedent
  base rate before any outcome); it is not yet wired into the sweep, so the 0.500 above is
  the definition that must not be reported as a predictor.

**Label = planted validity** (ground truth, n=1510):

| trait | AUC | Cliff's δ |
|---|---:|---:|
| novelty | 0.445 | −0.110 |
| overlap | 0.555 | +0.110 |
| uniqueness | 0.453 | −0.095 |
| sparsity | 0.458 | −0.083 |

Traits are *anti*-predictive of true validity, and the strongest single signal is
**overlap with stable memory** (0.555). This is largely by construction: valid candidates
here resemble existing stable rules, spurious ones do not. It is a warning, not a result —
see below.

## The most important caveat

`novelty` is distance from the stable tier. In any world where correctness correlates with
resemblance to already-known rules — which is plausible for a *refinement* proposal and
false for genuinely novel coverage — **novelty is anti-predictive by construction.**
The sweep's valid/spurious split was built that way, so its sign is a property of the world.

Therefore: the traits must be evaluated against **real** proposal distributions before any
claim is made. The offline result licenses only the weaker statement that the trait
pipeline produces interpretable, non-degenerate numbers, and that two of the six as
defined are uninformative.

## What this licenses

1. The analysis path works end to end at volume and produces the same artifacts a real tau3
   run will.
2. The gate-vs-no-gate difference is registerable, so the primary comparison is measurable.
3. Detection sensitivity depends on baseline integrity — a real constraint on interpreting
   a missed shift.
4. `surprise` and `lifespan` must be reported as controls, not predictors.

## What it does not license

Any claim about real agent behaviour, real proposal distributions, or the true
informativeness of the traits. Those require the tau3 binding, which does not exist yet.
