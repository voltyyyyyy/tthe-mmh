# tau3 × MMH — final experiment design

**Status:** specification. Nothing here is running yet; the previous model servers were stopped.

Framing: this is the experiment the MMH paper's own theory requires but never runs. The draft's
premise is a non-stationary stream (`P_i ≠ P_{i+1}`) and its dual tier exists to survive concept
drift — but every reported experiment (LawBench, S2D, USPTO-50k, IMO math, TerminalBench) is
stationary. The paper's largest ablation drop is **−Stable Tier: 52.3% → 49.2%**, yet the stable tier
is never exercised under the condition it was built for. This design does that.

---

## 1. The claim under test

> Under a genuinely non-stationary task stream, does MMH's two-tier memory with evidence-gated
> promotion preserve performance across regime shifts — and do the tiers actually diverge?

Secondary (the volatile-traits question): do novelty, surprise, uniqueness, sparsity, and
stable-overlap separate volatile from stable memory, and do they predict promotion?

If the tiers do **not** diverge under drift, the two-tier premise is decorative — a publishable
negative result.

---

## 2. Rule representation — behavioral guidelines

**Decided.** A rule is the paper's `r = ⟨φ, ψ, ω, c, τ⟩`: a natural-language applicability condition
`φ` and a harness intervention `ψ`.

Justification is forced by the paper's own Eq. 11: influence is estimated by embedding similarity over
the rule's context, `z = f(φ)`. That only works if `φ` is comparable natural language. A code diff has
no applicability context, which is precisely why the LiveCodeBench `--memory-mode mmh` interpretation
(patches as executable harness edits) makes Eq. 11 degenerate. The paper's worked examples are
behavioral (`contrastive_verify`, `truncate_long_symptoms`, `stopword_filter`).

Consequence: the editable surface is the **guideline set injected into the agent prompt**, not harness
code. The harness stays frozen in both arms.

---

## 3. Benchmark and split (frozen, non-negotiable)

Reuse exactly, so numbers are comparable to the recorded 31/75:

| Item | Value |
|---|---|
| Benchmark | `reproduction/qwen_tau3/all375/benchmark.yaml` |
| SHA-256 | `6ef47733039c009d702bd183b0b2c7c7b085969eaa887a4b72386b7937dc7a36` |
| Split manifest | `all375/split_manifest.json`, seed 42 |
| SHA-256 | `c1a5915765017bc48abb74eb1b207c74e411617e1c420e441b35358c99d1210c` |
| Domains | airline, retail, telecom, banking_knowledge |
| Partition | search 225 / validation 75 / test 75 |
| Base harness | all-domain copy `db0cdc8b…` (**not** historical `78b4aab9…`) |
| Model | `Qwen/Qwen3.8-27B` — actor, user simulator, and any judge |
| tau2 commit | `a2c024725189473d2d7cea3a5cfdbcc67478e41f` |
| Retrieval | bm25 |

Reference numbers already on record: frozen minimal 19/75 test (25.3%), human reference 31/75
(41.3%), fast Meta-Agent champion 31/75 test / 32/75 validation.

Reported baseline caveat: the frozen minimal harness crashed pre-first-call on all 42 telecom+banking
test tasks (numeric-ID assumption). Use the all-domain copy as the honest baseline; never present the
crashing baseline's number as a capability gap.

---

## 4. Non-stationarity: the regime schedule

`hard60`-style shuffling would destroy the drift signal, so **the stream is ordered by regime**. The
frozen split fixes *membership*, not order — order is ours. Two phases:

**Phase 1 — maturation** (regimes long enough for the lifecycle to actually run)

| Regime | Domain | Rounds | Turns test |
|---|---|---|---|
| A1 | airline | 5 | promotion reachable; stable retirement reachable |
| A2 | retail | 5 | same |

**Phase 2 — drift pressure** (regimes deliberately shortened)

| Regime | Domain | Rounds | Turns test |
|---|---|---|---|
| B1 | telecom | 3 | promotion reachable, retirement NOT |
| B2 | banking_knowledge | 3 | promotion reachable, retirement NOT |
| B3 | airline (revisit) | 4 | does the promoted stable set still hold on the origin domain? |

Total **20 rounds**, 8 tasks/round = **160 search tasks** (of 225).

### Why regime length is a hard design parameter

Verified against `MMHConfig` defaults (`promotion_age=3`, `promotion_subsets=2`,
`stable_failure_rounds=5`):

| Regime length | Promotion possible | Stable retirement possible |
|---:|---|---|
| 1–2 rounds | **no** | no |
| 3–4 rounds | yes | **no** |
| ≥5 rounds | yes | yes |

A 1-round-per-domain schedule would make the entire MMH lifecycle a no-op. This is why phases exist.

---

## 5. Validation protocol — round-scoped subsets

The paper's protocol (Appendix B) validates on a **fresh held-out sample each round**; the
implementation requires `subset_id` to match `(\d+):` (`mmh_adapter.py:331`) and forbids validating a
rule on its originating batch (`:338`).

**5 disjoint subsets × 15 tasks = the frozen 75 validation tasks.**

- Reaches `independent_subsets = 5` (gate needs ≥2).
- Each task validated **at most once** — no repeated-query contamination, and cost equals reuse.
- Covers the stable-edit gate (5 failure rounds across ≥2 subsets) during phase 1.

Every validation task is excluded from the proposal pool. The 75 test tasks are touched **once**, at
the end, and never enter the loop.

---

## 6. Arms

| Arm | Memory | Purpose |
|---|---|---|
| **Flat** | fixed guideline set, no tiers, no promotion | the thing MMH must beat |
| **MMH** | volatile→stable with the evidence gate | treatment |
| **Frozen** | all-domain harness, no guidelines | floor |

All arms share: base harness, model, endpoint, concurrency, timeouts, task order, and the frozen split.
One vLLM process per arm behind that arm's capture proxy, so arms do not contend for KV cache and each
arm's raw traffic is separable.

Optional, only if budget allows: **MMH-no-gate** (promote everything immediately) to isolate the gate
itself from tiering.

---

## 7. Measurements

**Per round / per arm** — success rate, accuracy where labels exist, cost (captured tokens), latency,
memory size (rules in each tier), promotions, rollbacks.

**Per rule at staging** — the five traits, computed against the state as of staging so a rule's own
promotion cannot retroactively redefine its novelty: novelty `1 − max cos` to stable, overlap
`max cos` to stable, uniqueness (min pairwise distance within volatile), sparsity (local density),
surprise. Lifespan is a **gate variable, not a headline** — reporting "volatile is younger than
stable" restates `engine.py:482-491`.

**Change points** — detected with CIs at regime boundaries, not eyeballed. The expected signature is
crash/recover/flip; a flip is a stable rule invalidated by a shift.

**Embedding provider (mandatory):** `sentence-transformers` + Qwen3-Embedding-0.6B, 1024-d. MMH's
default hash embedding scores `cos(paraphrase) = 0.000` on a verified pair versus `0.778` for
Qwen3-Embedding, so it measures token overlap, not meaning. Frozen at capture time.

---

## 8. Budget

At the measured **556 s/task** (telecom), concurrency 8:

| Scope | Rollouts | Wall clock (conc 8) |
|---|---:|---:|
| 160 search + 75 validation + 75 test, **one arm** | 310 | ~6 h |
| MMH + flat arms | 620 | ~12 h |
| add frozen arm | 930 | ~18 h |

Per-round: 8 tasks ≈ 9 min serial. Regime length ≥3 is therefore also a *cost* decision — longer
regimes buy lifecycle reachability and cost wall clock.

Honest limitation: 8 tasks/round on a 4-domain mixture means a single round is a noisy estimate of
domain performance (~1.5 tasks per domain). Report paired task-level outcomes and exact McNemar;
never compare bare percentages at this granularity.

---

## 9. Required observability

Frozen manifest before each launch (commit, all hashes above, endpoint, concurrency, timeouts, exact
command); capture-proxy JSONL per arm; solver cache; per-task records with `wall_time_s`,
`num_turns`, and **populated** token fields (they were `null` in the archived run, so cost was
unmeasurable); MMH SQLite snapshot per round (rules, patches, evidence, precedents, with embeddings);
server and GPU telemetry; exit codes. Export with remote checksum → copy → local verify → then tear
down.

---

## 10. What would falsify the framework

Stated before running:

1. **Tiers do not diverge** — volatile and stable trait distributions overlap once lifespan is
   excluded. The two-tier design is then decorative.
2. **No promotion benefit under drift** — MMH ≤ flat arm across phase-2 regime boundaries.
3. **Stable rules do not degrade at shifts** — if nothing crashes at B1/B2, there is no concept drift
   to manage, and the `τ` lifespan mechanism is unjustified.
4. **Promotion is unpredictable from traits** — AUC < 0.65 with CI including 0.5, after controlling
   for the gate's own variables.

Negative results on (1) and (3) are the most interesting outcomes available here.
