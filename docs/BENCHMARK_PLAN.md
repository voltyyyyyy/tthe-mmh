# Benchmark plan for the Meta-Memory Harness

Derived from the vendored draft `docs/papers/mmh-draft.pdf` (MMH) and the predecessor work it
compares against, Meta-Harness ([arXiv 2603.28052](https://arxiv.org/abs/2603.28052),
[artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)). Grounded in what this
repo already runs: `text_to_sql/` (BIRD), `livecodebench/`, `ds1000/`, `swe/`, `claweval/`,
`meta_memory/`.

## TL;DR

The draft's claims are about **adaptation over a non-stationary stream with recurring contexts**, but
every benchmark it cites (Tables 1-3) is evaluated as a **static, one-pass set**. A static eval cannot
exercise Eq. 5 (posterior needs repeated validations), Eq. 6 (age gate), Eq. 11 (needs a precedent log),
or Eq. 12 (a pending patch resolves only *the next time its context recurs*) — so MMH degenerates into
Meta-Harness on the paper's own suite. Therefore the suite must be the paper's three domain families
**re-instantiated as interleaved multi-context streams with scheduled shifts**, plus negative controls
and deterministic unit-scale probes.

Minimum defensible suite for Q1-Q4: **τ³-bench** as the agentic, genuinely non-stationary domain
(already split and already run at full scale in this lab), **DS-1000 + LiveCodeBench + BIRD** as cheap
execution streams under four drift schedules, the **classification trio**
(LawBench / Symptom2Disease / USPTO-50k) for the accuracy-context Pareto, and **Terminal-Bench 2.0**
(plus SWE-bench Verified) as frozen final claims.

---

## 1. Capability inventory from the paper

| # | Capability | Paper anchor | What would falsify it |
| --- | --- | --- | --- |
| C1 | Sustained gain over a frozen harness on a shifting stream | §3.1, Eq. 2, Q1 | mean online accuracy ≤ frozen baseline, or no advantage over Meta-Harness |
| C2 | No forgetting / drift resilience | Eq. 2, Eq. 5-6, Q1 | accuracy drops after a shift and does not recover; pre-shift rules go stale without being retired |
| C3 | Volatile/stable isolation; stable rules never rewritten by noise | §3.2, Eq. 3-4, App. Round 9 | any honored `REFINE`/`SPLIT` on a stable rule; a harmful patch reaching stable |
| C4 | Evidence-gated promotion on repeated cross-subset success + recovery | Eq. 5-8, App. Round 5-7 | rules promoted from one lucky validation; promoted rules that do not help held-out data |
| C5 | Five atomic context-aware edit ops, each with its own identity | Eq. 9-10, Table 5 | ops that never fire; per-op success rates far off Table 5 (Add 32%/68%, Delete 8%/42%, Refine 28%/72%, Split 18%/65%, Merge 14%/78%) |
| C6 | Patch-level credit assignment; precedent influence gate; conflict arbitration | Eq. 11-12, App. Round 8, Q2 | `I(δ)` uncorrelated with realized outcome; conflicts resolved by coin flip; search steps wasted |
| C7 | Calibrated confidence | Eq. 5 | posterior confidence diverges from realized per-context success; threshold-sensitive collapse |
| C8 | Context-aware routing (`φ` matching) | Prompt 3, 0.7 threshold | rules fire on out-of-context tasks (false-fire) and hurt accuracy |
| C9 | Cross-domain transfer without per-benchmark overfitting | §4 Q4, Table 1-3 | a harness tuned on one domain regresses on held-out domains |
| C10 | Efficiency: context tokens and search steps | Table 1 `Context (k)`, §4 Q2 | accuracy bought with more context than ACE/MCE; more candidate evaluations than baselines for the same score |
| C11 | Compact, interpretable, non-redundant rule base | §3.3 MERGE rationale, Table 5 | rule base grows monotonically; `φ` overlap accumulates |
| C12 | Label-free adaptation (hidden tests never enter the loop) | repo protocol | gold appears in a prompt, proposal, rule, or patch record |

## 2. Structural requirements a benchmark must satisfy

These are the filters that decide membership in the suite.

- **R1 — ≥ 6 natural contexts.** Promotion needs ≥ 2 independent validation subsets; stable
  `DELETE`/`MERGE` needs 5 distinct applicable failure rounds spanning ≥ 2 subsets. Contexts must be
  real partitions, not random splits: BIRD `db_id`, DS-1000 `library` × perturbation, LCB
  `platform` × `difficulty`, SWE-bench repo prefix, τ³ domain × task family, claw-eval task family.
- **R2 — Context recurrence.** Every context must be revisited ≥ 3× inside a run, or Eq. 12 can never
  resolve a patch and Eq. 6 can never be satisfied. This is the single most important design
  constraint and the one the draft's evaluation currently violates.
- **R3 — Public signal + hidden measurement.** Public tests (LCB `public_test_cases`) or a reserved
  validation split must exist so the loop stays label-free; the reported metric comes from an untouched
  final split. Benchmarks whose reward *is* the grader (τ³, SWE-bench) have no public/hidden seam, so
  there the discipline is the Meta-Harness one instead: the loop sees search-set scores only.
- **R4 — Near-miss contexts.** Routing (`φ`, threshold 0.7) is only testable if contexts are
  confusable: BIRD's similar schemas, DS-1000's sibling pandas APIs, LCB's AtCoder/ARC overlap.
- **R5 — Cheap enough for the loop.** The Meta-Harness protocol spends ~50 full evaluations per search
  run (50-100 classification samples, 88 math problems). Docker-per-instance benchmarks cannot sit
  inside an 20-iteration × G-branch loop.
- **R6 — Three-way split.** search set / validation subsets / final test, declared up front. The draft
  does not specify this; it must be fixed before any number is reported.
- **R7 — Objective drift axis.** Prefer a time or domain axis that is not hand-picked: LCB
  `contest_date`, SWE-bench repo blocks, τ³ domain mixture, dataset rotation.

## 3. The suite

### Tier A — paper-comparable core (needed for head-to-head with Meta-Harness / ACE / MCE)

| ID | Benchmark | Contexts | Purpose |
| --- | --- | --- | --- |
| A1 | **Online text classification**: LawBench (215 classes), Symptom2Disease (22), USPTO-50k (180) | 3 datasets, rotating | C1, C2, C7, C10, C11; reproduces Table 1 and the accuracy-context Pareto |
| A1b | **OOD suite** used by Meta-Harness: Banking77, AG News, SciTail, TweetEval-Hate, FiNER, Amazon-Reviews, GoEmotions, CLINC, citation-intent | 9 held-out datasets | C9; concatenate with A1 into one 12-context stream → drift *and* transfer, same harness |
| A2 | **Retrieval-augmented math**: 200 IMO-level problems = IMO-AnswerBench 100 + IMO-ProofBench 60 + ArXivMath Dec-2025 17 + Jan-2026 23; 5 frozen models; 535k-item corpus (OpenMathReasoning, DeepMath-103K, NuminaMath-1.5, PolyMath, Omni-MATH, FineProofs-SFT, AIME 1983-2024, Putnam-AXIOM) | 4 eval families × subject (algebra/geometry/number theory) | C2 (ArXivMath months are a genuine temporal shift), C8 (geometry routing), C10 |
| A3 | **Terminal-Bench 2.0** (89 tasks × 5 trials; easy 4 / medium 55 / hard 30) | task category, tool family | C1, C9, C10 final claim; baselines Claude Code, Terminus-2, Terminus-KIRA, ForgeCode, Goose |

### Tier B — execution-grounded domains already running (this repo + the lab)

| ID | Benchmark | Contexts | Drift axis | Notes |
| --- | --- | --- | --- | --- |
| B1 | **BIRD** (`text_to_sql/`) | `db_id`: formula_1, superhero, card_games, toxicology, debit_card_specializing, student_club, california_schools, … | DB-block rotation | SQLite execution = public signal; `slices/genuine_hard50.json` already indexes `[db, idx]` |
| B2 | **LiveCodeBench** (`livecodebench/`) | `platform` (abc/arc/…) × `difficulty` | **`contest_date`** — objective, contamination-free temporal stream | public tests label-free, hidden tests measurement; `recent_first` ordering already implemented |
| B3 | **DS-1000** (`ds1000/`) | `library` (numpy/pandas/scipy/sklearn/…) × perturbation (origin/surface/semantic/domain) ≈ 28 cells | library-block rotation | best op-coverage target: heterogeneous cells force `REFINE`/`SPLIT`/`MERGE` |
| B4 | **SWE-bench Verified** (`swe/`) | repo prefix: django, sympy, matplotlib, sphinx-doc, pylint-dev, pytest-dev, astropy, pydata/xarray | repo-block rotation | Docker per instance; `slices/hard40.json` for a final claim, not the inner loop |
| B5 | **τ³-bench** (customer-service tool use + simulated user; already wired in the lab at `/home/yangfan/meta-agent-test/tau2-bench`) | **domain × task family**: airline (50), retail (114), telecom (114), banking_knowledge (97) | domain-mixture rotation | 375 tasks; binary reward; **pass^k** is the official metric; natural drift (airline ≈ saturated, telecom/banking ≈ 10-15%) |
| B6 | **claw-eval** (`claweval/`) | task family (kb_search, email_reply_draft, calendar_scheduling, meeting_notes, todo, crm, sla_audit, …) × zh/en | family rotation | optional second agentic domain; `headroom30.json` exists (loop code is external) |

### B5 in detail: τ³-bench

τ³-bench extends the tau-bench / tau²-bench line: a policy-following assistant resolves customer-service
requests through domain tools and a simulated user, scored by the official evaluator as a **binary task
reward**, reported as **pass^k = C(s,k)/C(n,k)** with `num_trials=3`
([adapter reference](https://github.com/harbor-framework/harbor/blob/main/adapters/tau3-bench/README.md),
[EvalScope entry](https://evalscope.readthedocs.io/en/latest/benchmarks/tau3_bench.html)).

Three reasons it belongs at the front of the suite:

1. **The infrastructure already exists and the split is already frozen.** The archived run
   (`C:\Users\yyc\meta-agent-test-results\all375-metamem-metaagent-20260906T1025Z`) used all 375 official
   tasks partitioned **before** the run into 225 search (30/68/68/59 by domain), 75 validation
   (10/23/23/19) and 75 final test (10/23/23/19), with two frozen baselines: a minimal loop at
   25/75 (33.3%) and a human-engineered reference at 30/75 (40.0%). Nothing needs to be built to start.
2. **It has the largest real drift signal of any candidate.** Airline is saturated (10/10 both arms in
   the final test) while telecom (3/23) and banking_knowledge (2/19) are near floor. Domain-mixture
   rotation therefore moves expected accuracy by tens of points — a far stronger Eq. 2 test than any
   single-domain benchmark.
3. **It already produced the exact negative result MMH is designed to fix.** In that run the only scored
   memory candidate was **rejected**: `metamem_002_606715cf` improved search (34/64 vs parent 33/64) but
   regressed validation (30/75 vs parent 33/75), so zero memory updates were accepted and the final
   comparison (Meta 35/75 vs fast-only 31/75, McNemar p=0.289) is **not** a memory-on/off test. The
   reflection had applied 9 generic guidelines from 53 proposed operations in one notebook. That bundled
   intervention is precisely what Eq. 9-12 forbids: one patch, one identity, one validation subset, one
   rollback.

**Testable hypothesis for τ³:** per-patch staging should accept *some* rules where the monolithic
notebook regressed — or, equally publishable, the influence gate should reject them one at a time with a
recorded reason. Either outcome is a result; the archived run can produce neither.

τ³-specific design requirements:

- **Disjoint validation subsets.** MMH promotion needs ≥ 2 independent subsets; the archived run reused a
  single 75-task pool for selection at every epoch. Re-partition validation into ≥ 2 (ideally 4) disjoint
  domain-stratified subsets, and never let a subset double as final test.
- **A rotating stream, not an epoch batch.** The archived loop used a deterministic 64-task search batch
  (`seed=42`) per epoch — a sample, not a stream. Draw batches from a shifting domain mixture so contexts
  recur (Eq. 12) while the mixture moves (Eq. 2).
- **pass^k at ≥ 3 trials as the headline.** The stable tier is a reliability mechanism, so pass^2/pass^3
  is where its value shows; a single trial per task (as in the archive) cannot see it.
- **Headroom weighting.** Mirror the existing `claweval/slices/headroom30.json` pattern: build a
  `tau3/slices/headroom*.json` over telecom/banking task families, or airline's saturation will dilute
  every measured effect.
- **Isolate execution errors.** The archived final test carried 8 classified execution-error records in
  the Meta arm and 4 in fast-only (mostly request timeouts), recorded as zeros. MMH must never attribute a
  timeout to a rule: tag execution errors on a separate channel, exclude them from patch validation, and
  report them next to every score.
- **Two tau3 setups exist — do not mix them.** The earlier dual-loop run used a 35/15 airline-only split
  (`reference` harness at 14/15, `evo_003` at 15/15); the vanilla holdout varied 11-13/15 across reruns,
  so that 15/15 was plausibly sampling variance. Use the 375-task split for anything claim-bearing.

**Note on framing.** The draft's Table 3 is Terminal-Bench 2.0, not τ³. A τ³ result is new evidence from
this lab, not a reproduction, and its numbers are not comparable to the Meta-Harness TB2 rows.

### Tier C — non-stationarity construction (nothing off-the-shelf exists)

Nothing in the benchmark landscape ships a stream that satisfies R2 + R7, so the stream generator is
itself part of the contribution. Spec:

- Draw tasks as `Q_i ~ P_i` over a context pool `C = {c_1..c_K}`, `K ≥ 6`, `|C_c| ≥ 8` tasks each.
- Shift `P_i` every `B` batches, `B ≈ 1-2` at the current `--batch-size 5`.
- Four schedules, all reused across every domain:
  1. **abrupt** `A→B→C→D` (tests recovery);
  2. **cyclic/recurring** `A→B→C→A→B→C` (makes Eq. 5/12 resolvable — the mandatory control);
  3. **gradual** linear mixture drift over the batch index (tests graded staleness);
  4. **return-with-decay** a context returns after a long absence (tests retirement of stale rules).
- Tier C2 **negative controls** (these catch broken gates, and are cheap):
  - **label-shuffled stream** — no learnable signal: MMH must promote ≈ nothing and must not degrade;
  - **random-reward judge** — the influence gate must reject (Eq. 11);
  - **harmful-patch injection** — stable mutation attempts must be refused 100% of the time (App. Round 9);
  - **stationary permuted stream** — must not regress against the frozen harness.
- Tier C3 **transfer**: adapt on context set `X`, freeze, evaluate on unseen `Y` (Q4), including one
  rule base shared across BIRD + DS-1000 + LCB.

### Tier D — deterministic probes (cheap, and the only place exact precision is knowable)

- **D1 Synthetic rule-induction stream** with ground-truth rules and a known drift schedule: extend
  `meta_memory/demo.py`'s labeled lifecycle to a benchmark-scale stream. Only here can promotion
  precision/recall, calibration error, and drift-detection latency be computed against truth.
- **D2 Conformance** — keep `tests/test_meta_memory.py`, `tests/test_lcb_mmh_adapter.py`; add an
  op-coverage test that every one of the five ops is exercised.
- **D3 Threshold sweeps** on the cheapest domain: `θ_c ∈ {0.6,0.7,0.8,0.9}`,
  `θ_age ∈ {1,2,3,5}`, `ε ∈ {0.4,0.6,0.8}`, context-match `0.7`.

## 4. Capability → benchmark → metric → pass criterion

| Capability | Primary benchmarks | Metric | Pass criterion |
| --- | --- | --- | --- |
| C1 adaptation gain | A1, B1-B3, B5, C1 schedules | final accuracy vs frozen / zero-shot / few-shot / ACE / MCE; paired per-task flips (McNemar) | beats frozen on mean online accuracy with paired significance, ≥ 3 seeds |
| C2 no forgetting | C1 schedules on B2 (date-ordered) and B5 (domain mixture), A1b, B4 | AUC of the online accuracy curve; backward transfer (BWT); worst post-shift drop; rounds to recover; **count of shift points where mean accuracy falls** (Eq. 2 violations) | BWT ≥ 0; recovery within 1-2 batches; fewer Eq. 2 violations than Meta-Harness |
| C3 tier integrity | C2 harmful injection, B5, B3, B1 | honored stable `REFINE`/`SPLIT` (must be 0); harmful-patch-to-stable rate (0); post-rollback state hash equality | zero violations |
| C4 promotion quality | D1, C2 label-shuffled, B5, B1-B3 | promotion precision = P(held-out improves \| promoted); promotions on the shuffled control; survival of promoted rules to stream end | precision > 0.5 and control promotions = 0; **≥ 1 accepted patch on τ³**, where the monolithic notebook accepted none |
| C5 op coverage | B3, B1, B5, A3 | per-op usage/success vs Table 5 | every op exercised ≥ 10×; rates within a stated band of Table 5 |
| C6 credit assignment | B3, B5, B2, C2 random-reward | influence AUC (`I(δ)` vs realized `s_t`); drop-one counterfactual attribution error; false-reject rate at `I<ε`; conflict rate + arbitration correctness | influence AUC > 0.5 significantly; false-reject rate reported and bounded |
| C7 calibration | D1, A1, B5, B1 | ECE / Brier of `c_t` vs realized per-context success; Spearman(c_t, realized) | ECE below the frozen judge's; monotone ordering |
| C8 routing | B1, B3, B2, A2 | false-fire rate on out-of-context tasks; miss rate; Δaccuracy \| fired | net positive Δaccuracy given firing |
| C9 transfer | A1b, cross-domain C3, B5, B6 | held-out-domain accuracy after adaptation; regression count | no held-out domain regresses more than noise |
| C10 efficiency | A1 (Table 1 protocol), B3, B5 | context tokens; proposer calls and wall-clock per accepted rule; candidate evals to match baseline; accuracy-context Pareto | ≤ ACE/MCE context tokens; fewer evals than OpenEvolve/TTT-Discover for equal score |
| C11 rule-base quality | D1, B5, B3 | rule count over time; φ-overlap redundancy rate; stale-rule retirement rate | rule base bounded; redundancy not monotonically increasing |
| C12 label-free | all | leakage audit of prompts / state / proposals / patch records | zero gold artifacts |

## 5. Baselines and ablations

Required comparison arms:

1. **Frozen**: zero-shot; few-shot `N ∈ {4,8,16,32,all}`; the seed harnesses `agents/bare.py`,
   `agents/react.py`.
2. **Hand-designed context engineering**: ACE, MCE (the two the draft's Table 1 is built on).
3. **Search-based reuse**: Meta-Harness; the static experience bank (`experience/`) that Figure 1 calls
   "offline search-based reuse" — the direct foil for the bi-level memory claim.
4. **τ³-specific arms already measured** (reuse verbatim as baselines): minimal frozen loop 25/75,
   human-engineered reference 30/75, fast-only evolved 31/75, Meta+notebook-MetaMem 35/75.
5. **Text optimizers**: GEPA, OpenEvolve, TTT-Discover, Best-of-N (compute-matched control).
6. **Ablations** matching Table 4, run on the C1 stream rather than a static set:
   `−stable tier`, `−credit assignment`, `−influence gating`, `−atomic ops (Add/Delete only)`,
   `−full traces (scores only)`, `−full traces (summary only)`.
7. **Oracle controls**: promote-everything, promote-nothing, and (in D1) the ground-truth rule set.

## 6. Protocol and statistics

- **Declare the three-way split** (search / validation / final test) per domain before running; the
  draft leaves this open and it is the difference between a result and a leak.
- **≥ 3 seeds** per arm; report **median and best** (the draft reports both). Model stochasticity is
  material: on a 15-task holdout one task is 6.7 pp.
- **Paired, task-level** comparisons; do not compare aggregates across different runs.
- **Budget parity**: evaluation is the bottleneck, so match candidate-evaluation counts across methods
  (Meta-Harness matched prior text optimizers at 0.1× evals; that is the bar to beat or match).
- **Pre-register the acceptance policy** (holdout-first vs best-search) and record mechanism
  activation — which op fired, whether the influence gate rejected, why a patch rolled back.
- **Decontaminate** any retrieval corpus against eval families (exact prefix + fuzzy Jaccard ≈ 0.8,
  as Meta-Harness did) before A2 numbers mean anything.
- **Extend the leakage audit**: `audit_harness.py` and `mmh_adapter._FORBIDDEN` already gate gold
  fields; run the same check for every new adapter (BIRD gold SQL, DS-1000 `reference_code`, SWE
  FAIL_TO_PASS).

## 7. Gaps and risks that benchmark choice cannot fix

1. **Tables 1-3 are not protocol-matched.** Table 2 of the draft states its per-model numbers are
   "directly copied from Meta-Harness Table 6", and Table 1's Meta-Harness row (42.3 / 76.0 / 51.5,
   56.6 avg) does not match Meta-Harness's own reported search-set figures (median 50.0 / best 56.7).
   Re-run the baselines under one protocol, or label the comparison as indicative.
2. **Eq. 2 is not testable as written.** Expected accuracy under `P_{t+1}` is not measurable per shift;
   report online AUC + BWT and *count* violations instead of asserting monotonicity.
3. **Eq. 6 omits the recovery condition** that the prose and this implementation require. Streams must
   therefore contain recurring *failures*, not just recurring contexts, or C4 cannot be measured.
4. **Eq. 12 requires recurrence.** One-pass evaluation makes the entire memory mechanism inert — hence
   R2 above; state this explicitly in the paper's setup section.
5. **Terminal-Bench 2.0 leaderboard rows are not comparable.** ForgeCode's 81.8 is a vendor blog claim
   on a different protocol; only self-run harnesses belong in the table.
6. **Cost.** TB2 (89 × 5 trials) and SWE-bench (Docker per instance) cannot live inside the search
   loop. Use them for the frozen selected harness only; run the loop on B1-B3 + B5.
7. **Table 5 statistics need instrumentation** that does not exist yet: per-op logging with the
   validation subset id, or the numbers are unreproducible.
8. **Execution errors masquerade as rule failures.** In the archived τ³ final test, 8/75 (Meta) and
   4/75 (fast-only) records were classified execution errors — mostly request timeouts — and were
   recorded as zeros. Baseline runs carried 10 each. Any gate that reads the raw reward will treat these
   as evidence against whatever rule was active, so execution errors must be tagged and excluded from
   patch validation, and reported alongside every score.
9. **τ³ is not a label-free-in-the-LCB-sense benchmark.** Its reward *is* the official grader, so there
   is no public-tests / hidden-tests seam. The integrity discipline here is the Meta-Harness one: the
   loop may see search-set rewards, the final split stays untouched, and validation subsets are reserved
   rather than being the same pool used for selection (which is what the archived run did).
10. **Harness wins on τ³ are partly infrastructure.** The selected Meta harness's main mechanism was
    context compaction for ~80K-token banking KB payloads plus bounded retries — repairing timeouts, not
    policy skill. Report infra rules and policy rules separately, or the capability claim will be
    over-read.

## 8. Priority order

| Step | Work | Cost | Unlocks |
| --- | --- | --- | --- |
| 1 | D1/D2/D3: synthetic ground-truth stream, op coverage, threshold sweeps | hours, no credentials | C4, C5, C7 exactly measurable |
| 2 | **τ³**: re-partition validation into ≥ 2 disjoint subsets, wire MMH where the actor-only MetaMem plugin sits, rotate the domain mixture, re-run the 4 archived baselines | days; adapter work, not new infrastructure | C1-C4, C6, C10, C11, and the accepted-patch question the archive could not answer |
| 3 | Stream generator (Tier C1/C2) over B3 + B1 + B2 | days; execution is cheap | C2, C3, C6, C8, C12 |
| 4 | A1 + A1b classification stream | LLM calls, medium | C1, C9, C10, Table 1 parity |
| 5 | A2 math retrieval stream | medium-high | C8, C10, Table 2 parity |
| 6 | B6 claw-eval | agentic, high | C9 second agentic domain |
| 7 | A3 TB2 + B4 SWE-bench Verified, final claims | very high | Table 3 parity |
