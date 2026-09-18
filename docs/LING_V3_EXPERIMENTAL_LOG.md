# Ling v3 Experimental Log — tau3 all375 × MMH

**Run ID:** `full375-20260917-v3`  
**Model:** `inclusionAI/Ling-3.0-tiny`  
**Benchmark:** tau3 all375  
**Arms:** `baseline`, `flat`, `mmh`  
**Status:** complete  
**Dates:** 2026-09-17 08:20 → 2026-09-18 14:39 (Asia/Shanghai)  
**Runtime:** ~30 h 16 min  
**Infrastructure errors:** 0  
**Stable promotions:** 0

---

## 1. Bottom line

The Ling v3 run completed all planned task evaluations with no infrastructure errors.

However, **zero rules reached the stable tier**. The cause was candidate starvation in the full-study validation scheduler, compounded by two implementation/paper mismatches:

1. new rules started with confidence `0.5` instead of the paper-style `c0` prior;
2. the promotion gate was stricter than the paper’s Eq. 6.

Because no stable rule ever existed, this run **cannot test the stable-vs-volatile hypothesis**. The task-pass-rate results are exploratory only.

---

## 2. Setup

| Item | Value |
|---|---|
| Model server | Ling-3.0-tiny vLLM, port `8001` |
| GPU | GPU 1, `CUDA_VISIBLE_DEVICES=1` |
| Context limit | 131,072 tokens |
| Capture proxy | port `8103` |
| Study repo | `repo-ctxbudget-v3` |
| Run root | `/home/yangfan/mmh-exp/full375-20260917-v3` |
| Local copy | `C:\Users\yyc\TTHE\artifacts\ling-v3-20260917` |

The run used the frozen `all375-paired-v1` protocol:

- 225 search tasks,
- 75 validation tasks,
- 75 final-test tasks,
- evaluated separately under three arms.

---

## 3. Arms

| Arm | Memory policy | Promotion gate |
|---|---|---|
| `baseline` | No learned rules; frozen prompt only | None |
| `flat` | Single validated volatile pool | No promotion gate by design |
| `mmh` | Volatile + stable two-tier memory | Repeated validation gate |

Prompt budget for learned arms: up to 8 rules.

---

## 4. Preflight

Four search tasks were run before the full study, one per domain.

| Task | Result |
|---|---|
| airline_22 | FAIL |
| retail_43 | FAIL |
| telecom service_issue | FAIL |
| banking_knowledge_task_040 | FAIL |

Preflight completed cleanly. There were no infrastructure errors or context-limit retries.

---

## 5. Run timeline

| Time | Event |
|---|---|
| 2026-09-17 08:19 | v3 repo/root launched |
| 2026-09-17 08:20:26 | Ling preflight started |
| 2026-09-17 08:23:30 | Ling preflight complete |
| 2026-09-17 08:23:38 | Ling full study started |
| 2026-09-18 09:07 | Search phase complete; frozen validation/test started |
| 2026-09-18 14:39:46 | Full study complete |

No supervisor restart was needed.

---

## 6. Final task results

| Arm | Search | Validation | Final test | Promoted |
|---|---:|---:|---:|---:|
| baseline | 23/225 | 4/75 | 6/75 | 0 |
| mmh | 22/225 | 6/75 | 9/75 | 0 |
| flat | 25/225 | 5/75 | 9/75 | 0 |

The three arms were very close.

### Paired comparisons

**Search**

| Comparison | Difference | 95% CI | p |
|---|---:|---:|---:|
| MMH vs Flat | -0.013 | [-0.049, 0.018] | 0.607 |
| MMH vs Baseline | -0.004 | [-0.036, 0.027] | 1.000 |

**Final test**

| Comparison | Difference | 95% CI | p |
|---|---:|---:|---:|
| MMH vs Flat | 0.000 | [-0.053, 0.053] | 1.000 |
| MMH vs Baseline | +0.040 | [0.000, 0.093] | 0.250 |

No comparison reached statistical significance.

---

## 7. Mechanism behavior

### Staging

| Arm | Non-empty proposal files | Rules staged | Rules promoted |
|---|---:|---:|---:|
| MMH | 16 | 16 | 0 |
| Flat | 19 | 9 | 0 |
| Baseline | 0 | 0 | 0 |

### Validation decisions

| Arm | Success | Failure | Indecisive |
|---|---:|---:|---:|
| MMH | 5 | 1 | 37 |
| Flat | 6 | 4 | 35 |

### Final tier counts

| Arm | Stable | Volatile | Active | Pending |
|---|---:|---:|---:|---:|
| MMH | 0 | 15 | 4 | 11 |
| Flat | 0 | 5 | 2 | 3 |

MMH staged real rules and validated them, but no rule satisfied the promotion gate before search ended.

---

## 8. Root cause: candidate starvation

`full_study.adapt()` selected only two candidates per round:

```python
for candidate in candidates[:2]:
```

It sorted them by:

```python
provenance.get("last_measured", 0)
```

New rules had no `last_measured`, so they defaulted to `0` and always sorted first. Each round, the newest rule therefore took one of the two validation slots. Only one slot remained for all older rules.

### Evidence from `memory.sqlite`

| Rule | Created | Last measured | Outcome |
|---|---:|---:|---|
| g-17 | 17 | 18 | validated next round |
| g-18 | 18 | 19 | validated next round |
| g-19 | 19 | 20 | validated next round |
| g-20 | 20 | 21 | validated next round |
| g-21 | 21 | 22 | validated next round |
| g-22 | 22 | 23 | validated next round |
| g-23 | 23 | 24 | validated next round |
| g-24 | 24 | — | never validated |

Older rules were starved:

| Rule | Created | Last measured | Successes | Lifespan | Independent subsets |
|---|---:|---:|---:|---:|---:|
| g-9 | 9 | 22 | 2 | 2 | 1 |
| g-5 | 5 | 21 | 1 | 1 | 1 |
| g-3 | 3 | 19 | 1 | 1 | 1 |

`g-9-llm` was the closest candidate, but needed:

```text
successful_lifespan >= 3
independent_subsets >= 2
```

The search phase ended at round 24, so it never received enough validation attempts.

### Why this is not a model failure

The rules were not rejected. They were mostly never tested enough to resolve. Flat rules `g-3-llm` and `g-5-llm` eventually reached:

```text
confidence = 0.8
successful_lifespan = 3
independent_subsets = 2–3
```

with no gate blockers. They remained volatile only because Flat has no promotion gate by design.

---

## 9. Additional implementation/paper mismatches

### Confidence prior

Paper Eq. 5:

```text
c = (alpha0 + S) / (alpha0 + beta0 + S + F)
alpha0 = n0 * c0
beta0 = n0 * (1 - c0)
```

Code used `alpha = 1`, `beta = 1` and ignored `c0`. New rules therefore started at confidence `0.5`, even when the judge supplied `0.8` or `0.9`.

This made promotion harder.

### Promotion gate

Paper Eq. 6:

```text
promote if confidence >= theta_c and age >= theta_age
```

Code additionally required:

- `successful_lifespan >= 3`
- `independent_subsets >= 2`
- `original_failure_recovered`, when required

This made promotion even harder.

### Other gaps

- `omega` was not updated with recent validation feedback.
- `tau` was represented indirectly by `elapsed_age` / `successful_lifespan`, not as a single field.
- The scheduler did not match the intended lifecycle of resolving all pending patches and then probing maturing rules.

---

## 10. Conclusions

- The run is valid as an infrastructure and task-coverage result.
- It is **not** a valid test of stable-vs-volatile memory.
- Zero stable promotions were caused primarily by candidate starvation, amplified by the confidence-prior and promotion-gate mismatches.
- No MMH/Flat/Baseline task-performance differences were statistically significant.

---

## 11. Artifacts

Remote root:

```text
/home/yangfan/mmh-exp/full375-20260917-v3
```

Local copy:

```text
C:\Users\yyc\TTHE\artifacts\ling-v3-20260917
```

Key local files:

```text
unpacked\ling\REPORT.md
unpacked\ling\analysis.json
unpacked\operations\ling-full-1.log
unpacked\operations\ling-api.jsonl
unpacked\operations\ling-supervisor.json
unpacked\operations\ling-preflight-1.log
unpacked\operations\monitor-v3.log
unpacked\monitor.json
unpacked\telemetry.jsonl
```

Hugging Face upload staging:

```text
C:\Users\yyc\TTHE\artifacts\ling-v3-20260917\hf-upload\ling-v3-20260917
```

---

## 12. Next steps

1. Fix candidate scheduling:
   - validate all pending patches;
   - then probe up to `candidate_evals_per_round` maturing rules, oldest evidence first;
   - never let new rules default to `last_measured = 0`.
2. Align confidence calibration with Eq. 5:
   - derive `alpha0, beta0` from `c0` and `n0`.
3. Decide whether promotion should follow Eq. 6 exactly or keep the extra criteria.
4. Add an activation guard that flags or stops a run if no rule reaches the promotion gate after a meaningful number of rounds.
5. Rerun as v4 and treat the v3 Ling results as infrastructure/exploratory only.