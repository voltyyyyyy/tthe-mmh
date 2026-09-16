# tau3 × MMH — fair-comparison freeze spec

Purpose: make an MMH arm comparable to the archived **fast-only Meta-Agent** run
(`tau3-all375-qwen38-meta-5iter-s42-v2`) by holding every non-mechanism factor identical.

Rule: **the memory mechanism is the only planned difference.** Anything else that differs must be
declared as a deviation with a reason.

## 1. Frozen reference: the archived comparison

Source: `/home/yangfan/meta-agent-test/all375-qwen38-20260901/` and the diagnostics run
`all375-metamem-metaagent-20260906T1025Z`. Documentation:
`repo/docs/tau3_all375_experiment_results.md` (SHA-256 `b27b2c6f893038e889ba233325fda4ebdb830a4a1615db4ab7d9609d570e0d86`).

| System | Validation (75) | Untouched test (75) |
|---|---:|---:|
| Frozen minimal baseline | 18/75 (24.0%) | 19/75 (25.3%) |
| Human-engineered reference | 29/75 (38.7%) | 31/75 (41.3%) |
| Fast-only Meta-Agent champion | 32/75 (42.7%) | 31/75 (41.3%) |

Recorded integrity values — reproduce, do not re-derive:

| Artifact | SHA-256 |
|---|---|
| Benchmark yaml | `6ef47733039c009d702bd183b0b2c7c7b085969eaa887a4b72386b7937dc7a36` |
| Split manifest | `c1a5915765017bc48abb74eb1b207c74e411617e1c420e441b35358c99d1210c` |
| Historical harness | `78b4aab94604e30dc8833b40fc1ab4fec52463f65b6579960a98f586b1ae4bf9` |
| All-domain harness copy | `db0cdc8b783c3a8639c2ef6e904a6cc41991fe8b3d64154ff877aba69f065bc7` |
| Final evolved champion | `64c63866c059d557e1d8c397566556881a7c7b40b396f3d5032f7f058c78d82d` |

## 2. What must be held identical

| Factor | Value | Notes |
|---|---|---|
| Benchmark | `reproduction/qwen_tau3/all375/benchmark.yaml` | verify hash above |
| Domains | airline, retail, telecom, banking_knowledge | 4 domains |
| Split manifest | `all375/split_manifest.json`, seed 42 | 225/75/75 |
| Task IDs | exactly the frozen lists | see `all375_split_frozen.json` |
| Base harness | the all-domain copy (`db0cdc8b…`) | **not** the historical one |
| Model | `Qwen/Qwen3.8-27B` for actor, user sim, and any judge | served name must match |
| Endpoint | one vLLM process per arm, via that arm's capture proxy | separate proxy log per arm |
| Concurrency | 8 (`META_AGENT_CONCURRENCY`) | same for both arms |
| Task timeout | 3600 s (`TAU3_TASK_TIMEOUT_S`) | same for both arms |
| tau2 commit | `a2c024725189473d2d7cea3a5cfdbcc67478e41f` | pinned in manifest |
| Retrieval | `bm25` | per benchmark backend |

**Baseline-selection caveat (must be reported).** The archived frozen minimal harness crashed before
its first model call on all 42 telecom + banking test tasks, because it assumed numeric task IDs:
`seed = int(ctx.task["task_id"]) * 100 + step`. Its 19/75 is therefore partly a compatibility
artifact. The honest MMH baseline arm is the **all-domain copy** (`db0cdc8b…`) that runs every domain.
State which baseline is used in any claim, and do not compare MMH against the crashing baseline as if
that were a capability difference.

## 3. MMH partitioning — round-scoped, not split-scoped

MMH does not consume "a validation split". `mmh_adapter.py:331-333` requires
`subset_id` to match `(\d+):` — i.e. `"<round>:<label>"` — and `:338` forbids validating a rule on
the batch that created it. Promotion then needs `len(independent_subsets) >= 2` **and**
`successful_lifespan >= 3` **and** `elapsed_age >= 3`.

So the frozen 75 validation tasks must be **partitioned into round-scoped subsets**, e.g.

```
round 1 -> subset "1:val-a"   (tasks 1..25 of the frozen validation list)
round 2 -> subset "2:val-b"   (tasks 26..50)
round 3 -> subset "3:val-c"   (tasks 51..75)
round 4 -> subset "4:val-a"   (re-visit; a second independent subset for the same rules)
```

Every validation task must be excluded from the proposal/search pool. Never validate a rule on a task
whose trace produced it.

### Two hard gates that shape the budget

1. **Promotion needs recurrences, not just passes.** A rule must succeed across >= 3 distinct rounds
   and >= 2 distinct subsets, at age >= 3. Reusing the frozen 75 across rounds is what produces this —
   and on telecom (measured **556 s/task**) that repetition is the dominant cost.
2. **`minimum_comparable_precedents = 3`.** Until 3 resolved precedents exist within the Gaussian
   bandwidth, `estimate_influence` falls back to `judge_confidence` instead of precedent weighting.
   The first ~3 patches therefore carry no precedent signal; do not interpret early-round influence
   or trait values as if the precedent log were active.

## 4. Role assignment (what each frozen split is for)

| Split | Tasks | Role | MMH mechanism |
|---|---:|---|---|
| search | 225 | surfaces failures; proposal/subset identity only | never becomes validation evidence |
| validation | 75 | evidence events, round-scoped | `ValidationEvidence(subset_id="<round>:…")` |
| test | 75 | **report only, touched once** | never enters the loop |

The test split is the only honest final number. Validation is *not* untouched after the first round —
it is repeatedly consulted by the gate, so a score computed on it is validation, never a test result.

## 5. Embedding provider (mandatory)

Traits are cosines over rule embeddings, so the provider must be semantic and identical across arms.
MMH's default is normalized feature hashing, which scores **cos(paraphrase) = 0.0** on a verified pair
(`"index out of bounds in a slice operation"` vs `"list index error when slicing beyond the array
length"`). With Qwen3-Embedding-0.6B the same pair scores **0.778**, and an unrelated pair 0.413.

```
MMH_EMBEDDING_PROVIDER=sentence-transformers
MMH_EMBEDDING_LOCAL_MODEL=/dataset1/yangfan/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

Dimensions: **1024**. Embeddings must be frozen at capture time — never recomputed under a different
provider, or trait vectors become incomparable.

## 6. Observability requirements

Per arm, per run: frozen manifest before launch (commit, hashes, endpoint, concurrency, timeouts,
exact command); capture-proxy JSONL of all model traffic; solver cache; per-task records
(`wall_time_s`, `num_turns`, reward — the token fields were `null` in the archived run and should be
populated so cost is measured, not estimated); MMH SQLite snapshot per round; server and GPU logs;
exit code. Export with remote checksum, copy down, verify, then tear down.

## 7. Open decisions before launch

1. **MMH mechanism on tau3** — harness *code* edits (as in the LiveCodeBench `--memory-mode mmh` path,
   where a patch is an executable harness change) or *behavioral rules* injected into the agent prompt
   (as in the archived MetaMem / LifelongAgentBench units)? This changes what a "rule" is, what the
   traits measure, and whether the frozen 225/75/75 is even the right shape.
2. **Repeats vs breadth** — at 1 rollout/task a 75-task split has ~1.33 pts/task granularity, and the
   archived run showed 12 terminal `ReadTimeout`s confounding one arm. One repeat cannot separate
   mechanism from noise.
3. **Whether MMH replaces or augments the fast loop.** The name of the archive
   (`all375-metamem-metaagent-…`) suggests a MetaMem+MetaAgent combination already exists; if so, the
   fair comparison is against that, not against fast-only.

## 8. Known limitation to state up front

The archived 375-run is a **single rollout per task**, with different serving load across arms and
different nonnumeric seed schemes. Any MMH-vs-fast comparison inherits that noise floor. Report paired
task-level flips and exact McNemar, not bare percentage differences.
