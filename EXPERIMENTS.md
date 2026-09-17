# MMH on tau3 — experiment handoff

## Active full-study handoff — 2026-09-16 08:37 Asia/Shanghai

The user authorized all 375 tasks on **both Qwen and Ling**, with MMH, flat, and
frozen-baseline arms. Remote tmux supervisors are running; laptop shutdown is safe.
At this checkpoint both are performing four-domain search-only preflight. Each
automatically starts its full study when preflight finishes without infrastructure
errors. A scored model/harness failure does not fail preflight.

The user explicitly requested a `gpt-5.6-terra` subagent to manage the runs.
`/root/terra_manager` is assigned that role and confirmed the live remote supervisors.
The durable manager after laptop shutdown is the remote tmux supervisor, not the
chat subagent. Ling's banking preflight hit HTTP 400 and is under diagnosis; inspect
the exact captured response before deciding whether it is infrastructure or a
native context/model/harness limitation. Do not count an arbitrary HTTP 400 as a
transport failure merely because later requests return 200.

Confirmed response for that Ling attempt: native window 131072, requested output
8192, input at least 122881 (total at least 131073). This is a context-budget
capability/harness outcome, not a transport outage. Terra is assigned a narrowly
evidenced classification fix and regression test before full-study launch; preserve
the original attempt and do not silently truncate the task or exceed native context.

- Host: `yangfan@10.96.43.165`; tmux session `mmh-run`.
- Windows: `qwen-study`, `ling-study`, `study-monitor`, `qwen-capture`,
  `ling-capture`, `server27b`, `ling128k` (legacy `proxy` remains).
- Root: `/home/yangfan/mmh-exp/full375-20260916`.
- Run code: `/home/yangfan/meta-agent-test/all375-qwen38-20260901/repo/experiments_tau3_mmh_study`.
- Entry points: `full_study.py`, `supervise_study.py`, `monitor_study.py`,
  `analyze_study.py`. The older `round_driver.py` remains pilot-only.
- Launch scripts: `operations/qwen-study.sh`, `operations/ling-study.sh`.
- Read `monitor.json`, each model's `status.json`, and
  `operations/{qwen,ling}-supervisor.json`. Logs are in `operations/`.
- Qwen capture: port 8102 → 8000. Ling capture: port 8103 routes Ling requests to
  8001 and simulator/grader requests to Qwen 8000. All traffic stays on the lab.
- Ling now serves its supported 131072 context limit, using its existing isolated
  vLLM 0.29.0 environment. Qwen's existing vLLM environment was not upgraded.
- Official Qwen3-Embedding-0.6B snapshot `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
  runs through sentence-transformers on CPU. Actual load/inference verified:
  1024 dimensions, paraphrase cosine 0.74849 versus unrelated 0.27879.
- One-task retail smoke completed for both models. Ling's scored failure used no
  customer tools; separate required/automatic tool probes both parsed correctly.
  Do not relabel that behavioral failure as infrastructure.

Protocol: six rounds per domain, all 225 search tasks once per arm, three disjoint
validation groups/domain, then frozen-memory coverage of all 75 validation and 75
test tasks. Final test is never shown to proposer or used during adaptation. One
proposal/round and at most two paired candidate probes/round. Positive paired reward
difference validates; negative difference rejects; ties are indecisive. Both learned
arms retain initial validation; flat has one validated pool with no promotion gate.
Prompt cap is eight rules: MMH stable-first, flat newest-first. This compares those
memory policies, not an isolated gate with identical prompt ranking.

Infrastructure attempts and raw traces are preserved. Only failed infrastructure
tasks are retried (three attempts/evaluation; bounded supervisor restarts). A persistent
fault stops with `needs_attention`; the supervisor does not repair arbitrary bugs.
Interrupted rounds restore their start snapshots and reuse verified evaluation receipts.
Source/config hashes are frozen and resume refuses changed provenance.

Fixes: SQLite same-file backup hang, missing embedding wiring, proposer SDK
`extra_body`, dropped Ling reasoning alias, preservation of full baseline instructions,
telecom task-ID colon normalization (from the other session), baseline promoted-list
type. New runner fixes held-out task separation, repeated candidate evidence, paired
validation, trait freezing, trace error inspection, and task/attempt isolation.
Verification: 44 existing checks plus four new full-study regression tests pass.

Research limits: one stream/trial per model; no pass^2/pass^3 estimate or replicated
causal claim. Domain difficulty changes alone do not demonstrate conditional concept
drift. The new runner logs stable-rule degradation but does not implement stable
retirement. Gate-controlled trait prediction needs enough promotions; zero promotions
is an identifiability limitation, not proof of tier equivalence. Results remain pending.

Next session: inspect statuses before restarting anything. Do not edit deployed source
during a run. After completion, inspect `analysis.json`/`REPORT.md`, audit coverage and
infrastructure attempts, then archive, copy locally and verify checksums before any
server teardown. Earlier handoff sections below describe the superseded pilots.

---

**Status:** infrastructure complete and verified; the MMH lifecycle has **not yet been observed
firing on real data** (every attempt so far died on infrastructure before reaching the
mechanism). Read §7 before trusting any run.

**Last updated:** 2026-09-16, ~01:45 local (Asia/Shanghai)

---

## 1. The goal

### Primary research question

> Under a genuinely **non-stationary** task stream, does MMH's two-tier memory
> (volatile → stable, with an evidence-gated promotion rule) preserve performance across
> regime shifts — and do the two tiers actually diverge?

### Why this experiment exists (the gap it fills)

The MMH paper draft (`docs/papers/mmh-draft.pdf`, text at `docs/papers/mmh-draft.txt`)
is framed entirely around non-stationarity:

- §3.1: *"each query Q_i arises from a shifting task context P_i … with **P_i ≠ P_{i+1}**"*
- §3.2: the two-tier design exists because *"a single unfavorable round can corrupt the entire
  rule base"*, and the volatile tier *"absorbs transient noise"*
- The `τ` (lifespan) field exists *"to handle concept drift"*
- RQ1 asks whether Meta-Memory can *"sustain or enhance long-term average performance … under
  sequential task distribution shifts"*

**But every experiment the paper actually reports is stationary** (LawBench, S2D, USPTO-50k,
IMO-level math, TerminalBench-2). A `grep` for *non-stationary*, *concept drift*, or
*distribution shift* matches **only** the intro and method sections — never the experiments.

The consequence is that in a stationary stream the volatile and stable tiers are
indistinguishable in expectation, and the temporal-lifespan mechanism is untestable. The
paper's own largest ablation drop is **−Stable Tier (52.3% → 49.2%)**, i.e. its most valuable
component is never exercised under the condition it was designed for.

**This experiment runs that missing test.**

### Secondary question (from the original brief)

Do **novelty, surprise, uniqueness, sparsity, and overlap-with-stable** separate volatile from
stable memory, and do they predict promotion? (`lifespan` is a **gate variable, not a
finding** — see §6.)

### Falsification criteria (fixed in advance)

1. **Tiers do not diverge** — trait distributions overlap once lifespan is excluded ⇒ the
   two-tier design is decorative.
2. **No promotion benefit under drift** — MMH ≤ flat-memory arm across regime boundaries.
3. **Stable rules do not degrade at shifts** — if nothing crashes at a regime boundary there is
   no concept drift to manage, and `τ` is unjustified.
4. **Promotion unpredictable from traits** — AUC < 0.65 with CI including 0.5, after
   controlling for the gate's own variables.

Negative results on (1) and (3) are the most interesting available outcomes.

---

## 2. Core design decisions (and why)

| Decision | Choice | Rationale |
|---|---|---|
| Rule representation | **Behavioral guidelines** (natural language `φ`/`ψ`), not code diffs | Forced by the paper's Eq. 11: influence is a similarity-weighted success rate over `z = f(φ)`, the embedding of the rule's **applicability context**. Only coherent for comparable natural language. Code diffs have no applicability context — this is why the LiveCodeBench `--memory-mode mmh` code-edit interpretation makes Eq. 11 degenerate. |
| Editable surface | The **guideline block in the system prompt**; harness code frozen | Both arms run a byte-identical harness; only the guideline text differs. |
| Benchmark | tau3 **all375** (4 domains) | Domain boundaries are natural regime shifts, giving real non-stationarity for free. |
| Non-stationarity | **Regime schedule** (domain-ordered blocks), not shuffled | A shuffled i.i.d. stream destroys the drift signal. |
| Validation | **Round-scoped held-out subsets** | MMH's gate counts `independent_subsets`; see §6. |

### The constraint that silently breaks everything

From `MMHConfig` defaults — `promotion_age=3`, `promotion_subsets=2`,
`stable_failure_rounds=5`:

| Regime length | Promotion possible | Stable retirement possible |
|---:|---|---|
| 1–2 rounds | **no** | no |
| 3–4 rounds | yes | **no** |
| ≥5 rounds | yes | yes |

**A regime shorter than 3 rounds can never promote anything**, so a naive "one round per
domain" drift schedule turns the whole two-tier mechanism into a no-op — silently. The
implementation checks this up front (`schedule.assess_schedule`) and refuses non-viable
schedules.

Also: **effective promotion latency is `promotion_age + 1` rounds**, not `promotion_age`. A rule
staged in round N first ages at N+1, so both freshness conditions cannot be met before N+4.

---

## 3. The benchmark

**`reproduction/qwen_tau3/all375/`** in the lab repo — all 375 official tau3 `base` tasks.

| Item | Value |
|---|---|
| Domains | airline (50), retail (114), telecom (114), banking_knowledge (97) |
| Partition | search 225 / validation 75 / test 75 (seed 42, 60/20/20 per domain) |
| Benchmark yaml SHA-256 | `6ef47733039c009d702bd183b0b2c7c7b085969eaa887a4b72386b7937dc7a36` |
| Split manifest SHA-256 | `c1a5915765017bc48abb74eb1b207c74e411617e1c420e441b35358c99d1210c` |
| tau2 commit | `a2c024725189473d2d7cea3a5cfdbcc67478e41f` |
| Baseline harness SHA-256 | `db0cdc8b783c3a8639c2ef6e904a6cc41991fe8b3d64154ff877aba69f065bc7` (`reproduction/qwen_tau3/harnesses/reference_all375/harness.py`) |

### Archived comparison numbers (for reference, **not** reproduced by this work)

From the 2026-09-01/02 run `tau3-all375-qwen38-meta-5iter-s42-v2`:

| System | Validation (75) | Untouched test (75) |
|---|---:|---:|
| Frozen minimal baseline | 18/75 (24.0%) | 19/75 (25.3%) |
| Human-engineered reference | 29/75 (38.7%) | 31/75 (41.3%) |
| Fast-only Meta-Agent champion | 32/75 (42.7%) | 31/75 (41.3%) |

**Caveat carried from that run's own docs:** the frozen *minimal* harness crashed before its
first model call on all 42 telecom + banking test tasks (`int(ctx.task["task_id"])` on
non-numeric IDs). Its 19/75 is partly a compatibility artifact. Use the **all-domain copy**
above as the honest baseline.

**Task-pool ordering matters.** The search pool is flattened in manifest order:
`airline` starts at index 0, `retail` at **30**, `telecom` at **98**, `banking_knowledge` at
**166**. Airline is the *easiest* domain (23 recorded baseline failures vs 96 banking, 82
retail), so a low `--start-at` will not produce failures and the mechanism will never trigger.

---

## 4. What the tau3 runtime actually is (important — avoids a costly wrong turn)

**The 375-task run did NOT use Claude Code.** Every harness in it — vanilla, reference, the
human baseline, and all evolved candidates — uses the **program contract**:

```python
async def run(ctx):          # ctx is a TauProgramContext
    result = await ctx.call_model(messages=..., system=..., extra_body=...)
    ...
    observation = await ctx.execute_tool(call.name, call.arguments)
```

verified across all five harness files. `benchmarks/tau3/program_adapter.py` imports
`claude_agent_sdk` **zero** times; it pulls tau2 tools as OpenAI schemas
(`_openai_tools`) and posts to `<base_url>/chat/completions` directly.

`TauProgramContext` exposes: `ctx.system_prompt`, `ctx.call_model(...)`, `ctx.execute_tool`,
`ctx.task`, `ctx.customer_ended`, `ctx.log_event`, `ctx.finish`, plus `num_turns` /
`input_tokens` / `output_tokens` for cost metrics.

> **Trap:** `harnesses/agents/tau3_airline/harness.py` is a **Claude-style** artifact that
> exports `build_options() -> ClaudeAgentOptions`. It is unused, and it requires the `claude`
> CLI, which **does not exist on the lab host**. Building against it fails with
> `[claude-code:unrecognized_model]`. Use the program contract.

---

## 5. Code inventory

### 5.1 On this laptop — `C:\Users\yyc\TTHE\experiments\tau3_mmh\`

| Module | Purpose | Tests |
|---|---|---|
| `guidelines.py` | Two-tier guideline memory: `Guideline`, `TaskObservation`, `GuidelineMemory` (round-scoped evidence, promotion, snapshot/restore), reason-level gate reporting | — |
| `integrity.py` | Guards separating **infrastructure faults** from model failures (§6) | 13 |
| `proposer.py` | `OfflineProposer` (deterministic, credential-free) and `LLMProposer` (validated via `meta_memory.adapters.parse_patch_response`) | — |
| `schedule.py` | `RegimeSpec`, `Schedule`, validation subsets, `assess_schedule` viability check | (in lifecycle) |
| `runner.py` | Synthetic/offline round loop, per-round metrics, change-point detection | 11 |
| `traits.py` | Volatile traits frozen at staging; AUC / Cliff's δ / Spearman; `ex_ante_surprise` | — |
| `logging_utils.py` | Per-run structured artifacts (manifest, rounds, lifetimes, patches, traits, exit) | — |
| `sweep.py` | Seeded offline sweep, tiered-vs-flat arms, aggregate reporting | — |
| `tau3_binding.py` | **Real tau3 bridge**: score parsing, error classification, guideline rendering, eval invocation | 12 |
| `tau3_harness/harness.py` | The guideline-injecting program harness (derived from `reference_all375`) | 6 |
| `round_driver.py` | **The orchestrator** that drives the real benchmark through the MMH loop | — |
| `demo.py` | Offline end-to-end drift demo | — |
| `tests/` | 4 suites | **42 total** |

Run all suites:

```bash
cd C:\Users\yyc\TTHE
python -m experiments.tau3_mmh.tests.test_guidelines        # 11
python -m experiments.tau3_mmh.tests.test_integrity         # 13
python -m experiments.tau3_mmh.tests.test_binding           # 12
python -m experiments.tau3_mmh.tests.test_harness_contract  # 6
```

### 5.2 On the lab — `/home/yangfan/meta-agent-test/all375-qwen38-20260901/repo/`

- `experiments_tau3_mmh/` — the 13 `.py` modules above + `tau3_harness/`
- `meta_memory/` — **copied to the repo root** (the package imports it as top-level)
- `experiments_tau3_mmh/meta_memory` must NOT exist (creates a shadowing duplicate)

### 5.3 The harness edit surface

`experiments_tau3_mmh/tau3_harness/harness.py` is the all-domain baseline **byte-identical**
except for one line:

```python
# baseline:  system = ctx.system_prompt + "\n\n" + GUIDELINES        # a literal
# ours:      system = ctx.system_prompt + "\n\n" + load_guidelines()  # from memory
```

`load_guidelines()` reads `$MMH_GUIDELINES_FILE`; a missing/empty file falls back to the
baseline block (so a missing file can never make the agent *worse* than baseline and be blamed
on the memory system). Everything else — the 50-step loop, premature-stop recovery,
`_seed_for_task` (including the non-numeric-ID fix), tool execution, finish reasons — is
unchanged. **This is what makes the comparison a test of the memory rather than of the
harness.**

---

## 6. Correctness invariants (each backed by a test)

1. **A scored zero means the model or harness genuinely failed.** An infrastructure fault is
   fatal and loudly reported. No third category.
2. **Errored tasks are never scored.** Timeouts, crashes, and **unrun tasks** are `error`
   observations, excluded from the success rate and from the gate.
3. **A pending guideline cannot steer the round that judges it.** Staging happens after
   evaluation.
4. **Validation judges the candidate, not compliance.** A proposal is resolved by
   *re-running the held-out tasks with it in force* — never by checking whether the agent
   already used it (a pending rule is not injected, so that would make promotion impossible).
5. **`subset_id` is the validation data group, not the round.** Encoding the round into it
   (`"3:val-a"`) makes every round look like a fresh independent subset and silently dissolves
   the `>= promotion_subsets` gate.
6. **Mixed validation rounds are indecisive** — a patch stays pending rather than resolving on
   a coin flip.
7. **Validated rules keep collecting evidence** — `successful_lifespan` counts distinct
   successful *rounds*, so evidence must not stop at resolution.
8. **`lifespan` is a gate variable, not a finding.** The gate requires
   `successful_lifespan >= promotion_age`, so "volatile is younger than stable" restates
   `engine.py:482-491`. It is reported as a control the other traits must beat. Offline sweep
   confirmed AUC 0.500 (zero information).
9. **`surprise` as a posterior shift partly encodes the outcome** and must not be reported as
   a predictor. `traits.ex_ante_surprise` is the non-circular version (precedent base rate
   before any outcome).
10. **Runs are deterministic given a seed** — 14 scientific fields identical across repeats;
    only `wall_time_s` varies.

### Integrity guard taxonomy (`integrity.py`)

| Guard | Kind | Protects against |
|---|---|---|
| `error_rate_too_high` | **fatal** | timeouts/crashes reported as a model failure rate |
| `candidate_budget_exceeded` | defect | truncation stranding patches, misread as "failed to promote" |
| `no_guidelines_injected` | defect | mechanism never switched on ⇒ null result means nothing |
| `degenerate_score_series` | defect | saturation, where "both arms at 1.0" isn't equivalence |
| `proposals_all_rejected_at_stage` | defect | proposals vanishing without a record |
| `traits_missing_for_staged_rules` | defect | trait statistics silently omitting rules |
| `no_validation_decisions` | defect | the gate never resolving anything |
| `validation_call_failed` | warn | a validation crash recorded as evidence about a rule |
| unfinished run | **fatal** | an unverified run defaulting to "healthy" |

**fatal** raises `InfrastructureFault` (run stops). **defect** lets the run finish but marks
`analysable: false`; the sweep **excludes** it from aggregates. Every run writes
`integrity.json`. `ExperimentRunner.finish()` must be called; `require_finished()` fails closed
if it wasn't — a missing report is never read as "clean".

---

## 7. Bugs found (each produced wrong results silently)

1. **`ninja` missing from PATH** — vLLM selected FlashInfer for sampling, which JIT-compiles via
   `ninja`; `FileNotFoundError: 'ninja'` killed engine init after the model loaded. Fix:
   `.venv/bin` on PATH.
2. **Proxy forwarded stale `Content-Length`** while re-serialising the body ⇒
   `LocalProtocolError: Too little data for declared Content-Length`. Fix: strip
   `content-length`/`content-encoding`/`transfer-encoding`/`connection` when forwarding.
3. **Proxy dropped the reasoning field.** vLLM names it `reasoning` in some versions and
   `reasoning_content` in others; the parser only read the latter, so traces were silently lost.
   Fix: capture both, normalise to `reasoning_text`.
4. **Client bypassed the capture proxy.** `lcb_bridge` reads `base_url` from `config.yaml`, not
   from `OPENAI_BASE_URL`. The run went direct to :8000 and ran **entirely uncaptured**.
5. **Zombie generations.** Killing an eval client does not cancel upstream requests; 6 requests
   kept generating on the GPU, blocking the next run.
6. **Candidate-directory collision (severe).** Candidate dirs are keyed by `--name`, and
   `_locate_scores` takes the newest match, so a still-running eval read the **previous run's
   `scores.json`** — a 100% airline result would have been recorded as retail evidence and
   promoted a rule on data it never saw. Fix: names derive from the run directory
   (`mmh_<run-dir>_r_1`), **plus** `_load_per_task` now raises `InfrastructureFault` if the
   reported task set ≠ the requested one.
7. **User simulator unroutable (severe).** tau2 defaults to `gpt-4.1-2025-04-14`; the vLLM
   server did not serve that name, so every simulator call 404'd and tau2 retried **10× with
   15 s backoff** — 53 minutes for 2 tasks, after which the tasks were recorded as
   `reward: 0.0`. **The conversations never started.** Fix: serve the alias, exactly as the
   archived `server.sh` did.
8. **Unrun tasks scored as failures (severe).** Records with `num_turns: null`,
   `wall_time_s: 0.0` and `trial_dir: ""` and no trace files are *unrun* tasks, not failures.
   Feeding them to the gate injects fabricated negative evidence. Fix: `classify_error` now
   detects this (`_never_ran`) and excludes them — conservatively, requiring positive evidence
   of non-execution so genuine failures are never reclassified.

---

## 8. Infrastructure (lab host)

**Host:** `ssh yangfan@10.96.43.165` (hostname `ail2-jaehong-ws2`; the name may not resolve
from your machine — use the IP). 3× RTX PRO 6000 Blackwell, 97,887 MiB each.

### Two virtualenvs

| Purpose | Path | Notes |
|---|---|---|
| vLLM **0.27.1** (torch 2.13.0+cu130) | `/home/yangfan/meta-agent-test/.venv` | Qwen3.8-27B; **the working setup for the main run — do not upgrade** |
| vLLM **0.29.0** | `/dataset1/yangfan/venvs/ling-vllm` | Ling-3.0-tiny only; isolated to protect the above |

### Servers

**Qwen3.8-27B (GPU 0, :8000)** — must serve tau2's default alias or the user simulator 404s:

```bash
CUDA_VISIBLE_DEVICES=0 /home/yangfan/meta-agent-test/.venv/bin/vllm serve \
  /dataset1/yangfan/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --served-model-name Qwen/Qwen3.8-27B gpt-4.1-2025-04-14 claude-opus-4-5 \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 262144 \
  --max-num-seqs 32 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --language-model-only
```

**Ling-3.0-tiny (GPU 1, :8001)** — `architectures: [BailingMoeV3ForCausalLM]`; vLLM 0.27.1
does **not** register it, 0.29.0 does. Needs `--trust-remote-code` and the `ling3` parsers:

```bash
CUDA_VISIBLE_DEVICES=1 /dataset1/yangfan/venvs/ling-vllm/bin/vllm serve \
  /dataset1/yangfan/.cache/huggingface/hub/models--inclusionAI--Ling-3.0-tiny/snapshots/e3a47d5b986e7141b6efd62597d598ebb392060d \
  --served-model-name inclusionAI/Ling-3.0-tiny \
  --host 127.0.0.1 --port 8001 --trust-remote-code --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser ling3 --reasoning-parser ling3
```

Ling notes: thinking is per-request via `chat_template_kwargs: {"enable_thinking": bool}`;
recommended sampling `temperature=1.0, top_p=0.95, top_k=20`; it exposes the trace in
`reasoning` (not `reasoning_content`). Startup takes ~2 min (kernel JIT); Triton
deprecation warnings are normal, not errors.

**Capture proxy (:8100 → :8000)** — lossless JSONL of every model call (request, response,
reasoning, usage, latency). Script: `/home/yangfan/meta-agent-test/mmh-volatile-lab/capture_proxy.py`.

```bash
PROXY_LOG=/home/yangfan/mmh-exp/run_api.jsonl PROXY_PORT=8100 \
PROXY_UPSTREAM=http://127.0.0.1:8000 \
/home/yangfan/meta-agent-test/.venv/bin/python \
  /home/yangfan/meta-agent-test/mmh-volatile-lab/capture_proxy.py
```

**tmux session `mmh-run`** holds `server27b`, `ling`, `proxy`. Other sessions
(`meta-agent-test`, `tau3-all375`) belong to earlier work — **do not touch**.

### Environment for an eval

```
PYTHONPATH=<repo>
LOCAL_OPENAI_V1_BASE=http://127.0.0.1:8100/v1
LOCAL_MODEL_BASE_URL=http://127.0.0.1:8100/v1
OPENAI_BASE_URL=http://127.0.0.1:8100/v1
OPENAI_API_BASE=http://127.0.0.1:8100/v1
OPENAI_API_KEY=EMPTY
TAU2_DATA_DIR=/home/yangfan/meta-agent-test/tau2-bench/data
MMH_GUIDELINES_FILE=/home/yangfan/mmh-exp/harness_state/guidelines.md
META_AGENT_CONCURRENCY=2
TAU3_TASK_TIMEOUT_S=3600
```

`MMH_EMBEDDING_PROVIDER=sentence-transformers` +
`MMH_EMBEDDING_LOCAL_MODEL=.../models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
(1024-d) for semantic traits — MMH's default hash embedding scores `cos(paraphrase) = 0.000`,
i.e. it measures token overlap, not meaning.

---

## 9. How to run

```bash
cd /home/yangfan/meta-agent-test/all375-qwen38-20260901/repo

# ensure servers + proxy are up (see §8), then:
mkdir -p /home/yangfan/mmh-exp/run_N
nohup /home/yangfan/meta-agent-test/.venv/bin/python -m experiments_tau3_mmh.round_driver \
  --arm mmh \
  --rounds 3 --tasks-per-round 2 \
  --start-at 30 \
  --concurrency 2 \
  --out /home/yangfan/mmh-exp/run_N \
  > /home/yangfan/mmh-exp/run_N.log 2>&1 &
```

- `--arm mmh` (guidelines from memory) or `--arm baseline` (frozen block)
- `--start-at` selects the domain: **30 = retail, 98 = telecom, 166 = banking** (0 = airline,
  which is too easy to trigger failures)
- `--rounds`: **≥3 required** for promotion to be reachable at all
- Per-round flow: snapshot → render guidelines → eval → parse → stage proposals → **validate
  candidate by re-running the held-out tasks with it in force** → promote → log. Candidate
  validation costs one extra eval per round, which is why it's capped at 1/round.

Artifacts land in `--out`: `manifest.json`, `rounds.json`, `guidelines.md`, `memory.sqlite`,
`integrity.json`.

**Cost, measured:** ~343 s/task on airline; **retail took 53 min for 2 tasks during the
failed run** (that was retry backoff, not work). Expect **20–60 min per round** at
concurrency 2. A full 225/75/75 run is hundreds of rollouts per arm — days of GPU time.

---

## 10. Current state (at time of writing)

| Item | State |
|---|---|
| Qwen3.8-27B :8000 with aliases | **UP**, verified (`gpt-4.1-2025-04-14 → 'OK'`) |
| Ling-3.0-tiny :8001 | **UP**, verified (`content: 'LING_OK'`) |
| Capture proxy :8100 | UP |
| Test suites | **42/42 pass** |
| Run `run_r3` (retail, 3 rounds) | **RUNNING**, ~1 h elapsed, **0 NotFoundErrors** |
| MMH lifecycle observed on real data | **NO — not yet** |
| Ling harness wiring | **NOT DONE** — Ling has never run a tau3 task |
| Baseline arm | NOT RUN (contends with the MMH arm for GPU 0) |

### What is NOT done

1. **The mechanism has never been observed firing.** No rule staged, validated, or promoted on
   real data. Until `rounds.json` shows `staged > 0` and `memory.sqlite` has a `rules` row,
   treat the harness as unproven.
2. **Ling has never run a tau3 task.** Only a raw completion. Its harness path (different
   `reasoning` field, `enable_thinking`) is untested, and `tau3_harness` has no Ling-specific
   handling.
3. **Pilot validation reuses the originating tasks.** `_validate_candidate` re-runs the same
   tasks the candidate was derived from — held-out *by round*, not by task. Fine for a
   mechanism pilot; **must be tightened before any claim.**
4. **One failure signature only.** The proposer emits a single candidate type
   (`no_tool_call_before_answer`), so candidate variety is ~zero. The trait analysis needs a
   real proposal distribution.
5. **Per-round subset labels.** Disjoint by construction here, but a full 225/75/75 run needs
   proper `val-r1…r5` labels over the frozen validation split.
6. **No baseline arm, no regime/drift schedule** (the design calls for maturation regimes ≥5
   rounds then shortened drift regimes).
7. **Budget undecided.** Input needed.

### Immediate next steps

1. Read `run_r3`'s `rounds.json` + `memory.sqlite`. If `staged > 0` and a rule row exists,
   confirm the lifecycle end-to-end. If `n_errors > 0`, the unrun-detection fired and the
   failures are infrastructure, not model.
2. If it works: add the baseline arm on a disjoint task slice, then widen.
3. Wire Ling into the harness path and run one task to prove it.
4. Tighten validation to disjoint tasks, and add real failure-class signatures.
5. Decide the budget before committing to a large run.

### Watching guidance

Do not poll impatiently — a round takes 20–60 min. Use a background watcher on
`rounds.json` (non-empty ⇒ a round completed). Useful single checks:

```bash
pgrep -af round_driver                       # alive?
ls -la /home/yangfan/mmh-exp/run_N/rounds.json
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader
wc -l /home/yangfan/mmh-exp/run_api.jsonl    # model calls captured
grep -c NotFoundError /home/yangfan/mmh-exp/run_N.log
```

**Operational caution:** killing an eval client does **not** cancel upstream requests —
zombie generations keep the GPU busy. Kill by verified PID; never `pkill -f vllm` (other users
share this host).

---

## 11. Reference: the offline synthetic sweep

Independent of the lab, `sweep.py` ran a 20-seed × 2-arm synthetic drift study with full
logging (1,510 staged rule observations, 361 artifact files). Results in
`experiments/tau3_mmh/RESULTS_OFFLINE_SWEEP.md`. Headlines:

- Gate vs no-gate: pre-shift success **0.705 (tiered) vs 0.483 (flat)** — promoting everything
  builds a larger stable tier that performs worse (spurious rules displace useful ones under
  the prompt cap).
- The planted shift was detected in **20/20 tiered runs but only 5/20 flat** — a degraded
  baseline makes drift harder to diagnose.
- Trait informativeness (label = promoted): uniqueness 0.574, sparsity 0.584, overlap 0.539,
  novelty 0.461, **surprise 0.500, lifespan 0.500** (zero information).
- **Caveat:** the synthetic world plants validity as similarity to stable memory, so novelty is
  anti-predictive *by construction* there. These results bound what the *design* can detect;
  they are not evidence about real agents.
