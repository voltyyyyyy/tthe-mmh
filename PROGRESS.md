# Meta-Memory Harness implementation progress

Last updated: 2026-09-15 (Asia/Shanghai)

## Current status

Implementation is complete. The standalone Meta-Memory Harness (MMH) was built and tested before its LiveCodeBench integration.

| Area | Status | Notes |
| --- | --- | --- |
| Paper source | Complete | Vendored as `docs/papers/mmh-draft.pdf`; SHA-256 `1c048f5d74dd7588450777e0fded5994c94bed91ba896fe54df0e46881567435`. |
| MMH core | Complete | Typed rules and patches, transactional SQLite state, five operations, precedence filtering, exact rollback, evidence, and promotion are implemented. |
| Demo and conformance tests | Complete | Deterministic labeled demo and dependency-free lifecycle tests pass. |
| LiveCodeBench adapter | Complete | Optional, label-free integration uses public execution evidence only. |
| Final review | Complete | Static checks, offline test suite, adapter-to-core test, and documentation pass completed. |

## Decisions locked in

- The original paper PDF is preserved unchanged. Its worked examples use labels, but TTHE integration will only consume public tests, generated stress probes, and execution traces.
- The standalone MMH supports labeled validation for the paper-faithful demo.
- The implementation uses the paper's Algorithm 1 and equations when they conflict with appendix examples.
- Initial defaults: Beta prior `alpha=1`, `beta=1`; influence threshold `0.6`; promotion confidence `0.8`; minimum volatile age `3`; minimum two independent validation subsets.
- MMH rules guide proposer-authored executable harness edits. Rules are not automatically injected into frozen solver prompts.

## Verification

- `python -m meta_memory.demo` runs without credentials or network access. It demonstrates a validated repair, three independent successful recurrences, promotion, rollback of a harmful patch, and distribution-shift failure recording.
- Core smoke checks cover an interrupted SQLite run reopening with exact `REFINE` rollback, and the stable-tier delete gate: five consecutive failures across two subsets must precede staging, and the stable original remains until successful validation.
- `python tests/test_meta_memory.py` passes all seven standalone conformance tests: appendix response validation, every atomic operation, rollback, Eq. 5 and Eq. 11, conflict handling, resume behavior, recovery gating, and stable protection.
- `python tests/test_lcb_mmh_adapter.py` verifies a public-only future-task parent/candidate comparison reaches the core as MMH evidence. Every task receives a unique validation subset identifier.
- `python -m livecodebench.lcb_optimize --help`, `python -m compileall -q meta_memory livecodebench tests`, and whitespace validation pass without requiring API credentials.

## Integration behavior delivered

- `livecodebench.lcb_optimize --memory-mode mmh` creates separate public lineage JSON and SQLite memory state. The default `--memory-mode none` leaves the original loop unchanged.
- A candidate must supply an immutable, one-intervention proposal card before its MMH patch can be staged. MMH retrieves prior rules for the proposer only; it does not inject them into frozen solver prompts.
- Parent/candidate artifacts are re-run only on later matching public tasks. A public-test improvement validates a pending patch; an unsuccessful comparison rolls it back. Hidden-test scoring remains outside the memory path.
- Windows proposer worker timeouts terminate the entire subprocess tree through `taskkill /T /F`; POSIX retains process-group termination.

## Separate LCB online-MMH experiment (2026-09-17)

Status: implemented and offline-tested; no live paid benchmark run.

- Added `experiments/lcb_mmh_online/` with typed online evidence, explicit
  support/harmful/promotion thresholds, neutral ties, immutable content-addressed
  applications, failed-intervention retrieval, a hard SQLite budget ledger,
  replay caching, bounded scheduling, flat/none controls, and offline fixtures.
- Added behavior tests in
  `experiments/lcb_mmh_online/tests/test_online_mmh.py`; the existing
  `tests/test_meta_memory.py` and `tests/test_lcb_mmh_adapter.py` remain green.
- Added reproducible JSON configs for the `none`, `flat`, and `mmh` arms and
  documented the differences from the legacy paper/demo and LCB adapter.
- Added an optional live-loop shim (`experiments.lcb_mmh_online.live_runner`)
  that reuses the existing LiveCodeBench optimizer while swapping in the online
  memory adapter; `--arm none` remains the ordinary path.
- Remaining limitation: no paid live benchmark was run.  Provider-token
  enforcement for solver/proposer/judge subprocesses is represented by the
  process-safe ledger API and validation reservations; production live runs
  still need provider-usage instrumentation at every external call boundary.

## TTHE benchmark suite (`none` vs `mmh`) (2026-09-17)

Status: benchmark specifications, stream plans, dataset preparation/validation,
paired reporting helpers, and command planning implemented; no live paid runs.

- Added `experiments/tthe_benchmarks/` covering the paper's four hard slices:
  BIRD Mini-Dev hard50, LiveCodeBench hard60, SWE-bench Verified hard40, and
  DS-1000 hard50.
- The in-repo slices are validated for their paper counts and metadata.
- Dataset prep records which sources are already pullable in code
  (LiveCodeBench/DS-1000/SWE via HuggingFace) and which require an external local
  root (BIRD Mini-Dev).
- Added single-domain and mixed domain-blocked stream plans, two-arm command
  planning (`none` vs `mmh` only), and exact paired McNemar reporting helpers.
- LiveCodeBench is the only domain with an online-MMH adapter today; the mixed
  MMH arm and BIRD/DS-1000/SWE MMH adapters remain integration work.
