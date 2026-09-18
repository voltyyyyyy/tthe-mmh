# TTHE benchmark suite: `none` vs `mmh`

This package defines the two benchmark studies requested:

1. **Single-domain study** — one hard slice at a time.
2. **Mixed-domain study** — a domain-blocked round-robin stream over all four task families.

Only two arms are used: `none` (ordinary TTHE) and `mmh` (shared online memory).
No flat arm.

## What is already in the repo

The hard-slice manifests are present and validated by this package:

| Domain | Slice | Count | Contexts | Underlying data |
|---|---:|---:|---|---|
| BIRD Mini-Dev | `text_to_sql/slices/genuine_hard50.json` | 50 | `db_id` | not vendored |
| LiveCodeBench | `livecodebench/slices/hard60.json` | 60 | `platform`, `difficulty`, `contest_date` | HuggingFace hub |
| SWE-bench Verified | `swe/slices/hard40.json` | 40 | `repo` | HuggingFace `datasets` |
| DS-1000 | `ds1000/slices/hard50.json` | 50 | `library`, `perturbation` | HuggingFace `datasets` |

The underlying benchmark data are **not** vendored.  The repo has loaders for
LCB, DS-1000, and SWE-bench that pull from their public sources; BIRD still
requires a local `bird_root` with `mini_dev.json` and the SQLite databases.

## Commands

Validate the in-repo slices:

```bash
PYTHONPATH=. python -m experiments.tthe_benchmarks validate-slices
```

Prepare/pull external data where the environment allows it:

```bash
PYTHONPATH=. python -m experiments.tthe_benchmarks prepare \
  --data-root benchmark_data \
  --bird-root /path/to/bird_mini_dev
```

LCB downloads `livecodebench/code_generation_lite test6`; DS-1000 and SWE-bench
use `datasets.load_dataset`; BIRD is only verified locally because its official
data is not on the same simple HuggingFace path.

Print the two commands for a single-domain comparison:

```bash
PYTHONPATH=. python -m experiments.tthe_benchmarks commands-single \
  --domain livecodebench --total-budget 2000000 --run-name hard60
```

Write stream plans:

```bash
PYTHONPATH=. python -m experiments.tthe_benchmarks plan-single \
  --domain livecodebench --batch-size 5 --output benchmark_data/single_lcb.json

PYTHONPATH=. python -m experiments.tthe_benchmarks plan-mixed \
  --batch-size 5 --repeats 3 --output benchmark_data/mixed_all.json
```

## Mixed stream design

`plan-mixed` creates domain-blocked batches:

`BIRD blocks -> LCB blocks -> SWE blocks -> DS-1000 blocks -> repeat`

Each batch contains tasks from exactly one domain.  A single harness is not
shared across domains because the interfaces differ; the `mmh` arm would share
the memory layer, while each domain keeps its own incumbent harness.

## Current integration status

- **Single-domain LCB**: runnable now through
  `experiments.lcb_mmh_online.live_runner` with `--arm none` or `--arm mmh`.
- **Single-domain BIRD/DS-1000/SWE**: `none` commands are planned; MMH adapters
  do not exist yet.
- **Mixed-domain study**: stream plans and command metadata exist, but a
  cross-domain runner and public-only MMH adapters for BIRD/DS-1000/SWE are
  still required before the mixed `mmh` arm can run.

No paid benchmark was launched by these commands.
