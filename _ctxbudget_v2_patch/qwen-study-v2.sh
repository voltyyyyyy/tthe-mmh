#!/bin/bash
set -eu
REPO=/home/yangfan/meta-agent-test/all375-qwen38-20260901/repo-ctxbudget-v2
export PYTHONPATH="$REPO"
export MMH_EMBEDDING_PROVIDER=sentence-transformers
export MMH_EMBEDDING_DEVICE=cpu
export MMH_EMBEDDING_LOCAL_MODEL=/dataset1/yangfan/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/yangfan/meta-agent-test/.venv/bin/python -u -m experiments_tau3_mmh_study.supervise_study --repo "$REPO" --root /home/yangfan/mmh-exp/full375-20260916-v2 --label qwen
