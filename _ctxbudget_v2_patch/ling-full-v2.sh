#!/bin/bash
set -eu
REPO=/home/yangfan/meta-agent-test/all375-qwen38-20260901/repo-ctxbudget-v2
ROOT=/home/yangfan/mmh-exp/full375-20260916-v2
export PYTHONPATH="$REPO"
export MMH_EMBEDDING_PROVIDER=sentence-transformers MMH_EMBEDDING_DEVICE=cpu
export MMH_EMBEDDING_LOCAL_MODEL=/dataset1/yangfan/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/yangfan/meta-agent-test/.venv/bin/python -u -m experiments_tau3_mmh_study.full_study --repo "$REPO" --out "$ROOT/ling" --model inclusionAI/Ling-3.0-tiny --base-url http://127.0.0.1:8103/v1 --concurrency 2
