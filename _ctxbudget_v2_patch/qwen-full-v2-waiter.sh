#!/bin/bash
set -eu
REPO=/home/yangfan/meta-agent-test/all375-qwen38-20260901/repo-ctxbudget-v2
ROOT=/home/yangfan/mmh-exp/full375-20260916-v2
SCORES=/home/yangfan/meta-agent-test/all375-qwen38-20260901/repo/experience/qwen38-tau3-all375-search/candidates/study_2bdca7ea_3a90e4a76aa441/scores.json
while [ ! -s "$SCORES" ]; do sleep 30; done
export PYTHONPATH="$REPO"
/home/yangfan/meta-agent-test/.venv/bin/python -c "import json; r=json.load(open('$SCORES')); assert len(r.get('per_task',[]))==4, len(r.get('per_task',[]))"
export MMH_EMBEDDING_PROVIDER=sentence-transformers MMH_EMBEDDING_DEVICE=cpu
export MMH_EMBEDDING_LOCAL_MODEL=/dataset1/yangfan/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/yangfan/meta-agent-test/.venv/bin/python -u -m experiments_tau3_mmh_study.full_study --repo "$REPO" --out "$ROOT/qwen" --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8102/v1 --concurrency 2
