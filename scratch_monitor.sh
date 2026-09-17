#!/bin/bash
# Launch a TTHE/MMH run under full observability.
#
# Every run gets its own directory containing: a frozen manifest, the exact command,
# resource telemetry, the server log slice, a capture proxy log, and provenance.
# Nothing is inferred later; anything not written here cannot be analysed.
#
#   usage: monitor.sh <run-name> <command...>
#
set -euo pipefail

RUN="${1:?usage: monitor.sh <run-name> <command...>}"; shift
ROOT=/home/yangfan/meta-agent-test/mmh-volatile-lab
TTHE=/home/yangfan/TTHE
VENV=/home/yangfan/meta-agent-test/.venv
RUNS=/home/yangfan/mmh-exp/runs
RUN_DIR="$RUNS/$RUN"
PROXY_PORT="${PROXY_PORT:-8100}"
UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"

mkdir -p "$RUN_DIR"
cd "$TTHE"

# ---------- 1. manifest: freeze the comparison before anything runs ----------
{
  printf '{\n'
  printf '  "run_name": "%s",\n' "$RUN"
  printf '  "started_utc": "%s",\n' "$(date -u +%FT%TZ)"
  printf '  "host": "%s",\n' "$(hostname)"
  printf '  "user": "%s",\n' "$(whoami)"
  printf '  "experiment_root": "%s",\n' "$ROOT"
  printf '  "tthe_repo": "%s",\n' "$TTHE"
  printf '  "tthe_commit": "%s",\n' "$(git -C "$TTHE" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '  "tthe_dirty": "%s",\n' "$(git -C "$TTHE" status --porcelain 2>/dev/null | wc -l)"
  printf '  "python": "%s",\n' "$($VENV/bin/python --version 2>&1)"
  printf '  "vllm": "%s",\n' "$($VENV/bin/python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo unknown)"
  printf '  "model_snapshot": "%s",\n' "$(basename "$(ls -d /dataset1/yangfan/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ 2>/dev/null | head -1)")"
  printf '  "proxy_port": %s,\n' "$PROXY_PORT"
  printf '  "upstream": "%s",\n' "$UPSTREAM"
  printf '  "gpus": "%s",\n' "$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ';' | sed 's/"/\\"/g')"
  printf '  "command": "%s"\n' "$(printf '%s ' "$@" | sed 's/"/\\"/g')"
  printf '}\n'
} > "$RUN_DIR/manifest.json"

printf '%s\n' "$@" > "$RUN_DIR/command.txt"
sha256sum "$TTHE/livecodebench/slices/hard60.json" > "$RUN_DIR/benchmark.sha256" 2>/dev/null || true
cp "$TTHE/config.yaml" "$RUN_DIR/config.snapshot.yaml" 2>/dev/null || true
env | sort > "$RUN_DIR/env.snapshot.txt"

echo "=== run $RUN ==="
cat "$RUN_DIR/manifest.json"

# ---------- 2. resource telemetry ----------
tmux kill-window -t mmh-volatile:gpu-$RUN 2>/dev/null || true
tmux new-window -d -t mmh-volatile -n "gpu-$RUN" \
  "nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > '$RUN_DIR/gpu.csv'"

# ---------- 3. capture proxy: lossless record of every model call ----------
tmux kill-window -t mmh-volatile:proxy-$RUN 2>/dev/null || true
tmux new-window -d -t mmh-volatile -n "proxy-$RUN" \
  "PROXY_LOG='$RUN_DIR/api.jsonl' PROXY_PORT='$PROXY_PORT' PROXY_UPSTREAM='$UPSTREAM' bash '$ROOT/capture_proxy.py' 2>&1 | tee '$RUN_DIR/proxy.log'"
sleep 3

# ---------- 4. run, teeing output and recording the exit code ----------
export OPENAI_API_KEY=EMPTY
export TTHE_CONFIG="$TTHE/config.yaml"
export PYTHONPATH="$TTHE"
export PYTHONUNBUFFERED=1
# Route the solver through the capture proxy so api.jsonl holds the real traffic.
export OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1"
export OPENAI_API_BASE="http://127.0.0.1:$PROXY_PORT/v1"

set +e
"$@" 2>&1 | tee "$RUN_DIR/run.log"
RC=${PIPESTATUS[0]}
set -e

{
  printf '{"exit_code": %s, "finished_utc": "%s"}\n' "$RC" "$(date -u +%FT%TZ)"
} > "$RUN_DIR/exit.json"

# ---------- 5. self-contained evidence per run ----------
tmux kill-window -t mmh-volatile:gpu-$RUN 2>/dev/null || true
tmux kill-window -t mmh-volatile:proxy-$RUN 2>/dev/null || true
cp /home/yangfan/meta-agent-test/mmh-volatile-lab/logs/server0.log "$RUN_DIR/server.log" 2>/dev/null || true

# per-run copy of any solver cache / MMH state the run produced
mkdir -p "$RUN_DIR/artifacts"
for f in "$TTHE"/livecodebench/logs/*.json "$TTHE"/livecodebench/logs/*.sqlite \
         "$TTHE"/logs/*.json; do
  [ -e "$f" ] && cp "$f" "$RUN_DIR/artifacts/" 2>/dev/null || true
done

echo "--- run $RUN finished rc=$RC ---"
ls -la "$RUN_DIR"
exit "$RC"
