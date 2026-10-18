#!/usr/bin/env bash
# What sampling costs, which no other row here measures.
#
#   scripts/bench_sampling.sh [concurrency] [requests]
#
# Every row in the results table sends temperature 0. That is the right default
# for a benchmark -- greedy is reproducible and takes the model's own choices
# out of the measurement -- but it also meant the top-p path was never run
# under load, and a full sort of all 151,936 vocabulary entries sat in it
# unnoticed beneath a comment claiming a partial sort was enough.
#
# This runs the same workload twice, greedy and top-p 0.9, so the cost of
# nucleus sampling is a published number rather than an assumption. Pair it
# with ENGINE_PROFILE=1 to see the split inside the step.
set -euo pipefail
cd "$(dirname "$0")/.."

CONC=${1:-32}
REQUESTS=${2:-128}
MODEL=${MODEL:-models/Qwen2.5-0.5B-Instruct}
BACKEND=${BACKEND:-cuda}
export BACKEND
BIN=${BIN:-./build-cuda/engine}
[ -x "$BIN" ] || BIN=./build/engine
[ -x "$BIN" ] || { echo "no engine binary" >&2; exit 2; }

PORT=$((8700 + RANDOM % 200))
ENGINE_PROFILE=${ENGINE_PROFILE:-} \
  "$BIN" serve --model "$MODEL" --backend "$BACKEND" --port "$PORT" \
    --num-blocks 4096 --max-num-seqs "$CONC" --max-batched-tokens 4096 \
    > /tmp/engine-sampling.log 2>&1 &
PID=$!
trap 'kill -TERM $PID 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do curl -sf "localhost:$PORT/readyz" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/readyz" >/dev/null || { echo "server never became ready" >&2; exit 1; }

# A discarded pass first: the first requests after start-up pay for weight
# residency and allocator warm-up, and attributing that to sampling would be
# the same mistake this script exists to correct.
ENGINE_RESULTS_DIR="$(mktemp -d)" python3 bench/loadgen.py --url "http://localhost:$PORT" \
  --tag warmup --concurrency "$CONC" --num-requests 32 --output-len 64 >/dev/null

for row in "sampling-greedy 0.0 1.0" "sampling-topp 0.9 0.9"; do
  set -- $row
  echo "== $1 (temperature $2, top_p $3)"
  python3 bench/loadgen.py --url "http://localhost:$PORT" --tag "$1" \
    --concurrency "$CONC" --num-requests "$REQUESTS" --output-len 128 \
    --output-dist geometric --temperature "$2" --top-p "$3"
done
