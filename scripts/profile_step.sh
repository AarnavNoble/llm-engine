#!/usr/bin/env bash
# Where a decode step's time goes, under real load.
#
#   scripts/profile_step.sh [concurrency] [requests]
#
# Writes bench/results/kernel-time.json, rendered by bench/report.py.
#
# This exists because Nsight Compute does not run on a rented pod: ncu needs
# NVreg_RestrictProfilingToAdminUsers=0, a host kernel-module parameter a
# container cannot set, and returns ERR_NVGPUCTRPERM without it. CUDA events
# need no privileges. They cannot report achieved bandwidth, so this does not
# replace scripts/profile_kernels.sh; it answers the question that actually
# decides what to optimise next, which is which region of a step the time is
# in.
#
# Run under load rather than a single generate. At batch 1 the GEMMs are tiny
# and attention is trivial, so the proportions describe a workload nobody
# serves. The defaults match the serving table so the split can be read next
# to those numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

CONC=${1:-32}
REQUESTS=${2:-128}
MODEL=${MODEL:-models/Qwen2.5-0.5B-Instruct}
RESULTS_DIR=${ENGINE_RESULTS_DIR:-bench/results}
mkdir -p "$RESULTS_DIR"

[ -x ./build/engine ] || { echo "build/engine missing; build with -DENGINE_CUDA=ON" >&2; exit 2; }

PORT=$((8600 + RANDOM % 200))
echo "profiling $REQUESTS requests at concurrency $CONC"

# ENGINE_PROFILE makes the model time each region with CUDA events and print a
# table plus the JSON when it is destroyed, which happens on a clean shutdown.
ENGINE_PROFILE=1 ENGINE_PROFILE_OUT="$RESULTS_DIR/kernel-time.json" \
  ./build/engine serve --model "$MODEL" --backend cuda --port "$PORT" \
    --num-blocks 4096 --max-num-seqs "$CONC" --max-batched-tokens 4096 \
    > /tmp/engine-profile.log 2>&1 &
PID=$!

for _ in $(seq 1 300); do curl -sf "localhost:$PORT/readyz" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/readyz" >/dev/null || { echo "server never became ready" >&2; kill $PID; exit 1; }

# Results go to a scratch directory: this run is a profile, and its throughput
# is depressed by the event records, so it must never reach the serving table.
ENGINE_RESULTS_DIR="$(mktemp -d)" python3 bench/loadgen.py --url "http://localhost:$PORT" \
  --tag profile --concurrency "$CONC" --num-requests "$REQUESTS" \
  --output-len 128 --output-dist geometric >/dev/null

# SIGTERM drains in flight work and exits, which runs the model destructor and
# with it the report.
kill -TERM "$PID"; wait "$PID" 2>/dev/null || true

echo
sed -n '/per-step cost over/,$p' /tmp/engine-profile.log || true
[ -f "$RESULTS_DIR/kernel-time.json" ] && echo && echo "wrote $RESULTS_DIR/kernel-time.json"
