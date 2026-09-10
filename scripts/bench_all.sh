#!/usr/bin/env bash
# Run the standard benchmark 3x against a running server and record under a tag.
#   scripts/bench_all.sh v0.1-baseline [extra loadgen args]
set -euo pipefail
TAG="$1"; shift || true
for i in 1 2 3; do
  python3 "$(dirname "$0")/../bench/loadgen.py" --tag "$TAG" --concurrency 32 --num-requests 256 --output-len 128 "$@"
done
python3 "$(dirname "$0")/../bench/plot.py"
