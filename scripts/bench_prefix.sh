#!/usr/bin/env bash
# Prefix-caching experiment: N requests sharing one long system prompt, each
# with its own short tail. Runs the same workload with the cache on and off and
# prints the time-to-first-token difference.
#
# TTFT is the metric that moves: a cache hit skips prefill for the shared
# blocks entirely, so this is a real reduction in work rather than a scheduling
# effect, and it is therefore measurable on the CPU backend.
#
#   scripts/bench_prefix.sh [shared_prefix_tokens] [requests] [concurrency]
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX=${1:-512}
REQUESTS=${2:-32}
CONC=${3:-8}
PROMPT=$((PREFIX + 128))
MODEL=models/Qwen2.5-0.5B-Instruct
BACKEND=${BACKEND:-cpu}

run() {  # run <tag> <extra server flags...>
  local tag=$1; shift
  local port=$((8200 + RANDOM % 300))
  ./build/engine serve --model "$MODEL" --backend "$BACKEND" --port "$port" \
    --num-blocks 2048 --max-num-seqs "$CONC" "$@" > "/tmp/engine-$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do curl -sf "localhost:$port/readyz" >/dev/null && break; sleep 1; done
  python3 bench/loadgen.py --url "http://localhost:$port" --tag "$tag" \
    --concurrency "$CONC" --num-requests "$REQUESTS" --output-len 8 --output-dist fixed \
    --prompt-len "$PROMPT" --shared-prefix "$PREFIX" --seed 3
  kill -TERM $pid; wait $pid 2>/dev/null || true
}

RESULTS_DIR=${ENGINE_RESULTS_DIR:-bench/results}
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR/prefix-on.json" "$RESULTS_DIR/prefix-off.json"

# The off case always runs first, so without this it pays the cold-device cost
# -- clocks ramping from idle, driver context creation -- and the on case runs
# on a warm GPU. That difference lands straight in the TTFT ratio this script
# exists to report, in the direction that flatters prefix caching. Each case
# already gets its own fresh server, so the per-process costs were symmetric;
# this is the device-level asymmetry the ordering introduces.
#
# It matters more here than in the serving table: that table takes a median of
# three runs, while these two numbers are single runs with nothing to absorb an
# outlier.
echo "=== warmup (discarded) ==="
ENGINE_RESULTS_DIR="$(mktemp -d)" run prefix-warmup --no-prefix-cache >/dev/null

echo "=== prefix caching OFF ==="
run prefix-off --no-prefix-cache
echo "=== prefix caching ON ==="
run prefix-on

python3 - <<'PY'
import json, os, pathlib
d = pathlib.Path(os.environ.get("ENGINE_RESULTS_DIR", "bench/results"))
r = lambda t: json.loads((d / f"{t}.json").read_text())[-1]
off, on = r("prefix-off"), r("prefix-on")
print(f"\n{'':28} {'off':>10} {'on':>10} {'change':>10}")
for k, label, unit in (("ttft_p50_ms", "TTFT p50", "ms"), ("ttft_p95_ms", "TTFT p95", "ms"),
                       ("e2e_p50_s", "end-to-end p50", "s"), ("wall_s", "wall clock", "s"),
                       ("tokens_per_s", "tokens/s", "")):
    a, b = off[k], on[k]
    chg = f"{a/b:.1f}x faster" if k != "tokens_per_s" and b else (f"{b/a:.2f}x" if a else "-")
    print(f"{label:28} {a:10.1f} {b:10.1f} {chg:>10}")
hit = on["prefix_hit_blocks"] / max(on["prefix_total_blocks"], 1)
print(f"\nprefix cache hit ratio: {100*hit:.1f}% of full prompt blocks "
      f"({on['prefix_hit_blocks']:.0f} of {on['prefix_total_blocks']:.0f})")
PY
