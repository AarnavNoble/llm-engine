#!/usr/bin/env bash
# Reproduce every published number, from a clean checkout.
#
#   scripts/reproduce.sh            everything available on this machine
#   scripts/reproduce.sh --quick    smaller workloads, for a smoke check
#   scripts/reproduce.sh --cpu      skip the GPU rows even if a GPU is present
#
# What runs depends on what is available, and the script says which rows it is
# skipping rather than silently producing a partial table:
#
#   always        correctness suite, KV-cache efficiency, prefix caching,
#                 chunked prefill, static vs continuous batching
#   with a GPU    the CUDA backend rows
#   with vllm     the vLLM reference row
#
# Everything lands in bench/results/*.json and is rendered by bench/report.py.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
FORCE_CPU=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --cpu) FORCE_CPU=1 ;;
    *) echo "unknown flag $arg" >&2; exit 2 ;;
  esac
done

MODEL=models/Qwen2.5-0.5B-Instruct
# A quick run is a smoke check, not a measurement, so it must not overwrite the
# published numbers or the generated report.
if [ $QUICK = 1 ]; then
  export ENGINE_RESULTS_DIR=bench/results-quick
  export ENGINE_REPORT_DIR=bench/results-quick
  mkdir -p "$ENGINE_RESULTS_DIR"
fi
RESULTS_DIR=${ENGINE_RESULTS_DIR:-bench/results}
REQUESTS=$([ $QUICK = 1 ] && echo 32 || echo 256)
CONC=$([ $QUICK = 1 ] && echo 8 || echo 32)
RUNS=$([ $QUICK = 1 ] && echo 1 || echo 3)
BLOCKS=4096

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
have_gpu() { [ $FORCE_CPU = 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

# ----------------------------------------------------------------- preparation
say "model weights"
[ -d "$MODEL" ] || scripts/download_model.sh Qwen/Qwen2.5-0.5B-Instruct

say "build"
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if have_gpu; then CMAKE_ARGS+=(-DENGINE_CUDA=ON); BACKEND=cuda; else BACKEND=cpu; fi
cmake -S . -B build -G Ninja "${CMAKE_ARGS[@]}" >/dev/null
cmake --build build >/dev/null
echo "backend: $BACKEND"

say "correctness"
# The numerical comparison needs the PyTorch dumps; generate them if absent.
if [ ! -d tests/data/Qwen2.5-0.5B-Instruct/ref ]; then
  echo "generating reference dumps (one-off, a few minutes)"
  python3 reference/dump_logits.py "$MODEL"
fi
./build/tests/engine_tests

# ------------------------------------------------------------------ allocator
say "KV cache efficiency (hardware independent)"
./build/kv_waste --requests "$REQUESTS" --blocks "$BLOCKS" --sweep \
  --json "$RESULTS_DIR/kv-waste.json"

# --------------------------------------------------------- scheduler behaviour
run_server() {  # run_server <mode> <port> [extra flags...]
  local mode=$1 port=$2; shift 2
  ./build/engine serve --model "$MODEL" --backend "$BACKEND" --mode "$mode" --port "$port" \
    --num-blocks "$BLOCKS" --max-num-seqs "$CONC" --max-batched-tokens 4096 "$@" \
    > "/tmp/engine-repro-$mode-$port.log" 2>&1 &
  echo $!
  for _ in $(seq 1 300); do curl -sf "localhost:$port/readyz" >/dev/null && return 0; sleep 1; done
  echo "server on $port never became ready" >&2; exit 1
}

bench_mode() {  # bench_mode <mode> <tag>
  local mode=$1 tag=$2 port=$((8400 + RANDOM % 200))
  rm -f "$RESULTS_DIR/$tag.json"
  local pid; pid=$(run_server "$mode" "$port")
  for _ in $(seq 1 "$RUNS"); do
    python3 bench/loadgen.py --url "http://localhost:$port" --tag "$tag" \
      --concurrency "$CONC" --num-requests "$REQUESTS" --output-len 128 --output-dist geometric
  done
  kill -TERM "$pid"; wait "$pid" 2>/dev/null || true
}

say "static vs continuous batching ($BACKEND backend)"
bench_mode static "$BACKEND-static"
bench_mode continuous "$BACKEND-continuous"

say "prefix caching"
scripts/bench_prefix.sh 512 "$REQUESTS" "$CONC"

say "chunked prefill"
python3 scripts/bench_chunked_prefill.py --backend "$BACKEND" --chunks 2048 512 128

# --------------------------------------------------------------- reference row
if command -v vllm >/dev/null 2>&1 && have_gpu; then
  say "vLLM reference row"
  python3 bench/vllm_baseline.py --tag vllm --concurrency "$CONC" \
    --num-requests "$REQUESTS" --num-blocks "$BLOCKS" --runs "$RUNS"
else
  say "skipping vLLM row"
  echo "needs a GPU and 'pip install vllm'"
fi

if ! have_gpu; then
  say "skipping CUDA backend rows"
  echo "no GPU visible; see docs/gpu-setup.md"
fi

say "report"
python3 bench/report.py
echo
if [ $QUICK = 1 ]; then
  echo "quick mode: results in $RESULTS_DIR, published numbers untouched."
else
  echo "docs/results.md and docs/results.html are up to date."
fi
