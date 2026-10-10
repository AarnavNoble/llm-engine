#!/usr/bin/env bash
# The inner loop while writing kernels: rebuild, check one operation against its
# golden tensor, report which one disagrees.
#
#   scripts/kernel_dev.sh              rebuild and check every operation
#   scripts/kernel_dev.sh rmsnorm      just that one
#   scripts/kernel_dev.sh --model      the full per-layer comparison (slower)
#   scripts/kernel_dev.sh --watch      re-run on every save
#
# Order matters when several fail: fix the earliest operation in the layer
# first, because every later tensor is computed from its output.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/Qwen2.5-0.5B-Instruct}
GOLDEN=tests/data/golden
BUILD=${BUILD:-build}
WATCH=0
FULL=0
FILTER=""

for arg in "$@"; do
  case "$arg" in
    --watch) WATCH=1 ;;
    --model) FULL=1 ;;
    --help|-h) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "unknown flag $arg" >&2; exit 2 ;;
    *) FILTER="$arg" ;;
  esac
done

# Build with CUDA when the backend exists, so the same command works before and
# after the kernels are written.
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=RelWithDebInfo)
if [ -f src/kernels/attention_decode.cu ] && command -v nvcc >/dev/null 2>&1; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  CMAKE_ARGS+=(-DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-89}")
fi

run_once() {
  printf "\n\033[1m== build\033[0m\n"
  cmake -S . -B "$BUILD" -G Ninja "${CMAKE_ARGS[@]}" >/dev/null
  if ! cmake --build "$BUILD" 2>&1 | tail -40; then
    printf "\033[31mbuild failed\033[0m\n"; return 1
  fi

  # Golden tensors are produced by the CPU path, so regenerate them whenever the
  # reference moves. They are the thing kernels are compared against.
  if [ ! -f "$GOLDEN/manifest.json" ] || [ "src/model/cpu/cpu_model.cpp" -nt "$GOLDEN/manifest.json" ]; then
    printf "\n\033[1m== refreshing golden tensors\033[0m\n"
    mkdir -p "$GOLDEN"
    "./$BUILD/golden_dump" --model "$MODEL" --out "$GOLDEN"
  fi

  printf "\n\033[1m== operations\033[0m\n"
  local spec='[golden]'
  [ -n "$FILTER" ] && spec="*$FILTER*"
  if "./$BUILD/tests/engine_tests" "$spec" 2>&1 | tail -30; then
    printf "\033[32moperations agree with the reference\033[0m\n"
  else
    printf "\033[31man operation disagrees; fix the earliest one in the layer first\033[0m\n"
    return 1
  fi

  if [ "$FULL" = 1 ]; then
    printf "\n\033[1m== full comparison against PyTorch\033[0m\n"
    "./$BUILD/tests/engine_tests" "[model]" 2>&1 | tail -24
  fi
  return 0
}

if [ "$WATCH" = 0 ]; then
  run_once
  exit $?
fi

# Poll rather than depend on inotify or fswatch being installed on a rented box.
echo "watching src/ and include/ (ctrl-c to stop)"
LAST=""
while true; do
  NOW=$(find src include -type f \( -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.cuh' \) \
        -exec stat -f '%m %N' {} + 2>/dev/null || find src include -type f \
        \( -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.cuh' \) -printf '%T@ %p\n')
  if [ "$NOW" != "$LAST" ]; then
    LAST="$NOW"
    run_once || true
    printf "\n\033[2mwaiting for changes\033[0m\n"
  fi
  sleep 2
done
