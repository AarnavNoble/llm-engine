#!/usr/bin/env bash
# Second GPU session, in one command.
#
#   git clone https://github.com/AarnavNoble/llm-engine engine && cd engine
#   scripts/gpu_session2.sh
#
# The first session built and verified the CUDA backend and produced the
# serving table. This one closes the three gaps that remained, in the order
# that matters if the pod dies partway:
#
#   1. correctness, including the five invariance cases and the second
#      architecture, which had never run on a GPU at all
#   2. the vLLM reference row, the one number here measured by someone else's
#      code
#   3. per-kernel achieved bandwidth, which is what says whether any kernel is
#      worth optimising
#
# Correctness runs first on purpose. A measurement taken from a backend that
# turns out to be wrong is worse than no measurement, because it gets
# published.
#
# This does NOT re-run the serving benchmark. Those numbers were taken on an
# A40 and are still current; re-running them is only necessary on a different
# card, or once a kernel changes.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
FAILED=()

say "bring-up"
scripts/gpu_bringup.sh

say "second model"
# TinyLlama is what makes the two-architecture claim true on the GPU. Its
# reference dumps run on the CPU and take a few minutes.
if [ ! -d models/TinyLlama-1.1B-Chat-v1.0 ]; then
  scripts/download_model.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0
fi
if [ ! -d tests/data/TinyLlama-1.1B-Chat-v1.0/ref ]; then
  python3 reference/dump_logits.py models/TinyLlama-1.1B-Chat-v1.0
fi

say "build with CUDA"
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-86}" >/dev/null
cmake --build build >/dev/null
echo "built for sm_${ARCH}"
# The trap from session one: a binary that quietly ran on the CPU while
# labelled cuda.
./build/engine generate --model models/Qwen2.5-0.5B-Instruct --backend cuda \
  --prompt hi --max-tokens 1 >/dev/null

say "correctness, including the five new invariance cases"
./build/tests/engine_tests || FAILED+=("tests")

say "vLLM reference row"
# Its own virtualenv: vLLM pins a torch that must not displace the one
# generating the reference dumps.
VLLM_VENV=${VLLM_VENV:-/workspace/vllm-env}
if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  echo "installing vLLM into $VLLM_VENV (several minutes, ~3 GB)"
  python3 -m venv "$VLLM_VENV"
  "$VLLM_VENV/bin/pip" install -q --upgrade pip
  "$VLLM_VENV/bin/pip" install -q vllm || FAILED+=("vllm install")
fi
if [ -x "$VLLM_VENV/bin/vllm" ]; then
  "$VLLM_VENV/bin/python" bench/vllm_baseline.py --tag vllm \
    --concurrency 32 --num-requests 256 --num-blocks 4096 --runs 3 \
    || FAILED+=("vllm row")
fi

say "per-kernel bandwidth"
scripts/profile_kernels.sh || FAILED+=("ncu profile")

say "report"
python3 bench/report.py || FAILED+=("report")

say "summary"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "everything completed."
else
  # Named rather than hidden: a session that half worked should say which half.
  printf 'completed with failures: %s\n' "${FAILED[*]}"
fi
echo
echo "Pull the results before terminating the pod:"
echo "  scp -r <pod>:$(pwd)/bench/results ./"
