#!/usr/bin/env bash
# First-session setup on a fresh GPU box, in one command.
#
# Everything here is free to run and takes minutes, but discovering any of it
# mid-session costs rented GPU time. Order matters: the PyTorch reference dumps
# and the per-operation golden tensors have to exist before a kernel can be
# checked against anything.
#
#   git clone https://github.com/AarnavNoble/llm-engine engine && cd engine
#   scripts/gpu_bringup.sh
#
# Idempotent: re-run it after a pod restart.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/Qwen2.5-0.5B-Instruct}
step() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
missing=()

step "machine"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  echo "CMAKE_CUDA_ARCHITECTURES should be $CAP"
else
  echo "no nvidia-smi: this script is for a GPU box"; exit 1
fi
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || missing+=("nvcc (install a CUDA *devel* image or the toolkit)")
command -v ncu  >/dev/null 2>&1 || missing+=("ncu (Nsight Compute, ships with the toolkit)")
command -v nsys >/dev/null 2>&1 || missing+=("nsys (Nsight Systems)")

step "system packages"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq cmake ninja-build git g++ curl libopenblas-dev python3-pip >/dev/null
  echo "cmake $(cmake --version | head -1 | awk '{print $3}'), ninja $(ninja --version)"
else
  echo "no apt-get; ensure cmake, ninja, g++, OpenBLAS and pip are present"
fi

step "python packages"
# Ubuntu 24.04 marks the system Python externally managed (PEP 668) and
# refuses a plain pip install, which aborts bring-up before anything is built.
# A rented pod is a disposable container with one job, so installing into the
# system interpreter is the right answer here rather than a virtualenv; the
# flag is probed for instead of assumed because older images reject it.
PIP_FLAGS=""
if pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
  PIP_FLAGS="--break-system-packages"
fi
pip install -q $PIP_FLAGS -r reference/requirements.txt
python3 - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

step "model weights"
[ -d "$MODEL" ] || scripts/download_model.sh Qwen/Qwen2.5-0.5B-Instruct
du -sh "$MODEL"

step "CPU build and full test suite"
# The CPU backend is the oracle every kernel is checked against, so it has to be
# green here before any CUDA work starts.
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build >/dev/null
./build/tests/engine_tests "~[model]"

step "PyTorch reference dumps (per layer)"
if [ -d "tests/data/$(basename "$MODEL")/ref" ]; then
  echo "already present: $(ls "tests/data/$(basename "$MODEL")/ref" | wc -l) files"
else
  python3 reference/dump_logits.py "$MODEL"
fi

step "golden tensors (per operation)"
mkdir -p tests/data/golden
./build/golden_dump --model "$MODEL" --out tests/data/golden
./build/tests/engine_tests "[golden]"

step "numerical suite against the reference"
./build/tests/engine_tests "[model]"

step "CUDA build"
if [ -f src/kernels/attention_decode.cu ]; then
  cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CAP}"
  cmake --build build-cuda
  ./build-cuda/engine generate --model "$MODEL" --backend cuda --prompt "The capital of France is" --max-tokens 8
else
  echo "the CUDA backend is not implemented yet, which is the work this box is for."
  echo "start with src/model/cuda/cuda_model.cu; the remaining items and their"
  echo "exit criteria are listed at the top of docs/gpu-setup.md."
fi

step "ready"
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing tools:"
  for m in "${missing[@]}"; do echo "  - $m"; done
fi
cat <<'NOTES'
Next:
  1. write a kernel, then check it alone:   ./build/tests/engine_tests "[golden]"
  2. wire it in, then check the model:      ./build/tests/engine_tests "[model]"
  3. when the backend runs end to end:      scripts/reproduce.sh
Stop the pod when you walk away.
NOTES
