# GPU box setup, and what is left to build

## What remains

Step-by-step commands with exit criteria are in [runbook.md](runbook.md); the
design of each kernel, including the reference it must match and the mistakes
that actually happen, is in [kernel-plan.md](kernel-plan.md). This page is the
machine setup.

The CPU backend is complete, and the CUDA backend runs end to end against the
CPU oracle. Done:

| # | Item | How it was checked |
|---|---|---|
| 1 | `cuda_model.cu`: weight upload, cuBLAS GEMMs, rmsnorm / RoPE / silu·up / embedding / cache-write kernels | GPU logits match the CPU oracle and the PyTorch dumps |
| 2 | **`attention_decode.cu`**: paged, online softmax, GQA-aware | matches the oracle, including a deliberately fragmented block table |
| 3 | Device-resident forward pass, no host round-trip per op | one synchronise per step; greedy output identical to the CPU backend |

Remaining, in dependency order:

| # | Item | Exit criterion |
|---|---|---|
| 4 | `sampling.cu` | greedy and top-p on device, logits never reach the host |
| 5 | `attention_prefill.cu` (or cuBLAS + masked softmax) | matches the oracle; prefill currently works through the decode kernel |
| 6 | Benchmark rows via `scripts/reproduce.sh` | the GPU rows stop saying "not measured" |
| 7 | vLLM row | `bench/vllm_baseline.py`, already written |
| 8 | Fuse RoPE with the cache write, vectorised loads, split-K, `ncu` table | bandwidth against the card's peak |
| 9 | CUDA graphs for decode at fixed batch sizes | `nsys` before and after |
| 10 | k3s + KEDA on this node, Grafana screenshot | scaler reacts to queue depth |

Item 6 is what makes the project what it claims to be: the mechanisms are
built and verified, and until they are measured on a GPU the throughput claims
are unsupported. Items 8 and 9 are optimisation; 10 is packaging. The cut
order if time runs short is 10, then 9, then 8, then 5 via cuBLAS.

The silu·up fusion is already done — `src/kernels/silu_mul.cu` computes
`silu(gate) * up` in one pass, which is the point of that kernel existing
rather than two. Only the RoPE/cache-write fusion is outstanding, and
`store_kv.cu` is deliberately separate from `rope.cu` so the fusion can be
measured as a before/after rather than assumed.

# GPU box setup (RunPod RTX 4090)

Rules: stop the pod whenever you walk away; keep everything on the persistent volume; push a tag before stopping.

1. **Pod**: RunPod → Deploy → RTX 4090 (24 GB, sm_89). Template `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (any CUDA 12.x *devel* image; you need `nvcc`). Attach a 50 GB **network volume** mounted at `/workspace`. Expose TCP 22 (SSH) and 8000 (API).
2. **SSH**: copy the pod's `ssh root@<ip> -p <port>` line into `~/.ssh/config` as host `gpu`, then VS Code → Remote-SSH → `gpu`.
3. **Toolchain** (once per fresh image; volumes persist, the image does not):
   ```bash
   apt-get update && apt-get install -y cmake ninja-build git libopenblas-dev
   nvcc --version && nvidia-smi
   ```
   Nsight Compute / Systems ship with the CUDA toolkit: `ncu --version`, `nsys --version`. If `ncu` refuses (`ERR_NVGPUCTRPERM`), the pod lacks perf-counter permission — pick a "secure cloud" pod or use `nsys` only.
4. **Repo + model** (on the volume):
   ```bash
   cd /workspace && git clone https://github.com/AarnavNoble/llm-engine.git engine && cd engine
   pip install -r reference/requirements.txt
   scripts/download_model.sh
   python3 reference/dump_logits.py       # writes tests/data/ref/*.bin
   ```
5. **Golden tensors for kernel work** (do this before writing any kernel):
   ```bash
   cmake -S . -B build -G Ninja && cmake --build build
   ./build/golden_dump --out tests/data/golden      # per-operation input/output pairs
   ./build/tests/engine_tests "[golden]"            # harness green against the CPU reference
   ```
   Each kernel then gets a test that is `tests/test_golden_ops.cpp` with the
   reference call swapped for a launch, so a wrong kernel points at itself
   instead of at the end of a 24-layer forward pass. The reference each kernel
   must match lives in `include/engine/ops.h`, and it is the same code the CPU
   model runs, so it cannot drift from what was verified against PyTorch.

6. **Build + test**:
   ```bash
   cmake -S . -B build -G Ninja -DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build && ./build/tests/engine_tests
   ./build/engine serve --model models/Qwen2.5-0.5B-Instruct --backend cuda
   ```
7. **Profiling**:
   ```bash
   nsys profile -o docs/prof/decode ./build/engine generate --backend cuda --model models/Qwen2.5-0.5B-Instruct --prompt "hi" --max-tokens 64
   ncu --set full -k regex:attention_decode -c 3 ./build/engine generate --backend cuda --model models/Qwen2.5-0.5B-Instruct --prompt "hi" --max-tokens 8
   ```
8. **Sanitizers.** Run these once on the Linux box, because they do not work on
   the Mac: ThreadSanitizer segfaults on a two-thread hello world under macOS 26
   on arm64, and the Address/UB build hangs during configuration. The Engine is
   the only multi-threaded component (HTTP threads submit and abort, the engine
   thread owns the scheduler and fires callbacks), so it is worth a clean run:
   ```bash
   cmake -S . -B build-tsan -G Ninja -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" \
     -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
   cmake --build build-tsan && ./build-tsan/tests/engine_tests "[model]"
   # and again with -fsanitize=address,undefined
   ```
9. **Budget**: ~40 GPU-hours total. `runpodctl stop pod <id>` (or the web button) every time.

Colab fallback: T4 is sm_75 — build with `-DCMAKE_CUDA_ARCHITECTURES=75`; no bf16, so fp16 only. Fine for the first kernels, not for the published numbers.
