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
5. **Build + test**:
   ```bash
   cmake -S . -B build -G Ninja -DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build && ./build/tests/engine_tests
   ./build/engine serve --model models/Qwen2.5-0.5B-Instruct --backend cuda
   ```
6. **Profiling**:
   ```bash
   nsys profile -o docs/prof/decode ./build/engine generate --backend cuda --model models/Qwen2.5-0.5B-Instruct --prompt "hi" --max-tokens 64
   ncu --set full -k regex:attention_decode -c 3 ./build/engine generate --backend cuda --model models/Qwen2.5-0.5B-Instruct --prompt "hi" --max-tokens 8
   ```
7. **Budget**: ~40 GPU-hours total. `runpodctl stop pod <id>` (or the web button) every time.

Colab fallback: T4 is sm_75 — build with `-DCMAKE_CUDA_ARCHITECTURES=75`; no bf16, so fp16 only. Fine for the first kernels, not for the published numbers.
