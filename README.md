# engine — an LLM inference server in C++ and CUDA, from scratch

A small serving engine for Llama-architecture models (Qwen2.5-0.5B, TinyLlama) with the parts production engines are built from: a **paged KV cache** with block tables, **continuous batching** with chunked prefill and preemption, **prefix caching** with refcounted block sharing, an **OpenAI-compatible streaming API**, Prometheus metrics, and a Kubernetes deploy that scales on queue depth. No PyTorch at runtime: weights are read from safetensors, the tokenizer parses `tokenizer.json` directly, and the forward pass is hand-written kernels plus cuBLAS.

Every optimization is a tagged commit with a measured before/after number.

## Benchmark

Qwen2.5-0.5B-Instruct, fp16, RTX 4090, 32 concurrent requests, prompt mix 128/512/1024, 128 output tokens, median of 3 runs. Each row is a git tag: `scripts/bench_all.sh <tag>` reproduces it.

| Config | Tokens/s | TTFT p50 / p95 (ms) | Inter-token p50 / p95 (ms) | KV waste % |
|---|---|---|---|---|
| HF transformers, batch=1 | | | | |
| `v0.1-baseline` static batching, contiguous KV | | | | |
| `v0.2-paged` static batching, paged KV | | | | |
| `v0.3-continuous` + continuous batching | | | | |
| `v0.4-prefix` + prefix caching (shared 512-tok system prompt) | | | | |
| `v0.5-kernels` + fused rope/cache-write, silu·up, optimized decode attention | | | | |
| `v0.6-graphs` + CUDA graphs | | | | |
| vLLM, same model and GPU (reference) | | | | |

*Status: CPU reference path complete and verified against PyTorch (`v0.0-cpu`). GPU rows land as the CUDA backend is built; see the schedule in `docs/design.md`.*

## Architecture

```
client ──HTTP/SSE──▶ api/         /v1/completions  /v1/chat/completions  /metrics  /readyz
                      │
                      ▼
                    scheduler/    iteration-level batching: one step = {prefill chunks, decode tokens}
                      │           admission under a token budget, FCFS, preempt-youngest on OOM
                      ▼
                    kv_cache/     block allocator + block tables, hash→block prefix map, refcounts, LRU
                      │
                      ▼
                    model/        Llama forward pass over paged K/V: cpu/ (fp32 oracle) and cuda/
                      │
                    kernels/      rmsnorm · rope+cache-write · paged decode attention · prefill attention
                                  silu·up · embedding · sampling            (cuBLAS for the GEMMs)
```

## Quickstart

```bash
scripts/download_model.sh                         # Qwen2.5-0.5B-Instruct -> models/
cmake -S . -B build -G Ninja && cmake --build build
./build/engine serve --model models/Qwen2.5-0.5B-Instruct            # --backend cuda on a GPU box
python3 examples/openai_client.py                 # any OpenAI client works
curl localhost:8000/metrics
```

CPU-only build (default) needs CMake ≥ 3.24, a C++17 compiler and optionally BLAS (Accelerate on macOS, OpenBLAS on Linux). CUDA build: `-DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`.

`engine generate --model DIR --chat "Explain paged attention" --max-tokens 64` runs a one-off completion; `--copies 8` runs eight concurrently through the batcher.

## Correctness

`reference/dump_logits.py` runs the Hugging Face model in fp32 on 22 fixed prompts and dumps the residual stream after every layer plus final logits. `tests/test_cpu_model.cpp` replays the same prompts through the engine's forward pass and checks:

- residual stream after embedding and after each of the 24 layers: max |diff| ≤ 2e-3 · max |ref|
- final logits: max |diff| < 1e-2, argmax identical
- 16-token greedy generation through the scheduler with 6 sequences batched together: identical to `generate()`

The tokenizer is checked token-for-token against `tokenizers` on a 75-line corpus (CJK, emoji, contractions, whitespace runs, chat templates).

## Serving features

- **Paged KV cache** (`kv_cache/`): 16-token blocks, free list, per-sequence block tables. Prompts allocate exactly `ceil(n/16)` blocks; `engine_kv_waste_fraction` reports allocated-but-empty slots.
- **Prefix caching**: full blocks are hashed (chained with the previous block's hash, so position-aware) and published only once their K/V are fully computed; later prompts with the same prefix reuse the physical blocks with a refcount, and unowned blocks sit in an LRU until memory pressure evicts them.
- **Continuous batching** (`scheduler/`): finished sequences leave the batch at the step they finish; new ones join at the next step under a `max_num_batched_tokens` budget. Long prompts are prefilled in chunks so decode latency stays flat. Out of memory → the youngest sequence is preempted and recomputed on resume. `--mode static` selects the static-batching baseline on the same code path.
- **API** (`api/`): `/v1/completions`, `/v1/chat/completions` (ChatML template), SSE streaming with UTF-8-safe chunking, `prompt_token_ids` for benchmarks, `ignore_eos`, per-request `seed`.
- **Observability**: Prometheus counters, gauges and histograms — queue depth, running sequences, KV blocks free/used/cached, prefix hit ratio, tokens/s, TTFT, inter-token latency, step time, preemptions.
- **Graceful drain**: SIGTERM flips `/readyz` to 503, finishes in-flight sequences, then exits.

## Scope, honestly

Single GPU, small model, fp16 weights, no tensor parallelism, no speculative decoding, no quantization. GEMMs are cuBLAS; writing a competitive GEMM is a separate project. The Kubernetes deploy is validated on a single-node k3s with one GPU.

## Layout

```
include/engine/   public headers          src/          implementation (config, safetensors, tokenizer,
tests/            Catch2 unit + numerical                kv_cache, scheduler, model/cpu, model/cuda, kernels, api)
reference/        PyTorch ground-truth dumps            bench/        load generator, results, table renderer
deploy/           Dockerfile, Helm, KEDA                docs/         design notes, kernel notes, GPU setup
```
