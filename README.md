# engine — an LLM inference server, written from scratch

A serving engine for Llama-architecture models (Qwen2.5-0.5B, TinyLlama) with the parts production engines are built from: a **paged KV cache** with block tables, **continuous batching** with chunked prefill and preemption, **prefix caching** with refcounted block sharing, an **OpenAI-compatible streaming API**, Prometheus metrics, and a Kubernetes deploy that scales on queue depth. No PyTorch at runtime: weights are read from safetensors, the tokenizer parses `tokenizer.json` directly, and the forward pass is C++.

**Status.** The CPU backend is complete and verified layer by layer against PyTorch on two architectures; it serves real requests over the API. **The CUDA backend is not written yet** — `-DENGINE_CUDA=ON` tells you so rather than pretending. That means the serving mechanisms (paging, batching, prefix sharing, admission control) are built and measured, while GPU throughput and per-kernel bandwidth are not. Rows awaiting it are marked *not measured* on the results page instead of estimated. What remains, and the plan for it, is in [docs/gpu-setup.md](docs/gpu-setup.md).

Every optimization is a commit with a measured before/after number, and CI fails if any figure quoted in the docs disagrees with the data the harness produced.

## Benchmark

Full tables, the protocol, and what each figure does and does not mean: **[docs/results.md](docs/results.md)**, generated from `bench/results/*.json` by `bench/report.py`, so the published numbers cannot drift from what the harness measured. Rows awaiting the CUDA backend are rendered as *not measured* rather than estimated.

Reproduce everything on one machine with `scripts/reproduce.sh` (add `--quick` for a smoke check, which writes to a separate directory and leaves the published numbers alone).

Measured so far, all on a 10-core Mac with the CPU backend except the first row, which is hardware independent:

| Result | Number |
|---|---|
| KV slot utilization, paged vs contiguous with a declared cap | **98.8%** vs 55.5% (1.7% reserving full context) |
| Continuous vs static batching, throughput | **1.47x** |
| Continuous vs static batching, TTFT p50 | **5.9x faster** (13,673 ms to 2,330 ms) |
| Prefix caching, TTFT p50 on a shared 512-token prompt | **2.8x faster**, 77.5% block hit ratio (the ceiling for that workload) |
| Chunked prefill, worst decode stall with a 2,048-token prompt injected | **11.8x smaller** (23.2 s to 2.0 s) |
| Admission watermark, preemption recompute waste | **16.2% to 2.3%** of all token work, at 0.7% occupancy cost |

GPU throughput, per-kernel Nsight bandwidth, and a vLLM reference row on the same hardware are the remaining rows; `bench/vllm_baseline.py` and the harness are ready for them.

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

Deployment lives in [`deploy/`](deploy): a multi-stage Dockerfile with `gpu` and `cpu` targets, a Helm chart that requests one GPU per pod and splits startup/readiness/liveness probes so a draining pod is never killed mid-request, a KEDA ScaledObject that scales on `engine_queue_depth`, and a Grafana dashboard. `scripts/check_observability.py` runs in CI and fails the build if a dashboard panel or the autoscaler query references a metric the server does not expose.

## Correctness

`reference/dump_logits.py` runs the Hugging Face model in fp32 on 22 fixed prompts and dumps the residual stream after every layer plus final logits. `tests/test_cpu_model.cpp` replays the same prompts through the engine's forward pass and checks:

- residual stream after embedding and after each of the 24 layers: max |diff| ≤ 2e-3 · max |ref|
- final logits: max |diff| < 1e-2, argmax identical
- 16-token greedy generation through the scheduler with 6 sequences batched together: identical to `generate()`

The same comparison runs against a second architecture, **TinyLlama-1.1B**, which differs from Qwen in every way the loader could have hardcoded: no q/k/v bias, an untied `lm_head`, GQA ratio 8 instead of 7, `rms_norm_eps` 1e-5, `rope_theta` 10,000. It passes unchanged. Its sentencepiece-style tokenizer is supported too, so the whole path works end to end: `engine generate --model models/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is"` reproduces Hugging Face's greedy continuation token for token. Chat templates are still ChatML-only, so `/v1/chat/completions` against TinyLlama returns 400 rather than guessing a format.

The tokenizer is checked token-for-token against `tokenizers` on a 75-line corpus (CJK, emoji, contractions, whitespace runs, chat templates).

`scripts/e2e_smoke.py` runs 40 checks against a live server process: both endpoints, SSE framing, error codes, the metrics surface, prefix-cache hits observed through the metrics delta, and a SIGTERM mid-generation that must leave readiness at 503, refuse new work, and still return every token of the in-flight request.

## Serving features

- **Paged KV cache** (`kv_cache/`): 16-token blocks, free list, per-sequence block tables. Prompts allocate exactly `ceil(n/16)` blocks; `engine_kv_waste_fraction` reports allocated-but-empty slots.
- **Prefix caching**: full blocks are hashed (chained with the previous block's hash, so position-aware) and published only once their K/V are fully computed; later prompts with the same prefix reuse the physical blocks with a refcount, and unowned blocks sit in an LRU until memory pressure evicts them.
- **Continuous batching** (`scheduler/`): finished sequences leave the batch at the step they finish; new ones join at the next step under a `max_num_batched_tokens` budget. Long prompts are prefilled in chunks, which cuts the worst decode stall 11.8x when a 2,048-token prompt arrives mid-flight. Admission keeps a watermark of free blocks so a new sequence cannot steal the block a resident one needs next, which cut preemption recompute waste from 16.2% to 2.3% of all token work at a 0.7% occupancy cost. Out of memory → the youngest sequence is preempted and recomputed on resume. `--mode static` selects the static-batching baseline on the same code path.
- **API** (`api/`): `/v1/completions`, `/v1/chat/completions` (ChatML template), SSE streaming with UTF-8-safe chunking, `prompt_token_ids` for benchmarks, `ignore_eos`, per-request `seed`.
- **Observability**: Prometheus counters, gauges and histograms — queue depth, running sequences, KV blocks free/used/cached, prefix hit ratio, tokens/s, TTFT, inter-token latency, step time, preemptions.
- **Graceful drain**: SIGTERM flips `/readyz` to 503, finishes in-flight sequences, then exits.
- **Failure detection**: `/healthz` reports real state and fails when the step loop stalls with work queued, when weights are not loaded, or when the allocator's own block accounting stops adding up — a liveness probe that cannot fail never restarts a wedged process. An idle engine is explicitly not a stalled one.
- **Overload protection**: past `--max-queue-depth` the server returns 429 with `Retry-After` instead of queueing without limit, because unbounded acceptance only spreads the delay over every request. Queued work can be given a deadline, rejections are counted by cause, and nine Prometheus alert rules ship with the chart, each naming the action it implies.

## Scope, honestly

Single GPU, small model, fp16 weights, no tensor parallelism, no speculative decoding, no quantization. GEMMs are cuBLAS; writing a competitive GEMM is a separate project. The Kubernetes deploy is validated on a single-node k3s with one GPU.

## Layout

```
include/engine/   public headers          src/          implementation (config, safetensors, tokenizer,
tests/            Catch2 unit + numerical                kv_cache, scheduler, model/cpu, model/cuda, kernels, api)
reference/        PyTorch ground-truth dumps            bench/        load generator, results, table renderer
deploy/           Dockerfile, Helm, KEDA                docs/         design notes, kernel notes, GPU setup
```
