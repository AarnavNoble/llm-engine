# engine — an LLM inference server, written from scratch

[![ci](https://github.com/AarnavNoble/llm-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/AarnavNoble/llm-engine/actions/workflows/ci.yml) **899,082 assertions across 81 test cases**, two model architectures, CPU and CUDA backends. [Measured results](https://aarnavnoble.github.io/llm-engine/).

A serving engine for Llama-architecture models (Qwen2.5-0.5B, TinyLlama) with the parts production engines are built from: a **paged KV cache** with block tables, **continuous batching** with chunked prefill and preemption, **prefix caching** with refcounted block sharing, an **OpenAI-compatible streaming API**, Prometheus metrics, and a Kubernetes deploy that scales on queue depth. No PyTorch at runtime: weights are read from safetensors, the tokenizer parses `tokenizer.json` directly, and the forward pass is C++.

**Status.** The CPU backend is complete and verified layer by layer against PyTorch on two architectures; it serves real requests over the API. The CUDA backend runs: the forward pass is device-resident, with hand-written kernels for embedding, RMSNorm, RoPE, the K/V cache write, silu·up, the residual and bias adds, and paged decode attention, plus cuBLAS for the GEMMs. It is gated against the CPU oracle by `tests/test_cuda_model.cpp`, including a case that deliberately fragments the block table so a kernel computing slots arithmetically cannot pass.

Two pieces are deliberately not written: a dedicated prefill kernel (prefill tokens go through the decode kernel, which is correct but leaves parallelism on the table) and sampling (logits are still read back to the host). What remains, and the plan for it, is in [docs/gpu-setup.md](docs/gpu-setup.md), with a per-kernel design plan in [docs/kernel-plan.md](docs/kernel-plan.md).

Every optimization is a commit with a measured before/after number, and CI fails if any figure quoted in the docs disagrees with the data the harness produced.

## Benchmark

Full tables, the protocol, and what each figure does and does not mean: **[aarnavnoble.github.io/llm-engine](https://aarnavnoble.github.io/llm-engine/)** (or [docs/results.md](docs/results.md)), generated from `bench/results/*.json` by `bench/report.py`, so the published numbers cannot drift from what the harness measured. The page records the hardware each figure came from, and any row without data behind it is rendered as *not measured* rather than estimated.

Reproduce everything on one machine with `scripts/reproduce.sh` (add `--quick` for a smoke check, which writes to a separate directory and leaves the published numbers alone).

Each row names the instance it was measured on, and that matters more than it
should. Full tables, with the protocol and what each figure does and does not
mean, are in [docs/results.md](docs/results.md).

| Result | Number | Measured on |
|---|---|---|
| KV slot utilization, paged vs contiguous with a declared cap | **99.0%** vs 54.8% (1.7% reserving full context) | hardware independent |
| Continuous vs static batching, TTFT p50 | **53x faster** (3,993 ms to 75 ms) | A40, pod isx98c0 |
| Continuous vs static batching, TTFT p95 | **92x faster** (7,628 ms to 83 ms) | A40, pod isx98c0 |
| Continuous vs static batching, throughput | **1.37x** (564 to 771 tok/s) | A40, pod isx98c0 |
| Continuous vs static batching, inter-token p50 | **1.69x worse** (23.1 ms to 39.1 ms) | A40, pod isx98c0 |
| Sharing K/V reads across a GQA group, attention kernel | **1.62x faster** (31.56 ms to 19.43 ms per step) | A40, pod isx98c0 |
| Sharing K/V reads across a GQA group, end to end | **1.27x** (771 to 980 tok/s) | A40, pod isx98c0 |
| Prefix caching, TTFT p50 on a shared 512-token prompt | **14.4x faster** (1,627 ms to 113 ms), 95.8% block hit ratio | A40, pod b0flqxf |
| Chunked prefill, worst decode stall with a 2,048-token prompt injected | 47.8 ms to **36.8 ms** | A40, pod b0flqxf |
| Continuous vs static batching, throughput | **1.47x** | 10-core Mac, CPU |
| Continuous vs static batching, TTFT p50 | **5.9x faster** (13,673 ms to 2,330 ms) | 10-core Mac, CPU |
| Admission watermark, preemption recompute waste | **17.1% to 4.7%** of all token work at 8,192 KV slots, 64 preemptions to 11 | hardware independent |
| vLLM on the same model and GPU | **5,927 tok/s** against this engine's 386 on that instance | A40, pod b0flqxf |

**The same code on two rented A40s differed by a factor of two.** Same GPU
model, same Xeon 6342, same vCPU allocation: 386 tok/s against 771, and
inter-token latency of 80.3 ms against 39.1 ms. The hardware label described
both correctly. The same continuous-versus-static comparison came out at 38x
on time-to-first-token on pod b0flqxf and 53x on pod isx98c0, which is why
both figures appear in this repository's history and only the second is in the
table above. The cause is unknown and one observation per machine is not
enough to find it, so every row names its instance and a ratio is only
meaningful between rows that share one. The vLLM gap of 15.4x is real on pod
b0flqxf, where both sides were measured; dividing it against the engine
numbers above would flatter this engine by about half.

That is also why the figures worth quoting from this project are ratios rather
than absolutes. 1.62x on a kernel and 53x on time-to-first-token survive the
instance changing. 980 tok/s does not.

Four results contradict what the design predicted, and they are in the table
rather than omitted from it.

**Continuous batching barely moves throughput and transforms latency.** 1.37x
against 53x. At 32 concurrent requests with a 0.5B model the card is already
saturated, so there is no idle capacity for better scheduling to reclaim; what
it buys is admitting a request on the next step rather than after the batch
drains. The CPU backend, which is not saturated, shows 1.47x on throughput.
Same engine, different bottleneck.

**It makes inter-token latency 1.69x worse,** because every resident sequence
advances on every step. That is the trade.

**The attention optimisation was 1.33x slower before it was 1.62x faster.**
Profiling put 87.6% of a decode step in attention, and the obvious reading was
the sevenfold redundant K/V read that grouped-query attention causes. Removing
it needs one block per (token, kv head) instead of per (token, query head),
which also cuts the grid from 448 blocks to 64 on a card with 84 SMs. The
kernel was parallelism-starved, not bandwidth-starved. Widening the block from
128 to 512 threads fixed it. "87.6% of time" says where the time goes, not what
limits it.

**Chunked prefill is nearly pointless on this hardware,** worth 1.3x. A
2,048-token prefill takes about 37 ms on an A40, so there is barely a stall
left to subtract. It would matter again with a larger model or longer prompts.

Per-kernel achieved bandwidth is still unmeasured: Nsight Compute needs
`NVreg_RestrictProfilingToAdminUsers=0`, a host kernel-module parameter a
rented container cannot set, and returns `ERR_NVGPUCTRPERM` without it. The
remaining optimisation rows are unmeasured because the optimisations are
unwritten. Both are rendered as *not measured* on the results page rather than
estimated, and `bench/report.py` fails rather than publishing a figure with no
data behind it.

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
                    kernels/      rmsnorm · rope · cache write · paged decode attention
                                  silu·up · embedding · bias and residual adds
                                  (cuBLAS for the GEMMs; prefill shares the decode
                                   kernel, and sampling is still done on the host)
```

## Watch the scheduler

```bash
./build/engine serve --model models/Qwen2.5-0.5B-Instruct   # then open http://localhost:8000
```

The engine serves a live view of its own scheduler at `/`. It is not a chat
UI: streaming text into a box demonstrates nothing an inference server does.
It shows the three mechanisms that are invisible from a terminal and hard to
describe in prose. Requests joining a batch that is already running, one per
second, rather than waiting for it to drain. Six prompts sharing one system
prompt, where the prefix is stored once and five of them skip prefill for it.
And forty-eight at once, past what the pool holds, where preemptions rise as
the scheduler evicts live sequences and recomputes them later.

Each button says what to watch before it runs.

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
- **Continuous batching** (`scheduler/`): finished sequences leave the batch at the step they finish; new ones join at the next step under a `max_num_batched_tokens` budget. Long prompts are prefilled in chunks, which cuts the worst decode stall 1.3x when a 2,048-token prompt arrives mid-flight. Admission keeps a watermark of free blocks so a new sequence cannot steal the block a resident one needs next, which cut preemption recompute waste from 17.1% to 4.7% of all token work. Out of memory → the youngest sequence is preempted and recomputed on resume. `--mode static` selects the static-batching baseline on the same code path.
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
