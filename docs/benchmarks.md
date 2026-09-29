# Benchmarks: protocol, what is measurable where, and results so far

Two rules for this file. Every number states the hardware it came from, and a
comparison is only published when the baseline is configured to its own
advantage. A flattering benchmark is worth less than no benchmark.

## Protocol

- Prompts: 128 / 512 / 1024 tokens at 50 / 30 / 20 percent.
- Output: mean 128 tokens, **not** a fixed length (see below).
- Greedy decoding, `ignore_eos` so every request produces its full output.
- 32 concurrent requests, closed loop, warm up first, three runs, report the median.
- Each published row is a git tag; `scripts/bench_all.sh <tag>` reproduces it.

Output lengths must vary. With a single fixed output length every sequence
finishes on the same step, so nothing ever waits behind a longer neighbour and
static batching costs exactly nothing. Measured with a fixed length, static and
continuous batching came out within 1.5% of each other — an artifact of the
workload, not a property of the scheduler. `bench/loadgen.py --output-dist`
offers `fixed`, `geometric` and `bimodal`; the last is the interesting one, a
few long generations among many short ones.

## What can be measured without a GPU, and what cannot

**KV cache efficiency: hardware-independent.** Waste is a property of the
allocator, so `bench/kv_waste.cpp` measures it anywhere. It drives the real
`KVCacheManager` and `Scheduler` with a fake model; only the contiguous
baseline is simulated, because the engine never implements contiguous
allocation.

512 requests, 65,536 KV slots (4,096 blocks of 16), `max_num_seqs` 64,
geometric output lengths around 128:

| strategy | slot util | KV waste | resident seqs | decoding seqs |
|---|---|---|---|---|
| contiguous, reserve `max_model_len` (32,768) | 1.7% | 98.3% | 2.0 | 1.4 |
| contiguous, reserve prompt + cap, static | 44.4% | 55.6% | 51.4 | 12.4 |
| contiguous, reserve prompt + cap, continuous | 55.5% | 44.5% | 37.3 | 37.3 |
| paged, static batching | 98.8% | 1.2% | 63.7 | 14.6 |
| paged, continuous batching | **98.8%** | **1.2%** | 41.5 | **41.1** |

The two mechanisms separate cleanly. Paging is what fixes memory: 98.8% slot
utilization against 55.5% for the best contiguous variant, and 1.7% for an
allocator that must reserve the full context. Continuous batching is what fixes
occupancy: 41.1 sequences actually decoding per step against 14.6. The gap
between the *resident* and *decoding* columns is precisely the capacity static
batching burns on finished sequences that still hold their blocks.

Paged waste tracks half a block per sequence, as expected: 0.5% at block size
8, 1.1% at 16, 2.3% at 32.

**Throughput: needs the GPU, and here is why.** `CpuModel::forward` loops over
the step's slices and runs a separate forward pass for each one. Nothing is
batched, so the total work for N tokens is the same however those tokens are
scheduled, and tokens/s is by construction independent of the scheduler.
Measured on the Mac at 16 concurrent requests, static and continuous batching
both produced 12.58 tokens/s — identical to three significant figures, exactly
as the implementation predicts.

Continuous batching's throughput win comes from amortizing weight reads across
a batched GEMM. That only exists on the CUDA backend, so the throughput rows of
the table stay empty until the kernels are written. Publishing a CPU tokens/s
comparison as evidence for continuous batching would be meaningless.

**A side effect worth recording.** On a backend that does not batch, continuous
batching still changes *who waits*. Same workload, bimodal output lengths:

| | short third, e2e p50 | long third, e2e p50 |
|---|---|---|
| static batching | 11.06 s | 24.63 s |
| continuous batching | 10.57 s | 45.55 s |

Short requests finish sooner because they are released the step they complete
instead of waiting for the batch. Long requests finish much later because
continuous admission keeps more sequences resident, and a fixed compute budget
is now shared more ways. This is fair-sharing, not a throughput gain: on the
GPU the batch is close to free, so the long-request regression should largely
disappear. Whether it does is a measurement, not an assumption, and it is one
of the first things to check once the CUDA backend runs.

## Table status

| Row | Blocked on |
|---|---|
| HF transformers batch 1 and padded batch 32 | GPU (`bench/hf_baseline.py` is ready) |
| static batching, contiguous KV | GPU |
| static batching, paged KV | GPU |
| + continuous batching | GPU |
| + prefix caching | GPU |
| + fused kernels, optimized decode attention | kernels not written |
| + CUDA graphs | kernels not written |
| KV waste column | **done, above** |

## Reproducing

```bash
cmake -S . -B build -G Ninja && cmake --build build
./build/kv_waste --requests 512 --blocks 4096          # KV efficiency, any machine

./build/engine serve --model models/Qwen2.5-0.5B-Instruct --mode continuous &
python3 bench/loadgen.py --tag v0.3-continuous --concurrency 32 --num-requests 256
python3 bench/hf_baseline.py --tag hf-b1 --batch-size 1 --device cuda
python3 bench/plot.py
```
