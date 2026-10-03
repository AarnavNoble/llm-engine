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

**Prefix caching: measurable on CPU, because it removes work.** A cache hit
skips prefill for the shared blocks outright, so the saving is arithmetic
rather than a scheduling or batching effect. `scripts/bench_prefix.sh` runs 32
requests that share a 512-token system prompt and differ only in a 128-token
tail, once with the cache enabled and once with `--no-prefix-cache`, on
otherwise identical servers.

Mac, CPU backend, 8 concurrent, 8 output tokens:

| | cache off | cache on | change |
|---|---|---|---|
| TTFT p50 | 18,019 ms | 7,284 ms | **2.5x faster** |
| TTFT p95 | 20,973 ms | 10,267 ms | 2.0x faster |
| end-to-end p50 | 23.3 s | 10.2 s | 2.3x faster |
| wall clock | 93.9 s | 42.7 s | 2.2x faster |

Hit ratio was 75%: 960 of 1,280 full prompt blocks. The ceiling for this
workload is 77.5% — 32 of every 40 prompt blocks are shared, and the very
first request necessarily misses. The remaining gap is the cold start: with
eight concurrent requests, the first few begin prefilling before any block has
been published, so they cannot share with each other. Blocks are published as
each one fills, which is why only two requests missed rather than all eight.

The tokens/s figures in this experiment (2.7 to 6.0) are not a decode
throughput claim. Each request generates only 8 tokens against a 640-token
prompt, so the measurement is dominated by prefill; that is deliberate, since
prefill is what prefix caching changes.

**Chunked prefill: what it does and does not buy.** A long prompt must be
prefilled before it can generate. If the whole prompt goes into one forward
pass, every sequence already decoding waits for that pass and sees one enormous
gap between tokens. `scripts/bench_chunked_prefill.py` holds four streams
decoding, injects a 2,048-token prompt mid-flight, and measures the streams'
inter-token gaps around the injection, for several chunk sizes on otherwise
identical servers. Prefix caching is disabled so the prefill cost is real.

Mac, CPU backend:

| chunk | ITL p50 | ITL p95 | worst gap | injected prompt TTFT |
|---|---|---|---|---|
| 2048 (effectively unchunked) | 152 ms | 314 ms | **23,972 ms** | 24.38 s |
| 512 | 148 ms | 6,955 ms | 9,895 ms | 24.30 s |
| 128 | 148 ms | **2,556 ms** | **2,785 ms** | 25.00 s |

The worst-case stall falls 8.6x, from 24.0 s to 2.8 s, and the arriving
request's own TTFT is barely affected (24.4 s to 25.0 s, about 2.5%).

The p95 column moves the other way, and that is not a contradiction. Total
prefill work is conserved: chunking does not make the prompt cheaper, it splits
one catastrophic stall into many small ones. Unchunked there is exactly one
24-second gap, which barely registers in a p95 over roughly 150 samples; at
chunk 512 there are four gaps of several seconds each, so more of the
distribution is affected even though the worst case is far better. The metric
that matters to a user watching tokens appear is the worst gap, not p95, and
reporting p95 alone here would hide the entire effect.

Absolute numbers are CPU-bound and will shrink by orders of magnitude on the
GPU; what should carry over is the shape, with the worst stall bounded by the
chunk size rather than by the prompt length.

**Preemption is not free, and the engine was not measuring what it cost.**
`engine_preemptions_total` said how often a sequence was evicted but not how
much work that destroyed. Recompute-on-resume discards every token the victim
had computed, so the cost is counted directly as
`engine_recomputed_tokens_total` and reported by `kv_waste --sweep`.

Measuring it immediately exposed a design flaw. The engine admitted
optimistically and preempted reactively, so a fresh admission would steal the
block a resident sequence needed at its next boundary, and the pair would
thrash. Wasted work did not even fall monotonically with pool size: 1,024
blocks wasted more than 768. Holding the budget at 8,192 slots and varying only
the sequence cap showed the mechanism — past 32 the cap is irrelevant because
memory binds, yet going from 16 to 32 bought half a sequence of occupancy and
doubled the preemptions.

The fix is admission control rather than a bigger pool: refuse to admit a new
sequence unless enough free blocks remain for the resident sequences to each
reach their next block boundary, which is one block apiece. Reusable prefix
blocks cost nothing to adopt, so they do not count against the headroom, and
with nothing resident the watermark is zero, so a prompt that fits at all is
always admitted and the engine cannot deadlock. `SchedulerConfig::watermark_blocks`
defaults to this automatic policy; 0 restores the old behaviour.

256 requests, geometric outputs around 128, `max_num_seqs` 64, wasted work as a
fraction of all token work:

| blocks | KV slots | preemptions before → after | wasted work before → after |
|---|---|---|---|
| 256 | 4,096 | 86 → 30 | 20.6% → **8.1%** |
| 384 | 6,144 | 71 → 18 | 17.1% → **6.1%** |
| 512 | 8,192 | 71 → 11 | 16.9% → **3.2%** |
| 768 | 12,288 | 72 → 5 | 15.4% → **1.1%** |
| 1,024 | 16,384 | 84 → 1 | 18.3% → **0.1%** |
| 2,048 | 32,768 | 24 → 0 | 7.0% → **0.0%** |
| 4,096 | 65,536 | 0 → 0 | 0.0% → 0.0% |

The curve is now monotonic, as it should be, and the cost of the fix is almost
nothing: at 8,192 slots the sequences decoding per step went from 14.0 to 13.9,
a 0.7% loss of occupancy in exchange for 6.5x fewer preemptions. At the tightest
budget, where the engine is genuinely oversubscribed, preemption still happens —
the watermark reduces thrash, it cannot create memory that is not there.

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
