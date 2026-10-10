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

**Throughput: now measurable on CPU too, after the forward pass was batched.**
This section previously said throughput could not be measured here, and the
reason was an implementation detail rather than a law: `CpuModel::forward` ran a
separate forward pass per slice, so the work for N tokens was identical however
they were scheduled and tokens/s was scheduler-independent by construction.
Both modes measured 12.58 tokens/s, to three significant figures.

The forward pass now packs every slice in the step into one `[N, hidden]` batch,
so each layer runs one GEMM per projection instead of one per sequence.
Attention stays a loop over (token, head), because each token attends over its
own sequence's block table for its own length. That is the same packing the CUDA
backend needs, so this also rehearses the layout.

Throughput now scales with the batch, as it should:

| concurrent requests | tokens/s |
|---|---|
| 1 | 25.6 |
| 4 | 45.5 |
| 16 | 152.2 |

And the continuous-batching comparison finally says something. 48 requests,
bimodal output lengths, 16 concurrent, through the HTTP server:

| | static batching | continuous batching | change |
|---|---|---|---|
| tokens/s | 38.9 | 57.1 | **1.47x** |
| wall clock | 35.1 s | 24.0 s | 1.47x faster |
| TTFT p50 | 13,673 ms | 2,330 ms | **5.9x faster** |

The TTFT gap is the larger one and the easier to explain: under static batching a
new request cannot start until the entire current batch drains, so its first
token waits behind the longest generation in flight.

These are CPU numbers on a 0.5B model and they are not a prediction of GPU
throughput, where the arithmetic intensity of a batched GEMM is the whole point
and the gain should be larger. They do establish that the scheduler earns its
complexity, which could not be shown before.

Attention is threaded separately, since BLAS only covers the GEMMs. The benefit
depends entirely on context length, because attention cost grows with it while
the pool's overhead does not:

| | 1 thread | 8 threads |
|---|---|---|
| 2,048-token context, decode | 0.71 tok/s | **2.60 tok/s** (3.7x) |
| 30-token context, 16 concurrent | 108-149 tok/s | 140-152 tok/s |

The second row is noise in both columns, which is the point: below a work
threshold the model stays single-threaded, because threading unconditionally cost
short-context batches more in synchronisation and BLAS contention than it saved.
Thread count does not change the output; `ENGINE_THREADS=1` and `ENGINE_THREADS=8`
produce identical token ids.

Packing correctness is tested, not assumed: a sequence's logits must not depend
on who else is in the step. Two tests cover it, one comparing three sequences
packed together against each run alone, and one checking a mid-generation decode
token packed beside someone else's prefill chunk.

**Prefix caching: measurable on CPU, because it removes work.** A cache hit
skips prefill for the shared blocks outright, so the saving is arithmetic
rather than a scheduling or batching effect. `scripts/bench_prefix.sh` runs 32
requests that share a 512-token system prompt and differ only in a 128-token
tail, once with the cache enabled and once with `--no-prefix-cache`, on
otherwise identical servers.

Mac, CPU backend, 8 concurrent, 8 output tokens:

| | cache off | cache on | change |
|---|---|---|---|
| TTFT p50 | 20,118 ms | 7,268 ms | **2.8x faster** |
| TTFT p95 | 27,874 ms | 8,610 ms | 3.2x faster |
| end-to-end p50 | 24.4 s | 8.5 s | 2.9x faster |
| wall clock | 101.7 s | 35.0 s | 2.9x faster |

Hit ratio is 77.5%, 992 of 1,280 full prompt blocks, which is exactly the ceiling
for this workload: 32 of every 40 prompt blocks are shared and the very first
request must miss, giving 31 x 32 = 992. An earlier run of the same experiment
reached only 75%, because with the unbatched forward pass the first few
concurrent requests each began prefilling before any block had been published and
so could not share with one another. Packing the step shrank that cold-start
window to a single request.

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
| 2048 (effectively unchunked) | 92 ms | 561 ms | **23,203 ms** | 23.81 s |
| 512 | 89 ms | 6,813 ms | 9,750 ms | 23.63 s |
| 128 | 91 ms | **1,653 ms** | **1,969 ms** | 24.69 s |

The worst-case stall falls 11.8x, from 23.2 s to 2.0 s, and the arriving
request's own TTFT is barely affected (23.8 s to 24.7 s, about 3.6%). Note that
the injected prompt's own prefill time did not improve when the forward pass was
batched: a single 2,048-token prefill was already one slice, so there was nothing
to pack it with. Batching helps a step that contains several sequences, not a
step that contains one large one.

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
had computed, so the cost is counted as `engine_recomputed_tokens_total` and
reported by `kv_waste --sweep`, which now measures the admission watermark off
and on in the same run rather than quoting a number remembered from whenever it
was first taken.

512 requests, geometric outputs around 128, `max_num_seqs` 64. Wasted work is
recomputed tokens over recomputed plus delivered tokens:

| blocks | KV slots | preemptions, no watermark | wasted | preemptions, watermark | wasted |
|---|---|---|---|---|---|
| 256 | 4,096 | 162 | 19.2% | 65 | **9.1%** |
| 384 | 6,144 | 134 | 15.4% | 35 | **5.5%** |
| 512 | 8,192 | 135 | 16.2% | 15 | **2.3%** |
| 768 | 12,288 | 141 | 16.7% | 14 | **1.9%** |
| 1,024 | 16,384 | 150 | 17.0% | 2 | **0.1%** |
| 2,048 | 32,768 | 87 | 10.1% | 0 | **0.0%** |
| 4,096 | 65,536 | 0 | 0.0% | 0 | 0.0% |

Without the watermark the curve is not even monotonic: 1,024 blocks wasted more
than 768 did. That non-monotonicity was the clue. Holding memory at 8,192 slots
and varying only the sequence cap shows the mechanism:

| `max_num_seqs` | decoding, no watermark | preemptions | decoding, watermark | preemptions |
|---|---|---|---|---|
| 8 | 7.7 | 0 | 7.7 | 0 |
| 16 | 13.5 | 40 | 13.4 | **10** |
| 32 | 14.0 | 71 | 13.9 | **11** |
| 64 | 14.0 | 71 | 13.9 | **11** |

Past 32 the cap is irrelevant because memory binds. Between 16 and 32 it buys
half a sequence of occupancy and doubles the preemptions: admission was
optimistic and preemption reactive, so a new sequence took the block a resident
one needed at its next boundary and the pair thrashed.

The fix is admission control rather than a bigger pool: refuse to admit unless
enough free blocks remain for every resident sequence to cross one more block
boundary, which is one block apiece. A reusable prefix costs nothing to adopt so
it does not count against the headroom, and with nothing resident the watermark
is zero, so any prompt that fits is admitted and the engine cannot deadlock.
`SchedulerConfig::watermark_blocks` defaults to that policy; 0 restores the old
behaviour, which is what the left-hand columns above measure.

The cost is 0.7% of occupancy, 14.0 sequences decoding against 13.9. At the
tightest budget preemption still happens, because the watermark reduces thrash
and cannot invent memory that is not there.

**Queueing: strict FCFS versus bounded lookahead.** Admission serves the head
of the queue, so a large prompt that does not fit holds up smaller requests
behind it. `SchedulerConfig::admission_lookahead` lets the scheduler look that
many positions down the queue for something admissible, and
`starvation_wait_steps` bounds the unfairness: a request that has waited that
many rounds cannot be overtaken, so it blocks the queue until it fits.

Adversarial workload for FCFS — 192 requests, 90% of prompts at 64-96 tokens and
10% at 2,048, in a 4,096-slot pool, so a large prompt needs half the pool:

| lookahead | mean wait (steps) | drain time | preemptions | wasted work |
|---|---|---|---|---|
| 1 (strict FCFS) | 1,937 | 3,128 | 10 | **1.9%** |
| 4 | 1,653 | 2,890 | 24 | 8.5% |
| 16 | 1,653 | 2,890 | 24 | 8.5% |

Lookahead saturates at 4 positions and buys a 15% lower mean wait and a 7.6%
shorter drain, but it quadruples recompute waste from 1.9% to 8.5%, because
admitting more sequences into a tight pool is exactly what the watermark is
trying to avoid. Those two roughly cancel, so **strict FCFS stays the default**:
it is simpler, its latency is predictable, and it burns less compute. The knob
exists, is bounded, and is tested, but whether it pays depends on the relative
cost of recompute against idle capacity, which is a GPU measurement.

Large prompts do not starve under lookahead, which was the thing to check: at
lookahead 4 they waited 1,494 steps against 1,653 for the small ones.

Two bugs surfaced while building this experiment, and both had been hiding the
real behaviour. The simulation loop treated an empty step as end of work, so it
stopped at the first step where the scheduler admitted nothing and silently
reported results for a fraction of the workload. And a request whose prompt plus
`max_tokens` exceeds the whole KV pool was being admitted, generating one token,
then aborted when it could not grow: `Engine::validate` now rejects it at
submission with a 400 and an explanation, which is checked end to end.

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

## Keeping published numbers honest

Every number here is re-run when the code underneath it changes, and the shifts
are explained rather than quietly swapped. Batching the forward pass moved three
of them: continuous batching became measurable at all, the prefix hit ratio rose
from 75% to its 77.5% ceiling because the cold-start window shrank, and the
chunked-prefill stall improved from 8.6x to 11.8x because decode tokens now share
a step with the prefill chunk instead of running after it. The KV-efficiency
table is unaffected, because it runs against a fake model and measures the
allocator alone.

## Table status

| Row | Blocked on |
|---|---|
| HF transformers batch 1 and padded batch 32 | GPU (`bench/hf_baseline.py` is ready) |
| static batching, contiguous KV | GPU |
| static batching, paged KV | GPU |
| + continuous batching | GPU |
| + prefix caching | GPU |
| + fused kernels, optimized decode attention | kernels not written |
| + CUDA graphs | GPU, and graph capture not implemented |
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
