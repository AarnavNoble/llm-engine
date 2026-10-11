# Seven times my own benchmarks lied to me

I am building an LLM inference server from scratch — the thing that sits between
an API call and a GPU, holds the model's memory, decides which requests share
each forward pass, and streams tokens back. The parts you would recognise from
vLLM are all there: a paged KV cache, continuous batching, prefix caching, an
OpenAI-compatible streaming API. No PyTorch at runtime; weights are read
straight out of safetensors and the forward pass is my own code.

The interesting part has not been building it. It has been measuring it, because
almost every measurement I took was flattering me at first, and finding out why
taught me more about serving than the implementation did.

This is the CPU half of the project. The CUDA backend has since been written —
part two is about it — and only the GPU throughput numbers still come later;
everything below is measured, reproducible with
`scripts/reproduce.sh`, and the numbers on the results page are generated from
the harness output rather than typed in by hand.

---

## 1. Fixed output lengths made static batching look free

The first comparison I cared about was continuous batching against static
batching. Static batching admits a group of requests, runs them together, and
does not admit anyone new until every member finishes. Continuous batching
re-decides every step: finished requests leave immediately, new ones join
immediately.

I ran both. They came out within 1.5% of each other.

The workload was the problem. Every request asked for exactly 24 output tokens,
so every sequence in a static batch finished on the *same step*. Nobody ever
waited behind a longer neighbour, which means static batching's entire cost —
head-of-line blocking — had been designed out of the experiment.

Output lengths now come from a distribution, and the load generator offers
`fixed`, `geometric` and `bimodal`. The bimodal one, a few long generations
among many short ones, is where the difference lives.

**Lesson:** if your workload is uniform, you have accidentally benchmarked a
world where the thing you are measuring does not matter.

## 2. My throughput number could not move, by construction

With varying output lengths, the two modes still produced identical throughput:
12.58 tokens per second, to three significant figures. That is suspiciously
exact, and exactness is a clue.

The CPU forward pass ran one pass per *slice* — one per sequence in the step. So
the total work for N tokens was the same no matter how those tokens were
scheduled, and tokens per second was mathematically independent of the
scheduler. I had written in the design notes that "throughput needs a GPU, and
here is why," which was true of my implementation and not true of anything else.

So I packed the step: every slice is now flattened into one `[N, hidden]` batch,
and each layer runs one GEMM per projection instead of one per sequence.
Attention stays a loop over (token, head), because every token attends over its
own sequence's block table for its own length. That is also the layout the CUDA
kernels consume, so it turned out to be the port rehearsed in advance.

Throughput then scaled — 25.6 tokens/s at one concurrent request, 45.5 at four,
152.2 at sixteen — and the comparison finally said something:

| | static | continuous |
|---|---|---|
| tokens/s | 38.9 | **57.1** |
| TTFT p50 | 13,673 ms | **2,330 ms** |

**Lesson:** when a number refuses to move, suspect the measuring apparatus
before the thing being measured. "This cannot be measured here" deserves one
more round of why.

## 3. I counted finished work as live work

KV memory efficiency is a property of the allocator, not of the hardware, so I
could measure it on a laptop: drive the real block allocator and scheduler with
a fake model, and compare against a simulated contiguous allocator on the same
workload and memory budget.

The first table said paged allocation reached 99.0% slot utilization against
87.9% for contiguous — a thin win, and the static and continuous rows were
*identical*, which made no sense.

Two bugs. The first was lesson 1 again: fixed output lengths. The second was in
my simulator. Under static batching a sequence that has finished keeps its slot
to emulate padded generation, and my code was still counting those sequences as
producing tokens and still growing their length. The baseline appeared to keep
working after it had stopped.

Once finished sequences stopped counting as active, the two mechanisms separated
cleanly:

| strategy | slot utilization | sequences decoding |
|---|---|---|
| contiguous, reserve full context | 1.7% | 1.4 |
| contiguous, reserve prompt + cap | 54.8% | 20.2 |
| paged, static batching | 98.6% | 11.7 |
| paged, continuous batching | **99.0%** | **22.9** |

Paging is what fixes memory. Continuous batching is what fixes occupancy. The
gap between *resident* and *decoding* is exactly the capacity static batching
burns on finished sequences that still hold their blocks.

**Lesson:** a baseline is a piece of code you wrote, and it deserves the same
suspicion as the thing it is a baseline for.

## 4. I had handicapped the baseline

In the same table, paged static batching held only 13 sequences. That seemed
low, and it was: my static mode closed the batch after a single step's token
budget, so it admitted roughly `budget / prompt_length` sequences and then
refused everyone until the batch drained.

Real static batching fills the batch — up to the sequence limit or until memory
runs out — and *then* runs it to completion. Filling it takes several steps
because one step's token budget rarely covers every prompt. I fixed that, which
made the baseline stronger and my win smaller. That is the correct direction for
a fix to move a number.

**Lesson:** if the baseline is the part of the system you spent the least time
on, your comparison is measuring your attention, not your design.

## 5. The simulation stopped early and reported anyway

While building a queueing experiment I noticed a number that looked too clean.
The simulation loop treated an empty step — a step where the scheduler admitted
nothing — as "work is finished" and broke out. So the run ended at the first
moment of memory pressure and confidently printed results for a fraction of the
workload. In one configuration it reported numbers after 2 of 192 requests had
completed.

The loops now continue until progress genuinely stops, and the queueing
experiment asserts that *every* request finished with reason `length`. A silent
abort now fails the run instead of skewing it.

That assertion immediately paid for itself by catching a real bug: a request
whose prompt plus `max_tokens` exceeded the whole block pool was being admitted,
generating one token, hitting the end of the pool, getting preempted, and then
being quietly aborted because it could never be re-admitted. The client saw a
truncated stream with no explanation. It is now rejected at submission with a
400 and the arithmetic.

**Lesson:** make your harness assert what it assumes. "The run completed" is an
assumption.

## 6. A metric I did not have was hiding a design flaw

The engine counted preemptions but not what they cost. Preemption here is
recompute-on-resume: the victim's KV cache is dropped and it re-prefills its
prompt plus everything it had generated. The count tells you how often that
happened; it does not tell you how much work was destroyed.

So I added `engine_recomputed_tokens_total` and swept the pool size. Wasted work
ran from 19.2% of all token work at the tightest budget down to zero at the
loosest — but it was not monotonic. 1,024 blocks wasted more than 768 did.

Non-monotonicity in a quantity that should be monotone is a bug telling you
where it lives. Holding memory fixed and varying only the sequence cap showed
the mechanism:

| `max_num_seqs` | sequences decoding | preemptions |
|---|---|---|
| 8 | 7.7 | **0** |
| 16 | 13.5 | 40 |
| 32 | 14.0 | 71 |
| 64 | 14.0 | 71 |

Past 32 the cap is irrelevant because memory binds. Between 16 and 32, raising
it buys half a sequence of occupancy and *doubles* the preemptions. The engine
admitted optimistically and preempted reactively, so a new sequence would take
the very block a resident one needed at its next 16-token boundary, and the pair
would thrash.

The fix is admission control, not a bigger pool: refuse to admit unless enough
free blocks remain for every resident sequence to cross one more boundary — one
block each. Two details mattered. A reusable prefix costs nothing to adopt, so
it must not count against the headroom, or prefix caching would be throttled by
its own success. And with nothing resident the watermark is zero, so any prompt
that fits is admitted and the engine cannot deadlock.

| KV slots | preemptions | wasted work |
|---|---|---|
| 4,096 | 162 → 65 | 19.2% → **9.1%** |
| 8,192 | 64 → 11 | 17.1% → **4.7%** |
| 16,384 | 150 → 2 | 17.0% → **0.1%** |

Monotonic now, at a 0.7% occupancy cost. Both columns come from the same run, so
the comparison stays reproducible instead of becoming a number I remember.

**Lesson:** you cannot fix what you are not counting, and a counter without a
cost attached is only half a metric.

## 7. Two of the bugs were in the reference

The forward pass is checked against Hugging Face transformers after every single
layer. Twice, the engine and the reference disagreed and the engine was right.

Qwen's `generation_config.json` sets `repetition_penalty: 1.1`, and
`generate()` applies it even under greedy decoding, so my "greedy" ground truth
was not greedy. And without an explicit `attention_mask`, `generate()` masks
every position equal to `pad_token_id` — which, for a chat-templated prompt
containing `<|im_end|>`, silently deletes tokens from the middle of the prompt.

**Lesson:** "it disagrees with PyTorch" is a hypothesis, not a verdict. Control
every default in the reference, or the oracle is the thing with the bug.

---

## What I would take from this

Three habits, in order of how much they have been worth:

1. **Measure things that cannot move, to find out why.** Two of the biggest
   improvements in the project came from a number that stubbornly refused to
   change.
2. **Make the harness assert its assumptions.** Every one of these lies survived
   because something was reported rather than checked.
3. **Re-run published numbers when the code beneath them changes, and explain
   the shifts.** Batching the forward pass moved three published results. The
   prefix cache hit ratio rose to its theoretical ceiling of 95.8%,
   because packing the step shrank the cold-start window to a single request.
   That is a satisfying explanation, and I would have missed it by quietly
   swapping the figure.

A randomised stress test over the scheduler and allocator — random pool sizes,
budgets, chunk sizes, watermarks, continuous and static — found two more bugs
within seconds of being written, including one where static batching resurrected
a *completed* request and let it generate past its own `max_tokens`. 6,000
iterations are clean now, which I trust about as far as 6,000 iterations deserve.

The one feature I implemented and then chose not to enable: admission lookahead,
where the scheduler may skip past a head-of-queue request that does not fit. It
cuts mean queueing delay 15% and quadruples recompute waste. It ships off by
default, with the measurement written down, because "we built it and measured it
as not worth it" is a better answer than either enabling it or never trying.

Source, full tables and the reproduction script:
[github.com/AarnavNoble/llm-engine](https://github.com/AarnavNoble/llm-engine).

---

# Part two: the CUDA backend

The second half of the project is the backend that was the reason for all of it:
a device-resident forward pass with hand-written kernels for everything except
the GEMMs, over the same paged KV cache the CPU backend uses.

The honest status is that this half is written and verified but not yet timed.
The kernels are gated against the CPU oracle, including a case built
specifically to catch the one mistake that makes paged attention not paged. The
throughput rows are being measured as I write this, so every figure that would
have to come from a GPU appears below as `TODO(gpu)`. Part one is about
benchmarks that flattered me; it would be a poor sequel to fill this half with
numbers I expect rather than numbers I have.

## I ported the plumbing before I wrote a single kernel

The first CUDA commit is not a kernel. It is a host-side object that owns device
memory, uploads every weight — fp16 for the matrices, fp32 for the norms and
biases, because those are tiny and the reference uses them in fp32 — and reports
how much VRAM that took against what the card has. `forward()` still threw.

The commit after that one runs an actual forward pass, with the projections on
the device through cuBLAS and *every other operation routed back through the
reference implementations in `ops.h`*: RMSNorm, RoPE, the fused silu-mul and
attention all ran on the host, with the activations converting fp16 to fp32 and
back at each boundary, twice per operation. The KV cache stayed in host memory,
but in the layout the kernel would later want. Embedding rows were fetched with
a `cudaMemcpy` rather than a gather kernel.

That version is slow by construction, and it is the point. A port has two
distinct bodies of new code in it — the plumbing and the maths — and if both
arrive together then every disagreement with the oracle has two candidate
causes. The round-trip version proves the plumbing while the only new code *is*
plumbing: the packed step layout, the strides, the weight upload, the transpose
convention, the residual adds, the logits gather, the slot arithmetic. Once that
agrees with the CPU backend, each reference call is replaced by one kernel at a
time, and a disagreement after that has exactly one suspect.

The class of bug this makes impossible to hide is the compensating pair: a
kernel that is wrong in a way the plumbing happens to cancel, or a stride error
that only shows up through one particular operation. Those are the bugs that
survive a working end-to-end test, and the way to not have them is to never have
two unverified things in the pass at once.

The gate itself is `tests/test_cuda_model.cpp`. The CPU model exposes per-layer
taps and can be checked layer by layer; the CUDA model cannot, so it is checked
on what is observable from outside: final logits against the same PyTorch dumps
the CPU backend uses, argmax agreement, and a greedy continuation driven through
the real scheduler that must produce token-for-token the same output as the CPU
backend. The logits tolerance is looser than the CPU backend's on purpose —
tightening it would mean the kernels were not actually running in fp16 — and one
argmax flip across the prompt set is allowed, because an occasional flip on a
near-tie is expected in fp16 and a systematic one is not. Each kernel also has a
per-operation golden tensor dumped by `golden_dump`, so it can be checked alone
before it is checked in context.

**Lesson:** a deliberately slow intermediate version is worth building, because
its value is not what it does, it is that it contains only one new thing.

## The transpose, worked out once, on a hand-computed 2x3

Everything in this project is row-major; cuBLAS is column-major. There is no
flag for that, so the convention has to be chosen once and then be right
everywhere. The wrapper is five lines and deserves the space:

```cpp
// C[m,n] = A[m,k] * B[n,k]^T, all row-major, fp16 data with fp32 accumulate.
cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
             B, CUDA_R_16F, k, A, CUDA_R_16F, k, &beta,
             C, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

The reasoning runs backwards from the output. A row-major `[m,n]` buffer is,
byte for byte, a column-major `[n,m]` buffer — which is to say it is C^T as far
as cuBLAS is concerned. So do not ask cuBLAS for C; ask it for C^T, which lands
in exactly the bytes wanted. Transposing `C = A · B^T` gives `C^T = B · A^T`, so
the first operand is B and the second is A, which is why the call reads with the
weight first and the activations second, and why the output dimensions are
`n, m, k` rather than `m, n, k`.

The op flags follow from the same substitution. B is row-major `[n,k]`, which
cuBLAS reads as column-major `[k,n]`; the product wants B itself, so that view
has to be transposed: `CUBLAS_OP_T`. A is row-major `[m,k]`, which cuBLAS reads
as column-major `[k,m]`, and that *is* A^T, which is what the product wants
already: `CUBLAS_OP_N`. The three leading dimensions are `k`, `k`, `n` — each
one the row length of the row-major buffer, which is the column stride of the
transposed view cuBLAS is reading.

Data is fp16 and the compute type is `CUBLAS_COMPUTE_32F`. Accumulating a
896-term dot product in fp16 loses enough precision to look like a logic bug,
which is the same reason the reductions inside the kernels accumulate in float.

This was checked against a hand-computed 2x3 case on the device before anything
depended on it. That is not caution for its own sake: a wrong transpose produces
output of the right shape and plausible magnitude, and it is indistinguishable
from a wrong kernel. Had it been left unverified it would have been found three
kernels later, and the three kernels would have been suspected first.

## Seven kernels, and which wrong version of each one still passes

There are three decomposition shapes across the seven, and almost no cleverness
in any of them. What took the time in each case was working out which incorrect
implementation would have passed the test I was about to write.

**`embedding.cu`** — one block per token, threads striding the row, so both the
read and the write stay coalesced. Written first on purpose: it is the head of
the chain, and if it is wrong then everything downstream is wrong too, so the
golden tensor for the embedding output is the cheapest possible early signal.
The index arithmetic goes through `size_t` rather than `int`, because
`ids[token] * hidden` is a product of a vocabulary size and a hidden size and
has no business being computed in 32 bits.

**`rmsnorm.cu`** — one block per row, 256 threads, and the reason it is worth
reading is the reduction, because the same three-level pattern reappears inside
attention. 896 values have to be summed into one number that all 256 threads
then need. Level one: each thread sums the squares of its own strided slice.
Level two: `__shfl_down_sync` halves the lane offset five times, so the 32 lanes
of a warp sum into lane 0 without touching memory at all. Level three: the eight
warps hand their partials through a shared array, warp 0 reduces those, and the
scale goes back out through shared memory because thread 0 holds it and every
thread needs it. Every shared write that another thread reads has a
`__syncthreads()` in front of it. The traps are precision and placement: fp32
accumulation is not optional at 896 terms, and `eps` belongs inside the square
root rather than added to its result.

**`silu_mul.cu`** — a grid-stride loop, the other standard decomposition: the
grid is sized to the device rather than to the data, capped at 4096 blocks, and
each thread walks the array in strides of the whole grid, which keeps the launch
shape fixed for any N and every warp's accesses contiguous. The fusion is the
substance. As two kernels — silu, then multiply — the data crosses memory three
times; fused it is two reads and one write. At `intermediate` 4864 and 24 layers
that is tens of megabytes of traffic per step, and the operation is entirely
bandwidth-bound, so traffic is the only thing that matters. The sigmoid is
computed in float: `__expf` on a half loses accuracy exactly where it matters,
around zero, and the reference uses the float form, so matching it is what keeps
the comparison meaningful rather than being an improvement on it.

**`rope.cu`** — one block per (token, head), one thread per rotation pair. At
`head_dim` 64 that is 32 threads, exactly one warp, so no barrier is needed:
each pair is independent and nothing is shared. Three traps here, and only one
of them fails loudly.

The pairing is Hugging Face's `rotate_half` convention, where element `i` pairs
with `i + head_dim/2`, not with its neighbour. Pairing `i` with `i+1` is the
natural guess, and it produces perfectly plausible output that degrades quality
subtly rather than failing: a text sample looks fine, and only the golden tensor
catches it.

The position comes from an array uploaded per step, not from the token's index
within the step. On a simple prefill those two agree, which is to say they agree
in the first test anyone writes. They diverge the moment a prompt is chunked, or
a sequence is preempted and recomputed — because preemption here is
recompute-on-resume, so the "prompt" being re-prefilled is the prompt plus
everything generated so far, at its original positions. Deriving the position
inside the kernel would have worked until the exact circumstances under which it
must not.

V is never rotated. A kernel that rotated V would still pass a Q-only check, so
the golden test asserts V is untouched rather than only asserting Q is right.

One more, small: both halves are read before either is written, since the second
write would otherwise clobber an input of the first.

**`store_kv.cu`** — scatters this step's K and V rows into their cache slots. The
slots are computed on the host, `block_table[pos / block_size] * block_size +
pos % block_size`, the same arithmetic the CPU backend uses, so no kernel needs
to know what a block table is except the one that walks history rather than
writing one row.

**`attention_decode.cu`** — the kernel the project exists for. Grid is (token,
query head), 64 threads, with shared memory holding the query vector, one
accumulator row per thread, and a running max and sum per thread; 64 threads
times a 128-element head dimension times 4 bytes is 32 KiB, which is what bounds
both constants.

Three things make it different from textbook attention.

The history is *paged*. Positions are contiguous logically and scattered
physically, so every access goes through the block table: the kernel takes the
concatenated tables for the step plus a per-sequence offset, and indexes
`table[p / block_size]`. A kernel that computes the slot directly from `p` works
perfectly whenever blocks happen to have been allocated in order, which is
exactly what a freshly allocated sequence gives you, which is exactly what a
first test does. This is the one claim the project is actually about, so it gets
its own test: allocate forty single-block sequences, release every other one so
the free list hands back a shuffled set of ids, allocate the longest prompt in
the reference set into the resulting fragmented pool, `REQUIRE` that the block
table really is not consecutive — assert the premise, not just the conclusion —
and only then compare the two backends on it. Without that test the paged part
of paged attention is untested, and every other case in the file passes with the
bug in place.

The softmax is *online*. The score row is never materialised, because at a few
thousand positions it does not fit anywhere useful. A running max and sum are
carried instead, and everything accumulated so far is rescaled by
`exp(m_old - m_new)` whenever a larger score appears. That is the FlashAttention
rescaling rule, and it is also what keeps long contexts numerically stable. The
running max starts at `-FLT_MAX` rather than negative infinity, and the rescale
factor is special-cased at that sentinel, so the arithmetic never evaluates
`inf - inf`.

The work is split *by position*, not by dimension. Each thread owns a stride of
the history and keeps its own `(max, sum, accumulator)` triple; at the end the
partials are merged pairwise with the same rescaling rule that the inner loop
uses. Splitting by dimension instead would need a reduction at every position
rather than one at the end. A thread that saw no positions at all still holds
`-FLT_MAX` and contributes nothing, which the merge guards make explicit rather
than leaving to how `exp` of a large negative number happens to behave.

The GQA mapping is `head / group`, where `group = q_heads / kv_heads`. With 14
query heads over 2 KV heads, `head % group` also compiles, also runs, and is
wrong. Both produce output; the golden tensor distinguishes them and a text
sample often does not.

**`elementwise.cu`** — the residual add and the bias add. Neither is interesting
and both are necessary: once every other operation is a kernel, doing these two
on the host means copying the whole activation tensor across PCIe twice per
layer, which costs far more than the addition does. The bias is kept in fp32
because the reference adds it in fp32, and rounding it to half first would show
up in the comparison as an unexplained discrepancy.

What the kernels do *not* include is prefill attention and sampling, which is
the last section below.

**Lesson:** for every one of these, the arithmetic was the easy part. The work
was identifying the plausible wrong version — pair with `i+1`, `head % group`,
slot from position, rotate V, position from the step index — and then writing the
test that separates it from the right one. A test that only the correct
implementation passes is a different artefact from a test that passes.

## The step: one upload, one synchronise, nothing in between

Once all seven existed, the staging that let each one be introduced on its own
came out. Activations are now allocated once through a `Scratch<T>` — a device
buffer that grows on demand and is then reused — because allocating inside
`forward()` on every step is the classic way to make a GPU program mysteriously
slow. The fp32 conversions at every operation boundary went with it.

Per step, the only things crossing the bus are a handful of small index arrays
in and the logits out. `upload_step_indices` builds them from the flattened
step: the slot each token writes to, which sequence each token belongs to, the
concatenated block tables with a per-sequence offset, the positions, and the
token ids. Block tables are concatenated once per distinct sequence rather than
once per token, since many tokens in a packed step share a sequence.

Two ordering details matter inside the layer. K and V for *every* token in the
step are written to the cache before any attention runs, which is what lets
tokens that arrived together attend to each other and makes a mixed
prefill-and-decode step correct. And only the rows that actually need logits go
through the output head: they are gathered device to device first, so nothing
crosses the bus until the logits themselves.

Synchronisation is one point per step. While kernels were landing one at a time
there was a synchronise after each launch, which exists to attribute a fault to
the kernel that caused it — CUDA reports errors asynchronously, so an unchecked
failure in one kernel otherwise surfaces as wrong numbers inside a different
one. Once the pass was stable those came out, leaving `cudaGetLastError` plus a
single `cudaDeviceSynchronize` at the end of the step, which lets the launches
overlap. Every CUDA and cuBLAS call goes through a macro that reports file, line
and the error string, from the first line of code in the file.

The cost of a step, and how much of it is attention: **36.05 ms on an A40 at
32 concurrent requests, and 87.6% of it is attention** -- 31.57 ms of the 36.05,
measured with CUDA events over 728 steps. The rest is barely worth naming: the
MLP is 5.8%, the QKV projection 2.8%, the head 1.4%, RoPE and the cache write
0.6% between them, the embedding 0.03%.

That number exists because the obvious guess was wrong. The vLLM comparison
had localised the inter-token deficit to per-step cost, and the natural
inference from there was the host: every step copies 32 x 151,936 fp16 logits
back and widens them one value at a time with a branchy software conversion,
which is easy to estimate at tens of milliseconds and easy to believe. Acting
on that would have meant writing a device sampler to optimise something worth
1.4% of device time.

It also bounds the host cost rather than dismissing it. 36 ms of device time
against roughly 80 ms of measured inter-token latency leaves about half the
step outside the kernels, and that half is the host-side work. Both are real.
Attention is simply the larger single item and the one a kernel can address.

Why it is slow is legible in the kernel rather than mysterious: one block per
(token, head), 64 threads, scalar fp32 accumulation, no tensor cores, and with
GQA group 7 each of seven query heads re-reads the same kv head's history from
global memory independently. One block per (sequence, kv_head) serving all
seven removes a sevenfold redundant read of the largest structure in the step.

Nsight Compute would have added achieved bandwidth per kernel and does not run
on a rented pod: it needs `NVreg_RestrictProfilingToAdminUsers=0`, a host
kernel-module parameter a container cannot set, and returns `ERR_NVGPUCTRPERM`
without it. CUDA events need no privileges and answered the question that
actually decided what to do next.

## `--backend cuda` ran on the CPU and reported success

This is the worst bug in the project, and it was found by an audit rather than by
anything failing.

Before renting a GPU I went looking for failure modes that would waste rented
time. `--backend cuda`, on a binary built without the CUDA backend compiled in,
fell back to the CPU backend and ran happily to completion. Nothing printed a
warning. Nothing downstream could tell: the benchmark harness records the
backend it was asked for, so forgetting `-DENGINE_CUDA=ON` on the GPU box would
have produced an entire results table labelled `cuda` containing CPU numbers,
and the only hint would have been that the numbers were low.

The whole premise of this write-up is that no published figure lacks data behind
it. A silent fallback is the single bug that can violate that premise while
every individual number remains correctly measured and correctly reported — it
is just a measurement of something else.

Backend selection is now explicit. `cuda` without the backend compiled in is an
error that states how to fix it, an unknown backend name is rejected rather than
quietly treated as `cpu`, and `scripts/reproduce.sh` proves the binary really
has the backend before it runs anything labelled with it.

**Lesson:** a fallback is a decision to produce a plausible result instead of an
error. Anywhere a label and a measurement can drift apart, the label has to be
checked against the thing it names.

## Two build failures, both found before the clock was running

Neither of these is interesting as a bug. They are here because on rented
hardware the expensive failures are the boring ones, and both would have
happened in the first minutes of the first GPU session.

The static-library link cycle: `engine_core` calls `make_cuda_model`, which
lives in the CUDA target, while the CUDA target needs `engine_core`'s headers.
Two static archives in that cycle do not resolve in a single link pass, and it
would have surfaced as an unresolved symbol. `engine_cuda` is now an OBJECT
library, so its objects go straight into the final binaries instead of into an
archive that has to be ordered against another one.

The missing define: `cuda_kernels.h` guards its declarations with `ENGINE_CUDA`,
and the define was set only on `engine_core`. So the header was *empty* inside
`cuda_model.cu`, and the first kernel call failed to compile. Both targets need
it — `engine_core` to choose a backend, `engine_cuda` to see the launcher
declarations. In the same area, `CMAKE_CUDA_ARCHITECTURES` has to be set before
`enable_language(CUDA)`, because that call compiles a test program and bakes in
whatever architecture it finds.

Earlier still, CMake listed the `.cu` files before any of them existed, so
`-DENGINE_CUDA=ON` — the flag the README advertised — failed with a missing-file
error. Anyone following the documented GPU instructions hit a confusing build
break rather than a statement of where the project stood. The fix was to check
for the sources at configure time and say the backend was not implemented; that
fix is the subject of the second half of the next section, because it did not
measure what it claimed to.

**Lesson:** rehearse the build on the machine you have before you pay for the
machine you don't.

## Two guards against dishonest output, both of them broken

These two belong together, and they are worse than any kernel bug in this half.
Each is a mechanism that exists specifically to stop this write-up from
containing a claim with nothing behind it, and neither was doing that.

The first is a harness bug in the same family as part one's. `reproduce.sh`
derived
its result tags from the backend name, so a GPU run would write
`cuda-static.json` and `cuda-continuous.json`, while `report.py` asks for
`gpu-static` and `gpu-continuous`. The measurements would have been real,
correct, and on disk. Nothing would have rendered them.

The failure mode is what makes it worth recording: a full publishing run would
have written a headline table with *every row reading "not measured"*, exited
successfully, and mentioned the discrepancy only inside a parenthetical count.
After a benchmark session on a rented GPU, that is the output.

`report.py` now tracks every tag any table asks for and fails if a results file
exists under a tag that nothing renders. Measuring something and then discarding
it is a harness bug, and it should not be distinguishable from success only by
reading a count. Separately, the hardware a run was measured on is now recorded
by the harness into the data rather than written into the prose by hand, because
a throughput table that does not say which card produced it is not a result, and
hand-written hardware notes are how documentation comes to disagree with its own
data.

The second is the configure-time check from the previous section — the one added
so that `-DENGINE_CUDA=ON` would explain that the backend was unwritten instead
of failing with a missing-file error. It verified that every source in the CUDA
list existed, and reported a complete backend when they all did.

`attention_prefill.cu` and `sampling.cu` sat in that list for months as
single-line comments pointing at the kernel plan. A one-line comment satisfies an
existence check perfectly and compiles to an empty translation unit, so the build
succeeded, the check passed, and the thing it was built to detect — two of the
backend's kernels not existing — was exactly the state it was in. The guard was
not a weak signal; it was reporting the opposite of the truth, and it had been
written by someone who knew both files were stubs.

It has been removed rather than strengthened. A file listing cannot answer
whether a backend works, and any version of that check I could write would be
some other proxy for the same question. The question already has an answer that
cannot be faked by an empty file: the oracle test, which compares the CUDA
backend's output against the CPU backend's and fails if a kernel is missing,
wrong, or doing nothing.

**Lesson:** part one's version of this was a simulation that stopped early and
reported anyway. These are a pipeline that measured correctly and published
nothing, and a check whose subject was not what it was testing. All three
passed. A guard against false claims is itself a claim, and nothing was checking
it.

## What is deliberately not done

**No prefill kernel.** Prefill tokens go through the decode kernel. That is
correct rather than approximate: every token in a packed step carries its own
position and its own sequence, so a prefill token attends over `0..pos` through
its own block table, and the chunk-invariance tests cover it. What it leaves on
the table is parallelism. A chunk of prefill tokens all read the same sequence's
history, and a tiled kernel would load each tile of K and V once for a whole
block of queries instead of once per (token, head) pair. `kernel-plan.md` keeps
two acceptable implementations for it, one of which is explicitly the slower,
less interesting one, because prefill is not the steady-state hot path. The cost
of not having it: `TODO(gpu)`.

**Sampling is still on the host.** Logits are copied back and sampled on the
CPU, which means one row of 151,936 values per logits-producing sequence crosses
the bus every step. At large batch sizes that is noise against the GEMM; at
batch 1 it is not. A device-side greedy argmax has to tie-break on the lowest
index to match the host sampler exactly, and device top-p has to reproduce the
host's per-request reseeding, since `seed` is part of the API contract and the
sampler tests pin it. Current cost of the readback: `TODO(gpu)`.

**`store_kv` is not fused into RoPE.** K is written to a temporary and then read
back to be scattered, where a fused kernel would write it exactly once. The
saving is real and the reason it is still separate is that it belongs in the
benchmark table as a measured before and after rather than as an assumption:
`TODO(gpu)`.

**GQA sharing is not exploited.** Seven query heads share each KV head, and
seven blocks therefore read the same K and V. Correct, and wasteful; one block
per (token, kv\_head) handling all seven query heads reads that history once.
`TODO(gpu)`.

**No vectorised 16-byte loads, no split-K, no CUDA graphs.** Those have rows in
the results table, and until they are measured those rows read "not measured",
which is what they should read.

## What the port taught me

Three things, none of which I would have predicted before starting.

1. **The CPU backend's value was mostly as something to disagree with.** It is
   slow and it will never serve anything, but it is an fp32 oracle that reads
   through block tables with the same arithmetic the kernel uses, which means an
   allocator bug or a slot bug shows up on a laptop. Every kernel landed against
   it one at a time, and none of them was ever debugged alongside something else
   that was also new.
2. **Writing the test is the hard half.** Each kernel had a plausible wrong
   version that passed every test I had: slot from position, `head % group`,
   pair with `i+1`, rotate V, position from the step index. The fragmented
   block-table test exists because the correct version and the wrong version are
   indistinguishable on a freshly allocated sequence.
3. **The expensive bugs in this half were not in the kernels.** A wrong kernel
   fails a comparison. A silent CPU fallback, a results tag nothing reads, a
   build check whose subject was a file list rather than a backend, and a label
   that does not have to match what was measured all produce a document with
   numbers in it instead. Those are worse, and nothing in the numerical test
   suite has any opinion about them.

The GPU throughput table, the per-kernel bandwidth against the card's peak, and
a vLLM reference row on the same hardware are the rows that remain. The harness
is ready for them and the kernels are gated against the oracle; what is missing
is a measurement, and until it exists the honest thing to publish is the empty
cell.
