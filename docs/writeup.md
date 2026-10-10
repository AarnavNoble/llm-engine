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

This is the CPU half of the project. The CUDA backend and the GPU throughput
numbers come later; everything below is measured, reproducible with
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
kernels will consume, so it rehearses the port.

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

The first table said paged allocation reached 98.8% slot utilization against
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
| contiguous, reserve prompt + cap | 55.5% | 37.3 |
| paged, static batching | 98.8% | 14.6 |
| paged, continuous batching | **98.8%** | **41.1** |

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
| 8,192 | 135 → 15 | 16.2% → **2.3%** |
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
   prefix cache hit ratio rose from 75% to its theoretical ceiling of 77.5%,
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
