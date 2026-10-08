# Design notes

## Data flow of one step

```
Scheduler::schedule()        → StepInput { slices: [prefill(seq, start, len)…, decode(seq, start, 1)…] }
Model::forward(step, logits) → logits[num_logits × vocab], one row per slice with needs_logits
Sampler::sample_step()       → one token id per logits row
Scheduler::on_step_done()    → num_computed += len; publish full blocks to the prefix cache;
                                append token; stop checks; retire finished sequences (free blocks)
```

A slice is a contiguous token range of one sequence. A decode slice has `len = 1` and `start = num_tokens - 1`. A prefill slice covers a chunk of the prompt; only the chunk that ends at the last token has `needs_logits`. After a preemption the "prompt" that gets re-prefilled is prompt + everything generated so far — the sequence resumes with all positions recomputed, and RoPE positions come from the sequence, never from the step.

## KV cache

Physical layout (both backends): per layer, `[num_blocks × block_size][num_kv_heads][head_dim]` for K and for V. `slot(pos) = block_table[pos / 16] * 16 + pos % 16`. The CPU oracle reads through block tables exactly like the CUDA kernel so allocator bugs show up on the Mac.

Block lifecycle:

```
free ──allocate──▶ owned (ref ≥ 1) ──commit_computed (block full)──▶ owned + hashed
                        │                                                   │
                     free()                                              free()
                        ▼                                                   ▼
                      free                                        cached in LRU (ref 0)
                                                                    │              │
                                                            prefix hit         take_block() under pressure
                                                            (ref ← 1)          (evict: unhash, reuse)
```

Publication happens in `on_step_done`, after the forward pass that wrote the block's last slot. Publishing at allocation time would let a second request in the same step read K/V that the first request's kernel is still writing.

Duplicate content (two identical prompts prefilled in the same step): the first block to commit wins the hash; the second stays unhashed (goes straight back to the free list on release). Its chain is broken from that block on — a deliberate simplification.

Sharing only ever involves full blocks, so copy-on-write is never needed: a sequence can only append into its own, unshared, partial last block.

## Scheduler

- Budget: `max_num_batched_tokens` per step, decode tokens reserved first, remaining budget admits prefill chunks (`prefill_chunk_size`).
- Admission is FCFS; a preempted sequence goes back to the *head* of the queue. `starvation_wait_steps` is recorded per sequence for the metrics; there is no priority reordering beyond FCFS (documented non-feature).
- Preemption picks the youngest running sequence (most recently admitted) — the cheapest to recompute and the least likely to starve others. Recompute-on-resume rather than swap-to-host: simpler, correct, and on a 0.5B model recompute is cheap.
- Static mode: the batch is admitted together; members that finish keep their slot and their blocks (they stop computing, which is slightly *kind* to the baseline) until the last member finishes, then everything frees at once.

## CPU threading

GEMMs go to BLAS, which threads them already. Attention does not: it is a loop
over (token, head) where each item walks its own sequence's block table for its
own length, so `ThreadPool::parallel_for` splits that index space. Each item
writes only its own head slice, so there is no race and the result is
independent of the split, which `ENGINE_THREADS=1` versus `ENGINE_THREADS=8`
producing identical token ids confirms.

The pool is persistent, because the loop runs once per layer per step and
spawning threads per call would cost more than the work. It is also gated: the
model estimates the attention work first and stays single-threaded below a
threshold, since the pool costs a wake-up per layer per step and competes with
the BLAS threads. Threading unconditionally made short-context batches slower
while long contexts sped up several-fold, which is the shape the gate encodes.

## Numerics

- Weights are bf16 on disk. CPU path upcasts to fp32 and matches the PyTorch fp32 reference to ~1e-4 (tolerance 2e-3 relative on the residual stream). CUDA path uses fp16 weights/activations with fp32 accumulation; expect ~1e-2 on logits and occasional argmax flips on near-ties — the test tolerates a flip only when the reference top-2 gap is below the observed error.
- RoPE: HF `rotate_half` convention, pairs `(i, i + head_dim/2)`, `inv_freq = theta^(-2i/d)`, tables precomputed for `max_position_embeddings`.
- GQA: query head `h` reads KV head `h / (num_heads / num_kv_heads)`.
- Qwen2 has q/k/v bias; Llama does not. Loader treats each bias as optional.

## Architecture genericity

Every dimension comes from `config.json`, and that claim is tested rather than
asserted: the per-layer numerical comparison runs against both Qwen2.5-0.5B
(qwen2: q/k/v bias, tied embeddings, GQA 7, eps 1e-6, theta 1e6) and
TinyLlama-1.1B (llama: no bias, untied `lm_head`, GQA 8, eps 1e-5, theta 1e4).
TinyLlama was the first exercise of the untied-head path, and it passed
unmodified.

The tokenizer handles both BPE families that appear in `tokenizer.json`, chosen
by what the file declares:

- **ByteLevel** (GPT-2, Qwen): a regex pre-tokenizer splits the text, then every
  byte maps to a printable placeholder character.
- **Metaspace** (Llama, sentencepiece-derived): no pre-tokenizer at all. The
  normaliser prepends U+2581 and replaces every space with it, BPE runs over
  characters, and a character outside the vocabulary falls back to one `<0xNN>`
  token per UTF-8 byte. Decoding reverses that and strips the one leading space
  the normaliser added.

Both reproduce Hugging Face exactly on the corpus. The merge loop is shared;
merges are stored as `(left id, right id) -> (rank, merged id)` so neither path
has to concatenate token strings to find the result, which matters because
byte-fallback tokens do not concatenate meaningfully.

Special tokens are a model property too: Llama's post-processor inserts `<s>`
before every sequence and Qwen's does not, so `encode_for_generation` adds BOS
only when the tokenizer declares it. Prompts submitted as raw `prompt_token_ids`
skip this entirely, since the client controls those exactly.

Still not generic: chat templates. Only ChatML is implemented, so
`/v1/chat/completions` against TinyLlama returns 400 with
"only ChatML chat templates are implemented" rather than silently using the
wrong prompt format. `/v1/completions` works for both models.

## Invariances the tests pin

Three properties have to hold for paging and continuous batching to be safe,
and none of them is obvious from reading the code:

- **Batching invariance.** A token's logits must not depend on who else is in
  the step. Checked by running three sequences packed together against each one
  alone, and by putting a mid-generation decode token beside someone else's
  prefill chunk.
- **Chunk invariance.** Splitting a prompt changes how many tokens share a
  forward pass and how the block table is walked, but not the answer. Checked on
  a 492-token prompt spanning 31 blocks, at chunk sizes 16 (block-aligned), 128
  (spanning blocks) and 37 (coprime with the block size, so boundaries land
  mid-block).
- **Preemption invariance.** Recompute-on-resume discards the cache and
  re-prefills prompt plus everything generated so far. If positions came from
  anywhere other than the sequence itself, RoPE would be applied at the wrong
  offsets and the continuation would silently diverge. Checked by preempting
  after 1, 5 and 8 generated tokens and requiring the full continuation to match
  the uninterrupted run exactly.

The design notes had predicted a RoPE off-by-one after preemption as the most
likely hard bug. It never happened, because positions are read from the sequence
rather than tracked alongside it, but that is now a tested property instead of
an accident.

## Lessons recorded

- A missing test fixture must not read as a pass. The per-layer dumps are
  gitignored and regenerable, and the test originally `WARN`ed when they were
  absent, so after they were deleted the suite kept reporting green while the
  central correctness check silently did nothing. It now `SKIP`s, which is
  visible in the output.
- Reference dumps must control every `generation_config.json` knob: Qwen ships `repetition_penalty=1.1`, which `generate()` applies even under greedy decoding, and without an explicit `attention_mask` positions equal to `pad_token_id` are masked — chat prompts contain `<|im_end|>`.

## Not built (say so in interviews)

Speculative decoding, tensor parallelism, fp8/int4 weights, a custom GEMM, swap-to-CPU preemption, priority scheduling, multi-LoRA.
