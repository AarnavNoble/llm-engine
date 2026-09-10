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

## Numerics

- Weights are bf16 on disk. CPU path upcasts to fp32 and matches the PyTorch fp32 reference to ~1e-4 (tolerance 2e-3 relative on the residual stream). CUDA path uses fp16 weights/activations with fp32 accumulation; expect ~1e-2 on logits and occasional argmax flips on near-ties — the test tolerates a flip only when the reference top-2 gap is below the observed error.
- RoPE: HF `rotate_half` convention, pairs `(i, i + head_dim/2)`, `inv_freq = theta^(-2i/d)`, tables precomputed for `max_position_embeddings`.
- GQA: query head `h` reads KV head `h / (num_heads / num_kv_heads)`.
- Qwen2 has q/k/v bias; Llama does not. Loader treats each bias as optional.

## Lessons recorded

- Reference dumps must control every `generation_config.json` knob: Qwen ships `repetition_penalty=1.1`, which `generate()` applies even under greedy decoding, and without an explicit `attention_mask` positions equal to `pad_token_id` are masked — chat prompts contain `<|im_end|>`.

## Not built (say so in interviews)

Speculative decoding, tensor parallelism, fp8/int4 weights, a custom GEMM, swap-to-CPU preemption, priority scheduling, multi-LoRA.
