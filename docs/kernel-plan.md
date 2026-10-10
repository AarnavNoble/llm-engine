# Kernel plan

One section per kernel: what it computes, how to decompose it, the reference it
must agree with, and the mistakes that actually happen. Written before renting a
GPU so that design is not done on a clock.

Two kernels are not planned here because they were not foreseen here. The K/V
cache write became a kernel of its own rather than part of RoPE (see section 2),
and the residual and bias adds became `elementwise.cu` once the forward pass
stopped round-tripping through the host. Both are documented in their own
files.

Every kernel has a reference in `include/engine/ops.h` and a golden tensor dumped
by `./build/golden_dump`. The loop is always:

```bash
scripts/kernel_dev.sh <name>     # build, compare that operation, report
```

Two rules throughout:

- **Check every CUDA call.** A swallowed error turns into wrong numbers much
  later, in a different kernel. Wrap launches and API calls in a macro that
  reports file, line and `cudaGetErrorString`, from the first line of code.
- **Correct first, fast second.** Get agreement with the reference, commit that,
  then optimise with the test still passing. Every optimisation is a separate
  commit with a before/after number.

Shapes below use Qwen2.5-0.5B: `hidden` 896, `layers` 24, `q_heads` 14,
`kv_heads` 2 (GQA group 7), `head_dim` 64, `intermediate` 4864, `vocab` 151936,
`block_size` 16. Nothing may be hardcoded; read it from `ModelConfig`.

---

## 0. Host side first: `cuda_model.cu`

Before any kernel, the plumbing. This is the largest single chunk of work and
none of it is interesting, so do it deliberately rather than discovering it.

**Weight upload.** Read each tensor through `SafeTensorsDir`, convert with
`TensorView::to_f16()`, `cudaMalloc` and `cudaMemcpy` once at load. Keep a small
struct per layer holding device pointers, mirroring `Layer` in the CPU model.
Fail loudly on a `cudaMalloc` failure, with how much was requested and how much
the device has, because "the pool is too big for this card" is a configuration
mistake and should read like one.

**KV cache.** Two allocations per layer, `[num_blocks * block_size][kv_heads][head_dim]`
in fp16, matching the CPU layout exactly so `slot()` is the same arithmetic. At
4096 blocks this is `4096*16*2*64*2 bytes * 24 layers * 2` ≈ 800 MB; check it
against free VRAM at startup and report the number.

**The step layout.** The CPU model already packs a step into one `[N, hidden]`
batch with per-token arrays of owning sequence, position and whether logits are
wanted. Upload three int arrays per step through pinned host buffers:
`token_ids[N]`, `positions[N]`, `slot_ids[N]` (precomputed on the host from the
block tables, so no kernel needs to know about `BlockTable`), plus, for
attention, a flattened block table and per-sequence lengths.

Precomputing `slot_ids` on the host is deliberate: it keeps the page-table walk
out of every kernel, and the only kernel that needs the full block table is
decode attention, which walks history rather than writing one slot.

**GEMMs.** `cublasGemmEx`, fp16 in, **fp32 compute type**. cuBLAS is
column-major and the model is row-major, so compute `Cᵀ = Bᵀ Aᵀ`: to get
`C[m,n] = A[m,k] · B[n,k]ᵀ` pass B as the first matrix with `CUBLAS_OP_T` and A
second with `CUBLAS_OP_N`, with leading dimensions `k`, `k`, `n`. Write this once
in a wrapper and test the wrapper against `ref` output on a small case before
trusting it anywhere, because every wrong-transpose bug looks like a wrong
kernel.

**Exit criterion.** `scripts/kernel_dev.sh --model` green: GPU logits match the
CPU oracle after every layer, on both models.

---

## 1. `rmsnorm.cu`

**Computes** `out[r] = x[r] * rsqrt(mean(x[r]²) + eps) * weight`, over `N` rows
of `hidden`.

**Decomposition.** One block per row, 256 threads. Each thread accumulates a
partial sum of squares over a strided slice, reduce within the warp with
`__shfl_down_sync`, then across warps through a small shared array, then
broadcast the result. Second pass multiplies and writes.

**Accumulate in fp32** even with fp16 in and out. `hidden` is 896 and the values
are O(1), so fp16 accumulation loses real precision and shows up as a 1e-2
disagreement that looks like a different bug.

**Reference** `ref::rmsnorm`. **Golden** `input_norm.L0.bin` from
`embedding.L-1.bin`.

**Traps.** Forgetting the `eps` inside the sqrt rather than outside. Dividing by
the wrong count when `hidden` is not a multiple of the block size. Reading
`weight` per thread without caching it — correct, just slower than needed.

**Performance.** Pure streaming: reads `N*hidden` and writes `N*hidden`. Expect
to approach peak bandwidth. On a 4090 (~1008 GB/s) a 4096-token step at
hidden 896 in fp16 is ~15 MB of traffic, under 20 µs.

---

## 2. `rope.cu`, and the K/V cache write beside it

**Computes** rotary embedding in HF's `rotate_half` convention: for each head,
element `i` pairs with `i + head_dim/2`:

```
out[i]      = x[i] * cos[pos][i] - x[i+half] * sin[pos][i]
out[i+half] = x[i+half] * cos[pos][i] + x[i] * sin[pos][i]
```

Applied to Q in place and to K before K goes into the cache. V is copied to the
cache unrotated -- a kernel that rotates V still passes a Q-only check, so the
golden test asserts V is untouched.

**Decomposition.** Grid `(N, q_heads)` for Q and `(N, kv_heads)` for K, or one
kernel with `heads = q_heads + kv_heads` and a branch. `head_dim/2 = 32` threads
per head, each owning one pair.

**Fusing the cache write is the point, and it has not been done yet.** This
section planned one fused kernel; what shipped is two, `rope.cu` and
`store_kv.cu`, deliberately kept apart. Unfused, K is written to a temporary and
then copied: two extra passes over `N*kv_heads*head_dim`. Fused, the kernel
reads the projection output and writes straight to `slot_ids[i]`.

Keeping them separate first means the fusion can be published as a measured
before/after rather than asserted, which is what the optimisation rows in the
results table are for. The cost of that choice is the two extra passes, paid on
every step until the row is filled in.

**Reference** `ref::rope_inplace`. **Golden** `rope.L0.bin` (q and k) from
`qkv_proj.L0.bin`.

**Traps.** Rotating V, which the golden test catches because v is asserted
untouched. Pairing `i` with `i+1` instead of `i+half` — a plausible-looking
output that degrades quality subtly rather than failing loudly. Using the step
index as the position instead of `positions[i]`, which only breaks after a
preemption or with chunked prefill, i.e. not in your first test. Precompute the
cos and sin tables on the host and upload once; recomputing `powf` per element
is slow and drifts from the reference.

---

## 3. `silu_mul.cu`

**Computes** `gate[i] = silu(gate[i]) * up[i]` over `N * intermediate`, where
`silu(x) = x / (1 + e^-x)`.

**Decomposition.** Grid-stride loop over `N*intermediate`, `half2` or `float4`
loads so each thread moves 4–8 values. Entirely bandwidth-bound.

**Fusing matters here too.** As two kernels this is three passes over `N*4864`;
as one it is two reads and one write. At `N=4096` that is ~40 MB saved per layer
per step, 24 layers deep. Measure both.

**Reference** `ref::silu_mul`. **Golden** no direct pair, since the inputs are
GEMM outputs; the test compares against an independent scalar formulation, and
the per-layer comparison covers it in context.

**Traps.** Computing `silu` in fp16 — `expf` on a half loses accuracy; convert to
float, compute, convert back. Overflow for large negative `x` if you use
`1/(1+exp(-x))` naively; the reference uses the same form, so match it rather
than improving it, or the tolerance comparison becomes meaningless.

---

## 4. `embedding.cu`

**Computes** `out[i] = embed_table[token_ids[i]]`, a gather of `N` rows of
`hidden`.

**Decomposition.** Grid `(N, hidden/vec)`, vectorised loads. Trivial, but it is
the first kernel in the chain, so write it first: if it is wrong, everything
downstream is, and `embedding.L-1.bin` tells you immediately.

**Traps.** Out-of-range token ids from a tokenizer mismatch — assert
`id < vocab_size` in a debug build. Tied embeddings mean this table is also the
output head; upload it once and point both at it.

---

## 5. `attention_decode.cu` — the one that matters

**Computes**, for one query token at position `pos` of one sequence, and one
query head `h`:

```
kv_head = h / (q_heads / kv_heads)
scores[p] = dot(q, K[slot(p)][kv_head]) * rsqrt(head_dim)   for p in 0..pos
out       = sum_p softmax(scores)[p] * V[slot(p)][kv_head]
slot(p)   = block_table[p / block_size] * block_size + p % block_size
```

**Decomposition.** One thread block per `(sequence, query head)`. The block walks
the sequence's block table, and for each 16-token block loads K cooperatively,
computes 16 dot products, updates the running softmax, and accumulates V.

**Online softmax** (FlashAttention §3), so the `[1, pos]` score row is never
materialised. Keep `m` (running max) and `l` (running sum) plus an accumulator
`acc[head_dim]`. For each new block with local max `m_new`:

```
m' = max(m, m_new)
scale_old = exp(m - m')         # rescale what is already accumulated
l  = l * scale_old + sum_j exp(s_j - m')
acc = acc * scale_old + sum_j exp(s_j - m') * V_j
m  = m'
```
and divide `acc` by `l` at the end. This is also what keeps the kernel numerically
stable for long contexts, which a naive exp-then-normalise is not.

**GQA.** `kv_heads = 2` and `q_heads = 14`, so seven query heads share each KV
head. Seven blocks therefore read the same K/V. That is wasteful but correct;
the optimisation is to assign one block per `(sequence, kv_head)` and have it
process all seven query heads, reading K/V once. Do the simple version first.

**Reference** `ref::paged_attention_head`. **Golden** `attn_out.L0.bin`, with
`rope.L0.bin` for K and `qkv_proj.L0.bin` for V, scattered into a cache using the
block table in `manifest.json`. The golden test already does exactly this
reconstruction, so the kernel test is that file with the reference call replaced.

**Traps**, in the order they are likely to happen:
1. **GQA mapping.** `h / group` not `h % group`. Both produce output; one is
   wrong. The golden test catches it, a text sample often does not.
2. **Block table indirection.** Computing `slot` with the logical position
   instead of walking the table, which happens to work when blocks are allocated
   contiguously — exactly what your first test does. The golden manifest's block
   table is `[0, 1, 2]`, so **shuffle it in a second test** or the bug survives.
3. **Off-by-one on `pos`.** The query attends over `0..pos` inclusive. Excluding
   itself is a one-token error that looks like mild quality loss.
4. **Partial final block.** The last block holds `pos % 16 + 1` valid tokens;
   reading all 16 picks up stale K/V from a previous sequence, which is the kind
   of bug that only appears once blocks are being reused.
5. **fp16 accumulation** of the dot products or the V sum. Accumulate in fp32.

**Performance.** Memory-bound: per `(sequence, head)` it reads `pos * head_dim * 2`
bytes of K plus the same of V. At 2048 context, fp16, that is 512 KB per head;
with 32 sequences and GQA sharing it is the dominant traffic in a decode step.
Target: a high fraction of peak bandwidth with vectorised 16-byte loads
(`float4` / `half8`). Report achieved GB/s against 1008 GB/s from `ncu`'s
`dram__bytes.sum` and `gpu__time_duration`.

**Then optimise, in this order**, each a commit with a number:
1. 16-byte vectorised loads of K and V.
2. One block per `(sequence, kv_head)` handling all GQA query heads, so K/V is
   read once instead of `group` times.
3. Split-K for contexts over ~2K: partition the history across several blocks,
   each producing a partial `(acc, m, l)`, then a second kernel combines them
   with the same rescaling rule. This is what keeps a single long sequence from
   leaving the GPU mostly idle.

---

## 6. `attention_prefill.cu`

**Computes** causal attention over a whole chunk: `N` queries against the same
sequence's history, where query `t` attends `0..start+t`.

**Two acceptable implementations.** Tiled shared-memory attention, like decode
but with a causal mask and a tile of queries; or `cublasGemmStridedBatched` for
`QKᵀ` followed by a masked-softmax kernel and a second batched GEMM for `·V`.
The second is less code, slower, and perfectly defensible — prefill is not the
steady-state hot path. **Say which one you chose in the README.**

**Reference and golden** the same as decode: `attn_out.L0.bin` is a 40-token
prefill, so it exercises exactly this path.

**Traps.** The causal mask must use absolute positions, not positions within the
chunk, or chunked prefill silently lets tokens attend to the future. Your chunk
invariance test is what catches this — run it (`[model]`) after this kernel, not
just the golden check.

---

## 7. `sampling.cu`

**Computes** greedy or top-p over `[rows, vocab]`, on device, so logits never
cross PCIe. At vocab 151936 and fp32 that is 600 KB per row per step; moving it
to the host is a real cost at small batch sizes.

**Greedy.** Block reduction for argmax over the row, then a reduce across blocks.
Tie-break on the lowest index to match `Sampler::argmax`, or the comparison
against the CPU sampler fails on exact ties.

**Top-p.** Sorting 151936 values per row is overkill. Either a radix-select for
the k-th largest with k grown until the cumulative mass passes `p`, or a
two-pass threshold search on the probabilities. Match the reference's
tie-breaking and its reseeding behaviour per request, since `seed` is part of the
API contract and `test_sampler.cpp` pins it.

**Traps.** Sampling before the softmax has been normalised consistently with the
CPU path. Using a device RNG whose sequence differs per launch configuration,
which breaks seed reproducibility — derive the stream from the request's seed,
not from the thread index alone.

---

## Order of work, and what each unlocks

| Step | Unlocks |
|---|---|
| Host plumbing + cuBLAS wrapper | nothing runs without it |
| `embedding`, `rmsnorm`, `silu_mul` | the easy three; builds confidence in the harness |
| `rope` + cache write | K/V exists, so attention has something to read |
| naive contiguous attention | first end-to-end GPU logits, oracle parity |
| `attention_decode` paged | the actual claim of the project |
| `attention_prefill` | long prompts stop being slow |
| `sampling` | logits stay on device |
| benchmark rows | the table |
| fusions, vectorisation, split-K, graphs | the optimisation rows |
