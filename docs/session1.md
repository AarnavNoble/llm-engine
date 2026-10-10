# Session 1, step by step

Goal: a CUDA build that produces the same logits as the CPU oracle, plus the
three easy kernels. Budget about four hours. Everything here is a command to
run or a thing to write, in order.

Keep `docs/kernel-plan.md` open alongside this. That one explains *why*; this one
is the sequence.

---

## 1. Deploy the pod

| Setting | Value |
|---|---|
| Storage | Network Volume, 20 GB, note the region |
| Pod | Community Cloud, RTX 4090, same region as the volume |
| Template | `runpod/pytorch:*-cuda12.*-devel` — the **devel** suffix is what provides `nvcc` |
| Ports | TCP 22 and 8000 |

Set a timer on your phone for your intended session length. A forgotten running
pod is the only way to lose real money here.

## 2. Connect and bring up

```bash
cd /workspace
git clone https://github.com/AarnavNoble/llm-engine engine && cd engine
scripts/gpu_bringup.sh
```

Takes about ten minutes. **Do not watch it.** Read `docs/kernel-plan.md` §0 and
open `src/model/cpu/cpu_model.cpp` at `forward()`; that function is what you are
about to port.

It ends with `== ready`. Note the compute capability it prints, which should be
`89` for a 4090. If anything in it failed, fix that before continuing: everything
below assumes the CPU backend is green.

## 3. Make the CUDA build configure and link

CMake requires all eight `.cu` files to exist, and `engine_core` calls
`make_cuda_model`, so create stubs before writing anything real. This gives you a
tree that always builds, which means every later failure is caused by the thing
you just wrote.

```bash
mkdir -p src/kernels src/model/cuda
for f in rmsnorm rope silu_mul embedding attention_decode attention_prefill sampling; do
  echo "// $f" > src/kernels/$f.cu
done

cat > src/model/cuda/cuda_model.cu <<'EOF'
#include <stdexcept>
#include "engine/model.h"

namespace engine {
std::unique_ptr<Model> make_cuda_model(const std::string&, const KVCacheManager&) {
  throw std::runtime_error("CUDA backend not implemented yet");
}
}  // namespace engine
EOF

cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENGINE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda
```

**Checkpoint.** It compiles and links. You have a CUDA build whose model still
runs on the CPU. If this fails it is the toolkit path or the link setup, and it
is far cheaper to fix now than after you have written three kernels.

Export the build directory so the helper script uses it:

```bash
export BUILD=build-cuda
```

## 4. Error checking, first

Put this at the top of `cuda_model.cu` and use it on **every** CUDA and cuBLAS
call from here on. CUDA fails silently and asynchronously; an unchecked error
from one kernel surfaces as wrong numbers in a different one.

```cpp
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
  throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                           " " + cudaGetErrorString(e_)); } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) \
  throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                           " cuBLAS status " + std::to_string(int(s_))); } while (0)
```

While developing, follow each kernel launch with
`CUDA_CHECK(cudaDeviceSynchronize())`. It is slow and it attributes the error to
the kernel that actually caused it. Remove those syncs later.

## 5. The cuBLAS wrapper, and a test of it

You want the `nn.Linear` shape, with weights stored `[out, in]`:

```
C[m,n] = A[m,k] · B[n,k]ᵀ
```

cuBLAS is column-major while your data is row-major, so ask it for `Cᵀ`, which is
the same bytes as `C`. Since `Cᵀ = B · Aᵀ`:

```cpp
// C[m,n] = A[m,k] * B[n,k]^T, all row-major, fp16 data with fp32 accumulate.
void gemm(cublasHandle_t h, const __half* A, const __half* B, __half* C,
          int m, int n, int k) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(h,
      CUBLAS_OP_T, CUBLAS_OP_N,   // transpose B, not A
      n, m, k,                    // note the swap
      &alpha,
      B, CUDA_R_16F, k,
      A, CUDA_R_16F, k,
      &beta,
      C, CUDA_R_16F, n,
      CUBLAS_COMPUTE_32F,         // not 16F: accumulating 896 terms in fp16 loses real precision
      CUBLAS_GEMM_DEFAULT));
}
```

**Test it before anything depends on it.** This is the highest-value ten minutes
of the session, because a wrong transpose looks exactly like a wrong kernel.

```
A (m=2, k=3) = [[1, 2, 3],
                [4, 5, 6]]

B (n=2, k=3) = [[1, 0, -1],
                [2, 1,  0]]

C = A · Bᵀ   = [[-2,  4],
                [-2, 13]]
```

Upload those, call `gemm`, copy back, print. If you get the transpose of that, or
`[[-2,-2],[4,13]]`, your leading dimensions or op flags are swapped.

## 6. Weight upload

Mirror the CPU model's `Layer` struct with device pointers. For each tensor:

```
SafeTensorsDir → TensorView → .to_f16() → cudaMalloc → cudaMemcpy → store pointer
```

Load every per-layer tensor, plus `model.embed_tokens.weight`,
`model.norm.weight`, and `lm_head.weight` when `tie_word_embeddings` is false.
Remember q/k/v biases are optional, present on Qwen and absent on TinyLlama.

Print the total allocated. If a `cudaMalloc` fails, report how much was requested
and how much the device has free, because that is a configuration mistake and
should read like one:

```cpp
size_t free_b = 0, total_b = 0;
CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
```

## 7. KV cache and scratch buffers

Per layer, one K and one V allocation, laid out exactly as the CPU model does:

```
[num_blocks * block_size][kv_heads][head_dim]   in fp16
```

Identical layout means `slot(pos) = block_table[pos / block_size] * block_size +
pos % block_size` is the same arithmetic on both sides, which removes a whole
class of disagreement.

Allocate the per-step scratch **once**, sized for the largest step
(`max_num_batched_tokens`): `x`, `h`, `q`, `k`, `v`, `attn`, `o`, `gate`, `up`,
and the logits buffer. Never `cudaMalloc` inside `forward()`.

Also allocate three pinned host buffers and their device counterparts for the
per-token arrays: `token_ids[N]`, `positions[N]`, `slot_ids[N]`.

## 8. `forward()` with host round-trips

Port `CpuModel::forward` structurally: flatten the step into the same packed
`[N, hidden]` layout, fill the three int arrays on the host (compute `slot_ids`
from each sequence's block table here, so no kernel needs to know about block
tables), upload them, then run the layer loop.

For every operation, do the slow thing for now:

```
copy device → host → call the matching ref:: function → copy host → device
```

`include/engine/ops.h` has `ref::rmsnorm`, `ref::rope_inplace`, `ref::silu_mul`
and `ref::paged_attention_head`. Use the real `gemm` for the projections, since
you have already tested it.

```bash
scripts/kernel_dev.sh --model
```

**Checkpoint, and the real milestone of today.** Green means layout, strides,
uploads, GEMM transposes, residual adds and the logits gather are all correct,
*before* a single kernel is under suspicion.

```bash
git add -A && git commit -m "cuda: plumbing, gemm wrapper, round-trip forward" && git push
```

If the per-layer test fails here, the bug is in the plumbing rather than in math:
a wrong stride, a layer pointing at the previous layer's weights, the residual
added in the wrong place, or the logits gather taking the wrong rows.

## 9. Replace the round-trips, one kernel at a time

In this order, easiest first, so the harness proves itself on simple cases:

**`embedding.cu`** — `out[i] = embed[token_ids[i]]`, a gather of `N` rows of
`hidden`. Grid `(N, hidden/vec)`. Write it first because everything downstream
depends on it, and `embedding.L-1.bin` catches it immediately.

**`rmsnorm.cu`** — one block per row, 256 threads, each accumulating a partial
sum of squares over a strided slice; reduce with `__shfl_down_sync`, then across
warps through shared memory; **accumulate in fp32**; second pass multiplies by
`rsqrt(mean + eps)` and the weight.

**`silu_mul.cu`** — grid-stride loop over `N * intermediate`, `gate[i] =
silu(gate[i]) * up[i]`. Convert to float for the `expf`, back to half to store.
Vectorise the loads once it is correct.

After each one:

```bash
scripts/kernel_dev.sh                 # just the operations, seconds
scripts/kernel_dev.sh --model         # in context, slower
git commit -am "cuda: <name> kernel" && git push
```

When an operation disagrees, fix the earliest one in the layer first: every later
tensor is computed from its output. The test prints `max |diff|` against
`max |expected|`. A difference at the magnitude of the values is a wrong formula;
one around 1e-3 is almost always fp16 accumulation where the reference uses fp32.

## 10. End of session

```bash
git add -A && git commit -m "cuda: session 1" && git push
```

**Stop the pod from the dashboard.** Closing the tab does not stop billing.

---

## What counts as a good session

| Outcome | Verdict |
|---|---|
| Round-trip `forward()` green | the session paid for itself; the plumbing is the hard part |
| Plus one or two kernels | on plan |
| Plus all three kernels | ahead; session 2 is RoPE and naive attention |

If anything blocks you for more than about twenty minutes, push what you have and
ask, rather than grinding. Twenty minutes of a 4090 costs about twelve cents, but
two hours of it is most of a session.
