// Rotary position embedding, in Hugging Face's rotate_half convention:
// element i pairs with i + head_dim/2, not with its neighbour.
//
//   out[i]        = x[i] * cos[pos][i] - x[i+half] * sin[pos][i]
//   out[i+half]   = x[i+half] * cos[pos][i] + x[i] * sin[pos][i]
//
// Two traps live here, and only one of them fails loudly.
//
// Pairing i with i+1 instead of i+half produces perfectly plausible output that
// degrades quality subtly; the golden tensor is what catches it.
//
// The position must come from the sequence, not from the token's index within
// the step. They agree on a simple prefill and diverge the moment a prompt is
// chunked or a sequence is preempted and recomputed, which no first test
// exercises. That is why positions arrive as an array rather than being derived
// here.
//
// V is never rotated. A kernel that rotates it would still pass a q-only check,
// so the golden test asserts v is untouched.
//
// Reference: ref::rope_inplace in include/engine/ops.h.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace engine {

// One block per (token, head); one thread per rotation pair, so head_dim/2
// threads. At head_dim 64 that is 32 threads, exactly one warp, which means no
// __syncthreads is needed: each pair is independent and nothing is shared.
__global__ void rope_kernel(__half* x, const int* positions, const float* cos_table,
                            const float* sin_table, int head_dim, int row_stride) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int half = head_dim / 2;
  const int i = threadIdx.x;
  if (i >= half) return;

  __half* v = x + static_cast<size_t>(token) * row_stride + static_cast<size_t>(head) * head_dim;
  const size_t t = static_cast<size_t>(positions[token]) * half + i;
  const float c = cos_table[t];
  const float s = sin_table[t];

  // Read both halves before writing either: the second write would otherwise
  // clobber an input of the first.
  const float a = __half2float(v[i]);
  const float b = __half2float(v[i + half]);
  v[i] = __float2half(a * c - b * s);
  v[i + half] = __float2half(b * c + a * s);
}

void launch_rope(__half* x, const int* positions, const float* cos_table,
                 const float* sin_table, int n_tokens, int heads, int head_dim,
                 int row_stride, cudaStream_t stream) {
  if (n_tokens <= 0 || heads <= 0 || head_dim <= 0) return;
  const int half = head_dim / 2;
  // Round the block up to a warp so partial warps are not left idle mid-warp;
  // the bounds check above discards the extra lanes.
  const int threads = ((half + 31) / 32) * 32;
  const dim3 grid(n_tokens, heads);
  rope_kernel<<<grid, threads, 0, stream>>>(x, positions, cos_table, sin_table,
                                            head_dim, row_stride);
}

}  // namespace engine
