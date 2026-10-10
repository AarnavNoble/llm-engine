// Scatter this step's K and V rows into their slots in the paged cache.
//
// slot = block_table[pos / block_size] * block_size + pos % block_size, the
// same arithmetic the CPU backend and the attention kernel use. The slots are
// computed on the host and passed in, so no kernel needs to know what a block
// table is; only attention, which walks history rather than writing one row,
// needs the table itself.
//
// This is deliberately separate from the RoPE kernel for now. Fusing the two,
// so K is written exactly once instead of being stored and then re-read, is a
// real saving and belongs in the benchmark table as a measured before and
// after rather than as an assumption.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace engine {

__global__ void store_kv_kernel(const __half* k, const __half* v, const int* slots,
                                __half* k_cache, __half* v_cache, int kv_dim) {
  const int token = blockIdx.x;
  const size_t src = static_cast<size_t>(token) * kv_dim;
  const size_t dst = static_cast<size_t>(slots[token]) * kv_dim;
  for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
    k_cache[dst + d] = k[src + d];
    v_cache[dst + d] = v[src + d];
  }
}

void launch_store_kv(const __half* k, const __half* v, const int* slots,
                     __half* k_cache, __half* v_cache, int n_tokens, int kv_dim,
                     cudaStream_t stream) {
  if (n_tokens <= 0 || kv_dim <= 0) return;
  store_kv_kernel<<<n_tokens, 128, 0, stream>>>(k, v, slots, k_cache, v_cache, kv_dim);
}

}  // namespace engine
