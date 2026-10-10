#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine {

// out[token][j] = table[ids[token]][j], for N tokens of `hidden` elements.
__global__ void embedding_kernel(const __half* table, const int32_t* ids,
                                 __half* out, int hidden) {
  // One block per token.
  const int token = blockIdx.x;

  // size_t math: ids[token] * hidden can overflow int32 for large vocab * hidden.
  const __half* src = table + static_cast<size_t>(ids[token]) * hidden;
  __half* dst = out + static_cast<size_t>(token) * hidden;

  // Each thread handles elements j, j + blockDim.x, j + 2*blockDim.x, ...
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    dst[j] = src[j];
  }
}

// Host-side launcher, so cuda_model.cu can call it without being a .cu concern.
void launch_embedding(const __half* table, const int32_t* ids, __half* out,
                      int n_tokens, int hidden, cudaStream_t stream) {
  if (n_tokens <= 0 || hidden <= 0) return;  // a zero-block grid is an invalid launch
  embedding_kernel<<<n_tokens, 256, 0, stream>>>(table, ids, out, hidden);
}

}  // namespace engine