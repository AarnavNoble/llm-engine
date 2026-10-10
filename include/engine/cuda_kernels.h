#pragma once
// Host-side launchers for the CUDA kernels.
//
// Each kernel lives in its own .cu file and is reached through one of these, so
// cuda_model.cu orchestrates without needing to know block shapes, and each
// kernel can be swapped in or out independently. The reference each one must
// agree with is in ops.h; the design notes and traps are in docs/kernel-plan.md.
#ifdef ENGINE_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace engine {

// out[t][j] = table[ids[t]][j] for n_tokens rows of `hidden` elements.
void launch_embedding(const __half* table, const int32_t* ids, __half* out,
                      int n_tokens, int hidden, cudaStream_t stream);

// out[r][j] = x[r][j] * rsqrt(mean(x[r]^2) + eps) * weight[j], one block per row.
// `weight` is fp32 because the reference normalises in fp32.
void launch_rmsnorm(const __half* x, const float* weight, __half* out,
                    int rows, int cols, float eps, cudaStream_t stream);

}  // namespace engine

#endif  // ENGINE_CUDA
