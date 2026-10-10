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

// gate[i] = silu(gate[i]) * up[i], in place over n elements.
void launch_silu_mul(__half* gate, const __half* up, size_t n, cudaStream_t stream);

// Rotary embedding in place over x[n_tokens][row_stride], rotating `heads`
// heads of head_dim each. Positions come from the sequence, one per token, so
// chunked prefill and recompute-after-preemption stay correct.
void launch_rope(__half* x, const int* positions, const float* cos_table,
                 const float* sin_table, int n_tokens, int heads, int head_dim,
                 int row_stride, cudaStream_t stream);

// x[i] += y[i], the residual connection.
void launch_add_inplace(__half* x, const __half* y, size_t n, cudaStream_t stream);

// x[r][c] += bias[c], with the bias kept in fp32 as the reference has it.
void launch_add_bias(__half* x, const float* bias, int rows, int cols, cudaStream_t stream);

// Scatter k and v rows into their precomputed cache slots.
void launch_store_kv(const __half* k, const __half* v, const int* slots,
                     __half* k_cache, __half* v_cache, int n_tokens, int kv_dim,
                     cudaStream_t stream);

// Paged attention over the whole history of each token's own sequence.
// block_tables is every sequence's table concatenated, table_offset indexes
// into it, and token_seq maps a token to its sequence.
void launch_attention_decode(const __half* q, const __half* k_cache, const __half* v_cache,
                             const int* block_tables, const int* table_offset,
                             const int* token_seq, const int* token_pos, __half* out,
                             int n_tokens, int q_heads, int kv_heads, int head_dim,
                             int block_size, float scale, cudaStream_t stream);

}  // namespace engine

#endif  // ENGINE_CUDA
