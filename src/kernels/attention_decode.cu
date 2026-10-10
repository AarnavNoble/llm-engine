// Paged attention: the kernel this project exists for.
//
// For one query token at position `pos` of one sequence, and one query head h:
//
//   kv_head   = h / (q_heads / kv_heads)                     GQA sharing
//   score[p]  = dot(q, K[slot(p)][kv_head]) * 1/sqrt(head_dim)   for p in 0..pos
//   out       = sum_p softmax(score)[p] * V[slot(p)][kv_head]
//   slot(p)   = block_table[p / block_size] * block_size + p % block_size
//
// Three things make this different from textbook attention.
//
// The history is *paged*: positions are contiguous logically but scattered
// physically, so every access goes through the block table. A kernel that
// computes the slot directly from p works perfectly whenever blocks happen to
// have been allocated contiguously, which is exactly what a first test does.
//
// The softmax is *online*: the score row is never materialised, because at
// 2048 positions it would not fit anywhere useful. Instead a running max and
// sum are carried, and everything accumulated so far is rescaled whenever a
// larger value appears. That is the FlashAttention trick, and it is also what
// keeps long contexts numerically stable.
//
// The work is split *by position*, not by dimension. Each thread owns a stride
// of the history and keeps its own (max, sum, accumulator); they are merged at
// the end with the same rescaling rule. Splitting by dimension instead would
// need a reduction per position rather than one at the end.
//
// Reference: ref::paged_attention_head in include/engine/ops.h.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>

namespace engine {
namespace {

constexpr int kThreads = 64;     // partial accumulators live in shared memory,
constexpr int kMaxHeadDim = 128; // so both of these bound its size: 64*128*4 = 32 KiB

}  // namespace

__global__ void attention_decode_kernel(const __half* __restrict__ q,
                                        const __half* __restrict__ k_cache,
                                        const __half* __restrict__ v_cache,
                                        const int* __restrict__ block_tables,
                                        const int* __restrict__ table_offset,
                                        const int* __restrict__ token_seq,
                                        const int* __restrict__ token_pos,
                                        __half* __restrict__ out,
                                        int q_heads, int kv_heads, int head_dim,
                                        int block_size, float scale) {
  extern __shared__ float smem[];
  // Layout: the query vector, then one accumulator row per thread, then the
  // per-thread running max and sum.
  float* s_q = smem;                                   // [head_dim]
  float* s_acc = s_q + head_dim;                       // [kThreads][head_dim]
  float* s_max = s_acc + kThreads * head_dim;          // [kThreads]
  float* s_sum = s_max + kThreads;                     // [kThreads]

  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int t = threadIdx.x;

  const int pos = token_pos[token];
  const int seq = token_seq[token];
  const int* table = block_tables + table_offset[seq];
  const int group = q_heads / kv_heads;
  const int kv_head = head / group;      // not head % group; both run, one is wrong
  const int kv_dim = kv_heads * head_dim;

  // The whole block shares one query vector, so load it once.
  for (int d = t; d < head_dim; d += kThreads)
    s_q[d] = __half2float(q[(static_cast<size_t>(token) * q_heads + head) * head_dim + d]);
  __syncthreads();

  float* acc = s_acc + static_cast<size_t>(t) * head_dim;
  for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
  float m = -FLT_MAX;   // running max; -FLT_MAX rather than -inf keeps the
  float l = 0.0f;       // rescaling arithmetic free of inf - inf

  // Each thread walks its own stride of the history.
  for (int p = t; p <= pos; p += kThreads) {
    const int slot = table[p / block_size] * block_size + p % block_size;
    const __half* kp = k_cache + static_cast<size_t>(slot) * kv_dim + kv_head * head_dim;
    const __half* vp = v_cache + static_cast<size_t>(slot) * kv_dim + kv_head * head_dim;

    float s = 0.0f;
    for (int d = 0; d < head_dim; d++) s += s_q[d] * __half2float(kp[d]);
    s *= scale;

    // Online update: if this score is the new maximum, everything accumulated
    // so far is scaled down to match before the new term is added.
    const float m_new = fmaxf(m, s);
    const float rescale = (m == -FLT_MAX) ? 0.0f : __expf(m - m_new);
    const float w = __expf(s - m_new);
    l = l * rescale + w;
    for (int d = 0; d < head_dim; d++) acc[d] = acc[d] * rescale + w * __half2float(vp[d]);
    m = m_new;
  }
  s_max[t] = m;
  s_sum[t] = l;
  __syncthreads();

  // Merge the per-thread partials pairwise, with the same rescaling rule. A
  // thread that saw no positions still holds m = -FLT_MAX and contributes
  // nothing, which the guards below make explicit rather than accidental.
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      const float ma = s_max[t], mb = s_max[t + stride];
      const float m_new = fmaxf(ma, mb);
      const float ca = (ma == -FLT_MAX) ? 0.0f : __expf(ma - m_new);
      const float cb = (mb == -FLT_MAX) ? 0.0f : __expf(mb - m_new);
      float* a = s_acc + static_cast<size_t>(t) * head_dim;
      const float* b = s_acc + static_cast<size_t>(t + stride) * head_dim;
      for (int d = 0; d < head_dim; d++) a[d] = a[d] * ca + b[d] * cb;
      s_sum[t] = s_sum[t] * ca + s_sum[t + stride] * cb;
      s_max[t] = m_new;
    }
    __syncthreads();
  }

  // Thread 0's row now holds the unnormalised result for the whole history.
  const float denom = s_sum[0];
  const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
  for (int d = t; d < head_dim; d += kThreads)
    out[(static_cast<size_t>(token) * q_heads + head) * head_dim + d] =
        __float2half(s_acc[d] * inv);
}

void launch_attention_decode(const __half* q, const __half* k_cache, const __half* v_cache,
                             const int* block_tables, const int* table_offset,
                             const int* token_seq, const int* token_pos, __half* out,
                             int n_tokens, int q_heads, int kv_heads, int head_dim,
                             int block_size, float scale, cudaStream_t stream) {
  if (n_tokens <= 0 || q_heads <= 0) return;
  const size_t smem = (static_cast<size_t>(head_dim) +
                       static_cast<size_t>(kThreads) * head_dim + 2 * kThreads) * sizeof(float);
  const dim3 grid(n_tokens, q_heads);
  attention_decode_kernel<<<grid, kThreads, smem, stream>>>(
      q, k_cache, v_cache, block_tables, table_offset, token_seq, token_pos, out,
      q_heads, kv_heads, head_dim, block_size, scale);
}

}  // namespace engine
