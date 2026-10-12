// Paged decode attention, sharing each K/V read across a GQA group.
//
// Profiling said attention is 87.6% of a decode step, 31.6 ms of 36.1 ms at 32
// concurrent requests on an A40. The reason is visible in attention_decode.cu
// rather than subtle: that kernel runs one block per (token, query head), and
// with grouped-query attention several query heads share one kv head. At
// Qwen2.5-0.5B's group of 7, seven blocks each walk the same key/value history
// out of global memory independently. Six of those seven reads are redundant.
//
// This kernel runs one block per (token, kv head) and handles every query head
// in the group together, so each position's K and V are read once and reused
// across the group. The arithmetic is unchanged -- same online softmax, same
// paged indirection, same fp32 accumulation -- so it should agree with the
// reference to the last bit that fp16 inputs allow.
//
// The decomposition had to change to make that possible. attention_decode.cu
// gives every thread its own (max, sum, accumulator) over a stride of
// positions, which is simple and costs head_dim floats per thread. Keeping
// that while handling 7 heads at once would need 7x the accumulator state:
// 64 threads x 7 heads x 64 dims x 4 B is 114 KiB of shared memory, nearly
// three times what a block can have.
//
// So the accumulator is split across a warp instead of owned by a thread. Each
// lane owns head_dim/32 dimensions, which is 2 at head_dim 64, so one head's
// accumulator is 2 registers per lane rather than 64. Seven heads fit in 14.
// The cost is that a dot product now needs a warp reduction per position
// instead of being local to a thread, which is a shuffle tree of 5 steps
// against 6 saved global reads of a 128-byte row.
//
// Warps split the positions between them, exactly as threads did before, and
// their partial results are merged at the end with the same rescaling rule.
//
// This is deliberately a second kernel rather than a rewrite of the first. The
// point of the exercise is a measured before and after, and that needs both to
// exist at once. src/model/cuda/cuda_model.cu selects between them.
//
// Reference: ref::paged_attention_head in include/engine/ops.h.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace engine {
namespace {

constexpr int kWarp = 32;
// 16 warps, 512 threads. Measured, not chosen: at 4 warps this kernel was
// 1.33x SLOWER than the one it replaces, and the reason was occupancy rather
// than anything in the arithmetic. One block per (token, kv head) instead of
// per (token, query head) means 7x fewer blocks -- 64 instead of 448 at 32
// concurrent requests -- and 64 blocks do not fill an A40's 84 SMs. The saved
// bandwidth could not pay for a GPU that was three quarters idle.
//
// Threads per block are the one knob that compensates, since the grid is fixed
// by the decomposition. attention ms/step across the sweep, 728 steps at
// concurrency 32:
//
//     baseline (per token, head)   31.56
//     4 warps, 128 threads         41.90    1.33x slower
//     8 warps, 256 threads         37.42    1.19x slower
//    16 warps, 512 threads         19.47    1.62x faster
//
// 16 is close to the ceiling: the per-warp partials are kWarps * group *
// (head_dim + 2) floats, so 24 warps at group 8 would want 50 KiB against a
// 48 KiB limit.
constexpr int kWarps = 16;
constexpr int kThreadsG = kWarp * kWarps;
constexpr int kMaxGroup = 8;                   // Qwen2.5 is 7, TinyLlama is 8
constexpr int kMaxHeadDim = 128;
constexpr int kMaxDimsPerLane = kMaxHeadDim / kWarp;

// Sum across a warp, leaving the total in every lane.
//
// xor rather than down: the score is needed by all 32 lanes, because each lane
// then updates the accumulator dimensions it owns. __shfl_down_sync would
// leave the result only in lane 0 and need a second broadcast.
__device__ inline float warp_all_sum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

}  // namespace

__global__ void attention_decode_gqa_kernel(const __half* __restrict__ q,
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

  const int token = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int group = q_heads / kv_heads;
  const int dpl = head_dim / kWarp;            // dimensions per lane
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;

  const int pos = token_pos[token];
  const int seq = token_seq[token];
  const int* table = block_tables + table_offset[seq];
  const int kv_dim = kv_heads * head_dim;

  // [group][head_dim] queries, then [kWarps][group][head_dim + 2] partials,
  // the trailing two being the running max and sum.
  float* s_q = smem;
  const int part_stride = head_dim + 2;
  float* s_part = s_q + group * head_dim;

  for (int i = threadIdx.x; i < group * head_dim; i += blockDim.x) {
    const int h = i / head_dim, d = i % head_dim;
    const int head = kv_head * group + h;
    s_q[i] = __half2float(q[(static_cast<size_t>(token) * q_heads + head) * head_dim + d]);
  }
  __syncthreads();

  // Per head, per warp: a running max and sum that stay warp-uniform because
  // every value feeding them comes out of warp_all_sum, plus the slice of the
  // accumulator this lane owns.
  float m[kMaxGroup], l[kMaxGroup], acc[kMaxGroup][kMaxDimsPerLane];
  for (int h = 0; h < group; h++) {
    m[h] = -FLT_MAX;
    l[h] = 0.0f;
    for (int d = 0; d < dpl; d++) acc[h][d] = 0.0f;
  }

  for (int p = warp; p <= pos; p += kWarps) {
    // Paged: the slot comes from the block table, never from arithmetic on p.
    const int slot = table[p / block_size] * block_size + p % block_size;
    const __half* kp = k_cache + static_cast<size_t>(slot) * kv_dim + kv_head * head_dim;
    const __half* vp = v_cache + static_cast<size_t>(slot) * kv_dim + kv_head * head_dim;

    // The whole point: one read of this position's K and V, reused by every
    // query head in the group below.
    float kk[kMaxDimsPerLane], vv[kMaxDimsPerLane];
    for (int d = 0; d < dpl; d++) {
      const int idx = lane * dpl + d;
      kk[d] = __half2float(kp[idx]);
      vv[d] = __half2float(vp[idx]);
    }

    for (int h = 0; h < group; h++) {
      float dot = 0.0f;
      for (int d = 0; d < dpl; d++) dot += s_q[h * head_dim + lane * dpl + d] * kk[d];
      const float s = warp_all_sum(dot) * scale;

      const float m_new = fmaxf(m[h], s);
      const float rescale = (m[h] == -FLT_MAX) ? 0.0f : __expf(m[h] - m_new);
      const float wgt = __expf(s - m_new);
      l[h] = l[h] * rescale + wgt;
      for (int d = 0; d < dpl; d++) acc[h][d] = acc[h][d] * rescale + wgt * vv[d];
      m[h] = m_new;
    }
  }

  for (int h = 0; h < group; h++) {
    float* dst = s_part + (static_cast<size_t>(warp) * group + h) * part_stride;
    for (int d = 0; d < dpl; d++) dst[lane * dpl + d] = acc[h][d];
    // Warp-uniform, so one lane writes them.
    if (lane == 0) {
      dst[head_dim] = m[h];
      dst[head_dim + 1] = l[h];
    }
  }
  __syncthreads();

  // Merge the warps' partials, one head per warp. A warp that saw no positions
  // still carries m = -FLT_MAX and contributes nothing, which the guard makes
  // explicit rather than accidental.
  for (int h = warp; h < group; h += kWarps) {
    float mm = -FLT_MAX, ll = 0.0f, a[kMaxDimsPerLane];
    for (int d = 0; d < dpl; d++) a[d] = 0.0f;

    for (int w = 0; w < kWarps; w++) {
      const float* src = s_part + (static_cast<size_t>(w) * group + h) * part_stride;
      const float ms = src[head_dim], ls = src[head_dim + 1];
      const float m_new = fmaxf(mm, ms);
      const float ca = (mm == -FLT_MAX) ? 0.0f : __expf(mm - m_new);
      const float cb = (ms == -FLT_MAX) ? 0.0f : __expf(ms - m_new);
      for (int d = 0; d < dpl; d++) a[d] = a[d] * ca + src[lane * dpl + d] * cb;
      ll = ll * ca + ls * cb;
      mm = m_new;
    }

    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    const int head = kv_head * group + h;
    for (int d = 0; d < dpl; d++)
      out[(static_cast<size_t>(token) * q_heads + head) * head_dim + lane * dpl + d] =
          __float2half(a[d] * inv);
  }
}

void launch_attention_decode_gqa(const __half* q, const __half* k_cache, const __half* v_cache,
                                 const int* block_tables, const int* table_offset,
                                 const int* token_seq, const int* token_pos, __half* out,
                                 int n_tokens, int q_heads, int kv_heads, int head_dim,
                                 int block_size, float scale, cudaStream_t stream) {
  if (n_tokens <= 0 || q_heads <= 0 || kv_heads <= 0) return;

  // Three bounds, enforced rather than assumed. attention_decode.cu declared a
  // head_dim bound as a constant and never checked it, which meant a larger
  // model would have failed the launch silently and surfaced the error at an
  // unrelated line several kernels later.
  const int group = q_heads / kv_heads;
  if (q_heads % kv_heads != 0)
    throw std::runtime_error("attention_decode_gqa: q_heads " + std::to_string(q_heads) +
                             " is not a multiple of kv_heads " + std::to_string(kv_heads));
  if (group > kMaxGroup)
    throw std::runtime_error("attention_decode_gqa: GQA group " + std::to_string(group) +
                             " exceeds " + std::to_string(kMaxGroup) +
                             "; the per-head accumulators are held in registers, so raising "
                             "this trades occupancy for reach.");
  if (head_dim > kMaxHeadDim || head_dim % kWarp != 0)
    throw std::runtime_error("attention_decode_gqa: head_dim " + std::to_string(head_dim) +
                             " must be a multiple of " + std::to_string(kWarp) + " and at most " +
                             std::to_string(kMaxHeadDim) +
                             "; the accumulator is split one slice per lane.");

  const size_t smem = (static_cast<size_t>(group) * head_dim +
                       static_cast<size_t>(kWarps) * group * (head_dim + 2)) * sizeof(float);
  const dim3 grid(n_tokens, kv_heads);
  attention_decode_gqa_kernel<<<grid, kThreadsG, smem, stream>>>(
      q, k_cache, v_cache, block_tables, table_offset, token_seq, token_pos, out,
      q_heads, kv_heads, head_dim, block_size, scale);
}

}  // namespace engine
