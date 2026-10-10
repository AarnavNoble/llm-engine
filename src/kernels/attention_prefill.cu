// Paged attention for prefill: many query tokens of one sequence at once.
//
// The computation is the same as attention_decode. For a query token at
// position `pos` of one sequence and one query head h:
//
//   kv_head   = h / (q_heads / kv_heads)                           GQA sharing
//   score[p]  = dot(q, K[slot(p)][kv_head]) * scale   for p in 0..pos
//   out       = sum_p softmax(score)[p] * V[slot(p)][kv_head]
//   slot(p)   = block_table[p / block_size] * block_size + p % block_size
//
// The decode kernel is already *correct* for prefill, and it is worth being
// precise about why: each token's loop runs p = 0..token_pos[token] inclusive,
// and cuda_model stores K/V for every token of the step before any attention
// runs, so a token attends to its own prefix and to nothing after it. That is
// exactly the causal mask. Nothing here fixes a bug.
//
// What decode leaves on the table is reuse and shape. One block per query
// token means the N tokens of a prompt each walk the same history out of
// global memory, so a prompt of N tokens moves O(N^2) bytes of K/V with no
// sharing at all. And decode's 64 threads are split by position, so for a
// token at position 20 most of them own nothing and the block still pays a
// full log2(64) tree reduction over head_dim-wide rows afterwards.
//
// So this kernel tiles instead. One block owns a tile of up to kTileQ query
// tokens of one sequence, for one query head, and walks the history in tiles
// of kTileK positions. Each K/V tile is loaded into shared memory once and
// consumed by every query in the tile: the global traffic drops by the tile
// width. The online softmax is per-K-tile rather than per-position, which is
// the same rescaling rule applied to a group of scores at a time, so no score
// row is materialised and long prefixes stay stable.
//
// The causal mask uses *absolute* positions from token_pos, never the index of
// a query within its tile. With chunked prefill a tile can start at position
// 512, and a mask written against the in-tile index would let those tokens
// attend to the future while still passing a single-chunk golden test.
//
// The work split means no cross-thread reduction at all. kDimGroups threads
// share one query row and each owns a strided slice of head_dim; the running
// max and sum are recomputed redundantly by all of them from the shared score
// row rather than reduced across them. Redundant arithmetic over kTileK values
// is cheaper than the barrier it replaces.
//
// Reference: ref::paged_attention_head in include/engine/ops.h, the same one
// decode must agree with.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cfloat>
#include <cstddef>

namespace engine {
namespace {

constexpr int kThreads = 128;
constexpr int kTileQ = 16;        // query tokens per block
constexpr int kTileK = 16;        // history positions staged per iteration
constexpr int kMaxHeadDim = 128;
constexpr int kDimGroups = kThreads / kTileQ;   // threads sharing one query row
static_assert(kThreads % kTileQ == 0, "each query row needs a whole group");

// Shared memory, counted in floats, is bounded by those constants alone:
//
//   q and acc : 2 * kTileQ * (head_dim + kDimGroups)
//   k and v   : 2 * kTileK * (head_dim + 1)
//   scores    :     kTileQ * (kTileK + 1)
//   sums, pos : 2 * kTileQ
//
// At head_dim = kMaxHeadDim that is 4352 + 4128 + 272 + 32 = 8784 floats, or
// 34.3 KiB, inside the 48 KiB a block gets without opting in. Nothing in the
// kernel assumes head_dim is 128; the constant only pins this bound, and a
// head_dim large enough to break it fails the launch rather than quietly
// computing something else.
//
// The three row paddings are not decoration. The acc row stride is padded by
// kDimGroups so the 4 query rows and 8 lanes of one warp land on 32 distinct
// shared-memory banks during the V accumulation; with an unpadded stride of
// 128 floats all four rows sit on the same 8 banks. The K/V row stride is
// padded by 1 because the dot-product stage walks j across a warp at fixed d,
// which an unpadded stride collapses onto one bank. The score row is padded
// for the same reason.

}  // namespace

__global__ void attention_prefill_kernel(const __half* __restrict__ q,
                                         const __half* __restrict__ k_cache,
                                         const __half* __restrict__ v_cache,
                                         const int* __restrict__ block_tables,
                                         const int* __restrict__ table_offset,
                                         const int* __restrict__ token_seq,
                                         const int* __restrict__ token_pos,
                                         const int* __restrict__ q_tile_offset,
                                         __half* __restrict__ out,
                                         int q_heads, int kv_heads, int head_dim,
                                         int block_size, float scale) {
  const int q_stride = head_dim + kDimGroups;
  const int kv_stride = head_dim + 1;
  const int score_stride = kTileK + 1;

  extern __shared__ float smem[];
  float* s_q = smem;                                      // [kTileQ][q_stride]
  float* s_acc = s_q + kTileQ * q_stride;                 // [kTileQ][q_stride]
  float* s_k = s_acc + kTileQ * q_stride;                 // [kTileK][kv_stride]
  float* s_v = s_k + kTileK * kv_stride;                  // [kTileK][kv_stride]
  float* s_s = s_v + kTileK * kv_stride;                  // [kTileQ][score_stride]
  float* s_l = s_s + kTileQ * score_stride;               // [kTileQ]
  int* s_pos = reinterpret_cast<int*>(s_l + kTileQ);      // [kTileQ]

  const int tile = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;

  const int tok0 = q_tile_offset[tile];
  const int span = q_tile_offset[tile + 1] - tok0;
  // The launcher's contract is that a tile holds one sequence's tokens and is
  // no wider than kTileQ. Clamping rather than trusting keeps an out-of-
  // contract caller from writing past the shared rows, and the assert names
  // the mistake in a debug build instead of letting it look numerical.
  const int tile_n = span < kTileQ ? span : kTileQ;
  if (tile_n <= 0) return;
  assert(span <= kTileQ);

  const int seq = token_seq[tok0];
  const int* table = block_tables + table_offset[seq];
  const int group = q_heads / kv_heads;
  const int kv_head = head / group;      // not head % group; both run, one is wrong
  const int kv_dim = kv_heads * head_dim;

#ifndef NDEBUG
  // A tile straddling two sequences would attend over the wrong block table,
  // and silently: the output stays plausible, it just is not the reference's.
  if (tid == 0)
    for (int qi = 0; qi < tile_n; qi++) assert(token_seq[tok0 + qi] == seq);
#endif

  // Stage the tile's queries and zero its accumulators. d is the fast index so
  // the global read of q stays coalesced.
  for (int i = tid; i < tile_n * head_dim; i += kThreads) {
    const int qi = i / head_dim;
    const int d = i - qi * head_dim;
    const size_t row = (static_cast<size_t>(tok0 + qi) * q_heads + head) * head_dim;
    s_q[qi * q_stride + d] = __half2float(q[row + d]);
    s_acc[qi * q_stride + d] = 0.0f;
  }
  for (int qi = tid; qi < tile_n; qi += kThreads) s_pos[qi] = token_pos[tok0 + qi];
  __syncthreads();

  // Positions ascend within a tile, so its last token bounds the history the
  // block has to walk. Taking a max rather than reading the last element means
  // the kernel does not quietly depend on that ordering.
  int pos_max = -1;
  for (int qi = 0; qi < tile_n; qi++)
    if (s_pos[qi] > pos_max) pos_max = s_pos[qi];

  const int qi_me = tid / kDimGroups;
  const int lane = tid % kDimGroups;
  float m = -FLT_MAX;   // running max; -FLT_MAX rather than -inf keeps the
  float l = 0.0f;       // rescaling arithmetic free of inf - inf

  for (int p_base = 0; p_base <= pos_max; p_base += kTileK) {
    const int left = pos_max + 1 - p_base;
    const int n_keys = left < kTileK ? left : kTileK;

    // Every position goes through the block table. Deriving the slot from p
    // arithmetically works whenever blocks happen to be contiguous, which is
    // what an untouched first test allocates and what the shuffled-table test
    // exists to catch. kTileK is deliberately not tied to block_size, so one
    // staged tile may span two blocks and each position is looked up alone.
    for (int j = 0; j < n_keys; j++) {
      const int p = p_base + j;
      const int slot = table[p / block_size] * block_size + p % block_size;
      const __half* kp = k_cache + static_cast<size_t>(slot) * kv_dim + kv_head * head_dim;
      const __half* vp = v_cache + static_cast<size_t>(slot) * kv_dim + kv_head * head_dim;
      for (int d = tid; d < head_dim; d += kThreads) {
        s_k[j * kv_stride + d] = __half2float(kp[d]);
        s_v[j * kv_stride + d] = __half2float(vp[d]);
      }
    }
    __syncthreads();

    // One thread per (query, key) pair, up to kTileQ * kTileK of them. A
    // masked pair is not computed at all; it stores the -FLT_MAX sentinel the
    // accumulation below skips on. Accumulating in float from values already
    // widened in shared memory is the point: a head_dim-long dot of O(1) terms
    // summed in fp16 loses enough precision to look like a logic bug.
    const int pairs = tile_n * n_keys;
    for (int idx = tid; idx < pairs; idx += kThreads) {
      const int qi = idx / n_keys;
      const int j = idx - qi * n_keys;
      float s = -FLT_MAX;
      if (p_base + j <= s_pos[qi]) {     // absolute positions, causal, inclusive
        const float* qp = s_q + qi * q_stride;
        const float* kp = s_k + j * kv_stride;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += qp[d] * kp[d];
        s = dot * scale;
      }
      s_s[qi * score_stride + j] = s;
    }
    __syncthreads();

    // Online update, one query row at a time. Each of the kDimGroups threads
    // on a row recomputes that row's max and weights from shared memory and
    // owns a strided slice of head_dim, so m and l stay in registers and this
    // whole stage needs neither a reduction nor a barrier of its own.
    if (qi_me < tile_n) {
      const float* row = s_s + qi_me * score_stride;
      float tile_max = -FLT_MAX;
      for (int j = 0; j < n_keys; j++) tile_max = fmaxf(tile_max, row[j]);

      // Every score masked means this tile lies entirely in the query's
      // future, which is the common case for the early rows once p_base has
      // passed them. Skipping leaves m and l untouched, as it must.
      if (tile_max > -FLT_MAX) {
        const float m_new = fmaxf(m, tile_max);
        const float rescale = (m == -FLT_MAX) ? 0.0f : __expf(m - m_new);
        float* acc = s_acc + qi_me * q_stride;
        for (int d = lane; d < head_dim; d += kDimGroups) acc[d] *= rescale;

        float l_add = 0.0f;
        for (int j = 0; j < n_keys; j++) {
          const float s = row[j];
          if (s == -FLT_MAX) continue;
          const float w = __expf(s - m_new);
          l_add += w;
          const float* vp = s_v + j * kv_stride;
          for (int d = lane; d < head_dim; d += kDimGroups) acc[d] += w * vp[d];
        }
        l = l * rescale + l_add;
        m = m_new;
      }
    }
    // Held until every row has finished reading s_k, s_v and s_s, all of which
    // the next iteration overwrites.
    __syncthreads();
  }

  // The sums cross threads exactly once, so the write back can be laid out for
  // coalescing instead of inheriting the accumulation's thread mapping.
  if (lane == 0 && qi_me < tile_n) s_l[qi_me] = l;
  __syncthreads();

  for (int i = tid; i < tile_n * head_dim; i += kThreads) {
    const int qi = i / head_dim;
    const int d = i - qi * head_dim;
    const float denom = s_l[qi];
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    out[(static_cast<size_t>(tok0 + qi) * q_heads + head) * head_dim + d] =
        __float2half(s_acc[qi * q_stride + d] * inv);
  }
}

// q_tile_offset holds n_tiles + 1 entries, the same CSR shape as table_offset:
// tile g covers query tokens q_tile_offset[g] .. q_tile_offset[g+1] - 1 of the
// step's packed token arrays. Every tile must lie within one sequence and be
// at most kTileQ wide, which is why the caller builds it instead of the kernel
// deriving it: where one slice ends and the next begins is host knowledge.
void launch_attention_prefill(const __half* q, const __half* k_cache, const __half* v_cache,
                              const int* block_tables, const int* table_offset,
                              const int* token_seq, const int* token_pos,
                              const int* q_tile_offset, __half* out,
                              int n_tiles, int q_heads, int kv_heads, int head_dim,
                              int block_size, float scale, cudaStream_t stream) {
  // A zero-block grid is an invalid launch, and the rest would divide by zero.
  if (n_tiles <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || block_size <= 0) return;

  const size_t floats = 2u * kTileQ * (static_cast<size_t>(head_dim) + kDimGroups) +
                        2u * kTileK * (static_cast<size_t>(head_dim) + 1) +
                        static_cast<size_t>(kTileQ) * (kTileK + 1) + 2u * kTileQ;
  const dim3 grid(n_tiles, q_heads);
  attention_prefill_kernel<<<grid, kThreads, floats * sizeof(float), stream>>>(
      q, k_cache, v_cache, block_tables, table_offset, token_seq, token_pos, q_tile_offset,
      out, q_heads, kv_heads, head_dim, block_size, scale);
}

// Still the simple version, in the order decode's notes give: 16-byte
// vectorised staging of K and V, then one block per (tile, kv_head) handling
// all `group` query heads so K/V is staged once rather than `group` times,
// then tensor cores for the two matrix products. The last is where the real
// factor is, and it is a different kernel rather than an edit to this one.

}  // namespace engine
