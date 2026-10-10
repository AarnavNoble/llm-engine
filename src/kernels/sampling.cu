// Sampling on device, so the logits never cross PCIe.
//
// The head leaves a [rows, vocab] __half tensor in device memory. Today
// cuda_model copies all of it back and Sampler::sample picks a token on the
// host; at vocab 151936 that is 297 KiB per row per decode step, plus a
// 151936-element fp16-to-fp32 conversion loop, plus — when top_p < 1 — a full
// std::sort of 151936 indices. The sort is the part that actually hurts. This
// file returns one int32 per row instead.
//
// The semantics are copied from src/model/sampler.cpp rather than re-derived,
// because the greedy path is pinned against the CPU backend token-for-token by
// tests/test_cuda_model.cpp. Specific things that are easy to get wrong and are
// deliberately handled:
//
//   - argmax ties go to the *lowest* index, because the host uses a strict `>`.
//     A reduction that takes the other side on equality fails only on ties,
//     which is to say only on the inputs a test is likely to use.
//   - the host subtracts the max before dividing by temperature, not after.
//     Both orders are algebraically equal for temperature > 0, but only one
//     reads the same as the reference.
//   - greedy is `temperature <= 0`, tested before anything else, so the RNG is
//     untouched. Reversing that order makes seeded output depend on how many
//     greedy requests shared the step.
//   - the host walks the CDF in *index* order when top_p >= 1 and in
//     descending-probability order when top_p < 1. Those pick different tokens
//     from the same uniform draw, so the two cases cannot share one walk.
//   - probabilities are never materialised. 151936 floats is 594 KiB per row,
//     which fits in no shared memory worth having, so each pass recomputes
//     expf from the logits. Fewer passes would need a global scratch buffer.
//
// Reference: Sampler::sample and Sampler::argmax in src/model/sampler.cpp.
// Divergences from it are marked DIVERGES below; the top-p ones are real.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace engine {
namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kThreads / kWarp;   // 8

// Bisection steps used to locate the nucleus threshold. The search runs over
// the exponent of tau in [2^-64, 2^0], so each step halves a range of 64
// octaves: 20 steps pin tau to a factor of 2^(64/2^20) = 1.00004, far finer
// than the gap between adjacent fp16-derived probabilities anywhere it
// matters. Each step is one more streaming pass over the row, so this constant
// is the main knob on the top-p path's cost.
constexpr int kBisectSteps = 20;
constexpr float kTauMinExp = 64.0f;   // smallest tau considered is 2^-64

// ---------------------------------------------------------------------------
// Block reductions.
//
// Same three-level shape as rmsnorm.cu: shuffle within a warp, hand the eight
// warp partials through shared memory, reduce those in warp 0. Two differences
// matter here.
//
// First, these are called many times per kernel, not once, so `partial` is
// reused. The leading __syncthreads() is what makes that safe: a thread cannot
// enter the next call and overwrite partial[] until every thread has finished
// reading the previous call's result. Every call therefore has to be reached by
// the whole block — none of them may sit inside a divergent branch.
//
// Second, every thread needs the answer, not just thread 0, so the result goes
// back out through a shared scalar.
// ---------------------------------------------------------------------------

__device__ inline float warp_reduce_sum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__device__ inline float warp_reduce_max(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}

__device__ inline float block_sum(float v) {
  __shared__ float partial[kWarpsPerBlock];
  __shared__ float result;
  __syncthreads();
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;

  v = warp_reduce_sum(v);
  if (lane == 0) partial[warp] = v;
  __syncthreads();

  // Lanes past the warp count feed in zero so the sum is unaffected.
  v = (threadIdx.x < kWarpsPerBlock) ? partial[threadIdx.x] : 0.0f;
  if (warp == 0) {
    v = warp_reduce_sum(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  return result;
}

__device__ inline float block_max(float v) {
  __shared__ float partial[kWarpsPerBlock];
  __shared__ float result;
  __syncthreads();
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;

  v = warp_reduce_max(v);
  if (lane == 0) partial[warp] = v;
  __syncthreads();

  v = (threadIdx.x < kWarpsPerBlock) ? partial[threadIdx.x] : -FLT_MAX;
  if (warp == 0) {
    v = warp_reduce_max(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  return result;
}

// Max carrying the index of the winner, with ties going to the lower index.
//
// The index rides along as a second shuffled register. `other > mine` alone
// would be wrong on equality, and `>=` would be wrong in the other direction:
// the tie has to be resolved explicitly on the index, every round, or the
// winner among several equal maxima depends on the shape of the shuffle tree.
__device__ inline void warp_reduce_argmax(float& v, int& idx) {
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, v, off);
    const int oi = __shfl_down_sync(0xffffffffu, idx, off);
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
  }
}

__device__ inline int block_argmax(float v, int idx) {
  __shared__ float pv[kWarpsPerBlock];
  __shared__ int pi[kWarpsPerBlock];
  __shared__ int result;
  __syncthreads();
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;

  warp_reduce_argmax(v, idx);
  if (lane == 0) { pv[warp] = v; pi[warp] = idx; }
  __syncthreads();

  // An inactive slot must lose on both keys: -FLT_MAX never beats a real value
  // (see the note in argmax_kernel on why that is safe for __half input), and
  // INT_MAX never wins a tie-break.
  if (threadIdx.x < kWarpsPerBlock) { v = pv[threadIdx.x]; idx = pi[threadIdx.x]; }
  else                              { v = -FLT_MAX;        idx = 0x7fffffff;     }
  if (warp == 0) {
    warp_reduce_argmax(v, idx);
    if (lane == 0) result = idx;
  }
  __syncthreads();
  return result;
}

// ---------------------------------------------------------------------------
// One token's unnormalised weight, recomputed on demand.
//
//   w[i] = exp((logit[i] - mx) / temperature),   p[i] = w[i] / sum(w)
//
// matching the host's
//
//     probs_[i] = std::exp((logits[i] - mx) / p.temperature);
//     sum += probs_[i];                      // sum is double on the host
//     probs_[i] = static_cast<float>(probs_[i] / sum);
//
// Everything downstream needs either a comparison or a ratio, so the division
// by the normaliser is folded into the other side of each comparison instead
// of being done per element.
//
// DIVERGES: the host accumulates `sum` in double and normalises in double. The
// requirement here is fp32 accumulation, so the normaliser is a float tree
// reduction over a different order. The probabilities agree to a few ulps, not
// exactly, which can flip the chosen token when the uniform draw lands within
// those few ulps of a CDF boundary. A single step disagreeing that way is
// possible; a distribution disagreeing is not.
//
// expf, not __expf: the fast intrinsic is a couple of ulps off, and this is
// being compared against a host std::exp, so the accuracy is worth the cost.
// The subtraction of mx keeps the argument <= 0, so expf cannot overflow.
// ---------------------------------------------------------------------------
__device__ inline float row_weight(const __half* row, int i, float mx, float temp) {
  return expf((__half2float(row[i]) - mx) / temp);
}

// Shared body of the greedy path, so the two kernels cannot drift apart.
__device__ inline int row_argmax(const __half* src, int vocab) {
  // Threads with no element keep -FLT_MAX. That is only safe because the input
  // is __half: the most negative finite half is -65504, so a real logit can
  // never tie with the sentinel and steal the tie-break. If the head ever
  // emits fp32 logits this sentinel has to become -inf with an explicit
  // has-any-element flag.
  float best = -FLT_MAX;
  int best_i = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += kThreads) {
    const float v = __half2float(src[i]);
    if (v > best || (v == best && i < best_i)) { best = v; best_i = i; }
  }
  return block_argmax(best, best_i);
}

}  // namespace

// ---------------------------------------------------------------------------
// Greedy: per-row argmax. One block per row.
//
// Matches
//     int32_t best = 0;
//     for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
// exactly, including the lowest-index tie-break. No known divergence.
// ---------------------------------------------------------------------------
__global__ void argmax_kernel(const __half* __restrict__ logits,
                              int32_t* __restrict__ out, int vocab) {
  const int row = blockIdx.x;
  const int winner = row_argmax(logits + static_cast<size_t>(row) * vocab, vocab);
  if (threadIdx.x == 0) out[row] = static_cast<int32_t>(winner);
}

// ---------------------------------------------------------------------------
// Temperature + top-p, one block per row.
//
// Why the caller supplies the uniform draw instead of curand:
//
// The host owns the RNG stream and has to keep owning it. SamplingParams::seed
// is part of the public API and tests/test_sampler.cpp pins it — a seeded
// request must produce the same token whatever else happens to be in the batch.
// A per-thread curand state cannot give that: the sequence a row sees would
// depend on its position in the batch, on the block shape, and on how many rows
// the scheduler happened to pack, none of which the host can reproduce. So the
// host draws exactly as Sampler::sample does — a fresh mt19937_64 seeded per
// request when seed != 0, the engine's rolling generator otherwise — and passes
// one float per row. The device does no randomness at all, which also makes a
// disagreement with the CPU backend debuggable: feed both the same draw and the
// only remaining variable is arithmetic.
//
// `rand_uniform[row]` must be in [0, 1). The nucleus branch scales it by the
// kept mass, because the host draws uniform(0, cum) directly rather than
// renormalising.
// ---------------------------------------------------------------------------
__global__ void sample_kernel(const __half* __restrict__ logits,
                              const float* __restrict__ temperature,
                              const float* __restrict__ top_p,
                              const float* __restrict__ rand_uniform,
                              int32_t* __restrict__ out, int vocab) {
  // Chunk totals for the index-order CDF walk, plus scalars broadcast from
  // thread 0. Shared memory bound, all of it statically sized: kThreads + 1
  // floats + 2 ints = 256*4 + 12 = 1036 B here, and 76 B of reduction scratch
  // across the three helpers above (8 floats + 4 for block_sum, the same for
  // block_max, 8 floats + 9 ints for block_argmax). Under 1.2 KiB per block,
  // so occupancy is set by registers, not by this.
  __shared__ float s_chunk[kThreads];
  __shared__ float s_base;
  __shared__ int s_sel_chunk;
  __shared__ int s_result;

  const int row = blockIdx.x;
  const __half* src = logits + static_cast<size_t>(row) * vocab;
  const float temp = temperature[row];
  const float tp = top_p[row];

  // Greedy first, before the draw is even read, exactly as the host does.
  // The whole block takes the same side of this branch, so the reductions
  // inside remain block-uniform.
  if (temp <= 0.0f) {
    const int winner = row_argmax(src, vocab);
    if (threadIdx.x == 0) out[row] = static_cast<int32_t>(winner);
    return;
  }

  // Pass 1: the row max. Strided and coalesced; max does not care about order.
  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < vocab; i += kThreads) m = fmaxf(m, __half2float(src[i]));
  const float mx = block_max(m);

  // Pass 2: the normaliser.
  float acc = 0.0f;
  for (int i = threadIdx.x; i < vocab; i += kThreads) acc += row_weight(src, i, mx, temp);
  const float wsum = block_sum(acc);

  // The contiguous chunk this thread owns during the CDF walk. Contiguous, not
  // strided: the host walks the CDF in index order, so the prefix sums have to
  // be orderable by thread id. That costs coalescing — a warp touches 32
  // separate ~1.2 KiB spans — and it is the right trade, because the
  // index-order walk *is* the semantics and a strided walk would not be one.
  const int chunk = (vocab + kThreads - 1) / kThreads;
  const int begin = threadIdx.x * chunk;
  const int end = (begin + chunk < vocab) ? (begin + chunk) : vocab;

  // `thresh` is the unnormalised cutoff: a token is a candidate when its weight
  // is >= thresh. Zero keeps everything, which is the top_p >= 1 case.
  float thresh = 0.0f;
  float cum = wsum;          // unnormalised mass of the candidate set
  int fallback = vocab - 1;  // host's fallback when the walk runs off the end

  if (tp < 1.0f) {
    // ---------------------------------------------------------------------
    // Nucleus selection, matching
    //     std::sort(idx, probs descending);
    //     for (keep = 0; keep < vocab; keep++) { cum += probs[idx[keep]];
    //                                            if (cum >= top_p) { keep++; break; } }
    // without the sort. The sorted prefix of length `keep` is exactly the set
    // { i : p[i] >= p[idx[keep-1]] }, so finding that threshold finds the set.
    //
    // Bisect on the exponent of tau rather than on tau itself: the interesting
    // thresholds span many orders of magnitude, and a linear bisection on
    // [0, 1] would spend every step inside the top octave. The invariant below
    // is mass(2^-lo) < top_p <= mass(2^-hi), with hi the accepted side, so the
    // set finally kept always carries at least top_p of the mass. (The
    // lo-side half of that invariant does not hold when the single most likely
    // token already exceeds top_p; in that case hi converges to 0, the kept set
    // is just the argmax, and that is the right answer.)
    //
    // DIVERGES, two ways, both documented rather than hidden:
    //
    //   1. Ties at the boundary. The host's std::sort is unstable, so when
    //      several tokens share the boundary probability it keeps an arbitrary
    //      subset of them — however many it needs to cross top_p. This keeps
    //      *all* of them, so the kept set can be a strict superset and the
    //      kept mass slightly larger. Exact ties are rare in real logits and
    //      common in hand-written test vectors.
    //
    //   2. Threshold precision. tau is pinned to within a factor of 1.00004,
    //      not exactly, so a token whose probability sits inside that window
    //      below the true boundary is kept when the host would drop it. Its
    //      probability is by construction the smallest in the nucleus, so the
    //      effect on the sampled distribution is bounded by that one token's
    //      mass.
    // ---------------------------------------------------------------------
    float lo = 0.0f;              // tau = 2^-lo, the rejected side
    float hi = kTauMinExp;        // tau = 2^-hi, the accepted side
    for (int step = 0; step < kBisectSteps; step++) {
      const float mid = 0.5f * (lo + hi);
      const float t = exp2f(-mid) * wsum;
      float part = 0.0f;
      for (int i = threadIdx.x; i < vocab; i += kThreads) {
        const float w = row_weight(src, i, mx, temp);
        if (w >= t) part += w;
      }
      const float mass = block_sum(part);
      // Compared as mass >= tp * wsum rather than mass / wsum >= tp: one
      // multiply instead of a divide, and the same comparison.
      if (mass >= tp * wsum) hi = mid; else lo = mid;
    }
    thresh = exp2f(-hi) * wsum;

    // One more pass, for two things at once: the kept mass, recomputed at the
    // final threshold so `cum` and the walk below agree term for term, and the
    // highest kept index.
    //
    // The host's fallback, when floating-point error leaves the draw unspent,
    // is idx[keep - 1] — the *lowest-probability* kept token.
    //
    // DIVERGES: finding that token needs the sorted order. This uses the
    // highest-index kept token instead, which the pass is already positioned to
    // find. It is only ever returned when the subtraction loop runs off the end
    // of the kept set, which takes a draw within rounding of cum.
    float mass = 0.0f;
    float last = -1.0f;
    for (int i = threadIdx.x; i < vocab; i += kThreads) {
      const float w = row_weight(src, i, mx, temp);
      if (w >= thresh) { mass += w; last = fmaxf(last, static_cast<float>(i)); }
    }
    cum = block_sum(mass);
    // Indices up to 151936 are exact in fp32 (integers are exact below 2^24),
    // so reducing them as floats loses nothing.
    const float last_f = block_max(last);

    // The candidate set can come back empty only in the degenerate case where
    // the most likely token already carries top_p of the mass on its own: hi
    // converges towards 0, thresh towards wsum, and a single ulp of rounding in
    // exp2f(-hi) * wsum can then push the cutoff just past the one weight that
    // should have qualified. The correct answer there is the argmax, so take it
    // directly rather than letting the walk run off the end and return
    // vocab - 1, which would be a silently wrong token rather than a visibly
    // wrong one. last_f is block-uniform, so this branch stays block-uniform
    // and the reduction inside row_argmax is safe.
    if (last_f < 0.0f) {
      const int winner = row_argmax(src, vocab);
      if (threadIdx.x == 0) out[row] = static_cast<int32_t>(winner);
      return;
    }
    fallback = static_cast<int>(last_f);
  }

  // ---------------------------------------------------------------------
  // Index-order CDF walk over the candidate set, matching the host's
  //     r -= probs[i]; if (r <= 0) return i;
  // loop. The host runs that over the whole vocabulary in index order when
  // top_p >= 1, and over the descending-probability order when top_p < 1.
  //
  // DIVERGES, and this is the one to know about: for top_p < 1 this walks the
  // nucleus in index order, not in descending-probability order. Reproducing
  // the host's order would mean sorting the nucleus, which is unbounded in
  // size and is the thing this kernel exists to avoid. The consequence is
  // precise and limited: the *distribution* sampled is identical — every kept
  // token still owns exactly its own share of [0, cum) — but a given uniform
  // draw maps to a different token than it would on the CPU backend. Seeded
  // output is reproducible run to run on the GPU, and reproducible run to run
  // on the CPU; it is not identical *between* the two backends when top_p < 1.
  // Nothing in the test suite pins that today (test_cuda_model compares only
  // greedy), but it is a behavioural difference rather than an implementation
  // detail, and it belongs in the README next to the backend flag.
  // ---------------------------------------------------------------------
  const float r = rand_uniform[row] * cum;   // host: uniform_real_distribution(0, cum)

  // Each thread totals its own chunk.
  float mine = 0.0f;
  for (int i = begin; i < end; i++) {
    const float w = row_weight(src, i, mx, temp);
    if (w >= thresh) mine += w;
  }
  s_chunk[threadIdx.x] = mine;
  if (threadIdx.x == 0) { s_sel_chunk = -1; s_base = 0.0f; s_result = fallback; }
  __syncthreads();

  // Thread 0 finds the chunk the draw lands in. 256 sequential adds, set
  // against passes over 151936 elements on either side of it; a parallel scan
  // here would be invisible. The `> 0` guard skips chunks holding no candidate,
  // which otherwise match the `>= r` test when r is 0.
  if (threadIdx.x == 0) {
    float base = 0.0f;
    for (int c = 0; c < kThreads; c++) {
      if (s_chunk[c] > 0.0f && base + s_chunk[c] >= r) { s_sel_chunk = c; break; }
      base += s_chunk[c];
    }
    s_base = base;
  }
  __syncthreads();

  // The owning thread walks its own chunk, subtracting in index order.
  if (s_sel_chunk >= 0 && static_cast<int>(threadIdx.x) == s_sel_chunk) {
    float rr = r - s_base;
    for (int i = begin; i < end; i++) {
      const float w = row_weight(src, i, mx, temp);
      if (w < thresh) continue;
      rr -= w;
      if (rr <= 0.0f) { s_result = i; break; }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) out[row] = static_cast<int32_t>(s_result);
}

// ---------------------------------------------------------------------------
// Launchers.
// ---------------------------------------------------------------------------

// Greedy argmax over each row of a [rows, vocab] __half logits tensor —
// the layout the lm_head GEMM already writes. `out` is int32 device memory
// with `rows` entries.
void launch_argmax(const __half* logits, int32_t* out, int rows, int vocab,
                   cudaStream_t stream) {
  if (rows <= 0 || vocab <= 0) return;   // a zero-block grid is an invalid launch
  argmax_kernel<<<rows, kThreads, 0, stream>>>(logits, out, vocab);
}

// Greedy or temperature + top-p, chosen per row by temperature <= 0.
//
// temperature, top_p and rand_uniform are device arrays of `rows` floats:
// SamplingParams::temperature and ::top_p for each row that needed logits, in
// the same slice order the head wrote them, and one uniform draw in [0, 1)
// taken by the host from the generator Sampler::sample would have used. Rows
// with temperature <= 0 ignore their draw, so the host must not advance the
// generator for them either — see the RNG note above sample_kernel.
//
// `out` is int32 device memory with `rows` entries.
void launch_sample(const __half* logits, const float* temperature, const float* top_p,
                   const float* rand_uniform, int32_t* out, int rows, int vocab,
                   cudaStream_t stream) {
  if (rows <= 0 || vocab <= 0) return;
  sample_kernel<<<rows, kThreads, 0, stream>>>(logits, temperature, top_p,
                                               rand_uniform, out, vocab);
}

}  // namespace engine
