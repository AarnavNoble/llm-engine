// RMSNorm: out[r][j] = x[r][j] * rsqrt(mean_j(x[r][j]^2) + eps) * weight[j]
//
// The formula is trivial; the interesting part is that 896 values have to be
// summed into one number that all 256 threads then need. That is a parallel
// reduction, and the same three-level pattern appears again inside attention's
// softmax, so it is worth reading closely.
//
// Reference: ref::rmsnorm in include/engine/ops.h.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace engine {
namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = kThreads / kWarp;   // 8

// Level 2: sum across the 32 lanes of one warp.
//
// A warp executes in lockstep, so its threads can read each other's registers
// directly, with no memory involved. __shfl_down_sync(mask, v, off) returns the
// v held by the lane `off` higher than this one.
//
//   off=16: lanes 0..15 add in the values from lanes 16..31
//   off=8 : lanes 0..7  absorb lanes 8..15
//   ... halving each round
//
// After five rounds lane 0 holds the total. The mask says which lanes take
// part; 0xffffffff is all of them, correct here because every thread in the
// warp reaches this line.
__device__ inline float warp_reduce_sum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Level 3: sum across the whole block.
//
// After the warp reduction there are 8 partial sums, one per warp, each sitting
// in a different lane 0. Shared memory is a scratchpad visible to every thread
// in the block, so the warps hand their partials over through it and warp 0
// reduces those.
//
// __syncthreads() is a barrier: no thread passes until all have arrived.
// Without it, warp 0 could read partial[] slots other warps have not written
// yet. Every shared-memory write that another thread reads needs one.
//
// The return value is only meaningful in thread 0.
__device__ inline float block_reduce_sum(float v) {
  __shared__ float partial[kWarpsPerBlock];
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;

  v = warp_reduce_sum(v);
  if (lane == 0) partial[warp] = v;
  __syncthreads();

  // Only warp 0 does the final step. Lanes past the warp count feed in zero so
  // the sum is unaffected.
  v = (threadIdx.x < kWarpsPerBlock) ? partial[threadIdx.x] : 0.0f;
  if (warp == 0) v = warp_reduce_sum(v);
  return v;
}

}  // namespace

__global__ void rmsnorm_kernel(const __half* x, const float* weight, __half* out,
                               int cols, float eps) {
  // One block per row, so blockIdx.x is simply the row this block owns.
  const int row = blockIdx.x;
  const __half* src = x + static_cast<size_t>(row) * cols;
  __half* dst = out + static_cast<size_t>(row) * cols;

  // Level 1: each thread sums the squares of its own strided slice. Loading as
  // half but accumulating in float is not optional: 896 terms of O(1) values
  // accumulated in fp16 lose enough precision to look like a logic bug.
  float acc = 0.0f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    const float v = __half2float(src[j]);
    acc += v * v;
  }

  const float total = block_reduce_sum(acc);

  // Thread 0 holds the sum but every thread needs the scale, so it goes back
  // out through shared memory. rsqrtf is a single hardware instruction, and eps
  // belongs inside the square root rather than added to its result.
  __shared__ float inv_rms;
  if (threadIdx.x == 0) inv_rms = rsqrtf(total / static_cast<float>(cols) + eps);
  __syncthreads();
  const float inv = inv_rms;

  // Second pass over the row. Consecutive threads touch consecutive addresses,
  // so both the read and the write stay coalesced.
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    dst[j] = __float2half(__half2float(src[j]) * inv * weight[j]);
  }
}

void launch_rmsnorm(const __half* x, const float* weight, __half* out,
                    int rows, int cols, float eps, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;   // a zero-block grid is an invalid launch
  rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(x, weight, out, cols, eps);
}

}  // namespace engine
