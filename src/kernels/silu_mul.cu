// The elementwise half of SwiGLU: gate[i] = silu(gate[i]) * up[i],
// where silu(x) = x / (1 + e^-x).
//
// Two things to notice. The first is the grid-stride loop, which is the other
// standard decomposition alongside one-block-per-row: the grid is sized to the
// device rather than to the data, and each thread walks the array in strides of
// the whole grid. That keeps the launch shape fixed whatever N is, and keeps
// every warp's accesses contiguous.
//
// The second is that this is fused on purpose. As two kernels, silu then
// multiply, the data crosses memory three times. Fused it is two reads and one
// write, and at N=4096 by intermediate=4864 that saves tens of megabytes per
// layer per step, twenty-four layers deep. The op is entirely bandwidth-bound,
// so traffic is the only thing that matters.
//
// Reference: ref::silu_mul in include/engine/ops.h.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace engine {
namespace {

constexpr int kThreads = 256;

// The sigmoid is computed in float, not half. __expf on a half-precision value
// loses accuracy where it matters most, around zero, and the reference uses the
// float form; matching it is what keeps the comparison meaningful.
__device__ inline float silu(float x) { return x / (1.0f + __expf(-x)); }

}  // namespace

__global__ void silu_mul_kernel(__half* gate, const __half* up, size_t n) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    gate[i] = __float2half(silu(g) * u);
  }
}

void launch_silu_mul(__half* gate, const __half* up, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  // Cap the grid rather than sizing it to n: past a few thousand blocks there
  // is nothing left to gain, and the stride loop handles any remainder. The
  // cap also keeps the launch valid for very large n, where n/kThreads would
  // overflow the grid dimension.
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  silu_mul_kernel<<<blocks < 4096 ? blocks : 4096, kThreads, 0, stream>>>(gate, up, n);
}

}  // namespace engine
