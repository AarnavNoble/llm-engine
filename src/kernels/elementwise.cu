// The two small elementwise steps that stand between the big kernels: adding a
// projection's bias, and adding a block's output back into the residual stream.
//
// Neither is interesting on its own. They exist because once every other op is
// a kernel, doing these two on the host would mean copying the whole activation
// tensor across PCIe twice per layer, which would cost far more than the work
// itself. Keeping the data resident is the entire point.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace engine {
namespace {
constexpr int kThreads = 256;
}  // namespace

// x[i] += y[i], the residual connection.
__global__ void add_inplace_kernel(__half* x, const __half* y, size_t n) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

// x[r][c] += bias[c]. The bias is fp32 because it is tiny and the reference
// adds it in fp32; rounding it to half first would show up in the comparison.
__global__ void add_bias_kernel(__half* x, const float* bias, int rows, int cols) {
  const size_t n = static_cast<size_t>(rows) * cols;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    x[i] = __float2half(__half2float(x[i]) + bias[i % cols]);
}

void launch_add_inplace(__half* x, const __half* y, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  const size_t blocks = (n + kThreads - 1) / kThreads;
  add_inplace_kernel<<<static_cast<int>(blocks < 4096 ? blocks : 4096), kThreads, 0, stream>>>(x, y, n);
}

void launch_add_bias(__half* x, const float* bias, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || bias == nullptr) return;
  const size_t n = static_cast<size_t>(rows) * cols;
  const size_t blocks = (n + kThreads - 1) / kThreads;
  add_bias_kernel<<<static_cast<int>(blocks < 4096 ? blocks : 4096), kThreads, 0, stream>>>(
      x, bias, rows, cols);
}

}  // namespace engine
