// CUDA backend. This file is the host side: it owns device memory, uploads the
// weights, and orchestrates the per-step work. The maths lives in src/kernels/.
//
// It is a port of CpuModel::forward, deliberately keeping the same packed step
// layout and the same KV cache layout, so slot(pos) is identical arithmetic on
// both sides and a disagreement can never be blamed on the two backends
// organising memory differently.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/model.h"
#include "engine/safetensors.h"

namespace engine {
namespace {

// CUDA fails silently and asynchronously: an unchecked error from one kernel
// surfaces as wrong numbers in a different one. Every API call goes through
// these from the first line of code.
#define CUDA_CHECK(x)                                                              \
  do {                                                                             \
    cudaError_t e_ = (x);                                                          \
    if (e_ != cudaSuccess)                                                         \
      throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                               " " + cudaGetErrorString(e_));                      \
  } while (0)

#define CUBLAS_CHECK(x)                                                            \
  do {                                                                             \
    cublasStatus_t s_ = (x);                                                       \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                               \
      throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                               " cuBLAS status " + std::to_string(int(s_)));       \
  } while (0)

size_t g_allocated = 0;

template <typename T>
T* device_alloc(size_t count) {
  T* p = nullptr;
  const size_t bytes = count * sizeof(T);
  const cudaError_t e = cudaMalloc(&p, bytes);
  if (e != cudaSuccess) {
    // "The pool is too big for this card" is a configuration mistake and should
    // read like one, with the arithmetic, rather than like a crash.
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    throw std::runtime_error("cudaMalloc of " + std::to_string(bytes >> 20) +
                             " MiB failed: " + cudaGetErrorString(e) + ". Device has " +
                             std::to_string(free_b >> 20) + " MiB free of " +
                             std::to_string(total_b >> 20) + " MiB.");
  }
  g_allocated += bytes;
  return p;
}

// Weights ship as bf16 and the device wants fp16. Norm weights stay fp32:
// they are tiny, and the reference computes the normalisation in fp32.
__half* upload_f16(const TensorView& t) {
  const std::vector<uint16_t> h = t.to_f16();
  __half* d = device_alloc<__half>(h.size());
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  return d;
}

float* upload_f32(const TensorView& t) {
  const Tensor h = t.to_f32();
  float* d = device_alloc<float>(h.numel());
  CUDA_CHECK(cudaMemcpy(d, h.ptr(), h.numel() * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}

struct CudaLayer {
  float* in_norm = nullptr;
  float* post_norm = nullptr;
  __half* wq = nullptr;
  __half* wk = nullptr;
  __half* wv = nullptr;
  __half* wo = nullptr;
  __half* bq = nullptr;   // null on models without qkv bias, such as Llama
  __half* bk = nullptr;
  __half* bv = nullptr;
  __half* wgate = nullptr;
  __half* wup = nullptr;
  __half* wdown = nullptr;
};

class CudaModel final : public Model {
 public:
  CudaModel(const std::string& dir, const KVCacheManager& kv)
      : cfg_(ModelConfig::load(dir)), kv_(kv) {
    CUDA_CHECK(cudaSetDevice(0));
    CUBLAS_CHECK(cublasCreate(&blas_));
    load_weights(dir);
    report_memory();
  }

  ~CudaModel() override {
    if (blas_) cublasDestroy(blas_);
  }

  const ModelConfig& config() const override { return cfg_; }
  const char* backend() const override { return "cuda"; }

  void forward(const StepInput&, std::vector<float>&) override {
    throw std::runtime_error("cuda forward not implemented yet");
  }

 private:
  void load_weights(const std::string& dir) {
    SafeTensorsDir w(dir);
    embed_ = upload_f16(w.get("model.embed_tokens.weight"));
    final_norm_ = upload_f32(w.get("model.norm.weight"));
    // A tied head is the embedding table read a second time, not a copy of it.
    lm_head_ = cfg_.tie_word_embeddings ? embed_ : upload_f16(w.get("lm_head.weight"));

    layers_.resize(cfg_.num_hidden_layers);
    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      CudaLayer& L = layers_[l];
      L.in_norm = upload_f32(w.get(p + "input_layernorm.weight"));
      L.post_norm = upload_f32(w.get(p + "post_attention_layernorm.weight"));
      L.wq = upload_f16(w.get(p + "self_attn.q_proj.weight"));
      L.wk = upload_f16(w.get(p + "self_attn.k_proj.weight"));
      L.wv = upload_f16(w.get(p + "self_attn.v_proj.weight"));
      L.wo = upload_f16(w.get(p + "self_attn.o_proj.weight"));
      if (w.has(p + "self_attn.q_proj.bias")) {
        L.bq = upload_f16(w.get(p + "self_attn.q_proj.bias"));
        L.bk = upload_f16(w.get(p + "self_attn.k_proj.bias"));
        L.bv = upload_f16(w.get(p + "self_attn.v_proj.bias"));
      }
      L.wgate = upload_f16(w.get(p + "mlp.gate_proj.weight"));
      L.wup = upload_f16(w.get(p + "mlp.up_proj.weight"));
      L.wdown = upload_f16(w.get(p + "mlp.down_proj.weight"));
    }
  }

  void report_memory() const {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("cuda: %s, weights %zu MiB, device %zu MiB free of %zu MiB\n",
                cfg_.model_type.c_str(), g_allocated >> 20, free_b >> 20, total_b >> 20);
    std::fflush(stdout);
  }

  // C[m,n] = A[m,k] * B[n,k]^T, all row-major, fp16 data with fp32 accumulate.
  //
  // cuBLAS is column-major while this data is row-major, so ask it for C^T,
  // which occupies the same bytes as C. Since C^T = B * A^T, B comes first with
  // OP_T and A second with OP_N, and n and m swap places. Verified against a
  // hand-computed 2x3 case before anything depended on it.
  void gemm(const __half* A, const __half* B, __half* C, int m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                              B, CUDA_R_16F, k, A, CUDA_R_16F, k, &beta,
                              C, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }

  ModelConfig cfg_;
  const KVCacheManager& kv_;
  cublasHandle_t blas_ = nullptr;

  __half* embed_ = nullptr;
  __half* lm_head_ = nullptr;
  float* final_norm_ = nullptr;
  std::vector<CudaLayer> layers_;
};

}  // namespace

std::unique_ptr<Model> make_cuda_model(const std::string& dir, const KVCacheManager& kv) {
  return std::make_unique<CudaModel>(dir, kv);
}

}  // namespace engine
