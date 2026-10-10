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

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/cuda_kernels.h"
#include "engine/model.h"
#include "engine/ops.h"
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

// A weight that lives on the device, with a host copy kept only when it is
// small enough not to matter and the reference needs it (norms and biases).
struct Weight {
  __half* dev = nullptr;
  Tensor host;                      // empty unless a host copy was requested
  bool present() const { return dev != nullptr; }
};

struct CudaLayer {
  Tensor in_norm, post_norm;        // fp32 on the host; the reference normalises in fp32
  Tensor bq, bk, bv;                // host copies, tiny and added after the GEMM
  float* in_norm_dev = nullptr;
  float* post_norm_dev = nullptr;
  __half* wq = nullptr;
  __half* wk = nullptr;
  __half* wv = nullptr;
  __half* wo = nullptr;
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
    rope_.build(cfg_.head_dim, cfg_.max_position_embeddings, cfg_.rope_theta);
    kv_dim_ = cfg_.num_key_value_heads * cfg_.head_dim;
    const size_t slots = static_cast<size_t>(kv.num_total_blocks()) * kv.block_size();
    k_cache_.assign(cfg_.num_hidden_layers, std::vector<float>(slots * kv_dim_, 0.f));
    v_cache_.assign(cfg_.num_hidden_layers, std::vector<float>(slots * kv_dim_, 0.f));
    report_memory();
  }

  ~CudaModel() override {
    if (blas_) cublasDestroy(blas_);
  }

  const ModelConfig& config() const override { return cfg_; }
  const char* backend() const override { return "cuda"; }

  // Step one of the port: the projections run on the device through cuBLAS,
  // everything else goes through the reference in ops.h via host round-trips.
  // Slow and pointless as a product, but it proves the plumbing — layout,
  // strides, weight upload, the transpose convention, residuals, the logits
  // gather — before any kernel is in question. Each round-trip is then replaced
  // by one kernel at a time, so nothing is ever debugged alongside something
  // else new.
  void forward(const StepInput& step, std::vector<float>& logits) override {
    logits.assign(static_cast<size_t>(step.num_logits) * cfg_.vocab_size, 0.f);
    if (step.slices.empty()) return;

    const int H = cfg_.hidden_size, hd = cfg_.head_dim, nh = cfg_.num_attention_heads,
              nkv = cfg_.num_key_value_heads, I = cfg_.intermediate_size,
              group = cfg_.gqa_group(), bs = kv_.block_size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    flatten(step);
    const int N = static_cast<int>(tok_id_.size());

    x_.assign(static_cast<size_t>(N) * H, 0.f);
    embed_tokens(N, x_.data());

    h_.resize(static_cast<size_t>(N) * H);
    q_.resize(static_cast<size_t>(N) * H);
    k_.resize(static_cast<size_t>(N) * kv_dim_);
    v_.resize(static_cast<size_t>(N) * kv_dim_);
    attn_.resize(static_cast<size_t>(N) * H);
    o_.resize(static_cast<size_t>(N) * H);
    gate_.resize(static_cast<size_t>(N) * I);
    up_.resize(static_cast<size_t>(N) * I);

    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      const CudaLayer& L = layers_[l];
      float* kc = k_cache_[l].data();
      float* vc = v_cache_[l].data();

      ref::rmsnorm(x_.data(), L.in_norm.ptr(), h_.data(), N, H, cfg_.rms_norm_eps);
      linear(h_.data(), N, H, L.wq, H, L.bq, q_.data());
      linear(h_.data(), N, H, L.wk, kv_dim_, L.bk, k_.data());
      linear(h_.data(), N, H, L.wv, kv_dim_, L.bv, v_.data());

      // Rotate and write K/V for every token first, so tokens that arrived in
      // this same step can attend to one another.
      for (int i = 0; i < N; i++) {
        const int pos = tok_pos_[i];
        ref::rope_inplace(&q_[static_cast<size_t>(i) * H], nh, hd, pos, rope_);
        ref::rope_inplace(&k_[static_cast<size_t>(i) * kv_dim_], nkv, hd, pos, rope_);
        const int64_t slot = tok_seq_[i]->block_table.slot(pos, bs);
        std::copy_n(&k_[static_cast<size_t>(i) * kv_dim_], kv_dim_,
                    kc + static_cast<size_t>(slot) * kv_dim_);
        std::copy_n(&v_[static_cast<size_t>(i) * kv_dim_], kv_dim_,
                    vc + static_cast<size_t>(slot) * kv_dim_);
      }

      for (int i = 0; i < N; i++) {
        for (int hh = 0; hh < nh; hh++) {
          ref::paged_attention_head(&q_[static_cast<size_t>(i) * H + hh * hd], kc, vc,
                                    tok_seq_[i]->block_table, bs, tok_pos_[i], hh / group,
                                    nkv, hd, scale,
                                    &attn_[static_cast<size_t>(i) * H + hh * hd], scratch_);
        }
      }

      linear(attn_.data(), N, H, L.wo, H, Tensor(), o_.data());
      for (size_t i = 0; i < x_.size(); i++) x_[i] += o_[i];

      ref::rmsnorm(x_.data(), L.post_norm.ptr(), h_.data(), N, H, cfg_.rms_norm_eps);
      linear(h_.data(), N, H, L.wgate, I, Tensor(), gate_.data());
      linear(h_.data(), N, H, L.wup, I, Tensor(), up_.data());
      ref::silu_mul(gate_.data(), up_.data(), gate_.size());
      linear(gate_.data(), N, I, L.wdown, H, Tensor(), o_.data());
      for (size_t i = 0; i < x_.size(); i++) x_[i] += o_[i];
    }

    if (logit_rows_.empty()) return;
    const int R = static_cast<int>(logit_rows_.size());
    h_.resize(static_cast<size_t>(R) * H);
    for (int r = 0; r < R; r++)
      ref::rmsnorm(&x_[static_cast<size_t>(logit_rows_[r]) * H], final_norm_.ptr(),
                   &h_[static_cast<size_t>(r) * H], 1, H, cfg_.rms_norm_eps);
    linear(h_.data(), R, H, lm_head_, cfg_.vocab_size, Tensor(), logits.data());
  }

 private:
  // Per-token view of the step, in slice order: the same packing the CPU model
  // uses, and the layout the kernels will consume.
  void flatten(const StepInput& step) {
    tok_seq_.clear();
    tok_pos_.clear();
    tok_id_.clear();
    logit_rows_.clear();
    for (const auto& sl : step.slices) {
      for (int t = 0; t < sl.len; t++) {
        tok_seq_.push_back(sl.seq);
        tok_pos_.push_back(sl.start + t);
        tok_id_.push_back(sl.seq->tokens[sl.start + t]);
      }
      if (sl.needs_logits) logit_rows_.push_back(static_cast<int>(tok_id_.size()) - 1);
    }
  }

  // Gather every token's embedding row in one kernel launch, then bring the
  // result back for the ops that are still running on the host. One bulk copy
  // instead of a memcpy per token.
  void embed_tokens(int N, float* out) {
    const int H = cfg_.hidden_size;
    const size_t count = static_cast<size_t>(N) * H;

    if (d_token_ids_cap_ < static_cast<size_t>(N)) {
      if (d_token_ids_) CUDA_CHECK(cudaFree(d_token_ids_));
      d_token_ids_ = device_alloc<int32_t>(static_cast<size_t>(N));
      d_token_ids_cap_ = static_cast<size_t>(N);
    }
    ensure_capacity(&d_x_, &d_x_cap_, count);

    CUDA_CHECK(cudaMemcpy(d_token_ids_, tok_id_.data(),
                          static_cast<size_t>(N) * sizeof(int32_t), cudaMemcpyHostToDevice));
    launch_embedding(embed_, d_token_ids_, d_x_, N, H, nullptr);
    CUDA_CHECK(cudaGetLastError());        // catches a bad launch configuration
    CUDA_CHECK(cudaDeviceSynchronize());   // attribute any fault to this kernel

    stage_c_.resize(count);
    CUDA_CHECK(cudaMemcpy(stage_c_.data(), d_x_, count * sizeof(__half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) out[i] = f16_to_f32(stage_c_[i]);
  }

  void ensure_capacity(__half** buf, size_t* cap, size_t need) {
    if (*cap >= need) return;
    if (*buf) CUDA_CHECK(cudaFree(*buf));
    *buf = device_alloc<__half>(need);
    *cap = need;
  }

  // out[m,n] = in[m,k] * W[n,k]^T + bias, with the GEMM on the device and the
  // activations converted at the boundary. The conversions are the cost of an
  // intermediate step; they disappear as the surrounding ops become kernels.
  void linear(const float* in, int m, int k, const __half* W, int n, const Tensor& bias,
              float* out) {
    const size_t a_count = static_cast<size_t>(m) * k;
    const size_t c_count = static_cast<size_t>(m) * n;
    ensure_capacity(&dA_, &dA_cap_, a_count);
    ensure_capacity(&dC_, &dC_cap_, c_count);

    stage_a_.resize(a_count);
    for (size_t i = 0; i < a_count; i++) stage_a_[i] = f32_to_f16(in[i]);
    CUDA_CHECK(cudaMemcpy(dA_, stage_a_.data(), a_count * sizeof(__half), cudaMemcpyHostToDevice));

    gemm(dA_, W, dC_, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    stage_c_.resize(c_count);
    CUDA_CHECK(cudaMemcpy(stage_c_.data(), dC_, c_count * sizeof(__half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < c_count; i++) out[i] = f16_to_f32(stage_c_[i]);

    if (bias.numel() == static_cast<size_t>(n))
      for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) out[static_cast<size_t>(r) * n + c] += bias.ptr()[c];
  }

  void load_weights(const std::string& dir) {
    SafeTensorsDir w(dir);
    embed_ = upload_f16(w.get("model.embed_tokens.weight"));
    final_norm_dev_ = upload_f32(w.get("model.norm.weight"));
    final_norm_ = w.get("model.norm.weight").to_f32();
    // A tied head is the embedding table read a second time, not a copy of it.
    lm_head_ = cfg_.tie_word_embeddings ? embed_ : upload_f16(w.get("lm_head.weight"));

    layers_.resize(cfg_.num_hidden_layers);
    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      CudaLayer& L = layers_[l];
      L.in_norm_dev = upload_f32(w.get(p + "input_layernorm.weight"));
      L.post_norm_dev = upload_f32(w.get(p + "post_attention_layernorm.weight"));
      L.in_norm = w.get(p + "input_layernorm.weight").to_f32();
      L.post_norm = w.get(p + "post_attention_layernorm.weight").to_f32();
      L.wq = upload_f16(w.get(p + "self_attn.q_proj.weight"));
      L.wk = upload_f16(w.get(p + "self_attn.k_proj.weight"));
      L.wv = upload_f16(w.get(p + "self_attn.v_proj.weight"));
      L.wo = upload_f16(w.get(p + "self_attn.o_proj.weight"));
      if (w.has(p + "self_attn.q_proj.bias")) {
        L.bq = w.get(p + "self_attn.q_proj.bias").to_f32();
        L.bk = w.get(p + "self_attn.k_proj.bias").to_f32();
        L.bv = w.get(p + "self_attn.v_proj.bias").to_f32();
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
  float* final_norm_dev_ = nullptr;
  Tensor final_norm_;
  std::vector<CudaLayer> layers_;

  // Device scratch for the GEMMs, grown on demand and then reused. Allocating
  // inside forward() every step is the classic way to make a GPU program
  // mysteriously slow.
  __half* dA_ = nullptr;
  __half* dC_ = nullptr;
  size_t dA_cap_ = 0, dC_cap_ = 0;
  std::vector<uint16_t> stage_a_, stage_c_;

  // Per-step token ids on the device, and the embedding output. Grown on
  // demand like the GEMM scratch.
  int32_t* d_token_ids_ = nullptr;
  size_t d_token_ids_cap_ = 0;
  __half* d_x_ = nullptr;
  size_t d_x_cap_ = 0;

  // The KV cache still lives on the host while attention is the reference
  // implementation. It moves to the device with attention_decode.cu, and the
  // layout is already the one the kernel will use:
  // [slot][kv_head][head_dim], slot = block_table[pos / block_size] * block_size + pos % block_size.
  int kv_dim_ = 0;
  std::vector<std::vector<float>> k_cache_, v_cache_;

  // Per-token view of the step, identical to the CPU packing.
  std::vector<const Sequence*> tok_seq_;
  std::vector<int> tok_pos_;
  std::vector<int32_t> tok_id_;
  std::vector<int> logit_rows_;
  std::vector<float> x_, h_, q_, k_, v_, attn_, o_, gate_, up_, scratch_;
  ref::RopeTables rope_;
};

}  // namespace

std::unique_ptr<Model> make_cuda_model(const std::string& dir, const KVCacheManager& kv) {
  return std::make_unique<CudaModel>(dir, kv);
}

}  // namespace engine
