// CUDA backend. This file is the host side: it owns device memory, uploads the
// weights, and orchestrates the per-step work. The maths lives in src/kernels/.
//
// It is a port of CpuModel::forward, deliberately keeping the same packed step
// layout and the same KV cache layout, so slot(pos) is identical arithmetic on
// both sides and a disagreement can never be blamed on the two backends
// organising memory differently.
//
// Activations stay resident. The only transfers per step are a few small index
// arrays in and the logits out; everything between the embedding and the head
// is a kernel reading and writing device memory.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
#define CUDA_CHECK(x)                                                                \
  do {                                                                               \
    cudaError_t e_ = (x);                                                            \
    if (e_ != cudaSuccess)                                                           \
      throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                               " " + cudaGetErrorString(e_));                        \
  } while (0)

#define CUBLAS_CHECK(x)                                                              \
  do {                                                                               \
    cublasStatus_t s_ = (x);                                                         \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                                 \
      throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                               " cuBLAS status " + std::to_string(int(s_)));         \
  } while (0)

// Allocation is accounted into a caller-supplied counter rather than a global.
// A file-scope total only ever incremented, so with more than one model alive
// in a process -- the test suite builds five -- each reported a figure that
// included every model before it. The leak below looked like a large model.
template <typename T>
T* device_alloc(size_t count, size_t* acct = nullptr) {
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
  if (acct) *acct += bytes;
  return p;
}

// Weights ship as bf16 and the device wants fp16. Norms and biases stay fp32:
// they are tiny, and the reference uses them in fp32.
__half* upload_f16(const TensorView& t, size_t* acct = nullptr) {
  const std::vector<uint16_t> h = t.to_f16();
  __half* d = device_alloc<__half>(h.size(), acct);
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  return d;
}

float* upload_f32(const TensorView& t, size_t* acct = nullptr) {
  const Tensor h = t.to_f32();
  float* d = device_alloc<float>(h.numel(), acct);
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
  float* bq = nullptr;     // null on models without qkv bias, such as Llama
  float* bk = nullptr;
  float* bv = nullptr;
  __half* wgate = nullptr;
  __half* wup = nullptr;
  __half* wdown = nullptr;
};

// Where a decode step actually goes, measured with CUDA events.
//
// Nsight Compute is the obvious tool and is unavailable on a rented pod: it
// needs NVreg_RestrictProfilingToAdminUsers=0, a host kernel-module parameter
// a container cannot set, and returns ERR_NVGPUCTRPERM without it. CUDA events
// need no privileges at all. They cannot report achieved bandwidth, which is
// the thing ncu would have given, but they answer the question that actually
// decides what to optimise next: which region of a step the time is in.
//
// Opt in with ENGINE_PROFILE=1. Off, every method is an inlined branch on a
// bool. On, it records two events per region per layer, which is cheap but not
// free, so treat the absolute step time under profiling as slightly inflated
// and the proportions as the result.
//
// Events are recorded on the default stream, which is where every launch here
// goes, and read only after the step's cudaDeviceSynchronize. Reading one
// earlier would block on work still queued.
enum ProfRegion { kEmbed, kNorm, kQKV, kRope, kAttn, kProjO, kMLP, kHead, kRegions };
const char* const kRegionName[kRegions] = {
    "embedding", "rmsnorm", "qkv_proj", "rope+store_kv",
    "attention", "o_proj+residual", "mlp", "head"};

class StepProfile {
 public:
  StepProfile() {
    const char* e = std::getenv("ENGINE_PROFILE");
    on_ = e && *e && *e != '0';
  }
  ~StepProfile() {
    if (on_) report();
    for (cudaEvent_t e : pool_) cudaEventDestroy(e);
  }
  void begin(int region) { if (on_) { open_ = region; start_ = mark(); } }
  void end() { if (on_) spans_.push_back({open_, start_, mark()}); }

  // Must be called after the step has synchronised.
  void flush() {
    if (!on_) return;
    for (const Span& sp : spans_) {
      // mark() returns an unincremented index if event creation ever failed,
      // so a span can name an event that does not exist. Dropping it loses one
      // sample; indexing it would be a buffer overrun in a diagnostic.
      if (sp.a >= static_cast<int>(pool_.size()) || sp.b >= static_cast<int>(pool_.size()))
        continue;
      float ms = 0.f;
      if (cudaEventElapsedTime(&ms, pool_[sp.a], pool_[sp.b]) == cudaSuccess)
        total_ms_[sp.region] += ms;
    }
    spans_.clear();
    next_ = 0;
    steps_++;
  }

 private:
  struct Span { int region, a, b; };

  int mark() {
    if (next_ == static_cast<int>(pool_.size())) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreate(&e) != cudaSuccess) return next_;  // give up quietly
      pool_.push_back(e);
    }
    cudaEventRecord(pool_[next_], nullptr);
    return next_++;
  }

  void report() const {
    if (steps_ == 0) return;
    double sum = 0;
    for (int r = 0; r < kRegions; r++) sum += total_ms_[r];
    if (sum <= 0) return;
    std::printf("\nper-step cost over %d steps (CUDA events, default stream)\n", steps_);
    std::printf("%-18s %10s %8s\n", "region", "ms/step", "share");
    for (int r = 0; r < kRegions; r++)
      std::printf("%-18s %10.3f %7.1f%%\n", kRegionName[r], total_ms_[r] / steps_,
                  100.0 * total_ms_[r] / sum);
    std::printf("%-18s %10.3f\n", "total on device", sum / steps_);

    const char* out = std::getenv("ENGINE_PROFILE_OUT");
    FILE* f = std::fopen(out && *out ? out : "bench/results/kernel-time.json", "w");
    if (!f) return;
    std::fprintf(f, "{\n  \"steps\": %d,\n  \"regions\": [\n", steps_);
    for (int r = 0; r < kRegions; r++)
      std::fprintf(f, "    {\"region\": \"%s\", \"ms_per_step\": %.6f, \"share_pct\": %.3f}%s\n",
                   kRegionName[r], total_ms_[r] / steps_, 100.0 * total_ms_[r] / sum,
                   r + 1 == kRegions ? "" : ",");
    std::fprintf(f, "  ],\n  \"device_ms_per_step\": %.6f\n}\n", sum / steps_);
    std::fclose(f);
  }

  bool on_ = false;
  int open_ = 0, start_ = 0, next_ = 0, steps_ = 0;
  std::vector<cudaEvent_t> pool_;
  std::vector<Span> spans_;
  double total_ms_[kRegions] = {0};
};

// Times one region for as long as it is in scope.
struct ProfScope {
  StepProfile& p;
  ProfScope(StepProfile& p_, int region) : p(p_) { p.begin(region); }
  ~ProfScope() { p.end(); }
};

// A device buffer that grows on demand and is then reused. Allocating inside
// forward() every step is the classic way to make a GPU program mysteriously
// slow, so every scratch tensor goes through one of these.
template <typename T>
struct Scratch {
  T* ptr = nullptr;
  size_t cap = 0;

  Scratch() = default;
  // Owns a device pointer, so copying one would free it twice.
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    if (ptr) cudaFree(ptr);   // a destructor cannot throw; nothing to recover
  }

  T* get(size_t need) {
    if (cap < need) {
      if (ptr) CUDA_CHECK(cudaFree(ptr));
      ptr = device_alloc<T>(need);
      cap = need;
    }
    return ptr;
  }
};

class CudaModel final : public Model {
 public:
  CudaModel(const std::string& dir, const KVCacheManager& kv)
      : cfg_(ModelConfig::load(dir)), kv_(kv) {
    CUDA_CHECK(cudaSetDevice(0));

    // Four places in this file hardcode the Q projection's output width as
    // hidden_size: the Q GEMM, q_'s size, the row stride handed to RoPE, and
    // the row stride attention writes with. That holds only while
    // num_attention_heads * head_dim == hidden_size, which config.json is free
    // to violate -- Qwen3-0.6B has hidden_size 1024 with 16 heads of 128, so
    // 2048. There the Q GEMM would write half the width it should, RoPE would
    // stride wrong, and attention would write N*2048 halves into an N*1024
    // buffer, running off the end into whatever Scratch allocated next. That
    // is silent corruption of k or attn, not a crash, so it is checked here
    // rather than discovered as bad output.
    if (cfg_.num_attention_heads * cfg_.head_dim != cfg_.hidden_size) {
      throw std::runtime_error(
          "cuda backend: num_attention_heads * head_dim (" +
          std::to_string(cfg_.num_attention_heads) + " * " +
          std::to_string(cfg_.head_dim) + " = " +
          std::to_string(cfg_.num_attention_heads * cfg_.head_dim) +
          ") must equal hidden_size (" + std::to_string(cfg_.hidden_size) +
          ") for this model. The CPU backend handles the general case; see "
          "docs/gpu-setup.md.");
    }

    CUBLAS_CHECK(cublasCreate(&blas_));
    load_weights(dir);

    // Rotation tables, uploaded once. Recomputing powf per element in the
    // kernel would be slower and would drift from the reference.
    ref::RopeTables rope;
    rope.build(cfg_.head_dim, cfg_.max_position_embeddings, cfg_.rope_theta);
    d_cos_ = device_alloc<float>(rope.cos.size(), &weights_bytes_);
    d_sin_ = device_alloc<float>(rope.sin.size(), &weights_bytes_);
    CUDA_CHECK(cudaMemcpy(d_cos_, rope.cos.data(), rope.cos.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin_, rope.sin.data(), rope.sin.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    kv_dim_ = cfg_.num_key_value_heads * cfg_.head_dim;
    const size_t slots = static_cast<size_t>(kv.num_total_blocks()) * kv.block_size();
    k_cache_.resize(cfg_.num_hidden_layers);
    v_cache_.resize(cfg_.num_hidden_layers);
    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      k_cache_[l] = device_alloc<__half>(slots * kv_dim_, &weights_bytes_);
      v_cache_[l] = device_alloc<__half>(slots * kv_dim_, &weights_bytes_);
      CUDA_CHECK(cudaMemset(k_cache_[l], 0, slots * kv_dim_ * sizeof(__half)));
      CUDA_CHECK(cudaMemset(v_cache_[l], 0, slots * kv_dim_ * sizeof(__half)));
    }
    report_memory();
  }

  // Releases every device allocation, not just the cuBLAS handle.
  //
  // This used to free blas_ alone. One model is about a gigabyte of fp16
  // weights -- the 151936x896 embedding is 272 MiB by itself -- plus the KV
  // cache, and the test suite constructs five in a single process. The dead
  // memory accumulated until a cudaMalloc failed and reported the pool as too
  // large for the card, which is actively misleading: the pool was fine, the
  // previous models were still resident. On a server that reloads a model it
  // is an unbounded leak.
  ~CudaModel() override {
    for (__half* p : k_cache_) if (p) cudaFree(p);
    for (__half* p : v_cache_) if (p) cudaFree(p);
    for (CudaLayer& L : layers_) {
      for (void* p : {static_cast<void*>(L.in_norm), static_cast<void*>(L.post_norm),
                      static_cast<void*>(L.wq), static_cast<void*>(L.wk),
                      static_cast<void*>(L.wv), static_cast<void*>(L.wo),
                      static_cast<void*>(L.bq), static_cast<void*>(L.bk),
                      static_cast<void*>(L.bv), static_cast<void*>(L.wgate),
                      static_cast<void*>(L.wup), static_cast<void*>(L.wdown)})
        if (p) cudaFree(p);
    }
    if (d_cos_) cudaFree(d_cos_);
    if (d_sin_) cudaFree(d_sin_);
    if (final_norm_) cudaFree(final_norm_);
    // A tied head is the embedding read a second time, not a copy, so freeing
    // both would be a double free.
    if (lm_head_ && lm_head_ != embed_) cudaFree(lm_head_);
    if (embed_) cudaFree(embed_);
    // The Scratch members free themselves.
    if (blas_) cublasDestroy(blas_);
  }

  const ModelConfig& config() const override { return cfg_; }
  const char* backend() const override { return "cuda"; }

  void forward(const StepInput& step, std::vector<float>& logits) override {
    logits.assign(static_cast<size_t>(step.num_logits) * cfg_.vocab_size, 0.f);
    if (step.slices.empty()) return;

    const int H = cfg_.hidden_size, hd = cfg_.head_dim, nh = cfg_.num_attention_heads,
              nkv = cfg_.num_key_value_heads, I = cfg_.intermediate_size, bs = kv_.block_size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    flatten(step);
    const int N = static_cast<int>(tok_id_.size());
    upload_step_indices(N, bs);

    __half* x = x_.get(static_cast<size_t>(N) * H);
    __half* h = h_.get(static_cast<size_t>(N) * H);
    __half* q = q_.get(static_cast<size_t>(N) * H);
    __half* k = k_.get(static_cast<size_t>(N) * kv_dim_);
    __half* v = v_.get(static_cast<size_t>(N) * kv_dim_);
    __half* attn = attn_.get(static_cast<size_t>(N) * H);
    __half* o = o_.get(static_cast<size_t>(N) * H);
    __half* gate = gate_.get(static_cast<size_t>(N) * I);
    __half* up = up_.get(static_cast<size_t>(N) * I);

    { ProfScope _(prof_, kEmbed);
      launch_embedding(embed_, d_token_ids_, x, N, H, nullptr); }

    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      const CudaLayer& L = layers_[l];

      { ProfScope _(prof_, kNorm);
        launch_rmsnorm(x, L.in_norm, h, N, H, cfg_.rms_norm_eps, nullptr); }
      { ProfScope _(prof_, kQKV);
      gemm(h, L.wq, q, N, H, H);
      gemm(h, L.wk, k, N, kv_dim_, H);
      gemm(h, L.wv, v, N, kv_dim_, H);
      launch_add_bias(q, L.bq, N, H, nullptr);
      launch_add_bias(k, L.bk, N, kv_dim_, nullptr);
      launch_add_bias(v, L.bv, N, kv_dim_, nullptr); }

      // K and V for every token in this step are written before any attention
      // runs, which is what lets tokens that arrived together attend to each
      // other and makes a mixed prefill-and-decode step correct.
      { ProfScope _(prof_, kRope);
      launch_rope(q, d_positions_, d_cos_, d_sin_, N, nh, hd, H, nullptr);
      launch_rope(k, d_positions_, d_cos_, d_sin_, N, nkv, hd, kv_dim_, nullptr);
      launch_store_kv(k, v, d_slots_, k_cache_[l], v_cache_[l], N, kv_dim_, nullptr); }
      { ProfScope _(prof_, kAttn);
      launch_attention_decode(q, k_cache_[l], v_cache_[l], d_tables_, d_table_offset_,
                              d_token_seq_, d_positions_, attn, N, nh, nkv, hd, bs, scale,
                              nullptr); }

      { ProfScope _(prof_, kProjO);
      gemm(attn, L.wo, o, N, H, H);
      launch_add_inplace(x, o, static_cast<size_t>(N) * H, nullptr); }

      { ProfScope _(prof_, kNorm);
        launch_rmsnorm(x, L.post_norm, h, N, H, cfg_.rms_norm_eps, nullptr); }
      { ProfScope _(prof_, kMLP);
      gemm(h, L.wgate, gate, N, I, H);
      gemm(h, L.wup, up, N, I, H);
      launch_silu_mul(gate, up, static_cast<size_t>(N) * I, nullptr);
      gemm(gate, L.wdown, o, N, H, I);
      launch_add_inplace(x, o, static_cast<size_t>(N) * H, nullptr); }
    }

    if (logit_rows_.empty()) {
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      prof_.flush();
      return;
    }

    // Only the rows that need logits go through the head. They are gathered
    // device to device, so nothing crosses the bus until the logits themselves.
    const int R = static_cast<int>(logit_rows_.size());
    for (int r = 0; r < R; r++)
      CUDA_CHECK(cudaMemcpy(o + static_cast<size_t>(r) * H,
                            x + static_cast<size_t>(logit_rows_[r]) * H,
                            static_cast<size_t>(H) * sizeof(__half), cudaMemcpyDeviceToDevice));
    __half* dlog = logits_.get(static_cast<size_t>(R) * cfg_.vocab_size);
    { ProfScope _(prof_, kHead);
      launch_rmsnorm(o, final_norm_, h, R, H, cfg_.rms_norm_eps, nullptr);
      gemm(h, lm_head_, dlog, R, cfg_.vocab_size, H); }

    // One synchronise per step, here, rather than after every launch. Any error
    // from any kernel above surfaces at this point.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    prof_.flush();

    const size_t count = static_cast<size_t>(R) * cfg_.vocab_size;
    stage_.resize(count);
    CUDA_CHECK(cudaMemcpy(stage_.data(), dlog, count * sizeof(__half), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) logits[i] = f16_to_f32(stage_[i]);
  }

 private:
  // Per-token view of the step, in slice order: the same packing the CPU model
  // uses, and the layout the kernels consume.
  void flatten(const StepInput& step) {
    tok_pos_.clear();
    tok_id_.clear();
    tok_seq_.clear();
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

  // Everything the kernels need about where this step's tokens live: the slot
  // each one writes to, which sequence it belongs to, and the block table of
  // each distinct sequence. Block tables are concatenated once per step rather
  // than per token, because many tokens share a sequence.
  void upload_step_indices(int N, int bs) {
    h_slots_.resize(N);
    h_token_seq_.resize(N);
    h_tables_.clear();
    h_table_offset_.clear();

    std::vector<const Sequence*> seen;
    for (int i = 0; i < N; i++) {
      const Sequence* s = tok_seq_[i];
      int idx = -1;
      for (size_t j = 0; j < seen.size(); j++)
        if (seen[j] == s) { idx = static_cast<int>(j); break; }
      if (idx < 0) {
        idx = static_cast<int>(seen.size());
        seen.push_back(s);
        h_table_offset_.push_back(static_cast<int>(h_tables_.size()));
        for (BlockId b : s->block_table.blocks) h_tables_.push_back(b);
      }
      h_token_seq_[i] = idx;
      h_slots_[i] = static_cast<int>(s->block_table.slot(tok_pos_[i], bs));
    }

    auto send = [](Scratch<int>& dst, const std::vector<int>& src) {
      int* d = dst.get(src.empty() ? 1 : src.size());
      if (!src.empty())
        CUDA_CHECK(cudaMemcpy(d, src.data(), src.size() * sizeof(int), cudaMemcpyHostToDevice));
      return d;
    };
    d_slots_ = send(slots_buf_, h_slots_);
    d_token_seq_ = send(token_seq_buf_, h_token_seq_);
    d_tables_ = send(tables_buf_, h_tables_);
    d_table_offset_ = send(table_offset_buf_, h_table_offset_);
    d_positions_ = send(positions_buf_, tok_pos_);

    int32_t* ids = token_ids_buf_.get(static_cast<size_t>(N));
    CUDA_CHECK(cudaMemcpy(ids, tok_id_.data(), static_cast<size_t>(N) * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    d_token_ids_ = ids;
  }

  void load_weights(const std::string& dir) {
    SafeTensorsDir w(dir);
    embed_ = upload_f16(w.get("model.embed_tokens.weight"), &weights_bytes_);
    final_norm_ = upload_f32(w.get("model.norm.weight"), &weights_bytes_);
    // A tied head is the embedding table read a second time, not a copy of it.
    lm_head_ = cfg_.tie_word_embeddings ? embed_ : upload_f16(w.get("lm_head.weight"), &weights_bytes_);

    layers_.resize(cfg_.num_hidden_layers);
    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      const std::string p = "model.layers." + std::to_string(l) + ".";
      CudaLayer& L = layers_[l];
      L.in_norm = upload_f32(w.get(p + "input_layernorm.weight"), &weights_bytes_);
      L.post_norm = upload_f32(w.get(p + "post_attention_layernorm.weight"), &weights_bytes_);
      L.wq = upload_f16(w.get(p + "self_attn.q_proj.weight"), &weights_bytes_);
      L.wk = upload_f16(w.get(p + "self_attn.k_proj.weight"), &weights_bytes_);
      L.wv = upload_f16(w.get(p + "self_attn.v_proj.weight"), &weights_bytes_);
      L.wo = upload_f16(w.get(p + "self_attn.o_proj.weight"), &weights_bytes_);
      if (w.has(p + "self_attn.q_proj.bias")) {
        L.bq = upload_f32(w.get(p + "self_attn.q_proj.bias"), &weights_bytes_);
        L.bk = upload_f32(w.get(p + "self_attn.k_proj.bias"), &weights_bytes_);
        L.bv = upload_f32(w.get(p + "self_attn.v_proj.bias"), &weights_bytes_);
      }
      L.wgate = upload_f16(w.get(p + "mlp.gate_proj.weight"), &weights_bytes_);
      L.wup = upload_f16(w.get(p + "mlp.up_proj.weight"), &weights_bytes_);
      L.wdown = upload_f16(w.get(p + "mlp.down_proj.weight"), &weights_bytes_);
    }
  }

  void report_memory() const {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("cuda: %s, weights and cache %zu MiB, device %zu MiB free of %zu MiB\n",
                cfg_.model_type.c_str(), weights_bytes_ >> 20, free_b >> 20, total_b >> 20);
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
  // This model's weights, rotation tables and KV cache. Scratch buffers are
  // excluded deliberately: they are neither weights nor cache, which is what
  // report_memory() claims to be printing.
  size_t weights_bytes_ = 0;
  StepProfile prof_;

  __half* embed_ = nullptr;
  __half* lm_head_ = nullptr;
  float* final_norm_ = nullptr;
  float* d_cos_ = nullptr;
  float* d_sin_ = nullptr;
  std::vector<CudaLayer> layers_;

  // The KV cache, laid out exactly as the CPU backend has it:
  // [slot][kv_head][head_dim], with
  // slot = block_table[pos / block_size] * block_size + pos % block_size.
  int kv_dim_ = 0;
  std::vector<__half*> k_cache_, v_cache_;

  Scratch<__half> x_, h_, q_, k_, v_, attn_, o_, gate_, up_, logits_;
  Scratch<int> slots_buf_, token_seq_buf_, tables_buf_, table_offset_buf_, positions_buf_;
  Scratch<int32_t> token_ids_buf_;
  int *d_slots_ = nullptr, *d_token_seq_ = nullptr, *d_tables_ = nullptr;
  int *d_table_offset_ = nullptr, *d_positions_ = nullptr;
  int32_t* d_token_ids_ = nullptr;

  std::vector<const Sequence*> tok_seq_;
  std::vector<int> tok_pos_, logit_rows_;
  std::vector<int32_t> tok_id_;
  std::vector<int> h_slots_, h_token_seq_, h_tables_, h_table_offset_;
  std::vector<uint16_t> stage_;
};

}  // namespace

std::unique_ptr<Model> make_cuda_model(const std::string& dir, const KVCacheManager& kv) {
  return std::make_unique<CudaModel>(dir, kv);
}

}  // namespace engine
