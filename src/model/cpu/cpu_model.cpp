// CPU reference forward pass for Llama-architecture models (Qwen2, Llama,
// TinyLlama). Correctness first: fp32 throughout, naive attention that reads
// K/V through the paged block table exactly like the CUDA kernel will.
// GEMMs go through cblas when available (Accelerate on macOS) so the Mac can
// actually serve tokens at a usable rate.
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "engine/model.h"
#include "engine/ops.h"
#include "engine/parallel.h"
#include "engine/safetensors.h"

#ifdef ENGINE_HAVE_CBLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace engine {
namespace {

// C[m,n] = A[m,k] * B[n,k]^T (+ bias[n])   — the nn.Linear convention.
void linear(const float* A, int m, int k, const float* B, int n, const float* bias, float* C) {
#ifdef ENGINE_HAVE_CBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      const float* a = A + static_cast<size_t>(i) * k; const float* b = B + static_cast<size_t>(j) * k;
      float s = 0.f; for (int t = 0; t < k; t++) s += a[t] * b[t];
      C[static_cast<size_t>(i) * n + j] = s;
    }
#endif
  if (bias) for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) C[static_cast<size_t>(i) * n + j] += bias[j];
}

using ref::rmsnorm;

struct Layer {
  Tensor in_norm, post_norm, wq, wk, wv, wo, bq, bk, bv, wgate, wup, wdown;
};

// Attention work below this (multiply-adds, roughly) runs on one thread: the
// pool's synchronisation and its contention with BLAS cost more than the work.
// Measured on a 10-core Mac; short-context batches regressed 15% when threaded
// unconditionally, while a 2,048-token context sped up 3.4x.
constexpr int64_t kAttentionThreadThreshold = 1 << 20;

class CpuModel final : public Model {
 public:
  CpuModel(const std::string& dir, const KVCacheManager& kv, CpuModelOptions opts)
      : cfg_(ModelConfig::load(dir)), kv_(kv), opts_(std::move(opts)) {
    SafeTensorsDir w(dir);
    embed_ = w.get("model.embed_tokens.weight").to_f32();
    final_norm_ = w.get("model.norm.weight").to_f32();
    if (!cfg_.tie_word_embeddings) lm_head_ = w.get("lm_head.weight").to_f32();
    layers_.resize(cfg_.num_hidden_layers);
    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      std::string p = "model.layers." + std::to_string(l) + ".";
      Layer& L = layers_[l];
      L.in_norm = w.get(p + "input_layernorm.weight").to_f32();
      L.post_norm = w.get(p + "post_attention_layernorm.weight").to_f32();
      L.wq = w.get(p + "self_attn.q_proj.weight").to_f32();
      L.wk = w.get(p + "self_attn.k_proj.weight").to_f32();
      L.wv = w.get(p + "self_attn.v_proj.weight").to_f32();
      L.wo = w.get(p + "self_attn.o_proj.weight").to_f32();
      if (w.has(p + "self_attn.q_proj.bias")) {
        L.bq = w.get(p + "self_attn.q_proj.bias").to_f32();
        L.bk = w.get(p + "self_attn.k_proj.bias").to_f32();
        L.bv = w.get(p + "self_attn.v_proj.bias").to_f32();
      }
      L.wgate = w.get(p + "mlp.gate_proj.weight").to_f32();
      L.wup = w.get(p + "mlp.up_proj.weight").to_f32();
      L.wdown = w.get(p + "mlp.down_proj.weight").to_f32();
    }
    // RoPE tables for every position we can ever see.
    const int hd = cfg_.head_dim;
    rope_.build(hd, cfg_.max_position_embeddings, cfg_.rope_theta);
    // Paged KV cache: [layer][slot][kv_head][head_dim]
    const size_t slots = static_cast<size_t>(kv.num_total_blocks()) * kv.block_size();
    kv_dim_ = cfg_.num_key_value_heads * hd;
    k_cache_.assign(cfg_.num_hidden_layers, std::vector<float>(slots * kv_dim_, 0.f));
    v_cache_.assign(cfg_.num_hidden_layers, std::vector<float>(slots * kv_dim_, 0.f));
  }

  const ModelConfig& config() const override { return cfg_; }
  const char* backend() const override { return "cpu"; }

  // Every slice in the step is packed into one [N, hidden] batch, so each layer
  // runs one GEMM per projection instead of one per sequence. Attention cannot be
  // batched the same way, because each token attends over its own sequence's
  // block table for its own length, so it stays a loop over (token, head). This
  // is the same packing the CUDA backend needs.
  void forward(const StepInput& step, std::vector<float>& logits) override {
    logits.assign(static_cast<size_t>(step.num_logits) * cfg_.vocab_size, 0.f);
    if (step.slices.empty()) return;

    const int H = cfg_.hidden_size, hd = cfg_.head_dim, nh = cfg_.num_attention_heads,
              nkv = cfg_.num_key_value_heads, I = cfg_.intermediate_size,
              group = cfg_.gqa_group(), bs = kv_.block_size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // Flatten the step: one entry per token, in slice order.
    tok_seq_.clear(); tok_pos_.clear(); tok_id_.clear(); logit_rows_.clear();
    for (const auto& sl : step.slices) {
      for (int t = 0; t < sl.len; t++) {
        tok_seq_.push_back(sl.seq);
        tok_pos_.push_back(sl.start + t);
        tok_id_.push_back(sl.seq->tokens[sl.start + t]);
      }
      // Only the last token of a slice that wants logits produces a row.
      if (sl.needs_logits) logit_rows_.push_back(static_cast<int>(tok_id_.size()) - 1);
    }
    const int N = static_cast<int>(tok_id_.size());

    x_.assign(static_cast<size_t>(N) * H, 0.f);
    for (int i = 0; i < N; i++)
      std::memcpy(x_.data() + static_cast<size_t>(i) * H,
                  embed_.ptr() + static_cast<size_t>(tok_id_[i]) * H, sizeof(float) * H);
    if (opts_.on_layer) opts_.on_layer(-1, x_.data(), N);
    tap({Tap::Embedding, -1, N, H, x_.data()});

    h_.resize(static_cast<size_t>(N) * H); q_.resize(static_cast<size_t>(N) * H);
    k_.resize(static_cast<size_t>(N) * kv_dim_); v_.resize(static_cast<size_t>(N) * kv_dim_);
    attn_.resize(static_cast<size_t>(N) * H); o_.resize(static_cast<size_t>(N) * H);
    gate_.resize(static_cast<size_t>(N) * I); up_.resize(static_cast<size_t>(N) * I);

    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      const Layer& L = layers_[l];
      float* kc = k_cache_[l].data();
      float* vc = v_cache_[l].data();

      rmsnorm(x_.data(), L.in_norm.ptr(), h_.data(), N, H, cfg_.rms_norm_eps);
      tap({Tap::InputNorm, l, N, H, h_.data()});
      linear(h_.data(), N, H, L.wq.ptr(), H, L.bq.numel() ? L.bq.ptr() : nullptr, q_.data());
      linear(h_.data(), N, H, L.wk.ptr(), kv_dim_, L.bk.numel() ? L.bk.ptr() : nullptr, k_.data());
      linear(h_.data(), N, H, L.wv.ptr(), kv_dim_, L.bv.numel() ? L.bv.ptr() : nullptr, v_.data());
      tap({Tap::QkvProj, l, N, H, q_.data(), k_.data(), v_.data(), kv_dim_, kv_dim_});

      // Rotate and write K/V for every token first, so tokens that arrived in
      // this same step can attend to each other. The rope tap fires after the
      // rotation and before the cache write, because the CUDA version fuses
      // those two and still has to reproduce this intermediate.
      for (int i = 0; i < N; i++) {
        const int pos = tok_pos_[i];
        apply_rope(q_.data() + static_cast<size_t>(i) * H, nh, pos);
        apply_rope(k_.data() + static_cast<size_t>(i) * kv_dim_, nkv, pos);
        if (i + 1 == N) tap({Tap::Rope, l, N, H, q_.data(), k_.data(), nullptr, kv_dim_, 0});
        const int64_t slot = tok_seq_[i]->block_table.slot(pos, bs);
        std::memcpy(kc + static_cast<size_t>(slot) * kv_dim_,
                    k_.data() + static_cast<size_t>(i) * kv_dim_, sizeof(float) * kv_dim_);
        std::memcpy(vc + static_cast<size_t>(slot) * kv_dim_,
                    v_.data() + static_cast<size_t>(i) * kv_dim_, sizeof(float) * kv_dim_);
      }

      // One work item per (token, head), the same decomposition the CUDA decode
      // kernel uses: a thread block per (sequence, head) walking the block table.
      // Each item writes only its own head slice, so this is race-free and the
      // result does not depend on the split.
      //
      // Threading only pays when attention is actually the bottleneck. Its cost
      // grows with context length, while the pool costs a wake-up per layer per
      // step and competes with the BLAS threads running the GEMMs. Below the
      // threshold the serial loop is measurably faster, so estimate the work
      // first: roughly one multiply-add per (token, head, past position, dim).
      int64_t attn_work = 0;
      for (int i = 0; i < N; i++) attn_work += static_cast<int64_t>(tok_pos_[i]) + 1;
      attn_work *= static_cast<int64_t>(nh) * hd;
      auto attention_body = [&](int64_t begin, int64_t end) {
        std::vector<float> scores;
        for (int64_t w = begin; w < end; w++) {
          const int i = static_cast<int>(w / nh), h = static_cast<int>(w % nh);
          ref::paged_attention_head(
              q_.data() + static_cast<size_t>(i) * H + h * hd, kc, vc,
              tok_seq_[i]->block_table, bs, tok_pos_[i], h / group, nkv, hd, scale,
              attn_.data() + static_cast<size_t>(i) * H + h * hd, scores);
        }
      };
      const int64_t total_items = static_cast<int64_t>(N) * nh;
      if (attn_work >= kAttentionThreadThreshold) pool_.parallel_for(total_items, attention_body);
      else attention_body(0, total_items);

      tap({Tap::AttnOut, l, N, H, attn_.data()});
      linear(attn_.data(), N, H, L.wo.ptr(), H, nullptr, o_.data());
      for (size_t i = 0; i < x_.size(); i++) x_[i] += o_[i];
      tap({Tap::AttnResidual, l, N, H, x_.data()});

      rmsnorm(x_.data(), L.post_norm.ptr(), h_.data(), N, H, cfg_.rms_norm_eps);
      tap({Tap::PostNorm, l, N, H, h_.data()});
      linear(h_.data(), N, H, L.wgate.ptr(), I, nullptr, gate_.data());
      linear(h_.data(), N, H, L.wup.ptr(), I, nullptr, up_.data());
      ref::silu_mul(gate_.data(), up_.data(), gate_.size());
      tap({Tap::SiluMul, l, N, I, gate_.data()});
      linear(gate_.data(), N, I, L.wdown.ptr(), H, nullptr, o_.data());
      for (size_t i = 0; i < x_.size(); i++) x_[i] += o_[i];
      tap({Tap::LayerOut, l, N, H, x_.data()});
      if (opts_.on_layer) opts_.on_layer(l, x_.data(), N);
    }

    if (logit_rows_.empty()) return;
    // Gather only the rows that need logits, then one GEMM against the head.
    const int R = static_cast<int>(logit_rows_.size());
    h_.resize(static_cast<size_t>(R) * H);
    for (int r = 0; r < R; r++)
      rmsnorm(x_.data() + static_cast<size_t>(logit_rows_[r]) * H, final_norm_.ptr(),
              h_.data() + static_cast<size_t>(r) * H, 1, H, cfg_.rms_norm_eps);
    const Tensor& head = cfg_.tie_word_embeddings ? embed_ : lm_head_;
    linear(h_.data(), R, H, head.ptr(), cfg_.vocab_size, nullptr, logits.data());
    tap({Tap::Logits, -1, R, cfg_.vocab_size, logits.data()});
  }

  void tap(const TapData& d) const { if (opts_.on_tap) opts_.on_tap(d); }

 private:
  void apply_rope(float* x, int heads, int pos) const {
    ref::rope_inplace(x, heads, cfg_.head_dim, pos, rope_);
  }

  ModelConfig cfg_;
  const KVCacheManager& kv_;
  CpuModelOptions opts_;
  Tensor embed_, final_norm_, lm_head_;
  std::vector<Layer> layers_;
  ref::RopeTables rope_;
  int kv_dim_ = 0;
  std::vector<std::vector<float>> k_cache_, v_cache_;
  // scratch, reused across steps
  ThreadPool pool_{default_thread_count()};
  std::vector<float> x_, h_, q_, k_, v_, attn_, o_, gate_, up_;
  std::vector<const Sequence*> tok_seq_;   // per packed token: owning sequence
  std::vector<int> tok_pos_;               // per packed token: position in that sequence
  std::vector<int32_t> tok_id_;            // per packed token: token id
  std::vector<int> logit_rows_;            // packed indices whose logits are wanted
};

}  // namespace

std::unique_ptr<Model> make_cpu_model(const std::string& dir, const KVCacheManager& kv, CpuModelOptions opts) {
  return std::make_unique<CpuModel>(dir, kv, std::move(opts));
}

}  // namespace engine
