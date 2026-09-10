// CPU reference forward pass for Llama-architecture models (Qwen2, Llama,
// TinyLlama). Correctness first: fp32 throughout, naive attention that reads
// K/V through the paged block table exactly like the CUDA kernel will.
// GEMMs go through cblas when available (Accelerate on macOS) so the Mac can
// actually serve tokens at a usable rate.
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "engine/model.h"
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

void rmsnorm(const float* x, const float* w, float* out, int rows, int h, float eps) {
  for (int r = 0; r < rows; r++) {
    const float* xr = x + static_cast<size_t>(r) * h; float* o = out + static_cast<size_t>(r) * h;
    double ss = 0; for (int i = 0; i < h; i++) ss += static_cast<double>(xr[i]) * xr[i];
    float inv = 1.0f / std::sqrt(static_cast<float>(ss / h) + eps);
    for (int i = 0; i < h; i++) o[i] = xr[i] * inv * w[i];
  }
}

struct Layer {
  Tensor in_norm, post_norm, wq, wk, wv, wo, bq, bk, bv, wgate, wup, wdown;
};

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
    const int hd = cfg_.head_dim, half = hd / 2, P = cfg_.max_position_embeddings;
    cos_.assign(static_cast<size_t>(P) * half, 0.f); sin_.assign(static_cast<size_t>(P) * half, 0.f);
    for (int i = 0; i < half; i++) {
      double inv_freq = 1.0 / std::pow(static_cast<double>(cfg_.rope_theta), (2.0 * i) / hd);
      for (int p = 0; p < P; p++) {
        double a = p * inv_freq;
        cos_[static_cast<size_t>(p) * half + i] = static_cast<float>(std::cos(a));
        sin_[static_cast<size_t>(p) * half + i] = static_cast<float>(std::sin(a));
      }
    }
    // Paged KV cache: [layer][slot][kv_head][head_dim]
    const size_t slots = static_cast<size_t>(kv.num_total_blocks()) * kv.block_size();
    kv_dim_ = cfg_.num_key_value_heads * hd;
    k_cache_.assign(cfg_.num_hidden_layers, std::vector<float>(slots * kv_dim_, 0.f));
    v_cache_.assign(cfg_.num_hidden_layers, std::vector<float>(slots * kv_dim_, 0.f));
  }

  const ModelConfig& config() const override { return cfg_; }
  const char* backend() const override { return "cpu"; }

  void forward(const StepInput& step, std::vector<float>& logits) override {
    logits.resize(static_cast<size_t>(step.num_logits) * cfg_.vocab_size);
    int row = 0;
    for (const auto& sl : step.slices) {
      forward_slice(sl, sl.needs_logits ? logits.data() + static_cast<size_t>(row) * cfg_.vocab_size : nullptr);
      if (sl.needs_logits) row++;
    }
  }

 private:
  // RoPE in the HF "rotate_half" convention: pairs (i, i + half).
  void apply_rope(float* x, int heads, int pos) const {
    const int hd = cfg_.head_dim, half = hd / 2;
    const float* c = cos_.data() + static_cast<size_t>(pos) * half; const float* s = sin_.data() + static_cast<size_t>(pos) * half;
    for (int h = 0; h < heads; h++) {
      float* v = x + h * hd;
      for (int i = 0; i < half; i++) {
        float a = v[i], b = v[i + half];
        v[i] = a * c[i] - b * s[i];
        v[i + half] = b * c[i] + a * s[i];
      }
    }
  }

  void forward_slice(const StepSlice& sl, float* logits_out) {
    const Sequence& seq = *sl.seq;
    const int T = sl.len, H = cfg_.hidden_size, hd = cfg_.head_dim, nh = cfg_.num_attention_heads,
              nkv = cfg_.num_key_value_heads, I = cfg_.intermediate_size, group = cfg_.gqa_group(), bs = kv_.block_size();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    x_.assign(static_cast<size_t>(T) * H, 0.f);
    for (int t = 0; t < T; t++)
      std::memcpy(x_.data() + static_cast<size_t>(t) * H, embed_.ptr() + static_cast<size_t>(seq.tokens[sl.start + t]) * H, sizeof(float) * H);
    if (opts_.on_layer) opts_.on_layer(-1, x_.data(), T);

    h_.resize(static_cast<size_t>(T) * H); q_.resize(static_cast<size_t>(T) * H); k_.resize(static_cast<size_t>(T) * kv_dim_);
    v_.resize(static_cast<size_t>(T) * kv_dim_); attn_.resize(static_cast<size_t>(T) * H); o_.resize(static_cast<size_t>(T) * H);
    gate_.resize(static_cast<size_t>(T) * I); up_.resize(static_cast<size_t>(T) * I);
    scores_.resize(static_cast<size_t>(sl.start + T));

    for (int l = 0; l < cfg_.num_hidden_layers; l++) {
      const Layer& L = layers_[l];
      float* kc = k_cache_[l].data(); float* vc = v_cache_[l].data();

      // --- attention block ---
      rmsnorm(x_.data(), L.in_norm.ptr(), h_.data(), T, H, cfg_.rms_norm_eps);
      linear(h_.data(), T, H, L.wq.ptr(), H, L.bq.numel() ? L.bq.ptr() : nullptr, q_.data());
      linear(h_.data(), T, H, L.wk.ptr(), kv_dim_, L.bk.numel() ? L.bk.ptr() : nullptr, k_.data());
      linear(h_.data(), T, H, L.wv.ptr(), kv_dim_, L.bv.numel() ? L.bv.ptr() : nullptr, v_.data());
      for (int t = 0; t < T; t++) {
        const int pos = sl.start + t;
        apply_rope(q_.data() + static_cast<size_t>(t) * H, nh, pos);
        apply_rope(k_.data() + static_cast<size_t>(t) * kv_dim_, nkv, pos);
        const int64_t slot = seq.block_table.slot(pos, bs);
        std::memcpy(kc + static_cast<size_t>(slot) * kv_dim_, k_.data() + static_cast<size_t>(t) * kv_dim_, sizeof(float) * kv_dim_);
        std::memcpy(vc + static_cast<size_t>(slot) * kv_dim_, v_.data() + static_cast<size_t>(t) * kv_dim_, sizeof(float) * kv_dim_);
      }
      // Causal attention over the paged cache, one (token, head) at a time.
      for (int t = 0; t < T; t++) {
        const int pos = sl.start + t;
        for (int h = 0; h < nh; h++) {
          const int kvh = h / group;
          const float* q = q_.data() + static_cast<size_t>(t) * H + h * hd;
          float mx = -INFINITY;
          for (int p = 0; p <= pos; p++) {
            const float* k = kc + static_cast<size_t>(seq.block_table.slot(p, bs)) * kv_dim_ + kvh * hd;
            float s = 0.f; for (int d = 0; d < hd; d++) s += q[d] * k[d];
            s *= scale; scores_[p] = s; if (s > mx) mx = s;
          }
          double denom = 0; for (int p = 0; p <= pos; p++) { scores_[p] = std::exp(scores_[p] - mx); denom += scores_[p]; }
          float* out = attn_.data() + static_cast<size_t>(t) * H + h * hd;
          for (int d = 0; d < hd; d++) out[d] = 0.f;
          for (int p = 0; p <= pos; p++) {
            const float w = static_cast<float>(scores_[p] / denom);
            const float* v = vc + static_cast<size_t>(seq.block_table.slot(p, bs)) * kv_dim_ + kvh * hd;
            for (int d = 0; d < hd; d++) out[d] += w * v[d];
          }
        }
      }
      linear(attn_.data(), T, H, L.wo.ptr(), H, nullptr, o_.data());
      for (size_t i = 0; i < x_.size(); i++) x_[i] += o_[i];

      // --- MLP block: down(silu(gate(x)) * up(x)) ---
      rmsnorm(x_.data(), L.post_norm.ptr(), h_.data(), T, H, cfg_.rms_norm_eps);
      linear(h_.data(), T, H, L.wgate.ptr(), I, nullptr, gate_.data());
      linear(h_.data(), T, H, L.wup.ptr(), I, nullptr, up_.data());
      for (size_t i = 0; i < gate_.size(); i++) { float g = gate_[i]; gate_[i] = g / (1.0f + std::exp(-g)) * up_[i]; }
      linear(gate_.data(), T, I, L.wdown.ptr(), H, nullptr, o_.data());
      for (size_t i = 0; i < x_.size(); i++) x_[i] += o_[i];
      if (opts_.on_layer) opts_.on_layer(l, x_.data(), T);
    }

    if (!logits_out) return;
    // Logits only for the last token of the slice.
    rmsnorm(x_.data() + static_cast<size_t>(T - 1) * H, final_norm_.ptr(), h_.data(), 1, H, cfg_.rms_norm_eps);
    const Tensor& head = cfg_.tie_word_embeddings ? embed_ : lm_head_;
    linear(h_.data(), 1, H, head.ptr(), cfg_.vocab_size, nullptr, logits_out);
  }

  ModelConfig cfg_;
  const KVCacheManager& kv_;
  CpuModelOptions opts_;
  Tensor embed_, final_norm_, lm_head_;
  std::vector<Layer> layers_;
  std::vector<float> cos_, sin_;
  int kv_dim_ = 0;
  std::vector<std::vector<float>> k_cache_, v_cache_;
  // scratch
  std::vector<float> x_, h_, q_, k_, v_, attn_, o_, gate_, up_, scores_;
};

}  // namespace

std::unique_ptr<Model> make_cpu_model(const std::string& dir, const KVCacheManager& kv, CpuModelOptions opts) {
  return std::make_unique<CpuModel>(dir, kv, std::move(opts));
}

}  // namespace engine
