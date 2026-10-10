#pragma once
// Reference implementations of the operations a decoder layer is made of.
//
// There is one copy of each, used by three callers: the CPU model runs them as
// its forward pass, the golden tests check them against dumped tensors, and the
// CUDA kernel tests diff device output against them. A kernel is therefore
// compared to the same code that was verified layer by layer against PyTorch,
// rather than to a second reference that could drift.
//
// Everything here is fp32, row-major, and deliberately simple. Speed comes from
// the kernels; this is the thing they have to agree with.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "engine/kv_cache.h"

namespace engine::ref {

// RMSNorm over rows: out[r] = x[r] / rms(x[r]) * weight, fp32 accumulate.
inline void rmsnorm(const float* x, const float* weight, float* out, int rows, int cols, float eps) {
  for (int r = 0; r < rows; r++) {
    const float* xr = x + static_cast<size_t>(r) * cols;
    float* o = out + static_cast<size_t>(r) * cols;
    double ss = 0;
    for (int i = 0; i < cols; i++) ss += static_cast<double>(xr[i]) * xr[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / cols) + eps);
    for (int i = 0; i < cols; i++) o[i] = xr[i] * inv * weight[i];
  }
}

// Rotary embedding tables for every position the model can reach.
struct RopeTables {
  std::vector<float> cos, sin;
  int half = 0;

  void build(int head_dim, int max_positions, float theta) {
    half = head_dim / 2;
    cos.assign(static_cast<size_t>(max_positions) * half, 0.f);
    sin.assign(static_cast<size_t>(max_positions) * half, 0.f);
    for (int i = 0; i < half; i++) {
      const double inv_freq = 1.0 / std::pow(static_cast<double>(theta), (2.0 * i) / head_dim);
      for (int p = 0; p < max_positions; p++) {
        const double angle = p * inv_freq;
        cos[static_cast<size_t>(p) * half + i] = static_cast<float>(std::cos(angle));
        sin[static_cast<size_t>(p) * half + i] = static_cast<float>(std::sin(angle));
      }
    }
  }
};

// RoPE in Hugging Face's rotate_half convention: element i pairs with
// i + head_dim/2, not with its neighbour. Applied in place to `heads`
// consecutive heads of one token.
inline void rope_inplace(float* x, int heads, int head_dim, int pos, const RopeTables& t) {
  const int half = head_dim / 2;
  const float* c = t.cos.data() + static_cast<size_t>(pos) * half;
  const float* s = t.sin.data() + static_cast<size_t>(pos) * half;
  for (int h = 0; h < heads; h++) {
    float* v = x + h * head_dim;
    for (int i = 0; i < half; i++) {
      const float a = v[i], b = v[i + half];
      v[i] = a * c[i] - b * s[i];
      v[i + half] = b * c[i] + a * s[i];
    }
  }
}

// Fused SwiGLU elementwise half: gate = silu(gate) * up, in place.
inline void silu_mul(float* gate, const float* up, size_t n) {
  for (size_t i = 0; i < n; i++) {
    const float g = gate[i];
    gate[i] = g / (1.0f + std::exp(-g)) * up[i];
  }
}

// Attention for one (token, head) over a paged KV cache.
//
// This is the operation attention_decode.cu implements: walk the sequence's
// block table from position 0 to `pos`, dot the query against each key, softmax
// with the max subtracted, and accumulate the weighted values. GQA is handled by
// mapping query head h to KV head h / group.
//
// `k_cache` and `v_cache` are laid out [slot][kv_heads][head_dim], and
// slot = block_table[p / block_size] * block_size + p % block_size.
inline void paged_attention_head(const float* q, const float* k_cache, const float* v_cache,
                                 const BlockTable& bt, int block_size, int pos, int kv_head,
                                 int kv_heads, int head_dim, float scale, float* out,
                                 std::vector<float>& scratch) {
  const int kv_dim = kv_heads * head_dim;
  scratch.resize(static_cast<size_t>(pos) + 1);

  float mx = -INFINITY;
  for (int p = 0; p <= pos; p++) {
    const float* k = k_cache + static_cast<size_t>(bt.slot(p, block_size)) * kv_dim + kv_head * head_dim;
    float s = 0.f;
    for (int d = 0; d < head_dim; d++) s += q[d] * k[d];
    s *= scale;
    scratch[p] = s;
    if (s > mx) mx = s;
  }
  double denom = 0;
  for (int p = 0; p <= pos; p++) { scratch[p] = std::exp(scratch[p] - mx); denom += scratch[p]; }
  for (int d = 0; d < head_dim; d++) out[d] = 0.f;
  for (int p = 0; p <= pos; p++) {
    const float w = static_cast<float>(scratch[p] / denom);
    const float* v = v_cache + static_cast<size_t>(bt.slot(p, block_size)) * kv_dim + kv_head * head_dim;
    for (int d = 0; d < head_dim; d++) out[d] += w * v[d];
  }
}

}  // namespace engine::ref
