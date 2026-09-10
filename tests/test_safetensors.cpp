#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "engine/config.h"
#include "engine/safetensors.h"

using namespace engine;

TEST_CASE("safetensors: every expected tensor exists with the right shape", "[model]") {
  auto c = ModelConfig::load(ENGINE_MODEL_DIR);
  SafeTensorsDir w(ENGINE_MODEL_DIR);
  const int64_t H = c.hidden_size, KV = c.num_key_value_heads * c.head_dim, I = c.intermediate_size;

  REQUIRE(w.get("model.embed_tokens.weight").shape == std::vector<int64_t>{c.vocab_size, H});
  REQUIRE(w.get("model.norm.weight").shape == std::vector<int64_t>{H});
  REQUIRE_FALSE(w.has("lm_head.weight"));  // tied
  for (int l = 0; l < c.num_hidden_layers; l++) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    REQUIRE(w.get(p + "self_attn.q_proj.weight").shape == std::vector<int64_t>{H, H});
    REQUIRE(w.get(p + "self_attn.k_proj.weight").shape == std::vector<int64_t>{KV, H});
    REQUIRE(w.get(p + "self_attn.v_proj.weight").shape == std::vector<int64_t>{KV, H});
    REQUIRE(w.get(p + "self_attn.o_proj.weight").shape == std::vector<int64_t>{H, H});
    REQUIRE(w.get(p + "self_attn.q_proj.bias").shape == std::vector<int64_t>{H});
    REQUIRE(w.get(p + "self_attn.k_proj.bias").shape == std::vector<int64_t>{KV});
    REQUIRE(w.get(p + "self_attn.v_proj.bias").shape == std::vector<int64_t>{KV});
    REQUIRE_FALSE(w.has(p + "self_attn.o_proj.bias"));
    REQUIRE(w.get(p + "mlp.gate_proj.weight").shape == std::vector<int64_t>{I, H});
    REQUIRE(w.get(p + "mlp.up_proj.weight").shape == std::vector<int64_t>{I, H});
    REQUIRE(w.get(p + "mlp.down_proj.weight").shape == std::vector<int64_t>{H, I});
    REQUIRE(w.get(p + "input_layernorm.weight").shape == std::vector<int64_t>{H});
    REQUIRE(w.get(p + "post_attention_layernorm.weight").shape == std::vector<int64_t>{H});
  }
  REQUIRE(w.names().size() == static_cast<size_t>(2 + 12 * c.num_hidden_layers));
}

TEST_CASE("safetensors: weights are bf16 and convert to finite, sane fp32", "[model]") {
  SafeTensorsDir w(ENGINE_MODEL_DIR);
  const auto& v = w.get("model.norm.weight");
  REQUIRE(v.dtype == DType::BF16);
  Tensor t = v.to_f32();
  REQUIRE(t.numel() == 896);
  double mean = 0;
  for (float x : t.data) { REQUIRE(std::isfinite(x)); mean += x; }
  mean /= t.numel();
  REQUIRE(mean > 0.1);  // norm gains are O(1) positive
  REQUIRE(mean < 20.0);
  auto h = v.to_f16();
  REQUIRE(h.size() == 896);
  REQUIRE(std::fabs(f16_to_f32(h[0]) - t.data[0]) < 1e-2f * std::fabs(t.data[0]) + 1e-3f);
}
