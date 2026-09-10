#include <catch2/catch_test_macros.hpp>
#include "engine/config.h"

TEST_CASE("Qwen2.5-0.5B config parses with the expected dimensions", "[model]") {
  auto c = engine::ModelConfig::load(ENGINE_MODEL_DIR);
  REQUIRE(c.model_type == "qwen2");
  REQUIRE(c.hidden_size == 896);
  REQUIRE(c.num_hidden_layers == 24);
  REQUIRE(c.num_attention_heads == 14);
  REQUIRE(c.num_key_value_heads == 2);
  REQUIRE(c.gqa_group() == 7);
  REQUIRE(c.head_dim == 64);
  REQUIRE(c.intermediate_size == 4864);
  REQUIRE(c.vocab_size == 151936);
  REQUIRE(c.rms_norm_eps == 1e-6f);
  REQUIRE(c.rope_theta == 1000000.f);
  REQUIRE(c.tie_word_embeddings);
  REQUIRE(c.attention_bias);
  REQUIRE(c.eos_token_id == 151645);
}
