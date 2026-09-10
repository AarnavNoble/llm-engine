#include "engine/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace engine {

ModelConfig ModelConfig::load(const std::string& model_dir) {
  std::ifstream f(model_dir + "/config.json");
  if (!f) throw std::runtime_error("cannot open " + model_dir + "/config.json");
  nlohmann::json j = nlohmann::json::parse(f);

  ModelConfig c;
  c.model_type = j.value("model_type", "llama");
  c.hidden_size = j.at("hidden_size");
  c.intermediate_size = j.at("intermediate_size");
  c.num_hidden_layers = j.at("num_hidden_layers");
  c.num_attention_heads = j.at("num_attention_heads");
  c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
  c.head_dim = j.value("head_dim", c.hidden_size / c.num_attention_heads);
  c.vocab_size = j.at("vocab_size");
  c.max_position_embeddings = j.value("max_position_embeddings", 4096);
  c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
  c.rope_theta = j.value("rope_theta", 10000.f);
  c.tie_word_embeddings = j.value("tie_word_embeddings", false);
  // Qwen2 always uses qkv bias; llama exposes it as attention_bias.
  c.attention_bias = j.value("attention_bias", c.model_type == "qwen2");
  c.bos_token_id = j.value("bos_token_id", -1);
  if (j.contains("eos_token_id")) {
    const auto& e = j["eos_token_id"];
    c.eos_token_id = e.is_array() ? e[0].get<int>() : e.get<int>();
  }
  if (c.num_attention_heads % c.num_key_value_heads != 0)
    throw std::runtime_error("num_attention_heads must be a multiple of num_key_value_heads");
  return c;
}

std::string ModelConfig::summary() const {
  std::ostringstream o;
  o << model_type << ": hidden=" << hidden_size << " layers=" << num_hidden_layers
    << " heads=" << num_attention_heads << "/" << num_key_value_heads << " head_dim=" << head_dim
    << " inter=" << intermediate_size << " vocab=" << vocab_size
    << " rope_theta=" << rope_theta << " tied=" << tie_word_embeddings << " qkv_bias=" << attention_bias;
  return o.str();
}

}  // namespace engine
