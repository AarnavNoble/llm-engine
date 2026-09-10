#pragma once
#include <string>

namespace engine {

// Everything the forward pass needs, read from HF config.json. Nothing is hardcoded.
struct ModelConfig {
  std::string model_type;      // "qwen2", "llama"
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;            // hidden_size / num_attention_heads unless given
  int vocab_size = 0;
  int max_position_embeddings = 0;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.f;
  bool tie_word_embeddings = false;
  bool attention_bias = false; // Qwen2: q/k/v have bias; Llama: none
  int bos_token_id = -1;
  int eos_token_id = -1;

  int gqa_group() const { return num_attention_heads / num_key_value_heads; }
  static ModelConfig load(const std::string& model_dir);
  std::string summary() const;
};

}  // namespace engine
