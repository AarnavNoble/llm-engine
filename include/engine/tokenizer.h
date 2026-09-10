#pragma once
// Byte-level BPE tokenizer that reads a Hugging Face tokenizer.json directly
// (GPT-2 / Qwen / Llama-3 style). No external tokenizer library at runtime.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine {

struct ChatMessage {
  std::string role;     // system | user | assistant
  std::string content;
};

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& model_dir);

  std::vector<int32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  // Decode a single token; used by the streaming path. Returns raw bytes,
  // which may be an incomplete UTF-8 sequence — callers buffer until valid.
  std::string decode_token(int32_t id) const;

  // Render messages with the model's chat template (ChatML for Qwen) and
  // append the assistant generation prompt.
  std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;

  int32_t eos_id() const { return eos_id_; }
  bool is_stop(int32_t id) const { return id == eos_id_ || id == eot_id_; }
  // Default stop set: eos plus the chat end-of-turn token when there is one.
  std::vector<int32_t> stop_ids() const { return eot_id_ >= 0 && eot_id_ != eos_id_ ? std::vector<int32_t>{eos_id_, eot_id_} : std::vector<int32_t>{eos_id_}; }
  size_t vocab_size() const { return id_to_token_.size(); }

  // Exposed for tests.
  std::vector<std::string> pre_tokenize(const std::string& text) const;

 private:
  struct PairHash {
    size_t operator()(const std::pair<int32_t, int32_t>& p) const {
      return std::hash<int64_t>()((static_cast<int64_t>(p.first) << 32) ^ static_cast<uint32_t>(p.second));
    }
  };
  std::vector<int32_t> bpe(const std::string& word) const;

  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<std::string> id_to_token_;             // byte-level alphabet strings
  std::unordered_map<std::pair<int32_t, int32_t>, int32_t, PairHash> merge_rank_;
  std::vector<std::pair<std::string, int32_t>> added_tokens_;  // matched literally, longest first
  std::string byte_to_unicode_[256];                 // GPT-2 byte -> printable char (UTF-8)
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;
  int32_t eos_id_ = -1, eot_id_ = -1;
  std::string bos_token_, eos_token_;
  bool chatml_ = false;
};

}  // namespace engine
