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

// Two families of BPE tokenizer are supported, chosen by what tokenizer.json
// declares:
//   ByteLevel    GPT-2 / Qwen style. A regex pre-tokenizer splits the text, then
//                every byte maps to a printable placeholder character.
//   Metaspace    Llama / sentencepiece style. No pre-tokenizer at all; spaces
//                become U+2581 and any character outside the vocabulary falls
//                back to one <0xNN> token per UTF-8 byte.
enum class TokenizerFamily { ByteLevel, Metaspace };

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& model_dir);

  std::vector<int32_t> encode(const std::string& text) const;
  // Tokenize a prompt for generation, adding the special tokens the model
  // expects in front of it (Llama wants <s>; Qwen wants nothing). Prompts given
  // to the API as raw token ids bypass this: the client controls those exactly.
  std::vector<int32_t> encode_for_generation(const std::string& text) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  // Decode a single token; used by the streaming path. Returns raw bytes,
  // which may be an incomplete UTF-8 sequence — callers buffer until valid.
  std::string decode_token(int32_t id) const;

  // Render messages with the model's chat template (ChatML for Qwen) and
  // append the assistant generation prompt.
  std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;

  TokenizerFamily family() const { return family_; }
  int32_t eos_id() const { return eos_id_; }
  int32_t bos_id() const { return bos_id_; }
  // Whether the model expects a BOS token in front of the prompt. Llama-family
  // tokenizers declare it in their post-processor; Qwen does not use one.
  bool prepends_bos() const { return prepend_bos_; }
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
  struct Merge { int32_t rank; int32_t id; };

  // Repeatedly apply the lowest-ranked adjacent merge. Shared by both families;
  // only the initial symbol sequence differs.
  void bpe_merge(std::vector<int32_t>& ids) const;
  std::vector<int32_t> encode_byte_level(const std::string& text) const;
  std::vector<int32_t> encode_metaspace(const std::string& text) const;
  std::vector<int32_t> bpe(const std::string& word) const;

  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<std::string> id_to_token_;             // byte-level alphabet strings
  std::unordered_map<std::pair<int32_t, int32_t>, Merge, PairHash> merges_;
  std::vector<std::pair<std::string, int32_t>> added_tokens_;  // matched literally, longest first
  std::string byte_to_unicode_[256];                 // GPT-2 byte -> printable char (UTF-8)
  std::unordered_map<std::string, uint8_t> unicode_to_byte_;
  TokenizerFamily family_ = TokenizerFamily::ByteLevel;
  std::vector<int32_t> byte_fallback_;   // byte value -> <0xNN> token id, -1 if absent
  int32_t eos_id_ = -1, eot_id_ = -1, bos_id_ = -1, unk_id_ = -1;
  bool prepend_bos_ = false;
  std::string bos_token_, eos_token_;
  bool chatml_ = false;
};

}  // namespace engine
