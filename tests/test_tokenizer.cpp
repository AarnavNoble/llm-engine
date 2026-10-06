#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "engine/tokenizer.h"

using namespace engine;

static std::vector<nlohmann::json> load_rows(const std::string& model_dir) {
  const std::string name = model_dir.substr(model_dir.find_last_of('/') + 1);
  std::ifstream f(std::string(ENGINE_TEST_DATA_DIR) + "/" + name + "/tokens.jsonl");
  REQUIRE(f);
  std::vector<nlohmann::json> rows; std::string line;
  while (std::getline(f, line)) if (!line.empty()) rows.push_back(nlohmann::json::parse(line));
  return rows;
}

// Both families run the same corpus: a byte-level tokenizer (Qwen) and a
// metaspace/byte-fallback one (TinyLlama) must each reproduce Hugging Face exactly.
static void check_encode_matches(const char* model_dir) {
  Tokenizer tok(model_dir);
  for (const auto& r : load_rows(model_dir)) {
    std::string text = r["text"];
    auto want = r["ids"].get<std::vector<int32_t>>();
    INFO("text: " << nlohmann::json(text).dump());
    CHECK(tok.encode(text) == want);
  }
}

static void check_roundtrip(const char* model_dir) {
  Tokenizer tok(model_dir);
  for (const auto& r : load_rows(model_dir)) {
    std::string text = r["text"];
    INFO("text: " << nlohmann::json(text).dump());
    CHECK(tok.decode(tok.encode(text)) == text);
  }
}

TEST_CASE("tokenizer: byte-level encode matches Hugging Face", "[model]") {
  check_encode_matches(ENGINE_MODEL_DIR);
}

TEST_CASE("tokenizer: metaspace encode matches Hugging Face", "[model][model2]") {
  if (!std::filesystem::exists(std::string(ENGINE_MODEL_DIR_2) + "/tokenizer.json"))
    SKIP("second model not downloaded");
  Tokenizer tok(ENGINE_MODEL_DIR_2);
  REQUIRE(tok.family() == TokenizerFamily::Metaspace);
  REQUIRE(tok.prepends_bos());          // Llama expects <s> in front of the prompt
  REQUIRE(tok.bos_id() == 1);
  REQUIRE(tok.eos_id() == 2);
  check_encode_matches(ENGINE_MODEL_DIR_2);
}

TEST_CASE("tokenizer: metaspace decode round-trips", "[model][model2]") {
  if (!std::filesystem::exists(std::string(ENGINE_MODEL_DIR_2) + "/tokenizer.json"))
    SKIP("second model not downloaded");
  check_roundtrip(ENGINE_MODEL_DIR_2);
}

TEST_CASE("tokenizer: decode(encode(x)) == x", "[model]") {
  check_roundtrip(ENGINE_MODEL_DIR);
}

TEST_CASE("tokenizer: chat template matches HF rendering", "[model]") {
  Tokenizer tok(ENGINE_MODEL_DIR);
  for (const auto& r : load_rows(ENGINE_MODEL_DIR)) {
    if (!r.contains("chat")) continue;
    std::vector<ChatMessage> msgs;
    for (const auto& m : r["chat"]) msgs.push_back({m["role"], m["content"]});
    CHECK(tok.apply_chat_template(msgs) == r["text"].get<std::string>());
  }
}

TEST_CASE("tokenizer: special ids", "[model]") {
  Tokenizer tok(ENGINE_MODEL_DIR);
  REQUIRE(tok.family() == TokenizerFamily::ByteLevel);
  REQUIRE_FALSE(tok.prepends_bos());
  REQUIRE(tok.vocab_size() == 151665);
  REQUIRE(tok.is_stop(151645));  // <|im_end|>
  REQUIRE(tok.is_stop(151643));  // <|endoftext|>
  REQUIRE_FALSE(tok.is_stop(0));
  REQUIRE(tok.decode_token(151644) == "<|im_start|>");
}
