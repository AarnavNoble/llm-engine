#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include "engine/tokenizer.h"

using namespace engine;

static std::vector<nlohmann::json> load_rows() {
  std::ifstream f(std::string(ENGINE_TEST_DATA_DIR) + "/tokens.jsonl");
  REQUIRE(f);
  std::vector<nlohmann::json> rows; std::string line;
  while (std::getline(f, line)) if (!line.empty()) rows.push_back(nlohmann::json::parse(line));
  return rows;
}

TEST_CASE("tokenizer: encode matches Hugging Face on every corpus line", "[model]") {
  Tokenizer tok(ENGINE_MODEL_DIR);
  for (const auto& r : load_rows()) {
    std::string text = r["text"];
    auto want = r["ids"].get<std::vector<int32_t>>();
    INFO("text: " << nlohmann::json(text).dump());
    CHECK(tok.encode(text) == want);
  }
}

TEST_CASE("tokenizer: decode(encode(x)) == x", "[model]") {
  Tokenizer tok(ENGINE_MODEL_DIR);
  for (const auto& r : load_rows()) {
    std::string text = r["text"];
    INFO("text: " << nlohmann::json(text).dump());
    CHECK(tok.decode(tok.encode(text)) == text);
  }
}

TEST_CASE("tokenizer: chat template matches HF rendering", "[model]") {
  Tokenizer tok(ENGINE_MODEL_DIR);
  for (const auto& r : load_rows()) {
    if (!r.contains("chat")) continue;
    std::vector<ChatMessage> msgs;
    for (const auto& m : r["chat"]) msgs.push_back({m["role"], m["content"]});
    CHECK(tok.apply_chat_template(msgs) == r["text"].get<std::string>());
  }
}

TEST_CASE("tokenizer: special ids", "[model]") {
  Tokenizer tok(ENGINE_MODEL_DIR);
  REQUIRE(tok.vocab_size() == 151665);
  REQUIRE(tok.is_stop(151645));  // <|im_end|>
  REQUIRE(tok.is_stop(151643));  // <|endoftext|>
  REQUIRE_FALSE(tok.is_stop(0));
  REQUIRE(tok.decode_token(151644) == "<|im_start|>");
}
