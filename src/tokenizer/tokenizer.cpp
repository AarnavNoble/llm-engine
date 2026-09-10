#include "engine/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <climits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace engine {

#include "unicode_tables.inc"

namespace {

bool in_ranges(uint32_t cp, const uint32_t (*r)[2], size_t n) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < r[mid][0]) hi = mid;
    else if (cp > r[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}
bool is_letter(uint32_t cp) { return in_ranges(cp, kLetterRanges, sizeof(kLetterRanges) / sizeof(kLetterRanges[0])); }
bool is_number(uint32_t cp) { return in_ranges(cp, kNumberRanges, sizeof(kNumberRanges) / sizeof(kNumberRanges[0])); }
bool is_space(uint32_t cp)  { return in_ranges(cp, kSpaceRanges,  sizeof(kSpaceRanges)  / sizeof(kSpaceRanges[0])); }
bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Decode one UTF-8 codepoint starting at s[i]; returns its byte length.
size_t utf8_decode(const std::string& s, size_t i, uint32_t& cp) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) { cp = c; return 1; }
  size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
  if (len == 1 || i + len > s.size()) { cp = 0xFFFD; return 1; }
  cp = c & (0xFF >> (len + 1));
  for (size_t k = 1; k < len; k++) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  return len;
}

std::string utf8_encode(uint32_t cp) {
  std::string o;
  if (cp < 0x80) o += static_cast<char>(cp);
  else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
  else if (cp < 0x10000) { o += static_cast<char>(0xE0 | (cp >> 12)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
  else { o += static_cast<char>(0xF0 | (cp >> 18)); o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
  return o;
}

std::vector<uint32_t> to_codepoints(const std::string& s) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < s.size();) { uint32_t cp; i += utf8_decode(s, i, cp); out.push_back(cp); }
  return out;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& model_dir) {
  std::ifstream f(model_dir + "/tokenizer.json");
  if (!f) throw std::runtime_error("cannot open " + model_dir + "/tokenizer.json");
  nlohmann::json j = nlohmann::json::parse(f);

  // GPT-2 byte <-> unicode table.
  {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<uint32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; b++)
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
    for (size_t i = 0; i < bs.size(); i++) {
      byte_to_unicode_[bs[i]] = utf8_encode(cs[i]);
      unicode_to_byte_[byte_to_unicode_[bs[i]]] = static_cast<uint8_t>(bs[i]);
    }
  }

  const auto& model = j.at("model");
  if (model.at("type") != "BPE") throw std::runtime_error("only BPE tokenizers are supported");
  const auto& vocab = model.at("vocab");
  id_to_token_.resize(vocab.size());
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    int32_t id = it.value().get<int32_t>();
    token_to_id_[it.key()] = id;
    if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(id + 1);
    id_to_token_[id] = it.key();
  }
  const auto& merges = model.at("merges");
  int32_t rank = 0;
  for (const auto& m : merges) {
    std::string a, b;
    if (m.is_string()) { std::string s = m.get<std::string>(); size_t sp = s.find(' '); a = s.substr(0, sp); b = s.substr(sp + 1); }
    else { a = m[0].get<std::string>(); b = m[1].get<std::string>(); }
    auto ia = token_to_id_.find(a), ib = token_to_id_.find(b);
    if (ia == token_to_id_.end() || ib == token_to_id_.end()) continue;
    merge_rank_.emplace(std::make_pair(ia->second, ib->second), rank++);
  }

  for (const auto& t : j.value("added_tokens", nlohmann::json::array())) {
    std::string content = t.at("content");
    int32_t id = t.at("id");
    if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(id + 1);
    id_to_token_[id] = content;
    token_to_id_[content] = id;
    added_tokens_.emplace_back(content, id);
  }
  std::sort(added_tokens_.begin(), added_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  // eos / chat template from tokenizer_config.json + generation_config.json.
  std::ifstream tc(model_dir + "/tokenizer_config.json");
  if (tc) {
    nlohmann::json c = nlohmann::json::parse(tc);
    auto tok_str = [&](const char* key) -> std::string {
      if (!c.contains(key) || c[key].is_null()) return "";
      return c[key].is_string() ? c[key].get<std::string>() : c[key].value("content", "");
    };
    eos_token_ = tok_str("eos_token");
    bos_token_ = tok_str("bos_token");
    if (auto it = token_to_id_.find(eos_token_); it != token_to_id_.end()) eos_id_ = it->second;
    std::string tmpl = c.value("chat_template", "");
    chatml_ = tmpl.find("<|im_start|>") != std::string::npos;
    if (auto it = token_to_id_.find("<|im_end|>"); chatml_ && it != token_to_id_.end()) eot_id_ = it->second;
  }
  std::ifstream gc(model_dir + "/generation_config.json");
  if (gc) {  // Qwen: eos_token_id = [151645, 151643]; the chat eos is <|im_end|>
    nlohmann::json g = nlohmann::json::parse(gc);
    if (g.contains("eos_token_id")) {
      const auto& e = g["eos_token_id"];
      if (e.is_array()) { eot_id_ = e[0].get<int32_t>(); if (e.size() > 1) eos_id_ = e[1].get<int32_t>(); }
      else eos_id_ = e.get<int32_t>();
    }
  }
  if (eos_id_ < 0) throw std::runtime_error("could not determine eos token id");
}

// Implements the Qwen2 / GPT-4-style split pattern by hand:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
  std::vector<uint32_t> cps = to_codepoints(text);
  std::vector<std::string> out;
  const size_t n = cps.size();
  size_t i = 0;
  auto emit = [&](size_t a, size_t b) { std::string s; for (size_t k = a; k < b; k++) s += utf8_encode(cps[k]); out.push_back(std::move(s)); };
  auto lower = [](uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };

  while (i < n) {
    uint32_t c = cps[i];
    // 1. contractions
    if (c == '\'' && i + 1 < n) {
      uint32_t c1 = lower(cps[i + 1]);
      uint32_t c2 = i + 2 < n ? lower(cps[i + 2]) : 0;
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { emit(i, i + 2); i += 2; continue; }
      if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { emit(i, i + 3); i += 3; continue; }
    }
    // 2. optional non-letter/number/newline prefix + letters
    if (!is_newline(c) && !is_number(c)) {
      size_t j = i;
      if (!is_letter(c)) j++;  // prefix char
      if (j < n && is_letter(cps[j])) {
        while (j < n && is_letter(cps[j])) j++;
        emit(i, j); i = j; continue;
      }
    }
    // 3. single number
    if (is_number(c)) { emit(i, i + 1); i++; continue; }
    // 4. optional space + punctuation run + trailing newlines
    {
      size_t j = i;
      if (cps[j] == ' ') j++;
      if (j < n && !is_space(cps[j]) && !is_letter(cps[j]) && !is_number(cps[j])) {
        while (j < n && !is_space(cps[j]) && !is_letter(cps[j]) && !is_number(cps[j])) j++;
        while (j < n && is_newline(cps[j])) j++;
        emit(i, j); i = j; continue;
      }
    }
    // 5. \s*[\r\n]+
    if (is_space(c)) {
      size_t j = i;
      while (j < n && is_space(cps[j])) j++;
      size_t last_nl = std::string::npos;
      for (size_t k = i; k < j; k++) if (is_newline(cps[k])) last_nl = k;
      if (last_nl != std::string::npos) { emit(i, last_nl + 1); i = last_nl + 1; continue; }
      // 6. \s+(?!\S): whitespace run, but leave the last space if followed by non-space
      if (j < n && j - i > 1) { emit(i, j - 1); i = j - 1; continue; }
      // 7. \s+
      emit(i, j); i = j; continue;
    }
    // fallback: single char (should not happen)
    emit(i, i + 1); i++;
  }
  return out;
}

std::vector<int32_t> Tokenizer::bpe(const std::string& word) const {
  // Map bytes -> byte-level alphabet symbols -> initial ids.
  std::vector<int32_t> ids;
  for (unsigned char b : word) {
    auto it = token_to_id_.find(byte_to_unicode_[b]);
    if (it == token_to_id_.end()) throw std::runtime_error("byte symbol missing from vocab");
    ids.push_back(it->second);
  }
  if (ids.size() < 2) return ids;
  while (true) {
    int32_t best_rank = INT32_MAX; size_t best_i = 0;
    for (size_t i = 0; i + 1 < ids.size(); i++) {
      auto it = merge_rank_.find({ids[i], ids[i + 1]});
      if (it != merge_rank_.end() && it->second < best_rank) { best_rank = it->second; best_i = i; }
    }
    if (best_rank == INT32_MAX) break;
    const std::string merged = id_to_token_[ids[best_i]] + id_to_token_[ids[best_i + 1]];
    int32_t mid = token_to_id_.at(merged);
    // merge every occurrence of this pair (same rank), left to right
    std::vector<int32_t> next; next.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
      if (i + 1 < ids.size() && ids[i] == ids[best_i] && ids[i + 1] == ids[best_i + 1]) { next.push_back(mid); i++; }
      else next.push_back(ids[i]);
    }
    ids.swap(next);
    if (ids.size() < 2) break;
  }
  return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> out;
  // Split on added/special tokens first (matched literally, longest first).
  size_t pos = 0;
  while (pos < text.size()) {
    size_t best_at = std::string::npos; const std::pair<std::string, int32_t>* best = nullptr;
    for (const auto& t : added_tokens_) {
      size_t at = text.find(t.first, pos);
      if (at != std::string::npos && (best == nullptr || at < best_at || (at == best_at && t.first.size() > best->first.size()))) { best_at = at; best = &t; }
    }
    size_t end = best ? best_at : text.size();
    if (end > pos)
      for (const auto& w : pre_tokenize(text.substr(pos, end - pos))) { auto ids = bpe(w); out.insert(out.end(), ids.begin(), ids.end()); }
    if (best) { out.push_back(best->second); pos = best_at + best->first.size(); }
    else pos = end;
  }
  return out;
}

std::string Tokenizer::decode_token(int32_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "";
  const std::string& tok = id_to_token_[id];
  for (const auto& a : added_tokens_) if (a.second == id) return tok;  // special tokens are stored raw
  std::string out;
  for (size_t i = 0; i < tok.size();) {
    uint32_t cp; size_t len = utf8_decode(tok, i, cp);
    auto it = unicode_to_byte_.find(tok.substr(i, len));
    out += it != unicode_to_byte_.end() ? static_cast<char>(it->second) : '?';
    i += len;
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string out;
  for (int32_t id : ids) out += decode_token(id);
  return out;
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages) const {
  if (!chatml_) throw std::runtime_error("only ChatML chat templates are implemented");
  std::string s;
  bool has_system = !messages.empty() && messages[0].role == "system";
  if (!has_system) s += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
  for (const auto& m : messages) s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
  s += "<|im_start|>assistant\n";
  return s;
}

}  // namespace engine
