// Numerical tests of the CPU reference against PyTorch dumps produced by
// reference/dump_logits.py. Tier 1 (prompts.json, committed): per-position
// argmax and a 16-token greedy continuation. Tier 2 (ref/*.bin, generated
// locally): residual stream after every layer and last-token logits.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "engine/model.h"
#include "engine/sampler.h"
#include "engine/scheduler.h"

using namespace engine;

namespace {
nlohmann::json load_prompts(const std::string& data_dir) {
  std::ifstream f(data_dir + "/prompts.json");
  REQUIRE(f);
  return nlohmann::json::parse(f);
}

struct Ref {
  int n = 0, layers = 0, hidden = 0, vocab = 0;
  std::vector<float> resid, logits;
  bool load(const std::string& data_dir, int i) {
    std::string path = data_dir + "/ref/" + std::to_string(i) + ".bin";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int hdr[4];
    REQUIRE(std::fread(hdr, sizeof(int), 4, f) == 4);
    n = hdr[0]; layers = hdr[1]; hidden = hdr[2]; vocab = hdr[3];
    resid.resize(static_cast<size_t>(layers + 1) * n * hidden); logits.resize(vocab);
    REQUIRE(std::fread(resid.data(), sizeof(float), resid.size(), f) == resid.size());
    REQUIRE(std::fread(logits.data(), sizeof(float), logits.size(), f) == logits.size());
    std::fclose(f);
    return true;
  }
};

// Loading fp32 weights is expensive, so each model is constructed once per
// binary and reused across the test cases that need it.
struct Fixture {
  KVCacheManager kv{256, 16, true};
  std::vector<std::vector<float>> captured;  // [layer+1][ntok*hidden]
  std::unique_ptr<Model> model;
  std::string dir, data;
  explicit Fixture(const char* model_dir) : dir(model_dir) {
    data = std::string(ENGINE_TEST_DATA_DIR) + "/" + dir.substr(dir.find_last_of('/') + 1);
    CpuModelOptions o;
    o.on_layer = [this](int layer, const float* r, int ntok) {
      size_t n = static_cast<size_t>(ntok) * model->config().hidden_size;
      captured.resize(std::max(captured.size(), static_cast<size_t>(layer + 2)));
      captured[layer + 1].assign(r, r + n);
    };
    model = make_cpu_model(model_dir, kv, o);
  }
  static Fixture& primary() { static Fixture f(ENGINE_MODEL_DIR); return f; }
  static Fixture& second() { static Fixture f(ENGINE_MODEL_DIR_2); return f; }
};

bool model_present(const char* dir) {
  return std::filesystem::exists(std::string(dir) + "/config.json");
}

float max_abs_diff(const float* a, const float* b, size_t n) {
  float m = 0; for (size_t i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i])); return m;
}
}  // namespace

// Compares the engine's residual stream after every layer, and the final
// logits, against the PyTorch dumps for whichever model the fixture holds.
void check_layers_against_reference(Fixture& F) {
  auto P = load_prompts(F.data);
  const int H = F.model->config().hidden_size, V = F.model->config().vocab_size;
  int checked = 0;
  for (size_t i = 0; i < P["prompts"].size(); i++) {
    Ref ref; if (!ref.load(F.data, static_cast<int>(i))) continue;
    auto s = std::make_shared<Sequence>();
    s->tokens = P["prompts"][i]["ids"].get<std::vector<int32_t>>(); s->prompt_len = s->num_tokens();
    REQUIRE(F.kv.allocate_prompt(s->block_table, s->tokens).has_value());
    StepInput step; StepSlice sl; sl.seq = s.get(); sl.len = s->num_tokens(); sl.needs_logits = true; sl.is_prefill = true;
    step.slices.push_back(sl); step.num_prefill_tokens = sl.len; step.num_logits = 1;
    std::vector<float> logits;
    F.captured.clear();
    F.model->forward(step, logits);
    F.kv.free(s->block_table);

    INFO("prompt " << i << ": " << P["prompts"][i]["text"].get<std::string>());
    REQUIRE(ref.n == s->num_tokens());
    REQUIRE(F.captured.size() == static_cast<size_t>(ref.layers + 1));
    for (int l = 0; l <= ref.layers; l++) {
      const float* want = ref.resid.data() + static_cast<size_t>(l) * ref.n * H;
      float d = max_abs_diff(F.captured[l].data(), want, static_cast<size_t>(ref.n) * H);
      float mag = 0; for (size_t k = 0; k < static_cast<size_t>(ref.n) * H; k++) mag = std::max(mag, std::fabs(want[k]));
      INFO("layer " << (l - 1) << " max|diff| = " << d << " (max|ref| = " << mag << ")");
      CHECK(d <= 2e-3f * std::max(1.0f, mag));   // fp32 vs fp32, order-of-summation noise only
    }
    float d = max_abs_diff(logits.data(), ref.logits.data(), V);
    INFO("logits max|diff| = " << d);
    CHECK(d < 1e-2f);
    CHECK(Sampler::argmax(logits.data(), V) == P["prompts"][i]["argmax"].back().get<int32_t>());
    checked++;
  }
  // Missing dumps must not look like a pass: the per-layer comparison is the
  // core correctness claim, so say plainly that it did not run.
  if (checked == 0) SKIP("no " + F.data + "/ref/*.bin; run: python3 reference/dump_logits.py " + F.dir);
}

TEST_CASE("cpu model: per-layer residual stream and logits match PyTorch fp32", "[model][slow]") {
  check_layers_against_reference(Fixture::primary());
}

TEST_CASE("cpu model: a second architecture loads and matches PyTorch", "[model][slow][model2]") {
  // TinyLlama differs from Qwen in every way the loader could have hardcoded:
  // no q/k/v bias, an untied lm_head, GQA ratio 8 rather than 7, rms_norm_eps
  // 1e-5, and rope_theta 10,000. Its tokenizer is sentencepiece, which the C++
  // side does not read, so the prompts are supplied as token ids.
  if (!model_present(ENGINE_MODEL_DIR_2)) SKIP("second model not downloaded: scripts/download_model.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0");
  Fixture& F = Fixture::second();
  const auto& c = F.model->config();
  CHECK(c.model_type == "llama");
  CHECK_FALSE(c.attention_bias);
  CHECK_FALSE(c.tie_word_embeddings);
  CHECK(c.gqa_group() == 8);
  check_layers_against_reference(F);
}

TEST_CASE("cpu model: greedy generation through the scheduler matches HF generate", "[model][slow]") {
  auto& F = Fixture::primary();
  auto P = load_prompts(F.data);
  Sampler sampler;
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 4096;
  Scheduler sch(cfg, F.kv);
  // Run several prompts concurrently so continuous batching itself is under test.
  std::vector<SequencePtr> seqs;
  std::vector<std::vector<int32_t>> want;
  for (size_t i = 0; i < P["prompts"].size(); i += 4) {
    auto s = std::make_shared<Sequence>();
    s->id = i; s->tokens = P["prompts"][i]["ids"].get<std::vector<int32_t>>(); s->prompt_len = s->num_tokens();
    s->params.max_tokens = 16; s->params.temperature = 0.f; s->params.ignore_eos = true;
    sch.add(s); seqs.push_back(s); want.push_back(P["prompts"][i]["greedy_16"].get<std::vector<int32_t>>());
  }
  std::vector<float> logits;
  while (sch.has_work()) {
    auto step = sch.schedule(); REQUIRE_FALSE(step.empty());
    F.model->forward(step, logits);
    sch.on_step_done(step, sampler.sample_step(step, logits, F.model->config().vocab_size));
  }
  for (size_t i = 0; i < seqs.size(); i++) {
    std::vector<int32_t> got(seqs[i]->tokens.begin() + seqs[i]->prompt_len, seqs[i]->tokens.end());
    INFO("prompt " << seqs[i]->id);
    CHECK(got == want[i]);
  }
  REQUIRE(F.kv.num_used_blocks() == 0);
}
