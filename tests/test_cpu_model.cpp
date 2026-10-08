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

TEST_CASE("cpu model: packing is batching-invariant", "[model][slow]") {
  // The packed forward runs several sequences through one set of GEMMs. A token's
  // logits must not depend on who else happens to be in the step; if they did,
  // batching would silently change answers, which is the failure mode that
  // matters most once the CUDA backend packs the same way.
  Fixture& F = Fixture::primary();
  auto P = load_prompts(F.data);
  const int V = F.model->config().vocab_size;

  std::vector<SequencePtr> seqs;
  for (size_t i : {0u, 3u, 7u}) {
    auto s = std::make_shared<Sequence>();
    s->tokens = P["prompts"][i]["ids"].get<std::vector<int32_t>>();
    s->prompt_len = s->num_tokens();
    REQUIRE(F.kv.allocate_prompt(s->block_table, s->tokens).has_value());
    seqs.push_back(s);
  }

  auto slice_for = [](Sequence* s) {
    StepSlice sl; sl.seq = s; sl.start = 0; sl.len = s->num_tokens();
    sl.needs_logits = true; sl.is_prefill = true; return sl;
  };

  // One step containing all three sequences.
  StepInput together;
  for (auto& s : seqs) {
    together.slices.push_back(slice_for(s.get()));
    together.num_prefill_tokens += s->num_tokens();
    together.num_logits++;
  }
  std::vector<float> batched;
  F.model->forward(together, batched);
  REQUIRE(batched.size() == static_cast<size_t>(3) * V);

  // The same sequences, one per step.
  for (int i = 0; i < 3; i++) {
    StepInput alone;
    alone.slices.push_back(slice_for(seqs[i].get()));
    alone.num_prefill_tokens = seqs[i]->num_tokens();
    alone.num_logits = 1;
    std::vector<float> single;
    F.model->forward(alone, single);
    REQUIRE(single.size() == static_cast<size_t>(V));
    const float* row = batched.data() + static_cast<size_t>(i) * V;
    float worst = 0;
    for (int t = 0; t < V; t++) worst = std::max(worst, std::fabs(single[t] - row[t]));
    INFO("sequence " << i << " max |batched - alone| = " << worst);
    CHECK(worst < 1e-3f);
    CHECK(Sampler::argmax(single.data(), V) == Sampler::argmax(row, V));
  }
  for (auto& s : seqs) F.kv.free(s->block_table);
}

TEST_CASE("cpu model: a decode token packed beside a prefill chunk is unaffected", "[model][slow]") {
  // Mixed steps are the whole point of continuous batching, so check the case
  // directly: a sequence mid-generation must produce the same next token whether
  // or not someone else's prompt is being prefilled alongside it.
  Fixture& F = Fixture::primary();
  auto P = load_prompts(F.data);
  const int V = F.model->config().vocab_size;

  auto decoder = std::make_shared<Sequence>();
  decoder->tokens = P["prompts"][1]["ids"].get<std::vector<int32_t>>();
  decoder->prompt_len = decoder->num_tokens();
  REQUIRE(F.kv.allocate_prompt(decoder->block_table, decoder->tokens).has_value());

  // Prefill it, sample greedily, append, so it is genuinely mid-stream.
  StepInput prefill;
  StepSlice p; p.seq = decoder.get(); p.len = decoder->num_tokens(); p.needs_logits = true; p.is_prefill = true;
  prefill.slices.push_back(p); prefill.num_prefill_tokens = p.len; prefill.num_logits = 1;
  std::vector<float> lg;
  F.model->forward(prefill, lg);
  decoder->tokens.push_back(Sampler::argmax(lg.data(), V));
  REQUIRE(F.kv.ensure_slot(decoder->block_table, decoder->num_tokens()));

  auto decode_slice = [&] {
    StepSlice d; d.seq = decoder.get(); d.start = decoder->num_tokens() - 1; d.len = 1; d.needs_logits = true;
    return d;
  };

  StepInput alone;
  alone.slices.push_back(decode_slice());
  alone.num_decode_tokens = 1; alone.num_logits = 1;
  std::vector<float> solo;
  F.model->forward(alone, solo);

  auto other = std::make_shared<Sequence>();
  other->tokens = P["prompts"][5]["ids"].get<std::vector<int32_t>>();
  other->prompt_len = other->num_tokens();
  REQUIRE(F.kv.allocate_prompt(other->block_table, other->tokens).has_value());
  StepInput mixed;
  StepSlice op; op.seq = other.get(); op.len = other->num_tokens(); op.needs_logits = true; op.is_prefill = true;
  mixed.slices.push_back(op);                 // prefill first, as the scheduler orders it
  mixed.slices.push_back(decode_slice());
  mixed.num_prefill_tokens = op.len; mixed.num_decode_tokens = 1; mixed.num_logits = 2;
  std::vector<float> both;
  F.model->forward(mixed, both);

  const float* decode_row = both.data() + static_cast<size_t>(V);  // second logits row
  float worst = 0;
  for (int t = 0; t < V; t++) worst = std::max(worst, std::fabs(solo[t] - decode_row[t]));
  INFO("max |mixed - alone| = " << worst);
  CHECK(worst < 1e-3f);
  CHECK(Sampler::argmax(solo.data(), V) == Sampler::argmax(decode_row, V));
  F.kv.free(decoder->block_table);
  F.kv.free(other->block_table);
}

TEST_CASE("cpu model: chunked prefill gives the same logits as one-shot prefill", "[model][slow]") {
  // Chunking a prompt changes how many tokens share a forward pass and how the
  // block table is walked, but it must not change the answer. Position
  // arithmetic at chunk and block boundaries is where that would break.
  Fixture& F = Fixture::primary();
  auto P = load_prompts(F.data);
  const int V = F.model->config().vocab_size;

  // The long prompt is the only one that spans dozens of blocks.
  size_t longest = 0;
  for (size_t i = 0; i < P["prompts"].size(); i++)
    if (P["prompts"][i]["ids"].size() > P["prompts"][longest]["ids"].size()) longest = i;
  auto ids = P["prompts"][longest]["ids"].get<std::vector<int32_t>>();
  REQUIRE(ids.size() > 400);

  auto run = [&](int chunk) {
    auto s = std::make_shared<Sequence>();
    s->tokens = ids;
    s->prompt_len = s->num_tokens();
    REQUIRE(F.kv.allocate_prompt(s->block_table, s->tokens).has_value());
    std::vector<float> logits;
    int computed = 0;
    while (computed < s->num_tokens()) {
      const int len = std::min(chunk, s->num_tokens() - computed);
      StepInput step;
      StepSlice sl; sl.seq = s.get(); sl.start = computed; sl.len = len; sl.is_prefill = true;
      sl.needs_logits = (computed + len == s->num_tokens());
      step.slices.push_back(sl); step.num_prefill_tokens = len;
      step.num_logits = sl.needs_logits ? 1 : 0;
      F.model->forward(step, logits);
      computed += len;
    }
    F.kv.free(s->block_table);
    return logits;
  };

  const auto one_shot = run(static_cast<int>(ids.size()));
  REQUIRE(one_shot.size() == static_cast<size_t>(V));
  // 16 aligns with the block size, 128 spans blocks, 37 is deliberately coprime
  // with it so chunk boundaries fall mid-block.
  for (int chunk : {16, 37, 128}) {
    const auto chunked = run(chunk);
    float worst = 0;
    for (int t = 0; t < V; t++) worst = std::max(worst, std::fabs(one_shot[t] - chunked[t]));
    INFO("chunk " << chunk << ": max |one-shot - chunked| = " << worst);
    CHECK(worst < 2e-3f);
    CHECK(Sampler::argmax(chunked.data(), V) == Sampler::argmax(one_shot.data(), V));
  }
}

TEST_CASE("cpu model: a preempted sequence resumes with identical output", "[model][slow]") {
  // Recompute-on-resume throws away the KV cache and re-prefills prompt plus
  // everything generated so far. If positions came from anywhere other than the
  // sequence itself, RoPE would be applied at the wrong offsets and the
  // continuation would diverge. This is the bug the design notes predicted.
  Fixture& F = Fixture::primary();
  auto P = load_prompts(F.data);
  const int V = F.model->config().vocab_size;

  auto ids = P["prompts"][1]["ids"].get<std::vector<int32_t>>();
  const int kGenerate = 12;

  auto generate = [&](int preempt_after) {
    auto s = std::make_shared<Sequence>();
    s->tokens = ids;
    s->prompt_len = s->num_tokens();
    REQUIRE(F.kv.allocate_prompt(s->block_table, s->tokens).has_value());
    int computed = 0;
    std::vector<float> logits;
    std::vector<int32_t> produced;

    auto forward_range = [&](int start, int len, bool want_logits) {
      StepInput step;
      StepSlice sl; sl.seq = s.get(); sl.start = start; sl.len = len;
      sl.is_prefill = len > 1; sl.needs_logits = want_logits;
      step.slices.push_back(sl);
      if (len > 1) step.num_prefill_tokens = len; else step.num_decode_tokens = 1;
      step.num_logits = want_logits ? 1 : 0;
      F.model->forward(step, logits);
    };

    forward_range(0, s->num_tokens(), true);
    computed = s->num_tokens();
    for (int g = 0; g < kGenerate; g++) {
      const int32_t tok = Sampler::argmax(logits.data(), V);
      produced.push_back(tok);
      s->tokens.push_back(tok);
      if (g + 1 == preempt_after) {
        // Preempt exactly as the scheduler does: drop the cache, reallocate, and
        // re-prefill prompt plus generated tokens.
        F.kv.free(s->block_table);
        REQUIRE(F.kv.allocate_prompt(s->block_table, s->tokens).has_value());
        forward_range(0, s->num_tokens(), true);
        computed = s->num_tokens();
        continue;
      }
      REQUIRE(F.kv.ensure_slot(s->block_table, s->num_tokens()));
      forward_range(computed, 1, true);
      computed = s->num_tokens();
    }
    F.kv.free(s->block_table);
    return produced;
  };

  const auto baseline = generate(-1);
  REQUIRE(baseline.size() == static_cast<size_t>(kGenerate));
  // Preempt at a few points, including one that lands on a block boundary.
  for (int at : {1, 5, 8}) {
    const auto resumed = generate(at);
    INFO("preempted after " << at << " generated tokens");
    CHECK(resumed == baseline);
  }
}
