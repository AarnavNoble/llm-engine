// Numerical check of the CUDA backend against the same PyTorch dumps the CPU
// backend is verified with.
//
// The CPU model exposes per-layer taps, so it can be compared layer by layer.
// The CUDA model cannot, so this checks what is observable from outside: the
// final logits and the argmax, for every prompt in the reference set, plus
// agreement with the CPU backend on a greedy continuation through the
// scheduler. That is the parity gate for each kernel as it lands.
#ifdef ENGINE_CUDA

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

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

// Only the final logits are needed here, so just the tail of each dump.
bool load_ref_logits(const std::string& data_dir, int i, std::vector<float>& out) {
  const std::string path = data_dir + "/ref/" + std::to_string(i) + ".bin";
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  int hdr[4];
  REQUIRE(std::fread(hdr, sizeof(int), 4, f) == 4);
  const int n = hdr[0], layers = hdr[1], hidden = hdr[2], vocab = hdr[3];
  const long resid = static_cast<long>(layers + 1) * n * hidden * sizeof(float);
  REQUIRE(std::fseek(f, resid, SEEK_CUR) == 0);
  out.resize(static_cast<size_t>(vocab));
  REQUIRE(std::fread(out.data(), sizeof(float), out.size(), f) == out.size());
  std::fclose(f);
  return true;
}

StepInput whole_prompt(Sequence* s) {
  StepInput step;
  StepSlice sl;
  sl.seq = s;
  sl.start = 0;
  sl.len = s->num_tokens();
  sl.needs_logits = true;
  sl.is_prefill = true;
  step.slices.push_back(sl);
  step.num_prefill_tokens = sl.len;
  step.num_logits = 1;
  return step;
}

}  // namespace

TEST_CASE("cuda model: logits match the PyTorch reference", "[cuda][model][slow]") {
  const std::string dir = ENGINE_MODEL_DIR;
  const std::string data = std::string(ENGINE_TEST_DATA_DIR) + "/" +
                           dir.substr(dir.find_last_of('/') + 1);
  if (!std::filesystem::exists(data + "/prompts.json"))
    SKIP("no reference data; run: python3 reference/dump_logits.py " + dir);

  KVCacheManager kv(256, 16, false);
  auto model = make_cuda_model(dir, kv);
  const int V = model->config().vocab_size;
  auto P = load_prompts(data);

  int checked = 0, argmax_agree = 0;
  float worst = 0;
  for (size_t i = 0; i < P["prompts"].size(); i++) {
    std::vector<float> want;
    if (!load_ref_logits(data, static_cast<int>(i), want)) continue;

    auto s = std::make_shared<Sequence>();
    s->tokens = P["prompts"][i]["ids"].get<std::vector<int32_t>>();
    s->prompt_len = s->num_tokens();
    REQUIRE(kv.allocate_prompt(s->block_table, s->tokens).has_value());
    std::vector<float> got;
    model->forward(whole_prompt(s.get()), got);
    kv.free(s->block_table);
    REQUIRE(got.size() == static_cast<size_t>(V));

    float d = 0, mag = 0;
    for (int t = 0; t < V; t++) {
      d = std::max(d, std::fabs(got[t] - want[t]));
      mag = std::max(mag, std::fabs(want[t]));
    }
    worst = std::max(worst, d);
    const int32_t a = Sampler::argmax(got.data(), V), b = Sampler::argmax(want.data(), V);
    if (a == b) argmax_agree++;
    INFO("prompt " << i << ": max |diff| = " << d << " against max |ref| = " << mag
                   << ", argmax " << a << " vs " << b);
    // fp16 weights with fp32 accumulation, against an fp32 reference. The bar
    // is looser than the CPU backend's for that reason, and deliberately so:
    // tightening it would mean the kernels were not actually running in fp16.
    CHECK(d < 0.35f * std::max(mag, 1.0f) * 0.1f + 0.2f);
    checked++;
  }
  if (checked == 0) SKIP("no reference dumps found");
  INFO("worst |diff| across " << checked << " prompts = " << worst);
  // An occasional flip on a near-tie is expected in fp16; a systematic one is not.
  CHECK(argmax_agree >= checked - 1);
}

TEST_CASE("cuda model: greedy generation matches the CPU backend", "[cuda][model][slow]") {
  // The strongest statement available without per-layer taps: both backends,
  // driven through the same scheduler, produce the same tokens.
  const std::string dir = ENGINE_MODEL_DIR;
  const std::string data = std::string(ENGINE_TEST_DATA_DIR) + "/" +
                           dir.substr(dir.find_last_of('/') + 1);
  if (!std::filesystem::exists(data + "/prompts.json")) SKIP("no reference data");
  auto P = load_prompts(data);

  auto run = [&](bool cuda) {
    KVCacheManager kv(256, 16, false);
    auto model = cuda ? make_cuda_model(dir, kv) : make_cpu_model(dir, kv);
    Sampler sampler;
    SchedulerConfig cfg;
    cfg.max_num_batched_tokens = 4096;
    Scheduler sch(cfg, kv);
    std::vector<SequencePtr> seqs;
    for (size_t i = 0; i < P["prompts"].size(); i += 8) {
      auto s = std::make_shared<Sequence>();
      s->id = i + 1;
      s->tokens = P["prompts"][i]["ids"].get<std::vector<int32_t>>();
      s->prompt_len = s->num_tokens();
      s->params.max_tokens = 8;
      s->params.temperature = 0.f;
      s->params.ignore_eos = true;
      sch.add(s);
      seqs.push_back(s);
    }
    std::vector<float> logits;
    while (sch.has_work()) {
      auto st = sch.schedule();
      if (st.empty()) break;
      model->forward(st, logits);
      sch.on_step_done(st, sampler.sample_step(st, logits, model->config().vocab_size));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto& s : seqs)
      out.emplace_back(s->tokens.begin() + s->prompt_len, s->tokens.end());
    return out;
  };

  const auto on_cpu = run(false);
  const auto on_gpu = run(true);
  REQUIRE(on_cpu.size() == on_gpu.size());
  for (size_t i = 0; i < on_cpu.size(); i++) {
    INFO("sequence " << i);
    CHECK(on_gpu[i] == on_cpu[i]);
  }
}

#endif  // ENGINE_CUDA
