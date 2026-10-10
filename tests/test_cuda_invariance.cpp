// What the CUDA parity gate was not testing.
//
// test_cuda_model.cpp compares the GPU against the PyTorch dumps and against
// the CPU backend, which is the right shape of test. But all three of its
// cases drive one architecture, with prefix caching off, every slice starting
// at position 0, and a KV pool large enough that nothing is ever evicted. A
// kernel can be wrong in several specific ways and still pass all of them:
//
//   - deriving a token's position from its index within the step rather than
//     from the sequence. Decode slices are one token long, so the two agree
//     there, and a single-chunk prefill starts at 0, so they agree there too.
//     Only a prompt split across steps separates them.
//   - assuming a sequence's K/V is written once and never recomputed. A
//     preempted sequence is recomputed from scratch on readmission.
//   - assuming a prefill slice starts at position 0. With prefix caching on,
//     a sequence is admitted with num_computed already positive and its first
//     slice starts mid-sequence over blocks shared with another sequence.
//   - reading the wrong sequence's block table. The tables are concatenated
//     per step and indexed through table_offset; with one sequence in the
//     step, every offset is zero and the indirection is untested.
//
// Each case here is an invariance: the same workload computed two ways must
// produce the same tokens. That makes them robust to the reference data being
// absent and to fp16 drift, since greedy decoding either agrees or it does
// not.
//
// The second architecture matters as much as any of them. TinyLlama has GQA
// group 8 against Qwen's 7, no QKV bias, and untied embeddings, so the
// separate lm_head upload and the bias kernel's null path are only exercised
// here. tests/CMakeLists.txt defines ENGINE_MODEL_DIR_2 "to prove nothing is
// hardcoded" and until now only the CPU tests used it.
#ifdef ENGINE_CUDA

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/model.h"
#include "engine/sampler.h"
#include "engine/scheduler.h"

using namespace engine;

namespace {

std::string data_dir_for(const std::string& model_dir) {
  return std::string(ENGINE_TEST_DATA_DIR) + "/" +
         model_dir.substr(model_dir.find_last_of('/') + 1);
}

// Prompt ids, or an empty vector when the reference data is not present.
std::vector<std::vector<int32_t>> load_prompt_ids(const std::string& model_dir, size_t stride) {
  std::vector<std::vector<int32_t>> out;
  const std::string p = data_dir_for(model_dir) + "/prompts.json";
  std::ifstream f(p);
  if (!f) return out;
  auto j = nlohmann::json::parse(f);
  for (size_t i = 0; i < j["prompts"].size(); i += stride)
    out.push_back(j["prompts"][i]["ids"].get<std::vector<int32_t>>());
  return out;
}

struct RunOpts {
  bool cuda = true;
  int num_blocks = 256;
  int block_size = 16;
  bool prefix_cache = false;
  int prefill_chunk_size = 512;
  int max_num_batched_tokens = 4096;
  int max_num_seqs = 32;
  int max_tokens = 8;
  int fragment_blocks = 0;   // allocate this many single-block sequences, free
                             // every other one, before the real work starts
};

// Drive a set of prompts through the scheduler greedily and return the
// generated tokens per sequence. Greedy and ignore_eos, so the result is a
// deterministic function of the arithmetic: two configurations that disagree
// disagree because one of them is wrong.
std::vector<std::vector<int32_t>> greedy_run(const std::string& dir,
                                             const std::vector<std::vector<int32_t>>& prompts,
                                             const RunOpts& o) {
  KVCacheManager kv(o.num_blocks, o.block_size, o.prefix_cache);
  auto model = o.cuda ? make_cuda_model(dir, kv) : make_cpu_model(dir, kv);

  // Fragment the pool so block tables come back out of order. Held tables are
  // released at the end of the scope, after the measured work.
  std::vector<BlockTable> held;
  for (int i = 0; i < o.fragment_blocks; i++) {
    BlockTable bt;
    std::vector<int32_t> one(8, 7000 + i);
    if (!kv.allocate_prompt(bt, one).has_value()) break;
    held.push_back(bt);
  }
  for (size_t i = 0; i < held.size(); i += 2) kv.free(held[i]);

  Sampler sampler;
  SchedulerConfig cfg;
  cfg.max_num_batched_tokens = o.max_num_batched_tokens;
  cfg.prefill_chunk_size = o.prefill_chunk_size;
  cfg.max_num_seqs = o.max_num_seqs;
  Scheduler sch(cfg, kv);

  std::vector<SequencePtr> seqs;
  for (size_t i = 0; i < prompts.size(); i++) {
    auto s = std::make_shared<Sequence>();
    s->id = i + 1;
    s->tokens = prompts[i];
    s->prompt_len = s->num_tokens();
    s->params.max_tokens = o.max_tokens;
    s->params.temperature = 0.f;
    s->params.ignore_eos = true;
    sch.add(s);
    seqs.push_back(s);
  }

  std::vector<float> logits;
  // A bound rather than while(has_work()): a scheduling bug that stops making
  // progress should fail the test, not hang the suite.
  const int max_steps = 100000;
  for (int step = 0; step < max_steps && sch.has_work(); step++) {
    auto st = sch.schedule();
    if (st.empty()) break;
    model->forward(st, logits);
    sch.on_step_done(st, sampler.sample_step(st, logits, model->config().vocab_size));
  }

  std::vector<std::vector<int32_t>> out;
  for (auto& s : seqs)
    out.emplace_back(s->tokens.begin() + s->prompt_len, s->tokens.end());
  for (size_t i = 1; i < held.size(); i += 2) kv.free(held[i]);
  return out;
}

void require_same(const std::vector<std::vector<int32_t>>& a,
                  const std::vector<std::vector<int32_t>>& b) {
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); i++) {
    INFO("sequence " << i);
    CHECK(a[i] == b[i]);
  }
}

}  // namespace

TEST_CASE("cuda model: second architecture matches the CPU backend", "[cuda][model][slow]") {
  // The gap this file was written for. TinyLlama exercises three paths Qwen
  // never does on the GPU: untied embeddings, so lm_head is a separate upload
  // rather than the embedding table read twice; no QKV bias, so the bias
  // kernel's null early-return is the only path taken; and GQA group 8 rather
  // than 7, so a kernel that happened to work for one group size is retested.
  const std::string dir = ENGINE_MODEL_DIR_2;
  if (!std::filesystem::exists(dir + "/config.json"))
    SKIP("second model not downloaded: scripts/download_model.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0");
  auto prompts = load_prompt_ids(dir, 8);
  if (prompts.empty()) SKIP("no reference data for the second model");

  RunOpts gpu; gpu.cuda = true;
  RunOpts cpu; cpu.cuda = false;
  require_same(greedy_run(dir, prompts, cpu), greedy_run(dir, prompts, gpu));
}

TEST_CASE("cuda model: chunked prefill does not change the output", "[cuda][model][slow]") {
  // A kernel that takes a token's position from its index within the step
  // instead of from the sequence passes every existing case, because decode
  // slices are one token long and a single-chunk prefill starts at zero. Only
  // splitting a prompt across steps separates the two, and then RoPE rotates
  // by the wrong angle and attention masks the wrong range.
  const std::string dir = ENGINE_MODEL_DIR;
  auto prompts = load_prompt_ids(dir, 4);
  if (prompts.empty()) SKIP("no reference data");

  // Only worth running if some prompt is actually longer than the small chunk.
  size_t longest = 0;
  for (auto& p : prompts) longest = std::max(longest, p.size());
  if (longest <= 32) SKIP("no prompt long enough to be chunked");

  RunOpts whole; whole.prefill_chunk_size = 4096;
  RunOpts split; split.prefill_chunk_size = 16;   // forces several chunks per prompt
  require_same(greedy_run(dir, prompts, whole), greedy_run(dir, prompts, split));
}

TEST_CASE("cuda model: several sequences share a fragmented pool", "[cuda][model][slow]") {
  // The existing fragmentation case runs one sequence, so every entry in
  // table_offset is zero and the per-sequence indirection is never exercised.
  // With several sequences in a step, a kernel that reads the concatenated
  // table at the wrong offset attends over another sequence's history --
  // valid slots, plausible values, wrong tokens.
  const std::string dir = ENGINE_MODEL_DIR;
  auto prompts = load_prompt_ids(dir, 3);
  if (prompts.size() < 3) SKIP("need several prompts");

  RunOpts cpu; cpu.cuda = false; cpu.fragment_blocks = 40;
  RunOpts gpu; gpu.cuda = true;  gpu.fragment_blocks = 40;
  require_same(greedy_run(dir, prompts, cpu), greedy_run(dir, prompts, gpu));
}

TEST_CASE("cuda model: survives preemption and recompute", "[cuda][model][slow]") {
  // A pool too small for every sequence at once forces the scheduler to
  // preempt and later recompute. Recompute replays a sequence's original
  // positions into freshly allocated -- and now differently ordered -- blocks,
  // so a kernel that cached anything about a sequence's layout, or that
  // derives position from anything but the sequence, diverges here.
  const std::string dir = ENGINE_MODEL_DIR;
  auto prompts = load_prompt_ids(dir, 2);
  if (prompts.size() < 4) SKIP("need several prompts");

  RunOpts cpu;
  cpu.cuda = false; cpu.num_blocks = 48; cpu.max_num_seqs = 8; cpu.max_tokens = 24;
  RunOpts gpu = cpu;
  gpu.cuda = true;
  require_same(greedy_run(dir, prompts, cpu), greedy_run(dir, prompts, gpu));
}

TEST_CASE("cuda model: prefix caching does not change the output", "[cuda][model][slow]") {
  // With the cache on, a sequence whose prefix is already resident is admitted
  // with num_computed > 0, so its first prefill slice starts mid-sequence over
  // refcounted blocks it shares with another sequence. Nothing on the CUDA
  // path had ever run that admission path. The tokens must not depend on
  // whether the work was skipped or redone.
  const std::string dir = ENGINE_MODEL_DIR;
  auto base = load_prompt_ids(dir, 6);
  if (base.empty()) SKIP("no reference data");

  // Give several sequences a genuinely shared prefix, then distinct tails, so
  // the second and later ones hit in the cache.
  const std::vector<int32_t> shared(96, 1234);
  std::vector<std::vector<int32_t>> prompts;
  for (size_t i = 0; i < base.size() && i < 4; i++) {
    std::vector<int32_t> p = shared;
    p.insert(p.end(), base[i].begin(), base[i].end());
    prompts.push_back(p);
  }

  RunOpts off; off.prefix_cache = false;
  RunOpts on;  on.prefix_cache = true;
  require_same(greedy_run(dir, prompts, off), greedy_run(dir, prompts, on));
}


TEST_CASE("cuda model: destroying a model returns its device memory", "[cuda][model][slow]") {
  // Written because this leaked and 899k assertions did not notice.
  //
  // ~CudaModel() released the cuBLAS handle and nothing else: weights, the
  // rotation tables, the per-layer KV cache and every Scratch buffer stayed
  // resident, and Scratch had no destructor either. Nothing in the suite
  // looked at device memory, so the only symptom was a cudaMalloc failure on
  // a smaller card, reported as the KV pool being too large for the device --
  // an error that blames the configuration for a leak.
  //
  // The reporting hid it too: the allocation counter was a file-scope total
  // that only incremented, so three models in one process logged 998, 2011
  // and 3010 MiB and read like one large model rather than three live ones.
  //
  // This asserts the property that was missing rather than re-testing the
  // fix: construct and destroy repeatedly, and require device-free memory to
  // come back. A tolerance well under one model's footprint catches a whole
  // leaked model while ignoring driver-side bookkeeping.
  const std::string dir = ENGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir + "/config.json")) SKIP("no model");

  auto free_mib = []() {
    size_t f = 0, t = 0;
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemGetInfo(&f, &t) == cudaSuccess);
    return static_cast<long long>(f >> 20);
  };

  // One construction first, discarded. It pays for CUDA context creation and
  // any one-off driver allocation, neither of which a later destructor
  // returns, and both of which would otherwise look like the leak.
  { KVCacheManager kv(256, 16, false); auto m = make_cuda_model(dir, kv); }
  const long long base = free_mib();

  for (int i = 0; i < 3; i++) {
    { KVCacheManager kv(256, 16, false); auto m = make_cuda_model(dir, kv); }
    const long long now = free_mib();
    INFO("after destroying model " << i + 1 << ": " << now << " MiB free, baseline " << base);
    // Qwen2.5-0.5B is roughly 1 GiB of weights plus its cache, so 256 MiB is
    // far below one leaked model and far above driver noise.
    CHECK(base - now < 256);
  }
}

#endif  // ENGINE_CUDA
