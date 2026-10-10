// The scheduler and allocator are single-threaded by design, but the Engine is
// not: requests arrive from any number of HTTP threads, aborts arrive from
// those threads too, and token callbacks fire on the engine thread. These tests
// hammer that boundary. Run the binary under ThreadSanitizer to make them say
// something about races rather than only about outcomes.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "engine/engine.h"

using namespace engine;

namespace {

EngineConfig test_config() {
  EngineConfig c;
  c.model_dir = ENGINE_MODEL_DIR;
  c.num_blocks = 256;
  c.block_size = 16;
  c.sched.max_num_seqs = 8;
  c.sched.max_num_batched_tokens = 512;
  return c;
}

struct Outcome {
  std::atomic<int> tokens{0};
  std::atomic<int> terminal{0};
  std::atomic<int> finished{0}, aborted{0};
};

}  // namespace

TEST_CASE("engine: concurrent submits and aborts all reach a terminal state", "[model][slow]") {
  Engine eng(test_config());
  eng.start();

  const int kThreads = 6, kPerThread = 8;
  Outcome out;
  std::vector<uint64_t> ids(static_cast<size_t>(kThreads) * kPerThread, 0);
  std::mutex ids_mu;

  std::vector<std::thread> submitters;
  for (int t = 0; t < kThreads; t++) {
    submitters.emplace_back([&, t] {
      std::mt19937_64 rng(static_cast<uint64_t>(t) + 1);
      for (int i = 0; i < kPerThread; i++) {
        SamplingParams p;
        p.max_tokens = 1 + static_cast<int>(rng() % 12);
        p.temperature = (rng() & 1) ? 0.f : 0.8f;
        p.seed = rng();
        p.ignore_eos = true;
        std::vector<int32_t> prompt(1 + rng() % 24);
        for (auto& tk : prompt) tk = static_cast<int32_t>(1000 + (rng() % 2000));
        auto seq = eng.submit(prompt, p, [&out](const Sequence&, int32_t tok, FinishReason r) {
          if (tok >= 0) { out.tokens++; return; }
          if (r == FinishReason::Abort) out.aborted++; else out.finished++;
          out.terminal++;
        });
        REQUIRE(seq != nullptr);
        {
          std::lock_guard<std::mutex> g(ids_mu);
          ids[static_cast<size_t>(t) * kPerThread + i] = seq->id;
        }
        if ((rng() % 8) == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
  }

  // Abort a scattering of requests from a different thread while they run.
  std::thread aborter([&] {
    std::mt19937_64 rng(99);
    for (int k = 0; k < 10; k++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      std::lock_guard<std::mutex> g(ids_mu);
      for (size_t i = rng() % ids.size(); i < ids.size(); i += 7)
        if (ids[i] != 0) eng.abort(ids[i]);
    }
  });

  for (auto& th : submitters) th.join();
  aborter.join();

  // Every request must end exactly once, aborted or finished.
  const int total = kThreads * kPerThread;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
  while (out.terminal.load() < total && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(out.terminal.load() == total);
  REQUIRE(out.finished.load() + out.aborted.load() == total);
  CHECK(out.tokens.load() > 0);

  eng.stop();
  // Metrics must agree with what the callbacks saw, and the pool must be empty.
  const auto& m = eng.metrics();
  CHECK(m.requests_total.load() == static_cast<uint64_t>(total));
  CHECK(m.requests_finished.load() + m.requests_aborted.load() == static_cast<uint64_t>(total));
  CHECK(m.kv_blocks_used.load() == 0);
  CHECK(m.queue_depth.load() == 0);
  CHECK(m.running_seqs.load() == 0);
}

TEST_CASE("engine: drain finishes in-flight work and refuses new requests", "[model][slow]") {
  Engine eng(test_config());
  eng.start();

  std::atomic<int> terminal{0}, aborted{0};
  std::vector<SequencePtr> seqs;
  for (int i = 0; i < 6; i++) {
    SamplingParams p;
    p.max_tokens = 8;
    p.temperature = 0.f;
    p.ignore_eos = true;
    std::vector<int32_t> prompt(8, static_cast<int32_t>(1000 + i));
    seqs.push_back(eng.submit(prompt, p, [&](const Sequence&, int32_t tok, FinishReason r) {
      if (tok < 0) { if (r == FinishReason::Abort) aborted++; terminal++; }
    }));
    REQUIRE(seqs.back() != nullptr);
  }

  // Let a little work happen, then drain from this thread while the engine runs.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::thread drainer([&] { eng.drain(); });
  // Submissions during the drain must be refused rather than silently dropped.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  SamplingParams late;
  late.max_tokens = 4;
  CHECK(eng.submit({1234}, late, nullptr) == nullptr);
  drainer.join();

  REQUIRE(terminal.load() == 6);
  CHECK(aborted.load() == 0);               // drain finishes work, it does not cancel it
  for (const auto& s : seqs) {
    REQUIRE(s->is_finished());
    CHECK(s->num_generated() == 8);
    CHECK(s->block_table.empty());
  }
  CHECK(eng.metrics().kv_blocks_used.load() == 0);
  CHECK_FALSE(eng.accepting());
}
