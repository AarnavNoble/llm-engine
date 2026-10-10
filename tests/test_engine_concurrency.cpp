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
        auto res = eng.submit(prompt, p, [&out](const Sequence&, int32_t tok, FinishReason r) {
          if (tok >= 0) { out.tokens++; return; }
          if (r == FinishReason::Abort) out.aborted++; else out.finished++;
          out.terminal++;
        });
        REQUIRE(res.ok());
        {
          std::lock_guard<std::mutex> g(ids_mu);
          ids[static_cast<size_t>(t) * kPerThread + i] = res.seq->id;
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
    auto res = eng.submit(prompt, p, [&](const Sequence&, int32_t tok, FinishReason r) {
      if (tok < 0) { if (r == FinishReason::Abort) aborted++; terminal++; }
    });
    REQUIRE(res.ok());
    seqs.push_back(res.seq);
  }

  // Let a little work happen, then drain from this thread while the engine runs.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::thread drainer([&] { eng.drain(); });
  // Submissions during the drain must be refused rather than silently dropped.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  SamplingParams late;
  late.max_tokens = 4;
  const auto late_res = eng.submit({1234}, late, nullptr);
  CHECK_FALSE(late_res.ok());
  CHECK(late_res.status == SubmitStatus::Draining);
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

TEST_CASE("engine: an idle engine is healthy, however long it has been quiet", "[model][slow]") {
  // Regression. Health used to measure time since the last completed step
  // without asking whether there was anything to do, and the loop sleeps when
  // idle, so a quiet server reported itself dead and would have been
  // restarted in a loop by any orchestrator watching liveness.
  auto cfg = test_config();
  cfg.stall_timeout_seconds = 0.5;   // far shorter than the idle period below
  Engine eng(cfg);
  eng.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  auto h = eng.health();
  INFO("problems: " << (h.problems.empty() ? "none" : h.problems[0]));
  CHECK(h.alive);
  CHECK(h.ready);
  // The loop stamps progress on every pass, so an idle engine looks recent.
  CHECK(h.seconds_since_progress < 0.5);
  eng.stop();
}

TEST_CASE("engine: load shedding rejects past the queue depth and counts it", "[model][slow]") {
  auto cfg = test_config();
  cfg.max_queue_depth = 3;
  cfg.sched.max_num_seqs = 1;        // keep work in the queue rather than running it
  Engine eng(cfg);
  eng.start();

  int accepted = 0, overloaded = 0;
  for (int i = 0; i < 24; i++) {
    SamplingParams p;
    p.max_tokens = 64;
    p.temperature = 0.f;
    p.ignore_eos = true;
    std::vector<int32_t> prompt(16, static_cast<int32_t>(1000 + i));
    auto res = eng.submit(prompt, p, nullptr);
    if (res.ok()) {
      accepted++;
    } else {
      CHECK(res.status == SubmitStatus::Overloaded);
      CHECK_FALSE(res.reason.empty());
      CHECK(res.retry_after_seconds > 0);   // the client needs to know when to come back
      overloaded++;
    }
  }
  // Some are accepted and some shed; the point is that the queue is bounded
  // rather than growing until every request is slow.
  CHECK(accepted > 0);
  CHECK(overloaded > 0);
  CHECK(accepted + overloaded == 24);
  CHECK(eng.metrics().requests_rejected_overload.load() == static_cast<uint64_t>(overloaded));
  eng.stop();
}

TEST_CASE("engine: a request that waits past its deadline is abandoned", "[model][slow]") {
  // Capacity should go to work someone is still waiting for. Without a deadline
  // an overloaded engine keeps serving requests whose clients gave up long ago.
  auto cfg = test_config();
  cfg.sched.max_num_seqs = 1;
  cfg.sched.max_queue_wait_seconds = 0.5;
  Engine eng(cfg);
  eng.start();

  std::atomic<int> timed_out{0}, finished{0}, terminal{0};
  for (int i = 0; i < 8; i++) {
    SamplingParams p;
    p.max_tokens = 48;
    p.temperature = 0.f;
    p.ignore_eos = true;
    std::vector<int32_t> prompt(24, static_cast<int32_t>(2000 + i));
    auto res = eng.submit(prompt, p, [&](const Sequence&, int32_t tok, FinishReason r) {
      if (tok >= 0) return;
      if (r == FinishReason::Timeout) timed_out++; else finished++;
      terminal++;
    });
    REQUIRE(res.ok());
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
  while (terminal.load() < 8 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  REQUIRE(terminal.load() == 8);
  CHECK(timed_out.load() > 0);        // the queue was long enough that some expired
  CHECK(finished.load() > 0);         // and the engine still made progress
  CHECK(eng.metrics().requests_timed_out.load() == static_cast<uint64_t>(timed_out.load()));
  eng.stop();
  CHECK(eng.metrics().kv_blocks_used.load() == 0);   // expiry frees everything
}
