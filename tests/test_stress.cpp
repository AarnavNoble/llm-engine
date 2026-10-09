// Randomised stress test for the scheduler and the block allocator.
//
// The hand-written tests each pin one behaviour. This one hunts for the
// combinations nobody thought to write down: a tight pool with a large seq
// limit, static batching with a shared prefix, aborts landing mid-prefill,
// preemption while another sequence is crossing a block boundary. It runs with
// a fake model so it is fast enough for CI.
//
// Every iteration asserts the invariants that must hold for any schedule:
//   - block accounting is conserved at every step
//   - no sequence computes more tokens than it has, or generates past its cap
//   - every request reaches a terminal state
//   - the pool is empty once the queue has drained
//   - the scheduler always makes progress, so no configuration can livelock
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <numeric>
#include <random>
#include <set>

#include "engine/scheduler.h"

using namespace engine;

namespace {

struct Workload {
  SchedulerConfig cfg;
  int num_blocks, block_size;
  bool prefix_caching;
  int num_requests;
};

Workload random_workload(std::mt19937_64& rng) {
  auto pick = [&](std::initializer_list<int> xs) {
    std::vector<int> v(xs);
    return v[rng() % v.size()];
  };
  Workload w;
  w.block_size = pick({1, 2, 4, 8, 16});
  w.num_blocks = pick({4, 8, 16, 32, 64, 128});
  w.prefix_caching = (rng() & 1) != 0;
  w.cfg.continuous = (rng() % 4) != 0;            // mostly continuous, sometimes static
  w.cfg.max_num_seqs = pick({1, 2, 4, 8, 16, 64});
  w.cfg.max_num_batched_tokens = pick({1, 4, 16, 64, 256, 4096});
  w.cfg.prefill_chunk_size = pick({1, 3, 8, 32, 512});
  w.cfg.admission_lookahead = pick({1, 1, 2, 8});  // default weighted
  w.cfg.starvation_wait_steps = pick({1, 4, 64});
  w.cfg.watermark_blocks = pick({-1, -1, 0, 1, 4});
  w.num_requests = static_cast<int>(1 + rng() % 24);
  return w;
}

// Deterministic fake model: the token it emits depends only on the step index,
// so a failure is reproducible from the seed alone.
struct FakeModel {
  int32_t next = 5000;
  std::vector<int32_t> run(const StepInput& step) {
    std::vector<int32_t> out;
    for (const auto& sl : step.slices) {
      REQUIRE(sl.seq != nullptr);
      REQUIRE(sl.len > 0);
      REQUIRE(sl.start == sl.seq->num_computed);
      REQUIRE(sl.start + sl.len <= sl.seq->num_tokens());
      if (sl.needs_logits) out.push_back(next++);
    }
    REQUIRE(static_cast<int>(out.size()) == step.num_logits);
    return out;
  }
};

}  // namespace

TEST_CASE("stress: random schedules preserve every invariant") {
  // 400 iterations keeps CI under a second. ENGINE_STRESS_ITERS raises it for a
  // deeper local hunt; a failure prints the seed and the full configuration, so
  // any iteration can be reproduced on its own.
  int kIterations = 400;
  if (const char* e = std::getenv("ENGINE_STRESS_ITERS")) {
    const int n = std::atoi(e);
    if (n > 0) kIterations = n;
  }
  for (int iter = 0; iter < kIterations; iter++) {
    std::mt19937_64 rng(0xC0FFEE ^ static_cast<uint64_t>(iter));
    const Workload w = random_workload(rng);
    INFO("iteration " << iter << ": blocks=" << w.num_blocks << "x" << w.block_size
         << " seqs=" << w.cfg.max_num_seqs << " budget=" << w.cfg.max_num_batched_tokens
         << " chunk=" << w.cfg.prefill_chunk_size << " lookahead=" << w.cfg.admission_lookahead
         << " watermark=" << w.cfg.watermark_blocks << " prefix=" << w.prefix_caching
         << " continuous=" << w.cfg.continuous << " requests=" << w.num_requests);

    KVCacheManager kv(w.num_blocks, w.block_size, w.prefix_caching);
    Scheduler sch(w.cfg, kv);
    const int capacity = kv.capacity_tokens();

    // A shared prefix on some requests exercises block sharing and refcounts.
    std::vector<int32_t> shared(std::min(capacity / 2, 1 + static_cast<int>(rng() % 40)));
    std::iota(shared.begin(), shared.end(), 900);

    std::vector<SequencePtr> seqs;
    for (int i = 0; i < w.num_requests; i++) {
      auto s = std::make_shared<Sequence>();
      s->id = static_cast<uint64_t>(i) + 1;
      const int max_tokens = 1 + static_cast<int>(rng() % 8);
      // Keep prompt + output inside the pool: a request that cannot fit is
      // rejected by Engine::validate in production, not by the scheduler.
      const int room = std::max(1, capacity - max_tokens);
      int prompt = 1 + static_cast<int>(rng() % static_cast<uint64_t>(room));
      if ((rng() & 1) && static_cast<int>(shared.size()) < prompt) {
        s->tokens = shared;
        while (static_cast<int>(s->tokens.size()) < prompt) s->tokens.push_back(static_cast<int32_t>(rng() & 0x3ff));
      } else {
        s->tokens.resize(prompt);
        for (auto& t : s->tokens) t = static_cast<int32_t>(rng() & 0x3ff);
      }
      s->prompt_len = s->num_tokens();
      s->params.max_tokens = max_tokens;
      s->params.ignore_eos = true;
      seqs.push_back(s);
      sch.add(s);
    }

    FakeModel model;
    std::set<uint64_t> aborted;
    long long steps = 0, idle = 0;
    // Generous but finite: every request needs at most prompt+max_tokens tokens,
    // and preemption can repeat that work a bounded number of times.
    const long long step_budget = 200LL * w.num_requests * (capacity + 16);

    while (sch.has_work()) {
      REQUIRE(steps < step_budget);
      // Occasionally abort something still in flight.
      if ((rng() % 32) == 0 && !sch.running().empty()) {
        const uint64_t victim = sch.running()[rng() % sch.running().size()]->id;
        sch.abort(victim);
        aborted.insert(victim);
        continue;
      }
      StepInput step = sch.schedule();

      // Block accounting must balance at every step, including mid-prefill.
      REQUIRE(kv.num_free_blocks() + kv.num_cached_blocks() + kv.num_used_blocks() == w.num_blocks);
      REQUIRE(kv.num_free_blocks() >= 0);
      REQUIRE(kv.num_used_blocks() >= 0);
      // Each resident sequence owns exactly enough blocks for what it holds.
      for (const auto& s : sch.running()) {
        const int need = KVCacheManager::blocks_needed(s->num_computed, w.block_size);
        REQUIRE(s->block_table.num_blocks() >= need);
        REQUIRE(s->num_computed <= s->num_tokens());
      }
      REQUIRE(static_cast<int>(sch.num_running()) <= w.cfg.max_num_seqs);
      REQUIRE(step.num_tokens() <= w.cfg.max_num_batched_tokens);

      if (step.empty()) {
        // No progress is acceptable only transiently; a livelock would spin here.
        REQUIRE(++idle < 10000);
        continue;
      }
      idle = 0;
      sch.on_step_done(step, model.run(step));
      steps++;
    }

    // Everything terminated, nothing generated past its cap, pool fully returned.
    for (const auto& s : seqs) {
      REQUIRE(s->is_finished());
      REQUIRE(s->block_table.empty());
      REQUIRE(s->num_generated() <= s->params.max_tokens);
      if (aborted.count(s->id) == 0 && s->finish != FinishReason::Abort)
        REQUIRE(s->num_generated() == s->params.max_tokens);
    }
    REQUIRE(kv.num_used_blocks() == 0);
    REQUIRE(kv.num_free_blocks() + kv.num_cached_blocks() == w.num_blocks);
    REQUIRE(sch.num_running() == 0);
    REQUIRE(sch.num_waiting() == 0);
  }
}
