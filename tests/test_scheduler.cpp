#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <map>
#include "engine/scheduler.h"

using namespace engine;

namespace {
// Deterministic stand-in for the model: emits token = 1000 + step counter,
// and lets a test decide which sequences stop when.
struct FakeModel {
  int32_t next = 1000;
  std::vector<int32_t> run(const StepInput& step) {
    std::vector<int32_t> out;
    for (const auto& sl : step.slices) {
      REQUIRE(sl.len > 0);
      REQUIRE(sl.start == sl.seq->num_computed);
      REQUIRE(sl.start + sl.len <= sl.seq->num_tokens());
      // every token in the slice must have a physical slot
      REQUIRE(sl.seq->block_table.num_blocks() * 16 >= sl.start + sl.len);
      if (sl.needs_logits) out.push_back(next++);
    }
    REQUIRE(static_cast<int>(out.size()) == step.num_logits);
    return out;
  }
};

SequencePtr make_seq(uint64_t id, int prompt_len, int max_tokens, std::vector<int32_t> stop = {}) {
  auto s = std::make_shared<Sequence>();
  s->id = id; s->tokens.resize(prompt_len); for (int i = 0; i < prompt_len; i++) s->tokens[i] = static_cast<int32_t>(id * 100000 + i);
  s->prompt_len = prompt_len; s->params.max_tokens = max_tokens; s->params.stop_ids = std::move(stop);
  return s;
}

int drive(Scheduler& sch, FakeModel& m, int max_steps = 10000) {
  int steps = 0;
  while (sch.has_work() && steps < max_steps) {
    auto step = sch.schedule();
    if (step.empty()) break;
    sch.on_step_done(step, m.run(step));
    steps++;
  }
  return steps;
}
}  // namespace

TEST_CASE("scheduler: single request prefills then decodes to max_tokens") {
  KVCacheManager kv(64, 16, false);
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 64;
  Scheduler sch(cfg, kv);
  auto s = make_seq(1, 20, 5);
  sch.add(s);
  FakeModel m;
  auto st = sch.schedule();
  REQUIRE(st.slices.size() == 1);
  REQUIRE(st.slices[0].is_prefill);
  REQUIRE(st.slices[0].len == 20);
  REQUIRE(st.slices[0].needs_logits);
  sch.on_step_done(st, m.run(st));
  REQUIRE(s->num_generated() == 1);
  REQUIRE(s->tokens.back() == 1000);
  st = sch.schedule();
  REQUIRE(st.num_decode_tokens == 1);
  REQUIRE(st.slices[0].start == 20);
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  REQUIRE(s->is_finished());
  REQUIRE(s->finish == FinishReason::Length);
  REQUIRE(s->num_generated() == 5);
  REQUIRE(kv.num_used_blocks() == 0);
}

TEST_CASE("scheduler: chunked prefill splits a long prompt and only the last chunk samples") {
  KVCacheManager kv(64, 16, false);
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 32; cfg.prefill_chunk_size = 32;
  Scheduler sch(cfg, kv);
  auto s = make_seq(1, 100, 1);
  sch.add(s);
  FakeModel m;
  std::vector<int> lens;
  while (!s->is_finished()) { auto st = sch.schedule(); REQUIRE(st.slices.size() == 1); lens.push_back(st.slices[0].len); sch.on_step_done(st, m.run(st)); }
  REQUIRE(lens == std::vector<int>{32, 32, 32, 4});
  REQUIRE(s->num_generated() == 1);
}

TEST_CASE("scheduler: stop token ends the sequence and frees blocks immediately") {
  KVCacheManager kv(64, 16, false);
  Scheduler sch(SchedulerConfig{}, kv);
  auto s = make_seq(1, 8, 100, {1002});
  int cb_tokens = 0; FinishReason cb_reason = FinishReason::None;
  s->on_token = [&](const Sequence&, int32_t t, FinishReason r) { if (t >= 0) cb_tokens++; else cb_reason = r; };
  sch.add(s);
  FakeModel m;
  drive(sch, m);
  REQUIRE(s->tokens.back() == 1002);
  REQUIRE(s->num_generated() == 3);
  REQUIRE(s->finish == FinishReason::Stop);
  REQUIRE(cb_tokens == 2);                  // the stop token itself is not streamed
  REQUIRE(cb_reason == FinishReason::Stop);
  REQUIRE(kv.num_used_blocks() == 0);
}

TEST_CASE("scheduler: continuous batching admits new requests as others finish") {
  KVCacheManager kv(256, 16, false);
  SchedulerConfig cfg; cfg.max_num_seqs = 2; cfg.max_num_batched_tokens = 256;
  Scheduler sch(cfg, kv);
  auto a = make_seq(1, 10, 2), b = make_seq(2, 10, 10), c = make_seq(3, 10, 2);
  sch.add(a); sch.add(b); sch.add(c);
  FakeModel m;
  auto st = sch.schedule();               // a, b admitted (prefill)
  REQUIRE(st.slices.size() == 2);
  REQUIRE(sch.num_waiting() == 1);
  sch.on_step_done(st, m.run(st));
  st = sch.schedule();                    // a, b decode; c still waiting (seq cap)
  REQUIRE(st.num_decode_tokens == 2); REQUIRE(st.num_prefill_tokens == 0);
  sch.on_step_done(st, m.run(st));        // a hits max_tokens=2 -> leaves now
  REQUIRE(a->is_finished());
  st = sch.schedule();                    // b decode + c prefill in the SAME step
  REQUIRE(st.num_decode_tokens == 1);
  REQUIRE(st.num_prefill_tokens == 10);
  REQUIRE(st.slices[0].is_prefill);       // prefill slices come first
  REQUIRE(st.slices[0].seq == c.get());
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  REQUIRE(b->is_finished()); REQUIRE(c->is_finished());
  REQUIRE(sch.stats().preemptions == 0);
}

TEST_CASE("scheduler: static batching waits for the slowest member before admitting") {
  KVCacheManager kv(256, 16, false);
  SchedulerConfig cfg; cfg.continuous = false; cfg.max_num_seqs = 2; cfg.max_num_batched_tokens = 256;
  Scheduler sch(cfg, kv);
  auto a = make_seq(1, 10, 2), b = make_seq(2, 10, 6), c = make_seq(3, 10, 2);
  sch.add(a); sch.add(b); sch.add(c);
  FakeModel m;
  auto st = sch.schedule(); sch.on_step_done(st, m.run(st));  // prefill a,b
  st = sch.schedule(); sch.on_step_done(st, m.run(st));       // decode -> a done
  REQUIRE(a->is_finished());
  REQUIRE(a->block_table.num_blocks() > 0);                   // a keeps its blocks (padded slot)
  int steps_with_c_waiting = 0;
  while (!b->is_finished()) { st = sch.schedule(); REQUIRE(st.num_prefill_tokens == 0); sch.on_step_done(st, m.run(st)); steps_with_c_waiting++; }
  REQUIRE(steps_with_c_waiting == 4);
  REQUIRE(kv.num_used_blocks() == 0);                          // batch drained, all blocks freed
  st = sch.schedule();
  REQUIRE(st.num_prefill_tokens == 10);                        // only now c is admitted
  REQUIRE(st.slices[0].seq == c.get());
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  REQUIRE(c->is_finished());
}

TEST_CASE("scheduler: preemption under memory pressure recomputes and resumes at the right position") {
  // 6 blocks of 16 = 96 slots. Two prompts of 40 tokens each need 3 blocks each -> pool full.
  KVCacheManager kv(6, 16, false);
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 512;
  Scheduler sch(cfg, kv);
  auto a = make_seq(1, 40, 20), b = make_seq(2, 40, 20);
  sch.add(a); sch.add(b);
  FakeModel m;
  auto st = sch.schedule(); REQUIRE(st.slices.size() == 2); sch.on_step_done(st, m.run(st));
  // Both at 41 tokens in 3 blocks (48 slots). Decode until someone needs a 4th block: at 48 tokens.
  int steps = 0;
  while (sch.stats().preemptions == 0 && steps < 50) { st = sch.schedule(); sch.on_step_done(st, m.run(st)); steps++; }
  REQUIRE(sch.stats().preemptions == 1);
  REQUIRE(b->num_preemptions == 1);      // youngest (admitted last) is the victim
  REQUIRE(b->status == SeqStatus::Waiting);
  REQUIRE(b->num_computed == 0);
  int b_generated_at_preempt = b->num_generated();
  REQUIRE(b_generated_at_preempt > 0);
  // a finishes, frees memory, b is re-admitted and re-prefills prompt + generated tokens.
  bool saw_recompute = false;
  while (sch.has_work()) {
    st = sch.schedule();
    REQUIRE_FALSE(st.empty());
    for (auto& sl : st.slices) if (sl.seq == b.get() && sl.is_prefill) {
      REQUIRE(a->is_finished());                          // b only comes back once memory frees up
      saw_recompute = true; REQUIRE(sl.len == 40 + b_generated_at_preempt);  // prompt + generated so far, all recomputed
    }
    sch.on_step_done(st, m.run(st));
  }
  REQUIRE(saw_recompute);
  REQUIRE(b->is_finished());
  REQUIRE(b->num_generated() == 20);
  REQUIRE(a->num_generated() == 20);
  REQUIRE(kv.num_used_blocks() == 0);
}

TEST_CASE("scheduler: abort frees blocks and fires the callback") {
  KVCacheManager kv(64, 16, false);
  Scheduler sch(SchedulerConfig{}, kv);
  auto s = make_seq(1, 8, 100);
  FinishReason r = FinishReason::None;
  s->on_token = [&](const Sequence&, int32_t t, FinishReason fr) { if (t < 0) r = fr; };
  sch.add(s);
  FakeModel m;
  auto st = sch.schedule(); sch.on_step_done(st, m.run(st));
  sch.abort(1);
  REQUIRE(r == FinishReason::Abort);
  REQUIRE(kv.num_used_blocks() == 0);
  REQUIRE_FALSE(sch.has_work());
}

TEST_CASE("scheduler: prefix cache hit skips prefill of shared blocks") {
  KVCacheManager kv(64, 16, true);
  Scheduler sch(SchedulerConfig{}, kv);
  auto a = make_seq(7, 40, 1);
  sch.add(a);
  FakeModel m; drive(sch, m);
  auto b = make_seq(7, 45, 1);          // same id -> same first 40 prompt tokens
  sch.add(b);
  auto st = sch.schedule();
  REQUIRE(st.slices[0].start == 32);    // 2 full blocks reused
  REQUIRE(st.slices[0].len == 13);
  sch.on_step_done(st, m.run(st));
  REQUIRE(b->tokens.size() == 46);
}

TEST_CASE("scheduler: kv waste fraction reflects partial blocks") {
  KVCacheManager kv(64, 16, false);
  Scheduler sch(SchedulerConfig{}, kv);
  auto s = make_seq(1, 17, 100);        // 2 blocks = 32 slots, 17 used
  sch.add(s);
  auto st = sch.schedule();
  REQUIRE(sch.kv_waste_fraction() == Catch::Approx(15.0 / 32.0));
}
