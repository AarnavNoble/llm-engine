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
  cfg.watermark_blocks = 0;  // this test is about the preemption path itself
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

TEST_CASE("scheduler: a static batch fills across steps before it closes") {
  // Token budget of 64 cannot admit five 40-token prompts in one step, so a
  // fair static baseline must keep admitting until max_num_seqs is reached.
  KVCacheManager kv(256, 16, false);
  SchedulerConfig cfg;
  cfg.continuous = false;
  cfg.max_num_seqs = 5;
  cfg.max_num_batched_tokens = 64;
  Scheduler sch(cfg, kv);
  std::vector<SequencePtr> seqs;
  for (int i = 0; i < 6; i++) { seqs.push_back(make_seq(i + 1, 40, 3)); sch.add(seqs.back()); }
  FakeModel m;

  // One step's 64-token budget covers one whole prompt plus a partial chunk of
  // the next, so the batch cannot be full after a single step.
  auto st = sch.schedule();
  size_t after_first = sch.num_running();
  REQUIRE(after_first >= 1);
  REQUIRE(after_first < 5);
  sch.on_step_done(st, m.run(st));
  int steps = 1;
  while (sch.num_running() < 5 && steps < 20) { st = sch.schedule(); sch.on_step_done(st, m.run(st)); steps++; }
  REQUIRE(sch.num_running() == 5);       // batch kept filling on later steps
  REQUIRE(steps > 1);
  REQUIRE(sch.num_waiting() == 1);       // sixth request held back: batch is full

  // Now the batch is closed: the sixth request waits for every member to finish,
  // including the ones that finish early and keep their slot.
  while (sch.num_running() > 0) {
    st = sch.schedule();
    // Closing the batch stops new admissions; members still mid-prefill keep
    // getting their remaining chunks, so only the queue must stay untouched.
    REQUIRE(sch.num_running() <= 5);
    REQUIRE(sch.num_waiting() == 1);
    sch.on_step_done(st, m.run(st));
  }
  for (int i = 0; i < 5; i++) REQUIRE(seqs[i]->is_finished());
  REQUIRE(sch.num_waiting() == 1);
  st = sch.schedule();
  REQUIRE(st.slices.size() == 1);
  REQUIRE(st.slices[0].seq == seqs[5].get());
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  REQUIRE(seqs[5]->is_finished());
  REQUIRE(kv.num_used_blocks() == 0);
}

TEST_CASE("scheduler: a static batch closes when memory cannot take the next request") {
  // 8 blocks of 16 = 128 slots; each 40-token prompt needs 3 blocks, so only
  // two fit. The batch must close on memory rather than spin forever.
  KVCacheManager kv(8, 16, false);
  SchedulerConfig cfg;
  cfg.continuous = false;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 4096;
  Scheduler sch(cfg, kv);
  std::vector<SequencePtr> seqs;
  for (int i = 0; i < 4; i++) { seqs.push_back(make_seq(i + 1, 40, 2)); sch.add(seqs.back()); }
  FakeModel m;
  auto st = sch.schedule();
  REQUIRE(sch.num_running() == 2);
  REQUIRE(sch.num_waiting() == 2);
  sch.on_step_done(st, m.run(st));
  st = sch.schedule();
  REQUIRE(st.num_prefill_tokens == 0);   // closed: no third admission attempt
  sch.on_step_done(st, m.run(st));
  int steps = drive(sch, m);
  REQUIRE(steps > 0);
  for (auto& s : seqs) REQUIRE(s->is_finished());
  REQUIRE(sch.stats().preemptions == 0);
  REQUIRE(kv.num_used_blocks() == 0);
}

TEST_CASE("scheduler: preemption accounts for the work it throws away") {
  // Same setup as the preemption test: two 40-token prompts in 6 blocks of 16.
  KVCacheManager kv(6, 16, false);
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 512;
  cfg.watermark_blocks = 0;  // this test is about the preemption path itself
  Scheduler sch(cfg, kv);
  auto a = make_seq(1, 40, 20), b = make_seq(2, 40, 20);
  sch.add(a); sch.add(b);
  FakeModel m;
  auto st = sch.schedule(); sch.on_step_done(st, m.run(st));
  REQUIRE(sch.stats().recomputed_tokens == 0);

  int steps = 0;
  while (sch.stats().preemptions == 0 && steps < 50) { st = sch.schedule(); sch.on_step_done(st, m.run(st)); steps++; }
  REQUIRE(sch.stats().preemptions == 1);
  // The victim lost everything it had computed: its prompt plus what it generated.
  REQUIRE(sch.stats().recomputed_tokens == static_cast<uint64_t>(40 + b->num_generated() - 1));
  REQUIRE(b->num_computed == 0);

  drive(sch, m);
  REQUIRE(a->num_generated() == 20);
  REQUIRE(b->num_generated() == 20);
  // Recomputed work is bounded by what the victim had at preemption time.
  REQUIRE(sch.stats().recomputed_tokens < 120);
}

TEST_CASE("scheduler: the admission watermark prevents preempt thrash") {
  // 6 blocks of 16. Each 40-token prompt needs 3 blocks, so two fit exactly and
  // leave no headroom for either to cross its next block boundary.
  auto run = [](int watermark, int* preemptions, int* admitted_first_step) {
    static KVCacheManager* kvp = nullptr;
    KVCacheManager kv(6, 16, false);
    SchedulerConfig cfg; cfg.max_num_batched_tokens = 512; cfg.watermark_blocks = watermark;
    Scheduler sch(cfg, kv);
    auto a = make_seq(1, 40, 20), b = make_seq(2, 40, 20);
    sch.add(a); sch.add(b);
    FakeModel m;
    auto st = sch.schedule();
    *admitted_first_step = static_cast<int>(sch.num_running());
    sch.on_step_done(st, m.run(st));
    drive(sch, m);
    *preemptions = static_cast<int>(sch.stats().preemptions);
    REQUIRE(a->num_generated() == 20);
    REQUIRE(b->num_generated() == 20);     // both still complete either way
    REQUIRE(kv.num_used_blocks() == 0);
    (void)kvp;
  };

  int preempt_off = 0, preempt_on = 0, admitted_off = 0, admitted_on = 0;
  run(0, &preempt_off, &admitted_off);     // no watermark: admit both, then thrash
  run(-1, &preempt_on, &admitted_on);      // auto watermark: hold the second back

  REQUIRE(admitted_off == 2);
  REQUIRE(admitted_on == 1);
  REQUIRE(preempt_off > 0);
  REQUIRE(preempt_on == 0);                // the point: no work is thrown away
}

TEST_CASE("scheduler: the watermark never blocks the first sequence") {
  // With nothing resident the auto watermark is zero, so a prompt that fits at
  // all must be admitted; otherwise the engine would deadlock under pressure.
  KVCacheManager kv(4, 16, false);
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 512;  // watermark_blocks = -1 (auto)
  Scheduler sch(cfg, kv);
  auto s = make_seq(1, 60, 4);             // needs all 4 blocks
  sch.add(s);
  FakeModel m;
  auto st = sch.schedule();
  REQUIRE(sch.num_running() == 1);
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  REQUIRE(s->num_generated() == 4);
}

TEST_CASE("scheduler: a reusable prefix does not count against the watermark") {
  // The second request's prompt is entirely cached, so admitting it allocates
  // nothing and must not be held back by headroom accounting.
  KVCacheManager kv(8, 16, true);
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 512;
  Scheduler sch(cfg, kv);
  auto a = make_seq(7, 64, 1);
  sch.add(a);
  FakeModel m;
  drive(sch, m);
  REQUIRE(kv.num_cached_blocks() == 4);    // 4 full blocks published and retained

  auto keep = make_seq(1, 48, 8);          // occupies 3 blocks, so headroom is tight
  sch.add(keep);
  auto st = sch.schedule(); sch.on_step_done(st, m.run(st));
  REQUIRE(sch.num_running() == 1);
  auto b = make_seq(7, 64, 1);             // same tokens as a: fully cached
  sch.add(b);
  st = sch.schedule();
  REQUIRE(sch.num_running() == 2);         // admitted despite the watermark
  REQUIRE(st.slices[0].seq == b.get());
  // The whole prompt is cached, but the last token is always recomputed because
  // the step has to produce logits to sample from.
  REQUIRE(st.slices[0].start == 63);
  REQUIRE(st.slices[0].len == 1);
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  REQUIRE(b->is_finished());
}

TEST_CASE("scheduler: lookahead admits a smaller request when the head does not fit") {
  // 8 blocks of 16. The head needs 6 blocks and cannot be admitted alongside the
  // resident sequence; the 1-block request behind it can.
  KVCacheManager kv(8, 16, false);
  SchedulerConfig cfg;
  cfg.max_num_batched_tokens = 512;
  cfg.admission_lookahead = 4;
  Scheduler sch(cfg, kv);
  auto resident = make_seq(1, 32, 8);   // 2 blocks
  sch.add(resident);
  FakeModel m;
  auto st = sch.schedule(); sch.on_step_done(st, m.run(st));
  REQUIRE(sch.num_running() == 1);

  auto big = make_seq(2, 96, 4);        // 6 blocks: leaves no headroom
  auto small = make_seq(3, 8, 4);       // 1 block
  sch.add(big); sch.add(small);
  st = sch.schedule();
  REQUIRE(sch.num_running() == 2);
  bool small_admitted = false, big_admitted = false;
  for (auto& sl : st.slices) {
    if (sl.seq == small.get()) small_admitted = true;
    if (sl.seq == big.get()) big_admitted = true;
  }
  REQUIRE(small_admitted);
  REQUIRE_FALSE(big_admitted);
  REQUIRE(sch.num_waiting() == 1);
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  REQUIRE(big->is_finished());          // and the head still gets served eventually
  REQUIRE(small->is_finished());
  REQUIRE(kv.num_used_blocks() == 0);
}

TEST_CASE("scheduler: strict FCFS is the default and does not reorder") {
  KVCacheManager kv(8, 16, false);
  SchedulerConfig cfg; cfg.max_num_batched_tokens = 512;   // admission_lookahead = 1
  Scheduler sch(cfg, kv);
  auto resident = make_seq(1, 32, 8);
  sch.add(resident);
  FakeModel m;
  auto st = sch.schedule(); sch.on_step_done(st, m.run(st));
  auto big = make_seq(2, 96, 4), small = make_seq(3, 8, 4);
  sch.add(big); sch.add(small);
  st = sch.schedule();
  REQUIRE(sch.num_running() == 1);       // head blocks the queue; small waits behind it
  REQUIRE(sch.num_waiting() == 2);
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  for (auto& s : {resident, big, small}) REQUIRE(s->is_finished());
}

TEST_CASE("scheduler: a long-waiting request may not be overtaken") {
  KVCacheManager kv(8, 16, false);
  SchedulerConfig cfg;
  cfg.max_num_batched_tokens = 512;
  cfg.admission_lookahead = 8;
  cfg.starvation_wait_steps = 3;         // small bound so the test is short
  Scheduler sch(cfg, kv);
  auto resident = make_seq(1, 32, 40);   // stays resident for the whole test
  sch.add(resident);
  FakeModel m;
  auto st = sch.schedule(); sch.on_step_done(st, m.run(st));

  auto big = make_seq(2, 96, 4);         // never fits while resident holds blocks
  sch.add(big);
  for (int i = 0; i < 3; i++) { st = sch.schedule(); sch.on_step_done(st, m.run(st)); }
  REQUIRE(big->wait_steps >= 3);

  // A small request arriving now must not jump the starving head.
  auto small = make_seq(3, 8, 4);
  sch.add(small);
  st = sch.schedule();
  for (auto& sl : st.slices) REQUIRE(sl.seq != small.get());
  REQUIRE(sch.num_waiting() == 2);
  sch.on_step_done(st, m.run(st));
  drive(sch, m);
  for (auto& s : {resident, big, small}) REQUIRE(s->is_finished());
  REQUIRE(kv.num_used_blocks() == 0);
}
