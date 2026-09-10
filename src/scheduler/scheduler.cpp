#include "engine/scheduler.h"

#include <algorithm>
#include <cassert>

namespace engine {

Scheduler::Scheduler(SchedulerConfig cfg, KVCacheManager& kv) : cfg_(cfg), kv_(kv) {}

void Scheduler::add(SequencePtr seq) {
  seq->status = SeqStatus::Waiting;
  seq->t_arrival = Clock::now();
  waiting_.push_back(std::move(seq));
}

void Scheduler::abort(uint64_t id) {
  for (auto it = waiting_.begin(); it != waiting_.end(); ++it)
    if ((*it)->id == id) { auto s = *it; waiting_.erase(it); retire(s, FinishReason::Abort); return; }
  for (auto it = running_.begin(); it != running_.end(); ++it)
    if ((*it)->id == id) { auto s = *it; running_.erase(it); retire(s, FinishReason::Abort); return; }
}

void Scheduler::retire(SequencePtr s, FinishReason r) {
  s->status = SeqStatus::Finished;
  s->finish = r;
  s->t_finish = Clock::now();
  kv_.free(s->block_table);
  stats_.finished++;
  if (s->on_token) s->on_token(*s, -1, r);
}

void Scheduler::add_prefill_slice(StepInput& step, Sequence& s, int len) {
  StepSlice sl;
  sl.seq = &s; sl.start = s.num_computed; sl.len = len; sl.is_prefill = true;
  sl.needs_logits = (s.num_computed + len == s.num_tokens());
  step.slices.push_back(sl);
  step.num_prefill_tokens += len;
  if (sl.needs_logits) step.num_logits++;
}

// Try to admit the head of the waiting queue. Returns false if it cannot
// be admitted right now (budget, seq limit, or memory).
bool Scheduler::admit_one(StepInput& step, int& budget) {
  if (waiting_.empty() || budget <= 0) return false;
  if (static_cast<int>(running_.size()) >= cfg_.max_num_seqs) return false;
  SequencePtr s = waiting_.front();

  if (s->block_table.empty()) {
    auto cached = kv_.allocate_prompt(s->block_table, s->tokens);
    if (!cached) return false;
    s->num_computed = std::min(*cached, s->num_tokens() - 1);  // always compute >= 1 token so we get logits
    if (s->t_first_scheduled == Clock::time_point{}) s->t_first_scheduled = Clock::now();
  }
  int remaining = s->num_tokens() - s->num_computed;
  int len = std::min({remaining, budget, cfg_.prefill_chunk_size});
  if (len <= 0) return false;
  add_prefill_slice(step, *s, len);
  budget -= len;
  waiting_.pop_front();
  running_.push_back(s);
  s->status = SeqStatus::Running;
  stats_.admitted++;
  return true;
}

void Scheduler::preempt_youngest() {
  // Recompute-on-resume: drop the most recently admitted sequence's cache and
  // put it back at the head of the queue. Its generated tokens are kept as
  // part of the "prompt" it will re-prefill.
  assert(!running_.empty());
  SequencePtr s = running_.back();
  running_.pop_back();
  kv_.free(s->block_table);
  s->num_computed = 0;
  s->num_preemptions++;
  s->status = SeqStatus::Waiting;
  waiting_.push_front(s);
  stats_.preemptions++;
}

StepInput Scheduler::schedule() {
  StepInput step;
  int budget = cfg_.max_num_batched_tokens;
  stats_.steps++;

  // Static batching: only refill when the batch has fully drained.
  const bool may_admit = cfg_.continuous || running_.empty();

  // 1. Running sequences. A sequence still mid-prefill (chunked) gets its next
  //    chunk now; sequences past their prompt are collected for decode.
  std::vector<Sequence*> decode;
  for (auto& s : running_) {
    if (s->is_finished()) continue;  // static mode keeps finished members around
    if (!s->prefill_done()) {
      int len = std::min({s->num_tokens() - s->num_computed, budget, cfg_.prefill_chunk_size});
      if (len > 0) { add_prefill_slice(step, *s, len); budget -= len; }
    } else {
      decode.push_back(s.get());
    }
  }

  // 2. Every decode token needs a physical slot for its K/V. When the pool is
  //    exhausted, preempt the youngest running sequence (recompute-on-resume)
  //    and retry; the victim may be the sequence we were trying to place.
  for (size_t i = 0; i < decode.size();) {
    Sequence* s = decode[i];
    if (kv_.ensure_slot(s->block_table, s->num_tokens())) { i++; continue; }
    Sequence* victim = running_.back().get();
    preempt_youngest();
    decode.erase(std::remove(decode.begin(), decode.end(), victim), decode.end());
    // if victim == s, decode[i] is now the next sequence; either way re-examine index i
  }
  if (static_cast<int>(decode.size()) > budget) decode.resize(static_cast<size_t>(budget));
  budget -= static_cast<int>(decode.size());

  // 3. Admit waiting sequences under the remaining budget (FCFS). Their prefill
  //    slices go before the decode slices so the model sees [prefill..., decode...].
  if (may_admit) {
    for (auto& w : waiting_) w->wait_steps++;
    while (admit_one(step, budget)) {}
    if (!waiting_.empty() && running_.empty() && decode.empty()) {
      // Nothing is running and the head request still cannot be admitted:
      // its prompt is larger than the whole pool. Reject it rather than hang.
      auto s = waiting_.front(); waiting_.pop_front(); retire(s, FinishReason::Abort);
    }
  }

  for (Sequence* s : decode) {
    StepSlice sl; sl.seq = s; sl.start = s->num_computed; sl.len = 1; sl.needs_logits = true;
    step.slices.push_back(sl); step.num_decode_tokens++; step.num_logits++;
  }
  return step;
}

void Scheduler::on_step_done(const StepInput& step, const std::vector<int32_t>& sampled) {
  size_t k = 0;
  std::vector<SequencePtr> done;
  auto now = Clock::now();
  for (const auto& sl : step.slices) {
    Sequence& s = *sl.seq;
    s.num_computed += sl.len;
    kv_.commit_computed(s.block_table, s.tokens, s.num_computed);
    if (!sl.needs_logits) continue;
    assert(k < sampled.size());
    int32_t tok = sampled[k++];
    if (s.is_finished()) continue;  // static mode: padded member, discard
    s.tokens.push_back(tok);
    s.token_times.push_back(now);
    if (s.t_first_token == Clock::time_point{}) s.t_first_token = now;
    bool stop = !s.params.ignore_eos &&
                std::find(s.params.stop_ids.begin(), s.params.stop_ids.end(), tok) != s.params.stop_ids.end();
    FinishReason r = stop ? FinishReason::Stop
                   : s.num_generated() >= s.params.max_tokens ? FinishReason::Length : FinishReason::None;
    if (s.on_token && !stop) s.on_token(s, tok, FinishReason::None);  // stop tokens are not emitted
    if (r == FinishReason::None) continue;
    s.finish = r;
    auto it = std::find_if(running_.begin(), running_.end(), [&](const SequencePtr& p) { return p.get() == &s; });
    if (cfg_.continuous) {
      done.push_back(*it); running_.erase(it);
    } else {
      // Static batching: mark finished but keep the slot and the blocks until the batch drains.
      s.status = SeqStatus::Finished; s.t_finish = now;
      if (s.on_token) s.on_token(s, -1, r);
    }
  }
  for (auto& s : done) retire(s, s->finish);
  if (!cfg_.continuous && !running_.empty() &&
      std::all_of(running_.begin(), running_.end(), [](const SequencePtr& p) { return p->is_finished(); })) {
    for (auto& s : running_) { kv_.free(s->block_table); stats_.finished++; }
    running_.clear();
  }
}

double Scheduler::kv_waste_fraction() const {
  int64_t allocated = 0, used = 0;
  for (const auto& s : running_) {
    allocated += static_cast<int64_t>(s->block_table.num_blocks()) * kv_.block_size();
    used += s->num_tokens();
  }
  return allocated == 0 ? 0.0 : 1.0 - static_cast<double>(used) / static_cast<double>(allocated);
}

}  // namespace engine
