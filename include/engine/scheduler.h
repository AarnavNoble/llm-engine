#pragma once
// Iteration-level scheduler (the Orca idea). Every step it builds one
// StepInput mixing decode tokens from running sequences with chunks of
// prompt from newly admitted ones, under a token budget. Finished sequences
// leave immediately and their blocks return to the pool.
//
// `SchedulerConfig::continuous = false` selects the static-batching baseline:
// a batch is admitted together and nothing new joins until every member is
// finished (finished members keep their KV blocks, exactly like padded
// HF generate). Same code path, one flag, so the comparison is honest.
#include <deque>
#include <vector>

#include "engine/kv_cache.h"
#include "engine/sequence.h"

namespace engine {

struct SchedulerConfig {
  bool continuous = true;
  int max_num_seqs = 32;             // running sequences per step
  int max_num_batched_tokens = 2048; // prefill + decode tokens per step
  int prefill_chunk_size = 512;      // chunked prefill; long prompts are split
  int starvation_wait_steps = 64;    // waiting this many rounds gets priority
};

struct SchedulerStats {
  uint64_t steps = 0, preemptions = 0, admitted = 0, finished = 0;
};

class Scheduler {
 public:
  Scheduler(SchedulerConfig cfg, KVCacheManager& kv);

  void add(SequencePtr seq);
  void abort(uint64_t seq_id);

  // Build the next step. Empty if nothing is runnable.
  StepInput schedule();
  // Apply sampled tokens: `sampled[i]` corresponds to the i-th slice with
  // needs_logits. Advances num_computed, appends tokens, retires finished
  // sequences (freeing their blocks) and fires callbacks.
  void on_step_done(const StepInput& step, const std::vector<int32_t>& sampled);

  bool has_work() const { return !waiting_.empty() || !running_.empty(); }
  size_t num_waiting() const { return waiting_.size(); }
  size_t num_running() const { return running_.size(); }
  const SchedulerStats& stats() const { return stats_; }
  // KV waste: allocated slots that hold no token, over allocated slots.
  double kv_waste_fraction() const;
  const std::vector<SequencePtr>& running() const { return running_; }

 private:
  bool admit_one(StepInput& step, int& budget);
  void preempt_youngest();
  void retire(SequencePtr seq, FinishReason r);
  void add_prefill_slice(StepInput& step, Sequence& s, int len);

  SchedulerConfig cfg_;
  KVCacheManager& kv_;
  std::deque<SequencePtr> waiting_;
  std::vector<SequencePtr> running_;
  SchedulerStats stats_;
};

}  // namespace engine
