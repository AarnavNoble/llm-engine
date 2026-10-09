#pragma once
// Iteration-level scheduler (the Orca idea). Every step it builds one
// StepInput mixing decode tokens from running sequences with chunks of
// prompt from newly admitted ones, under a token budget. Finished sequences
// leave immediately and their blocks return to the pool.
//
// `SchedulerConfig::continuous = false` selects the static-batching baseline:
// requests are admitted until the batch is full (max_num_seqs) or the next one
// does not fit in memory, after which the batch is closed and nothing new joins
// until every member has finished. Finished members keep their KV blocks, just
// like padded HF generate. Filling the batch may take several steps, because a
// single step's token budget rarely covers every prompt; closing it only once
// it is actually full keeps the baseline fair.
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
  // Admission is first-come-first-served, so a request at the head of the queue
  // that does not fit holds up everything behind it, including requests that
  // would fit. admission_lookahead > 1 lets the scheduler look that many
  // positions down the queue for something admissible. starvation_wait_steps
  // bounds the resulting unfairness: a request that has waited this many
  // scheduling rounds may not be overtaken, so it blocks the queue until it
  // fits. Lookahead 0 or 1 is strict FCFS and the bound never applies.
  int admission_lookahead = 1;
  int starvation_wait_steps = 64;
  // Admission watermark: free blocks that must remain after admitting a new
  // sequence. The engine admits optimistically and preempts reactively, so
  // without headroom a fresh admission steals the block a resident sequence
  // needs at its next block boundary, and the pair thrash. -1 keeps one block
  // per resident sequence, which is exactly what they can collectively need
  // before the next boundary; 0 disables the check.
  int watermark_blocks = -1;
};

struct SchedulerStats {
  uint64_t steps = 0, preemptions = 0, admitted = 0, finished = 0;
  // Tokens whose K/V were computed, then discarded by a preemption and have to
  // be computed again. This is the price of recompute-on-resume, and without it
  // the preemption counter says how often it happened but not what it cost.
  uint64_t recomputed_tokens = 0;
  uint64_t max_wait_steps = 0;   // worst scheduling rounds any request spent queued
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
  // Frees blocks for one resident sequence so another can grow, removing it from
  // the caller's candidate lists. False when nothing could be freed.
  bool evict_for_memory(std::vector<Sequence*>& mid_prefill, std::vector<Sequence*>& decode);
  bool admission_fits(const Sequence& s) const;
  void retire(SequencePtr seq, FinishReason r);
  void add_prefill_slice(StepInput& step, Sequence& s, int len);

  SchedulerConfig cfg_;
  KVCacheManager& kv_;
  std::deque<SequencePtr> waiting_;
  std::vector<SequencePtr> running_;
  bool static_batch_open_ = true;   // static mode: still accepting into this batch
  SchedulerStats stats_;
};

}  // namespace engine
