#pragma once
// Request / sequence state shared by the scheduler, the model and the API.
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine/kv_cache.h"

namespace engine {

using Clock = std::chrono::steady_clock;

struct SamplingParams {
  int max_tokens = 128;
  float temperature = 1.0f;   // 0 => greedy
  float top_p = 1.0f;
  uint64_t seed = 0;          // 0 => nondeterministic
  std::vector<int32_t> stop_ids;
  bool ignore_eos = false;    // benchmark mode: always run to max_tokens
  bool greedy() const { return temperature <= 0.f; }
};

enum class SeqStatus { Waiting, Running, Finished };
enum class FinishReason { None, Stop, Length, Abort };

inline const char* finish_reason_str(FinishReason r) {
  switch (r) { case FinishReason::Stop: return "stop"; case FinishReason::Length: return "length"; case FinishReason::Abort: return "abort"; default: return ""; }
}

struct Sequence {
  uint64_t id = 0;
  std::vector<int32_t> tokens;   // prompt followed by generated tokens
  int prompt_len = 0;
  int num_computed = 0;          // tokens whose K/V are in the cache
  BlockTable block_table;
  SamplingParams params;
  SeqStatus status = SeqStatus::Waiting;
  FinishReason finish = FinishReason::None;
  int num_preemptions = 0;
  int wait_steps = 0;            // scheduling rounds spent waiting (starvation guard)

  Clock::time_point t_arrival, t_first_scheduled, t_first_token, t_finish;
  std::vector<Clock::time_point> token_times;

  // Streaming callback: called on the engine thread for every generated
  // token, and once more with finish != None. Must not block.
  std::function<void(const Sequence&, int32_t token, FinishReason)> on_token;

  int num_generated() const { return static_cast<int>(tokens.size()) - prompt_len; }
  int num_tokens() const { return static_cast<int>(tokens.size()); }
  // True when only the last token is left to compute, i.e. the next step is a
  // one-token decode. After a preemption the generated tokens are re-prefilled too.
  bool prefill_done() const { return num_tokens() - num_computed <= 1; }
  bool is_finished() const { return status == SeqStatus::Finished; }
};

using SequencePtr = std::shared_ptr<Sequence>;

// One forward pass. Each slice computes tokens[start, start+len) of a
// sequence; `needs_logits` marks slices that end at the sequence's last
// token, i.e. whose output we sample from.
struct StepSlice {
  Sequence* seq = nullptr;
  int start = 0;
  int len = 0;
  bool needs_logits = false;
  bool is_prefill = false;
};

struct StepInput {
  std::vector<StepSlice> slices;   // prefill slices first, then decode
  int num_prefill_tokens = 0;
  int num_decode_tokens = 0;
  int num_logits = 0;
  bool empty() const { return slices.empty(); }
  int num_tokens() const { return num_prefill_tokens + num_decode_tokens; }
};

}  // namespace engine
