#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace engine {

Engine::Engine(EngineConfig cfg)
    : cfg_(std::move(cfg)),
      tokenizer_(std::make_unique<Tokenizer>(cfg_.model_dir)),
      kv_(cfg_.num_blocks, cfg_.block_size, cfg_.prefix_caching),
      model_(
#ifdef ENGINE_CUDA
          cfg_.backend == "cuda" ? make_cuda_model(cfg_.model_dir, kv_) :
#endif
          make_cpu_model(cfg_.model_dir, kv_)),
      scheduler_(cfg_.sched, kv_),
      sampler_(cfg_.seed) {
  metrics_.kv_blocks_total = cfg_.num_blocks;
  ready_ = true;
}

Engine::~Engine() { stop(); }

void Engine::start() {
  if (running_) return;
  running_ = true; accepting_ = true;
  thread_ = std::thread([this] { run(); });
}

void Engine::stop() {
  accepting_ = false;
  running_ = false;
  inbox_cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Engine::drain() {
  accepting_ = false;
  draining_ = true;
  inbox_cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  running_ = false;
}

SequencePtr Engine::submit(std::vector<int32_t> prompt, SamplingParams params, TokenCallback cb) {
  if (!accepting_) return nullptr;
  auto s = std::make_shared<Sequence>();
  s->id = next_id_++;
  s->tokens = std::move(prompt);
  s->prompt_len = s->num_tokens();
  s->params = std::move(params);
  for (int32_t id : tokenizer_->stop_ids())
    if (std::find(s->params.stop_ids.begin(), s->params.stop_ids.end(), id) == s->params.stop_ids.end()) s->params.stop_ids.push_back(id);
  s->on_token = std::move(cb);
  metrics_.requests_total++;
  metrics_.prompt_tokens_total += static_cast<uint64_t>(s->prompt_len);
  {
    std::lock_guard<std::mutex> g(inbox_mu_);
    inbox_.push_back(s);
  }
  inbox_cv_.notify_one();
  return s;
}

void Engine::abort(uint64_t id) {
  { std::lock_guard<std::mutex> g(inbox_mu_); abort_inbox_.push_back(id); }
  inbox_cv_.notify_one();
}

void Engine::refresh_gauges() {
  metrics_.queue_depth = static_cast<int64_t>(scheduler_.num_waiting());
  metrics_.running_seqs = static_cast<int64_t>(scheduler_.num_running());
  metrics_.kv_blocks_free = kv_.num_free_blocks();
  metrics_.kv_blocks_used = kv_.num_used_blocks();
  metrics_.kv_blocks_cached = kv_.num_cached_blocks();
  metrics_.kv_waste_fraction = scheduler_.kv_waste_fraction();
  metrics_.steps_total = scheduler_.stats().steps;
  metrics_.preemptions_total = scheduler_.stats().preemptions;
  metrics_.prefix_cache_queries = kv_.stats().prefix_queries;
  metrics_.prefix_cache_hit_blocks = kv_.stats().prefix_hit_blocks;
  metrics_.prefix_cache_total_blocks = kv_.stats().prefix_total_blocks;
  metrics_.kv_evictions = kv_.stats().evictions;
}

void Engine::run() {
  std::vector<float> logits;
  const int vocab = model_->config().vocab_size;
  auto window_start = Clock::now(); uint64_t window_tokens = 0;

  while (running_) {
    // Pull new requests / aborts.
    {
      std::unique_lock<std::mutex> lk(inbox_mu_);
      if (!scheduler_.has_work() && inbox_.empty() && abort_inbox_.empty()) {
        if (draining_) break;
        inbox_cv_.wait(lk, [&] { return !running_ || draining_ || !inbox_.empty() || !abort_inbox_.empty(); });
        if (!running_) break;
      }
      for (auto& s : inbox_) {
        // Wrap the user callback so we can record timings + metrics centrally.
        auto user_cb = std::move(s->on_token);
        s->on_token = [this, user_cb = std::move(user_cb)](const Sequence& seq, int32_t tok, FinishReason r) {
          if (tok >= 0) {
            metrics_.generated_tokens_total++;
            if (seq.num_generated() == 1) metrics_.ttft.observe(std::chrono::duration<double>(seq.t_first_token - seq.t_arrival).count());
            else if (seq.token_times.size() >= 2) metrics_.inter_token.observe(std::chrono::duration<double>(seq.token_times.back() - seq.token_times[seq.token_times.size() - 2]).count());
          } else {
            if (r == FinishReason::Abort) metrics_.requests_aborted++; else metrics_.requests_finished++;
            metrics_.e2e.observe(std::chrono::duration<double>(Clock::now() - seq.t_arrival).count());
          }
          if (user_cb) user_cb(seq, tok, r);
        };
        scheduler_.add(s);
      }
      inbox_.clear();
      for (uint64_t id : abort_inbox_) scheduler_.abort(id);
      abort_inbox_.clear();
    }
    if (!scheduler_.has_work()) { refresh_gauges(); continue; }

    auto t0 = Clock::now();
    StepInput step = scheduler_.schedule();
    if (step.empty()) { refresh_gauges(); if (!scheduler_.has_work()) continue; std::this_thread::yield(); continue; }
    model_->forward(step, logits);
    auto sampled = sampler_.sample_step(step, logits, vocab);
    scheduler_.on_step_done(step, sampled);
    auto t1 = Clock::now();
    metrics_.step_time.observe(std::chrono::duration<double>(t1 - t0).count());
    metrics_.batch_tokens.observe(step.num_tokens());
    window_tokens += sampled.size();
    double win = std::chrono::duration<double>(t1 - window_start).count();
    if (win >= 1.0) { metrics_.tokens_per_second = window_tokens / win; window_tokens = 0; window_start = t1; }
    refresh_gauges();
  }
  // Abort whatever is left so waiting clients are released.
  for (auto& s : std::vector<SequencePtr>(scheduler_.running())) scheduler_.abort(s->id);
  while (scheduler_.num_waiting() > 0) { auto st = scheduler_.schedule(); for (auto& sl : st.slices) scheduler_.abort(sl.seq->id); if (st.empty()) break; }
  refresh_gauges();
}

}  // namespace engine
