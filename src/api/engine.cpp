#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace engine {

namespace {

// Selecting a backend that was not compiled in must be an error, never a quiet
// fallback. A benchmark run labelled cuda that silently used the CPU backend
// would corrupt every number it produced, and nothing downstream could tell.
std::unique_ptr<Model> build_model(const std::string& backend, const std::string& model_dir,
                                   const KVCacheManager& kv) {
  if (backend == "cpu") return make_cpu_model(model_dir, kv);
  if (backend == "cuda") {
#ifdef ENGINE_CUDA
    return make_cuda_model(model_dir, kv);
#else
    throw std::runtime_error(
        "--backend cuda requested but this binary has no CUDA backend. "
        "Rebuild with -DENGINE_CUDA=ON (see docs/gpu-setup.md), or use --backend cpu. "
        "Refusing to fall back silently, because results labelled cuda that ran on the CPU "
        "would be indistinguishable from real ones.");
#endif
  }
  throw std::runtime_error("unknown --backend '" + backend + "'; expected cpu or cuda");
}

}  // namespace

Engine::Engine(EngineConfig cfg)
    : cfg_(std::move(cfg)),
      tokenizer_(std::make_unique<Tokenizer>(cfg_.model_dir)),
      kv_(cfg_.num_blocks, cfg_.block_size, cfg_.prefix_caching),
      model_(build_model(cfg_.backend, cfg_.model_dir, kv_)),
      scheduler_(cfg_.sched, kv_),
      sampler_(cfg_.seed) {
  metrics_.kv_blocks_total = cfg_.num_blocks;
  ready_ = true;
}

// Which backend this binary can actually run, for startup banners and scripts.
bool Engine::cuda_available() {
#ifdef ENGINE_CUDA
  return true;
#else
  return false;
#endif
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

std::string Engine::validate(size_t prompt_tokens, const SamplingParams& params) const {
  if (prompt_tokens == 0) return "prompt is empty";
  if (params.max_tokens <= 0) return "max_tokens must be positive";
  // A sequence needs room for its prompt and everything it will generate. If
  // that exceeds the whole KV pool it can never finish, even alone: it would be
  // admitted, generate a token, hit the end of the pool and be aborted.
  const size_t need = prompt_tokens + static_cast<size_t>(params.max_tokens);
  const size_t capacity = static_cast<size_t>(kv_.capacity_tokens());
  if (need > capacity) {
    return "prompt (" + std::to_string(prompt_tokens) + " tokens) plus max_tokens (" +
           std::to_string(params.max_tokens) + ") exceeds the KV cache capacity of " +
           std::to_string(capacity) + " tokens; raise --num-blocks or shorten the request";
  }
  const size_t model_max = static_cast<size_t>(model_->config().max_position_embeddings);
  if (need > model_max) {
    return "prompt plus max_tokens (" + std::to_string(need) + ") exceeds the model's context length of " +
           std::to_string(model_max) + " tokens";
  }
  return "";
}

void Engine::mark_progress() {
  last_progress_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
      Clock::now().time_since_epoch()).count();
}

HealthReport Engine::health() const {
  HealthReport h;
  const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
      Clock::now().time_since_epoch()).count();
  const int64_t last = last_progress_us_.load();
  h.seconds_since_progress = last == 0 ? 0.0 : (now_us - last) / 1e6;

  if (!ready_) { h.alive = false; h.problems.push_back("model not loaded"); }
  if (!running_) {
    // Not running is fatal while serving, but is also the state before start()
    // and after a completed drain, so it is reported rather than interpreted.
    h.alive = false;
    h.problems.push_back("engine loop not running");
  }
  // A stall means work is pending and the loop is not advancing it. An engine
  // with an empty queue is healthy however long it has been quiet, and that
  // distinction has to come from a counter both threads update rather than from
  // elapsed time: the loop sleeps on a condition variable when idle, so "no
  // recent step" is the normal state of a healthy, unused server. Reporting
  // that as dead would crash-loop an idle pod.
  if (cfg_.stall_timeout_seconds > 0 && last != 0 && queued_.load() > 0 &&
      h.seconds_since_progress > cfg_.stall_timeout_seconds) {
    h.alive = false;
    h.problems.push_back("no step completed in " + std::to_string(h.seconds_since_progress) + "s");
  }
  if (!invariants_ok_) {
    // The block pool accounting does not add up, which means the allocator has
    // leaked or double-counted. Results past this point are not trustworthy.
    h.alive = false;
    h.problems.push_back("KV block accounting invariant violated");
  }
  h.ready = h.alive && accepting_;
  if (h.alive && !accepting_) h.problems.push_back("draining");
  return h;
}

SubmitResult Engine::submit(std::vector<int32_t> prompt, SamplingParams params, TokenCallback cb) {
  SubmitResult out;
  if (!accepting_) {
    out.status = SubmitStatus::Draining;
    out.reason = "server is draining";
    return out;
  }
  // Load shedding. Rejecting early is kinder than accepting work that cannot be
  // served in time: the client learns immediately and can retry or shed its own
  // load, instead of every request in flight getting slower.
  if (cfg_.max_queue_depth > 0 && queued_.load() >= cfg_.max_queue_depth) {
    out.status = SubmitStatus::Overloaded;
    out.reason = "too many queued requests (" + std::to_string(queued_.load()) + " of " +
                 std::to_string(cfg_.max_queue_depth) + "); retry shortly";
    out.retry_after_seconds = 1;
    metrics_.requests_rejected_overload++;
    return out;
  }
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
  queued_++;
  inbox_cv_.notify_one();
  out.seq = s;
  return out;
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
  metrics_.requests_queue_timeout_total = scheduler_.stats().timed_out;
  // Cheap enough to check every step, and the one invariant whose violation
  // invalidates everything downstream.
  const bool balanced = kv_.num_free_blocks() + kv_.num_cached_blocks() + kv_.num_used_blocks() ==
                        kv_.num_total_blocks();
  if (!balanced) invariants_ok_ = false;
  metrics_.kv_accounting_ok = balanced ? 1 : 0;
  metrics_.recomputed_tokens_total = scheduler_.stats().recomputed_tokens;
  metrics_.prefix_cache_queries = kv_.stats().prefix_queries;
  metrics_.prefix_cache_hit_blocks = kv_.stats().prefix_hit_blocks;
  metrics_.prefix_cache_total_blocks = kv_.stats().prefix_total_blocks;
  metrics_.kv_evictions = kv_.stats().evictions;
}

void Engine::run() {
  mark_progress();
  std::vector<float> logits;
  const int vocab = model_->config().vocab_size;
  auto window_start = Clock::now(); uint64_t window_tokens = 0;

  while (running_) {
    // Pull new requests / aborts.
    {
      std::unique_lock<std::mutex> lk(inbox_mu_);
      if (!scheduler_.has_work() && inbox_.empty() && abort_inbox_.empty()) {
        if (draining_) break;
        // Bounded wait rather than an indefinite one, so an idle loop keeps
        // stamping progress and seconds_since_progress stays a real measure of
        // whether the loop is turning.
        mark_progress();
        inbox_cv_.wait_for(lk, std::chrono::milliseconds(500), [&] {
          return !running_ || draining_ || !inbox_.empty() || !abort_inbox_.empty();
        });
        if (!running_) break;
        if (inbox_.empty() && abort_inbox_.empty() && !scheduler_.has_work()) {
          mark_progress();
          lk.unlock();
          refresh_gauges();
          continue;
        }
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
            if (r == FinishReason::Timeout) metrics_.requests_timed_out++;
            else if (r == FinishReason::Abort) metrics_.requests_aborted++;
            else metrics_.requests_finished++;
            metrics_.e2e.observe(std::chrono::duration<double>(Clock::now() - seq.t_arrival).count());
            queued_--;
          }
          if (user_cb) user_cb(seq, tok, r);
        };
        scheduler_.add(s);
      }
      inbox_.clear();
      for (uint64_t id : abort_inbox_) scheduler_.abort(id);
      abort_inbox_.clear();
    }
    // An idle loop is healthy, so it counts as progress; only a loop with work
    // that fails to advance is a stall.
    if (!scheduler_.has_work()) { mark_progress(); refresh_gauges(); continue; }

    auto t0 = Clock::now();
    StepInput step = scheduler_.schedule();
    if (step.empty()) { refresh_gauges(); if (!scheduler_.has_work()) { mark_progress(); continue; } std::this_thread::yield(); continue; }
    model_->forward(step, logits);
    auto sampled = sampler_.sample_step(step, logits, vocab);
    scheduler_.on_step_done(step, sampled);
    auto t1 = Clock::now();
    mark_progress();
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
