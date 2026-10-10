#pragma once
// The engine loop: one thread that repeatedly asks the scheduler for a step,
// runs the model, samples, and feeds tokens back. Requests arrive from any
// thread through submit(); callbacks fire on the engine thread.
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "engine/kv_cache.h"
#include "engine/metrics.h"
#include "engine/model.h"
#include "engine/sampler.h"
#include "engine/scheduler.h"
#include "engine/tokenizer.h"

namespace engine {

struct EngineConfig {
  std::string model_dir;
  std::string backend = "cpu";     // cpu | cuda
  int num_blocks = 512;            // 512 * 16 = 8192 token slots
  int block_size = 16;
  bool prefix_caching = true;
  SchedulerConfig sched;
  uint64_t seed = 42;

  // --- overload protection ---
  // Accepting work without limit does not make it finish any sooner: if arrivals
  // outpace service, the queue and every request's latency grow without bound.
  // Past this many queued requests the engine rejects new ones so clients can
  // retry or fail fast instead of all waiting. 0 disables the limit.
  int max_queue_depth = 256;

  // --- failure detection ---
  // If work is pending and no step completes within this long, the engine is
  // considered stalled and liveness fails, so an orchestrator restarts it. Must
  // comfortably exceed the slowest single forward pass. 0 disables the check.
  double stall_timeout_seconds = 120.0;
};

// Why a request was not accepted. The API maps these to status codes; they are
// separated because they mean different things to a client: retry later,
// go away, or fix the request.
enum class SubmitStatus { Accepted, Draining, Overloaded, Invalid };

struct SubmitResult {
  SubmitStatus status = SubmitStatus::Accepted;
  SequencePtr seq;              // only when accepted
  std::string reason;           // client-facing, empty when accepted
  int retry_after_seconds = 0;  // advisory, set when overloaded
  bool ok() const { return status == SubmitStatus::Accepted; }
};

// Result of the engine's self-check. Liveness and readiness ask different
// questions of it: a draining engine is unready but perfectly alive, while a
// stalled one is the opposite and must be restarted rather than waited for.
struct HealthReport {
  bool alive = true;      // the step loop is running and making progress
  bool ready = true;      // alive, loaded, and accepting work
  std::vector<std::string> problems;
  double seconds_since_progress = 0;
};

// The last fatal error from the step loop, empty when there has not been one.
// A backend failure (a CUDA error, say) kills the loop, and the process should
// report that rather than calling std::terminate from a thread nobody is
// watching.

class Engine {
 public:
  explicit Engine(EngineConfig cfg);
  ~Engine();

  void start();
  void stop();     // drains nothing; aborts in-flight requests
  void drain();    // stop accepting, finish in-flight, then stop (SIGTERM path)

  using TokenCallback = std::function<void(const Sequence&, int32_t token, FinishReason)>;
  HealthReport health() const;
  // Empty when the request can run; otherwise a client-facing reason why it
  // never could, so the API can reject it instead of accepting work that is
  // guaranteed to be aborted partway through.
  std::string validate(size_t prompt_tokens, const SamplingParams& params) const;
  SubmitResult submit(std::vector<int32_t> prompt, SamplingParams params, TokenCallback cb);
  void abort(uint64_t id);
  bool accepting() const { return accepting_; }

  const Tokenizer& tokenizer() const { return *tokenizer_; }
  const ModelConfig& config() const { return model_->config(); }
  const EngineConfig& engine_config() const { return cfg_; }
  Metrics& metrics() { return metrics_; }
  bool ready() const { return ready_; }
  // True when the binary was built with the CUDA backend compiled in.
  static bool cuda_available();
  // Empty unless the step loop died. Set before the loop exits, so a caller
  // that notices requests failing can say why.
  std::string fatal_error() const;

 private:
  void run();
  void run_loop();
  void refresh_gauges();

  EngineConfig cfg_;
  std::unique_ptr<Tokenizer> tokenizer_;
  KVCacheManager kv_;
  std::unique_ptr<Model> model_;
  Scheduler scheduler_;
  Sampler sampler_;
  Metrics metrics_;

  std::thread thread_;
  std::mutex inbox_mu_;
  std::condition_variable inbox_cv_;
  std::vector<SequencePtr> inbox_;
  std::vector<uint64_t> abort_inbox_;
  std::atomic<bool> running_{false}, accepting_{false}, ready_{false}, draining_{false};
  std::atomic<uint64_t> next_id_{1};
  // Progress watchdog. The engine thread stamps this after every completed step
  // and whenever it goes idle with nothing to do, so a stalled loop is
  // distinguishable from an idle one.
  std::atomic<int64_t> last_progress_us_{0};
  std::atomic<bool> invariants_ok_{true};
  std::atomic<int64_t> queued_{0};
  void mark_progress();
  std::string fatal_error_;          // guarded by inbox_mu_
  mutable std::mutex fatal_mu_;
};

}  // namespace engine
