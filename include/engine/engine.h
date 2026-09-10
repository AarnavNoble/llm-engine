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
};

class Engine {
 public:
  explicit Engine(EngineConfig cfg);
  ~Engine();

  void start();
  void stop();     // drains nothing; aborts in-flight requests
  void drain();    // stop accepting, finish in-flight, then stop (SIGTERM path)

  using TokenCallback = std::function<void(const Sequence&, int32_t token, FinishReason)>;
  SequencePtr submit(std::vector<int32_t> prompt, SamplingParams params, TokenCallback cb);
  void abort(uint64_t id);
  bool accepting() const { return accepting_; }

  const Tokenizer& tokenizer() const { return *tokenizer_; }
  const ModelConfig& config() const { return model_->config(); }
  const EngineConfig& engine_config() const { return cfg_; }
  Metrics& metrics() { return metrics_; }
  bool ready() const { return ready_; }

 private:
  void run();
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
};

}  // namespace engine
