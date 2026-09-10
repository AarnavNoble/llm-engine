#pragma once
// Minimal Prometheus text-format metrics. Counters/gauges/histograms with a
// mutex; the engine thread updates once per step, the HTTP thread scrapes.
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace engine {

class Histogram {
 public:
  explicit Histogram(std::vector<double> buckets) : buckets_(std::move(buckets)), counts_(buckets_.size() + 1, 0) {}
  void observe(double v) {
    std::lock_guard<std::mutex> g(m_);
    size_t i = 0; while (i < buckets_.size() && v > buckets_[i]) i++;
    counts_[i]++; sum_ += v; count_++;
  }
  std::string render(const std::string& name, const std::string& help) const;
  // p in [0,1]; linear interpolation inside the bucket. Good enough for a dashboard.
  double quantile(double p) const;

 private:
  mutable std::mutex m_;
  std::vector<double> buckets_;
  std::vector<uint64_t> counts_;
  double sum_ = 0; uint64_t count_ = 0;
};

struct Metrics {
  std::atomic<uint64_t> requests_total{0}, requests_finished{0}, requests_aborted{0};
  std::atomic<uint64_t> prompt_tokens_total{0}, generated_tokens_total{0};
  std::atomic<uint64_t> steps_total{0}, preemptions_total{0};
  std::atomic<uint64_t> prefix_cache_queries{0}, prefix_cache_hit_blocks{0}, prefix_cache_total_blocks{0}, kv_evictions{0};
  // gauges, refreshed each step
  std::atomic<int64_t> queue_depth{0}, running_seqs{0};
  std::atomic<int64_t> kv_blocks_total{0}, kv_blocks_free{0}, kv_blocks_used{0}, kv_blocks_cached{0};
  std::atomic<double> kv_waste_fraction{0.0}, tokens_per_second{0.0};

  Histogram ttft{{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30}};
  Histogram inter_token{{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5}};
  Histogram e2e{{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120}};
  Histogram step_time{{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5}};
  Histogram batch_tokens{{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}};

  std::string render_prometheus() const;
};

}  // namespace engine
