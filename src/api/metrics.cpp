#include "engine/metrics.h"

#include <cmath>
#include <sstream>

namespace engine {

std::string Histogram::render(const std::string& name, const std::string& help) const {
  std::lock_guard<std::mutex> g(m_);
  std::ostringstream o;
  o << "# HELP " << name << " " << help << "\n# TYPE " << name << " histogram\n";
  uint64_t cum = 0;
  for (size_t i = 0; i < buckets_.size(); i++) { cum += counts_[i]; o << name << "_bucket{le=\"" << buckets_[i] << "\"} " << cum << "\n"; }
  cum += counts_.back();
  o << name << "_bucket{le=\"+Inf\"} " << cum << "\n" << name << "_sum " << sum_ << "\n" << name << "_count " << count_ << "\n";
  return o.str();
}

double Histogram::quantile(double p) const {
  std::lock_guard<std::mutex> g(m_);
  if (count_ == 0) return 0.0;
  double target = p * static_cast<double>(count_), cum = 0, lo = 0;
  for (size_t i = 0; i < buckets_.size(); i++) {
    double next = cum + counts_[i];
    if (next >= target) { double frac = counts_[i] ? (target - cum) / counts_[i] : 1.0; return lo + frac * (buckets_[i] - lo); }
    cum = next; lo = buckets_[i];
  }
  return buckets_.back();
}

std::string Metrics::render_prometheus() const {
  std::ostringstream o;
  auto counter = [&](const char* n, const char* h, uint64_t v) { o << "# HELP " << n << " " << h << "\n# TYPE " << n << " counter\n" << n << " " << v << "\n"; };
  auto gauge = [&](const char* n, const char* h, double v) { o << "# HELP " << n << " " << h << "\n# TYPE " << n << " gauge\n" << n << " " << v << "\n"; };
  counter("engine_requests_total", "Requests accepted", requests_total);
  counter("engine_requests_finished_total", "Requests finished", requests_finished);
  counter("engine_requests_aborted_total", "Requests aborted", requests_aborted);
  counter("engine_prompt_tokens_total", "Prompt tokens received", prompt_tokens_total);
  counter("engine_generated_tokens_total", "Tokens generated", generated_tokens_total);
  counter("engine_steps_total", "Forward passes executed", steps_total);
  counter("engine_preemptions_total", "Sequences preempted for memory", preemptions_total);
  counter("engine_prefix_cache_queries_total", "Prompts looked up in the prefix cache", prefix_cache_queries);
  counter("engine_prefix_cache_hit_blocks_total", "Prompt blocks served from the prefix cache", prefix_cache_hit_blocks);
  counter("engine_prefix_cache_total_blocks_total", "Full prompt blocks looked up", prefix_cache_total_blocks);
  counter("engine_kv_evictions_total", "Cached KV blocks evicted", kv_evictions);
  gauge("engine_queue_depth", "Requests waiting to be scheduled", static_cast<double>(queue_depth));
  gauge("engine_running_seqs", "Sequences in the current batch", static_cast<double>(running_seqs));
  gauge("engine_kv_blocks_total", "KV cache blocks", static_cast<double>(kv_blocks_total));
  gauge("engine_kv_blocks_free", "Free KV blocks", static_cast<double>(kv_blocks_free));
  gauge("engine_kv_blocks_used", "KV blocks owned by running sequences", static_cast<double>(kv_blocks_used));
  gauge("engine_kv_blocks_cached", "Unowned KV blocks retained for prefix reuse", static_cast<double>(kv_blocks_cached));
  gauge("engine_kv_waste_fraction", "Allocated-but-empty KV slots / allocated slots", kv_waste_fraction);
  gauge("engine_tokens_per_second", "Generated tokens per second (moving)", tokens_per_second);
  o << ttft.render("engine_ttft_seconds", "Time to first token");
  o << inter_token.render("engine_inter_token_seconds", "Inter-token latency");
  o << e2e.render("engine_request_seconds", "End-to-end request latency");
  o << step_time.render("engine_step_seconds", "Forward pass wall time");
  o << batch_tokens.render("engine_batch_tokens", "Tokens per forward pass");
  return o.str();
}

}  // namespace engine
