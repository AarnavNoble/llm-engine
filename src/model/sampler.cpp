#include "engine/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace engine {

int32_t Sampler::argmax(const float* logits, int vocab) {
  int32_t best = 0;
  for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

int32_t Sampler::sample(const float* logits, int vocab, const SamplingParams& p) {
  if (p.greedy()) return argmax(logits, vocab);
  std::mt19937_64* rng = &rng_;
  std::mt19937_64 seeded;
  if (p.seed != 0) { seeded.seed(p.seed); rng = &seeded; }

  probs_.resize(vocab);
  float mx = *std::max_element(logits, logits + vocab);
  double sum = 0;
  for (int i = 0; i < vocab; i++) { probs_[i] = std::exp((logits[i] - mx) / p.temperature); sum += probs_[i]; }
  for (int i = 0; i < vocab; i++) probs_[i] = static_cast<float>(probs_[i] / sum);

  if (p.top_p < 1.0f) {
    idx_.resize(vocab); std::iota(idx_.begin(), idx_.end(), 0);
    // partial sort is enough: nucleus is small for peaked distributions
    std::sort(idx_.begin(), idx_.end(), [&](int a, int b) { return probs_[a] > probs_[b]; });
    double cum = 0; int keep = 0;
    for (; keep < vocab; keep++) { cum += probs_[idx_[keep]]; if (cum >= p.top_p) { keep++; break; } }
    std::uniform_real_distribution<double> u(0.0, cum);
    double r = u(*rng);
    for (int i = 0; i < keep; i++) { r -= probs_[idx_[i]]; if (r <= 0) return idx_[i]; }
    return idx_[keep - 1];
  }
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(*rng);
  for (int i = 0; i < vocab; i++) { r -= probs_[i]; if (r <= 0) return i; }
  return vocab - 1;
}

std::vector<int32_t> Sampler::sample_step(const StepInput& step, const std::vector<float>& logits, int vocab) {
  std::vector<int32_t> out;
  out.reserve(step.num_logits);
  int row = 0;
  for (const auto& sl : step.slices) {
    if (!sl.needs_logits) continue;
    out.push_back(sample(logits.data() + static_cast<size_t>(row) * vocab, vocab, sl.seq->params));
    row++;
  }
  return out;
}

}  // namespace engine
