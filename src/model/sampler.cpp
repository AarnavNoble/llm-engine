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

    // Descending by probability, ties broken by index. The tie-break is not
    // cosmetic: std::sort is unstable, so without a total order the nucleus
    // could come out in a different order here than under a full sort, and
    // the same seed would then pick a different token. With it, the partial
    // path below is required to agree with the full sort exactly.
    auto by_prob = [&](int a, int b) {
      return probs_[a] > probs_[b] || (probs_[a] == probs_[b] && a < b);
    };

    // The comment that used to sit here said a partial sort was enough,
    // directly above a full sort of all 151,936 indices. It was right: the
    // nucleus of a peaked distribution is a few dozen tokens, and everything
    // below it is ordered for nothing. Measured at batch 1, that sort was
    // 17.95 ms of an 18.2 ms sampling step -- 80x the cost of greedy, and
    // invisible to every benchmark here because they all send temperature 0.
    //
    // The nucleus size is not known in advance, so take the top k, check
    // whether it holds top_p of the mass, and widen if it does not.
    // nth_element partitions in linear time and leaves the range holding the
    // same indices, so it can be repeated on a wider k.
    double cum = 0;
    int keep = 0;
    for (int k = 64;; k *= 4) {
      if (k >= vocab) {
        std::sort(idx_.begin(), idx_.end(), by_prob);
        k = vocab;
      } else {
        std::nth_element(idx_.begin(), idx_.begin() + k, idx_.end(), by_prob);
        std::sort(idx_.begin(), idx_.begin() + k, by_prob);
      }
      cum = 0;
      keep = 0;
      for (; keep < k; keep++) { cum += probs_[idx_[keep]]; if (cum >= p.top_p) { keep++; break; } }
      // keep < k means the nucleus closed inside the window, so widening it
      // would order tokens that cannot be chosen.
      if (keep < k || k == vocab) break;
    }
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
