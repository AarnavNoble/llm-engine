#pragma once
#include <cstdint>
#include <random>
#include <vector>

#include "engine/sequence.h"

namespace engine {

// Host sampler over a [rows, vocab] logits matrix. Greedy when
// temperature <= 0, otherwise temperature + top-p (nucleus) sampling.
class Sampler {
 public:
  explicit Sampler(uint64_t seed = 42) : rng_(seed) {}
  int32_t sample(const float* logits, int vocab, const SamplingParams& p);
  std::vector<int32_t> sample_step(const StepInput& step, const std::vector<float>& logits, int vocab);
  static int32_t argmax(const float* logits, int vocab);

 private:
  std::mt19937_64 rng_;
  std::vector<float> probs_;
  std::vector<int32_t> idx_;
};

}  // namespace engine
