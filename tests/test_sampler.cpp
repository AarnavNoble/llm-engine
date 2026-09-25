#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <map>
#include <vector>

#include "engine/sampler.h"

using namespace engine;

namespace {
// Logits whose softmax at temperature 1 is a known distribution.
// exp(2)=7.389, exp(1)=2.718, exp(0)=1 -> p = .665, .245, .090
std::vector<float> peaked() { return {2.f, 1.f, 0.f}; }

SamplingParams greedy_params() { SamplingParams p; p.temperature = 0.f; return p; }
}  // namespace

TEST_CASE("sampler: argmax") {
  std::vector<float> v{-1.f, 5.f, 2.f, 5.f};
  REQUIRE(Sampler::argmax(v.data(), 4) == 1);  // first of the tied maxima
  std::vector<float> one{3.f};
  REQUIRE(Sampler::argmax(one.data(), 1) == 0);
  std::vector<float> neg{-9.f, -8.f, -100.f};
  REQUIRE(Sampler::argmax(neg.data(), 3) == 1);
}

TEST_CASE("sampler: temperature <= 0 is greedy and ignores the rng") {
  Sampler a(1), b(999);
  auto v = peaked();
  for (int i = 0; i < 20; i++) {
    REQUIRE(a.sample(v.data(), 3, greedy_params()) == 0);
    REQUIRE(b.sample(v.data(), 3, greedy_params()) == 0);
  }
}

TEST_CASE("sampler: a per-request seed is reproducible across instances") {
  auto v = peaked();
  SamplingParams p; p.temperature = 1.0f; p.seed = 12345;
  std::vector<int32_t> first, second;
  Sampler s1(7), s2(8);  // different engine seeds must not matter
  for (int i = 0; i < 16; i++) first.push_back(s1.sample(v.data(), 3, p));
  for (int i = 0; i < 16; i++) second.push_back(s2.sample(v.data(), 3, p));
  REQUIRE(first == second);
  // With a fixed seed every draw is identical (the generator is reseeded per call).
  REQUIRE(std::count(first.begin(), first.end(), first[0]) == 16);
}

TEST_CASE("sampler: unseeded draws follow the softmax distribution") {
  auto v = peaked();
  SamplingParams p; p.temperature = 1.0f;
  Sampler s(42);
  std::map<int32_t, int> hist;
  const int N = 20000;
  for (int i = 0; i < N; i++) hist[s.sample(v.data(), 3, p)]++;
  REQUIRE(hist.size() == 3);
  // 3-sigma on a binomial with p=.665, n=20000 is ~2%; allow 3 points of slack.
  CHECK(hist[0] / double(N) == Catch::Approx(0.665).margin(0.03));
  CHECK(hist[1] / double(N) == Catch::Approx(0.245).margin(0.03));
  CHECK(hist[2] / double(N) == Catch::Approx(0.090).margin(0.03));
}

TEST_CASE("sampler: top-p truncates the tail") {
  auto v = peaked();
  Sampler s(42);
  SECTION("top_p below the leading probability keeps only the top token") {
    SamplingParams p; p.temperature = 1.0f; p.top_p = 0.5f;  // p0 = .665 > .5
    for (int i = 0; i < 500; i++) REQUIRE(s.sample(v.data(), 3, p) == 0);
  }
  SECTION("top_p inside the second token's mass excludes only the third") {
    SamplingParams p; p.temperature = 1.0f; p.top_p = 0.9f;  // .665 + .245 = .910 >= .9
    std::map<int32_t, int> hist;
    for (int i = 0; i < 5000; i++) hist[s.sample(v.data(), 3, p)]++;
    REQUIRE(hist.count(2) == 0);
    CHECK(hist[0] > 0);
    CHECK(hist[1] > 0);
  }
}

TEST_CASE("sampler: temperature widens the distribution") {
  auto v = peaked();
  Sampler cold(3), hot(3);
  SamplingParams c; c.temperature = 0.1f;
  SamplingParams h; h.temperature = 5.0f;
  int cold_top = 0, hot_top = 0;
  for (int i = 0; i < 4000; i++) {
    cold_top += cold.sample(v.data(), 3, c) == 0;
    hot_top += hot.sample(v.data(), 3, h) == 0;
  }
  CHECK(cold_top > 3900);   // nearly deterministic
  CHECK(hot_top < 2000);    // close to uniform over 3 tokens
}

TEST_CASE("sampler: sample_step returns one token per logits-producing slice, in order") {
  // Three sequences; the middle slice is a prefill chunk that does not sample.
  std::vector<SequencePtr> seqs;
  for (int i = 0; i < 3; i++) {
    auto s = std::make_shared<Sequence>();
    s->params.temperature = 0.f;  // greedy, so expectations are exact
    seqs.push_back(s);
  }
  StepInput step;
  auto add = [&](Sequence* s, bool logits) {
    StepSlice sl; sl.seq = s; sl.len = 1; sl.needs_logits = logits; step.slices.push_back(sl);
    if (logits) step.num_logits++;
  };
  add(seqs[0].get(), true);
  add(seqs[1].get(), false);
  add(seqs[2].get(), true);

  const int vocab = 4;
  std::vector<float> logits = {
      0.f, 9.f, 0.f, 0.f,   // row 0 -> token 1
      0.f, 0.f, 0.f, 9.f,   // row 1 -> token 3
  };
  Sampler s(1);
  auto out = s.sample_step(step, logits, vocab);
  REQUIRE(out == std::vector<int32_t>{1, 3});
}
