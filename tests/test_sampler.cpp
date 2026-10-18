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

// The nucleus is found with nth_element over a widening window rather than a
// full sort of the vocabulary. That is only safe if it picks the same token as
// the full sort would for every seed, so this holds a reference full-sort
// implementation beside it and demands exact agreement.
//
// The cases are chosen to cover what the widening loop does: a peaked
// distribution whose nucleus closes inside the first window of 64, a flat one
// whose nucleus is most of the vocabulary and forces the loop out to the full
// sort, and a distribution with deliberate exact ties, which is what the
// index tie-break in the comparator exists for.
namespace {

int32_t reference_top_p(const std::vector<float>& logits, const engine::SamplingParams& p) {
  const int vocab = static_cast<int>(logits.size());
  std::vector<float> probs(vocab);
  const float mx = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (int i = 0; i < vocab; i++) { probs[i] = std::exp((logits[i] - mx) / p.temperature); sum += probs[i]; }
  for (int i = 0; i < vocab; i++) probs[i] = static_cast<float>(probs[i] / sum);

  std::vector<int32_t> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return probs[a] > probs[b] || (probs[a] == probs[b] && a < b);
  });
  double cum = 0; int keep = 0;
  for (; keep < vocab; keep++) { cum += probs[idx[keep]]; if (cum >= p.top_p) { keep++; break; } }

  std::mt19937_64 rng(p.seed);
  std::uniform_real_distribution<double> u(0.0, cum);
  double r = u(rng);
  for (int i = 0; i < keep; i++) { r -= probs[idx[i]]; if (r <= 0) return idx[i]; }
  return idx[keep - 1];
}

}  // namespace

TEST_CASE("sampler: the partial-sort nucleus matches a full sort exactly") {
  const int vocab = 2000;
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<float> noise(-1.0f, 1.0f);

  std::vector<std::pair<const char*, std::vector<float>>> cases;

  // Peaked: a handful of tokens hold almost all the mass, so the nucleus
  // closes inside the first window and the loop never widens.
  {
    std::vector<float> l(vocab);
    for (int i = 0; i < vocab; i++) l[i] = noise(gen) - 12.0f;
    for (int i = 0; i < 5; i++) l[i * 37] = 9.0f - i;
    cases.emplace_back("peaked", l);
  }
  // Flat: every token is within noise of every other, so top-p 0.95 needs
  // almost the whole vocabulary and the loop is forced to the full sort.
  {
    std::vector<float> l(vocab);
    for (int i = 0; i < vocab; i++) l[i] = noise(gen) * 0.01f;
    cases.emplace_back("flat", l);
  }
  // Ties: many exactly equal logits, which is what the index tie-break in the
  // comparator is for. std::sort is unstable, so without it the two paths
  // could order the nucleus differently and draw different tokens.
  {
    std::vector<float> l(vocab, -3.0f);
    for (int i = 0; i < 400; i++) l[i] = 2.0f;
    cases.emplace_back("ties", l);
  }

  engine::Sampler s(1);
  for (const auto& [name, logits] : cases) {
    INFO("distribution: " << name);
    for (float top_p : {0.3f, 0.5f, 0.9f, 0.95f, 0.999f}) {
      for (uint64_t seed = 1; seed <= 40; seed++) {
        engine::SamplingParams p;
        p.temperature = 0.8f;
        p.top_p = top_p;
        p.seed = seed;
        INFO("top_p=" << top_p << " seed=" << seed);
        REQUIRE(s.sample(logits.data(), vocab, p) == reference_top_p(logits, p));
      }
    }
  }
}
