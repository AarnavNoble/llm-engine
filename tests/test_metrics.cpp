#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "engine/metrics.h"

using namespace engine;

namespace {
bool has_line(const std::string& text, const std::string& line) {
  return text.find("\n" + line + "\n") != std::string::npos || text.rfind(line + "\n", 0) == 0;
}
}  // namespace

TEST_CASE("histogram: cumulative buckets, sum and count in Prometheus format") {
  Histogram h({1, 2, 4});
  h.observe(0.5);   // le=1
  h.observe(1.5);   // le=2
  h.observe(4.0);   // le=4 (boundary is inclusive)
  h.observe(10.0);  // +Inf only
  std::string out = h.render("t_seconds", "test");

  CHECK(has_line(out, "# TYPE t_seconds histogram"));
  CHECK(has_line(out, "t_seconds_bucket{le=\"1\"} 1"));
  CHECK(has_line(out, "t_seconds_bucket{le=\"2\"} 2"));
  CHECK(has_line(out, "t_seconds_bucket{le=\"4\"} 3"));
  CHECK(has_line(out, "t_seconds_bucket{le=\"+Inf\"} 4"));
  CHECK(has_line(out, "t_seconds_sum 16"));
  CHECK(has_line(out, "t_seconds_count 4"));
}

TEST_CASE("histogram: empty renders zeros and quantiles are 0") {
  Histogram h({1, 2});
  std::string out = h.render("e_seconds", "empty");
  CHECK(has_line(out, "e_seconds_bucket{le=\"+Inf\"} 0"));
  CHECK(has_line(out, "e_seconds_count 0"));
  CHECK(h.quantile(0.5) == 0.0);
}

TEST_CASE("histogram: quantile interpolates inside the containing bucket") {
  Histogram h({1, 2, 4});
  h.observe(1.5);  // the only sample, in bucket (1, 2]
  // Half of the single sample sits halfway across the bucket: 1 + 0.5 * (2 - 1).
  CHECK(h.quantile(0.5) == Catch::Approx(1.5));
  CHECK(h.quantile(1.0) == Catch::Approx(2.0));

  Histogram g({1, 2, 4});
  for (int i = 0; i < 4; i++) g.observe(0.5);   // 4 in (0, 1]
  for (int i = 0; i < 4; i++) g.observe(3.0);   // 4 in (2, 4]
  CHECK(g.quantile(0.5) == Catch::Approx(1.0));   // boundary between the two groups
  CHECK(g.quantile(0.25) == Catch::Approx(0.5));  // halfway into the first bucket
  // p95 = sample 7.6 of 8; 4 samples sit at or below 2, so it lands 90% of the
  // way through the (2, 4] bucket: 2 + 0.9 * 2.
  CHECK(g.quantile(0.95) == Catch::Approx(3.8));
}

TEST_CASE("histogram: values above the last bucket are counted but clamp the quantile") {
  Histogram h({1, 2});
  h.observe(100.0);
  CHECK(has_line(h.render("o", "overflow"), "o_bucket{le=\"+Inf\"} 1"));
  CHECK(h.quantile(0.99) == Catch::Approx(2.0));  // cannot report beyond the last bound
}

TEST_CASE("metrics: prometheus output carries counters, gauges and every histogram") {
  Metrics m;
  m.requests_total += 3;
  m.generated_tokens_total += 128;
  m.preemptions_total++;
  m.queue_depth = 7;
  m.kv_blocks_used = 12;
  m.kv_waste_fraction = 0.25;
  m.ttft.observe(0.2);
  m.inter_token.observe(0.01);

  std::string out = m.render_prometheus();
  CHECK(has_line(out, "# TYPE engine_requests_total counter"));
  CHECK(has_line(out, "engine_requests_total 3"));
  CHECK(has_line(out, "engine_generated_tokens_total 128"));
  CHECK(has_line(out, "engine_preemptions_total 1"));
  CHECK(has_line(out, "# TYPE engine_queue_depth gauge"));
  CHECK(has_line(out, "engine_queue_depth 7"));
  CHECK(has_line(out, "engine_kv_blocks_used 12"));
  CHECK(has_line(out, "engine_kv_waste_fraction 0.25"));
  CHECK(has_line(out, "engine_ttft_seconds_count 1"));
  CHECK(has_line(out, "engine_inter_token_seconds_count 1"));
  // Every metric must be preceded by HELP and TYPE, or Prometheus drops the type.
  size_t help = 0, type = 0, pos = 0;
  while ((pos = out.find("# HELP ", pos)) != std::string::npos) { help++; pos += 7; }
  pos = 0;
  while ((pos = out.find("# TYPE ", pos)) != std::string::npos) { type++; pos += 7; }
  CHECK(help == type);
  CHECK(help >= 20);
}

TEST_CASE("metrics: the KEDA scaling metric is exposed under its documented name") {
  // deploy/helm/templates/scaledobject.yaml queries engine_queue_depth; if this
  // name ever changes, autoscaling silently stops working.
  Metrics m;
  m.queue_depth = 42;
  CHECK(has_line(m.render_prometheus(), "engine_queue_depth 42"));
}
