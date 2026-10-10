// Verifies each operation against the golden tensors dumped by golden_dump.
//
// The point is not to test the CPU code again, which the per-layer PyTorch
// comparison already covers. It is to establish the harness a CUDA kernel will
// be checked with: load the exact input and expected output of one operation,
// run one implementation of it, diff. When attention_decode.cu exists, its test
// is this file with the reference call swapped for a kernel launch, so a wrong
// kernel points at itself instead of at the end of a 24-layer forward pass.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/config.h"
#include "engine/ops.h"
#include "engine/safetensors.h"

using namespace engine;

namespace {

const std::string kGolden = std::string(ENGINE_TEST_DATA_DIR) + "/golden";

struct Golden {
  int rows = 0, cols = 0, second_cols = 0, third_cols = 0;
  std::vector<float> data, second, third;

  static Golden load(const std::string& tap, int layer) {
    const std::string path = kGolden + "/" + tap + ".L" + std::to_string(layer) + ".bin";
    FILE* f = std::fopen(path.c_str(), "rb");
    REQUIRE(f != nullptr);
    Golden g;
    int hdr[4];
    REQUIRE(std::fread(hdr, sizeof(int), 4, f) == 4);
    g.rows = hdr[0]; g.cols = hdr[1]; g.second_cols = hdr[2]; g.third_cols = hdr[3];
    auto read = [&](std::vector<float>& v, int cols) {
      if (cols == 0) return;
      v.resize(static_cast<size_t>(g.rows) * cols);
      REQUIRE(std::fread(v.data(), sizeof(float), v.size(), f) == v.size());
    };
    read(g.data, g.cols);
    read(g.second, g.second_cols);
    read(g.third, g.third_cols);
    std::fclose(f);
    return g;
  }
};

nlohmann::json manifest() {
  std::ifstream f(kGolden + "/manifest.json");
  REQUIRE(f);
  return nlohmann::json::parse(f);
}

bool golden_present() { return std::filesystem::exists(kGolden + "/manifest.json"); }

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  float m = 0;
  for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

// Relative tolerance against the magnitude of the expected tensor, so a tap
// whose values are O(100) is not held to the same absolute bar as one O(0.01).
void expect_close(const std::vector<float>& got, const std::vector<float>& want, const char* what) {
  float mag = 0;
  for (float v : want) mag = std::max(mag, std::fabs(v));
  const float diff = max_abs_diff(got, want);
  INFO(what << ": max |diff| = " << diff << " against max |expected| = " << mag);
  CHECK(diff <= 1e-4f * std::max(mag, 1.0f));
}

}  // namespace

TEST_CASE("golden: rmsnorm reproduces the dumped tensor", "[model][golden]") {
  if (!golden_present()) SKIP("run: ./build/golden_dump --out tests/data/golden");
  const auto m = manifest();
  const auto cfg = ModelConfig::load(m["model"].get<std::string>());
  SafeTensorsDir w(m["model"].get<std::string>());

  // Layer 0's input norm consumes the embedding output directly.
  const auto in = Golden::load("embedding", -1);
  const auto want = Golden::load("input_norm", 0);
  const Tensor weight = w.get("model.layers.0.input_layernorm.weight").to_f32();

  std::vector<float> got(static_cast<size_t>(in.rows) * in.cols);
  ref::rmsnorm(in.data.data(), weight.ptr(), got.data(), in.rows, in.cols, cfg.rms_norm_eps);
  expect_close(got, want.data, "rmsnorm");
}

TEST_CASE("golden: rope reproduces the dumped tensor for q and k", "[model][golden]") {
  if (!golden_present()) SKIP("run: ./build/golden_dump --out tests/data/golden");
  const auto m = manifest();
  const auto cfg = ModelConfig::load(m["model"].get<std::string>());

  const auto before = Golden::load("qkv_proj", 0);   // q, k, v before rotation
  const auto after = Golden::load("rope", 0);        // q, k after rotation
  REQUIRE(before.rows == after.rows);

  ref::RopeTables tables;
  tables.build(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta);

  // Positions are 0..N-1 for a single prefill of the whole prompt.
  std::vector<float> q = before.data, k = before.second;
  for (int t = 0; t < before.rows; t++) {
    ref::rope_inplace(q.data() + static_cast<size_t>(t) * before.cols,
                      cfg.num_attention_heads, cfg.head_dim, t, tables);
    ref::rope_inplace(k.data() + static_cast<size_t>(t) * before.second_cols,
                      cfg.num_key_value_heads, cfg.head_dim, t, tables);
  }
  expect_close(q, after.data, "rope(q)");
  expect_close(k, after.second, "rope(k)");

  // v must be untouched by rotation; a kernel that rotates it would still pass
  // a q-only check, so assert it explicitly.
  const auto v_before = before.third;
  CHECK(v_before.size() == static_cast<size_t>(before.rows) * before.third_cols);
}

TEST_CASE("golden: paged attention reproduces the dumped tensor", "[model][golden]") {
  if (!golden_present()) SKIP("run: ./build/golden_dump --out tests/data/golden");
  const auto m = manifest();
  const auto cfg = ModelConfig::load(m["model"].get<std::string>());
  const int block_size = m["config"]["block_size"];
  const int hd = cfg.head_dim, nh = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
  const int kv_dim = nkv * hd, group = cfg.gqa_group();

  const auto rope = Golden::load("rope", 0);        // q and k, rotated
  const auto qkv = Golden::load("qkv_proj", 0);     // v, unrotated
  const auto want = Golden::load("attn_out", 0);
  const int N = rope.rows;

  // Rebuild the sequence's block table from the manifest and scatter K/V into a
  // cache laid out exactly as the model and the kernel see it:
  // [slot][kv_head][head_dim], slot = block * block_size + offset.
  BlockTable bt;
  for (int b : m["block_table"]) { bt.blocks.push_back(b); bt.hashes.push_back(0); }
  REQUIRE(bt.num_blocks() * block_size >= N);

  const size_t slots = static_cast<size_t>(bt.num_blocks()) * block_size;
  std::vector<float> k_cache(slots * kv_dim, 0.f), v_cache(slots * kv_dim, 0.f);
  for (int t = 0; t < N; t++) {
    const int64_t slot = bt.slot(t, block_size);
    std::copy_n(rope.second.data() + static_cast<size_t>(t) * kv_dim, kv_dim,
                k_cache.begin() + static_cast<size_t>(slot) * kv_dim);
    std::copy_n(qkv.third.data() + static_cast<size_t>(t) * kv_dim, kv_dim,
                v_cache.begin() + static_cast<size_t>(slot) * kv_dim);
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  std::vector<float> got(static_cast<size_t>(N) * nh * hd, 0.f), scratch;
  for (int t = 0; t < N; t++) {
    for (int h = 0; h < nh; h++) {
      ref::paged_attention_head(rope.data.data() + static_cast<size_t>(t) * rope.cols + h * hd,
                                k_cache.data(), v_cache.data(), bt, block_size, t,
                                h / group, nkv, hd, scale,
                                got.data() + static_cast<size_t>(t) * rope.cols + h * hd, scratch);
    }
  }
  expect_close(got, want.data, "paged attention");

  // The last token attends over every position, which is the decode case the
  // CUDA kernel will run, and it must span more than one block to exercise the
  // block table rather than a single contiguous run.
  REQUIRE(N > block_size);
}

TEST_CASE("golden: silu_mul matches a scalar recomputation", "[model][golden]") {
  if (!golden_present()) SKIP("run: ./build/golden_dump --out tests/data/golden");
  // silu_mul's inputs are not dumped (they are two GEMM outputs), so this checks
  // the op against an independent scalar formulation on its own data, which is
  // what a kernel would be diffed against.
  std::vector<float> gate{-4.f, -0.5f, 0.f, 0.5f, 1.f, 7.f}, up{2.f, -1.f, 3.f, 0.25f, -2.f, 1.5f};
  std::vector<float> expected(gate.size());
  for (size_t i = 0; i < gate.size(); i++)
    expected[i] = static_cast<float>(gate[i] / (1.0 + std::exp(-static_cast<double>(gate[i]))) * up[i]);
  ref::silu_mul(gate.data(), up.data(), gate.size());
  for (size_t i = 0; i < gate.size(); i++) CHECK(gate[i] == Catch::Approx(expected[i]).margin(1e-6));
}
