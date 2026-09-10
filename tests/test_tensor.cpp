#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "engine/tensor.h"

using namespace engine;

TEST_CASE("fp16 round trip on representable values") {
  for (float f : {0.f, 1.f, -1.f, 0.5f, 65504.f, -65504.f, 6.103515625e-05f, 5.960464477539063e-08f, 3.140625f}) {
    REQUIRE(f16_to_f32(f32_to_f16(f)) == f);
  }
}

TEST_CASE("fp16 conversion matches known bit patterns") {
  REQUIRE(f32_to_f16(1.0f) == 0x3C00);
  REQUIRE(f32_to_f16(-2.0f) == 0xC000);
  REQUIRE(f32_to_f16(65504.f) == 0x7BFF);
  REQUIRE(f32_to_f16(1e5f) == 0x7C00);  // overflow -> inf
  REQUIRE(f16_to_f32(0x0001) == 5.960464477539063e-08f);  // smallest subnormal
  REQUIRE(std::isnan(f16_to_f32(0x7E00)));
}

TEST_CASE("fp16 rounds to nearest even") {
  // 1 + 2^-11 is exactly halfway between 1.0 and 1+2^-10; ties-to-even -> 1.0
  REQUIRE(f32_to_f16(1.0f + 0.00048828125f) == 0x3C00);
  // 1 + 3*2^-11 is halfway between 1+2^-10 and 1+2^-9; ties-to-even -> 1+2^-9 (0x3C02)
  REQUIRE(f32_to_f16(1.0f + 3 * 0.00048828125f) == 0x3C02);
}

TEST_CASE("bf16 round trip") {
  for (float f : {0.f, 1.f, -1.f, 3.125f, 1024.f, -0.375f}) REQUIRE(bf16_to_f32(f32_to_bf16(f)) == f);
  REQUIRE(f32_to_bf16(1.0f) == 0x3F80);
  REQUIRE(bf16_to_f32(0x4049) == 3.140625f);
}

TEST_CASE("tensor shape helpers") {
  Tensor t({2, 3, 4});
  REQUIRE(t.numel() == 24);
  REQUIRE(t.shape_str() == "[2,3,4]");
}
