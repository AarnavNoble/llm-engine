#pragma once
// Minimal host-side tensor utilities. The CPU reference path works in fp32;
// weights on disk are bf16/fp16 and get converted on load.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace engine {

enum class DType { F32, F16, BF16 };

inline size_t dtype_size(DType d) { return d == DType::F32 ? 4 : 2; }
inline const char* dtype_name(DType d) {
  switch (d) { case DType::F32: return "F32"; case DType::F16: return "F16"; default: return "BF16"; }
}

// ---- scalar conversions (bit-exact, round-to-nearest-even for f32->f16) ----
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16; float f; std::memcpy(&f, &u, 4); return f;
}
inline uint16_t f32_to_bf16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return static_cast<uint16_t>(u >> 16);  // inf/nan
  uint32_t lsb = (u >> 16) & 1u, bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((u + bias) >> 16);
}
inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, u;
  if (exp == 0) {
    if (man == 0) u = sign;
    else {  // subnormal: normalise
      int e = -1; do { e++; man <<= 1; } while ((man & 0x400u) == 0);
      man &= 0x3ffu; u = sign | ((127 - 15 - e) << 23) | (man << 13);
    }
  } else if (exp == 31) u = sign | 0x7f800000u | (man << 13);
  else u = sign | ((exp + 127 - 15) << 23) | (man << 13);
  float f; std::memcpy(&f, &u, 4); return f;
}
inline uint16_t f32_to_f16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u; int32_t exp = static_cast<int32_t>((u >> 23) & 0xffu) - 127 + 15;
  uint32_t man = u & 0x7fffffu;
  if (((u >> 23) & 0xffu) == 0xffu) return static_cast<uint16_t>(sign | 0x7c00u | (man ? 0x200u : 0));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    man |= 0x800000u; uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = man >> shift, rem = man & ((1u << shift) - 1), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u))) half++;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13), rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
  return static_cast<uint16_t>(half);
}

// ---- host tensor: owns an fp32 buffer ----
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;

  Tensor() = default;
  explicit Tensor(std::vector<int64_t> s) : shape(std::move(s)), data(numel_of(shape), 0.f) {}
  static size_t numel_of(const std::vector<int64_t>& s) { size_t n = 1; for (auto d : s) n *= static_cast<size_t>(d); return n; }
  size_t numel() const { return data.size(); }
  int64_t dim(size_t i) const { return shape[i]; }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
  std::string shape_str() const {
    std::string s = "["; for (size_t i = 0; i < shape.size(); i++) s += (i ? "," : "") + std::to_string(shape[i]); return s + "]";
  }
};

}  // namespace engine
