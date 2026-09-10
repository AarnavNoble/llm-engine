#include "engine/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace engine {

static DType parse_dtype(const std::string& s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  throw std::runtime_error("unsupported safetensors dtype: " + s);
}

Tensor TensorView::to_f32() const {
  Tensor t(shape);
  const size_t n = numel();
  switch (dtype) {
    case DType::F32: std::memcpy(t.ptr(), data, n * 4); break;
    case DType::F16: { auto* p = reinterpret_cast<const uint16_t*>(data); for (size_t i = 0; i < n; i++) t.data[i] = f16_to_f32(p[i]); break; }
    case DType::BF16: { auto* p = reinterpret_cast<const uint16_t*>(data); for (size_t i = 0; i < n; i++) t.data[i] = bf16_to_f32(p[i]); break; }
  }
  return t;
}

std::vector<uint16_t> TensorView::to_f16() const {
  const size_t n = numel();
  std::vector<uint16_t> out(n);
  switch (dtype) {
    case DType::F16: std::memcpy(out.data(), data, n * 2); break;
    case DType::F32: { auto* p = reinterpret_cast<const float*>(data); for (size_t i = 0; i < n; i++) out[i] = f32_to_f16(p[i]); break; }
    case DType::BF16: { auto* p = reinterpret_cast<const uint16_t*>(data); for (size_t i = 0; i < n; i++) out[i] = f32_to_f16(bf16_to_f32(p[i])); break; }
  }
  return out;
}

SafeTensorsFile::SafeTensorsFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st{};
  if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat failed: " + path); }
  map_len_ = static_cast<size_t>(st.st_size);
  map_ = ::mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("mmap failed: " + path); }

  const auto* base = static_cast<const uint8_t*>(map_);
  if (map_len_ < 8) throw std::runtime_error("safetensors file too small: " + path);
  uint64_t hlen = 0;
  std::memcpy(&hlen, base, 8);  // little-endian on every platform we care about
  if (8 + hlen > map_len_) throw std::runtime_error("safetensors header overruns file: " + path);
  const uint8_t* blob = base + 8 + hlen;
  const size_t blob_len = map_len_ - 8 - hlen;

  nlohmann::json header = nlohmann::json::parse(base + 8, base + 8 + hlen);
  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") {
      for (auto m = it->begin(); m != it->end(); ++m) metadata_[m.key()] = m->get<std::string>();
      continue;
    }
    TensorView v;
    v.dtype = parse_dtype(it->at("dtype").get<std::string>());
    v.shape = it->at("shape").get<std::vector<int64_t>>();
    auto off = it->at("data_offsets").get<std::vector<size_t>>();
    if (off.size() != 2 || off[1] < off[0] || off[1] > blob_len)
      throw std::runtime_error("bad data_offsets for tensor " + it.key());
    v.data = blob + off[0];
    v.nbytes = off[1] - off[0];
    if (v.nbytes != v.numel() * dtype_size(v.dtype))
      throw std::runtime_error("size mismatch for tensor " + it.key());
    tensors_[it.key()] = v;
  }
}

SafeTensorsFile::~SafeTensorsFile() {
  if (map_) ::munmap(map_, map_len_);
}

const TensorView& SafeTensorsFile::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
  return it->second;
}

SafeTensorsDir::SafeTensorsDir(const std::string& model_dir) {
  std::vector<std::string> paths;
  for (const auto& e : std::filesystem::directory_iterator(model_dir))
    if (e.path().extension() == ".safetensors") paths.push_back(e.path().string());
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) throw std::runtime_error("no *.safetensors in " + model_dir);
  for (const auto& p : paths) {
    files_.push_back(std::make_unique<SafeTensorsFile>(p));
    for (const auto& [name, view] : files_.back()->tensors()) index_[name] = &view;
  }
}

bool SafeTensorsDir::has(const std::string& name) const { return index_.count(name) != 0; }

const TensorView& SafeTensorsDir::get(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) throw std::runtime_error("tensor not found: " + name);
  return *it->second;
}

std::vector<std::string> SafeTensorsDir::names() const {
  std::vector<std::string> out;
  out.reserve(index_.size());
  for (const auto& [n, _] : index_) out.push_back(n);
  return out;
}

}  // namespace engine
