#pragma once
// Zero-copy safetensors reader: JSON header + raw byte blob, mmap'd.
// Format: u64 little-endian header length, UTF-8 JSON header, then data.
#include <cstdint>
#include <map>
#include <string>
#include <memory>
#include <vector>

#include "engine/tensor.h"

namespace engine {

struct TensorView {
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;  // points into the mmap
  size_t nbytes = 0;
  size_t numel() const { return Tensor::numel_of(shape); }
  // Materialise as fp32 (host copy).
  Tensor to_f32() const;
  // Materialise as fp16 bits (host copy), for upload to the GPU.
  std::vector<uint16_t> to_f16() const;
};

class SafeTensorsFile {
 public:
  explicit SafeTensorsFile(const std::string& path);
  ~SafeTensorsFile();
  SafeTensorsFile(const SafeTensorsFile&) = delete;
  SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorView& get(const std::string& name) const;
  const std::map<std::string, TensorView>& tensors() const { return tensors_; }
  const std::map<std::string, std::string>& metadata() const { return metadata_; }

 private:
  void* map_ = nullptr;
  size_t map_len_ = 0;
  std::map<std::string, TensorView> tensors_;
  std::map<std::string, std::string> metadata_;
};

// A model directory may shard weights across several *.safetensors files.
class SafeTensorsDir {
 public:
  explicit SafeTensorsDir(const std::string& model_dir);
  bool has(const std::string& name) const;
  const TensorView& get(const std::string& name) const;
  std::vector<std::string> names() const;

 private:
  std::vector<std::unique_ptr<SafeTensorsFile>> files_;
  std::map<std::string, const TensorView*> index_;
};

}  // namespace engine
