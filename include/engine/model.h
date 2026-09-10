#pragma once
// The forward-pass boundary. A Model consumes one StepInput (a mix of
// prefill chunks and single decode tokens over paged KV block tables) and
// produces logits for every slice that asked for them.
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine/config.h"
#include "engine/kv_cache.h"
#include "engine/sequence.h"

namespace engine {

class Model {
 public:
  virtual ~Model() = default;
  virtual const ModelConfig& config() const = 0;
  // `logits` is resized to step.num_logits * vocab, row-major, rows in slice order.
  virtual void forward(const StepInput& step, std::vector<float>& logits) = 0;
  virtual const char* backend() const = 0;
};

struct CpuModelOptions {
  // Test hook: called after the embedding (layer = -1) and after every
  // decoder layer with the residual stream of the current slice [ntok, hidden].
  std::function<void(int layer, const float* resid, int ntok)> on_layer;
};

// Plain C++ reference implementation. Slow, fp32, and the oracle for every
// other backend. Reads the same paged KV layout the CUDA backend uses.
std::unique_ptr<Model> make_cpu_model(const std::string& model_dir, const KVCacheManager& kv,
                                      CpuModelOptions opts = {});

#ifdef ENGINE_CUDA
std::unique_ptr<Model> make_cuda_model(const std::string& model_dir, const KVCacheManager& kv);
#endif

}  // namespace engine
