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

// Named points inside one decoder layer. Each CUDA kernel will be responsible
// for exactly one of these, so capturing the tensors here gives every kernel its
// own expected output instead of only a per-layer residual to diff against.
enum class Tap {
  Embedding,     // [ntok, hidden]   token embeddings, before layer 0
  InputNorm,     // [ntok, hidden]   RMSNorm of the residual stream
  QkvProj,       // q [ntok, hidden], k and v [ntok, kv_dim]  before RoPE
  Rope,          // q and k after rotary embedding, before the cache write
  AttnOut,       // [ntok, hidden]   attention output, before o_proj
  AttnResidual,  // [ntok, hidden]   residual stream after the attention block
  PostNorm,      // [ntok, hidden]   RMSNorm before the MLP
  SiluMul,       // [ntok, inter]    silu(gate) * up
  LayerOut,      // [ntok, hidden]   residual stream after the whole layer
  Logits,        // [rows, vocab]    final logits
};

inline const char* tap_name(Tap t) {
  switch (t) {
    case Tap::Embedding: return "embedding";
    case Tap::InputNorm: return "input_norm";
    case Tap::QkvProj: return "qkv_proj";
    case Tap::Rope: return "rope";
    case Tap::AttnOut: return "attn_out";
    case Tap::AttnResidual: return "attn_residual";
    case Tap::PostNorm: return "post_norm";
    case Tap::SiluMul: return "silu_mul";
    case Tap::LayerOut: return "layer_out";
    default: return "logits";
  }
}

// One captured tensor. `second` and `third` carry the extra outputs of taps that
// produce more than one (q, k, v), and are null otherwise.
struct TapData {
  Tap tap;
  int layer;              // -1 for taps outside a decoder layer
  int rows, cols;
  const float* data;
  const float* second = nullptr;
  const float* third = nullptr;
  int second_cols = 0, third_cols = 0;
};

struct CpuModelOptions {
  // Test hook: called after the embedding (layer = -1) and after every
  // decoder layer with the residual stream of the current slice [ntok, hidden].
  std::function<void(int layer, const float* resid, int ntok)> on_layer;
  // Finer hook: called at every Tap above. Used to dump per-kernel golden
  // tensors, so a CUDA kernel can be verified in isolation rather than by
  // diffing the end of a 24-layer forward pass.
  std::function<void(const TapData&)> on_tap;
};

// Plain C++ reference implementation. Slow, fp32, and the oracle for every
// other backend. Reads the same paged KV layout the CUDA backend uses.
std::unique_ptr<Model> make_cpu_model(const std::string& model_dir, const KVCacheManager& kv,
                                      CpuModelOptions opts = {});

#ifdef ENGINE_CUDA
std::unique_ptr<Model> make_cuda_model(const std::string& model_dir, const KVCacheManager& kv);
#endif

}  // namespace engine
