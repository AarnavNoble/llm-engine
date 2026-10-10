// Writes per-operation golden tensors for kernel development.
//
// The per-layer PyTorch comparison tells you that a forward pass is wrong. It
// does not tell you which of the eight operations inside a layer is at fault,
// and on a GPU that difference is hours. This dumps the exact input and output
// of every operation, for a fixed prompt, from the CPU implementation that is
// already verified against PyTorch layer by layer.
//
// A CUDA kernel can then be checked on its own, the moment it compiles, before
// it is wired into the model.
//
//   ./build/golden_dump --model models/Qwen2.5-0.5B-Instruct --out tests/data/golden
//
// Layout: one file per tap per layer, plus a manifest.
//   <out>/<tap>.L<layer>.bin    int32 rows, cols, extra_cols, extra2_cols
//                               then float32 data[rows*cols], then the extras
//   <out>/manifest.json         shapes, the prompt, and the config it came from
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "engine/model.h"
#include "engine/scheduler.h"

using namespace engine;

namespace {

struct Writer {
  std::string dir;
  int max_layer = 1;          // dump layer 0 and 1 only; the rest are identical in shape
  std::vector<std::string> written;

  void operator()(const TapData& d) {
    if (d.layer > max_layer) return;
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%s.L%d.bin", dir.c_str(), tap_name(d.tap), d.layer);
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return; }
    const int hdr[4] = {d.rows, d.cols, d.second ? d.second_cols : 0, d.third ? d.third_cols : 0};
    std::fwrite(hdr, sizeof(int), 4, f);
    std::fwrite(d.data, sizeof(float), static_cast<size_t>(d.rows) * d.cols, f);
    if (d.second) std::fwrite(d.second, sizeof(float), static_cast<size_t>(d.rows) * d.second_cols, f);
    if (d.third) std::fwrite(d.third, sizeof(float), static_cast<size_t>(d.rows) * d.third_cols, f);
    std::fclose(f);
    char entry[256];
    std::snprintf(entry, sizeof(entry),
                  "{\"tap\": \"%s\", \"layer\": %d, \"rows\": %d, \"cols\": %d, "
                  "\"second_cols\": %d, \"third_cols\": %d}",
                  tap_name(d.tap), d.layer, d.rows, d.cols, hdr[2], hdr[3]);
    written.push_back(entry);
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string model = "models/Qwen2.5-0.5B-Instruct", out = "tests/data/golden";
  int prompt_len = 40, max_layer = 1;
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--model")) model = argv[++i];
    else if (!std::strcmp(argv[i], "--out")) out = argv[++i];
    else if (!std::strcmp(argv[i], "--prompt-len")) prompt_len = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--max-layer")) max_layer = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
  }

  KVCacheManager kv(64, 16, false);
  Writer w;
  w.dir = out;
  w.max_layer = max_layer;
  CpuModelOptions opts;
  opts.on_tap = [&w](const TapData& d) { w(d); };
  auto cpu = make_cpu_model(model, kv, opts);
  const auto& cfg = cpu->config();

  // A deterministic prompt, long enough to cross block boundaries so the
  // attention tap exercises a real block table rather than a single block.
  auto seq = std::make_shared<Sequence>();
  seq->tokens.resize(prompt_len);
  for (int i = 0; i < prompt_len; i++) seq->tokens[i] = static_cast<int32_t>(1000 + 7 * i);
  seq->prompt_len = prompt_len;
  if (!kv.allocate_prompt(seq->block_table, seq->tokens)) {
    std::fprintf(stderr, "prompt does not fit the test pool\n");
    return 1;
  }

  StepInput step;
  StepSlice sl;
  sl.seq = seq.get();
  sl.len = prompt_len;
  sl.needs_logits = true;
  sl.is_prefill = true;
  step.slices.push_back(sl);
  step.num_prefill_tokens = prompt_len;
  step.num_logits = 1;

  std::vector<float> logits;
  cpu->forward(step, logits);

  char path[512];
  std::snprintf(path, sizeof(path), "%s/manifest.json", out.c_str());
  FILE* f = std::fopen(path, "wb");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return 1; }
  std::fprintf(f, "{\n  \"model\": \"%s\",\n  \"prompt_len\": %d,\n  \"max_layer\": %d,\n",
               model.c_str(), prompt_len, max_layer);
  std::fprintf(f, "  \"config\": {\"hidden\": %d, \"layers\": %d, \"q_heads\": %d, \"kv_heads\": %d,"
                  " \"head_dim\": %d, \"intermediate\": %d, \"vocab\": %d, \"rms_eps\": %g,"
                  " \"rope_theta\": %g, \"block_size\": %d},\n",
               cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
               cfg.num_key_value_heads, cfg.head_dim, cfg.intermediate_size, cfg.vocab_size,
               cfg.rms_norm_eps, cfg.rope_theta, kv.block_size());
  std::fprintf(f, "  \"prompt_ids\": [");
  for (int i = 0; i < prompt_len; i++) std::fprintf(f, "%s%d", i ? ", " : "", seq->tokens[i]);
  std::fprintf(f, "],\n  \"block_table\": [");
  for (int i = 0; i < seq->block_table.num_blocks(); i++)
    std::fprintf(f, "%s%d", i ? ", " : "", seq->block_table.blocks[i]);
  std::fprintf(f, "],\n  \"tensors\": [\n");
  for (size_t i = 0; i < w.written.size(); i++)
    std::fprintf(f, "    %s%s\n", w.written[i].c_str(), i + 1 < w.written.size() ? "," : "");
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);

  std::printf("wrote %zu tensors to %s\n", w.written.size(), out.c_str());
  kv.free(seq->block_table);
  return 0;
}
