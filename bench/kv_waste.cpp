// Measures KV cache memory efficiency: paged allocation (the real
// KVCacheManager and Scheduler) against contiguous reservation, on the same
// workload and the same memory budget.
//
// This is the one row of the benchmark table that is a property of the
// allocator rather than of the GPU, so it is measured here with a fake model
// instead of on rented hardware. The paged numbers come from the production
// code path; only the contiguous baseline is simulated, because the engine
// never implements contiguous allocation.
//
//   ./build/kv_waste --requests 512 --blocks 4096 --output-len 128
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "engine/scheduler.h"

using namespace engine;

namespace {

struct Workload {
  std::vector<int> prompt_len;
  std::vector<int> output_len;   // per request: real traffic does not stop in lockstep
  int mean_output = 128;
  int max_output = 0;            // what a length-declaring allocator must reserve
  int max_model_len = 32768;     // config.json max_position_embeddings for Qwen2.5
};

// Prompts follow the benchmark protocol's 128/512/1024 mix at 50/30/20.
// Output lengths are geometric around the mean unless --fixed-output is given:
// with identical output lengths every sequence finishes on the same step, which
// hides the entire cost of static batching and would flatter the comparison.
Workload make_workload(int n, int mean_output, bool fixed_output, uint64_t seed) {
  Workload w;
  w.mean_output = mean_output;
  std::mt19937_64 rng(seed);
  std::discrete_distribution<int> pick({0.5, 0.3, 0.2});
  std::geometric_distribution<int> out(1.0 / std::max(mean_output, 1));
  const int lens[3] = {128, 512, 1024};
  w.prompt_len.reserve(n);
  w.output_len.reserve(n);
  for (int i = 0; i < n; i++) {
    w.prompt_len.push_back(lens[pick(rng)]);
    w.output_len.push_back(fixed_output ? mean_output
                                        : std::clamp(out(rng) + 1, 1, 8 * mean_output));
  }
  w.max_output = *std::max_element(w.output_len.begin(), w.output_len.end());
  return w;
}

struct Result {
  std::string strategy;
  double slot_utilization = 0;   // time-averaged K/V-holding slots / allocated slots
  double mean_resident = 0;      // sequences holding KV per step
  double mean_active = 0;        // sequences producing a token per step: the throughput proxy
  int peak_resident = 0;
  long long steps = 0;
  long long preemptions = 0;
  long long recomputed_tokens = 0;   // work thrown away by preemption
  long long useful_tokens = 0;       // prompt + generated tokens actually delivered
};

// ---- paged: drive the real scheduler and allocator ----
Result run_paged(const Workload& w, int num_blocks, int block_size, const SchedulerConfig& cfg,
                 bool prefix_caching) {
  KVCacheManager kv(num_blocks, block_size, prefix_caching);
  Scheduler sch(cfg, kv);
  std::mt19937_64 rng(1234);

  for (size_t i = 0; i < w.prompt_len.size(); i++) {
    auto s = std::make_shared<Sequence>();
    s->id = i + 1;
    s->tokens.resize(w.prompt_len[i]);
    // Distinct tokens per request: this measures allocation, not prefix sharing.
    for (auto& t : s->tokens) t = static_cast<int32_t>(rng() & 0x7fff);
    s->prompt_len = s->num_tokens();
    s->params.max_tokens = w.output_len[i];
    s->params.ignore_eos = true;
    sch.add(s);
  }

  Result r;
  r.strategy = "paged";
  double util_sum = 0;
  long long resident_sum = 0, active_sum = 0;
  while (sch.has_work()) {
    StepInput step = sch.schedule();
    if (step.empty()) break;
    // Occupancy is sampled after scheduling, before the tokens are appended:
    // that is the moment the cache is holding everything the step needs.
    long long allocated = 0, used = 0;
    for (const auto& s : sch.running()) {
      allocated += static_cast<long long>(s->block_table.num_blocks()) * block_size;
      used += s->num_tokens();
    }
    if (allocated > 0) util_sum += static_cast<double>(used) / static_cast<double>(allocated);
    resident_sum += static_cast<long long>(sch.running().size());
    r.peak_resident = std::max(r.peak_resident, static_cast<int>(sch.running().size()));
    // Only sequences that actually decode this step count as active. In static
    // mode finished members keep their slot and their blocks but produce nothing.
    active_sum += step.num_decode_tokens;
    r.steps++;

    std::vector<int32_t> sampled(step.num_logits, 42);
    sch.on_step_done(step, sampled);
  }
  r.slot_utilization = r.steps ? util_sum / static_cast<double>(r.steps) : 0;
  r.mean_resident = r.steps ? static_cast<double>(resident_sum) / static_cast<double>(r.steps) : 0;
  r.mean_active = r.steps ? static_cast<double>(active_sum) / static_cast<double>(r.steps) : 0;
  r.preemptions = static_cast<long long>(sch.stats().preemptions);
  r.recomputed_tokens = static_cast<long long>(sch.stats().recomputed_tokens);
  for (size_t i = 0; i < w.prompt_len.size(); i++) r.useful_tokens += w.prompt_len[i] + w.output_len[i];
  return r;
}

// ---- contiguous: simulate a reservation allocator on the same budget ----
// Each admitted sequence holds one contiguous region for its whole lifetime.
// `reserve_max_model_len` picks the region size: the model's maximum context
// (what an allocator that cannot know the output length must reserve) or
// prompt + max_tokens (the best case, when the request declares its length).
Result run_contiguous(const Workload& w, long long total_slots, bool reserve_max_model_len,
                      int max_num_seqs, bool static_batching) {
  struct Live { long long reserved; int len; int remaining; };
  std::vector<Live> live;
  size_t next = 0;
  long long free_slots = total_slots;

  Result r;
  r.strategy = reserve_max_model_len ? "contiguous (reserve max_model_len)"
                                     : "contiguous (reserve prompt + max_tokens)";
  double util_sum = 0;
  long long resident_sum = 0, active_sum = 0;

  // A contiguous allocator must size the region before generation starts. It
  // either reserves the model's full context, or, if requests declare a token
  // cap, the prompt plus that cap — never the length actually generated.
  auto reservation_for = [&](int prompt) -> long long {
    return reserve_max_model_len ? w.max_model_len : prompt + w.max_output;
  };

  while (next < w.prompt_len.size() || !live.empty()) {
    // Static batching only refills once the batch has fully drained.
    const bool may_admit = !static_batching || live.empty();
    if (may_admit) {
      while (next < w.prompt_len.size() && static_cast<int>(live.size()) < max_num_seqs) {
        long long need = reservation_for(w.prompt_len[next]);
        if (need > free_slots) break;
        free_slots -= need;
        live.push_back({need, w.prompt_len[next], w.output_len[next]});
        next++;
      }
    }
    if (live.empty()) break;  // a single request does not fit the pool at all

    long long allocated = 0, used = 0, active = 0;
    for (const auto& l : live) {
      allocated += l.reserved;
      used += l.len;
      if (l.remaining > 0) active++;
    }
    util_sum += static_cast<double>(used) / static_cast<double>(allocated);
    resident_sum += static_cast<long long>(live.size());
    r.peak_resident = std::max(r.peak_resident, static_cast<int>(live.size()));
    active_sum += active;
    r.steps++;

    // A finished sequence stops growing: its region is held, not extended.
    for (auto& l : live) if (l.remaining > 0) { l.len++; l.remaining--; }
    if (!static_batching) {
      // Finished sequences release their region immediately.
      for (auto it = live.begin(); it != live.end();) {
        if (it->remaining <= 0) { free_slots += it->reserved; it = live.erase(it); } else ++it;
      }
    } else if (std::all_of(live.begin(), live.end(), [](const Live& l) { return l.remaining <= 0; })) {
      for (const auto& l : live) free_slots += l.reserved;
      live.clear();
    }
  }
  r.slot_utilization = r.steps ? util_sum / static_cast<double>(r.steps) : 0;
  r.mean_resident = r.steps ? static_cast<double>(resident_sum) / static_cast<double>(r.steps) : 0;
  r.mean_active = r.steps ? static_cast<double>(active_sum) / static_cast<double>(r.steps) : 0;
  return r;
}

void print_row(const Result& r) {
  std::printf("| %-42s | %9.1f%% | %9.1f%% | %8.1f | %10.1f | %6d | %7lld |\n",
              r.strategy.c_str(), 100.0 * r.slot_utilization, 100.0 * (1.0 - r.slot_utilization),
              r.mean_resident, r.mean_active, r.peak_resident, r.preemptions);
}

// Recompute-on-resume is the simplest correct preemption policy, but it is not
// free: a preempted sequence loses every token it had computed. This sweeps the
// memory budget to show where that cost starts to matter.
void sweep_pressure(const Workload& w, int block_size, const SchedulerConfig& cfg) {
  std::printf("\nrecompute cost against memory pressure (paged, continuous batching)\n");
  std::printf("| %8s | %9s | %8s | %11s | %14s |\n",
              "blocks", "slots", "preempt", "recomputed", "wasted work");
  std::printf("|%s|%s|%s|%s|%s|\n", std::string(10, '-').c_str(), std::string(11, '-').c_str(),
              std::string(10, '-').c_str(), std::string(13, '-').c_str(), std::string(16, '-').c_str());
  for (int blocks : {256, 384, 512, 768, 1024, 2048, 4096}) {
    Result r = run_paged(w, blocks, block_size, cfg, false);
    if (r.steps == 0) { std::printf("| %8d | %9d | %8s | %11s | %14s |\n",
                                    blocks, blocks * block_size, "-", "-", "does not fit"); continue; }
    const double wasted = static_cast<double>(r.recomputed_tokens) /
                          static_cast<double>(r.recomputed_tokens + r.useful_tokens);
    std::printf("| %8d | %9d | %8lld | %11lld | %13.1f%% |\n",
                blocks, blocks * block_size, r.preemptions, r.recomputed_tokens, 100.0 * wasted);
  }
}

void emit_json(const std::string& path, const std::vector<Result>& rs, const Workload& w,
               long long slots, int block_size) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
  std::fprintf(f, "{\n  \"workload\": {\"requests\": %zu, \"mean_output_len\": %d, \"max_output_len\": %d,"
                  " \"prompt_mix\": \"128/512/1024 at 50/30/20\", \"kv_slots\": %lld, \"block_size\": %d,"
                  " \"max_model_len\": %d},\n  \"results\": [\n",
               w.prompt_len.size(), w.mean_output, w.max_output, slots, block_size, w.max_model_len);
  for (size_t i = 0; i < rs.size(); i++) {
    std::fprintf(f, "    {\"strategy\": \"%s\", \"slot_utilization\": %.4f, \"kv_waste_fraction\": %.4f,"
                    " \"mean_resident_seqs\": %.2f, \"mean_active_seqs\": %.2f, \"peak_resident_seqs\": %d,"
                    " \"steps\": %lld, \"preemptions\": %lld, \"recomputed_tokens\": %lld}%s\n",
                 rs[i].strategy.c_str(), rs[i].slot_utilization, 1.0 - rs[i].slot_utilization,
                 rs[i].mean_resident, rs[i].mean_active, rs[i].peak_resident, rs[i].steps, rs[i].preemptions,
                 rs[i].recomputed_tokens, i + 1 < rs.size() ? "," : "");
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::printf("\nwrote %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  int requests = 512, blocks = 4096, block_size = 16, output_len = 128, max_num_seqs = 64;
  bool fixed_output = false, sweep = false;
  std::string json_out = "bench/results/kv-waste.json";
  for (int i = 1; i < argc; i++) {
    auto val = [&] { return std::atoi(argv[++i]); };
    if (!std::strcmp(argv[i], "--requests")) requests = val();
    else if (!std::strcmp(argv[i], "--blocks")) blocks = val();
    else if (!std::strcmp(argv[i], "--block-size")) block_size = val();
    else if (!std::strcmp(argv[i], "--output-len")) output_len = val();
    else if (!std::strcmp(argv[i], "--max-num-seqs")) max_num_seqs = val();
    else if (!std::strcmp(argv[i], "--fixed-output")) fixed_output = true;
    else if (!std::strcmp(argv[i], "--sweep")) sweep = true;
    else if (!std::strcmp(argv[i], "--json")) json_out = argv[++i];
    else { std::fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
  }

  const Workload w = make_workload(requests, output_len, fixed_output, 7);
  const long long slots = static_cast<long long>(blocks) * block_size;
  const double mean_prompt = std::accumulate(w.prompt_len.begin(), w.prompt_len.end(), 0.0) / requests;
  const double mean_out = std::accumulate(w.output_len.begin(), w.output_len.end(), 0.0) / requests;

  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = 4096;
  cfg.prefill_chunk_size = 512;

  std::printf("workload: %d requests, prompts 128/512/1024 at 50/30/20 (mean %.0f), outputs %s"
              " (mean %.0f, max %d)\n",
              requests, mean_prompt, fixed_output ? "fixed" : "geometric", mean_out, w.max_output);
  std::printf("budget:   %lld KV token slots (%d blocks x %d), max_num_seqs %d, max_model_len %d\n\n",
              slots, blocks, block_size, max_num_seqs, w.max_model_len);

  std::vector<Result> rs;
  rs.push_back(run_contiguous(w, slots, /*reserve_max_model_len=*/true, max_num_seqs, /*static=*/true));
  rs.push_back(run_contiguous(w, slots, /*reserve_max_model_len=*/false, max_num_seqs, /*static=*/true));
  rs.push_back(run_contiguous(w, slots, /*reserve_max_model_len=*/false, max_num_seqs, /*static=*/false));
  SchedulerConfig st = cfg; st.continuous = false;
  rs.push_back(run_paged(w, blocks, block_size, st, false));
  rs.back().strategy = "paged, static batching";
  rs.push_back(run_paged(w, blocks, block_size, cfg, false));
  rs.back().strategy = "paged, continuous batching";

  std::printf("| %-42s | %10s | %10s | %8s | %10s | %6s | %7s |\n",
              "strategy", "slot util", "KV waste", "resident", "decoding", "peak", "preempt");
  std::printf("|%s|%s|%s|%s|%s|%s|%s|\n", std::string(44, '-').c_str(), std::string(12, '-').c_str(),
              std::string(12, '-').c_str(), std::string(10, '-').c_str(), std::string(12, '-').c_str(),
              std::string(8, '-').c_str(), std::string(9, '-').c_str());
  for (const auto& r : rs) print_row(r);

  // "resident" counts sequences holding KV; "decoding" counts those producing a
  // token. Static batching keeps finished members resident, so the gap between
  // the two columns is exactly the capacity it wastes.
  const Result& best = rs.back();
  const Result& naive = rs.front();
  const Result& fair = rs[2];
  std::printf("\nslot utilization: %.1f%% paged vs %.1f%% contiguous with a declared cap (%.1fx)\n",
              100 * best.slot_utilization, 100 * fair.slot_utilization,
              best.slot_utilization / std::max(fair.slot_utilization, 1e-9));
  if (naive.steps == 0) {
    std::printf("reserving the full %d-token context does not fit a single request in %lld slots\n",
                w.max_model_len, slots);
  } else {
    std::printf("reserving the full %d-token context reaches only %.1f%% (%.0fx worse)\n",
                w.max_model_len, 100 * naive.slot_utilization,
                best.slot_utilization / naive.slot_utilization);
  }
  std::printf("sequences decoding per step: %.1f paged+continuous vs %.1f paged+static (%.1fx),"
              " %.1f contiguous+continuous\n", best.mean_active, rs[3].mean_active,
              best.mean_active / std::max(rs[3].mean_active, 1e-9), fair.mean_active);
  if (sweep) sweep_pressure(w, block_size, cfg);
  emit_json(json_out, rs, w, slots, block_size);
  return 0;
}
