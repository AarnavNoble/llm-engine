#pragma once
// Where a step's *host* time goes.
//
// src/model/cuda/cuda_model.cu already profiles the device with CUDA events,
// and that profile is complete in the sense that every kernel is inside some
// region. It is also, on its own, misleading: the regions summed to 23.97
// ms/step while the server was turning over a step every 32.66 ms at 980 tok/s
// on an A40. A quarter of every step was somewhere a device profiler cannot
// look, because CUDA events measure a position in a stream, not CPU work.
//
// Three candidates live in that gap:
//
//   - the logits readback, R x 151,936 fp16 -- 9.7 MB at 32 sequences -- copied
//     device to host every step
//   - widening those 4.9 M values with a branchy software f16_to_f32
//   - host-side sampling, which for top-p sorts the vocabulary per sequence
//
// This measures all three, plus the whole step as a denominator, so the
// unaccounted remainder becomes a number rather than a subtraction from an
// end-to-end throughput figure measured in a different run.
//
// Opt in with ENGINE_PROFILE=1, the same switch as the device profiler, so one
// run produces both halves. Output goes to ENGINE_HOST_PROFILE_OUT or
// bench/results/host-time.json.
//
// Backend-agnostic on purpose. The sampler does not know whether it was handed
// logits by the CPU or the CUDA model and should not have to take a profiler
// argument to be measured, so this is a process-global rather than something
// threaded through signatures for the sake of a diagnostic.
//
// Not thread safe, and does not need to be: Engine::run() owns one step thread
// and everything from schedule() to on_step_done() happens on it. A region
// entered from any other thread would corrupt the totals silently, so don't.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace engine {

class HostProfile {
 public:
  static HostProfile& get() {
    static HostProfile p;
    return p;
  }

  bool on() const { return on_; }

  // Called once per scope site, through the macro's static local, so the string
  // comparison happens once per site rather than once per step.
  int region(const char* name) {
    for (size_t i = 0; i < names_.size(); i++)
      if (names_[i] == name) return static_cast<int>(i);
    names_.push_back(name);
    total_ns_.push_back(0);
    return static_cast<int>(names_.size()) - 1;
  }

  void add(int id, long long ns) {
    if (id >= 0 && id < static_cast<int>(total_ns_.size())) total_ns_[id] += ns;
  }

  // The step counter is advanced by the outermost region rather than by a
  // separate call, so a step that throws part way through is still counted
  // once and its partial regions are not divided by a step that never ended.
  void count_step() { steps_++; }

  ~HostProfile() {
    if (on_) report();
  }

 private:
  HostProfile() {
    const char* e = std::getenv("ENGINE_PROFILE");
    on_ = e && *e && *e != '0';
  }

  // The denominator, if it was instrumented. Everything else is a share of it.
  double wall_ms_per_step() const {
    for (size_t i = 0; i < names_.size(); i++)
      if (names_[i] == "step_wall") return total_ns_[i] / 1e6 / steps_;
    return 0.0;
  }

  void report() const {
    if (steps_ == 0) return;
    const double wall = wall_ms_per_step();

    std::printf("\nper-step host cost over %d steps (steady_clock, step thread)\n", steps_);
    std::printf("%-20s %10s %8s\n", "region", "ms/step", "of step");
    for (size_t i = 0; i < names_.size(); i++) {
      const double ms = total_ns_[i] / 1e6 / steps_;
      if (wall > 0)
        std::printf("%-20s %10.3f %7.1f%%\n", names_[i].c_str(), ms, 100.0 * ms / wall);
      else
        std::printf("%-20s %10.3f %8s\n", names_[i].c_str(), ms, "-");
    }
    if (wall > 0)
      std::printf("\nstep_wall is the whole step: scheduling, forward, sampling and bookkeeping.\n"
                  "Subtract the device total in kernel-time.json from it to see what is left.\n");

    const char* out = std::getenv("ENGINE_HOST_PROFILE_OUT");
    FILE* f = std::fopen(out && *out ? out : "bench/results/host-time.json", "w");
    if (!f) return;
    std::fprintf(f, "{\n  \"steps\": %d,\n  \"regions\": [\n", steps_);
    for (size_t i = 0; i < names_.size(); i++) {
      const double ms = total_ns_[i] / 1e6 / steps_;
      std::fprintf(f, "    {\"region\": \"%s\", \"ms_per_step\": %.6f, \"share_of_step_pct\": %.3f}%s\n",
                   names_[i].c_str(), ms, wall > 0 ? 100.0 * ms / wall : 0.0,
                   i + 1 == names_.size() ? "" : ",");
    }
    std::fprintf(f, "  ],\n  \"wall_ms_per_step\": %.6f\n}\n", wall);
    std::fclose(f);
  }

  bool on_ = false;
  int steps_ = 0;
  std::vector<std::string> names_;
  std::vector<long long> total_ns_;
};

// Times one host region for as long as it is in scope. Off, the constructor is
// a branch on a bool and the clock is never read.
struct HostScope {
  int id;
  bool on;
  std::chrono::steady_clock::time_point t0;
  explicit HostScope(int id_) : id(id_), on(HostProfile::get().on()) {
    if (on) t0 = std::chrono::steady_clock::now();
  }
  ~HostScope() {
    if (!on) return;
    const auto dt = std::chrono::steady_clock::now() - t0;
    HostProfile::get().add(
        id, std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
  }
};

}  // namespace engine

// The static local means region() runs once per site, not once per step. The
// two-level paste is the usual dance: ## suppresses expansion of its operands,
// so __LINE__ has to go through another macro to become a number.
#define ENGINE_HP_CAT2(a, b) a##b
#define ENGINE_HP_CAT(a, b) ENGINE_HP_CAT2(a, b)
#define ENGINE_HOST_SCOPE(name)                                                     \
  static const int ENGINE_HP_CAT(_ehs_id_, __LINE__) =                               \
      ::engine::HostProfile::get().region(name);                                     \
  ::engine::HostScope ENGINE_HP_CAT(_ehs_, __LINE__)(ENGINE_HP_CAT(_ehs_id_, __LINE__))
