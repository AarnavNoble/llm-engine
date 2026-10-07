#pragma once
// A persistent worker pool for the CPU backend's attention loop.
//
// Attention is the one part of the forward pass that is not a GEMM, so BLAS
// does not thread it. Each (token, head) pair reads the KV cache and writes its
// own slice of the output, so the work is independent and the result is
// deterministic regardless of how it is split.
//
// The pool is persistent because the loop runs once per layer per step: at 24
// layers and hundreds of steps, spawning threads each time would cost more than
// the work itself.
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace engine {

class ThreadPool {
 public:
  // threads <= 1 runs everything on the calling thread, which keeps the
  // single-threaded path available for debugging and for tiny batches.
  explicit ThreadPool(int threads) : n_(threads > 1 ? threads : 1) {
    if (n_ == 1) return;
    workers_.reserve(static_cast<size_t>(n_) - 1);
    for (int i = 1; i < n_; i++) workers_.emplace_back([this, i] { worker(i); });
  }

  ~ThreadPool() {
    if (n_ == 1) return;
    {
      std::lock_guard<std::mutex> g(m_);
      stop_ = true;
      generation_++;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return n_; }

  // Runs body(begin, end) over a partition of [0, count) and returns once every
  // part is done. Blocking and reentrant-unsafe: one parallel_for at a time.
  void parallel_for(int64_t count, const std::function<void(int64_t, int64_t)>& body) {
    if (count <= 0) return;
    if (n_ == 1 || count == 1) { body(0, count); return; }
    {
      std::lock_guard<std::mutex> g(m_);
      count_ = count;
      body_ = &body;
      remaining_ = n_ - 1;
      generation_++;
    }
    cv_.notify_all();
    run_part(0);  // the calling thread takes the first part
    std::unique_lock<std::mutex> lk(m_);
    done_cv_.wait(lk, [this] { return remaining_ == 0; });
    body_ = nullptr;
  }

 private:
  // Contiguous ranges rather than strided ones: a token's attention cost grows
  // with its position, but contiguous blocks keep cache behaviour sane and the
  // imbalance is small compared to the cost of false sharing.
  void run_part(int idx) {
    const int64_t per = (count_ + n_ - 1) / n_;
    const int64_t begin = per * idx;
    const int64_t end = std::min(count_, begin + per);
    if (begin < end) (*body_)(begin, end);
  }

  void worker(int idx) {
    uint64_t seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&] { return generation_ != seen; });
      seen = generation_;
      if (stop_) return;
      lk.unlock();
      run_part(idx);
      lk.lock();
      if (--remaining_ == 0) done_cv_.notify_one();
    }
  }

  const int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  uint64_t generation_ = 0;
  int remaining_ = 0;
  bool stop_ = false;
  int64_t count_ = 0;
  const std::function<void(int64_t, int64_t)>* body_ = nullptr;
};

// Threads to use for CPU work: ENGINE_THREADS if set, else the hardware count.
inline int default_thread_count() {
  if (const char* e = std::getenv("ENGINE_THREADS")) {
    const int n = std::atoi(e);
    if (n > 0) return n;
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return hw ? static_cast<int>(hw) : 1;
}

}  // namespace engine
