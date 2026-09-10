#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <condition_variable>

#include "engine/api.h"
#include "engine/engine.h"

using namespace engine;

namespace {

void usage() {
  std::cerr <<
    "usage:\n"
    "  engine serve    --model DIR [--port 8000] [--host 0.0.0.0] [engine options]\n"
    "  engine generate --model DIR (--prompt TEXT | --chat TEXT | --ids 1,2,3) [--max-tokens N]\n"
    "                  [--temperature T] [--top-p P] [--copies N] [engine options]\n"
    "engine options:\n"
    "  --backend cpu|cuda        (default cpu)\n"
    "  --mode continuous|static  scheduler (default continuous)\n"
    "  --num-blocks N            KV cache blocks (default 512)\n"
    "  --block-size N            tokens per block (default 16)\n"
    "  --no-prefix-cache         disable prefix sharing\n"
    "  --max-num-seqs N          running sequences per step (default 32)\n"
    "  --max-batched-tokens N    prefill+decode tokens per step (default 2048)\n"
    "  --chunk N                 prefill chunk size (default 512)\n"
    "  --seed N\n";
}

struct Args {
  std::string cmd, model, prompt, chat, ids, host = "0.0.0.0";
  int port = 8000, max_tokens = 64, copies = 1;
  float temperature = 0.f, top_p = 1.f;
  EngineConfig ecfg;
};

bool parse(int argc, char** argv, Args& a) {
  if (argc < 2) return false;
  a.cmd = argv[1];
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + k); return argv[++i]; };
    if (k == "--model") a.model = val();
    else if (k == "--port") a.port = std::stoi(val());
    else if (k == "--host") a.host = val();
    else if (k == "--prompt") a.prompt = val();
    else if (k == "--chat") a.chat = val();
    else if (k == "--ids") a.ids = val();
    else if (k == "--max-tokens") a.max_tokens = std::stoi(val());
    else if (k == "--temperature") a.temperature = std::stof(val());
    else if (k == "--top-p") a.top_p = std::stof(val());
    else if (k == "--copies") a.copies = std::stoi(val());
    else if (k == "--backend") a.ecfg.backend = val();
    else if (k == "--mode") a.ecfg.sched.continuous = (val() != "static");
    else if (k == "--num-blocks") a.ecfg.num_blocks = std::stoi(val());
    else if (k == "--block-size") a.ecfg.block_size = std::stoi(val());
    else if (k == "--no-prefix-cache") a.ecfg.prefix_caching = false;
    else if (k == "--max-num-seqs") a.ecfg.sched.max_num_seqs = std::stoi(val());
    else if (k == "--max-batched-tokens") a.ecfg.sched.max_num_batched_tokens = std::stoi(val());
    else if (k == "--chunk") a.ecfg.sched.prefill_chunk_size = std::stoi(val());
    else if (k == "--seed") a.ecfg.seed = std::stoull(val());
    else { std::cerr << "unknown option " << k << "\n"; return false; }
  }
  if (a.model.empty()) { std::cerr << "--model is required\n"; return false; }
  a.ecfg.model_dir = a.model;
  return true;
}

int cmd_generate(const Args& a) {
  Engine eng(a.ecfg);
  eng.start();
  const Tokenizer& tok = eng.tokenizer();
  std::vector<int32_t> prompt;
  if (!a.ids.empty()) { size_t p = 0; while (p < a.ids.size()) { size_t q = a.ids.find(',', p); if (q == std::string::npos) q = a.ids.size(); prompt.push_back(std::stoi(a.ids.substr(p, q - p))); p = q + 1; } }
  else if (!a.chat.empty()) prompt = tok.encode(tok.apply_chat_template({{"user", a.chat}}));
  else prompt = tok.encode(a.prompt);

  std::mutex mu; std::condition_variable cv; int done = 0;
  std::vector<std::string> outputs(a.copies);
  std::vector<std::vector<int32_t>> out_ids(a.copies);
  auto t0 = Clock::now();
  for (int c = 0; c < a.copies; c++) {
    SamplingParams sp; sp.max_tokens = a.max_tokens; sp.temperature = a.temperature; sp.top_p = a.top_p; sp.seed = a.ecfg.seed + c;
    eng.submit(prompt, sp, [&, c](const Sequence&, int32_t t, FinishReason r) {
      if (t >= 0) { out_ids[c].push_back(t); std::string piece = tok.decode_token(t); outputs[c] += piece; if (a.copies == 1) { std::cout << piece << std::flush; } }
      else { std::lock_guard<std::mutex> g(mu); done++; cv.notify_all(); (void)r; }
    });
  }
  { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return done == a.copies; }); }
  double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  if (a.copies == 1) std::cout << "\n";
  else for (int c = 0; c < a.copies; c++) std::cout << "[" << c << "] " << outputs[c] << "\n";
  size_t total = 0; for (auto& v : out_ids) total += v.size();
  std::cerr << "ids:"; for (int32_t t : out_ids[0]) std::cerr << " " << t; std::cerr << "\n";
  std::cerr << "prompt_tokens=" << prompt.size() << " generated=" << total << " time=" << secs << "s tok/s=" << total / secs << "\n";
  eng.stop();
  return 0;
}

std::atomic<bool> g_shutdown{false};
void on_signal(int) { g_shutdown = true; }  // async-signal-safe: just raise the flag

int cmd_serve(const Args& a) {
  Engine eng(a.ecfg);
  std::cerr << "model: " << eng.config().summary() << "\n"
            << "backend=" << a.ecfg.backend << " mode=" << (a.ecfg.sched.continuous ? "continuous" : "static")
            << " blocks=" << a.ecfg.num_blocks << "x" << a.ecfg.block_size << " prefix_cache=" << a.ecfg.prefix_caching << "\n";
  eng.start();
  ApiServer server(eng);
  std::signal(SIGTERM, on_signal); std::signal(SIGINT, on_signal);
  // Graceful drain on SIGTERM: readiness flips to 503 so the load balancer /
  // kubelet stops routing, in-flight sequences run to completion, then the
  // listener closes. Done on a helper thread, never inside the signal handler.
  std::thread drainer([&] {
    while (!g_shutdown) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cerr << "SIGTERM: draining in-flight requests\n";
    server.stop_accepting();
    eng.drain();
    server.stop();
  });
  std::cerr << "listening on http://" << a.host << ":" << a.port << "\n";
  server.listen(a.host, a.port);  // blocks until stop()
  g_shutdown = true;
  drainer.join();
  eng.stop();
  std::cerr << "bye\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  try {
    if (!parse(argc, argv, a)) { usage(); return 2; }
    if (a.cmd == "generate") return cmd_generate(a);
    if (a.cmd == "serve") return cmd_serve(a);
    usage(); return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n"; return 1;
  }
}
