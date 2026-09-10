#pragma once
// OpenAI-compatible HTTP API on cpp-httplib:
//   POST /v1/completions        prompt (string | token id array), stream via SSE
//   POST /v1/chat/completions   messages[], stream via SSE
//   GET  /v1/models  /metrics  /healthz  /readyz
#include <memory>
#include <string>

namespace httplib { class Server; }

namespace engine {

class Engine;

class ApiServer {
 public:
  explicit ApiServer(Engine& engine);
  ~ApiServer();
  void listen(const std::string& host, int port);  // blocks
  void stop_accepting();  // readiness -> 503, new generations rejected
  void stop();

 private:
  Engine& engine_;
  std::unique_ptr<httplib::Server> svr_;
  bool accepting_ = true;
};

}  // namespace engine
