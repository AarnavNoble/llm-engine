#include "engine/api.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "engine/engine.h"

namespace engine {
namespace {

using json = nlohmann::json;

// Per-request channel from the engine thread to the HTTP handler thread.
struct Stream {
  std::mutex mu; std::condition_variable cv;
  std::deque<std::pair<int32_t, FinishReason>> q;   // token >= 0, or -1 with finish reason
  bool closed = false;
  void push(int32_t t, FinishReason r) { { std::lock_guard<std::mutex> g(mu); q.emplace_back(t, r); if (t < 0) closed = true; } cv.notify_one(); }
  bool pop(std::pair<int32_t, FinishReason>& out) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return !q.empty(); });
    out = q.front(); q.pop_front(); return true;
  }
};

// Buffers bytes until they form complete UTF-8 so streamed chunks never
// split a multibyte character.
struct Utf8Buffer {
  std::string pending;
  std::string push(const std::string& bytes) {
    pending += bytes;
    size_t valid = 0;
    while (valid < pending.size()) {
      unsigned char c = static_cast<unsigned char>(pending[valid]);
      size_t len = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
      if (valid + len > pending.size()) break;
      valid += len;
    }
    std::string out = pending.substr(0, valid); pending.erase(0, valid); return out;
  }
};

std::string now_id(const char* prefix) {
  static std::atomic<uint64_t> n{0};
  return std::string(prefix) + "-" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + "-" + std::to_string(n++);
}

SamplingParams parse_params(const json& body) {
  SamplingParams p;
  p.max_tokens = body.value("max_tokens", 128);
  p.temperature = body.value("temperature", 1.0f);
  p.top_p = body.value("top_p", 1.0f);
  p.seed = body.value("seed", 0ull);
  p.ignore_eos = body.value("ignore_eos", false);
  if (body.contains("stop_token_ids")) p.stop_ids = body["stop_token_ids"].get<std::vector<int32_t>>();
  return p;
}

void write_error(httplib::Response& res, int code, const std::string& msg) {
  res.status = code;
  res.set_content(json{{"error", {{"message", msg}, {"type", "invalid_request_error"}}}}.dump(), "application/json");
}

}  // namespace

ApiServer::ApiServer(Engine& engine) : engine_(engine), svr_(std::make_unique<httplib::Server>()) {
  auto& svr = *svr_;
  svr.new_task_queue = [] { return new httplib::ThreadPool(64); };

  svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) { res.set_content("ok", "text/plain"); });
  svr.Get("/readyz", [this](const httplib::Request&, httplib::Response& res) {
    if (engine_.ready() && accepting_) res.set_content("ready", "text/plain"); else { res.status = 503; res.set_content("not ready", "text/plain"); }
  });
  svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(engine_.metrics().render_prometheus(), "text/plain; version=0.0.4");
  });
  svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"object", "list"}, {"data", {{{"id", engine_.engine_config().model_dir}, {"object", "model"}, {"owned_by", "engine"}}}}}.dump(), "application/json");
  });

  // Shared implementation for both completion endpoints.
  auto handle = [this](const httplib::Request& req, httplib::Response& res, bool chat) {
    if (!accepting_ || !engine_.accepting()) { write_error(res, 503, "server is draining"); return; }
    json body;
    try { body = json::parse(req.body); } catch (const std::exception& e) { write_error(res, 400, std::string("invalid JSON: ") + e.what()); return; }
    const Tokenizer& tok = engine_.tokenizer();
    std::vector<int32_t> prompt;
    try {
      if (chat) {
        std::vector<ChatMessage> msgs;
        for (const auto& m : body.at("messages")) msgs.push_back({m.at("role"), m.at("content")});
        prompt = tok.encode(tok.apply_chat_template(msgs));
      } else if (body.contains("prompt_token_ids")) {
        prompt = body["prompt_token_ids"].get<std::vector<int32_t>>();
      } else {
        const auto& p = body.at("prompt");
        if (p.is_array()) prompt = p.get<std::vector<int32_t>>(); else prompt = tok.encode(p.get<std::string>());
      }
    } catch (const std::exception& e) { write_error(res, 400, e.what()); return; }
    if (prompt.empty()) { write_error(res, 400, "empty prompt"); return; }
    const bool stream = body.value("stream", false);
    const std::string id = now_id(chat ? "chatcmpl" : "cmpl");
    const std::string model = body.value("model", engine_.engine_config().model_dir);
    const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    auto ch = std::make_shared<Stream>();
    SamplingParams params = parse_params(body);
    SequencePtr seq = engine_.submit(prompt, params, [ch](const Sequence&, int32_t t, FinishReason r) { ch->push(t, r); });
    if (!seq) { write_error(res, 503, "server is draining"); return; }
    const size_t prompt_tokens = prompt.size();

    if (!stream) {
      std::vector<int32_t> out; FinishReason fr = FinishReason::None;
      for (std::pair<int32_t, FinishReason> m; ch->pop(m);) { if (m.first < 0) { fr = m.second; break; } out.push_back(m.first); }
      std::string text = tok.decode(out);
      json usage{{"prompt_tokens", prompt_tokens}, {"completion_tokens", out.size()}, {"total_tokens", prompt_tokens + out.size()}};
      json choice = chat ? json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", text}}}, {"finish_reason", finish_reason_str(fr)}}
                         : json{{"index", 0}, {"text", text}, {"finish_reason", finish_reason_str(fr)}};
      res.set_content(json{{"id", id}, {"object", chat ? "chat.completion" : "text_completion"}, {"created", created}, {"model", model},
                           {"choices", {choice}}, {"usage", usage}}.dump(), "application/json");
      return;
    }

    // SSE streaming. The content provider runs on the HTTP thread and blocks
    // on the channel; if the client disconnects we abort the sequence.
    auto seq_id = seq->id;
    auto utf8 = std::make_shared<Utf8Buffer>();
    auto first = std::make_shared<bool>(true);
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream",
      [ch, &tok, id, model, created, chat, utf8, first](size_t, httplib::DataSink& sink) {
        std::pair<int32_t, FinishReason> m;
        ch->pop(m);
        std::string payload;
        if (m.first >= 0) {
          std::string piece = utf8->push(tok.decode_token(m.first));
          json delta = chat ? json{{"role", "assistant"}, {"content", piece}} : json{{"text", piece}};
          if (chat && *first) { *first = false; }
          json choice = chat ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                             : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
          payload = "data: " + json{{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created}, {"model", model}, {"choices", {choice}}}.dump() + "\n\n";
          if (!sink.write(payload.data(), payload.size())) return false;
          return true;
        }
        std::string tail = utf8->pending; utf8->pending.clear();
        json choice = chat ? json{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason_str(m.second)}}
                           : json{{"index", 0}, {"text", tail}, {"finish_reason", finish_reason_str(m.second)}};
        payload = "data: " + json{{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created}, {"model", model}, {"choices", {choice}}}.dump() + "\n\ndata: [DONE]\n\n";
        sink.write(payload.data(), payload.size());
        sink.done();
        return true;
      },
      [this, seq_id, ch](bool success) {
        if (!success) engine_.abort(seq_id);   // client went away
      });
  };
  svr.Post("/v1/completions", [handle](const httplib::Request& req, httplib::Response& res) { handle(req, res, false); });
  svr.Post("/v1/chat/completions", [handle](const httplib::Request& req, httplib::Response& res) { handle(req, res, true); });
}

ApiServer::~ApiServer() = default;

void ApiServer::listen(const std::string& host, int port) {
  if (!svr_->listen(host.c_str(), port)) throw std::runtime_error("failed to bind " + host + ":" + std::to_string(port));
}
void ApiServer::stop_accepting() { accepting_ = false; }
void ApiServer::stop() { svr_->stop(); }

}  // namespace engine
