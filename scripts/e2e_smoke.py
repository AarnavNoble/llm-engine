#!/usr/bin/env python3
"""End-to-end smoke test against a real server process.

Starts `engine serve`, exercises the HTTP contract, and shuts it down with
SIGTERM to prove the drain path. This covers what the C++ unit tests cannot:
the API surface, SSE framing, and process lifecycle.

  python3 scripts/e2e_smoke.py                       # CPU backend
  python3 scripts/e2e_smoke.py --backend cuda        # on a GPU box

Exits non-zero on the first failure. Not part of CI: it needs model weights.
"""
import argparse, json, os, signal, socket, subprocess, sys, time, urllib.error, urllib.request

PASS, FAIL = "\033[32mok\033[0m", "\033[31mFAIL\033[0m"
failures = []


def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL}  {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def post(url, body, stream=False, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else (r.status, json.loads(r.read()))


def post_status(url, body):
    """POST returning the status code even for errors."""
    try:
        status, _ = post(url, body)
        return status
    except urllib.error.HTTPError as e:
        return e.code


def get(url, timeout=10):
    r = urllib.request.urlopen(url, timeout=timeout)
    return r.status, r.read().decode()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(a):
    port = a.port or free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [a.binary, "serve", "--model", a.model, "--backend", a.backend,
           "--host", "127.0.0.1", "--port", str(port), "--num-blocks", str(a.num_blocks)]
    print("$", " ".join(cmd))
    log = open(a.log, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    try:
        print("\nwaiting for readiness (weights load first)")
        deadline = time.time() + a.startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                sys.exit(f"server exited early with code {proc.returncode}; see {a.log}")
            try:
                if get(base + "/readyz", timeout=2)[0] == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            sys.exit(f"server never became ready; see {a.log}")
        check("GET /healthz", get(base + "/healthz")[0] == 200)
        check("GET /v1/models lists the loaded model",
              a.model in get(base + "/v1/models")[1])

        print("\ncompletions")
        status, body = post(base + "/v1/completions",
                            {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
        text = body["choices"][0]["text"]
        check("non-streaming completion returns text", status == 200 and len(text) > 0, repr(text))
        check("usage accounting is consistent",
              body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"],
              str(body["usage"]))
        check("respects max_tokens", body["usage"]["completion_tokens"] <= 8,
              str(body["usage"]["completion_tokens"]))
        check("greedy decoding is coherent", "Paris" in text, repr(text))
        check("finish_reason is length when truncated", body["choices"][0]["finish_reason"] == "length",
              body["choices"][0]["finish_reason"])

        _, body2 = post(base + "/v1/completions",
                        {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
        check("greedy is deterministic across requests", body2["choices"][0]["text"] == text)

        _, ids_body = post(base + "/v1/completions",
                           {"prompt_token_ids": [785, 6722, 315, 9625, 374], "max_tokens": 8, "temperature": 0})
        check("prompt_token_ids path matches the text path (same tokens)",
              ids_body["choices"][0]["text"] == text, repr(ids_body["choices"][0]["text"]))

        print("\nsampling parameters")
        seeded = [post(base + "/v1/completions",
                       {"prompt": "Write one word:", "max_tokens": 6, "temperature": 1.0, "seed": 4242})[1]
                  ["choices"][0]["text"] for _ in range(2)]
        check("a fixed seed reproduces the sample", seeded[0] == seeded[1], repr(seeded))
        _, ig = post(base + "/v1/completions",
                     {"prompt": "hi", "max_tokens": 12, "temperature": 0, "ignore_eos": True})
        check("ignore_eos runs to max_tokens", ig["usage"]["completion_tokens"] == 12,
              str(ig["usage"]["completion_tokens"]))

        print("\nchat completions")
        status, chat = post(base + "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Name one GPU vendor."}],
                             "max_tokens": 24, "temperature": 0})
        msg = chat["choices"][0]["message"]
        check("chat returns an assistant message", status == 200 and msg["role"] == "assistant" and msg["content"])
        check("chat template applied (model answers the question)",
              any(v in msg["content"] for v in ("NVIDIA", "AMD", "Intel")), repr(msg["content"][:80]))
        check("stop token is not echoed into the content", "<|im_end|>" not in msg["content"])

        print("\nSSE streaming")
        r = post(base + "/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "Count: one two three"}],
                  "max_tokens": 16, "temperature": 0, "stream": True}, stream=True)
        check("content-type is text/event-stream", "text/event-stream" in r.headers.get("content-type", ""),
              r.headers.get("content-type"))
        frames, done, pieces = 0, False, ""
        for raw in r:
            line = raw.decode().strip()
            if not line:
                continue
            check("every frame is an SSE data line", line.startswith("data: "), line[:40]) if frames == 0 else None
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done = True
                break
            frames += 1
            obj = json.loads(payload)
            pieces += obj["choices"][0]["delta"].get("content", "")
            last = obj
        check("stream produced multiple chunks", frames > 1, str(frames))
        check("stream terminated with [DONE]", done)
        check("final chunk carries a finish_reason", bool(last["choices"][0]["finish_reason"]),
              json.dumps(last["choices"][0]))
        check("streamed text is non-empty and valid UTF-8", len(pieces) > 0 and pieces == pieces.encode().decode())

        print("\nerror handling")
        check("empty prompt is rejected with 400",
              post_status(base + "/v1/completions", {"prompt": "", "max_tokens": 4}) == 400)
        check("missing prompt is rejected with 400",
              post_status(base + "/v1/completions", {"max_tokens": 4}) == 400)
        try:
            req = urllib.request.Request(base + "/v1/completions", data=b"{not json",
                                         headers={"content-type": "application/json"})
            code = urllib.request.urlopen(req, timeout=10).status
        except urllib.error.HTTPError as e:
            code = e.code
        check("malformed JSON is rejected with 400", code == 400, str(code))

        print("\nmetrics")
        status, metrics = get(base + "/metrics")
        check("GET /metrics returns Prometheus text", status == 200 and "# TYPE" in metrics)
        for name in ("engine_requests_total", "engine_generated_tokens_total", "engine_queue_depth",
                     "engine_kv_blocks_used", "engine_kv_waste_fraction", "engine_ttft_seconds_count",
                     "engine_inter_token_seconds_count", "engine_prefix_cache_hit_blocks_total"):
            check(f"exposes {name}", f"\n{name} " in metrics or f"\n{name}_bucket" in metrics)
        served = [l for l in metrics.splitlines() if l.startswith("engine_requests_total ")]
        check("request counter advanced", served and int(float(served[0].split()[1])) >= 8, str(served))
        check("all KV blocks returned to the pool after idle",
              "\nengine_kv_blocks_used 0\n" in metrics,
              [l for l in metrics.splitlines() if l.startswith("engine_kv_blocks_used")])

        print("\nprefix caching")
        shared = "You are a helpful assistant. " * 40  # long enough to fill several blocks
        post(base + "/v1/completions", {"prompt": shared + "First question?", "max_tokens": 4, "temperature": 0})
        before = get(base + "/metrics")[1]
        post(base + "/v1/completions", {"prompt": shared + "Second question?", "max_tokens": 4, "temperature": 0})
        after = get(base + "/metrics")[1]

        def metric(text, name):
            for l in text.splitlines():
                if l.startswith(name + " "):
                    return float(l.split()[1])
            return 0.0
        hits = metric(after, "engine_prefix_cache_hit_blocks_total") - metric(before, "engine_prefix_cache_hit_blocks_total")
        check("a shared prompt prefix produced cache hits", hits > 0, f"{hits} blocks")

        print("\ngraceful drain on SIGTERM")
        import threading
        result = {}

        def long_request():
            try:
                result["body"] = post(base + "/v1/completions",
                                      {"prompt": "Count slowly:", "max_tokens": 24,
                                       "temperature": 0, "ignore_eos": True}, timeout=180)[1]
            except Exception as e:
                result["error"] = repr(e)
        t = threading.Thread(target=long_request)
        t.start()
        time.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.4)
        try:
            ready_code = get(base + "/readyz", timeout=5)[0]
        except urllib.error.HTTPError as e:
            ready_code = e.code
        check("/readyz turns 503 while draining", ready_code == 503, str(ready_code))
        check("new requests are refused with 503 while draining",
              post_status(base + "/v1/completions", {"prompt": "x", "max_tokens": 1}) == 503)
        t.join(timeout=180)
        check("the in-flight request finished normally",
              "body" in result and result["body"]["usage"]["completion_tokens"] == 24,
              result.get("error") or json.dumps(result.get("body", {}).get("usage", {})))
        check("server exited 0 after draining", proc.wait(timeout=60) == 0, str(proc.returncode))
        proc = None
    finally:
        if proc and proc.poll() is None:
            proc.kill()
        log.close()

    print()
    if failures:
        print(f"\033[31m{len(failures)} check(s) failed:\033[0m " + ", ".join(failures))
        print(f"server log: {a.log}")
        sys.exit(1)
    print("\033[32mall checks passed\033[0m")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./build/engine")
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--num-blocks", type=int, default=256)
    ap.add_argument("--startup-timeout", type=int, default=300)
    ap.add_argument("--log", default=os.path.join(os.getenv("TMPDIR", "/tmp"), "engine-e2e.log"))
    main(ap.parse_args())
