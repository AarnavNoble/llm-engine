#!/usr/bin/env python3
"""Does chunked prefill actually protect decode latency?

A long prompt has to be prefilled before it can generate. If the whole prompt
goes into one forward pass, every sequence already decoding waits for that pass
to finish and sees one enormous gap between tokens. Splitting the prompt into
chunks bounds that gap at the cost of a slightly later first token for the
arriving request.

The experiment: hold N streams decoding steadily, inject one long prompt, and
measure the inter-token latency of the streams around the injection. Repeated
for several --chunk sizes against otherwise identical servers.

This is a property of how a step is composed, not of batched GEMMs, so it is
measurable on the CPU backend.

  python3 scripts/bench_chunked_prefill.py --chunks 2048 512 128
"""
import argparse, json, pathlib, socket, statistics, subprocess, sys, threading, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stream_tokens(base, body, out):
    """Record a wall-clock timestamp for every streamed token."""
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    stamps = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            stamps.append(time.perf_counter())
    out.append({"t0": t0, "stamps": stamps})


def pct(xs, p):
    if not xs:
        return float("nan")
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else xs[0]


def run_one(a, chunk):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(f"/tmp/engine-chunk-{chunk}.log", "w")
    proc = subprocess.Popen(
        [a.binary, "serve", "--model", a.model, "--backend", a.backend, "--host", "127.0.0.1",
         "--port", str(port), "--num-blocks", "2048", "--max-num-seqs", "16",
         "--max-batched-tokens", str(max(chunk, 2048)), "--chunk", str(chunk),
         "--no-prefix-cache"],  # prefix reuse would hide the prefill cost
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    try:
        for _ in range(a.startup_timeout):
            try:
                if urllib.request.urlopen(base + "/readyz", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            sys.exit(f"server did not start for chunk={chunk}")

        results, threads = [], []
        bank = [1000 + 7 * i for i in range(4096)]
        # Steady decoders: short prompts, long generations.
        for i in range(a.streams):
            body = {"prompt_token_ids": bank[i * 32:(i * 32) + 32], "max_tokens": a.decode_tokens,
                    "temperature": 0, "ignore_eos": True, "stream": True}
            t = threading.Thread(target=stream_tokens, args=(base, body, results))
            t.start()
            threads.append(t)

        time.sleep(a.inject_after)
        inject = []
        long_body = {"prompt_token_ids": bank[100:100 + a.prompt_tokens], "max_tokens": 1,
                     "temperature": 0, "ignore_eos": True, "stream": True}
        it = threading.Thread(target=stream_tokens, args=(base, long_body, inject))
        inject_at = time.perf_counter()
        it.start()
        for t in threads:
            t.join()
        it.join()

        # Inter-token gaps of the steady decoders, split around the injection.
        before, after = [], []
        for r in results[: a.streams]:
            for prev, nxt in zip(r["stamps"], r["stamps"][1:]):
                (after if prev >= inject_at else before).append(nxt - prev)
        all_gaps = before + after
        ttft_long = (inject[0]["stamps"][0] - inject[0]["t0"]) if inject and inject[0]["stamps"] else float("nan")
        return {
            "chunk": chunk, "streams": a.streams, "prompt_tokens": a.prompt_tokens,
            "itl_p50_ms": 1000 * pct(all_gaps, 50), "itl_p95_ms": 1000 * pct(all_gaps, 95),
            "itl_max_ms": 1000 * max(all_gaps) if all_gaps else float("nan"),
            "itl_p50_before_ms": 1000 * pct(before, 50),
            "itl_max_after_ms": 1000 * max(after) if after else float("nan"),
            "long_prompt_ttft_s": ttft_long,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main(a):
    rows = [run_one(a, c) for c in a.chunks]
    print(f"\n{a.streams} streams decoding, one {a.prompt_tokens}-token prompt injected mid-flight\n")
    print(f"| {'chunk':>6} | {'ITL p50':>9} | {'ITL p95':>9} | {'worst gap':>10} | {'long TTFT':>10} |")
    print(f"|{'-'*8}|{'-'*11}|{'-'*11}|{'-'*12}|{'-'*12}|")
    for r in rows:
        print(f"| {r['chunk']:>6} | {r['itl_p50_ms']:>7.0f}ms | {r['itl_p95_ms']:>7.0f}ms | "
              f"{r['itl_max_ms']:>8.0f}ms | {r['long_prompt_ttft_s']:>8.2f}s |")
    out = ROOT / "bench/results/chunked-prefill.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./build/engine")
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--chunks", type=int, nargs="+", default=[2048, 512, 128])
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--decode-tokens", type=int, default=40)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--inject-after", type=float, default=3.0)
    ap.add_argument("--startup-timeout", type=int, default=300)
    main(ap.parse_args())
