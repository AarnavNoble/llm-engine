#!/usr/bin/env python3
"""vLLM reference row, measured the same way as the engine's own rows.

vLLM serves an OpenAI-compatible API, so bench/loadgen.py drives it unchanged.
This script only starts the server with settings matched to the engine's, waits
for readiness, and records the result under the same schema.

Losing to vLLM on a 0.5B model is the expected outcome and is the reason to
measure it: a comparison nobody ran is worth less than one that is unflattering.
Matched where it matters (model, dtype, KV capacity, max concurrency, greedy
decoding, identical prompt and output distributions) and documented where it
cannot be (vLLM's kernels, its CUDA graphs, its scheduler internals).

  pip install vllm
  python3 bench/vllm_baseline.py --tag vllm --num-requests 256
"""
import argparse
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(base, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(base + "/v1/models", timeout=3).status == 200:
                return True
        except Exception:
            time.sleep(2)
    return False


def main(a):
    if shutil.which("vllm") is None and not a.url:
        sys.exit("vllm not installed (pip install vllm), or pass --url for a server you already run")

    port = a.port or free_port()
    base = a.url or f"http://127.0.0.1:{port}"
    proc = None
    if not a.url:
        # num-gpu-blocks-override pins the KV capacity to the engine's, so the
        # comparison is at equal memory rather than equal hardware.
        cmd = [
            "vllm", "serve", a.model,
            "--port", str(port),
            "--dtype", "float16",
            "--max-num-seqs", str(a.max_num_seqs),
            "--max-num-batched-tokens", str(a.max_batched_tokens),
            "--num-gpu-blocks-override", str(a.num_blocks),
            "--block-size", "16",
            "--disable-log-requests",
        ]
        if not a.prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        print("$", " ".join(cmd))
        log = open("/tmp/vllm-baseline.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    try:
        if not wait_ready(base, a.startup_timeout):
            sys.exit("vLLM did not become ready; see /tmp/vllm-baseline.log")
        # Same load generator, same workload, same result schema as every other row.
        loadgen = [
            sys.executable, str(ROOT / "bench/loadgen.py"),
            "--url", base, "--tag", a.tag,
            "--concurrency", str(a.concurrency),
            "--num-requests", str(a.num_requests),
            "--output-len", str(a.output_len),
            "--output-dist", a.output_dist,
        ]
        for _ in range(a.runs):
            subprocess.run(loadgen, check=True, cwd=ROOT)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()

    out = ROOT / "bench/results" / f"{a.tag}.json"
    if out.exists():
        runs = json.loads(out.read_text())
        # Record what could not be matched, so the row is read with the caveat.
        for r in runs[-a.runs:]:
            r["engine"] = "vllm"
            r["matched"] = ["model", "dtype", "kv_blocks", "block_size", "max_num_seqs",
                            "max_num_batched_tokens", "greedy", "workload"]
            r["not_matched"] = ["attention kernels", "CUDA graphs", "scheduler internals",
                                "chunked prefill policy"]
        out.write_text(json.dumps(runs, indent=2))
        print(f"recorded {a.runs} run(s) in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tag", default="vllm")
    ap.add_argument("--url", default="", help="benchmark an already-running server instead")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--output-dist", default="geometric")
    ap.add_argument("--num-blocks", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-batched-tokens", type=int, default=4096)
    ap.add_argument("--prefix-caching", action="store_true", default=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--startup-timeout", type=int, default=600)
    main(ap.parse_args())
