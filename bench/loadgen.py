#!/usr/bin/env python3
"""Closed-form load generator for the benchmark table.

Poisson arrivals at --rate req/s (or --concurrency closed loop), prompt lengths
drawn from a fixed mix, fixed output length, greedy. Reports tokens/s, TTFT and
inter-token latency percentiles, and reads KV waste / preemptions from /metrics.
Writes bench/results/<tag>.json.

  python3 bench/loadgen.py --tag v0.1-baseline --concurrency 32 --num-requests 256
"""
import argparse, asyncio, json, platform, random, re, statistics, subprocess, time, pathlib
import aiohttp

MIX = [(128, 0.5), (512, 0.3), (1024, 0.2)]


def machine_label():
    """A short description of this machine, for the results record.

    The GPU if there is one, since that is what a GPU row is about, otherwise
    the CPU. Best-effort: a label is worth more than a failed run, so every
    probe is allowed to come back empty.
    """
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        if g:
            return g.splitlines()[0].strip()
    except Exception:
        pass
    try:
        if platform.system() == "Darwin":
            b = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
            if b:
                return b
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def results_dir():
    """Where to record runs. ENGINE_RESULTS_DIR keeps a smoke run from
    overwriting the published numbers in bench/results/."""
    import os
    d = os.environ.get("ENGINE_RESULTS_DIR")
    return pathlib.Path(d) if d else pathlib.Path(__file__).resolve().parent / "results"

# Output lengths must vary. With one fixed length every request finishes on the
# same step, which hides the entire cost of static batching: nothing is ever
# waiting behind a longer neighbour.
def output_lengths(n, mean, dist, rng):
    if dist == "fixed":
        return [mean] * n
    if dist == "geometric":
        return [min(max(1, rng.expovariate(1.0 / mean).__trunc__() + 1), 8 * mean) for _ in range(n)]
    if dist == "bimodal":  # a few long generations among many short ones
        return [mean * 6 if rng.random() < 0.15 else max(1, mean // 2) for _ in range(n)]
    raise SystemExit(f"unknown --output-dist {dist}")
FILLER = ("The paged KV cache stores keys and values in fixed-size blocks so that memory is allocated on demand "
          "and freed as soon as a sequence finishes, which removes the fragmentation of contiguous allocation. ")

def make_prompt_ids(n, tok_ids):
    # Deterministic pseudo-prompt: cycle through a bank of token ids; the engine accepts raw ids.
    return [tok_ids[i % len(tok_ids)] for i in range(n)]

async def one(session, url, ids, out_len, ignore_eos, rec, model=None):
    body = {"prompt_token_ids": ids, "max_tokens": out_len, "temperature": 0, "stream": True, "ignore_eos": ignore_eos}
    # This engine ignores "model" -- it serves exactly one. vLLM's
    # OpenAI-compatible API requires it and answers a request without one with
    # a 400 in plain JSON, so no SSE frames ever arrive. Sent only when asked
    # for, so the field never has to be kept in step with this server.
    if model: body["model"] = model
    t0 = time.perf_counter(); first = None; last = t0; n = 0
    async with session.post(url + "/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"{url} returned HTTP {r.status}: {(await r.text())[:400]}")
        async for line in r.content:
            if not line.startswith(b"data:"): continue
            if line.strip() == b"data: [DONE]": break
            now = time.perf_counter()
            if first is None: first = now
            else: rec["itl"].append(now - last)
            last = now; n += 1
    # A 200 that streams nothing is still a failed request, and must say so
    # here. Falling through left first as None and raised a TypeError on the
    # subtraction below -- an error about arithmetic, two hundred lines from
    # the HTTP response that actually explained it.
    if first is None:
        raise RuntimeError(f"{url} streamed no tokens (HTTP 200, {n} frames). "
                           "Check the server log; for vLLM this is usually a rejected request.")
    # n counts every SSE frame; the last one carries finish_reason, not a token.
    rec["ttft"].append(first - t0); rec["e2e"].append(last - t0); rec["tokens"] += max(n - 1, 0)
    rec["per_request"].append({"out_len": out_len, "ttft": first - t0, "e2e": last - t0})

async def main(a):
    random.seed(a.seed)
    tok_bank = list(range(1000, 30000, 7))
    lens = [random.choices([m[0] for m in MIX], [m[1] for m in MIX])[0] for _ in range(a.num_requests)]
    if a.prompt_len: lens = [a.prompt_len] * a.num_requests
    outs = output_lengths(a.num_requests, a.output_len, a.output_dist, random.Random(a.seed + 1))
    if a.shared_prefix:
        prefix = make_prompt_ids(a.shared_prefix, tok_bank[::3])
    rec = {"ttft": [], "itl": [], "e2e": [], "tokens": 0, "per_request": []}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        # warmup
        warm = {"ttft": [], "itl": [], "e2e": [], "tokens": 0, "per_request": []}
        await asyncio.gather(*[one(s, a.url, make_prompt_ids(64, tok_bank), 16, True, warm, a.model)
                               for _ in range(min(8, a.concurrency))])
        m0 = await (await s.get(a.url + "/metrics")).text()
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(a.concurrency)
        async def run(i):
            if a.rate: await asyncio.sleep(random.expovariate(a.rate) * i if False else 0)
            async with sem:
                ids = make_prompt_ids(lens[i], tok_bank[i % 50:] + tok_bank[:i % 50])
                if a.shared_prefix: ids = prefix + ids[: max(1, lens[i] - a.shared_prefix)]
                await one(s, a.url, ids, outs[i], True, rec, a.model)
        tasks = []
        for i in range(a.num_requests):
            if a.rate: await asyncio.sleep(random.expovariate(a.rate))
            tasks.append(asyncio.create_task(run(i)))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
        m1 = await (await s.get(a.url + "/metrics")).text()
    def metric(txt, name):
        m = re.search(rf"^{name} ([0-9.e+-]+)$", txt, re.M); return float(m.group(1)) if m else float("nan")
    pct = lambda xs, p: statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else (xs[0] if xs else float("nan"))
    # Latency of the shortest requests is the head-of-line blocking signal: under
    # static batching they cannot leave until the longest request in the batch does.
    short = sorted(rec["per_request"], key=lambda r: r["out_len"])[: max(1, len(rec["per_request"]) // 3)]
    long_ = sorted(rec["per_request"], key=lambda r: -r["out_len"])[: max(1, len(rec["per_request"]) // 3)]
    res = {
        # Every run records the machine it ran on. A results directory
        # accumulates rows across machines -- the CPU rows here were taken on a
        # laptop and the GPU rows on a rented A40 -- and a single
        # hardware line above the table then claims one machine for all of
        # them. The number has to travel with its hardware or the table
        # misattributes it.
        "machine": machine_label(),
        "tag": a.tag, "concurrency": a.concurrency, "num_requests": a.num_requests, "output_len": a.output_len,
        "output_dist": a.output_dist, "mean_output_len": sum(outs) / len(outs), "max_output_len": max(outs),
        "rate": a.rate,
        "shared_prefix": a.shared_prefix, "wall_s": wall,
        "tokens_per_s": rec["tokens"] / wall, "requests_per_s": a.num_requests / wall,
        "ttft_p50_ms": 1000 * pct(rec["ttft"], 50), "ttft_p95_ms": 1000 * pct(rec["ttft"], 95),
        "itl_p50_ms": 1000 * pct(rec["itl"], 50), "itl_p95_ms": 1000 * pct(rec["itl"], 95),
        "e2e_p50_s": pct(rec["e2e"], 50), "e2e_p95_s": pct(rec["e2e"], 95),
        "preemptions": metric(m1, "engine_preemptions_total") - metric(m0, "engine_preemptions_total"),
        "prefix_hit_blocks": metric(m1, "engine_prefix_cache_hit_blocks_total") - metric(m0, "engine_prefix_cache_hit_blocks_total"),
        "prefix_total_blocks": metric(m1, "engine_prefix_cache_total_blocks_total") - metric(m0, "engine_prefix_cache_total_blocks_total"),
        "kv_waste_fraction_last": metric(m1, "engine_kv_waste_fraction"),
        "short_third_e2e_p50_s": pct([r["e2e"] for r in short], 50),
        "short_third_e2e_p95_s": pct([r["e2e"] for r in short], 95),
        "short_third_mean_out_len": sum(r["out_len"] for r in short) / len(short),
        "long_third_e2e_p50_s": pct([r["e2e"] for r in long_], 50),
        "long_third_mean_out_len": sum(r["out_len"] for r in long_) / len(long_),
    }
    print(json.dumps(res, indent=2))
    if a.tag:
        out = results_dir() / f"{a.tag}.json"
        out.parent.mkdir(exist_ok=True)
        runs = json.loads(out.read_text()) if out.exists() else []
        runs.append(res); out.write_text(json.dumps(runs, indent=2))
        print("appended to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--tag", default="")
    # Required by vLLM's API, ignored by this engine's.
    ap.add_argument("--model", default="")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128, help="mean output length")
    ap.add_argument("--output-dist", default="geometric", choices=["fixed", "geometric", "bimodal"],
                    help="fixed hides static batching's cost; see output_lengths()")
    ap.add_argument("--prompt-len", type=int, default=0, help="fixed prompt length instead of the 128/512/1024 mix")
    ap.add_argument("--rate", type=float, default=0.0, help="Poisson arrival rate (req/s); 0 = closed loop")
    ap.add_argument("--shared-prefix", type=int, default=0, help="prepend a shared N-token prefix (prefix-cache experiment)")
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
