#!/usr/bin/env python3
"""Closed-form load generator for the benchmark table.

Poisson arrivals at --rate req/s (or --concurrency closed loop), prompt lengths
drawn from a fixed mix, fixed output length, greedy. Reports tokens/s, TTFT and
inter-token latency percentiles, and reads KV waste / preemptions from /metrics.
Writes bench/results/<tag>.json.

  python3 bench/loadgen.py --tag v0.1-baseline --concurrency 32 --num-requests 256
"""
import argparse, asyncio, json, random, statistics, time, pathlib, re
import aiohttp

MIX = [(128, 0.5), (512, 0.3), (1024, 0.2)]
FILLER = ("The paged KV cache stores keys and values in fixed-size blocks so that memory is allocated on demand "
          "and freed as soon as a sequence finishes, which removes the fragmentation of contiguous allocation. ")

def make_prompt_ids(n, tok_ids):
    # Deterministic pseudo-prompt: cycle through a bank of token ids; the engine accepts raw ids.
    return [tok_ids[i % len(tok_ids)] for i in range(n)]

async def one(session, url, ids, out_len, ignore_eos, rec):
    body = {"prompt_token_ids": ids, "max_tokens": out_len, "temperature": 0, "stream": True, "ignore_eos": ignore_eos}
    t0 = time.perf_counter(); first = None; last = t0; n = 0
    async with session.post(url + "/v1/completions", json=body) as r:
        async for line in r.content:
            if not line.startswith(b"data:"): continue
            if line.strip() == b"data: [DONE]": break
            now = time.perf_counter()
            if first is None: first = now
            else: rec["itl"].append(now - last)
            last = now; n += 1
    rec["ttft"].append(first - t0); rec["e2e"].append(last - t0); rec["tokens"] += n - 1  # last chunk is the finish frame

async def main(a):
    random.seed(a.seed)
    tok_bank = list(range(1000, 30000, 7))
    lens = [random.choices([m[0] for m in MIX], [m[1] for m in MIX])[0] for _ in range(a.num_requests)]
    if a.prompt_len: lens = [a.prompt_len] * a.num_requests
    if a.shared_prefix:
        prefix = make_prompt_ids(a.shared_prefix, tok_bank[::3])
    rec = {"ttft": [], "itl": [], "e2e": [], "tokens": 0}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        # warmup
        await asyncio.gather(*[one(s, a.url, make_prompt_ids(64, tok_bank), 16, True, {"ttft": [], "itl": [], "e2e": [], "tokens": 0}) for _ in range(min(8, a.concurrency))])
        m0 = await (await s.get(a.url + "/metrics")).text()
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(a.concurrency)
        async def run(i):
            if a.rate: await asyncio.sleep(random.expovariate(a.rate) * i if False else 0)
            async with sem:
                ids = make_prompt_ids(lens[i], tok_bank[i % 50:] + tok_bank[:i % 50])
                if a.shared_prefix: ids = prefix + ids[: max(1, lens[i] - a.shared_prefix)]
                await one(s, a.url, ids, a.output_len, True, rec)
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
    res = {
        "tag": a.tag, "concurrency": a.concurrency, "num_requests": a.num_requests, "output_len": a.output_len, "rate": a.rate,
        "shared_prefix": a.shared_prefix, "wall_s": wall,
        "tokens_per_s": rec["tokens"] / wall, "requests_per_s": a.num_requests / wall,
        "ttft_p50_ms": 1000 * pct(rec["ttft"], 50), "ttft_p95_ms": 1000 * pct(rec["ttft"], 95),
        "itl_p50_ms": 1000 * pct(rec["itl"], 50), "itl_p95_ms": 1000 * pct(rec["itl"], 95),
        "e2e_p50_s": pct(rec["e2e"], 50), "e2e_p95_s": pct(rec["e2e"], 95),
        "preemptions": metric(m1, "engine_preemptions_total") - metric(m0, "engine_preemptions_total"),
        "prefix_hit_blocks": metric(m1, "engine_prefix_cache_hit_blocks_total") - metric(m0, "engine_prefix_cache_hit_blocks_total"),
        "prefix_total_blocks": metric(m1, "engine_prefix_cache_total_blocks_total") - metric(m0, "engine_prefix_cache_total_blocks_total"),
        "kv_waste_fraction_last": metric(m1, "engine_kv_waste_fraction"),
    }
    print(json.dumps(res, indent=2))
    if a.tag:
        out = pathlib.Path(__file__).parent / "results" / f"{a.tag}.json"
        out.parent.mkdir(exist_ok=True)
        runs = json.loads(out.read_text()) if out.exists() else []
        runs.append(res); out.write_text(json.dumps(runs, indent=2))
        print("appended to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--tag", default="")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=0, help="fixed prompt length instead of the 128/512/1024 mix")
    ap.add_argument("--rate", type=float, default=0.0, help="Poisson arrival rate (req/s); 0 = closed loop")
    ap.add_argument("--shared-prefix", type=int, default=0, help="prepend a shared N-token prefix (prefix-cache experiment)")
    ap.add_argument("--seed", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
