#!/usr/bin/env python3
"""Reference row for the benchmark table: Hugging Face transformers generate().

Runs the same workload the engine's loadgen drives (prompt-length mix, fixed
output length, greedy) and writes bench/results/<tag>.json in the identical
schema, so bench/plot.py renders every row the same way.

Two modes, both honest baselines:
  --batch-size 1   sequential requests, the "no batching at all" floor
  --batch-size N   padded static batching, which is what naive batching costs:
                   every sequence in the batch is padded to the longest prompt
                   and runs until the longest generation finishes

Fairness knobs are pinned to match reference/dump_logits.py: greedy, an explicit
all-ones attention mask, and repetition_penalty forced to 1.0 because
Qwen's generation_config.json sets 1.1 and applies it even under greedy.

  python3 bench/hf_baseline.py --tag hf-b1 --batch-size 1 --num-requests 32
  python3 bench/hf_baseline.py --tag hf-b32 --batch-size 32 --device cuda
"""
import argparse, json, pathlib, random, statistics, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor

MIX = [(128, 0.5), (512, 0.3), (1024, 0.2)]


class StepTimer(LogitsProcessor):
    """generate() calls this once per decoding step, so it gives per-token wall time."""

    def __init__(self):
        self.times = []

    def __call__(self, input_ids, scores):
        self.times.append(time.perf_counter())
        return scores

    def reset(self):
        self.times = []


def make_prompt_ids(n, bank, offset=0):
    return [bank[(i + offset) % len(bank)] for i in range(n)]


def percentile(xs, p):
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    return statistics.quantiles(xs, n=100)[p - 1]


def main(a):
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.float16 if a.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only models must be left-padded

    bank = list(range(1000, 30000, 7))
    lens = [a.prompt_len or random.choices([m[0] for m in MIX], [m[1] for m in MIX])[0]
            for _ in range(a.num_requests)]
    prompts = [make_prompt_ids(n, bank, offset=i * 13) for i, n in enumerate(lens)]

    timer = StepTimer()
    gen_kwargs = dict(max_new_tokens=a.output_len, min_new_tokens=a.output_len, do_sample=False,
                      repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                      eos_token_id=None, pad_token_id=tok.pad_token_id,
                      logits_processor=[timer], use_cache=True)

    # Warm up: kernel autotuning and allocator growth must not land in the measurement.
    with torch.no_grad():
        warm = torch.tensor([make_prompt_ids(64, bank)], device=a.device)
        model.generate(warm, attention_mask=torch.ones_like(warm),
                       **{**gen_kwargs, "max_new_tokens": 8, "min_new_tokens": 8, "logits_processor": []})
    if a.device == "cuda":
        torch.cuda.synchronize()

    ttft, itl, e2e = [], [], []
    generated = 0
    padding_slots = 0
    used_slots = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, a.num_requests, a.batch_size):
            batch = prompts[i:i + a.batch_size]
            longest = max(len(p) for p in batch)
            # Left-pad to the longest prompt in the batch: this padding is exactly
            # the waste that a paged KV cache removes.
            ids = torch.tensor([[tok.pad_token_id] * (longest - len(p)) + p for p in batch], device=a.device)
            mask = torch.tensor([[0] * (longest - len(p)) + [1] * len(p) for p in batch], device=a.device)

            timer.reset()
            t0 = time.perf_counter()
            out = model.generate(ids, attention_mask=mask, **gen_kwargs)
            if a.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            steps = timer.times
            assert len(steps) >= a.output_len, f"expected {a.output_len} steps, saw {len(steps)}"
            # Every sequence in a padded batch shares the batch's timings: they
            # start together and none of them is released early.
            for _ in batch:
                ttft.append(steps[0] - t0)
                e2e.append(t1 - t0)
            itl.extend(b - a_ for a_, b in zip(steps, steps[1:]))

            new_tokens = out.shape[1] - ids.shape[1]
            generated += new_tokens * len(batch)
            total = (longest + new_tokens) * len(batch)
            real = sum(len(p) + new_tokens for p in batch)
            padding_slots += total - real
            used_slots += total

    wall = time.perf_counter() - t_start
    res = {
        "tag": a.tag, "engine": "hf-transformers", "device": a.device, "dtype": str(dtype).split(".")[-1],
        "batch_size": a.batch_size, "concurrency": a.batch_size, "num_requests": a.num_requests,
        "output_len": a.output_len, "rate": 0.0, "shared_prefix": 0, "wall_s": wall,
        "tokens_per_s": generated / wall, "requests_per_s": a.num_requests / wall,
        "ttft_p50_ms": 1000 * percentile(ttft, 50), "ttft_p95_ms": 1000 * percentile(ttft, 95),
        "itl_p50_ms": 1000 * percentile(itl, 50), "itl_p95_ms": 1000 * percentile(itl, 95),
        "e2e_p50_s": percentile(e2e, 50), "e2e_p95_s": percentile(e2e, 95),
        "preemptions": 0, "prefix_hit_blocks": 0, "prefix_total_blocks": 0,
        # Padding is the contiguous-allocation analogue of the engine's KV waste.
        "kv_waste_fraction_last": padding_slots / used_slots if used_slots else 0.0,
    }
    print(json.dumps(res, indent=2))
    if a.tag:
        out_path = pathlib.Path(__file__).parent / "results" / f"{a.tag}.json"
        out_path.parent.mkdir(exist_ok=True)
        runs = json.loads(out_path.read_text()) if out_path.exists() else []
        runs.append(res)
        out_path.write_text(json.dumps(runs, indent=2))
        print("appended to", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=0, help="fixed length instead of the 128/512/1024 mix")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
