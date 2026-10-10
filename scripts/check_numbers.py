#!/usr/bin/env python3
"""Verify that every headline figure quoted in prose matches the results files.

Prose drifts. A number gets measured, written into the README and the write-up,
and then the code underneath it changes and the figure quietly becomes a lie.
bench/report.py solves that for generated tables; this solves it for the
hand-written claims that give those tables their meaning.

Each claim names the file it appears in, how to compute the figure from
bench/results/, and the exact string that must therefore be present. A failure
prints the claim, the current value and the file to fix.

  python3 scripts/check_numbers.py
"""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench/results"


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def last(name, key):
    runs = load(name)
    if not runs:
        return None
    vals = [r[key] for r in runs if key in r]
    return statistics.median(vals) if vals else None


def kv(strategy, key):
    d = load("kv-waste")
    if not d:
        return None
    # Two contiguous rows share a strategy string (static and continuous); the
    # continuous one is second, and is the fair baseline to quote.
    rows = [r for r in d["results"] if r["strategy"] == strategy]
    return rows[-1][key] if rows else None


def sweep(blocks, key):
    d = load("kv-waste")
    if not d or "pressure_sweep" not in d:
        return None
    for p in d["pressure_sweep"]:
        if p["blocks"] == blocks:
            return p[key]
    return None


def chunk(size, key):
    runs = load("chunked-prefill")
    if not runs:
        return None
    vals = [r[key] for r in runs if r["chunk"] == size]
    return statistics.median(vals) if vals else None


def prefix_ratio():
    on = load("prefix-on")
    if not on:
        return None
    hits = statistics.median(r["prefix_hit_blocks"] for r in on)
    total = statistics.median(r["prefix_total_blocks"] for r in on)
    return 100 * hits / total if total else None


def ratio(a, b):
    return a / b if a is not None and b else None


# Each claim: which docs must contain it, the computed value, and how it is
# written. Keep the formatting identical to the prose.
CLAIMS = [
    ("paged slot utilization", ["README.md", "docs/benchmarks.md", "docs/writeup.md"],
     lambda: kv("paged, continuous batching", "slot_utilization") * 100, "{:.1f}%"),
    ("contiguous slot utilization", ["README.md", "docs/benchmarks.md", "docs/writeup.md"],
     lambda: kv("contiguous (reserve prompt + max_tokens)", "slot_utilization") * 100, "{:.1f}%"),
    ("paged sequences decoding", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda: kv("paged, continuous batching", "mean_active_seqs"), "{:.1f}"),
    ("static sequences decoding", ["docs/benchmarks.md"],
     lambda: kv("paged, static batching", "mean_active_seqs"), "{:.1f}"),
    ("continuous tokens/s", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda: last("cpu-continuous", "tokens_per_s"), "{:.1f}"),
    ("static tokens/s", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda: last("cpu-static", "tokens_per_s"), "{:.1f}"),
    ("continuous TTFT p50 ms", ["README.md", "docs/benchmarks.md", "docs/writeup.md"],
     lambda: last("cpu-continuous", "ttft_p50_ms"), "{:,.0f}"),
    ("static TTFT p50 ms", ["README.md", "docs/benchmarks.md", "docs/writeup.md"],
     lambda: last("cpu-static", "ttft_p50_ms"), "{:,.0f}"),
    ("prefix cache hit ratio", ["README.md", "docs/benchmarks.md", "docs/writeup.md"],
     prefix_ratio, "{:.1f}%"),
    ("prefix TTFT speedup", ["README.md", "docs/benchmarks.md"],
     lambda: ratio(last("prefix-off", "ttft_p50_ms"), last("prefix-on", "ttft_p50_ms")), "{:.1f}x"),
    ("unchunked worst gap ms", ["docs/benchmarks.md"],
     lambda: chunk(2048, "itl_max_ms"), "{:,.0f}"),
    ("chunk 128 worst gap ms", ["docs/benchmarks.md"],
     lambda: chunk(128, "itl_max_ms"), "{:,.0f}"),
    ("chunked prefill speedup", ["README.md", "docs/benchmarks.md"],
     lambda: ratio(chunk(2048, "itl_max_ms"), chunk(128, "itl_max_ms")), "{:.1f}x"),
    ("wasted work without watermark at 8192 slots", ["README.md", "docs/benchmarks.md", "docs/writeup.md"],
     lambda: sweep(512, "wasted_no_watermark") * 100, "{:.1f}%"),
    ("wasted work with watermark at 8192 slots", ["README.md", "docs/benchmarks.md", "docs/writeup.md"],
     lambda: sweep(512, "wasted_watermark") * 100, "{:.1f}%"),
]


def main():
    missing_data, mismatches, checked = [], [], 0
    for name, docs, compute, fmt in CLAIMS:
        try:
            value = compute()
        except (TypeError, KeyError, IndexError):
            value = None
        if value is None:
            missing_data.append(name)
            continue
        text = fmt.format(value)
        for doc in docs:
            checked += 1
            body = (ROOT / doc).read_text()
            if text not in body:
                mismatches.append((doc, name, text))

    print(f"checked {checked} claim/document pairs against {RESULTS.relative_to(ROOT)}")
    for name in missing_data:
        print(f"  no data yet: {name}")
    if mismatches:
        print("\nprose does not match the measurements:")
        for doc, name, text in mismatches:
            print(f"  - {doc} should state {name} as {text}")
        print("\nupdate the prose, or re-run the benchmark if the data is stale.")
        sys.exit(1)
    print("every quoted figure matches the data")


if __name__ == "__main__":
    main()
