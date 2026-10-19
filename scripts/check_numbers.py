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
    # Registered after a round of doc edits left eight figures stale that this
    # script passed over: it checks claims someone thought to name, so a column
    # nobody registered drifts silently. These are the rest of the allocator
    # table and the workload it was run at.
    ("paged resident seqs", ["docs/benchmarks.md"],
     lambda: kv("paged, continuous batching", "mean_resident_seqs"), "{:.1f}"),
    ("static resident seqs", ["docs/benchmarks.md"],
     lambda: kv("paged, static batching", "mean_resident_seqs"), "{:.1f}"),
    ("paged KV waste", ["docs/benchmarks.md"],
     lambda: kv("paged, continuous batching", "kv_waste_fraction") * 100, "{:.1f}%"),
    ("contiguous KV waste", ["docs/benchmarks.md"],
     lambda: kv("contiguous (reserve prompt + max_tokens)", "kv_waste_fraction") * 100, "{:.1f}%"),
    ("allocator workload requests", ["docs/benchmarks.md"],
     lambda: json.loads((RESULTS / "kv-waste.json").read_text())["workload"]["requests"], "{:d}"),
    # The GPU serving figures were never registered, so when a second instance
    # produced numbers twice as fast, the README kept quoting the first one's
    # and this script reported success. Registered now.
    ("gpu continuous tokens/s", ["README.md"],
     lambda: last("gpu-continuous", "tokens_per_s"), "{:,.0f}"),
    ("gpu static tokens/s", ["README.md"],
     lambda: last("gpu-static", "tokens_per_s"), "{:,.0f}"),
    ("gpu kernels tokens/s", ["README.md"],
     lambda: last("gpu-kernels", "tokens_per_s"), "{:,.0f}"),
    ("gpu continuous TTFT p50 ms", ["README.md"],
     lambda: last("gpu-continuous", "ttft_p50_ms"), "{:,.0f}"),
    ("gpu static TTFT p50 ms", ["README.md"],
     lambda: last("gpu-static", "ttft_p50_ms"), "{:,.0f}"),
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
    # The rest of the pressure sweep. Registered after six of its seven rows
    # were found contradicting kv-waste.json: only the 8,192-slot row above
    # was named here, so only that row had stayed true.
    ("preemptions without watermark at 256 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=256: sweep(b, "preemptions_no_watermark"), "{:,}"),
    ("wasted work without watermark at 256 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=256: sweep(b, "wasted_no_watermark") * 100, "{:.1f}%"),
    ("preemptions with watermark at 256 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=256: sweep(b, "preemptions_watermark"), "{:,}"),
    ("wasted work with watermark at 256 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=256: sweep(b, "wasted_watermark") * 100, "{:.1f}%"),
    ("preemptions without watermark at 384 blocks", ["docs/benchmarks.md"],
     lambda b=384: sweep(b, "preemptions_no_watermark"), "{:,}"),
    ("wasted work without watermark at 384 blocks", ["docs/benchmarks.md"],
     lambda b=384: sweep(b, "wasted_no_watermark") * 100, "{:.1f}%"),
    ("preemptions with watermark at 384 blocks", ["docs/benchmarks.md"],
     lambda b=384: sweep(b, "preemptions_watermark"), "{:,}"),
    ("wasted work with watermark at 384 blocks", ["docs/benchmarks.md"],
     lambda b=384: sweep(b, "wasted_watermark") * 100, "{:.1f}%"),
    ("preemptions without watermark at 768 blocks", ["docs/benchmarks.md"],
     lambda b=768: sweep(b, "preemptions_no_watermark"), "{:,}"),
    ("wasted work without watermark at 768 blocks", ["docs/benchmarks.md"],
     lambda b=768: sweep(b, "wasted_no_watermark") * 100, "{:.1f}%"),
    ("preemptions with watermark at 768 blocks", ["docs/benchmarks.md"],
     lambda b=768: sweep(b, "preemptions_watermark"), "{:,}"),
    ("wasted work with watermark at 768 blocks", ["docs/benchmarks.md"],
     lambda b=768: sweep(b, "wasted_watermark") * 100, "{:.1f}%"),
    ("preemptions without watermark at 1024 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=1024: sweep(b, "preemptions_no_watermark"), "{:,}"),
    ("wasted work without watermark at 1024 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=1024: sweep(b, "wasted_no_watermark") * 100, "{:.1f}%"),
    ("preemptions with watermark at 1024 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=1024: sweep(b, "preemptions_watermark"), "{:,}"),
    ("wasted work with watermark at 1024 blocks", ["docs/benchmarks.md", "docs/writeup.md"],
     lambda b=1024: sweep(b, "wasted_watermark") * 100, "{:.1f}%"),
    ("preemptions without watermark at 2048 blocks", ["docs/benchmarks.md"],
     lambda b=2048: sweep(b, "preemptions_no_watermark"), "{:,}"),
    ("wasted work without watermark at 2048 blocks", ["docs/benchmarks.md"],
     lambda b=2048: sweep(b, "wasted_no_watermark") * 100, "{:.1f}%"),
    ("preemptions with watermark at 2048 blocks", ["docs/benchmarks.md"],
     lambda b=2048: sweep(b, "preemptions_watermark"), "{:,}"),
    ("wasted work with watermark at 2048 blocks", ["docs/benchmarks.md"],
     lambda b=2048: sweep(b, "wasted_watermark") * 100, "{:.1f}%"),
    # The prefix table's other three rows. Its TTFT p50 was registered and
    # correct while the rows beside it came from no committed file at all.
    ("prefix TTFT p95 off", ["docs/benchmarks.md"],
     lambda: last("prefix-off", "ttft_p95_ms"), "{:,.0f} ms"),
    ("prefix TTFT p95 on", ["docs/benchmarks.md"],
     lambda: last("prefix-on", "ttft_p95_ms"), "{:,.0f} ms"),
    ("prefix e2e p50 off", ["docs/benchmarks.md"],
     lambda: last("prefix-off", "e2e_p50_s"), "{:.2f} s"),
    ("prefix e2e p50 on", ["docs/benchmarks.md"],
     lambda: last("prefix-on", "e2e_p50_s"), "{:.2f} s"),
]


# Figures that were once true and must not reappear.
#
# The claims above check that the right string is *present*. That is not the
# same as the wrong one being absent, and the difference is not theoretical:
# "chunked prefill speedup" was registered and passing while README.md:160
# said 11.8x, because README.md:77 also said 1.3x and presence was all anyone
# asked for. A document can contradict itself and satisfy every claim in the
# list. Each entry here is a superseded value, with the figure that replaced
# it, so the same self-contradiction fails instead of passing.
STALE = [
    ("11.8x", "chunked prefill worst-stall speedup, now 1.3x"),
    ("16.2% to 2.3%", "watermark recompute waste, now 17.1% to 4.7%"),
    ("0.7% occupancy cost", "never recorded in kv-waste.json; no source"),
    ("| 162 |", "preemptions at 4,096 slots without the watermark, now 80"),
    ("19.2%", "wasted work at 4,096 slots without the watermark, now 18.3%"),
    ("| 150 |", "preemptions at 16,384 slots without the watermark, now 49"),
    ("27,874", "prefix-off TTFT p95, now 3,698 ms"),
    ("101.7 s", "prefix-off wall clock, now 44.0 s"),
]
STALE_DOCS = ["README.md", "docs/benchmarks.md", "docs/writeup.md"]


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

    stale = []
    for doc in STALE_DOCS:
        body = (ROOT / doc).read_text()
        for text, why in STALE:
            if text in body:
                stale.append((doc, text, why))

    print(f"checked {checked} claim/document pairs against {RESULTS.relative_to(ROOT)}")
    print(f"checked {len(STALE)} superseded figures are absent from "
          f"{len(STALE_DOCS)} documents")
    for name in missing_data:
        print(f"  no data yet: {name}")
    if mismatches:
        print("\nprose does not match the measurements:")
        for doc, name, text in mismatches:
            print(f"  - {doc} should state {name} as {text}")
    if stale:
        print("\nsuperseded figures are still quoted:")
        for doc, text, why in stale:
            print(f"  - {doc} still contains {text!r}: {why}")
    if mismatches or stale:
        print("\nupdate the prose, or re-run the benchmark if the data is stale.")
        sys.exit(1)
    print("every quoted figure matches the data")


if __name__ == "__main__":
    main()
