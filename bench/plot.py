#!/usr/bin/env python3
"""Render bench/results/*.json into the README benchmark table (median of runs per tag)."""
import json, pathlib, statistics
rows = []
for f in sorted(pathlib.Path(__file__).parent.glob("results/*.json")):
    runs = json.loads(f.read_text())
    med = lambda k: statistics.median(r[k] for r in runs)
    rows.append((f.stem, med("tokens_per_s"), med("ttft_p50_ms"), med("ttft_p95_ms"), med("itl_p50_ms"), med("itl_p95_ms"), med("kv_waste_fraction_last"), len(runs)))
print("| Config | Tokens/s | TTFT p50 / p95 (ms) | Inter-token p50 / p95 (ms) | KV waste % | runs |")
print("|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]:.0f} | {r[2]:.0f} / {r[3]:.0f} | {r[4]:.1f} / {r[5]:.1f} | {100*r[6]:.1f} | {r[7]} |")
