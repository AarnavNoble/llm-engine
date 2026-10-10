#!/usr/bin/env python3
"""Turn bench/results/*.json into the published artifacts.

Writes:
  docs/results.md    the benchmark tables, for the README and the write-up
  docs/results.html  a standalone page, no assets, hostable anywhere

Every figure comes from a results file, so a row that has not been measured is
rendered as "not measured" rather than guessed at or quietly omitted. That is the
point of generating this rather than hand-maintaining it: the published numbers
cannot drift from the files the harness produced, and the hardware they were
measured on is recorded beside them.

  python3 bench/report.py
"""
import json
import pathlib
import statistics
from datetime import date

import os

ROOT = pathlib.Path(__file__).resolve().parent.parent
# ENGINE_RESULTS_DIR lets a smoke run render its own numbers without touching the
# published ones.
RESULTS = pathlib.Path(os.environ["ENGINE_RESULTS_DIR"]) if os.environ.get("ENGINE_RESULTS_DIR") \
    else ROOT / "bench/results"


# Every tag any table asks for, whether or not a file existed. An orphaned
# results file means the harness measured something under a name nothing
# renders, which otherwise shows up only as a silently empty row.
REQUESTED = set()


def load(name):
    REQUESTED.add(name)
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def environment():
    """The hardware the numbers came from.

    A throughput figure is meaningless without it: the same code and the same
    table read as unremarkable on one card and excellent on another. Written by
    reproduce.sh; absent for older results, which are then simply unlabelled
    rather than labelled with this machine's hardware.
    """
    REQUESTED.add("env")
    p = RESULTS / "env.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def provenance_line():
    """Describes the machine of the most recent full run, and says so.

    It deliberately does not say 'measured on', because rows older than that
    run were measured elsewhere. Each row carries its own hardware instead.
    """
    e = environment()
    if not e:
        return "_Hardware not recorded for the last run._"
    bits = [b for b in (e.get("gpu"), e.get("cpu"), e.get("cuda"), e.get("driver")) if b]
    return ("_Last full run on " + ", ".join(bits) + f", commit `{e.get('commit', 'unknown')}`. "
            "Each row below names the machine it was measured on; rows may come "
            "from different machines._")


def med(runs, key):
    vals = [r[key] for r in runs if key in r and r[key] is not None]
    return statistics.median(vals) if vals else None


def fmt(v, digits=1, suffix=""):
    return "not measured" if v is None else f"{v:,.{digits}f}{suffix}"


# ---------------------------------------------------------------- serving table
def serving_rows():
    """The headline table. GPU rows stay empty until the CUDA backend exists."""
    rows = []
    for tag, label, note in [
        ("hf-b1", "HF transformers, batch 1", "reference floor"),
        ("hf-b32", "HF transformers, padded batch 32", "naive batching"),
        ("cpu-static", "engine, static batching", "CPU backend"),
        ("cpu-continuous", "engine, continuous batching", "CPU backend"),
        ("gpu-static", "engine, static batching", "CUDA backend"),
        ("gpu-continuous", "engine, continuous batching", "CUDA backend"),
        ("gpu-prefix", "engine, + prefix caching", "CUDA backend"),
        ("gpu-kernels", "engine, + fused kernels", "CUDA backend"),
        ("gpu-graphs", "engine, + CUDA graphs", "CUDA backend"),
        ("vllm", "vLLM, same model and GPU", "reference"),
    ]:
        runs = load(tag)
        if not runs:
            rows.append((label, note, None, None, None, None, None, 0))
            continue
        # The hardware travels with the row, not with the table. A results
        # directory accumulates rows measured on different machines -- the CPU
        # rows here were taken on a laptop and the GPU rows on a rented A40 --
        # and one hardware line above the table silently claims a single
        # machine for all of them.
        machines = sorted({r["machine"] for r in runs if r.get("machine")})
        # A backfilled label was derived afterwards from the run's env.json
        # rather than recorded by the run itself, and says so.
        back = any(r.get("machine_backfilled") for r in runs)
        if machines:
            note = note + " · " + ", ".join(machines) + (" (backfilled)" if back else "")
        else:
            note = note + " · hardware not recorded"
        rows.append((label, note, med(runs, "tokens_per_s"), med(runs, "ttft_p50_ms"),
                     med(runs, "ttft_p95_ms"), med(runs, "itl_p50_ms"), med(runs, "itl_p95_ms"),
                     len(runs)))
    return rows


# ------------------------------------------------------------- allocator table
def kv_rows():
    d = load("kv-waste")
    if not d:
        return None, []
    return d["workload"], [
        (r["strategy"], 100 * r["slot_utilization"], 100 * r["kv_waste_fraction"],
         r["mean_resident_seqs"], r["mean_active_seqs"], r["preemptions"])
        for r in d["results"]
    ]


def prefix_rows():
    off, on = load("prefix-off"), load("prefix-on")
    if not off or not on:
        return []
    hit = med(on, "prefix_hit_blocks"), med(on, "prefix_total_blocks")
    ratio = 100 * hit[0] / hit[1] if hit[0] and hit[1] else None
    return [
        ("TTFT p50", med(off, "ttft_p50_ms"), med(on, "ttft_p50_ms"), "ms"),
        ("TTFT p95", med(off, "ttft_p95_ms"), med(on, "ttft_p95_ms"), "ms"),
        ("end-to-end p50", med(off, "e2e_p50_s"), med(on, "e2e_p50_s"), "s"),
        ("wall clock", med(off, "wall_s"), med(on, "wall_s"), "s"),
    ], ratio


def step_rows():
    d = load("kernel-time")
    if not d:
        return None, []
    return d.get("device_ms_per_step"), [(r["region"], r["ms_per_step"], r["share_pct"])
                                         for r in d.get("regions", [])]


def bandwidth_rows():
    d = load("kernel-bandwidth")
    if not d:
        return []
    return [(r["kernel"], r["calls"], r["total_ms"], r["mean_us"], r.get("pct_of_peak_dram"))
            for r in d]


def chunk_rows():
    runs = load("chunked-prefill")
    if not runs:
        return []
    by_chunk = {}
    for r in runs:
        by_chunk.setdefault(r["chunk"], []).append(r)
    return [(c, med(v, "itl_p50_ms"), med(v, "itl_p95_ms"), med(v, "itl_max_ms"),
             med(v, "long_prompt_ttft_s")) for c, v in sorted(by_chunk.items(), reverse=True)]


# ----------------------------------------------------------------- markdown out
def markdown():
    L = []
    L.append("<!-- generated by bench/report.py; do not edit by hand -->")
    L.append(f"_Generated {date.today().isoformat()} from `bench/results/`._\n")
    L.append(provenance_line() + "\n")

    L.append("## Serving throughput and latency\n")
    L.append("Qwen2.5-0.5B-Instruct, 32 concurrent requests, prompt mix 128/512/1024 at 50/30/20, "
             "geometric output lengths around 128, median of the recorded runs.\n")
    L.append("| Config | Backend | Tokens/s | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | runs |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, note, tps, t50, t95, i50, i95, n in serving_rows():
        L.append(f"| {label} | {note} | {fmt(tps)} | {fmt(t50, 0, ' ms')} | {fmt(t95, 0, ' ms')} | "
                 f"{fmt(i50, 1, ' ms')} | {fmt(i95, 1, ' ms')} | {n or '-'} |")

    wl, rows = kv_rows()
    if rows:
        L.append("\n## KV cache efficiency\n")
        L.append(f"Allocator property, measured with a fake model, so it is hardware independent. "
                 f"{wl['requests']} requests, {wl['kv_slots']:,} KV token slots "
                 f"({wl['kv_slots'] // wl['block_size']} blocks of {wl['block_size']}), "
                 f"outputs around {wl['mean_output_len']} tokens.\n")
        L.append("| Strategy | Slot utilization | KV waste | Resident seqs | Decoding seqs | Preemptions |")
        L.append("|---|---|---|---|---|---|")
        for strat, util, waste, res, act, pre in rows:
            L.append(f"| {strat} | {util:.1f}% | {waste:.1f}% | {res:.1f} | {act:.1f} | {pre} |")

    pref = prefix_rows()
    if pref and pref[0]:
        metrics, ratio = pref
        L.append("\n## Prefix caching\n")
        L.append("32 requests sharing a 512-token system prompt with a 128-token tail each.\n")
        L.append("| Metric | Cache off | Cache on | Change |")
        L.append("|---|---|---|---|")
        for name, a, b, unit in metrics:
            change = f"{a / b:.1f}x faster" if a and b else "-"
            L.append(f"| {name} | {fmt(a, 1, ' ' + unit)} | {fmt(b, 1, ' ' + unit)} | **{change}** |")
        if ratio:
            L.append(f"\nBlock hit ratio {ratio:.1f}%.")

    step_ms, step = step_rows()
    if step:
        L.append("## Where a decode step goes\n")
        L.append(f"CUDA events on the default stream, 32 concurrent requests, "
                 f"{step_ms:,.2f} ms of device time per step. Nsight Compute would give achieved "
                 "bandwidth too and does not run on a rented pod, so this measures which region "
                 "the time is in rather than how close each is to the memory limit. Recording "
                 "events costs something, so read the proportions and not the absolute total.\n")
        L.append("| Region | ms/step | Share |")
        L.append("|---|---|---|")
        for name, ms, pct in step:
            L.append(f"| `{name}` | {ms:,.3f} | {pct:.1f}% |")
        L.append("")

    bw = bandwidth_rows()
    if bw:
        L.append("## Per-kernel achieved memory bandwidth\n")
        L.append("Nsight Compute, one generate run. Decode is memory bound, so the question "
                 "is how close each kernel runs to the memory system's limit rather than how "
                 "many GB/s it moves; a kernel well under peak has an access-pattern problem "
                 "worth fixing, and one near it is finished. Profiling serialises and replays "
                 "kernels, so the absolute times are not wall-clock figures.\n")
        L.append("| Kernel | Calls | Total ms | Mean us | % of peak DRAM |")
        L.append("|---|---|---|---|---|")
        for k, c, tot, mean, pct in bw:
            L.append(f"| `{k}` | {c:,} | {tot:,.2f} | {mean:,.1f} | "
                     f"{'not measured' if pct is None else f'{pct:.1f}%'} |")
        L.append("")

    ch = chunk_rows()
    if ch:
        L.append("\n## Chunked prefill\n")
        L.append("Four streams decoding while a 2,048-token prompt is injected mid-flight. "
                 "The worst gap is what a user watching tokens appear actually feels; p95 over a "
                 "few hundred samples hides a single long stall.\n")
        L.append("| Chunk size | ITL p50 | ITL p95 | Worst gap | Injected prompt TTFT |")
        L.append("|---|---|---|---|---|")
        for c, i50, i95, imax, ttft in ch:
            L.append(f"| {c} | {fmt(i50, 0, ' ms')} | {fmt(i95, 0, ' ms')} | **{fmt(imax, 0, ' ms')}** | "
                     f"{fmt(ttft, 2, ' s')} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- html out
CSS = """
:root { --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --accent:#0b5fff; --bg:#fff; --code:#f6f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e6e6; --muted:#9aa0a6; --line:#2d2f33; --accent:#6ea8ff; --bg:#141518; --code:#1d1f23; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width:70rem; margin:0 auto; padding:3rem 1.25rem 5rem; }
h1 { font-size:1.9rem; margin:0 0 .25rem; letter-spacing:-.02em; }
h2 { font-size:1.2rem; margin:2.75rem 0 .5rem; padding-top:1.25rem; border-top:1px solid var(--line); }
p { color:var(--fg); max-width:60rem; }
.sub { color:var(--muted); margin:0 0 2rem; }
a { color:var(--accent); }
code { background:var(--code); padding:.1em .35em; border-radius:4px; font-size:.9em; }
.scroll { overflow-x:auto; margin:1rem 0; }
table { border-collapse:collapse; width:100%; font-size:.9rem; min-width:40rem; }
th,td { text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line); white-space:nowrap; }
th { font-weight:600; color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
td:not(:first-child),th:not(:first-child) { text-align:right; font-variant-numeric:tabular-nums; }
tr.pending td { color:var(--muted); font-style:italic; }
strong { font-weight:600; }
footer { margin-top:3rem; color:var(--muted); font-size:.85rem; }
"""


def html(md_text):
    """Minimal markdown-to-HTML for the subset this report emits."""
    out, in_table, rows = [], False, []

    def flush_table():
        nonlocal rows, in_table
        if not rows:
            return
        head, body = rows[0], rows[2:]
        out.append('<div class="scroll"><table>')
        out.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>")
        for r in body:
            cells = "".join(f"<td>{c}</td>" for c in r)
            cls = ' class="pending"' if any("not measured" in c for c in r) else ""
            out.append(f"<tr{cls}>{cells}</tr>")
        out.append("</tbody></table></div>")
        rows, in_table = [], False

    def inline(t):
        import re
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        t = re.sub(r"_(.+?)_", r"<em>\1</em>", t)
        return t

    for line in md_text.splitlines():
        if line.startswith("<!--"):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells) and cells:
                rows.append(cells)
                continue
            rows.append([inline(c) for c in cells])
            in_table = True
            continue
        if in_table:
            flush_table()
        if line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip():
            out.append(f"<p>{inline(line.strip())}</p>")
    flush_table()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>engine — benchmark results</title>
<style>{CSS}</style></head>
<body><main>
<h1>engine</h1>
<p class="sub">An LLM inference server written from scratch in C++ and CUDA.
Paged KV cache, continuous batching, prefix caching, OpenAI-compatible streaming API.
<a href="https://github.com/AarnavNoble/llm-engine">Source on GitHub</a>.</p>
{chr(10).join(out)}
<footer>Every figure is generated from the JSON the benchmark harness wrote, by
<code>bench/report.py</code>. Rows marked <em>not measured</em> have no data behind them
yet and are left empty rather than estimated. Reproduce with
<code>scripts/reproduce.sh</code>.</footer>
</main></body></html>
"""


def main():
    md = markdown()
    out_dir = pathlib.Path(os.environ.get("ENGINE_REPORT_DIR", ROOT / "docs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(md)
    page = html(md)
    (out_dir / "results.html").write_text(page)
    # GitHub Pages serves this directory and wants an index. Writing it here
    # rather than committing a hand-made landing page keeps the published site
    # and the measured numbers the same artifact: there is no second copy to
    # fall out of date, because both come from this function.
    (out_dir / "index.html").write_text(page)
    rows = serving_rows()
    measured = sum(1 for r in rows if r[2] is not None)
    print(f"wrote {out_dir}/results.md and {out_dir}/results.html "
          f"({measured} of {len(rows)} serving rows measured)")

    # A results file under a tag no table reads is a harness bug, not an
    # unmeasured row: the work was done and the number thrown away. This was a
    # real failure -- reproduce.sh tagged its GPU runs "cuda-static" from the
    # backend name while this file asked for "gpu-static", so a full run
    # published a headline table with every row empty and said so only in the
    # count above. Reported loudly, and non-zero exit so a harness can catch it.
    present = {p.stem for p in RESULTS.glob("*.json")}
    orphans = sorted(present - REQUESTED)
    if orphans:
        print(f"\nERROR: {len(orphans)} results file(s) under tags no table renders:")
        for o in orphans:
            print(f"  {RESULTS / (o + '.json')}")
        print("Either the benchmark is writing the wrong tag or a table is missing.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
