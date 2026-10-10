#!/usr/bin/env bash
# Per-kernel achieved memory bandwidth, from Nsight Compute.
#
#   scripts/profile_kernels.sh
#
# Writes bench/results/kernel-bandwidth.json, which bench/report.py renders.
#
# Why percentage of peak rather than GB/s. Decode is memory bound: almost every
# kernel here reads more bytes than it does arithmetic on, so the honest
# question is not "how fast is it" but "how close to the memory system's limit
# is it". A kernel at 80% of peak is finished; one at 15% has something wrong
# with its access pattern, and the number says which kernels are worth
# optimising before any time is spent on them.
#
# ncu reports that directly as gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,
# which is preferable to dividing measured bytes by a datasheet figure: the
# datasheet number is rarely achievable and the comparison would flatter every
# kernel by the same unknown factor.
#
# Profiling serialises kernels and replays them to collect counters, so the
# wall-clock time here means nothing. Only the ratios do.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/Qwen2.5-0.5B-Instruct}
PROMPT=${PROMPT:-"Explain paged attention in two sentences."}
TOKENS=${TOKENS:-8}
RESULTS_DIR=${ENGINE_RESULTS_DIR:-bench/results}
OUT="$RESULTS_DIR/kernel-bandwidth.json"
mkdir -p "$RESULTS_DIR"

if ! command -v ncu >/dev/null 2>&1; then
  for d in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/nvidia/nsight-compute/*; do
    [ -x "$d/ncu" ] && { export PATH="$d:$PATH"; break; }
  done
fi
command -v ncu >/dev/null 2>&1 || {
  echo "ncu not found (Nsight Compute ships with the CUDA toolkit)." >&2
  echo "On a container you may also need --cap-add=CAP_SYS_ADMIN or" >&2
  echo "NVreg_RestrictProfilingToAdminUsers=0; see docs/gpu-setup.md." >&2
  exit 2
}
[ -x ./build/engine ] || { echo "build/engine missing; build with -DENGINE_CUDA=ON first" >&2; exit 2; }

CSV=$(mktemp)
# --target-processes all because the engine may fork; --replay-mode kernel is
# the default and is what makes per-kernel counters possible at all.
echo "profiling ${TOKENS} decode steps; this is slow by design"
ncu --csv --target-processes all \
    --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,dram__bytes.sum \
    ./build/engine generate --model "$MODEL" --backend cuda \
      --prompt "$PROMPT" --max-tokens "$TOKENS" \
    > "$CSV" 2>/dev/null || {
      echo "ncu failed; rerun without --csv to see its error" >&2; exit 1; }

python3 - "$CSV" "$OUT" <<'PY'
import csv, collections, json, sys, re

src, dst = sys.argv[1], sys.argv[2]
rows = []
with open(src) as f:
    # ncu prefixes the CSV with banner lines; the header is the first line that
    # actually looks like one.
    lines = [l for l in f if l.strip()]
    start = next((i for i, l in enumerate(lines) if l.lstrip().startswith('"ID"')), 0)
    rows = list(csv.DictReader(lines[start:]))

agg = collections.defaultdict(lambda: {"calls": 0, "ns": 0.0, "bytes": 0.0, "pct": []})
for r in rows:
    name = (r.get("Kernel Name") or "").strip()
    metric = (r.get("Metric Name") or "").strip()
    unit = (r.get("Metric Unit") or "").strip()
    raw = (r.get("Metric Value") or "").replace(",", "").strip()
    if not name or not metric:
        continue
    try:
        v = float(raw)
    except ValueError:
        continue
    # Demangle lightly: engine's kernels are plain C-style names already.
    short = re.sub(r"\(.*", "", name).split("::")[-1]
    a = agg[short]
    if metric.startswith("gpu__time_duration"):
        a["calls"] += 1
        a["ns"] += v * (1000 if unit == "usecond" else 1e6 if unit == "msecond" else 1)
    elif metric.startswith("dram__bytes"):
        a["bytes"] += v * (1024 if unit == "Kbyte" else 1024**2 if unit == "Mbyte" else 1)
    elif "pct_of_peak" in metric:
        a["pct"].append(v)

out = []
for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["ns"]):
    if not a["calls"]:
        continue
    out.append({
        "kernel": k,
        "calls": a["calls"],
        "total_ms": a["ns"] / 1e6,
        "mean_us": a["ns"] / a["calls"] / 1e3,
        "bytes": a["bytes"],
        "pct_of_peak_dram": (sum(a["pct"]) / len(a["pct"])) if a["pct"] else None,
    })
json.dump(out, open(dst, "w"), indent=2)

w = max([len(r["kernel"]) for r in out], default=10)
print(f"\n{'kernel':<{w}} {'calls':>6} {'total ms':>9} {'mean us':>9} {'% peak BW':>10}")
for r in out:
    p = f"{r['pct_of_peak_dram']:.1f}" if r["pct_of_peak_dram"] is not None else "-"
    print(f"{r['kernel']:<{w}} {r['calls']:>6} {r['total_ms']:>9.2f} {r['mean_us']:>9.1f} {p:>10}")
print(f"\nwrote {dst}")
PY
rm -f "$CSV"
