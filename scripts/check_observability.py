#!/usr/bin/env python3
"""Cross-check every metric referenced outside the server against what it exposes.

Dashboards and autoscalers fail silently: a renamed metric shows an empty panel
and an autoscaler that never triggers. This parses the metric names out of
src/api/metrics.cpp and asserts that every name used by the Grafana dashboard
and the KEDA ScaledObject exists, with the right suffix for its type.

  python3 scripts/check_observability.py
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
errors, checked = [], 0


def exposed_metrics():
    """Names and kinds from the single place they are rendered."""
    src = (ROOT / "src/api/metrics.cpp").read_text()
    kinds = {}
    for name in re.findall(r'counter\("(engine_[a-z_]+)"', src):
        kinds[name] = "counter"
    for name in re.findall(r'gauge\("(engine_[a-z_]+)"', src):
        kinds[name] = "gauge"
    for name in re.findall(r'\.render\("(engine_[a-z_]+)"', src):
        kinds[name] = "histogram"
    if not kinds:
        sys.exit("could not parse any metric names out of src/api/metrics.cpp")
    return kinds


def check_reference(name, where, kinds):
    """A reference may be a bare metric or a histogram-derived series."""
    global checked
    checked += 1
    base, suffix = name, None
    for s in ("_bucket", "_sum", "_count"):
        if name.endswith(s):
            base, suffix = name[: -len(s)], s
            break
    if base not in kinds:
        # A counter reference like engine_requests_total_count would land here too.
        errors.append(f"{where}: unknown metric {name!r}")
        return
    kind = kinds[base]
    if suffix and kind != "histogram":
        errors.append(f"{where}: {name!r} uses {suffix} but {base} is a {kind}")
    if kind == "histogram" and suffix is None:
        errors.append(f"{where}: {base!r} is a histogram and must be queried with _bucket/_sum/_count")


def main():
    kinds = exposed_metrics()
    print(f"src/api/metrics.cpp exposes {len(kinds)} metrics "
          f"({sum(1 for k in kinds.values() if k == 'counter')} counters, "
          f"{sum(1 for k in kinds.values() if k == 'gauge')} gauges, "
          f"{sum(1 for k in kinds.values() if k == 'histogram')} histograms)")

    dash_path = ROOT / "deploy/grafana-dashboard.json"
    dash = json.loads(dash_path.read_text())

    # Panels, including any nested inside collapsed rows.
    panels = list(dash["panels"])
    for p in dash["panels"]:
        panels.extend(p.get("panels", []))
    graph_panels = [p for p in panels if p["type"] != "row"]

    used = set()
    for p in graph_panels:
        if not p.get("targets"):
            errors.append(f"dashboard panel {p['title']!r} has no targets")
        if not p.get("description"):
            errors.append(f"dashboard panel {p['title']!r} has no description")
        for t in p.get("targets", []):
            expr = t["expr"]
            if "$__rate_interval" not in expr and re.search(r"\brate\(", expr) and "[5m]" not in expr:
                errors.append(f"dashboard panel {p['title']!r} uses rate() with a hardcoded window")
            for name in re.findall(r"\bengine_[a-z_]+", expr):
                used.add(name)
                check_reference(name, f"dashboard/{p['title']}", kinds)

    # Panel ids must be unique or Grafana silently drops panels.
    ids = [p["id"] for p in panels]
    if len(ids) != len(set(ids)):
        errors.append("dashboard has duplicate panel ids")

    # Template variables the queries rely on must be declared.
    declared = {v["name"] for v in dash["templating"]["list"]}
    for var in re.findall(r"\$(\w+)", json.dumps([p.get("targets", []) for p in graph_panels])):
        if var.startswith("__"):
            continue
        if var not in declared:
            errors.append(f"dashboard uses undeclared template variable ${var}")

    # The KEDA trigger query is the one that silently stops autoscaling.
    keda = (ROOT / "deploy/helm/templates/scaledobject.yaml").read_text()
    for name in re.findall(r"\bengine_[a-z_]+", keda):
        used.add(name)
        check_reference(name, "keda/scaledobject", kinds)

    unused = sorted(set(kinds) - {n.replace("_bucket", "").replace("_sum", "").replace("_count", "") for n in used})
    print(f"checked {checked} references across the dashboard and the KEDA trigger")
    if unused:
        print(f"note: {len(unused)} exposed metric(s) are not on the dashboard: {', '.join(unused)}")

    if errors:
        print("\nproblems:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("observability references are consistent")


if __name__ == "__main__":
    main()
