"""Summarize seeded, order-balanced Cake/baseline A/B runs.

Usage: make_report_seeded.py --output DIR NAME=RUN_DIR [NAME=RUN_DIR ...]
Each RUN_DIR holds results/pareto/{baseline,cake}_r{round}_c{concurrency}.json.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir):
    rows = {}
    for f in (run_dir / "results" / "pareto").glob("*_r*_c*.json"):
        arm, r, c = f.stem.split("_")
        rows[(arm, int(r[1:]), int(c[1:]))] = json.loads(f.read_text())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for spec in args.runs:
        name, run_dir = spec.split("=", 1)
        rows = load(Path(run_dir))
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        with open(args.output / f"{slug}_runs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["arm", "round", "concurrency", "completed", "output_throughput",
                             "mean_tpot_ms", "spec_decode_acceptance_length"])
            for (arm, r, c), v in sorted(rows.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
                writer.writerow([arm, r, c, v["completed"], round(v["output_throughput"], 2),
                                 round(v["mean_tpot_ms"], 4),
                                 round(v["spec_decode_acceptance_length"], 4)])
        per_c = {}
        for c in sorted({k[2] for k in rows}):
            deltas, normed, orders = [], [], []
            for r in sorted({k[1] for k in rows}):
                b, k = rows.get(("baseline", r, c)), rows.get(("cake", r, c))
                if not (b and k):
                    continue
                deltas.append((k["output_throughput"] / b["output_throughput"] - 1) * 100)
                normed.append((k["output_throughput"] / k["spec_decode_acceptance_length"])
                              / (b["output_throughput"] / b["spec_decode_acceptance_length"])
                              * 100 - 100)
                orders.append("baseline first" if r % 2 == 0 else "cake first")
            se = statistics.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else None
            per_c[c] = {
                "rounds": len(deltas),
                "delta_pct": [round(x, 2) for x in deltas],
                "order": orders,
                "mean_delta_pct": round(statistics.mean(deltas), 2),
                "se_pct": round(se, 2) if se is not None else None,
                "mean_delta_per_acceptance_pct": round(statistics.mean(normed), 2),
            }
        summary[name] = per_c
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, per_c) in enumerate(summary.items()):
        cs = sorted(per_c)
        xs = [c * (1 + 0.06 * (i - 1)) for c in cs]
        ax.errorbar(xs, [per_c[c]["mean_delta_pct"] for c in cs],
                    yerr=[per_c[c]["se_pct"] or 0 for c in cs],
                    marker="o", capsize=3, label=name)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Cake vs baseline output throughput (%)")
    ax.set_title("Seeded requests, 4 order-balanced A/B rounds (mean ± SE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output / "delta.png", dpi=150)
    for name, per_c in summary.items():
        print(f"## {name}")
        for c, v in per_c.items():
            print(f"  C{c}: {v['mean_delta_pct']:+.1f}% ± {v['se_pct']} (rounds {v['delta_pct']}), "
                  f"per-acceptance {v['mean_delta_per_acceptance_pct']:+.1f}%")


if __name__ == "__main__":
    main()
