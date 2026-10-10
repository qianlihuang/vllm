import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.results / "pareto" / "manifest.json").read_text())
    rows = []
    keys = ["completed", "failed", "duration", "total_input_tokens", "total_output_tokens",
            "output_throughput", "mean_tpot_ms", "median_tpot_ms", "p95_tpot_ms",
            "mean_ttft_ms", "p95_ttft_ms", "mean_e2el_ms", "p95_e2el_ms",
            "spec_decode_acceptance_rate", "spec_decode_acceptance_length"]
    for run in manifest["runs"]:
        value = json.loads((args.results / "pareto" / run["file"]).read_text())
        rows.append({k: run[k] for k in ("arm", "round", "concurrency")} |
                    {k: value.get(k) for k in keys})
    assert len(rows) == 42, "The full three-round, seven-concurrency sweep is incomplete"
    for row in rows:
        assert row["completed"] == max(8, 4 * row["concurrency"])
        assert row["total_output_tokens"] == row["completed"] * 512
        assert row["spec_decode_acceptance_rate"] > 0
        assert row["spec_decode_acceptance_length"] > 1
        peer = next(x for x in rows if x["round"] == row["round"]
                    and x["concurrency"] == row["concurrency"] and x["arm"] != row["arm"])
        assert row["total_input_tokens"] == peer["total_input_tokens"], "A/B prompts differ"
    with (args.output / "pareto_runs.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["arm"], row["concurrency"]), []).append(row)
    summary = []
    for (arm, concurrency), values in sorted(groups.items()):
        entry = {"arm": arm, "concurrency": concurrency, "runs": len(values)}
        for key in ("output_throughput", "mean_tpot_ms", "p95_tpot_ms",
                    "mean_ttft_ms", "spec_decode_acceptance_rate", "spec_decode_acceptance_length"):
            numbers = [v[key] for v in values if v[key] is not None]
            if numbers:
                entry[key] = statistics.median(numbers)
                entry[key + "_min"] = min(numbers)
                entry[key + "_max"] = max(numbers)
        summary.append(entry)
    (args.output / "pareto_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "route_probes.json").write_text(json.dumps(
        {key: manifest[key] for key in ("probe_baseline", "probe_cake")}, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for arm, color in (("baseline", "#777777"), ("cake", "#176AC6")):
        points = [x for x in summary if x["arm"] == arm]
        for ax, metric, label in zip(axes, ("mean_tpot_ms", "p95_tpot_ms"), ("Mean TPOT", "P95 TPOT")):
            xs = [x["output_throughput"] for x in points]
            ys = [x[metric] for x in points]
            front = [p for p in points if not any(
                q["output_throughput"] >= p["output_throughput"] and q[metric] <= p[metric]
                and (q["output_throughput"] > p["output_throughput"] or q[metric] < p[metric])
                for q in points)]
            front.sort(key=lambda p: p["output_throughput"])
            ax.scatter(xs, ys, color=color, label=arm)
            ax.plot([p["output_throughput"] for p in front], [p[metric] for p in front], color=color)
            ax.errorbar(xs, ys,
                        xerr=[[x["output_throughput"]-x["output_throughput_min"] for x in points],
                              [x["output_throughput_max"]-x["output_throughput"] for x in points]],
                        yerr=[[x[metric]-x[metric+"_min"] for x in points],
                              [x[metric+"_max"]-x[metric] for x in points]],
                        fmt="none", color=color, alpha=0.5, capsize=3)
            for x, y, p in zip(xs, ys, points):
                ax.annotate(f'C{p["concurrency"]}', (x, y), xytext=(5, 6), textcoords="offset points", fontsize=8)
            ax.set_xlabel("Output throughput (tokens/s)")
            ax.set_ylabel(label + " (ms/token)")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle("Qwen3.8-Flash-Next · 4×H200 · TP4 + EP · MTP3 probabilistic\n"
                 "SPEED-Bench throughput_1k / 512 output · T=1, k=20, p=0.95 · 3 alternating A/B rounds")
    fig.savefig(args.output / "pareto.png", dpi=200)
    micro = json.loads((args.results / "microbench_flash_next.json").read_text())
    (args.output / "microbench_flash_next.json").write_text(json.dumps(micro, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, paths, title in zip(axes, (("mask_triton", "mask_cake"), ("sample_flashinfer", "sample_cake")),
                                ("Top-k/top-p filtering", "Direct sampling")):
        for path, color in zip(paths, ("#777777", "#176AC6")):
            batches = [x["rows"] for x in micro["cases"]]
            ax.plot(batches,
                    [x["median_us"][path] for x in micro["cases"]], "o-", label=path, color=color)
            ax.fill_between(batches,
                            [min(x["timings_us"][path]) for x in micro["cases"]],
                            [max(x["timings_us"][path]) for x in micro["cases"]],
                            color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sampling rows (vocab=248,320)")
        ax.set_ylabel("Latency (µs)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].axvline(64, color="#999999", linestyle=":")
    axes[0].text(0.97, 0.97, "Cake dispatch: rows ≤ 64", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8)
    fig.suptitle("H200 · captured Flash-Next logits · k=20, p=0.95 · 3 rounds")
    fig.savefig(args.output / "microbench.png", dpi=200)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
