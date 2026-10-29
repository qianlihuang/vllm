# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native worker iteration and forward spans around decode admission.

Input: one benchmark JSON and its eight per-rank torch traces. No engine patch
or artificial CUDA synchronization is needed. CPU start-to-start periods include
forward, draft/sampling, scheduler and waits. GPU annotation bounding spans cover
the annotated main forward, including waits, rather than summed active kernels.
"""

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import regex as re

EXECUTE = re.compile(r"execute_(\d+)_context_(\d+)\(sq(\d+)")


def describe(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
        "max": max(values),
    }


def analyze_trace(path):
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    gpu = defaultdict(list)
    for event in data["traceEvents"]:
        if event.get("cat") == "gpu_user_annotation" and EXECUTE.match(event["name"]):
            key = (event["name"], event.get("args", {}).get("External id"))
            gpu[key].append((event["ts"], event["ts"] + event["dur"]))
    cpu = sorted(
        (
            event
            for event in data["traceEvents"]
            if event.get("cat") == "user_annotation"
            and event.get("ph") == "X"
            and EXECUTE.match(event.get("name", ""))
        ),
        key=lambda event: event["ts"],
    )
    assert len({(e["pid"], e["tid"]) for e in cpu}) == 1, path
    rows = []
    for index, (current, following) in enumerate(zip(cpu, cpu[1:])):
        parsed = EXECUTE.match(current["name"])
        key = (current["name"], current.get("args", {}).get("External id"))
        fragments = gpu[key]
        rows.append(
            {
                "trace": path.name,
                "iteration_index": index,
                "total_scheduled_tokens": int(parsed[1]),
                "context_requests": int(parsed[2]),
                "context_scheduled_tokens": int(parsed[3]),
                "annotation": current["name"],
                "external_id": key[1],
                "cpu_start_us": current["ts"],
                "next_start_us": following["ts"],
                "worker_period_ms": (following["ts"] - current["ts"]) / 1000,
                "cpu_forward_ms": current["dur"] / 1000,
                "gpu_forward_span_ms": (
                    (
                        max(end for _, end in fragments)
                        - min(start for start, _ in fragments)
                    )
                    / 1000
                    if fragments
                    else None
                ),
                "gpu_annotation_fragments": len(fragments),
            }
        )
    assert rows, path
    return rows, {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "base_time_nanoseconds": data.get("baseTimeNanoseconds"),
        "complete_cpu_iterations": len(cpu),
        "measured_periods": len(rows),
    }


def analyze(args):
    case_bytes = args.case.read_bytes()
    benchmark = json.loads(
        gzip.decompress(case_bytes) if args.case.suffix == ".gz" else case_bytes
    )
    assert not benchmark["partial"] and benchmark.get("validation") == "passed"
    prefixes = benchmark["profile"]["targets"]
    all_rows, ranks, instances = [], [], {}
    for target, prefix in prefixes.items():
        files = sorted(args.traces.glob(prefix + "_*.pt.trace.json.gz"))
        assert len(files) == 4, (prefix, len(files))
        rows = []
        bases = set()
        for path in files:
            rank_rows, provenance = analyze_trace(path)
            for row in rank_rows:
                row["target"] = target
            rows.extend(rank_rows)
            ranks.append(provenance)
            bases.add(provenance["base_time_nanoseconds"])
        assert len(bases) == 1 and None not in bases, (target, bases)
        admission = [row for row in rows if row["context_requests"]]
        assert admission, (
            target,
            "No annotated admission; cannot infer handoff window",
        )
        start = min(row["cpu_start_us"] for row in admission)
        end = max(row["next_start_us"] for row in admission)
        for row in rows:
            row["window"] = (
                "before"
                if row["next_start_us"] <= start
                else "after"
                if row["cpu_start_us"] >= end
                else "handoff"
            )
        affected = [row for row in rows if row["window"] == "handoff"]
        normal_before = [row for row in rows if row["window"] == "before"]
        normal_after = [row for row in rows if row["window"] == "after"]
        longest = max(affected, key=lambda row: row["worker_period_ms"])
        instances[target] = {
            "handoff_window_duration_ms": (end - start) / 1000,
            "admission_annotations": len(admission),
            "admitted_requests": sum(row["context_requests"] for row in admission),
            "actual_scheduled_context_tokens": sum(
                row["context_scheduled_tokens"] for row in admission
            ),
            "before_worker_period_ms": describe(
                [row["worker_period_ms"] for row in normal_before]
            ),
            "after_worker_period_ms": describe(
                [row["worker_period_ms"] for row in normal_after]
            ),
            "handoff_worker_period_ms": describe(
                [row["worker_period_ms"] for row in affected]
            ),
            "handoff_gpu_forward_span_ms": describe(
                [
                    row["gpu_forward_span_ms"]
                    for row in affected
                    if row["gpu_forward_span_ms"] is not None
                ]
            ),
            "longest_affected_iteration": longest,
        }
        all_rows.extend(rows)
    admitted = sum(instance["admitted_requests"] for instance in instances.values())
    assert admitted == benchmark["summary"]["handoffs"], (
        admitted,
        benchmark["summary"]["handoffs"],
    )
    summary = {
        "case": str(args.case),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "mode": benchmark["config"]["mode"],
        "handoff_tokens": benchmark["config"]["handoff_tokens"],
        "profiling_enabled": True,
        "token_accounting": {
            "uncached_suffix_total": sum(
                benchmark["summary"]["uncached_suffix_tokens"]
            ),
            "scheduled_context_total": sum(
                instance["actual_scheduled_context_tokens"]
                for instance in instances.values()
            ),
        },
        "definitions": {
            "worker_period_ms": (
                "Consecutive native CPU execute annotation start times on the "
                "same worker; includes sampling/draft, scheduler and waits "
                "between forwards."
            ),
            "gpu_forward_span_ms": (
                "Bounding wall span of all GPU annotation fragments linked by "
                "(name, External id) to that CPU main-forward scope; includes "
                "GPU waits and excludes separately annotated draft/sampling."
            ),
            "handoff_window": (
                "Per EP4 instance, earliest context-admission CPU start through "
                "latest admission next-start, covering affected peers too."
            ),
            "baseline": (
                "All complete periods before/after that instance's handoff "
                "window; medians reported separately because active batch size "
                "changes."
            ),
            "limitation": (
                "Diagnostic torch profiling adds overhead. These are worker/"
                "forward wall spans, not sums of active GPU kernel time or "
                "unprofiled TPOT."
            ),
        },
        "instances": instances,
        "ranks": ranks,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.case.name.removesuffix(".gz").removesuffix(".json")
    (args.out / f"{stem}.steps.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / f"{stem}.steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({"case": stem, "instances": instances}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args())
