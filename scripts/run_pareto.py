"""Alternate Cake/baseline on the same loaded model and physical GPUs."""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def rpc(base, method, *args):
    payload = json.dumps({"method": method, "args": list(args), "timeout": 180}).encode()
    request = urllib.request.Request(
        base + "/collective_rpc", payload, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=200) as response:
        result = json.load(response)
    print(method, result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8500")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=512)
    args = parser.parse_args()
    result_dir = args.root / "results" / "pareto"
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.root / "datasets" / "speed_bench"
    while not (dataset / "throughput_1k.jsonl").exists():
        print("Waiting for prepared SPEED-Bench throughput_1k", flush=True)
        time.sleep(30)
    common = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm", "--base-url", args.base_url,
        "--model", "flashnext", "--tokenizer", args.model,
        "--dataset-name", "speed_bench", "--dataset-path", str(dataset),
        "--speed-bench-dataset-subset", "throughput_1k",
        "--speed-bench-output-len", str(args.output_len),
        "--chat-template-kwargs", '{"enable_thinking":true}',
        "--ignore-eos", "--request-rate", "inf", "--num-warmups", "4",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
        "--extra-body", '{"temperature":1.0,"top_k":20,"top_p":0.95}',
        "--save-result", "--result-dir", str(result_dir), "--disable-tqdm",
    ]
    metadata = {"arguments": vars(args).copy(), "runs": []}
    metadata["arguments"]["root"] = str(args.root)
    # Probes are outside timing. The second arm must produce no Cake calls.
    for arm in ("cake", "baseline"):
        rpc(args.base_url, "cake_set", "1" if arm == "cake" else "0")
        rpc(args.base_url, "cake_probe_start", "512" if arm == "cake" else "0")
        subprocess.run(common + ["--num-prompts", "16", "--max-concurrency", "16",
                                "--num-warmups", "0", "--ready-check-timeout-sec", "0",
                                "--result-filename", f"probe_{arm}.json"], check=True)
        metadata[f"probe_{arm}"] = rpc(
            args.base_url, "cake_probe_stop", str(args.root / "results" / "flash_next_logits.pt")
        )
    cake_calls = sum(sum(x["calls"].values()) for x in metadata["probe_cake"]["results"])
    baseline_calls = sum(sum(x["calls"].values()) for x in metadata["probe_baseline"]["results"])
    assert cake_calls > 0 and baseline_calls == 0, "A/B did not exercise distinct sampling paths"
    (args.root / "results" / "probes.json").write_text(json.dumps(metadata, indent=2) + "\n")
    micro_exit = args.root / "results" / "microbench_flash_next.exit"
    while not micro_exit.exists():
        print("Waiting for captured-logits microbenchmark to finish", flush=True)
        time.sleep(15)
    assert micro_exit.read_text().strip() == "0", "captured-logits microbenchmark failed"
    for round_idx in range(args.rounds):
        for concurrency in args.concurrencies[::1 if round_idx % 2 == 0 else -1]:
            arms = ("baseline", "cake") if round_idx % 2 == 0 else ("cake", "baseline")
            for arm in arms:
                rpc(args.base_url, "cake_set", "1" if arm == "cake" else "0")
                filename = f"{arm}_r{round_idx}_c{concurrency}.json"
                subprocess.run(common + [
                    "--seed", str(100 + round_idx), "--num-prompts", str(max(8, 4 * concurrency)),
                    "--max-concurrency", str(concurrency), "--result-filename", filename,
                ], check=True)
                row = json.loads((result_dir / filename).read_text())
                assert row["completed"] == max(8, 4 * concurrency)
                assert row["total_output_tokens"] == row["completed"] * args.output_len
                assert row.get("spec_decode_acceptance_rate", 0) > 0
                assert row.get("spec_decode_acceptance_length", 0) > 1
                metadata["runs"].append({"arm": arm, "round": round_idx,
                                         "concurrency": concurrency, "file": filename})
                (result_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Pareto complete", flush=True)


if __name__ == "__main__":
    main()
