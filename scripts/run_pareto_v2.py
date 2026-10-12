"""Alternate Cake/baseline on one loaded model, with seeded requests.

Changes from run_pareto.py: every request carries a fixed per-round sampling
seed, so both arms draw the same Gumbel noise and acceptance does not vary by
chance; runs are longer (max(min_prompts, factor * concurrency) prompts); the
captured-logits microbenchmark is not awaited.
"""

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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8500")
    parser.add_argument("--served-name", default="flashnext")
    parser.add_argument("--capture-name", default="")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--prompt-factor", type=int, default=8)
    parser.add_argument("--min-prompts", type=int, default=16)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--first-round", type=int, default=0)
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args()
    result_dir = args.root / "results" / "pareto"
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset_root / "speed_bench"
    assert (dataset / "throughput_1k.jsonl").exists()

    def bench(extra_body, extra_args):
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "vllm", "--base-url", args.base_url,
            "--model", args.served_name, "--tokenizer", args.model,
            "--dataset-name", "speed_bench", "--dataset-path", str(dataset),
            "--speed-bench-dataset-subset", "throughput_1k",
            "--speed-bench-output-len", str(args.output_len),
            "--chat-template-kwargs", '{"enable_thinking":true}',
            "--ignore-eos", "--request-rate", "inf",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
            "--extra-body", json.dumps(extra_body),
            "--save-result", "--result-dir", str(result_dir), "--disable-tqdm",
        ]
        subprocess.run(cmd + extra_args, check=True)

    manifest = result_dir / "manifest.json"
    metadata = {"arguments": {k: str(v) for k, v in vars(args).items()}, "runs": []}
    if args.first_round and manifest.exists():
        metadata["runs"] = json.loads(manifest.read_text())["runs"]
    for arm in () if args.skip_probes else ("cake", "baseline"):
        rpc(args.base_url, "cake_set", "1" if arm == "cake" else "0")
        capture = "512" if (arm == "cake" and args.capture_name) else "0"
        rpc(args.base_url, "cake_probe_start", capture)
        bench({"temperature": 1.0, "top_k": 20, "top_p": 0.95},
              ["--num-prompts", "16", "--max-concurrency", "16", "--num-warmups", "0",
               "--ready-check-timeout-sec", "0", "--result-filename", f"probe_{arm}.json"])
        out = str(args.root / "results" / args.capture_name) if capture != "0" else ""
        metadata[f"probe_{arm}"] = rpc(args.base_url, "cake_probe_stop", out)
    if not args.skip_probes:
        cake_calls = sum(sum(x["calls"].values()) for x in metadata["probe_cake"]["results"])
        baseline_calls = sum(
            sum(x["calls"].values()) for x in metadata["probe_baseline"]["results"]
        )
        assert cake_calls > 0 and baseline_calls == 0, "A/B did not exercise distinct paths"
        (args.root / "results" / "probes.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for round_idx in range(args.first_round, args.rounds):
        body = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 1000 + round_idx}
        for concurrency in args.concurrencies[::1 if round_idx % 2 == 0 else -1]:
            arms = ("baseline", "cake") if round_idx % 2 == 0 else ("cake", "baseline")
            num_prompts = max(args.min_prompts, args.prompt_factor * concurrency)
            if args.max_prompts:
                num_prompts = min(num_prompts, max(args.max_prompts, concurrency))
            for arm in arms:
                rpc(args.base_url, "cake_set", "1" if arm == "cake" else "0")
                filename = f"{arm}_r{round_idx}_c{concurrency}.json"
                bench(body, ["--seed", str(100 + round_idx), "--num-warmups", "4",
                             "--num-prompts", str(num_prompts),
                             "--max-concurrency", str(concurrency),
                             "--result-filename", filename])
                row = json.loads((result_dir / filename).read_text())
                assert row["completed"] == num_prompts
                assert row["total_output_tokens"] == num_prompts * args.output_len
                assert row.get("spec_decode_acceptance_length", 0) > 1
                metadata["runs"].append({"arm": arm, "round": round_idx,
                                         "concurrency": concurrency, "file": filename})
                (result_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"done {filename} tput={row['output_throughput']:.1f} "
                      f"AL={row.get('spec_decode_acceptance_length', 0):.3f} "
                      f"{time.strftime('%H:%M:%S')}", flush=True)
    print("Pareto complete", flush=True)


if __name__ == "__main__":
    main()
