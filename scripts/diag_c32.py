"""C32 diagnostics: count Cake calls by row count, then an A/A baseline spread."""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


def rpc(base, method, *args):
    payload = json.dumps({"method": method, "args": list(args), "timeout": 180}).encode()
    request = urllib.request.Request(
        base + "/collective_rpc", payload, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=200) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--aa-runs", type=int, default=4)
    args = parser.parse_args()
    out = args.root / "results" / "diag"
    out.mkdir(parents=True, exist_ok=True)
    num_prompts = 8 * args.concurrency

    def bench(filename):
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "vllm", "--base-url", args.base_url,
            "--model", "flashnext", "--tokenizer", args.model,
            "--dataset-name", "speed_bench",
            "--dataset-path", str(args.dataset_root / "speed_bench"),
            "--speed-bench-dataset-subset", "throughput_1k",
            "--speed-bench-output-len", "512",
            "--chat-template-kwargs", '{"enable_thinking":true}',
            "--ignore-eos", "--request-rate", "inf", "--num-warmups", "4",
            "--extra-body", json.dumps(
                {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 1000}
            ),
            "--seed", "100", "--num-prompts", str(num_prompts),
            "--max-concurrency", str(args.concurrency),
            "--save-result", "--result-dir", str(out),
            "--result-filename", filename, "--disable-tqdm",
        ]
        subprocess.run(cmd, check=True)
        row = json.loads((out / filename).read_text())
        return row["output_throughput"], row.get("spec_decode_acceptance_length", 0)

    rpc(args.base_url, "cake_set", "1")
    rpc(args.base_url, "cake_probe_start", "0")
    tput, al = bench(f"probe_c{args.concurrency}.json")
    calls = rpc(args.base_url, "cake_probe_stop", "")
    rank0 = next(r for r in calls["results"] if r["rank"] == 0)
    by_rows = sorted(
        ((int(k.split("rows=")[1]), v) for k, v in rank0["calls"].items()), reverse=True
    )
    print(f"probe C{args.concurrency}: tput={tput:.1f} AL={al:.3f} cake calls by rows "
          f"(rank 0): {by_rows}", flush=True)

    rpc(args.base_url, "cake_set", "0")
    results = []
    for i in range(args.aa_runs):
        tput, al = bench(f"aa_baseline_{i}_c{args.concurrency}.json")
        results.append(tput)
        print(f"A/A baseline run {i}: tput={tput:.1f} AL={al:.3f}", flush=True)
    mean = sum(results) / len(results)
    spread = (max(results) - min(results)) / mean * 100
    print(f"A/A baseline C{args.concurrency}: mean={mean:.1f} range={min(results):.1f}-"
          f"{max(results):.1f} ({spread:.1f}% of mean)", flush=True)


if __name__ == "__main__":
    main()
