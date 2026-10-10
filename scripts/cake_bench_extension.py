"""Benchmark-only controls; no instrumentation remains during timed requests."""

from collections import Counter
from pathlib import Path

import torch

from vllm import envs
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.sample.ops import topk_topp_sampler as ops


class CakeBenchExtension:
    def cake_set(self, enabled):
        envs.VLLM_USE_FLASHINFER_CAKE_SAMPLER = str(enabled) == "1"
        return {"rank": self.rank, "enabled": envs.VLLM_USE_FLASHINFER_CAKE_SAMPLER}

    def cake_probe_start(self, capture_rows="0"):
        self._cake_originals = {
            name: getattr(ops, name)
            for name in ("apply_top_k_top_p_cake", "cake_sample")
        }
        self._cake_counts = Counter()
        self._cake_captured = []
        self._cake_capture_remaining = int(capture_rows) if self.rank == 0 else 0
        for name, original in self._cake_originals.items():
            def probe(logits, *args, _name=name, _original=original, **kwargs):
                self._cake_counts[f"{_name}:rows={len(logits)}"] += 1
                if self._cake_capture_remaining:
                    take = min(len(logits), self._cake_capture_remaining)
                    with gpu_sync_allowed():
                        self._cake_captured.append(logits[:take].detach().float().cpu())
                    self._cake_capture_remaining -= take
                return _original(logits, *args, **kwargs)
            setattr(ops, name, probe)
        return {"rank": self.rank, "capture_rows": int(capture_rows)}

    def cake_probe_stop(self, output_path=""):
        for name, original in self._cake_originals.items():
            setattr(ops, name, original)
        count = sum(len(x) for x in self._cake_captured)
        if count and output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.cat(self._cake_captured), output_path)
        self._cake_captured = []
        return {"rank": self.rank, "calls": dict(self._cake_counts), "captured": count}
