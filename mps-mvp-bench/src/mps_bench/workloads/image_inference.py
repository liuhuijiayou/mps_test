"""Image inference workload: torchvision ResNet18 / ResNet50.

Both online (small batch, open loop) and offline (large batch, saturated) use the
exact same adapter and the same pre-generated input pool -- only the batch size
and the arrival mode differ, so cross-mode comparison stays honest.

Weights default to offline synthetic init with a fixed seed so the container
never needs network access. Results carry `synthetic_weights=true`; they say
nothing about real model accuracy or production gains.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

from .base import BatchResult, Workload, WorkloadInfo

_DTYPES = {"fp32": "float32", "tf32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ImageInferenceWorkload(Workload):
    unit = "images"
    kind = "image_inference"

    def __init__(self,
                 model_name: str = "resnet18",
                 input_size: int = 224,
                 batch_size: int = 8,
                 precision: str = "fp16",
                 input_pool_size: int = 64,
                 cpu_threads: int = 4,
                 seed: int = 20251001,
                 weights_path: Optional[str] = None,
                 device: str = "cuda"):
        import torch  # local import: host orchestrator must not need torch
        import torchvision

        if model_name not in ("resnet18", "resnet50"):
            raise ValueError(f"image_inference 仅支持 resnet18/resnet50，收到 {model_name}")
        if precision not in _DTYPES:
            raise ValueError(f"未知 precision {precision}")

        self.torch = torch
        self.model_name = model_name
        self.input_size = int(input_size)
        self.batch_size = int(batch_size)
        self.precision = precision
        self.input_pool_size = int(input_pool_size)
        self.seed = int(seed)
        self.weights_path = weights_path
        self.device = torch.device(device)
        self._asset_hashes: Dict[str, str] = {}

        torch.set_num_threads(int(cpu_threads))
        # TF32 is an explicit precision choice, not a silent default.
        torch.backends.cuda.matmul.allow_tf32 = (precision == "tf32")
        torch.backends.cudnn.allow_tf32 = (precision == "tf32")
        torch.backends.cudnn.benchmark = True

        torch.manual_seed(self.seed)
        factory = getattr(torchvision.models, model_name)
        # weights=None -> deterministic local init, no download.
        self.model = factory(weights=None)
        self.synthetic_weights = True
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            state = state.get("state_dict", state)
            self.model.load_state_dict(state)
            self.synthetic_weights = False
            self._asset_hashes["weights"] = _sha256_file(weights_path)

        self.dtype = getattr(torch, _DTYPES[precision])
        self.model = self.model.eval().to(self.device)
        if precision in ("fp16", "bf16"):
            self.model = self.model.to(self.dtype)

        # Input pool is generated once, in the warmup phase, from a fixed seed.
        gen = torch.Generator(device="cpu").manual_seed(self.seed + 7)
        self.pool: List[Any] = []
        for _ in range(self.input_pool_size):
            cpu_batch = torch.randn(self.batch_size, 3, self.input_size, self.input_size,
                                    generator=gen, dtype=torch.float32)
            self.pool.append(cpu_batch.to(self.device, dtype=self.dtype, non_blocking=False))
        # One reference per pool entry, filled during warmup (see `warmup`).
        self._references: Optional[List[float]] = None

    # ------------------------------------------------------------------ #
    def info(self) -> WorkloadInfo:
        return WorkloadInfo(
            kind=self.kind, unit=self.unit,
            synthetic_inputs=True, synthetic_weights=self.synthetic_weights,
            detail={"model": self.model_name, "input_size": self.input_size,
                    "batch_size": self.batch_size, "precision": self.precision,
                    "input_pool_size": self.input_pool_size, "seed": self.seed,
                    "weights_path": self.weights_path},
            asset_hashes=self._asset_hashes,
        )

    def warmup(self, iterations: int = 3) -> None:
        """Model load, pool creation and any autotuning happen before measuring."""
        torch = self.torch
        with torch.inference_mode():
            for i in range(max(1, iterations)):
                self.model(self.pool[i % len(self.pool)])
        torch.cuda.synchronize(self.device)
        # Pre-compute the reference for EVERY pool entry before READY, so the
        # request path never runs an extra forward pass just to validate output.
        self._references = [self._compute_checksum(i) for i in range(len(self.pool))]

    def _compute_checksum(self, batch_index: int) -> float:
        torch = self.torch
        with torch.inference_mode():
            out = self.model(self.pool[batch_index % len(self.pool)])
        torch.cuda.synchronize(self.device)
        return float(out.float().sum().item())

    def run_batch(self, batch_index: int) -> BatchResult:
        torch = self.torch
        batch = self.pool[batch_index % len(self.pool)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record()
            out = self.model(batch)
            end.record()
        # Blocking sync: CPU timing must cover real GPU completion.
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
        flat = out.float()
        return BatchResult(batch_size=batch.shape[0], gpu_ms=gpu_ms,
                           checksum=float(flat.sum().item()),
                           output_shape=tuple(out.shape),
                           finite=bool(torch.isfinite(flat).all().item()),
                           samples=int(batch.shape[0]))

    def reference_checksum(self, batch_index: int = 0) -> float:
        if self._references is None:
            self._references = [self._compute_checksum(i) for i in range(len(self.pool))]
        return self._references[batch_index % len(self._references)]
