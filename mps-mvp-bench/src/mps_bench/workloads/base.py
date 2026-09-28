"""Common workload adapter contract.

One adapter interface serves both business workloads so the service layer, the
load generator and the reporting code never branch on workload type.

Timing contract (enforced by the service, documented here):
* `run_batch` must return only after the GPU work is truly complete
  (torch.cuda.synchronize / event synchronize). An async launch returning is NOT
  an inference completion.
* the returned `gpu_ms` comes from CUDA events around the batch. That interval
  includes queueing/waiting inside the stream; it is not a pure kernel
  instruction time and must not be reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BatchResult:
    batch_size: int
    gpu_ms: Optional[float]
    checksum: float
    output_shape: tuple
    finite: bool
    samples: int  # unit-bearing count: images for image_inference, samples for recsys


@dataclass
class WorkloadInfo:
    kind: str
    unit: str            # "images" or "samples"
    synthetic_inputs: bool
    synthetic_weights: bool
    detail: Dict[str, Any] = field(default_factory=dict)
    asset_hashes: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "unit": self.unit,
                "synthetic_inputs": self.synthetic_inputs,
                "synthetic_weights": self.synthetic_weights,
                "detail": self.detail, "asset_hashes": self.asset_hashes}


class Workload:
    """Adapter base class. Subclasses must not mutate global torch settings
    beyond what is configured (precision / thread count)."""

    unit = "samples"
    kind = "base"

    def info(self) -> WorkloadInfo:  # pragma: no cover - overridden
        raise NotImplementedError

    def warmup(self, iterations: int = 3) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def run_batch(self, batch_index: int) -> BatchResult:  # pragma: no cover
        raise NotImplementedError

    def reference_checksum(self) -> float:  # pragma: no cover
        raise NotImplementedError


def checksum_tolerance_ok(reference: float, observed: float, tolerance: float) -> bool:
    """Fixed-tolerance comparison.

    The tolerance comes from config and is identical across MPS and non-MPS runs;
    it must never be relaxed because an MPS result differed.
    """
    if reference == 0.0:
        return abs(observed) <= tolerance
    return abs(observed - reference) / abs(reference) <= tolerance
