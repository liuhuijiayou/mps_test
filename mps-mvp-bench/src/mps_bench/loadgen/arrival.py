"""Arrival trace generation.

Key property: the trace is a function of (seed, arrival mode, qps, duration) only.
It is generated ONCE, frozen, and reused across every mode. A slower service can
never cause fewer requests to be planned -- that is how coordinated omission
sneaks in.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ArrivalTrace:
    mode: str
    seed: int
    duration_s: float
    planned: List[float]  # offsets in seconds from measurement start
    detail: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.planned)

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "seed": self.seed, "duration_s": self.duration_s,
                "count": len(self.planned), "detail": self.detail}


def fixed_interval(qps: float, duration_s: float) -> List[float]:
    if qps <= 0:
        raise ValueError("target_qps 必须为正")
    step = 1.0 / qps
    n = int(math.floor(duration_s * qps))
    return [i * step for i in range(n)]


def poisson(qps: float, duration_s: float, seed: int) -> List[float]:
    if qps <= 0:
        raise ValueError("target_qps 必须为正")
    rng = random.Random(seed)
    out: List[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(qps)
        if t >= duration_s:
            break
        out.append(t)
    return out


def burst_trace(segments: Sequence[Dict[str, Any]], duration_s: float, seed: int) -> List[float]:
    """Repeatable burst trace: segments = [{t: start_s, qps: float}, ...].

    Each segment's qps holds until the next segment start (or duration end).
    Deterministic given the same seed.
    """
    if not segments:
        raise ValueError("burst_trace 为空")
    pts = sorted(({"t": float(s["t"]), "qps": float(s["qps"])} for s in segments),
                 key=lambda s: s["t"])
    rng = random.Random(seed)
    out: List[float] = []
    for i, seg in enumerate(pts):
        start = seg["t"]
        end = pts[i + 1]["t"] if i + 1 < len(pts) else duration_s
        end = min(end, duration_s)
        if end <= start or seg["qps"] <= 0:
            continue
        t = start
        while True:
            t += rng.expovariate(seg["qps"])
            if t >= end:
                break
            out.append(t)
    return sorted(out)


def build_trace(mode: str, qps: Optional[float], duration_s: float, seed: int,
                segments: Optional[Sequence[Dict[str, Any]]] = None) -> ArrivalTrace:
    if mode == "closed_loop":
        # Closed loop has no planned arrival times; capacity is measured by
        # saturation, and its results are reported separately from latency tests.
        return ArrivalTrace(mode=mode, seed=seed, duration_s=duration_s, planned=[],
                            detail={"note": "闭环饱和模式无计划到达时间，用于容量测试"})
    if mode == "fixed_interval":
        planned = fixed_interval(float(qps or 0), duration_s)
        detail = {"qps": qps}
    elif mode == "poisson":
        planned = poisson(float(qps or 0), duration_s, seed)
        detail = {"qps": qps}
    elif mode == "burst_trace":
        planned = burst_trace(segments or [], duration_s, seed)
        detail = {"segments": list(segments or [])}
    else:
        raise ValueError(f"未知到达模式 {mode}")
    return ArrivalTrace(mode=mode, seed=seed, duration_s=duration_s, planned=planned,
                        detail=detail)
