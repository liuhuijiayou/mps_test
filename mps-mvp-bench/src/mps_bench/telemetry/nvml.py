"""NVML-style device sampling via nvidia-smi query.

We label the utilization field explicitly as a *device busy* indicator, because
`utilization.gpu` is not SM active and not SM occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..control.exec import CommandLog, run

QUERY = ("uuid,utilization.gpu,memory.used,memory.free,memory.total,"
         "temperature.gpu,power.draw,clocks.sm,clocks.throttle_reasons.active")

_SENTINELS = {"n/a", "[n/a]", "not supported", "[not supported]", "", "unknown"}


def _num(token: str) -> Optional[float]:
    raw = (token or "").strip()
    if raw.lower() in _SENTINELS:
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


_METRICS = ("gpu_util_device_busy_pct", "memory_used_mib", "memory_free_mib",
           "memory_total_mib", "temperature_c", "power_w", "sm_clock_mhz")


@dataclass
class NvmlSample:
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    uuid: Optional[str] = None
    throttle_reasons: Optional[str] = None
    unavailable: Dict[str, str] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Optional[float]:
        """Attribute access for known metrics only.

        Deliberately restricted to `_METRICS`: `sample.sm_active` must raise,
        because nvidia-smi does not provide SM activity and silently returning
        `utilization.gpu` for it would be exactly the substitution we forbid.
        """
        if name in _METRICS:
            return self.__dict__.get("values", {}).get(name)
        raise AttributeError(
            f"NvmlSample \u65e0 {name!r}\uff1anvidia-smi \u4e0d\u63d0\u4f9b\u8be5\u6307\u6807"
            "\uff08\u5982 SM active \u9700\u7531 DCGM 1002 \u63d0\u4f9b\uff0c\u4e0d\u5f97\u7528 GPU util \u4ee3\u66ff\uff09")


def parse_query(stdout: str) -> Optional[NvmlSample]:
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 8:
            continue
        sample = NvmlSample(uuid=cols[0] or None)
        # Column order mirrors QUERY exactly; each sample carries its own UUID so
        # a timeseries row can never be silently attributed to the wrong GPU.
        mapping = [("gpu_util_device_busy_pct", 1), ("memory_used_mib", 2),
                   ("memory_free_mib", 3), ("memory_total_mib", 4),
                   ("temperature_c", 5), ("power_w", 6), ("sm_clock_mhz", 7)]
        for name, idx in mapping:
            value = _num(cols[idx])
            sample.values[name] = value
            if value is None:
                sample.unavailable[name] = f"nvidia-smi 返回 {cols[idx]!r}"
        if len(cols) > 8:
            raw = cols[8]
            sample.throttle_reasons = None if raw.lower() in _SENTINELS else raw
        return sample
    return None


def sample_device(uuid: str, binary: str = "nvidia-smi",
                  log: Optional[CommandLog] = None) -> Optional[NvmlSample]:
    res = run([binary, "-i", uuid, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
              timeout_s=10, log=log, note="telemetry query-gpu")
    if not res.ok:
        return None
    return parse_query(res.stdout)
