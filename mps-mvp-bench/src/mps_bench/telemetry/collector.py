"""Lightweight sampling collector.

* one background thread, default 1 Hz, configurable
* rolling terminal summary + raw timeseries to gpu_metrics.csv
* never stops anyone else's profiler; if DCGM refuses (permission / group
  conflict / existing profiler), we record `unavailable` with the reason.
* identical sampling overhead on both sides of a comparison: the collector runs
  per-run, not per-client, so the main A/B pair always shares one collector.
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..control.exec import CommandLog, run, which
from . import dcgm as dcgm_mod
from . import nvml as nvml_mod

CSV_COLUMNS = [
    "wall_ts", "monotonic_s", "gpu_uuid",
    "gpu_util_device_busy_pct", "sm_active", "sm_occupancy", "dram_active",
    "memory_used_mib", "memory_free_mib", "memory_total_mib",
    "temperature_c", "power_w", "sm_clock_mhz", "throttle_reasons",
    "unavailable_metrics", "phase",
]


@dataclass
class CollectorStatus:
    dcgm_available: bool = False
    dcgm_reason: str = ""
    sm_active_available: bool = False
    samples: int = 0
    missing_counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"dcgm_available": self.dcgm_available, "dcgm_reason": self.dcgm_reason,
                "sm_active_available": self.sm_active_available, "samples": self.samples,
                "missing_counts": self.missing_counts}


class Collector:
    def __init__(self, gpu_uuid: str, csv_path: str,
                 interval_s: float = 1.0,
                 nvidia_smi: str = "nvidia-smi",
                 dcgmi: str = "dcgmi",
                 dcgm_fields: Optional[List[int]] = None,
                 require_sm_metrics: bool = True,
                 allow_degraded: bool = False,
                 log: Optional[CommandLog] = None,
                 on_summary: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.gpu_uuid = gpu_uuid
        self.csv_path = csv_path
        self.interval_s = float(interval_s)
        self.nvidia_smi = nvidia_smi
        self.dcgmi = dcgmi
        self.dcgm_fields = list(dcgm_fields or [1002, 1003, 1005])
        self.require_sm_metrics = require_sm_metrics
        self.allow_degraded = allow_degraded
        self.log = log
        self.on_summary = on_summary
        self.status = CollectorStatus()
        self._phase = "idle"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = time.monotonic()
        self._handle = None
        self._writer = None

    # ---------------- phase annotation ----------------
    def set_phase(self, phase: str) -> None:
        """Annotate the timeseries (warmup / measure / inject / recover / ...)."""
        self._phase = phase

    # ---------------- capability ----------------
    def _probe_dcgm_raw(self) -> Dict[str, Any]:
        """Isolated probe step (separate method so it can be stubbed in tests)."""
        if not which(self.dcgmi):
            return {"available": False, "reason": f"未找到 {self.dcgmi}", "sample": None}
        res = self._dcgm_once()
        if res is None:
            return {"available": False,
                    "reason": ("dcgmi 存在但采样失败：可能权限不足、采样分组受限"
                               "或与既有 profiler 冲突（不会停止他人的采集器）"),
                    "sample": None}
        return {"available": True, "reason": "", "sample": res}

    def probe_dcgm(self) -> Dict[str, Any]:
        probe = self._probe_dcgm_raw()
        self.status.dcgm_available = bool(probe.get("available"))
        self.status.dcgm_reason = probe.get("reason", "")
        if self.status.dcgm_available:
            ok, reason = dcgm_mod.required_fields_available(probe.get("sample"),
                                                           self.require_sm_metrics)
            sample = probe.get("sample")
            value = (sample.values.get("sm_active")
                     if isinstance(sample, dcgm_mod.DcgmSample) else None)
            self.status.sm_active_available = value is not None
            if not self.status.sm_active_available:
                self.status.dcgm_reason = reason
        else:
            self.status.sm_active_available = False
        if self.require_sm_metrics and not self.status.sm_active_available and not self.allow_degraded:
            raise RuntimeError(
                "require_sm_metrics=true 但 DCGM_FI_PROF_SM_ACTIVE 不可用："
                f"{self.status.dcgm_reason}。不得用 GPU util 代替 SM active，也不得填 0；"
                "如确需降级采集请显式设置 telemetry.allow_degraded=true（报告将显著标记缺项）")
        return self.status.as_dict()

    def _dcgm_once(self) -> Optional[dcgm_mod.DcgmSample]:
        res = run([self.dcgmi, "dmon", "-e", ",".join(str(f) for f in self.dcgm_fields),
                   "-c", "1"], timeout_s=15, log=self.log, note="dcgmi dmon sample")
        if not res.ok:
            return None
        samples = dcgm_mod.parse_dmon(res.stdout, self.dcgm_fields)
        return samples[-1] if samples else None

    # ---------------- loop ----------------
    def start(self) -> None:
        self._handle = open(self.csv_path, "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            row = self.sample_once()
            if self._writer is not None:
                self._writer.writerow(row)
                if self._handle is not None:
                    self._handle.flush()
            if self.on_summary is not None:
                self.on_summary(row)
            self._stop.wait(self.interval_s)

    def sample_once(self) -> Dict[str, Any]:
        unavailable: List[str] = []
        row: Dict[str, Any] = {col: None for col in CSV_COLUMNS}
        row["wall_ts"] = time.time()
        row["monotonic_s"] = round(time.monotonic() - self._t0, 4)
        row["gpu_uuid"] = self.gpu_uuid
        row["phase"] = self._phase

        nv = nvml_mod.sample_device(self.gpu_uuid, self.nvidia_smi, log=None)
        if nv is None:
            unavailable.append("nvml:all")
        else:
            nv_uuid = getattr(nv, "uuid", None)
        if nv_uuid and self.gpu_uuid and nv_uuid != self.gpu_uuid:
            # Never attribute another card's numbers to our target GPU.
            unavailable.append(f"nvml:uuid_mismatch({nv_uuid})")
        else:
            row.update({k: v for k, v in nv.values.items() if k in CSV_COLUMNS})
            row["throttle_reasons"] = nv.throttle_reasons
            unavailable += [f"nvml:{k}" for k in nv.unavailable]

        if self.status.dcgm_available:
            sample = self._dcgm_once()
            if sample is None:
                unavailable.append("dcgm:all")
            else:
                for metric in ("sm_active", "sm_occupancy", "dram_active"):
                    row[metric] = sample.values.get(metric)
                    if row[metric] is None:
                        unavailable.append(f"dcgm:{metric}")
        else:
            unavailable += ["dcgm:sm_active", "dcgm:sm_occupancy", "dcgm:dram_active"]

        row["unavailable_metrics"] = ";".join(unavailable)
        self.status.samples += 1
        for key in unavailable:
            self.status.missing_counts[key] = self.status.missing_counts.get(key, 0) + 1
        return row

    def stop(self) -> CollectorStatus:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3 + 5)
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return self.status


def rolling_summary(row: Dict[str, Any]) -> str:
    """One-line terminal summary; explicitly distinguishes the three SM-ish metrics."""
    def fmt(key: str, scale: float = 1.0, suffix: str = "") -> str:
        value = row.get(key)
        return "n/a" if value is None else f"{value * scale:.1f}{suffix}"
    return (f"t={row.get('monotonic_s')}s phase={row.get('phase')} "
            f"gpu_busy={fmt('gpu_util_device_busy_pct', 1.0, '%')} "
            f"sm_active={fmt('sm_active', 100.0, '%')} "
            f"sm_occ={fmt('sm_occupancy', 100.0, '%')} "
            f"dram={fmt('dram_active', 100.0, '%')} "
            f"mem_used={fmt('memory_used_mib')}MiB")
