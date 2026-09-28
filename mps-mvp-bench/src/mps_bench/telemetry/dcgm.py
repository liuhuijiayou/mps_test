"""DCGM profiling field parsing.

Field semantics we must not blur:
  1002 DCGM_FI_PROF_SM_ACTIVE     -- fraction of time >=1 warp was resident on an SM
  1003 DCGM_FI_PROF_SM_OCCUPANCY  -- resident warps / max warps
  and neither equals "GPU util" (a device-busy indicator) nor proves ALU saturation.

BLANK / sentinel values (N/A, Not Supported, -1, and NVML's 0xFFFF... sentinels)
become None with a reason. They are never coerced to 0, and GPU util is never
substituted for SM active.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

FIELD_NAMES = {
    1001: "GRACT",
    1002: "SMACT",
    1003: "SMOCC",
    1004: "TENSO",
    1005: "DRAMA",
    1009: "PCITX",
    1010: "PCIRX",
}
FIELD_METRICS = {
    1002: "sm_active",
    1003: "sm_occupancy",
    1005: "dram_active",
    1001: "graphics_engine_active",
    1004: "tensor_active",
}

_SENTINELS = {"n/a", "na", "not supported", "notsupported", "nosupport", "-", "",
              "blank", "null", "nan"}
# NVML blank sentinels surface through DCGM as these magic numbers.
_NUMERIC_SENTINELS = {-1.0, 9223372036854775794.0, 9223372036854775792.0,
                      9223372036854775807.0, 4294967295.0}


@dataclass
class DcgmSample:
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    unavailable: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"values": self.values, "unavailable": self.unavailable}


def parse_scalar(token: str) -> "tuple[Optional[float], str]":
    """Return (value, unavailable_reason). Reason is "" when the value is good."""
    raw = (token or "").strip()
    low = raw.lower().strip("[]")
    if low in _SENTINELS:
        return None, f"DCGM 返回哨兵/空值: {raw!r}"
    try:
        value = float(raw)
    except ValueError:
        return None, f"无法解析为数值: {raw!r}"
    if value != value:  # NaN
        return None, f"DCGM 返回 NaN: {raw!r}"
    if value in _NUMERIC_SENTINELS:
        return None, f"DCGM/NVML 哨兵值: {raw!r}"
    return value, ""


_HEADER_RE = re.compile(r"[A-Za-z_]+")


def parse_dmon(stdout: str, fields: List[int]) -> Any:
    """Parse `dcgmi dmon -e <fields>` table output.

    Format (DCGM 2.x/3.x):
        #Entity   SMACT   SMOCC   DRAMA
        GPU 0     0.512   0.204   0.101

    Returns a `DmonResult`: a list of DcgmSample that ALSO behaves like the last
    sample's flat dict (`result["sm_active"]`, `result["sm_active_reason"]`), so
    callers can use whichever view fits without re-parsing.
    """
    samples: List[DcgmSample] = []
    header: List[str] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.lower().startswith("entity"):
            header = [tok for tok in _HEADER_RE.findall(stripped) if tok.lower() not in
                      ("entity", "id")]
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        # Entity column may be "GPU 0" (two tokens) or "GPU-0".
        offset = 2 if parts[0].upper() == "GPU" and parts[1].isdigit() else 1
        data_tokens = parts[offset:]
        names = header if len(header) == len(data_tokens) else [
            FIELD_NAMES.get(f, f"field_{f}") for f in fields]
        sample = DcgmSample()
        for name, token in zip(names, data_tokens):
            metric = _metric_for_header(name)
            value, reason = parse_scalar(token)
            sample.values[metric] = value
            if reason:
                sample.unavailable[metric] = reason
        samples.append(sample)
    return DmonResult(samples)


class DmonResult(list):
    """List of DcgmSample with flat dict-style access to the newest sample."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if not len(self):
                return None
            latest: DcgmSample = list.__getitem__(self, -1)
            if key.endswith("_reason"):
                return latest.unavailable.get(key[: -len("_reason")])
            return latest.values.get(key)
        return list.__getitem__(self, key)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str) and len(self):
            latest: DcgmSample = list.__getitem__(self, -1)
            return key in latest.values or (
                key.endswith("_reason") and key[: -len("_reason")] in latest.unavailable)
        return list.__contains__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        value = self[key]
        return default if value is None else value


def _metric_for_header(header: str) -> str:
    upper = header.upper()
    for fid, short in FIELD_NAMES.items():
        if short == upper:
            return FIELD_METRICS.get(fid, short.lower())
    return header.lower()


def required_fields_available(sample: Any, require_sm_active: bool) -> "tuple[bool, str]":
    """Standard acceptance requires SM active. If it is missing we say so loudly
    rather than filling a 0 or substituting GPU util.

    Accepts a DcgmSample, a DmonResult, or a plain dict.
    """
    if sample is None or (isinstance(sample, list) and not sample):
        return (not require_sm_active), "未取得任何 DCGM 采样"
    if isinstance(sample, DcgmSample):
        value = sample.values.get("sm_active")
        reason = sample.unavailable.get("sm_active", "")
    else:
        value = sample.get("sm_active")
        reason = sample.get("sm_active_reason") or ""
    if value is None:
        return (not require_sm_active), (reason or "DCGM_FI_PROF_SM_ACTIVE 不可用")
    return True, ""
