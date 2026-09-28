"""Case discovery, matrix expansion and budget estimation.

Expansion rules:
* a case file is one case; `case.sweep` adds explicitly enumerated variants
  (list of {name, set:{key:value}}). We never take a cartesian product over all
  parameters -- sweeps are curated and auditable.
* `plan` prints the expanded case count and the estimated wall time before
  anything executes, and refuses to run if `measure.total_budget_s` is exceeded.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import config as cfgmod


@dataclass
class ExpandedCase:
    case_id: str
    variant: str
    case_file: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    config: Optional[cfgmod.LoadedConfig] = None

    @property
    def full_id(self) -> str:
        return self.case_id if self.variant == "base" else f"{self.case_id}[{self.variant}]"


def discover_case_files(cases_dir: str, families: Optional[Sequence[str]] = None,
                        ids: Optional[Sequence[str]] = None) -> List[str]:
    paths = sorted(glob.glob(os.path.join(cases_dir, "**", "*.yaml"), recursive=True))
    if families:
        wanted = set(families)
        paths = [p for p in paths if os.path.basename(os.path.dirname(p)) in wanted]
    if ids:
        wanted_ids = set(ids)
        kept = []
        for path in paths:
            raw = cfgmod._read_yaml(path)
            case_id = (raw.get("case") or {}).get("id")
            if case_id in wanted_ids:
                kept.append(path)
        paths = kept
    return paths


def expand(case_files: Sequence[str],
           base_configs: Sequence[str] = (),
           profile: Optional[str] = None,
           overrides: Sequence[str] = (),
           profiles_dir: Optional[str] = None,
           validate_cases: bool = True) -> List[ExpandedCase]:
    expanded: List[ExpandedCase] = []
    for path in case_files:
        raw = cfgmod._read_yaml(path)
        case_block = raw.get("case") or {}
        case_id = case_block.get("id") or os.path.splitext(os.path.basename(path))[0]
        if case_block.get("enabled") is False:
            continue
        sweeps = case_block.get("sweep") or []
        variants: List[Dict[str, Any]] = [{"name": "base", "set": {}}]
        for item in sweeps:
            if not isinstance(item, dict) or "set" not in item:
                raise cfgmod.ConfigError(f"{path}: case.sweep 元素需为 {{name, set}}")
            variants.append({"name": str(item.get("name", "sweep")), "set": item["set"]})
        for variant in variants:
            extra = [f"{k}={_yaml_scalar(v)}" for k, v in (variant["set"] or {}).items()]
            case = ExpandedCase(case_id=case_id, variant=variant["name"], case_file=path,
                                overrides=dict(variant["set"] or {}))
            if validate_cases:
                case.config = cfgmod.load(config_files=base_configs, profile=profile,
                                          case_file=path,
                                          overrides=list(extra) + list(overrides),
                                          profiles_dir=profiles_dir)
            expanded.append(case)
    return expanded


def _yaml_scalar(value: Any) -> str:
    import json
    if isinstance(value, str):
        return value
    return json.dumps(value)


def estimate_duration_s(cases: Sequence[ExpandedCase]) -> Dict[str, Any]:
    """Wall-clock estimate, including per-case startup/teardown overhead."""
    STARTUP_S = 45.0  # container start + warmup handshake + telemetry probe
    TEARDOWN_S = 20.0
    total = 0.0
    rows = []
    for case in cases:
        cfg = case.config
        if cfg is None:
            continue
        per_round = cfg["measure.warmup_s"] + cfg["measure.duration_s"] + cfg["measure.drain_timeout_s"]
        case_total = cfg["measure.repeats"] * per_round + STARTUP_S + TEARDOWN_S
        if cfg["case.fault"] is not None:
            case_total += cfg["faults.repeats"] * (cfg["faults.observe_window_s"]
                                                   + cfg["faults.recovery_timeout_s"] * 0.5)
        total += case_total
        rows.append({"case": case.full_id, "rounds": cfg["measure.repeats"],
                     "per_round_s": per_round, "estimated_s": round(case_total, 1),
                     "mode": cfg["case.mode"], "family": cfg["case.family"]})
    return {"case_count": len(rows), "estimated_total_s": round(total, 1),
            "estimated_total_min": round(total / 60.0, 1), "cases": rows}


def check_budget(estimate: Dict[str, Any], budget_s: Optional[float]) -> Optional[str]:
    if budget_s is None:
        return None
    if estimate["estimated_total_s"] > budget_s:
        return (f"展开用例预计耗时 {estimate['estimated_total_min']} 分钟，"
                f"超过 measure.total_budget_s={budget_s}s。请用 --case-id / --family 选择子集，"
                "或降低 measure.repeats / duration_s。")
    return None


def interleave_order(case_ids: Sequence[str], mode: str, repeats: int,
                     seed: int = 0) -> List["tuple[str, int]"]:
    """Round ordering for comparison runs.

    ABBA alternates direction between rounds so monotonic drift (thermals, other
    host activity) does not systematically favour one mode.
    """
    import random
    plan: List[tuple[str, int]] = []
    ids = list(case_ids)
    for r in range(repeats):
        if mode == "none":
            order = ids
        elif mode == "ABBA":
            order = ids if r % 2 == 0 else list(reversed(ids))
        elif mode == "random":
            order = ids[:]
            random.Random(seed + r).shuffle(order)
        else:
            raise ValueError(f"未知 interleave 模式 {mode}")
        plan += [(cid, r) for cid in order]
    return plan
