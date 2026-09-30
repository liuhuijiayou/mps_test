#!/usr/bin/env python3
"""Build one scenario-level comparison CSV from several run directories.

Each run directory must contain exactly one scenario (B2 / HH70 / HH50), which
is why the runbook starts a separate run per scenario: `gpu_metrics.csv` is
collected per run, so mixing scenarios in one run makes scenario-level GPU
telemetry impossible to attribute correctly.

Unit conventions (do not "fix" these without reading DCGM/NVML docs):
* DCGM SM_ACTIVE / SM_OCCUPANCY / DRAM_ACTIVE are ratios in 0..1  -> x100 for %
* nvidia-smi utilization.gpu is already 0..100                    -> NOT scaled
* GPU util is a device-busy indicator and is NEVER a stand-in for SM active.

Usage:
    python tools/build_comparison.py --out comparison.csv \
        --run B2=results/<b2-run-id> \
        --run HH70=results/<hh70-run-id> \
        --run HH50=results/<hh50-run-id>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

COLUMNS = [
    "scenario", "mps", "memory_policy", "sm_policy",
    "avg_gpu_util_pct", "avg_sm_active_pct", "avg_sm_occupancy_pct",
    "avg_dram_active_pct", "avg_gpu_memory_mib",
    "avg_throughput_a_images_s", "avg_throughput_b_images_s",
    "avg_total_throughput_images_s",
    "avg_latency_a_ms", "avg_latency_b_ms",
    "success_rate_pct", "output_error_count", "rounds", "run_dir",
]


def _read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _pct(value: Optional[float], scale: float) -> Optional[float]:
    return None if value is None else round(value * scale, 3)


def _policy(run_dir: str) -> Dict[str, Any]:
    """Read the actual enforced policy out of the run's own evidence."""
    cases = _read_json(os.path.join(run_dir, "case_results.json"), []) or []
    atp: List[Any] = []
    mem: List[Any] = []
    mps_on = False
    for case in cases:
        for rnd in case.get("rounds", []):
            for key, ev in (rnd.get("mps_evidence") or {}).items():
                rt = (ev or {}).get("worker_runtime") or {}
                if rt.get("mps_active_thread_percentage") is not None:
                    atp.append(rt["mps_active_thread_percentage"])
                    mps_on = True
                if rt.get("mps_pinned_device_mem_limit") is not None:
                    mem.append(rt["mps_pinned_device_mem_limit"])
    sm_policy = ("ATP=" + "/".join(str(v) for v in sorted({str(a) for a in atp}))
                 if atp else "none (non-MPS)")
    mem_policy = ("mem_limit=" + "/".join(sorted({str(m) for m in mem}))
                  if mem else "none (non-MPS)")
    return {"mps": mps_on, "sm_policy": sm_policy, "memory_policy": mem_policy}


def build_row(scenario: str, run_dir: str) -> Dict[str, Any]:
    from mps_bench.summarize import build_summary

    built = build_summary(run_dir)
    rows = built["summary_rows"]
    if not rows:
        raise SystemExit(f"{run_dir}: summary_rows 为空，无法生成对比行")

    # GPU telemetry is run-scoped (one scenario per run), so any row carries it.
    tel = rows[0]
    per_slot: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        per_slot.setdefault(str(r.get("worker_id")), []).append(r)

    def slot_mean(slot: str, key: str) -> Optional[float]:
        return _mean([r.get(key) for r in per_slot.get(slot, [])])

    tp_a = slot_mean("a", "throughput_units_s")
    tp_b = slot_mean("b", "throughput_units_s")
    units = {str(r.get("unit")) for r in rows}
    # images/s and samples/s must never be summed; only combine identical units.
    total = (sum(v for v in (tp_a, tp_b) if v is not None)
             if len(units) == 1 and (tp_a is not None or tp_b is not None) else None)

    success = sum(int(r.get("success") or 0) for r in rows)
    cohort = sum(int(r.get("cohort_size") or 0) for r in rows)
    output_errors = sum(int(r.get("output_errors") or 0) for r in rows)
    rounds = len({r.get("round_index") for r in rows})

    pol = _policy(run_dir)
    return {
        "scenario": scenario,
        "mps": pol["mps"],
        "memory_policy": pol["memory_policy"],
        "sm_policy": pol["sm_policy"],
        # utilization.gpu is already a percentage -> scale 1.0
        "avg_gpu_util_pct": _pct(tel.get("gpu_util_mean"), 1.0),
        # DCGM ratios 0..1 -> percent
        "avg_sm_active_pct": _pct(tel.get("sm_active_mean"), 100.0),
        "avg_sm_occupancy_pct": _pct(tel.get("sm_occupancy_mean"), 100.0),
        "avg_dram_active_pct": _pct(tel.get("dram_active_mean"), 100.0),
        "avg_gpu_memory_mib": _pct(tel.get("memory_used_mib_mean"), 1.0),
        "avg_throughput_a_images_s": None if tp_a is None else round(tp_a, 3),
        "avg_throughput_b_images_s": None if tp_b is None else round(tp_b, 3),
        "avg_total_throughput_images_s": None if total is None else round(total, 3),
        "avg_latency_a_ms": _pct(slot_mean("a", "latency_mean_ms"), 1.0),
        "avg_latency_b_ms": _pct(slot_mean("b", "latency_mean_ms"), 1.0),
        "success_rate_pct": (round(100.0 * success / cohort, 3) if cohort else None),
        "output_error_count": output_errors,
        "rounds": rounds,
        "run_dir": run_dir,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="生成 scenario 级 comparison CSV")
    ap.add_argument("--run", action="append", required=True, metavar="SCENARIO=RUN_DIR",
                    help="如 HH70=results/20260929-hh70；可重复")
    ap.add_argument("--out", default="comparison.csv")
    args = ap.parse_args(argv)

    rows = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run 需要 SCENARIO=RUN_DIR 形式，收到 {spec!r}")
        scenario, run_dir = spec.split("=", 1)
        rows.append(build_row(scenario.strip(), run_dir.strip()))

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    widths = {c: max(len(c), *(len(str(r.get(c))) for r in rows)) for c in COLUMNS}
    print(" | ".join(c.ljust(widths[c]) for c in COLUMNS))
    for row in rows:
        print(" | ".join(str(row.get(c)).ljust(widths[c]) for c in COLUMNS))
    print(f"\ncomparison CSV -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
