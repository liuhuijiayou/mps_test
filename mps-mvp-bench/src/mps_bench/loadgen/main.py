"""Standalone load generator process: `python -m mps_bench.loadgen.main`.

Deliberately a separate process from the worker so its CPU cost never shares a
GIL with the CUDA executor, and so it demonstrably holds no CUDA context.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

from .arrival import build_trace
from .driver import DriverConfig, run_closed_loop, run_open_loop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MPS bench load generator (no CUDA context)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--round-index", type=int, required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--mode", required=True, choices=("online", "offline"))
    ap.add_argument("--arrival", required=True,
                    choices=("closed_loop", "fixed_interval", "poisson", "burst_trace"))
    ap.add_argument("--qps", type=float, default=None)
    ap.add_argument("--burst-trace", default="[]")
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--request-timeout-s", type=float, default=5.0)
    ap.add_argument("--drain-timeout-s", type=float, default=30.0)
    ap.add_argument("--unit", default="samples")
    ap.add_argument("--seed", type=int, default=20251001)
    ap.add_argument("--out", required=True, help="requests jsonl 输出路径")
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--start-at-wall", type=float, default=None,
                    help="所有压测端对齐的 wall clock 起始时间，保证两侧窗口一致")
    args = ap.parse_args(argv)

    cfg = DriverConfig(run_id=args.run_id, case_id=args.case_id, round_index=args.round_index,
                       worker_id=args.worker_id, url=args.url, unit=args.unit,
                       request_timeout_s=args.request_timeout_s, concurrency=args.concurrency,
                       drain_timeout_s=args.drain_timeout_s)

    if args.start_at_wall:
        delay = args.start_at_wall - time.time()
        if delay > 0:
            time.sleep(delay)

    t0 = time.monotonic()
    if args.arrival == "closed_loop":
        outcome = run_closed_loop(cfg, duration_s=args.duration_s,
                                  measurement_start_monotonic=t0)
        trace_info: Dict[str, Any] = {"mode": "closed_loop", "planned": 0}
    else:
        trace = build_trace(args.arrival, args.qps, args.duration_s, args.seed,
                            segments=json.loads(args.burst_trace))
        trace_info = trace.as_dict()
        outcome = run_open_loop(cfg, trace, measurement_start_monotonic=t0)

    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in outcome.records:
            fh.write(json.dumps(rec.as_dict(), ensure_ascii=False) + "\n")

    summary = {
        "run_id": args.run_id, "case_id": args.case_id, "round_index": args.round_index,
        "worker_id": args.worker_id, "mode": args.mode, "arrival": args.arrival,
        "unit": args.unit, "duration_s": args.duration_s,
        "planned_count": outcome.planned_count,
        "recorded_count": len(outcome.records),
        "measurement_start_wall": outcome.measurement_start_wall,
        "measurement_window_s": outcome.measurement_end_monotonic - outcome.measurement_start_monotonic,
        "trace": trace_info, "notes": outcome.notes,
    }
    with open(args.summary_out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
