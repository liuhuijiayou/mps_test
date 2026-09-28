"""Statistics with explicit, auditable conventions.

Conventions (documented in docs/METHODOLOGY.zh-CN.md):
* Quantiles use the nearest-rank (inverse CDF) method on the *cohort* of requests
  whose planned arrival falls inside the measurement window. Every quantile keeps
  its sample count.
* P99.9 is only emitted when the cohort is at least `min_samples_p999`; otherwise
  it is reported as insufficient, never as a number.
* Throughput counts SUCCESSFULLY COMPLETED work within the stated window, with
  the unit named. Rejected/timeout/incomplete requests are counted separately so
  the tail is never lost.
* Averaging per-round P99s does NOT produce an overall P99. We report both: the
  pooled P99 over all rounds, and the per-round P99 list plus a round-level CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..loadgen.record import (STATUS_INCOMPLETE, STATUS_NOT_SENT, STATUS_OK,
                              STATUS_OUTPUT_ERROR, STATUS_REJECTED, STATUS_TIMEOUT)

INSUFFICIENT = "insufficient_samples"


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank quantile. Returns None for an empty sample."""
    if not values:
        return None
    if not 0 < q <= 1:
        raise ValueError("q 必须在 (0,1]")
        # nearest-rank: ceil(q*n)-th smallest
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def stddev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def t_ci95(values: Sequence[float]) -> "tuple[Optional[float], Optional[float]]":
    """Round-level 95% CI using a small-sample t table.

    Rounds are the independent unit of repetition: a million requests in one
    round are NOT a million independent experiments.
    """
    block = t_ci95_block(values)
    return block["low"], block["high"]


def t_ci95_block(values: Sequence[float]) -> Dict[str, Any]:
    n = len(values)
    if n < 2:
        return {"mean": (values[0] if n == 1 else None), "half_width": None,
                "low": None, "high": None, "n": n,
                "reason": "轮级置信区间需至少 2 轮，单轮结果不给出 CI"}
    # keyed by degrees of freedom (n-1); two-sided 95% critical values
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
               8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
               14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
               20: 2.086, 25: 2.060, 30: 2.042}
    df = n - 1
    tval = t_table.get(df)
    if tval is None:
        # No exact row: fall back to the nearest tabulated df that is <= ours, so
        # the interval is never narrower than the table justifies.
        candidates = [k for k in t_table if k <= df]
        tval = t_table[max(candidates)] if candidates else 12.706
        if df > 30:
            tval = 1.96
    m = sum(values) / n
    sd = stddev(values) or 0.0
    half = tval * sd / math.sqrt(n)
    return {"mean": m, "half_width": half, "low": m - half, "high": m + half, "n": n,
            "reason": ""}


# --------------------------------------------------------------------------- #
# cohort accounting
# --------------------------------------------------------------------------- #

@dataclass
class Cohort:
    """Requests attributed to the measurement window.

    Attribution is by planned arrival time (open loop) or actual send time
    (closed loop). A request that arrived inside the window but finished after it
    still belongs to the cohort -- that is what keeps the tail honest.
    """
    window_start_s: float
    window_end_s: float
    records: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, rec: Dict[str, Any]) -> bool:
        key = rec.get("planned_arrival_s")
        if key is None:
            key = rec.get("actual_send_s", rec.get("send_s"))
        if key is None:
            return False
        if self.window_start_s <= key < self.window_end_s:
            self.records.append(rec)
            return True
        return False


@dataclass
class WorkloadStats:
    worker_id: str
    role: str
    unit: str
    window_s: float
    cohort_size: int
    success: int
    errors: int
    timeouts: int
    rejected: int
    not_sent: int
    incomplete: int
    output_errors: int
    success_units: int
    throughput_req_s: Optional[float]
    throughput_units_s: Optional[float]
    slo_throughput_req_s: Optional[float]
    latency: Dict[str, Any]
    gpu_ms: Dict[str, Any]
    queue_s: Dict[str, Any]
    send_delay_s: Dict[str, Any]
    checksum_ok: bool

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @property
    def success_rate(self) -> Optional[float]:
        return (self.success / self.cohort_size) if self.cohort_size else None

    @property
    def error_rate(self) -> Optional[float]:
        if not self.cohort_size:
            return None
        return (self.errors + self.output_errors) / self.cohort_size

    @property
    def timeout_rate(self) -> Optional[float]:
        return (self.timeouts / self.cohort_size) if self.cohort_size else None

    @property
    def reject_rate(self) -> Optional[float]:
        return (self.rejected / self.cohort_size) if self.cohort_size else None


def _quantile_block(values: Sequence[float], min_samples_p999: int,
                    scale: float = 1.0) -> Dict[str, Any]:
    n = len(values)
    block: Dict[str, Any] = {"n": n}
    for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        value = quantile(values, q)
        block[label] = None if value is None else value * scale
    block["mean"] = None if not values else (sum(values) / n) * scale
    block["max"] = None if not values else max(values) * scale
    if n >= min_samples_p999:
        block["p999"] = (quantile(values, 0.999) or 0.0) * scale
        block["p999_status"] = "ok"
    else:
        block["p999"] = None
        block["p999_status"] = INSUFFICIENT
        block["p999_required_samples"] = min_samples_p999
    return block


def _e2e(rec: Dict[str, Any]) -> Optional[float]:
    """End-to-end latency from the PLANNED arrival when one exists.

    Recomputed here (rather than trusting a pre-stored field) so that records
    loaded from any source obey the same anti-coordinated-omission rule.
    """
    if rec.get("e2e_s") is not None:
        return rec["e2e_s"]
    start = rec.get("planned_arrival_s")
    if start is None:
        start = rec.get("actual_send_s", rec.get("send_s"))
    end = rec.get("response_done_s", rec.get("response_s"))
    if start is None or end is None:
        return None
    return end - start


def summarize_workload(records: Iterable[Dict[str, Any]],
                       worker_id: str,
                       role: str,
                       unit: str,
                       window_s: float,
                       window_start_s: float = 0.0,
                       window_end_s: Optional[float] = None,
                       min_samples_p999: int = 100000,
                       slo_latency_ms: Optional[float] = None) -> WorkloadStats:
    end = window_start_s + window_s if window_end_s is None else window_end_s
    cohort = Cohort(window_start_s=window_start_s, window_end_s=end)
    outside = 0
    for rec in records:
        if rec.get("worker_id") != worker_id:
            continue
        if rec.get("status") == STATUS_INCOMPLETE:
            # no arrival timestamp; attribute to this cohort explicitly
            cohort.records.append(rec)
            continue
        if not cohort.add(rec):
            outside += 1

    recs = cohort.records
    success = [r for r in recs if r.get("status") == STATUS_OK]
    e2e = [_e2e(r) for r in success]
    e2e = [v for v in e2e if v is not None]
    gpu_ms = [r["gpu_ms"] for r in success if r.get("gpu_ms") is not None]
    queue = [r["queue_s"] for r in success if r.get("queue_s") is not None]
    delays = [r["send_delay_s"] for r in recs if r.get("send_delay_s") is not None]

    counts = {status: sum(1 for r in recs if r.get("status") == status)
              for status in (STATUS_OK, "error", STATUS_TIMEOUT, STATUS_REJECTED,
                             STATUS_NOT_SENT, STATUS_INCOMPLETE, STATUS_OUTPUT_ERROR)}
    success_units = sum(int(r.get("samples") or 0) for r in success)

    slo_success = None
    if slo_latency_ms is not None:
        slo_success = sum(1 for r in success
                          if (_e2e(r) is not None and _e2e(r) * 1000 <= slo_latency_ms))

    return WorkloadStats(
        worker_id=worker_id, role=role, unit=unit, window_s=window_s,
        cohort_size=len(recs),
        success=counts[STATUS_OK], errors=counts["error"], timeouts=counts[STATUS_TIMEOUT],
        rejected=counts[STATUS_REJECTED], not_sent=counts[STATUS_NOT_SENT],
        incomplete=counts[STATUS_INCOMPLETE], output_errors=counts[STATUS_OUTPUT_ERROR],
        success_units=success_units,
        throughput_req_s=(counts[STATUS_OK] / window_s) if window_s > 0 else None,
        throughput_units_s=(success_units / window_s) if window_s > 0 else None,
        slo_throughput_req_s=(slo_success / window_s) if (slo_success is not None and window_s > 0)
        else None,
        latency=_quantile_block(e2e, min_samples_p999, scale=1000.0),  # ms
        gpu_ms=_quantile_block(gpu_ms, min_samples_p999),
        queue_s=_quantile_block(queue, min_samples_p999, scale=1000.0),
        send_delay_s=_quantile_block(delays, min_samples_p999, scale=1000.0),
        checksum_ok=all(r.get("status") != STATUS_OUTPUT_ERROR for r in recs),
    )


# --------------------------------------------------------------------------- #
# deltas / speedup
# --------------------------------------------------------------------------- #

def latency_delta(p99_test: Optional[float], p99_reference: Optional[float]) -> Optional[float]:
    if not p99_test or not p99_reference:
        return None
    return p99_test / p99_reference - 1.0


def throughput_delta(q_test: Optional[float], q_reference: Optional[float]) -> Optional[float]:
    if q_test is None or not q_reference:
        return None
    return q_test / q_reference - 1.0


def weighted_speedup(pairs: Sequence["tuple[Optional[float], Optional[float]]"]) -> Optional[float]:
    """sum_i (colocated_throughput_i / solo_throughput_i).

    Only meaningful for capacity experiments with comparable definitions; the
    solo denominator configuration must be stated alongside the number.
    An empty list returns None rather than 0.0 -- "no data" is not "no speedup".
    """
    if not pairs:
        return None
    total = 0.0
    for test, solo in pairs:
        if test is None or not solo:
            return None
        total += test / solo
    return total


def combine_throughput(stats: Sequence[Any]) -> Dict[str, Any]:
    """Sum throughput ONLY across identical units/semantics.

    images/s and samples/s are never added. Heterogeneous pairs get per-task
    numbers plus (optionally) a normalized weighted speedup computed elsewhere.

    Accepts either WorkloadStats objects or (unit, throughput) tuples.
    """
    normalized: List[Dict[str, Any]] = []
    for item in stats:
        if isinstance(item, tuple):
            unit, value = item
            normalized.append({"worker_id": None, "role": None, "unit": unit,
                               "throughput_req_s": None, "throughput_units_s": value})
        else:
            normalized.append({"worker_id": item.worker_id, "role": item.role,
                               "unit": item.unit,
                               "throughput_req_s": item.throughput_req_s,
                               "throughput_units_s": item.throughput_units_s})
    units = {entry["unit"] for entry in normalized}
    if len(units) == 1 and normalized:
        unit = units.pop()
        total_units = sum(e["throughput_units_s"] or 0.0 for e in normalized)
        return {"combinable": True, "unit": unit, "total": total_units,
                "total_throughput_units_s": total_units,
                "total_throughput_req_s": sum(e["throughput_req_s"] or 0.0
                                              for e in normalized),
                "reason": "", "per_task": normalized}
    return {"combinable": False, "unit": None, "total": None,
            "total_throughput_units_s": None,
            "reason": "任务单位/语义不同（如 images/s 与 samples/s），不可合计；"
                      "请使用各自吞吐与归一化 weighted speedup",
            "per_task": normalized}


def aggregate_rounds(rounds: Sequence[Dict[str, Any]],
                     min_samples_p999: int = 100000) -> Dict[str, Any]:
    """Round-level aggregation.

    Each element is `{latency_samples_ms: [...], throughput_units_s: float}`.

    `pooled_p99_ms` is computed over ALL requests from all rounds. The per-round
    mean of P99 is reported separately and explicitly labeled -- it is not the
    overall P99. Degraded rounds are kept: silently dropping the bad round is how
    a colocation result gets flattered.
    """
    pooled: List[float] = []
    per_round_p99: List[Optional[float]] = []
    tps: List[float] = []
    for rnd in rounds:
        samples = list(rnd.get("latency_samples_ms") or [])
        pooled += samples
        per_round_p99.append(quantile(samples, 0.99))
        value = rnd.get("throughput_units_s")
        if value is not None:
            tps.append(float(value))
    p99s = [v for v in per_round_p99 if v is not None]
    return {
        "rounds": len(rounds),
        "pooled_p99_ms": quantile(pooled, 0.99),
        "pooled_p50_ms": quantile(pooled, 0.5),
        "pooled_samples": len(pooled),
        "pooled_p999_ms": (quantile(pooled, 0.999) if len(pooled) >= min_samples_p999
                           else None),
        "pooled_p999_status": ("ok" if len(pooled) >= min_samples_p999 else INSUFFICIENT),
        "per_round_p99_ms": per_round_p99,
        "per_round_p99_mean_ms": mean(p99s),
        "per_round_p99_note": "这是各轮 P99 的均值，不等于整体 P99",
        "per_round_p99_ci95": t_ci95_block(p99s),
        "per_round_throughput_units_s": tps,
        "throughput_units_s_mean": mean(tps),
        "throughput_units_s_min": (min(tps) if tps else None),
        "throughput_units_s_max": (max(tps) if tps else None),
        "throughput_units_s_ci95": t_ci95_block(tps),
        "note": "保留全部轮次（包括退化轮），不剔除异常轮",
    }
