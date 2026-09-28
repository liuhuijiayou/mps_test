"""Statistics rules: quantiles, timeout accounting, deltas, round aggregation."""

import math

import pytest

from mps_bench.reporting import stats


def _records(n=100, base_ms=10.0, status="ok", worker="a", start=0.0, unit="images",
             samples=1):
    out = []
    for i in range(n):
        planned = start + i * 0.01
        out.append({"run_id": "r", "case_id": "c", "worker_id": worker,
                    "request_id": i, "planned_arrival_s": planned,
                    "send_s": planned, "response_s": planned + (base_ms + i) / 1000.0,
                    "status": status, "unit": unit, "samples": samples,
                    "gpu_ms": base_ms, "service_recv_s": planned,
                    "exec_start_s": planned, "gpu_done_s": planned})
    return out


# ------------------------------ quantiles ------------------------------ #

def test_quantile_nearest_rank():
    data = list(range(1, 101))
    assert stats.quantile(data, 0.5) == 50
    assert stats.quantile(data, 0.99) == 99
    assert stats.quantile(data, 1.0) == 100


def test_quantile_empty_is_none_not_zero():
    assert stats.quantile([], 0.5) is None


def test_quantile_single_sample():
    assert stats.quantile([7.0], 0.99) == 7.0


# ------------------------------ timeouts / errors ------------------------------ #

def test_timeouts_excluded_from_latency_but_counted():
    recs = _records(90) + _records(10, status="timeout")
    for i, r in enumerate(recs[90:]):
        r["request_id"] = 1000 + i
    st = stats.summarize_workload(recs, worker_id="a", role="high", unit="images",
                                  window_s=10.0)
    assert st.cohort_size == 100
    assert st.success == 90
    assert st.timeouts == 10
    assert st.latency["n"] == 90  # timeouts must not silently become fast successes


def test_not_sent_and_rejected_counted_separately():
    recs = (_records(5) + _records(3, status="rejected") +
            _records(2, status="not_sent") + _records(1, status="output_error"))
    for i, r in enumerate(recs):
        r["request_id"] = i
    st = stats.summarize_workload(recs, worker_id="a", role="high", unit="images",
                                  window_s=1.0)
    assert (st.success, st.rejected, st.not_sent, st.output_errors) == (5, 3, 2, 1)
    assert st.cohort_size == 11


def test_p999_requires_enough_samples():
    st = stats.summarize_workload(_records(50), worker_id="a", role="high",
                                  unit="images", window_s=1.0, min_samples_p999=1000)
    assert st.latency["p999"] is None
    assert st.latency["p999_status"] == "insufficient_samples"

    st2 = stats.summarize_workload(_records(50), worker_id="a", role="high",
                                   unit="images", window_s=1.0, min_samples_p999=10)
    assert st2.latency["p999"] is not None
    assert st2.latency["p999_status"] == "ok"


def test_e2e_measured_from_planned_arrival():
    """Coordinated omission guard: latency includes the send delay."""
    rec = _records(1)[0]
    rec["planned_arrival_s"] = 0.0
    rec["send_s"] = 1.0          # driver was late by 1s
    rec["response_s"] = 1.05
    st = stats.summarize_workload([rec], worker_id="a", role="high", unit="images",
                                  window_s=1.0)
    assert st.latency["p50"] == pytest.approx(1050.0, rel=1e-6)


def test_throughput_units_and_requests():
    recs = _records(20, samples=8)
    st = stats.summarize_workload(recs, worker_id="a", role="high", unit="images",
                                  window_s=10.0)
    assert st.throughput_req_s == pytest.approx(2.0)
    assert st.throughput_units_s == pytest.approx(16.0)


def test_slo_throughput_only_counts_requests_under_threshold():
    recs = _records(10, base_ms=10.0)   # latencies 10,11,...,19 ms
    st = stats.summarize_workload(recs, worker_id="a", role="high", unit="images",
                                  window_s=1.0, slo_latency_ms=15.0)
    # 10..15 ms inclusive = 6 requests meet the SLO; 16..19 ms do not
    assert st.slo_throughput_req_s == pytest.approx(6.0)
    assert st.throughput_req_s == pytest.approx(10.0)


# ------------------------------ deltas ------------------------------ #

def test_latency_delta_sign():
    # worse latency -> positive delta
    assert stats.latency_delta(120.0, 100.0) == pytest.approx(0.2)
    assert stats.latency_delta(80.0, 100.0) == pytest.approx(-0.2)
    assert stats.latency_delta(None, 100.0) is None
    assert stats.latency_delta(100.0, 0.0) is None


def test_throughput_delta_sign():
    assert stats.throughput_delta(120.0, 100.0) == pytest.approx(0.2)
    assert stats.throughput_delta(None, 100.0) is None


def test_weighted_speedup():
    assert stats.weighted_speedup([(100.0, 100.0), (50.0, 100.0)]) == pytest.approx(1.5)
    assert stats.weighted_speedup([]) is None
    assert stats.weighted_speedup([(10.0, 0.0)]) is None


def test_combine_throughput_refuses_cross_unit_sum():
    ok = stats.combine_throughput([("images", 10.0), ("images", 5.0)])
    assert ok["total"] == pytest.approx(15.0) and ok["unit"] == "images"
    mixed = stats.combine_throughput([("images", 10.0), ("samples", 5.0)])
    assert mixed["total"] is None
    assert "不可合计" in mixed["reason"]


# ------------------------------ rounds ------------------------------ #

def test_aggregate_rounds_pooled_vs_mean_of_p99():
    rounds = []
    for k in range(5):
        rounds.append({"latency_samples_ms": [10.0 + k] * 90 + [500.0 + k * 100] * 10,
                       "throughput_units_s": 100.0 + k})
    agg = stats.aggregate_rounds(rounds)
    assert agg["rounds"] == 5
    assert agg["pooled_p99_ms"] is not None
    assert agg["per_round_p99_mean_ms"] is not None
    assert agg["pooled_p99_ms"] != agg["per_round_p99_mean_ms"]
    assert "不等于整体 P99" in agg["per_round_p99_note"]


def test_round_ci_uses_rounds_as_unit():
    rounds = [{"latency_samples_ms": [10.0], "throughput_units_s": v}
              for v in (100.0, 102.0, 98.0, 101.0, 99.0)]
    agg = stats.aggregate_rounds(rounds)
    ci = agg["throughput_units_s_ci95"]
    assert ci["n"] == 5
    assert ci["low"] < agg["throughput_units_s_mean"] < ci["high"]


def test_ci_single_round_is_none():
    agg = stats.aggregate_rounds([{"latency_samples_ms": [1.0], "throughput_units_s": 1.0}])
    assert agg["throughput_units_s_ci95"]["low"] is None
    assert "至少 2 轮" in agg["throughput_units_s_ci95"]["reason"]


def test_degraded_rounds_are_kept():
    rounds = [{"latency_samples_ms": [10.0] * 10, "throughput_units_s": 100.0},
              {"latency_samples_ms": [900.0] * 10, "throughput_units_s": 3.0}]
    agg = stats.aggregate_rounds(rounds)
    assert agg["rounds"] == 2
    assert agg["throughput_units_s_min"] == pytest.approx(3.0)


def test_t_ci95_known_value():
    lo, hi = stats.t_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo == pytest.approx(3.0 - 2.776 * (math.sqrt(2.5) / math.sqrt(5)), rel=1e-6)
    assert hi == pytest.approx(3.0 + 2.776 * (math.sqrt(2.5) / math.sqrt(5)), rel=1e-6)


def test_cohort_attribution_by_planned_arrival():
    recs = _records(10, start=0.0)
    st = stats.summarize_workload(recs, worker_id="a", role="high", unit="images",
                                  window_s=1.0, window_start_s=0.0, window_end_s=0.05)
    assert st.cohort_size == 5   # only requests planned inside the window
