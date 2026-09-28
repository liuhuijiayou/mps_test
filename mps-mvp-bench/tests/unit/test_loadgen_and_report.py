"""Arrival traces, request records, summary/report generation."""

import json
import os

import pytest

from mps_bench.loadgen import arrival, record
from mps_bench.reporting import report as reportmod
from mps_bench import summarize


# --------------------------- arrival --------------------------- #

def test_fixed_interval_is_exact():
    times = arrival.fixed_interval(qps=10.0, duration_s=1.0)
    assert len(times) == 10
    assert times[1] - times[0] == pytest.approx(0.1)


def test_poisson_is_reproducible_and_frozen():
    a = arrival.poisson(qps=50.0, duration_s=2.0, seed=7)
    b = arrival.poisson(qps=50.0, duration_s=2.0, seed=7)
    assert a == b
    c = arrival.poisson(qps=50.0, duration_s=2.0, seed=8)
    assert c != a


def test_poisson_mean_rate_is_sane():
    times = arrival.poisson(qps=100.0, duration_s=10.0, seed=1)
    assert 800 < len(times) < 1200


def test_burst_trace_segments():
    times = arrival.burst_trace([{"t": 0.0, "qps": 10.0}, {"t": 1.0, "qps": 100.0}],
                                duration_s=2.0, seed=3)
    early = [t for t in times if t < 1.0]
    late = [t for t in times if t >= 1.0]
    assert len(late) > len(early) * 3


def test_closed_loop_has_no_planned_times():
    tr = arrival.build_trace("closed_loop", qps=None, duration_s=5.0, seed=1)
    assert tr.planned == []
    assert "闭环" in tr.detail["note"]


def test_open_loop_requires_qps():
    with pytest.raises(ValueError):
        arrival.build_trace("poisson", qps=None, duration_s=1.0, seed=1)


# --------------------------- records --------------------------- #

def test_record_derived_fields():
    rec = record.RequestRecord(run_id="r", case_id="c", worker_id="a", request_id=1,
                               planned_arrival_s=0.0, send_s=0.2, service_recv_s=0.21,
                               enqueue_s=0.21, exec_start_s=0.3, gpu_done_s=0.4,
                               response_s=0.45, status="ok", unit="images", samples=8,
                               gpu_ms=100.0)
    d = rec.as_dict()
    assert d["send_delay_s"] == pytest.approx(0.2)
    assert d["e2e_s"] == pytest.approx(0.45)     # from PLANNED arrival
    assert d["service_s"] == pytest.approx(0.25)
    assert d["queue_s"] == pytest.approx(0.09)


def test_record_e2e_falls_back_to_send_when_no_plan():
    rec = record.RequestRecord(run_id="r", case_id="c", worker_id="a", request_id=1,
                               planned_arrival_s=None, send_s=1.0, response_s=1.5,
                               status="ok", unit="images", samples=1)
    assert rec.as_dict()["e2e_s"] == pytest.approx(0.5)


def test_record_writer_roundtrip(tmp_path):
    path = tmp_path / "req.jsonl"
    with record.RequestWriter(str(path)) as w:
        w.write(record.RequestRecord(run_id="r", case_id="c", worker_id="a",
                                     request_id=1, planned_arrival_s=0.0, send_s=0.0,
                                     response_s=0.01, status="ok", unit="images",
                                     samples=1))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "ok"


def test_status_vocabulary_is_closed():
    assert set(record.STATUSES) == {"ok", "error", "timeout", "rejected", "output_error",
                                    "not_sent", "incomplete"}
    with pytest.raises(ValueError):
        record.RequestRecord(run_id="r", case_id="c", worker_id="a", request_id=1,
                             planned_arrival_s=0.0, send_s=0.0, response_s=0.1,
                             status="weird", unit="images", samples=1)


# --------------------------- reporting --------------------------- #

def test_summary_csv_has_stable_columns(tmp_path):
    rows = [{"run_id": "r", "case_id": "HH70", "worker_id": "a",
             "latency_p99_ms": 12.0, "status": "PASS"}]
    path = tmp_path / "summary.csv"
    reportmod.write_summary_csv(str(path), rows)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(reportmod.SUMMARY_COLUMNS)


def test_svg_chart_leaves_gaps_for_missing_values():
    svg = reportmod._svg_line_chart("sm_active", [
        {"name": "sm_active",
         "points": [(0.0, 1.0), (1.0, 2.0), (2.0, None), (3.0, 3.0), (4.0, 4.0)]}])
    # a break means more than one polyline and no interpolation across the hole
    assert svg.count("<polyline") == 2


def test_svg_chart_all_none_states_unavailable():
    svg = reportmod._svg_line_chart("sm_active", [
        {"name": "sm_active", "points": [(0.0, None), (1.0, None)]}])
    assert "不可用" in svg
    assert "<polyline" not in svg


def test_report_html_renders_statuses(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    mvp = [{"mvp_item": "SM 隔离各 50%", "case_id": "HH-S", "status": "SKIP",
            "config_summary": "-", "reason": "驱动不支持", "evidence": "help 输出"},
           {"mvp_item": "SM 各 70%", "case_id": "HH70", "status": "PASS",
            "config_summary": "ATP=70", "reason": "", "evidence": "summary.csv"}]
    path = reportmod.render_html(
        run_dir=str(run_dir), run_id="run1", summary_rows=[],
        mvp_status=mvp, events=[], capability={"capabilities": []}, environment={},
        telemetry_status={"sm_active_available": False, "dcgm_reason": "dcgmi 不存在"},
        findings={"positive": [], "negative": [], "not_reproduced": [], "sweeps": []},
        fault_summary=[], memory_summary=[], not_verified=["静态 SM 分区未验证"])
    html = open(path, encoding="utf-8").read()
    assert "SKIP" in html and "PASS" in html
    assert "静态 SM 分区未验证" in html
    assert "SM active" in html  # missing-metric banner


def test_build_summary_on_empty_dir(tmp_path):
    built = summarize.build_summary(str(tmp_path))
    assert built["summary_rows"] == []
    # every MVP item must be explicitly NOT_RUN, never silently absent
    assert len(built["mvp_status"]) == len(summarize.MVP_ITEMS)
    assert all(item["status"] == "NOT_RUN" for item in built["mvp_status"])
    assert any("NOT_RUN" in n for n in built["not_verified"])

def test_build_summary_marks_inconclusive_without_mps_evidence(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    (tmp_path / "case_results.json").write_text(json.dumps([{
        "case": "HH70",
        "rounds": [{"round": 0, "healthy": True, "notes": [],
                    "mps_evidence": {"client_a": {"mps_attachment_proven": False}},
                    "stats": {"a": {"role": "high", "unit": "images", "window_s": 10,
                                    "cohort_size": 10, "success": 10,
                                    "latency": {"p99": 11.0}}}}]}]),
        encoding="utf-8")
    built = summarize.build_summary(str(tmp_path))
    row = built["summary_rows"][0]
    assert row["status"] == "INCONCLUSIVE"
    assert "MPS" in row["status_reason"]


def test_deltas_computed_against_b0_and_b2(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")

    def rnd(p99, thr):
        return {"round": 0, "healthy": True, "notes": [],
                "mps_evidence": {"client_a": {"mps_attachment_proven": True}},
                "stats": {"a": {"role": "high", "unit": "images", "window_s": 10,
                                "cohort_size": 10, "success": 10,
                                "throughput_units_s": thr,
                                "latency": {"p99": p99}}}}

    (tmp_path / "case_results.json").write_text(json.dumps([
        {"case": "B0", "rounds": [rnd(10.0, 100.0)]},
        {"case": "B2", "rounds": [rnd(20.0, 50.0)]},
        {"case": "HH70", "rounds": [rnd(15.0, 80.0)]}]), encoding="utf-8")
    built = summarize.build_summary(str(tmp_path))
    hh = next(r for r in built["summary_rows"] if r["case_id"] == "HH70")
    assert hh["latency_delta_vs_B0"] == pytest.approx(0.5)
    assert hh["latency_delta_vs_B2"] == pytest.approx(-0.25)
    assert hh["throughput_delta_vs_B0"] == pytest.approx(-0.2)
    assert hh["throughput_delta_vs_B2"] == pytest.approx(0.6)
