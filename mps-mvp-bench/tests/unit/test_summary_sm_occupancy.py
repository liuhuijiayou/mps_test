"""`sm_occupancy` must reach summary.csv alongside SM active / GPU util.

These are three semantically different metrics; the summary must carry all of
them rather than silently dropping occupancy.
"""

from __future__ import annotations

import csv
import json
import os

import pytest

from mps_bench.reporting.report import SUMMARY_COLUMNS, write_summary_csv
from mps_bench.summarize import _metric_means, build_summary

_METRICS_CSV = (
    "wall_ts,monotonic_s,gpu_uuid,gpu_util_device_busy_pct,sm_active,sm_occupancy,"
    "dram_active,memory_used_mib,memory_free_mib,memory_total_mib,temperature_c,"
    "power_w,sm_clock_mhz,throttle_reasons,unavailable_metrics,phase\n"
    "1.0,0.0,GPU-abc,60,0.50,0.30,0.10,8000,16576,24576,55,120,1400,,,measure\n"
    "2.0,1.0,GPU-abc,80,0.70,0.50,0.20,10000,14576,24576,57,140,1400,,,measure\n"
)


def _write_metrics(run_dir: str) -> None:
    with open(os.path.join(run_dir, "gpu_metrics.csv"), "w", encoding="utf-8") as fh:
        fh.write(_METRICS_CSV)


def test_metric_means_includes_sm_occupancy(tmp_path):
    _write_metrics(str(tmp_path))
    means = _metric_means(os.path.join(str(tmp_path), "gpu_metrics.csv"))
    assert means["sm_occupancy"] == pytest.approx(0.40)
    assert means["sm_active"] == pytest.approx(0.60)
    assert means["gpu_util_device_busy_pct"] == pytest.approx(70.0)
    assert means["dram_active"] == pytest.approx(0.15)
    assert means["memory_used_mib"] == pytest.approx(9000.0)
    assert means["memory_used_mib_max"] == pytest.approx(10000.0)
    # occupancy is a distinct metric, not aliased to SM active
    assert means["sm_occupancy"] != means["sm_active"]


def test_sm_occupancy_column_exists_in_summary_csv():
    assert "sm_occupancy_mean" in SUMMARY_COLUMNS
    assert "sm_active_mean" in SUMMARY_COLUMNS
    assert "gpu_util_mean" in SUMMARY_COLUMNS
    assert "memory_used_mib_mean" in SUMMARY_COLUMNS


def test_sm_occupancy_flows_from_metrics_into_summary_csv(tmp_path):
    run_dir = str(tmp_path)
    _write_metrics(run_dir)
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_id": "R1"}, fh)
    with open(os.path.join(run_dir, "telemetry_status.json"), "w", encoding="utf-8") as fh:
        json.dump({"sm_active_available": True}, fh)
    with open(os.path.join(run_dir, "case_results.json"), "w", encoding="utf-8") as fh:
        json.dump([{"case": "HH70", "rounds": [
            {"round": 0, "healthy": True, "notes": [], "mps_evidence": {},
             "stats": {"a": {"role": "high", "unit": "images", "cohort_size": 10,
                             "success": 10, "output_errors": 0}}}]}], fh)

    built = build_summary(run_dir)
    row = built["summary_rows"][0]
    assert row["sm_occupancy_mean"] == pytest.approx(0.40)
    assert row["sm_active_mean"] == pytest.approx(0.60)
    assert row["gpu_util_mean"] == pytest.approx(70.0)
    assert row["memory_used_mib_mean"] == pytest.approx(9000.0)

    out = os.path.join(run_dir, "summary.csv")
    write_summary_csv(out, built["summary_rows"])
    with open(out, "r", encoding="utf-8") as fh:
        written = list(csv.DictReader(fh))
    assert written[0]["sm_occupancy_mean"] == "0.4"
    assert written[0]["sm_active_mean"] == "0.6"
