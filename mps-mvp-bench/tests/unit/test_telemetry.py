"""Telemetry parsing: unavailable metrics must be null with a reason, never 0."""

import pytest

from mps_bench.telemetry import dcgm, nvml


def test_parse_scalar_sentinels_become_none():
    for raw in ("N/A", "[N/A]", "Not Supported", "[Not Supported]", "", "-", "nan"):
        value, reason = dcgm.parse_scalar(raw)
        assert value is None
        assert reason  # must state why
        assert value != 0


def test_parse_scalar_numeric_sentinel():
    value, reason = dcgm.parse_scalar("9223372036854775794")
    assert value is None and "哨兵" in reason


def test_parse_scalar_valid():
    value, reason = dcgm.parse_scalar("0.84")
    assert value == pytest.approx(0.84) and reason == ""


def test_parse_dmon_maps_fields_to_distinct_metrics():
    text = ("# Entity   SMACT   SMOCC   DRAMA\n"
            "    GPU 0   0.812   0.401   0.233\n")
    sample = dcgm.parse_dmon(text, fields=[1002, 1003, 1005])
    assert sample["sm_active"] == pytest.approx(0.812)
    assert sample["sm_occupancy"] == pytest.approx(0.401)
    assert sample["dram_active"] == pytest.approx(0.233)
    # these are three different things; none of them is GPU utilization
    assert "gpu_util_device_busy_pct" not in sample


def test_parse_dmon_blank_is_none_with_reason():
    text = ("# Entity   SMACT   SMOCC   DRAMA\n"
            "    GPU 0     N/A   0.401   0.233\n")
    sample = dcgm.parse_dmon(text, fields=[1002, 1003, 1005])
    assert sample["sm_active"] is None
    assert sample["sm_active_reason"]
    assert sample["sm_occupancy"] == pytest.approx(0.401)


def test_required_fields_available():
    ok, reason = dcgm.required_fields_available({"sm_active": 0.5}, require_sm_active=True)
    assert ok and reason == ""
    bad, reason = dcgm.required_fields_available(
        {"sm_active": None, "sm_active_reason": "BLANK"}, require_sm_active=True)
    assert not bad and "BLANK" in reason
    # when not required, absence is tolerated but still reported
    ok2, _ = dcgm.required_fields_available({"sm_active": None}, require_sm_active=False)
    assert ok2


def test_nvml_gpu_util_is_labelled_device_busy():
    row = "GPU-abc, 55, 12000, 12576, 24576, 61, 120.5, 1410, Not Active"
    sample = nvml.parse_query(row)
    assert sample.gpu_util_device_busy_pct == 55
    assert sample.memory_used_mib == 12000
    assert sample.memory_total_mib == 24576
    # explicitly not an SM-activity metric
    assert not hasattr(sample, "sm_active")


def test_nvml_not_supported_fields_become_none():
    row = "GPU-abc, [N/A], 12000, 12576, 24576, [Not Supported], [N/A], 1410, [N/A]"
    sample = nvml.parse_query(row)
    assert sample.gpu_util_device_busy_pct is None
    assert sample.temperature_c is None
    assert sample.power_w is None
    assert sample.memory_used_mib == 12000


def test_nvml_core_query_excludes_optional_throttle_field():
    """Regression: an unsupported optional field must not be able to fail the
    whole query and blank out GPU util / memory for the entire run."""
    assert "clocks.throttle_reasons.active" not in nvml.CORE_QUERY
    for required in ("uuid", "utilization.gpu", "memory.used", "memory.free",
                     "memory.total", "temperature.gpu", "power.draw", "clocks.sm"):
        assert required in nvml.CORE_QUERY
    assert nvml.QUERY == nvml.CORE_QUERY


def test_nvml_parses_core_only_row_without_throttle_column():
    """Driver 580.x rejecting the throttle field -> 8-column output must parse."""
    row = "GPU-abc, 73, 9001, 15575, 24576, 58, 130.25, 1395"
    sample = nvml.parse_query(row)
    assert sample is not None
    assert sample.gpu_util_device_busy_pct == 73
    assert sample.memory_used_mib == 9001
    assert sample.memory_free_mib == 15575
    assert sample.memory_total_mib == 24576
    assert sample.throttle_reasons is None
    assert sample.unavailable == {}


def test_nvml_falls_back_to_core_query_when_optional_field_rejected(monkeypatch):
    calls = []

    class _Res:
        def __init__(self, ok, stdout=""):
            self.ok = ok
            self.stdout = stdout

    def fake_run(argv, **kwargs):
        fields = [a for a in argv if a.startswith("--query-gpu=")][0]
        calls.append(fields)
        if "throttle" in fields:
            return _Res(False, "")  # driver rejects the whole query
        return _Res(True, "GPU-abc, 73, 9001, 15575, 24576, 58, 130.25, 1395")

    monkeypatch.setattr(nvml, "run", fake_run)
    monkeypatch.setattr(nvml, "_throttle_supported", None)

    sample = nvml.sample_device("GPU-abc")
    assert sample is not None
    assert sample.gpu_util_device_busy_pct == 73
    assert sample.memory_used_mib == 9001
    assert len(calls) == 2  # probed once, then fell back

    # the unsupported field is not retried on every sample
    calls.clear()
    sample2 = nvml.sample_device("GPU-abc")
    assert sample2.gpu_util_device_busy_pct == 73
    assert len(calls) == 1
    assert all("throttle" not in c for c in calls)


def test_collector_refuses_to_substitute_gpu_util(monkeypatch):
    from mps_bench.telemetry.collector import Collector
    col = Collector(gpu_uuid="GPU-abc", csv_path="/tmp/does-not-matter.csv",
                    require_sm_metrics=True)
    monkeypatch.setattr(col, "_probe_dcgm_raw",
                        lambda: {"available": False, "reason": "dcgmi 不存在"})
    with pytest.raises(RuntimeError) as exc:
        col.probe_dcgm()
    msg = str(exc.value)
    assert "SM" in msg
    assert "GPU util" in msg or "不得" in msg


def test_collector_degraded_mode_records_reason(monkeypatch, tmp_path):
    from mps_bench.telemetry.collector import Collector
    col = Collector(gpu_uuid="GPU-abc", csv_path=str(tmp_path / "m.csv"),
                    require_sm_metrics=False, allow_degraded=True)
    monkeypatch.setattr(col, "_probe_dcgm_raw",
                        lambda: {"available": False, "reason": "dcgmi 不存在"})
    status = col.probe_dcgm()
    assert status["sm_active_available"] is False
    assert "dcgmi" in status["dcgm_reason"]
