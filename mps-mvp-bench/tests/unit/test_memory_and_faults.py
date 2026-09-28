"""Memory quota classification and fault bookkeeping."""

import pytest

from mps_bench import faults as faultmod
from mps_bench import memory_probe_runner as mem


GIB = 1024 ** 3


# --------------------------- memory quota --------------------------- #

def test_expected_quota_oom_when_near_limit_and_device_has_room():
    cls, reason = mem.classify_oom(allocated_bytes=11 * GIB, quota_bytes=12 * GIB,
                                   cuda_error="CUDA_ERROR_OUT_OF_MEMORY",
                                   device_free_bytes=10 * GIB)
    assert cls == "expected_quota_oom"
    assert "配额" in reason


def test_unexpected_oom_when_device_itself_is_full():
    cls, reason = mem.classify_oom(allocated_bytes=11 * GIB, quota_bytes=12 * GIB,
                                   cuda_error="CUDA_ERROR_OUT_OF_MEMORY",
                                   device_free_bytes=64 * 1024 * 1024)
    assert cls == "unexpected_oom"
    assert "物理" in reason


def test_unexpected_oom_far_below_quota():
    cls, _ = mem.classify_oom(allocated_bytes=2 * GIB, quota_bytes=12 * GIB,
                              cuda_error="CUDA_ERROR_OUT_OF_MEMORY",
                              device_free_bytes=10 * GIB)
    assert cls == "unexpected_oom"


def test_no_oom_without_error():
    cls, _ = mem.classify_oom(allocated_bytes=11 * GIB, quota_bytes=12 * GIB,
                              cuda_error=None, device_free_bytes=10 * GIB)
    assert cls == "no_oom"


def test_non_oom_cuda_error_is_not_quota_oom():
    cls, _ = mem.classify_oom(allocated_bytes=11 * GIB, quota_bytes=12 * GIB,
                              cuda_error="CUDA_ERROR_ILLEGAL_ADDRESS",
                              device_free_bytes=10 * GIB)
    assert cls == "other_error"


def test_no_quota_configured_is_never_expected_quota_oom():
    cls, reason = mem.classify_oom(allocated_bytes=23 * GIB, quota_bytes=None,
                                   cuda_error="CUDA_ERROR_OUT_OF_MEMORY",
                                   device_free_bytes=0)
    assert cls != "expected_quota_oom"


def test_parse_probe_output_takes_last_json_line():
    text = ('{"stage":"ready"}\nsome noise\n'
            '{"ok":true,"allocated_mib":100,"cuda_error":"CUDA_ERROR_OUT_OF_MEMORY"}\n')
    parsed = mem.parse_probe_output(text)
    assert parsed["allocated_mib"] == 100


def test_parse_probe_output_invalid():
    assert mem.parse_probe_output("not json at all") is None


def test_probe_argv_is_argv_list():
    argv = mem.probe_argv(step_mib=256, limit_mib=None, touch=True)
    assert isinstance(argv, list)
    assert "--step-mib" in argv and "256" in argv and "--touch" in argv
    assert "--json" in argv
    assert all(";" not in a and "|" not in a for a in argv)


def test_verify_expectations_pass():
    outcomes = {"a": {"classification": "expected_quota_oom", "peer_correctness": True},
                "b": {"classification": "expected_quota_oom", "peer_correctness": True}}
    verdict = mem.verify_expectations({"client_a_quota_oom": True,
                                       "client_b_quota_oom": True,
                                       "peer_correctness": True}, outcomes)
    assert verdict["verdict"] == "PASS"


def test_verify_expectations_fail_on_unexpected_oom():
    outcomes = {"a": {"classification": "unexpected_oom", "peer_correctness": True}}
    verdict = mem.verify_expectations({"client_a_quota_oom": True}, outcomes)
    assert verdict["verdict"] == "FAIL"
    assert "unexpected_oom" in str(verdict["checks"])


def test_verify_expectations_inconclusive_when_missing_data():
    verdict = mem.verify_expectations({"client_a_quota_oom": True}, {})
    assert verdict["verdict"] == "INCONCLUSIVE"


def test_quota_report_row_records_raw_bytes_and_env():
    row = mem.quota_report_row(client="b", quota_bytes=12288 * 1024 * 1024,
                               env_value="0=12288M", total_bytes=24576 * 1024 * 1024,
                               outcome={"classification": "expected_quota_oom",
                                        "allocated_bytes": 11 * GIB})
    assert row["quota_bytes"] == 12288 * 1024 * 1024
    assert row["quota_env"] == "0=12288M"
    assert row["quota_fraction_of_total"] == pytest.approx(0.5)


# --------------------------- faults --------------------------- #

def test_fault_catalog_covers_f1_to_f9():
    assert set(faultmod.FAULTS) == {f"F{i}" for i in range(1, 10)}
    assert faultmod.FAULTS["F3"]["disruptive"] is False
    assert faultmod.FAULTS["F6"]["disruptive"] is True


def test_injection_and_peer_outcome_are_separate_dimensions():
    recs = [faultmod.InjectionRecord(fault_id="F1", attempt=0,
                                     injection_status="injected",
                                     peer_outcome="unaffected"),
            faultmod.InjectionRecord(fault_id="F1", attempt=1,
                                     injection_status="not_injected",
                                     peer_outcome="unknown")]
    summary = faultmod.summarize("F1", recs)
    # a failed injection must not be counted as "isolation held"
    assert summary["injected"] == 1
    assert summary["not_injected"] == 1
    assert summary["propagated"] == 0
    assert "1 次" in summary["propagation_statement"]


def test_propagation_statement_never_claims_total_isolation():
    recs = [faultmod.InjectionRecord("F7", i, "injected", "unaffected") for i in range(3)]
    summary = faultmod.summarize("F7", recs)
    assert "0/3" in summary["propagation_statement"]
    assert "不等于完全隔离" in summary["propagation_statement"]


def test_propagation_counted():
    recs = [faultmod.InjectionRecord("F7", 0, "injected", "failed"),
            faultmod.InjectionRecord("F7", 1, "injected", "unaffected")]
    summary = faultmod.summarize("F7", recs)
    assert summary["propagated"] == 1
    assert "1/2" in summary["propagation_statement"]


def test_safe_exit_fence_drain_exit():
    state = {"signalled": False, "phase": "running"}

    def send_signal(sig):
        state["signalled"] = True
        state["phase"] = "draining"

    calls = {"n": 0}

    def read_status():
        calls["n"] += 1
        if calls["n"] >= 2:
            return {"state": "exited", "inflight": 0, "completed": 10, "failed": 0}
        return {"state": "draining", "inflight": 3}

    res = faultmod.safe_exit(send_signal, read_status, drain_timeout_s=5.0,
                             terminate_client=None, poll_interval_s=0.01)
    assert res.exited_cleanly
    assert res.escalated_to_terminate_client is False
    assert res.inflight_at_exit == 0


def test_safe_exit_escalates_then_watchdogs():
    terminated = {"called": False}

    def send_signal(sig):
        pass

    def read_status():
        return {"state": "draining", "inflight": 5}

    def terminate_client():
        terminated["called"] = True
        return {"ok": True, "cuda_status": "CUDA_SUCCESS"}

    res = faultmod.safe_exit(send_signal, read_status, drain_timeout_s=0.05,
                             terminate_client=terminate_client,
                             poll_interval_s=0.01, watchdog_s=0.3)
    assert not res.exited_cleanly
    assert res.escalated_to_terminate_client and terminated["called"]
    assert res.inflight_at_exit == 5


def test_recover_r1_when_mps_and_witness_healthy():
    res = faultmod.recover(["R1", "R2"], rebuild_client=lambda: True,
                           mps_healthy=lambda: True, witness_healthy=lambda: True,
                           restart_mps=lambda: True, diagnostics=lambda: {})
    assert res["recovered"] and res["level"] == "R1"


def test_recover_escalates_to_r2():
    calls = {"restart": 0}

    def restart_mps():
        calls["restart"] += 1
        return True

    healthy = {"v": False}

    res = faultmod.recover(["R1", "R2"],
                           rebuild_client=lambda: healthy["v"],
                           mps_healthy=lambda: healthy["v"],
                           witness_healthy=lambda: True,
                           restart_mps=lambda: (healthy.__setitem__("v", True),
                                                restart_mps())[1],
                           diagnostics=lambda: {})
    assert res["recovered"] and res["level"] == "R2"
    assert calls["restart"] == 1


def test_recover_r3_never_resets_gpu():
    res = faultmod.recover(["R1", "R2", "R3"], rebuild_client=lambda: False,
                           mps_healthy=lambda: False, witness_healthy=lambda: False,
                           restart_mps=lambda: False,
                           diagnostics=lambda: {"xid": []})
    assert not res["recovered"] and res["level"] == "R3"
    note = str(res["steps"])
    assert "不自动 GPU reset" in note
