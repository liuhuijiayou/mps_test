"""Safety gates: GPU ownership, locking, disruptive gating, cleanup idempotence."""

import os

import pytest

from mps_bench.control import safety
from mps_bench.control.cleanup import cleanup
from mps_bench.control.gpu import GpuProcess, parse_gpu_query


class _FakeDocker:
    def __init__(self, existing):
        self.existing = list(existing)
        self.calls = []
        self.project = "proj"

    def assert_owned(self, name):
        if name not in self.existing:
            raise KeyError(name)

    def logs(self, name):
        self.calls.append(("logs", name))
        return "log-of-" + name

    def stop(self, name, timeout_s=10):
        self.calls.append(("stop", name))
        if name not in self.existing:
            return {"ok": False, "reason": "not found"}
        return {"ok": True}

    def remove(self, name):
        self.calls.append(("remove", name))
        if name in self.existing:
            self.existing.remove(name)
            return {"ok": True}
        return {"ok": True, "already_absent": True}


def test_ownership_check_flags_foreign_process():
    procs = [GpuProcess(pid=999, name="VLLM::EngineCore", memory_mib=22669,
                        gpu_uuid="GPU-a")]
    check = safety.check_gpu_processes(procs, owned_pids=[], gpu_uuid="GPU-a")
    assert not check.ok
    assert check.foreign and check.foreign[0]["pid"] == 999
    assert "拒绝" in check.reason or "外部" in check.reason


def test_ownership_check_accepts_own_processes():
    procs = [GpuProcess(pid=100, name="python", memory_mib=1024, gpu_uuid="GPU-a")]
    check = safety.check_gpu_processes(procs, owned_pids=[100], gpu_uuid="GPU-a")
    assert check.ok and not check.foreign


def test_ownership_ignores_other_gpus():
    procs = [GpuProcess(pid=999, name="VLLM::EngineCore", memory_mib=22669,
                        gpu_uuid="GPU-b")]
    check = safety.check_gpu_processes(procs, owned_pids=[], gpu_uuid="GPU-a")
    assert check.ok


def test_gpu_lock_is_exclusive_and_idempotent(tmp_path):
    a = safety.GpuLock(str(tmp_path), "GPU-a", "run1")
    a.acquire()
    b = safety.GpuLock(str(tmp_path), "GPU-a", "run2")
    with pytest.raises(safety.SafetyError):
        b.acquire()
    # another run must never delete our lock
    b.release()
    assert os.path.exists(a.path)
    a.release()
    assert not os.path.exists(a.path)
    a.release()  # idempotent


def test_different_gpus_lock_independently(tmp_path):
    a = safety.GpuLock(str(tmp_path), "GPU-a", "run1")
    b = safety.GpuLock(str(tmp_path), "GPU-b", "run2")
    a.acquire()
    b.acquire()
    a.release()
    b.release()


def test_disruptive_requires_flag_config_and_uuid():
    with pytest.raises(safety.SafetyError):
        safety.gate_disruptive("F6", cli_flag=False, config_flag=True,
                               confirm_uuid="GPU-a", gpu_uuid="GPU-a")
    with pytest.raises(safety.SafetyError):
        safety.gate_disruptive("F6", cli_flag=True, config_flag=False,
                               confirm_uuid="GPU-a", gpu_uuid="GPU-a")
    with pytest.raises(safety.SafetyError):
        safety.gate_disruptive("F6", cli_flag=True, config_flag=True,
                               confirm_uuid="GPU-b", gpu_uuid="GPU-a")
    safety.gate_disruptive("F6", cli_flag=True, config_flag=True,
                           confirm_uuid="GPU-a", gpu_uuid="GPU-a")


def test_non_disruptive_fault_needs_no_gate():
    safety.gate_disruptive("F1", cli_flag=False, config_flag=False,
                           confirm_uuid=None, gpu_uuid="GPU-a")
    safety.gate_disruptive(None, False, False, None, "GPU-a")


def test_compute_mode_check():
    class G:
        compute_mode = "Exclusive_Process"
    ok, reason = safety.check_compute_mode(G(), mps_mode=False,
                                           require_default_for_nonmps=True)
    assert not ok and "Default" in reason

    class G2:
        compute_mode = "Default"
    ok2, _ = safety.check_compute_mode(G2(), mps_mode=False,
                                       require_default_for_nonmps=True)
    assert ok2


def test_cleanup_is_idempotent():
    docker = _FakeDocker(["c1", "c2"])
    saved = {}
    r1 = cleanup(docker, ["c1", "c2"], None,
                 save_logs=lambda name, text: saved.__setitem__(name, text))
    assert set(r1.removed) == {"c1", "c2"}
    assert saved  # logs saved BEFORE removal
    r2 = cleanup(docker, ["c1", "c2"], None)
    assert r2.errors == [] or all("not found" not in e for e in r2.errors)
    # second run must not raise and must not report false removals
    assert docker.existing == []


def test_cleanup_does_not_stop_unowned_mps():
    class FakeMps:
        owns_daemon = False
        stopped = False

        def stop_daemon(self):
            self.stopped = True
            return {"ok": False, "reason": "not owned"}

        def destroy_partitions(self):
            return {"supported": False}

    mps = FakeMps()
    report = cleanup(_FakeDocker([]), [], mps)
    assert mps.stopped is False
    assert any("未由本 run 启动" in n or "not owned" in n for n in report.notes)


def test_parse_gpu_query_handles_na():
    # column order must match control.gpu._QUERY_FIELDS
    line = ("0, GPU-abc, NVIDIA A30, 24576, 100, 24476, Default, Disabled, "
            "580.105.08, [N/A]")
    gpu = parse_gpu_query(line)
    assert gpu.uuid == "GPU-abc"
    assert gpu.memory_total_mib == 24576
    assert gpu.compute_mode == "Default"
    assert gpu.mig_mode == "Disabled"
    assert gpu.memory_total_bytes == 24576 * 1024 * 1024
