"""MPS env construction, control-output parsing and PID mapping."""

import pytest

from mps_bench.control import mps as mpsmod
from mps_bench.control import pidmap


# ------------------------- client env ------------------------- #

def test_client_env_priority_mapping():
    env = mpsmod.client_env("/tmp/pipe", priority="NORMAL")
    assert env["CUDA_MPS_CLIENT_PRIORITY"] == "0"
    env = mpsmod.client_env("/tmp/pipe", priority="BELOW_NORMAL")
    assert env["CUDA_MPS_CLIENT_PRIORITY"] == "1"


def test_client_env_rejects_unknown_priority():
    with pytest.raises(mpsmod.MpsError):
        mpsmod.client_env("/tmp/pipe", priority="HIGH")


def test_client_env_atp_range():
    assert mpsmod.client_env("/p", active_thread_percentage=70)[
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "70"
    with pytest.raises(mpsmod.MpsError):
        mpsmod.client_env("/p", active_thread_percentage=0)
    with pytest.raises(mpsmod.MpsError):
        mpsmod.client_env("/p", active_thread_percentage=101)


def test_client_env_mem_limit_units():
    total = 24576 * 1024 * 1024
    limit = mpsmod.memory_limit_bytes(total, 0.5)
    assert limit == total // 2
    env = mpsmod.client_env("/p", pinned_device_mem_limit_bytes=limit)
    assert env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=12288M"


def test_memory_limit_validation():
    with pytest.raises(mpsmod.MpsError):
        mpsmod.memory_limit_bytes(0, 0.5)
    with pytest.raises(mpsmod.MpsError):
        mpsmod.memory_limit_bytes(1024, 1.5)


def test_client_env_pipe_always_set():
    env = mpsmod.client_env("/tmp/mine/pipe", log_dir="/tmp/mine/log")
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mine/pipe"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/mine/log"
    # No MPS knobs unless explicitly requested
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in env
    assert "CUDA_MPS_CLIENT_PRIORITY" not in env


# ------------------------- output parsing ------------------------- #

def test_parse_server_and_client_lists():
    assert mpsmod.parse_server_list("12345\n") == [12345]
    assert mpsmod.parse_server_list("No server\n") == []
    assert mpsmod.parse_client_list("54321\n54322\n") == [54321, 54322]
    assert mpsmod.parse_client_list("") == []


def test_parse_terminate_requires_cuda_success():
    ok = mpsmod.parse_terminate_output("CUDA_SUCCESS\n", "", 0)
    assert ok.ok and ok.cuda_status == "CUDA_SUCCESS"

    # shell rc 0 but a CUDA error must NOT be treated as success
    bad = mpsmod.parse_terminate_output("CUDA_ERROR_NOT_FOUND\n", "", 0)
    assert not bad.ok and bad.cuda_status == "CUDA_ERROR_NOT_FOUND"

    # non-zero rc
    rc = mpsmod.parse_terminate_output("", "boom", 1)
    assert not rc.ok

    # timeout is never a success
    to = mpsmod.parse_terminate_output("CUDA_SUCCESS", "", 0, timed_out=True)
    assert not to.ok and to.timed_out


def test_parse_terminate_ambiguous_output_is_not_success_when_error_present():
    res = mpsmod.parse_terminate_output("client terminated with error\n", "", 0)
    assert not res.ok


def test_parse_server_state():
    assert mpsmod.parse_server_state("server is ACTIVE") == "ACTIVE"
    assert mpsmod.parse_server_state("state: FAULT") == "FAULT"
    assert mpsmod.parse_server_state("nothing here") is None


def test_parse_help_detects_commands_and_static_partitioning():
    legacy = ("get_server_list\nget_client_list\nterminate_client <server pid> <client pid>\n"
              "set_default_active_thread_percentage\nquit\n")
    caps = mpsmod.parse_help(legacy)
    assert caps.has("terminate_client")
    assert not caps.static_partitioning

    modern = legacy + "create_device_partition <dev> <sm>\nlist_device_partition\n"
    caps2 = mpsmod.parse_help(modern)
    assert caps2.static_partitioning


def test_help_absence_means_unsupported_not_emulated():
    caps = mpsmod.parse_help("get_server_list\nquit\n")
    assert caps.static_partitioning is False
    assert not caps.has("terminate_client")


# ------------------------- PID mapping ------------------------- #

def test_parse_nspid():
    assert pidmap.parse_nspid("Name:\tpython\nNSpid:\t40321\t7\n") == [40321, 7]
    assert pidmap.parse_nspid("Name:\tpython\n") == []


def test_resolve_by_nspid(tmp_path):
    proc = tmp_path
    for host_pid, inner in ((1000, 1), (1001, 7)):
        d = proc / str(host_pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\tx\nNSpid:\t{host_pid}\t{inner}\n", encoding="utf-8")
    res = pidmap.resolve_worker_host_pid([1000, 1001], worker_container_pid=7,
                                         mps_client_pids=[1001], proc_root=str(proc))
    assert res.host_pid == 1001
    assert res.method == "nspid"
    assert res.confirmed_by_mps_client_list is True
    # crucially, docker PID 1 (host 1000) was NOT chosen
    assert res.host_pid != 1000


def test_resolve_falls_back_to_mps_client_list(tmp_path):
    res = pidmap.resolve_worker_host_pid([2000, 2001], worker_container_pid=9,
                                         mps_client_pids=[2001], proc_root=str(tmp_path))
    assert res.host_pid == 2001 and res.method == "mps_client_list"


def test_resolve_unresolved_when_ambiguous(tmp_path):
    res = pidmap.resolve_worker_host_pid([3000, 3001], worker_container_pid=None,
                                         mps_client_pids=[3000, 3001], proc_root=str(tmp_path))
    assert res.host_pid is None and res.method == "unresolved"
