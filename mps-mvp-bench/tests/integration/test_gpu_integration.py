"""GPU integration tests. Skipped entirely unless MPS_BENCH_GPU_TESTS=1.

These are the checks that genuinely cannot run without hardware. They are
ordered from cheapest/least invasive to most invasive, and none of them touch a
GPU other than MPS_BENCH_GPU_UUID.

Required environment:
    MPS_BENCH_GPU_TESTS=1
    MPS_BENCH_GPU_UUID=GPU-xxxxxxxx-....      target card (must be idle)
    MPS_BENCH_IMAGE=mps-mvp-bench:local       built container image
    MPS_BENCH_DOCKER_RUNTIME_ARGS='--gpus device=GPU-xxxx...'   (as the host needs)

Every assertion rests on evidence read back from the machine; nothing is assumed
from the fact that a command exited 0.
"""

from __future__ import annotations

import json
import os
import shlex
import uuid as uuidlib

import pytest

from mps_bench.control import gpu as gpumod
from mps_bench.control import safety
from mps_bench.control.docker import ContainerSpec, DockerClient, make_labels
from mps_bench.control.exec import run, which
from mps_bench.control.mps import MpsController, client_env, memory_limit_bytes
from mps_bench.control.pidmap import resolve_worker_host_pid

pytestmark = pytest.mark.gpu

UUID = os.environ.get("MPS_BENCH_GPU_UUID", "")
IMAGE = os.environ.get("MPS_BENCH_IMAGE", "mps-mvp-bench:local")
RUNTIME_ARGS = shlex.split(os.environ.get("MPS_BENCH_DOCKER_RUNTIME_ARGS", ""))

DEVICE_PROBE = "/opt/mps-bench/bin/device_probe"
MEMORY_PROBE = "/opt/mps-bench/bin/memory_probe"


def _require_env() -> None:
    if not UUID:
        pytest.skip("未提供 MPS_BENCH_GPU_UUID，拒绝猜测目标卡")
    if not RUNTIME_ARGS:
        pytest.skip("未提供 MPS_BENCH_DOCKER_RUNTIME_ARGS，"
                    "拒绝使用 --gpus all / --runtime=nvidia 之类的默认值")


def _spec(name: str, command: list, env: dict | None = None,
          slot: str = "a") -> ContainerSpec:
    return ContainerSpec(
        name=name, image=IMAGE, slot=slot, role="probe", command=command,
        env=dict(env or {}), runtime_args=list(RUNTIME_ARGS),
        labels=make_labels("mps-mvp-bench", "itest", slot, "probe", UUID),
        network="none")


def _oneshot(docker: DockerClient, name: str, command: list,
             env: dict | None = None, timeout_s: float = 180):
    """Run a container to completion and return (result, stdout)."""
    from mps_bench.control.docker import build_run_argv
    spec = _spec(name, command, env)
    argv = build_run_argv(spec, docker_binary=docker.binary, detach=False)
    try:
        return run(argv, timeout_s=timeout_s, note=f"itest {name}")
    finally:
        docker.remove(name)


def _last_json(stdout: str) -> dict:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"probe 未输出 JSON，无法取证：{stdout[-500:]!r}")


# --------------------------------------------------------------------------- #
# 1. host-side, read-only
# --------------------------------------------------------------------------- #

def test_target_gpu_exists_and_is_addressable_by_uuid():
    _require_env()
    gpus = gpumod.GpuQuery().list_gpus()
    assert gpus, "nvidia-smi 未返回任何 GPU"
    match = [g for g in gpus if g.uuid == UUID]
    assert match, f"未找到 UUID={UUID}；可见: {[g.uuid for g in gpus]}"
    assert match[0].memory_total_mib and match[0].memory_total_mib > 0


def test_target_gpu_has_no_foreign_processes():
    """The whole experiment is invalid if someone else is on the card.

    This is the same check the runner performs before every round.
    """
    _require_env()
    procs = gpumod.GpuQuery().compute_processes(UUID)
    check = safety.check_gpu_processes(procs, owned_pids=(), gpu_uuid=UUID)
    assert check.ok, (
        f"目标卡上存在非本实验进程，拒绝继续：{check.foreign}。"
        "请改用一张空闲卡（切勿使用已被其他长期服务占用的那张）")


def test_compute_mode_allows_multi_process_baseline():
    """A non-MPS two-process baseline needs Default compute mode.

    If this fails, B2 numbers would be meaningless and MPS must not be credited.
    """
    _require_env()
    gpu = gpumod.GpuQuery().get(UUID)
    assert gpu is not None
    ok, reason = safety.check_compute_mode(gpu, mps_mode=False)
    assert ok, reason


def test_dcgm_sm_active_is_actually_available():
    """SM active is a mandatory acceptance metric; prove it before measuring."""
    from mps_bench.telemetry import dcgm

    if not which("dcgmi"):
        pytest.skip("未安装 dcgmi")
    res = run(["dcgmi", "dmon", "-e", "1002,1003,1005", "-c", "1"], timeout_s=20)
    assert res.ok, f"dcgmi dmon 失败: {res.stderr}"
    samples = dcgm.parse_dmon(res.stdout, [1002, 1003, 1005])
    ok, reason = dcgm.required_fields_available(samples, require_sm_active=True)
    assert ok, f"DCGM_FI_PROF_SM_ACTIVE 不可用：{reason}（不得用 GPU util 代替，也不得填 0）"


# --------------------------------------------------------------------------- #
# 2. container + device visibility
# --------------------------------------------------------------------------- #

def test_container_sees_exactly_the_target_device():
    """Container device isolation must be proven by UUID, not by --gpus working."""
    _require_env()
    docker = DockerClient()
    if not docker.available():
        pytest.skip("docker 不可用")
    res = _oneshot(docker, f"mps-itest-dev-{uuidlib.uuid4().hex[:8]}",
                   [DEVICE_PROBE, "--json"])
    assert res.ok, res.stderr
    payload = _last_json(res.stdout)
    uuids = [d.get("uuid") for d in payload.get("devices", [])]
    assert uuids == [UUID], (
        f"容器可见设备 {uuids} 与目标 {UUID} 不一致：设备传递参数不正确，"
        "此时任何隔离结论都不成立")


def test_container_records_its_own_toolkit_versions():
    """nvidia-smi's CUDA field says nothing about the image; read the image's own
    recorded versions instead of inferring them."""
    _require_env()
    docker = DockerClient()
    if not docker.available():
        pytest.skip("docker 不可用")
    res = _oneshot(docker, f"mps-itest-ver-{uuidlib.uuid4().hex[:8]}",
                   ["cat", "/opt/mps-bench/build-versions.json"])
    assert res.ok, res.stderr
    versions = json.loads(res.stdout)
    for key in ("cuda_toolkit", "torch", "torchvision"):
        assert versions.get(key), f"build-versions.json 缺少 {key}"


# --------------------------------------------------------------------------- #
# 3. MPS attachment -- the single most-faked claim in this whole area
# --------------------------------------------------------------------------- #

def test_mps_client_attachment_is_provable(tmp_path):
    """Env vars being set is NOT proof. We require the worker's host PID to show
    up in `get_client_list` for a server WE started."""
    _require_env()
    docker = DockerClient()
    if not docker.available():
        pytest.skip("docker 不可用")
    pipe_dir, log_dir = str(tmp_path / "pipe"), str(tmp_path / "log")
    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    mps = MpsController(pipe_dir=pipe_dir, log_dir=log_dir, gpu_uuid=UUID)
    if mps.binary_path() is None:
        pytest.skip("未安装 nvidia-cuda-mps-control")
    if mps.daemon_present():
        pytest.skip("该 pipe 目录已存在 MPS 实例：本项目不接管他人的 MPS")

    started = mps.start_daemon()
    assert started.get("ok"), started
    name = f"mps-itest-attach-{uuidlib.uuid4().hex[:8]}"
    try:
        env = client_env(pipe_dir=pipe_dir, log_dir=log_dir)
        spec = _spec(name, [DEVICE_PROBE, "--json", "--hold-s", "25"], env)
        spec.mounts = [{"source": pipe_dir, "target": pipe_dir},
                       {"source": log_dir, "target": log_dir}]
        assert docker.start(spec).ok
        host_pids = docker.top_pids(name)
        assert host_pids, "docker top 未返回任何 host PID"
        server_pids = mps.server_pids()
        assert server_pids, "MPS server 未出现在 get_server_list 中"
        client_pids = mps.client_pids(server_pids[0])
        resolution = resolve_worker_host_pid(host_pids, worker_container_pid=None,
                                             mps_client_pids=client_pids)
        assert resolution.host_pid is not None, (
            f"无法把容器内 worker 映射到 host PID（候选 {host_pids}，"
            f"MPS client list {client_pids}）；"
            "仅凭环境变量已设置不能断定已接入 MPS")
        assert resolution.host_pid in client_pids, (
            f"host PID {resolution.host_pid} 未出现在 MPS client list {client_pids}")
    finally:
        docker.stop(name)
        docker.remove(name)
        mps.stop_daemon()


def test_static_partitioning_capability_is_probed_not_assumed(tmp_path):
    """Either the control interface offers it, or HH-S is SKIP.

    A pass here is not required; a *false claim* is what must never happen.
    """
    _require_env()
    mps = MpsController(pipe_dir=str(tmp_path / "pipe"), log_dir=str(tmp_path / "log"),
                        gpu_uuid=UUID)
    if mps.binary_path() is None:
        pytest.skip("未安装 nvidia-cuda-mps-control")
    supported = mps.static_partitioning_supported()
    assert isinstance(supported, bool)
    if not supported:
        pytest.skip("本驱动的 MPS 控制接口不提供静态 SM 分区；HH-S 应记为 SKIP，"
                    "不得用 ACTIVE_THREAD_PERCENTAGE / 亲和性 / MIG 冒充")


# --------------------------------------------------------------------------- #
# 4. memory quota -- driver-API probe, not the framework allocator
# --------------------------------------------------------------------------- #

def test_pinned_device_mem_limit_actually_caps_allocation(tmp_path):
    """Allocate past the quota with the driver API and require an OOM that is
    provably a *quota* OOM (device still has physical headroom)."""
    _require_env()
    from mps_bench import memory_probe_runner as mpr

    docker = DockerClient()
    if not docker.available():
        pytest.skip("docker 不可用")
    pipe_dir, log_dir = str(tmp_path / "pipe"), str(tmp_path / "log")
    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    mps = MpsController(pipe_dir=pipe_dir, log_dir=log_dir, gpu_uuid=UUID)
    if mps.binary_path() is None:
        pytest.skip("未安装 nvidia-cuda-mps-control")
    if mps.daemon_present():
        pytest.skip("该 pipe 目录已存在 MPS 实例：本项目不接管他人的 MPS")

    gpu = gpumod.GpuQuery().get(UUID)
    assert gpu is not None and gpu.memory_total_bytes
    limit_bytes = memory_limit_bytes(gpu.memory_total_bytes, 0.5)
    limit_mib = limit_bytes // (1024 * 1024)

    assert mps.start_daemon().get("ok")
    name = f"mps-itest-mem-{uuidlib.uuid4().hex[:8]}"
    try:
        env = client_env(pipe_dir=pipe_dir, log_dir=log_dir,
                         pinned_device_mem_limit_bytes=limit_bytes)
        # Ask for well past the quota so the boundary must be hit.
        argv = mpr.probe_argv(step_mib=512, limit_mib=int(limit_mib * 2), touch=True,
                              binary=MEMORY_PROBE)
        spec = _spec(name, argv, env)
        spec.mounts = [{"source": pipe_dir, "target": pipe_dir},
                       {"source": log_dir, "target": log_dir}]
        from mps_bench.control.docker import build_run_argv
        res = run(build_run_argv(spec, docker_binary=docker.binary, detach=False),
                  timeout_s=300, note="itest memory probe")
        parsed = mpr.parse_probe_output(res.stdout)
        assert parsed is not None, (
            f"probe 输出无法解析，缺少证据（不记为 0 结果）：{res.stdout[-500:]!r}")
        after = gpumod.GpuQuery().get(UUID)
        outcome = mpr.build_outcome("a", limit_bytes,
                                   env.get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"),
                                   parsed,
                                   device_free_mib=(after.memory_free_mib if after else None))
        assert outcome.oom_class == mpr.EXPECTED_QUOTA_OOM, (
            f"未能证明配额生效：{outcome.oom_class} / {outcome.oom_reason}")
        assert outcome.released, "分配失败后未能释放，配额机制的可用性未通过"
        assert outcome.recovered_compute, "释放后无法继续计算，配额机制的可用性未通过"
    finally:
        docker.remove(name)
        mps.stop_daemon()
