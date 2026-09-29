"""Read-only preflight -> capability.json.

Rules encoded here:
* preflight never changes compute mode and never touches the user's own MPS
  instance.
* "found the binary / found the symbol" is NOT a verified capability. Anything
  that can only be proven by executing something is recorded as `unknown` with
  `verify_by: smoke`.
* every entry carries raw evidence so a human can re-check the conclusion.

The one deliberate exception
----------------------------
`mps_static_partitioning` cannot be settled by inspection. The control binary's
`help` text on R580 does not reliably advertise the feature, so parsing it
yields false negatives and silently SKIPs HH-S. We therefore *actively probe*:
a throwaway MPS daemon is started in its own pipe directory under
`paths.probe_dir`, a 1-chunk partition is requested, and the verdict comes from
the return value. The probe instance is isolated from any pre-existing MPS and
is always torn down. Set `mps.static_partitioning.probe=false` to skip it, in
which case the capability is reported as `unknown`, never as `supported`.
"""

from __future__ import annotations

import getpass
import os
import platform
import shutil
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import __version__
from .exec import CommandLog, run, which
from .gpu import GpuQuery
from .mps import MpsController, probe_static_partitioning
from .safety import check_compute_mode, check_gpu_processes

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
UNKNOWN = "unknown"


@dataclass
class Capability:
    name: str
    status: str
    detail: str = ""
    evidence: Any = None
    verify_by: Optional[str] = None  # "smoke" when only execution can prove it

    def as_dict(self) -> Dict[str, Any]:
        out = {"name": self.name, "status": self.status, "detail": self.detail,
               "evidence": self.evidence}
        if self.verify_by:
            out["verify_by"] = self.verify_by
        return out


@dataclass
class PreflightReport:
    capabilities: List[Capability] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    blocking: List[str] = field(default_factory=list)

    def add(self, cap: Capability) -> None:
        self.capabilities.append(cap)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "harness_version": __version__,
            "capabilities": [c.as_dict() for c in self.capabilities],
            "blocking": self.blocking,
            "summary": {
                "supported": [c.name for c in self.capabilities if c.status == SUPPORTED],
                "unsupported": [c.name for c in self.capabilities if c.status == UNSUPPORTED],
                "unknown": [c.name for c in self.capabilities if c.status == UNKNOWN],
            },
        }


def _host_environment() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "user": getpass.getuser(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "gid": os.getgid() if hasattr(os, "getgid") else None,
        "cpu_count": os.cpu_count(),
        "harness_version": __version__,
    }


def _dir_writable(path: str) -> bool:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return bool(probe) and os.access(probe, os.W_OK)


def run_preflight(cfg, log: Optional[CommandLog] = None) -> PreflightReport:
    report = PreflightReport()
    report.environment = _host_environment()
    log = log if log is not None else CommandLog()

    # ---------------- driver / nvidia-smi ----------------
    smi = cfg["telemetry.nvidia_smi_binary"]
    smi_path = which(smi)
    if not smi_path:
        report.add(Capability("nvidia_smi", UNSUPPORTED, f"未找到 {smi}", None))
        report.blocking.append("nvidia-smi 不可用：无法进行任何 GPU 相关验证")
        gpus = []
        gpu = None
    else:
        gq = GpuQuery(binary=smi, log=log)
        gpus = gq.list_gpus()
        report.add(Capability("nvidia_smi", SUPPORTED, smi_path,
                              {"gpu_count": len(gpus),
                               "driver_version": gpus[0].driver_version if gpus else None,
                               "raw_rows": [g.raw for g in gpus]}))
        target_uuid = cfg["gpu.uuid"]
        gpu = next((g for g in gpus if g.uuid == target_uuid), None) if target_uuid else None

        if not target_uuid:
            report.add(Capability("target_gpu", UNKNOWN,
                                  "gpu.uuid 未指定：plan / CPU 单测允许，真实 GPU 执行前必须填写",
                                  {"visible_uuids": [g.uuid for g in gpus]},
                                  verify_by="user-input"))
        elif gpu is None:
            report.add(Capability("target_gpu", UNSUPPORTED,
                                  f"未在本机找到 gpu.uuid={target_uuid}",
                                  {"visible_uuids": [g.uuid for g in gpus]}))
            report.blocking.append(f"目标 GPU {target_uuid} 不存在")
        else:
            report.add(Capability("target_gpu", SUPPORTED, gpu.name, {
                "uuid": gpu.uuid, "index": gpu.index,
                "memory_total_mib": gpu.memory_total_mib,
                "memory_used_mib": gpu.memory_used_mib,
                "memory_free_mib": gpu.memory_free_mib,
                "driver_version": gpu.driver_version,
                "note": "nvidia-smi 的 CUDA 字段是驱动能力，不能证明容器内 Toolkit/Runtime 版本",
            }))
            expect = cfg["gpu.expect_total_memory_mib"]
            if expect is not None and gpu.memory_total_mib != expect:
                report.blocking.append(
                    f"显存断言失败：期望 {expect} MiB，实际 {gpu.memory_total_mib} MiB")

            # MIG / compute mode
            mig = (gpu.mig_mode or "unknown").lower()
            report.add(Capability("mig_disabled",
                                  SUPPORTED if "disabled" in mig or mig in ("n/a", "unknown")
                                  else UNSUPPORTED,
                                  f"mig.mode.current={gpu.mig_mode}", {"raw": gpu.mig_mode}))
            if "enabled" in mig:
                report.blocking.append("目标 GPU 处于 MIG Enabled：本项目不使用 MIG 冒充 MPS 隔离")

            cm_ok, cm_reason = check_compute_mode(
                gpu,
                mps_mode=False,
                require_default_for_nonmps=cfg["compute_mode.require_default_for_nonmps"],
            )
            report.add(Capability(
                "compute_mode_allows_multiprocess",
                SUPPORTED if cm_ok else UNSUPPORTED,
                cm_reason or f"compute_mode={gpu.compute_mode}",
                {"ok": cm_ok, "compute_mode": gpu.compute_mode, "reason": cm_reason},
            ))
            if not cm_ok and not cfg["compute_mode.allow_change"]:
                report.blocking.append(cm_reason or "compute mode 不允许多进程")

            # SM count
            sm = gq.sm_count(gpu.uuid)
            report.add(Capability("sm_count", SUPPORTED if sm else UNKNOWN,
                                  "来自 nvidia-smi -q" if sm else "nvidia-smi -q 未给出 SM 数",
                                  {"sm_count": sm}, verify_by=None if sm else "device-probe"))

            # foreign processes
            procs = gq.compute_processes(gpu.uuid)
            own = check_gpu_processes(procs, owned_pids=[], gpu_uuid=gpu.uuid)
            report.add(Capability("target_gpu_idle_for_experiment",
                                  SUPPORTED if own.ok else UNSUPPORTED, own.reason,
                                  own.as_dict()))
            if not own.ok and cfg["safety.abort_on_foreign_gpu_process"]:
                report.blocking.append(own.reason)

    # ---------------- docker ----------------
    dbin = cfg["docker.binary"]
    dpath = which(dbin)
    if not dpath:
        report.add(Capability("docker", UNSUPPORTED, f"未找到 {dbin}", None))
        report.blocking.append("docker 不可用")
    else:
        ver = run([dbin, "version", "--format", "{{.Server.Version}}"], timeout_s=20, log=log,
                  note="docker version")
        report.add(Capability("docker", SUPPORTED if ver.ok else UNSUPPORTED,
                              dpath, {"server_version": ver.stdout.strip(),
                                      "stderr_tail": ver.stderr[-500:]}))
        if not ver.ok:
            report.blocking.append("docker daemon 不可达")

        img = run([dbin, "image", "inspect", cfg["docker.image"], "--format",
                   "{{index .RepoDigests 0}}|{{.Id}}"], timeout_s=30, log=log,
                  note="image inspect")
        report.add(Capability("image_present", SUPPORTED if img.ok else UNKNOWN,
                              cfg["docker.image"],
                              {"digest_or_id": img.stdout.strip(), "stderr_tail": img.stderr[-500:]},
                              verify_by=None if img.ok else "docker build"))

    # runtime args are user-supplied; we can only check they are non-empty and
    # name the target UUID. Whether the container really sees ONLY that device
    # can only be proven by executing a probe.
    runtime_args = cfg["docker.runtime_args"]
    if runtime_args:
        names_uuid = bool(cfg["gpu.uuid"]) and cfg["gpu.uuid"] in " ".join(runtime_args)
        report.add(Capability("docker_runtime_args", SUPPORTED if names_uuid else UNKNOWN,
                              "用户提供的 GPU runtime/设备参数" +
                              ("" if names_uuid else "；未包含 gpu.uuid，真实执行前会被拒绝"),
                              {"argv": runtime_args},
                              verify_by=None if names_uuid else "user-input"))
    else:
        report.add(Capability("docker_runtime_args", UNKNOWN,
                              "docker.runtime_args 为空：plan/CPU 测试允许，真实 GPU 执行前必填",
                              {"argv": []}, verify_by="user-input"))
    report.add(Capability("container_sees_only_target_device", UNKNOWN,
                          "容器实际设备透传只能由执行证明；preflight 不启动 CUDA context",
                          None, verify_by="smoke"))

    # ---------------- MPS ----------------
    ctl = MpsController(pipe_dir=cfg["paths.mps_pipe_dir"], log_dir=cfg["paths.mps_log_dir"],
                       binary=cfg["mps.control_binary"], gpu_uuid=cfg["gpu.uuid"], log=log)
    ctl_path = ctl.binary_path()
    if not ctl_path:
        report.add(Capability("mps_control_binary", UNSUPPORTED,
                              f"未找到 {cfg['mps.control_binary']}", None))
        report.blocking.append("MPS 控制工具不可用")
        report.add(Capability("mps_static_partitioning", UNKNOWN,
                              "无控制工具，无法探测静态分区能力", None, verify_by="smoke"))
    else:
        server = which("nvidia-cuda-mps-server")
        caps = ctl.capabilities()
        report.add(Capability("mps_control_binary", SUPPORTED, ctl_path, {
            "control_path": ctl_path, "server_path": server,
            "advertised_commands": caps.commands,
            "help_raw_tail": caps.raw[-4000:],
            "note": "以本机 help 为准，不把最新 MPS v3 命令硬套到 R580",
        }))
        # Actively probe: `help` is not authoritative on R580. See module docstring.
        if not cfg["mps.static_partitioning.probe"]:
            report.add(Capability("mps_static_partitioning", UNKNOWN,
                                  "mps.static_partitioning.probe=false，未做实际探测；"
                                  "不据此判定支持与否",
                                  {"help_mentions": caps.static_partitioning},
                                  verify_by="smoke"))
        elif not cfg["gpu.uuid"]:
            report.add(Capability("mps_static_partitioning", UNKNOWN,
                                  "未指定 gpu.uuid，无法对具体设备执行分区探测",
                                  {"help_mentions": caps.static_partitioning},
                                  verify_by="user-input"))
        else:
            probe = probe_static_partitioning(
                binary=cfg["mps.control_binary"], gpu_uuid=cfg["gpu.uuid"],
                probe_root=cfg["paths.probe_dir"], log=log)
            supported = bool(probe.get("supported"))
            report.add(Capability(
                "mps_static_partitioning",
                SUPPORTED if supported else UNSUPPORTED,
                probe.get("reason", "") if supported else
                (probe.get("reason", "") + "；HH-S 用例 SKIP，"
                 "禁止用 ACTIVE_THREAD_PERCENTAGE/亲和性/MIG 冒充"),
                {"mps_static_partitioning": {
                    "supported": supported,
                    "commands": probe.get("commands", []),
                    "reason": probe.get("reason", ""),
                 },
                 "probe": {k: v for k, v in probe.items()
                           if k not in ("supported", "commands", "reason")},
                 "help_mentions": caps.static_partitioning,
                 "note": "结论来自实际执行 sm_partition，而非 help 文本"}))
            if supported and probe.get("probe_partition_removed") is False:
                report.blocking.append(
                    "静态分区探测后未能释放探测分区，GPU 可能仍处于被切分状态："
                    f"{probe.get('probe_cleanup_errors')}")
        existing = ctl.daemon_present()
        report.add(Capability("mps_pipe_free",
                              UNSUPPORTED if existing and not cfg["mps.allow_adopt_existing"]
                              else SUPPORTED,
                              "该 pipe 目录已有可用控制实例；默认不接管" if existing else
                              "pipe 目录无既有实例",
                              {"pipe_dir": cfg["paths.mps_pipe_dir"], "existing_daemon": existing}))
        if existing and not cfg["mps.allow_adopt_existing"]:
            report.blocking.append("已存在 MPS 实例且未授权接管：停止并报告")
        report.add(Capability("mps_client_attached", UNKNOWN,
                              "仅设置环境变量或看到 daemon 存在都不能证明业务已接入 MPS；"
                              "需运行期核对 server/client 列表与实际 worker PID",
                              None, verify_by="smoke"))

    # ---------------- DCGM ----------------
    dcgmi = cfg["telemetry.dcgmi_binary"]
    dcgmi_path = which(dcgmi)
    if not dcgmi_path:
        report.add(Capability("dcgm_sm_active", UNSUPPORTED,
                              f"未找到 {dcgmi}：SM active(1002) 不可用",
                              {"required": cfg["telemetry.require_sm_metrics"]}))
        if cfg["telemetry.require_sm_metrics"]:
            report.blocking.append("require_sm_metrics=true 但 dcgmi 不可用："
                                   "不得用 GPU util 代替 SM active，也不得填 0")
    else:
        probe = run([dcgmi, "dmon", "-e", ",".join(str(f) for f in cfg["telemetry.dcgm_fields"]),
                     "-c", "1"], timeout_s=40, log=log, note="dcgmi dmon probe")
        ok = probe.ok and "1002" not in probe.stderr
        report.add(Capability("dcgm_sm_active", SUPPORTED if ok else UNKNOWN,
                              "dcgmi dmon 可返回 profiling field" if ok else
                              "dcgmi 存在但采样探测未成功（权限/分组/已有 profiler 冲突）",
                              {"path": dcgmi_path, "fields": cfg["telemetry.dcgm_fields"],
                               "stdout_tail": probe.stdout[-2000:],
                               "stderr_tail": probe.stderr[-2000:]},
                              verify_by=None if ok else "smoke"))

    # ---------------- directories / uid / pid namespace ----------------
    for key in ("paths.results_dir", "paths.mps_pipe_dir", "paths.mps_log_dir", "safety.lock_dir"):
        path = cfg[key]
        writable = _dir_writable(path)
        report.add(Capability(f"writable:{key}", SUPPORTED if writable else UNSUPPORTED,
                              path, {"exists": os.path.exists(path), "writable": writable}))
        if not writable:
            report.blocking.append(f"{key}={path} 不可写")

    nspid_ok = os.path.exists("/proc/self/status") and "NSpid" in _read_self_status()
    report.add(Capability("pid_namespace_mapping", SUPPORTED if nspid_ok else UNKNOWN,
                          "/proc/<pid>/status 提供 NSpid，可将容器内 PID 映射到宿主 PID"
                          if nspid_ok else
                          "本机 /proc 未提供 NSpid（非 Linux 或内核较旧）："
                          "terminate_client 的 PID 解析将退化为 MPS client 列表交叉确认",
                          {"nspid_available": nspid_ok}, verify_by=None if nspid_ok else "smoke"))

    report.add(Capability("uid_compatibility", UNKNOWN,
                          "容器 UID 与宿主 MPS pipe 所属 UID 的兼容性需执行证明",
                          {"host_uid": report.environment.get("uid"),
                           "configured_container_user": cfg["docker.user"]},
                          verify_by="smoke"))

    # native probes are inside the image; presence can only be proven by running
    report.add(Capability("native_probes_built", UNKNOWN,
                          "device_probe / memory_probe / fault_injector 是否在镜像中成功编译，"
                          "需构建镜像后由 smoke 验证",
                          None, verify_by="smoke"))
    report.add(Capability("torch_cuda_runtime", UNKNOWN,
                          "容器内 PyTorch/CUDA Runtime/Toolkit 版本需运行期采集，"
                          "不能用 nvidia-smi 的 CUDA 字段代替",
                          None, verify_by="smoke"))
    return report


def _read_self_status() -> str:
    try:
        with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""
