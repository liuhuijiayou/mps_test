"""Docker argv construction and ownership-scoped container lifecycle.

Design constraints:
* GPU runtime / device pass-through args come verbatim from
  `docker.runtime_args` (user supplied). We never inject `--runtime=nvidia`,
  never default to `--gpus all`, and never pick a "looks idle" GPU.
* Every container carries project/run labels; destructive operations are only
  ever applied to containers matching those labels. No `docker rm -f $(docker ps)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .exec import CommandLog, CommandResult, run

LABEL_PROJECT = "mps_bench.project"
LABEL_RUN = "mps_bench.run_id"
LABEL_ROLE = "mps_bench.role"
LABEL_SLOT = "mps_bench.slot"
LABEL_GPU = "mps_bench.gpu_uuid"


class DockerArgError(ValueError):
    pass


_FORBIDDEN_ARGS = ("--privileged", "-v /var/run/docker.sock", "/var/run/docker.sock")


def validate_runtime_args(runtime_args: Sequence[str]) -> List[str]:
    """User-supplied GPU runtime args must be an explicit argv list.

    We reject shell metacharacters (they would only matter if someone later
    string-joined them) and reject implicit whole-GPU requests, because an
    experiment must target exactly the UUID the user named.
    """
    args = [str(a) for a in runtime_args]
    for a in args:
        if any(ch in a for ch in ";|&`$\n"):
            raise DockerArgError(f"docker.runtime_args 含 shell 元字符，拒绝执行: {a!r}")
        if a.replace(" ", "") in ("--gpus=all", "--gpusall"):
            raise DockerArgError("禁止 --gpus all：必须显式指定目标 GPU 设备")
        if "/var/run/docker.sock" in a:
            raise DockerArgError("禁止把 docker socket 挂入业务容器")
    if "all" in args:
        idx = args.index("all")
        if idx > 0 and args[idx - 1] == "--gpus":
            raise DockerArgError("禁止 --gpus all：必须显式指定目标 GPU 设备")
    return args


def requires_explicit_device(runtime_args: Sequence[str], gpu_uuid: Optional[str]) -> None:
    """Any real GPU execution must name the device explicitly."""
    if not gpu_uuid:
        raise DockerArgError("gpu.uuid 未指定：真实 GPU 执行前必须由用户明确指定目标 GPU UUID")
    if not runtime_args:
        raise DockerArgError("docker.runtime_args 为空：需由用户提供 GPU runtime / 设备透传参数")
    joined = " ".join(runtime_args)
    if gpu_uuid not in joined:
        raise DockerArgError(
            "docker.runtime_args 未包含 gpu.uuid；不允许依赖 GPU 序号或默认设备选择。"
            f" 期望参数中出现 {gpu_uuid}")


@dataclass
class ContainerSpec:
    name: str
    image: str
    slot: str
    role: str
    command: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    mounts: List[Dict[str, str]] = field(default_factory=list)  # {source,target,ro}
    ports: List[Dict[str, Any]] = field(default_factory=list)   # {host,container,bind}
    cpuset: Optional[str] = None
    memory: Optional[str] = None
    network: str = "none"
    user: Optional[str] = None
    ipc: Optional[str] = None
    shm_size: str = "1g"
    runtime_args: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    pid_mode: Optional[str] = None


def build_run_argv(spec: ContainerSpec, docker_binary: str = "docker",
                   detach: bool = True) -> List[str]:
    argv: List[str] = [docker_binary, "run"]
    argv += ["--detach"] if detach else ["--rm"]
    if detach:
        # keep the container around so logs/exit codes survive fault injection
        argv += ["--name", spec.name]
    else:
        argv += ["--name", spec.name]
    argv += ["--init"]  # reap zombies; entrypoint still forwards signals itself
    argv += ["--network", spec.network]
    argv += ["--shm-size", spec.shm_size]
    if spec.user:
        argv += ["--user", spec.user]
    if spec.ipc:
        argv += ["--ipc", spec.ipc]
    if spec.pid_mode:
        argv += ["--pid", spec.pid_mode]
    if spec.cpuset:
        argv += ["--cpuset-cpus", spec.cpuset]
    if spec.memory:
        argv += ["--memory", spec.memory]
    for key in sorted(spec.env):
        argv += ["--env", f"{key}={spec.env[key]}"]
    for mount in spec.mounts:
        ro = ",readonly" if mount.get("ro") else ""
        argv += ["--mount", f"type=bind,source={mount['source']},target={mount['target']}{ro}"]
    for port in spec.ports:
        bind = port.get("bind", "127.0.0.1")
        argv += ["--publish", f"{bind}:{port['host']}:{port['container']}"]
    for key in sorted(spec.labels):
        argv += ["--label", f"{key}={spec.labels[key]}"]
    argv += validate_runtime_args(spec.runtime_args)
    argv += [str(a) for a in spec.extra_args]
    argv += [spec.image]
    argv += [str(a) for a in spec.command]
    return argv


def make_labels(project: str, run_id: str, slot: str, role: str,
                gpu_uuid: Optional[str]) -> Dict[str, str]:
    labels = {LABEL_PROJECT: project, LABEL_RUN: run_id, LABEL_SLOT: slot, LABEL_ROLE: role}
    if gpu_uuid:
        labels[LABEL_GPU] = gpu_uuid
    return labels


class DockerClient:
    """Thin wrapper; only ever touches containers labeled by this project."""

    def __init__(self, binary: str = "docker", project: str = "mps-mvp-bench",
                 log: Optional[CommandLog] = None):
        self.binary = binary
        self.project = project
        self.log = log

    def _run(self, argv: Sequence[str], timeout_s: float = 120, note: str = "") -> CommandResult:
        return run(argv, timeout_s=timeout_s, log=self.log, note=note)

    def available(self) -> bool:
        return self._run([self.binary, "version", "--format", "{{.Server.Version}}"],
                         timeout_s=20, note="docker version").ok

    def start(self, spec: ContainerSpec) -> CommandResult:
        return self._run(build_run_argv(spec, self.binary), note=f"start {spec.name}")

    def inspect(self, name: str) -> Optional[Dict[str, Any]]:
        res = self._run([self.binary, "inspect", name], note=f"inspect {name}")
        if not res.ok:
            return None
        try:
            data = json.loads(res.stdout)
        except json.JSONDecodeError:
            return None
        return data[0] if data else None

    def owned_containers(self, run_id: Optional[str] = None) -> List[str]:
        argv = [self.binary, "ps", "-a", "--filter", f"label={LABEL_PROJECT}={self.project}"]
        if run_id:
            argv += ["--filter", f"label={LABEL_RUN}={run_id}"]
        argv += ["--format", "{{.Names}}"]
        res = self._run(argv, note="list owned containers")
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    def assert_owned(self, name: str) -> None:
        info = self.inspect(name)
        labels = ((info or {}).get("Config") or {}).get("Labels") or {}
        if labels.get(LABEL_PROJECT) != self.project:
            raise DockerArgError(
                f"容器 {name} 不属于本项目（缺少 {LABEL_PROJECT}={self.project}），拒绝操作")

    def signal(self, name: str, sig: str) -> CommandResult:
        self.assert_owned(name)
        return self._run([self.binary, "kill", "--signal", sig, name], note=f"signal {sig} {name}")

    def stop(self, name: str, timeout_s: int = 30) -> CommandResult:
        """Graceful stop. Note: normal cleanup must not *rely* on docker's
        post-timeout SIGKILL; the drain protocol runs before this is called."""
        self.assert_owned(name)
        return self._run([self.binary, "stop", "--timeout", str(int(timeout_s)), name],
                         timeout_s=timeout_s + 30, note=f"stop {name}")

    def logs(self, name: str) -> str:
        self.assert_owned(name)
        return self._run([self.binary, "logs", name], note=f"logs {name}").stdout

    def remove(self, name: str) -> CommandResult:
        self.assert_owned(name)
        return self._run([self.binary, "rm", "--force", name], note=f"rm {name}")

    def top_pids(self, name: str) -> List[int]:
        """Host-namespace PIDs of processes inside the container.

        Needed because MPS `terminate_client` wants the client PID as seen by the
        MPS control process namespace -- docker PID 1 is NOT necessarily the CUDA
        worker.
        """
        self.assert_owned(name)
        res = self._run([self.binary, "top", name, "-eo", "pid,comm,args"], note=f"top {name}")
        pids: List[int] = []
        for line in res.stdout.splitlines()[1:]:
            parts = line.split(None, 1)
            if parts and parts[0].isdigit():
                pids.append(int(parts[0]))
        return pids
