"""MPS control-daemon driver (legacy `nvidia-cuda-mps-control` interface).

Version discipline
------------------
The target machine runs driver R580, whose MPS control binary exposes the
*legacy* command set. We therefore:

* probe `help` at runtime and record it as evidence,
* only issue commands that the probed help actually advertises,
* treat static SM partitioning as an independent, separately-probed capability
  (`-S` / `--static-partitioning` and the partition management commands). If it
  is absent, the HH-S case is SKIPped -- we never emulate it with
  ACTIVE_THREAD_PERCENTAGE=50, context affinity or MIG.

Per-client knobs (ACTIVE_THREAD_PERCENTAGE, CLIENT_PRIORITY,
PINNED_DEVICE_MEM_LIMIT) are environment variables that must be set *before*
CUDA initialization in the client process. Changing them requires starting a new
client; they cannot retune a live context.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .exec import CommandLog, CommandResult, run, which

PRIORITY_VALUES = {"NORMAL": "0", "BELOW_NORMAL": "1"}


class MpsError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# client environment
# --------------------------------------------------------------------------- #

def client_env(pipe_dir: str,
               log_dir: Optional[str] = None,
               active_thread_percentage: Optional[int] = None,
               priority: Optional[str] = None,
               pinned_device_mem_limit_bytes: Optional[int] = None,
               gpu_uuid: Optional[str] = None,
               static_partition: Optional[str] = None) -> Dict[str, str]:
    """Build the env a client must receive *before* CUDA init.

    PINNED_DEVICE_MEM_LIMIT is expressed per device. We emit the explicit byte
    count with an `M` suffix (MiB) because the control interface accepts
    K/M/G-suffixed sizes and plain bytes are easy to mis-scale; the raw byte
    count is recorded separately in the run manifest.
    """
    env: Dict[str, str] = {"CUDA_MPS_PIPE_DIRECTORY": pipe_dir}
    if log_dir:
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    if active_thread_percentage is not None:
        if not 1 <= int(active_thread_percentage) <= 100:
            raise MpsError("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 必须在 1..100")
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(active_thread_percentage))
    if priority is not None:
        if priority not in PRIORITY_VALUES:
            raise MpsError(f"未知优先级 {priority}，仅支持 {sorted(PRIORITY_VALUES)}")
        env["CUDA_MPS_CLIENT_PRIORITY"] = PRIORITY_VALUES[priority]
    if pinned_device_mem_limit_bytes is not None:
        mib = int(pinned_device_mem_limit_bytes) // (1024 * 1024)
        if mib < 1:
            raise MpsError("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT 小于 1MiB，配置无意义")
        # Device-scoped form: "<dev_id>=<size>" is accepted; for a single visible
        # device we use index 0 of the *container* view, and we separately verify
        # the container really only sees the target physical UUID.
        env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"0={mib}M"
    if static_partition is not None:
        # Recorded for auditability; actual binding is performed by the control
        # daemon when the capability exists.
        env["MPS_BENCH_STATIC_PARTITION"] = static_partition
    if gpu_uuid:
        env["MPS_BENCH_TARGET_GPU_UUID"] = gpu_uuid
    return env


def memory_limit_bytes(total_bytes: int, fraction: float) -> int:
    if total_bytes <= 0:
        raise MpsError("物理总显存必须为正数（应来自运行时查询，而非硬编码）")
    if not 0 < fraction <= 1:
        raise MpsError("显存比例必须在 (0,1]")
    return int(total_bytes * fraction)


# --------------------------------------------------------------------------- #
# control output parsing
# --------------------------------------------------------------------------- #

_PID_RE = re.compile(r"\b(\d{2,7})\b")


def parse_server_list(stdout: str) -> List[int]:
    return [int(m) for line in stdout.splitlines() for m in _PID_RE.findall(line.strip())
            if line.strip() and not line.lower().startswith(("no ", "error"))]


def parse_client_list(stdout: str) -> List[int]:
    pids: List[int] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("no ", "error")):
            continue
        found = _PID_RE.findall(line)
        if found:
            pids.append(int(found[0]))
    return pids


@dataclass
class TerminateResult:
    """`terminate_client` semantics.

    Success means the MPS server confirmed the client's CUDA contexts were
    destroyed. It does NOT kill the OS process -- that is a separate, explicit
    decision. Failure/timeout must be recorded as such and escalated per the
    recovery policy; it is never a "safe exit success".
    """
    ok: bool
    cuda_status: Optional[str]
    raw: str
    timed_out: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "cuda_status": self.cuda_status,
                "timed_out": self.timed_out, "raw_tail": self.raw[-2000:]}


_CUDA_STATUS_RE = re.compile(r"(CUDA_(?:SUCCESS|ERROR_[A-Z_]+))")


def parse_terminate_output(stdout: str, stderr: str, returncode: int,
                           timed_out: bool = False) -> TerminateResult:
    raw = (stdout or "") + (stderr or "")
    match = _CUDA_STATUS_RE.search(raw)
    status = match.group(1) if match else None
    low = raw.lower()
    # Shell exit code alone is not evidence: we require an explicit CUDA_SUCCESS
    # or an unambiguous success phrase, and any non-success CUDA status vetoes.
    ok = (not timed_out and returncode == 0
          and (status == "CUDA_SUCCESS"
               or "successfully terminated" in low
               or ("terminated" in low and status is None
                   and "error" not in low and "fail" not in low)))
    if status is not None and status != "CUDA_SUCCESS":
        ok = False
    return TerminateResult(ok=ok, cuda_status=status, raw=raw, timed_out=timed_out)


_SERVER_STATE_RE = re.compile(r"\b(ACTIVE|INACTIVE|FAULT|PENDING)\b", re.IGNORECASE)


def parse_server_state(stdout: str) -> Optional[str]:
    match = _SERVER_STATE_RE.search(stdout or "")
    return match.group(1).upper() if match else None


@dataclass
class HelpCapabilities:
    raw: str
    commands: List[str] = field(default_factory=list)

    def has(self, command: str) -> bool:
        return command in self.commands

    @property
    def static_partitioning(self) -> bool:
        low = self.raw.lower()
        return ("static_partitioning" in low or "--static-partitioning" in low
                or "set_default_device_partition" in low or "create_device_partition" in low)


_HELP_CMD_RE = re.compile(r"^\s*([a-z_][a-z0-9_]{3,})", re.IGNORECASE)

_KNOWN_COMMANDS = (
    "get_server_list", "get_client_list", "get_server_status", "server_status",
    "terminate_client", "quit", "quit_if_idle",
    "set_default_active_thread_percentage", "get_default_active_thread_percentage",
    "set_active_thread_percentage", "get_device_client_list",
    "set_default_device_pinned_mem_limit", "get_default_device_pinned_mem_limit",
    "create_device_partition", "destroy_device_partition", "list_device_partition",
    "set_default_device_partition", "get_device_partition_list",
)


def parse_help(stdout: str) -> HelpCapabilities:
    found = []
    low = stdout or ""
    for cmd in _KNOWN_COMMANDS:
        if cmd in low:
            found.append(cmd)
    return HelpCapabilities(raw=stdout or "", commands=sorted(set(found)))


# --------------------------------------------------------------------------- #
# controller
# --------------------------------------------------------------------------- #

@dataclass
class MpsController:
    pipe_dir: str
    log_dir: str
    binary: str = "nvidia-cuda-mps-control"
    gpu_uuid: Optional[str] = None
    log: Optional[CommandLog] = None
    owns_daemon: bool = False
    _capabilities: Optional[HelpCapabilities] = None

    # ---- low level ----
    def _env(self) -> Dict[str, str]:
        return {"CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
                "CUDA_MPS_LOG_DIRECTORY": self.log_dir}

    def _control_stdin(self, command: str, timeout_s: float) -> CommandResult:
        # The documented form is `echo <cmd> | nvidia-cuda-mps-control`. We keep
        # the same contract without a shell: python writes the command to stdin.
        import subprocess
        argv = [self.binary]
        env = dict(os.environ)
        env.update(self._env())
        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(argv, input=command + "\n", capture_output=True, text=True,
                                  timeout=timeout_s, env=env, shell=False)
            rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out, rc = True, -9
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            err = exc.stderr if isinstance(exc.stderr, str) else ""
        except FileNotFoundError as exc:
            rc, out, err = 127, "", str(exc)
        result = CommandResult(argv=argv + ["<stdin>", command], returncode=rc, stdout=out,
                               stderr=err, duration_s=time.monotonic() - start, timed_out=timed_out)
        if self.log is not None:
            self.log.add(result, env=self._env(), note=f"mps-control: {command}")
        return result

    # ---- capability probing (read-only) ----
    def binary_path(self) -> Optional[str]:
        return which(self.binary)

    def capabilities(self) -> HelpCapabilities:
        if self._capabilities is None:
            res = self._control_stdin("help", timeout_s=20)
            self._capabilities = parse_help(res.stdout + res.stderr)
        return self._capabilities

    def daemon_present(self) -> bool:
        """A reachable control pipe. NOTE: daemon presence alone never proves a
        workload is actually attached to MPS."""
        res = self._control_stdin("get_server_list", timeout_s=15)
        return res.ok

    # ---- lifecycle ----
    def start_daemon(self, timeout_s: float = 30, allow_adopt_existing: bool = False) -> Dict[str, Any]:
        os.makedirs(self.pipe_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.daemon_present():
            if not allow_adopt_existing:
                raise MpsError(
                    f"{self.pipe_dir} 已存在可用的 MPS 控制实例。默认不接管既有实例："
                    "无法证明资源归属/安全共存时停止并报告。如确认归属本实验，"
                    "设置 mps.allow_adopt_existing=true。")
            self.owns_daemon = False
            return {"adopted": True, "pipe_dir": self.pipe_dir}
        res = run([self.binary, "-d"], timeout_s=timeout_s, env=self._env(), log=self.log,
                  note="start mps daemon")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.daemon_present():
                self.owns_daemon = True
                return {"adopted": False, "pipe_dir": self.pipe_dir,
                        "start_stdout": res.stdout[-2000:]}
            time.sleep(0.5)
        raise MpsError(f"MPS daemon 启动超时：{res.stderr.strip() or res.stdout.strip()}")

    def stop_daemon(self, timeout_s: float = 60) -> Dict[str, Any]:
        """Only ever stops the daemon this experiment started. We never issue a
        global quit against someone else's MPS instance."""
        if not self.owns_daemon:
            return {"stopped": False, "reason": "本实验不拥有该 MPS 实例，拒绝执行 quit"}
        res = self._control_stdin("quit", timeout_s=timeout_s)
        return {"stopped": res.ok, "stdout_tail": res.stdout[-1000:], "returncode": res.returncode}

    # ---- queries (evidence) ----
    def server_pids(self) -> List[int]:
        return parse_server_list(self._control_stdin("get_server_list", timeout_s=15).stdout)

    def client_pids(self, server_pid: Optional[int] = None) -> List[int]:
        cmd = f"get_client_list {server_pid}" if server_pid is not None else "get_client_list"
        return parse_client_list(self._control_stdin(cmd, timeout_s=15).stdout)

    def server_state(self) -> Optional[str]:
        caps = self.capabilities()
        for cmd in ("get_server_status", "server_status"):
            if caps.has(cmd):
                return parse_server_state(self._control_stdin(cmd, timeout_s=15).stdout)
        return None

    def evidence(self) -> Dict[str, Any]:
        servers = self.server_pids()
        clients: Dict[str, List[int]] = {}
        for pid in servers:
            clients[str(pid)] = self.client_pids(pid)
        return {"pipe_dir": self.pipe_dir, "log_dir": self.log_dir,
                "server_pids": servers, "client_pids_by_server": clients,
                "server_state": self.server_state(),
                "control_commands": self.capabilities().commands}

    # ---- fault handling ----
    def terminate_client(self, server_pid: int, client_pid: int,
                         timeout_s: float = 30) -> TerminateResult:
        """client_pid must be the PID as seen by the MPS control process'
        namespace -- resolve it via pidmap, not docker PID 1."""
        if not self.capabilities().has("terminate_client"):
            return TerminateResult(ok=False, cuda_status=None,
                                   raw="terminate_client 不在本机 MPS control help 中")
        res = self._control_stdin(f"terminate_client {server_pid} {client_pid}", timeout_s=timeout_s)
        return parse_terminate_output(res.stdout, res.stderr, res.returncode, res.timed_out)

    # ---- static SM partitioning (independent capability) ----
    def static_partitioning_supported(self) -> bool:
        return self.capabilities().static_partitioning

    def create_partitions(self, partitions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.static_partitioning_supported():
            return {"supported": False,
                    "reason": "本机 MPS control 未提供静态 SM 分区命令；HH-S 用例 SKIP。"
                              "禁止用 ACTIVE_THREAD_PERCENTAGE / 上下文亲和性 / MIG 冒充静态隔离"}
        created = []
        for part in partitions:
            name, sm = part.get("name"), part.get("sm_count")
            res = self._control_stdin(f"create_device_partition 0 {sm}", timeout_s=30)
            created.append({"name": name, "requested_sm": sm, "ok": res.ok,
                            "stdout_tail": res.stdout[-1000:]})
        listing = self._control_stdin("list_device_partition", timeout_s=20)
        return {"supported": True, "created": created, "listing": listing.stdout[-4000:]}

    def destroy_partitions(self) -> Dict[str, Any]:
        if not self.static_partitioning_supported():
            return {"supported": False}
        res = self._control_stdin("destroy_device_partition 0", timeout_s=30)
        return {"supported": True, "ok": res.ok, "stdout_tail": res.stdout[-1000:]}
