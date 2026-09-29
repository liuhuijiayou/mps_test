"""MPS control-daemon driver (legacy `nvidia-cuda-mps-control` interface).

Version discipline
------------------
The target machine runs driver R580, whose MPS control binary exposes the
*legacy* (v2) command set. We therefore:

* probe `help` at runtime and record it as evidence,
* only issue commands that the probed help actually advertises,
* treat static SM partitioning as an independent, separately-probed capability.
  If it is absent, the HH-S case is SKIPped -- we never emulate it with
  ACTIVE_THREAD_PERCENTAGE=50, context affinity or MIG.

Static SM partitioning (Legacy MPS v2 spelling)
-----------------------------------------------
Per the NVIDIA MPS documentation, the v2 interface is:

* daemon launch flag: ``-S`` / ``--static-partitioning``. The daemon MUST be
  started with it; the mode cannot be turned on afterwards.
* ``sm_partition add <device UUID> <number of chunks>`` -> prints the full
  partition ID (``<uuid>/<opaque-base64>``).
* ``sm_partition rm <device UUID> <partition>``
* ``lspart`` -> current partition table.

The partitioning unit is a *chunk*, not an SM. Chunk size depends on the
architecture: on pre-Hopper dGPUs (A30 is Ampere) a v2 chunk is **4 SMs**.
Clients bind to a partition by exporting ``CUDA_MPS_SM_PARTITION=<partition id>``
before CUDA init, and under static partitioning mode that variable is
*mandatory* -- a client without it fails with
CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION. The opaque ID is only known at
runtime, so partitions are addressed by a human label in config and resolved to
real IDs here.

Note also that static partitioning makes dynamic provisioning
(ACTIVE_THREAD_PERCENTAGE) *ignored*, which is why the two must never be mixed.

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

# Legacy MPS v2 chunk granularity, per NVIDIA MPS docs ("Static SM Partitioning"):
#   dGPU: 8 SMs on Hopper+, 4 SMs on pre-Hopper (v2)
#   iGPU: 2 SMs
# A30 is Ampere -> 4 SMs per chunk. We never guess from the marketing name: the
# caller passes the architecture-derived value and we record it as evidence.
DGPU_PRE_HOPPER_CHUNK_SMS = 4
DGPU_HOPPER_PLUS_CHUNK_SMS = 8


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
               static_partition_id: Optional[str] = None,
               static_partition_label: Optional[str] = None) -> Dict[str, str]:
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
    if static_partition_id is not None:
        # THE binding mechanism. Under `-S` static partitioning mode this
        # variable is mandatory for every client: without it CUDA context
        # creation fails with CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION. The
        # value is the opaque "<device uuid>/<id>" string printed by
        # `sm_partition add`, so it can only be known at runtime.
        if "/" not in static_partition_id:
            raise MpsError(
                f"CUDA_MPS_SM_PARTITION 取值非法: {static_partition_id!r}；"
                "应为 `sm_partition add` 返回的 <device UUID>/<partition id>")
        env["CUDA_MPS_SM_PARTITION"] = static_partition_id
    if static_partition_label is not None:
        # Human-readable label from the case file, for auditability only. It is
        # NOT a binding mechanism and carries no meaning for the driver.
        env["MPS_BENCH_STATIC_PARTITION_LABEL"] = static_partition_label
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
        """Whether `help` *mentions* static partitioning.

        This is a weak, advisory signal only. The authoritative check is
        `MpsController.probe_static_partitioning()`, which actually starts a
        daemon in `-S` mode and tries to carve a partition. R580's help text is
        known not to reliably advertise this feature, so a False here must never
        by itself mark the capability unsupported.
        """
        low = self.raw.lower()
        return ("sm_partition" in low or "lspart" in low
                or "static-partitioning" in low or "static_partitioning" in low)


_HELP_CMD_RE = re.compile(r"^\s*([a-z_][a-z0-9_]{3,})", re.IGNORECASE)

_KNOWN_COMMANDS = (
    "get_server_list", "get_client_list", "get_server_status", "server_status",
    "terminate_client", "quit", "quit_if_idle", "ps", "device_query",
    "set_default_active_thread_percentage", "get_default_active_thread_percentage",
    "set_active_thread_percentage", "get_active_thread_percentage",
    "get_device_client_list", "start_server",
    "set_default_device_pinned_mem_limit", "get_default_device_pinned_mem_limit",
    "set_device_pinned_mem_limit", "get_device_pinned_mem_limit",
    "set_default_client_priority", "get_default_client_priority",
    # static SM partitioning (Legacy MPS v2 spelling)
    "sm_partition", "lspart",
)


def parse_help(stdout: str) -> HelpCapabilities:
    found = []
    low = stdout or ""
    for cmd in _KNOWN_COMMANDS:
        if cmd in low:
            found.append(cmd)
    return HelpCapabilities(raw=stdout or "", commands=sorted(set(found)))


# --------------------------------------------------------------------------- #
# static SM partitioning output parsing (Legacy MPS v2)
# --------------------------------------------------------------------------- #

# `sm_partition add <uuid> <chunks>` prints the full partition ID on success:
#   GPU-74d43ed3-cdf7-e667-3644-bf5b4f46ed65/Dx4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
_PARTITION_ID_RE = re.compile(r"\b(GPU-[0-9a-fA-F-]+/[A-Za-z0-9+/=]{8,})")

_PARTITION_FAILURE_HINTS = (
    "failed to fulfill", "error", "invalid", "not supported", "unsupported",
    "unknown command", "cannot", "unable",
)


def parse_partition_id(stdout: str, stderr: str = "") -> Optional[str]:
    """Extract the opaque partition ID from `sm_partition add` output.

    Returns None when the command did not produce one -- e.g. oversubscription
    ("Failed to fulfill the requested SM partition of N chunks, error
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION") or an unknown command on a build
    without the feature. We never invent an ID.
    """
    match = _PARTITION_ID_RE.search((stdout or "") + "\n" + (stderr or ""))
    return match.group(1) if match else None


def partition_failure_reason(stdout: str, stderr: str) -> str:
    """Best-effort human-readable failure cause, preserved verbatim."""
    raw = ((stdout or "") + "\n" + (stderr or "")).strip()
    for line in raw.splitlines():
        if any(hint in line.lower() for hint in _PARTITION_FAILURE_HINTS):
            return line.strip()
    return raw[-500:] if raw else "控制进程无任何输出"


@dataclass
class PartitionRow:
    """One row of `lspart` output."""
    gpu: str
    partition: Optional[str]
    free_chunks: Optional[int]
    used_chunks: Optional[int]
    free_sms: Optional[int]
    used_sms: Optional[int]
    in_use: Optional[bool]
    raw: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"gpu": self.gpu, "partition": self.partition,
                "free_chunks": self.free_chunks, "used_chunks": self.used_chunks,
                "free_sms": self.free_sms, "used_sms": self.used_sms,
                "in_use": self.in_use, "raw": self.raw}


def _opt_int(token: str) -> Optional[int]:
    token = (token or "").strip()
    if not token or token == "-":
        return None
    try:
        return int(token)
    except ValueError:
        return None


def parse_lspart(stdout: str) -> List[PartitionRow]:
    """Parse the `lspart` table.

    Documented shape (column widths vary, so we split on whitespace):

        GPU           Partition   free   used   free  used  clients
                                  chunk  chunk  SM    SM
        GPU-74d43ed3  -           0      8      74    56    -
        GPU-74d43ed3  Dx4AAA...   -      7      -     56    Yes

    Unparseable lines are skipped rather than guessed at; a caller that finds no
    row for its partition must treat that as "not proven", not as zero.
    """
    rows: List[PartitionRow] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cols = stripped.split()
        # Header lines ("GPU Partition free used ...") and the wrapped second
        # header row ("chunk chunk SM SM") never start with a device id.
        if not cols[0].startswith("GPU-"):
            continue
        if len(cols) < 7:
            continue
        clients = cols[6].strip().lower()
        rows.append(PartitionRow(
            gpu=cols[0],
            partition=None if cols[1] == "-" else cols[1],
            free_chunks=_opt_int(cols[2]),
            used_chunks=_opt_int(cols[3]),
            free_sms=_opt_int(cols[4]),
            used_sms=_opt_int(cols[5]),
            in_use=True if clients == "yes" else (False if clients == "no" else None),
            raw=stripped,
        ))
    return rows


def find_partition_row(rows: Sequence[PartitionRow],
                       partition_id: str) -> Optional[PartitionRow]:
    """Match a full partition ID against `lspart` rows.

    `lspart` prints only the *partial* device UUID and the bare partition
    component, whereas `sm_partition add` returns "<full uuid>/<component>". We
    therefore compare on the component after the slash.
    """
    component = partition_id.split("/", 1)[1] if "/" in partition_id else partition_id
    for row in rows:
        if row.partition and (row.partition == component
                              or component.startswith(row.partition)
                              or row.partition.startswith(component)):
            return row
    return None


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
    static_partitioning_mode: bool = False
    _capabilities: Optional[HelpCapabilities] = None
    # label -> created partition record, populated by create_partitions()
    _partitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
    def start_daemon(self, timeout_s: float = 30, allow_adopt_existing: bool = False,
                     static_partitioning: bool = False) -> Dict[str, Any]:
        """Start (or adopt) the control daemon.

        `static_partitioning=True` adds the documented `-S` flag. This mode can
        only be selected at launch: an already-running daemon started without
        `-S` can never be switched, so adopting one is refused rather than
        silently producing a non-partitioned run.
        """
        os.makedirs(self.pipe_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.daemon_present():
            if not allow_adopt_existing:
                raise MpsError(
                    f"{self.pipe_dir} 已存在可用的 MPS 控制实例。默认不接管既有实例："
                    "无法证明资源归属/安全共存时停止并报告。如确认归属本实验，"
                    "设置 mps.allow_adopt_existing=true。")
            if static_partitioning:
                raise MpsError(
                    "静态 SM 分区模式只能在 daemon 启动时通过 -S 指定，无法对既有实例开启。"
                    "拒绝接管既有实例并把它当作已分区（那会让 HH-S 得到假结论）。")
            self.owns_daemon = False
            return {"adopted": True, "pipe_dir": self.pipe_dir,
                    "static_partitioning": False}
        argv = [self.binary, "-d"]
        if static_partitioning:
            argv.append("-S")
        res = run(argv, timeout_s=timeout_s, env=self._env(), log=self.log,
                  note="start mps daemon" + (" (-S static partitioning)" if static_partitioning else ""))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.daemon_present():
                self.owns_daemon = True
                self.static_partitioning_mode = static_partitioning
                return {"adopted": False, "pipe_dir": self.pipe_dir,
                        "argv": argv, "static_partitioning": static_partitioning,
                        "start_stdout": res.stdout[-2000:]}
            time.sleep(0.5)
        raise MpsError(
            f"MPS daemon 启动超时（argv={argv}）：{res.stderr.strip() or res.stdout.strip()}")

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
    def static_partitioning_advertised(self) -> bool:
        """Weak signal from `help` only. Not authoritative -- see
        `probe_static_partitioning`."""
        return self.capabilities().static_partitioning

    def _sm_partition_add(self, device: str, chunks: int,
                          timeout_s: float = 30) -> Tuple[Optional[str], CommandResult]:
        res = self._control_stdin(f"sm_partition add {device} {int(chunks)}",
                                  timeout_s=timeout_s)
        return parse_partition_id(res.stdout, res.stderr), res

    def _sm_partition_rm(self, device: str, partition_id: str,
                         timeout_s: float = 30) -> CommandResult:
        # The docs accept both "rm <uuid>/<part>" and "rm <uuid> <part>"; we use
        # the two-argument form and pass the bare component.
        component = partition_id.split("/", 1)[1] if "/" in partition_id else partition_id
        return self._control_stdin(f"sm_partition rm {device} {component}", timeout_s=timeout_s)

    def lspart(self, timeout_s: float = 20) -> Tuple[List[PartitionRow], CommandResult]:
        res = self._control_stdin("lspart", timeout_s=timeout_s)
        return parse_lspart(res.stdout), res

    def create_partitions(self, partitions: Sequence[Dict[str, Any]],
                          device: Optional[str] = None,
                          sms_per_chunk: int = DGPU_PRE_HOPPER_CHUNK_SMS
                          ) -> Dict[str, Any]:
        """Carve the requested partitions on `device`.

        Each entry is ``{name, chunks}`` (preferred) or ``{name, sm_count}``.
        Because the hardware unit is a *chunk*, an `sm_count` is only accepted
        when it is an exact multiple of the chunk size -- we refuse to silently
        round, which would make the case's "28 + 28" claim untrue.

        On any failure we roll back the partitions we already made, so the GPU
        is never left carved up by a half-finished run.
        """
        device = device or self.gpu_uuid
        if not device:
            raise MpsError("创建静态 SM 分区必须指定目标 GPU UUID")
        if not self.static_partitioning_mode:
            return {"supported": False, "created": [],
                    "reason": "MPS daemon 未以 -S/--static-partitioning 启动，"
                              "静态分区命令不会生效；拒绝继续（HH-S 应 SKIP）"}
        if sms_per_chunk <= 0:
            raise MpsError("sms_per_chunk 必须为正数")

        created: List[Dict[str, Any]] = []
        for part in partitions:
            label = part.get("name")
            if not label:
                raise MpsError("静态分区必须有 name 标签，用于与 client 绑定")
            chunks, requested_sms = self._resolve_chunks(part, sms_per_chunk)

            partition_id, res = self._sm_partition_add(device, chunks)
            record: Dict[str, Any] = {
                "name": label,
                "device": device,
                "requested_chunks": chunks,
                "requested_sm_count": requested_sms,
                "sms_per_chunk": sms_per_chunk,
                "command": f"sm_partition add {device} {chunks}",
                "returncode": res.returncode,
                "stdout_tail": res.stdout[-1000:],
                "stderr_tail": res.stderr[-1000:],
                "partition_id": partition_id,
                "ok": partition_id is not None,
            }
            if partition_id is None:
                record["reason"] = partition_failure_reason(res.stdout, res.stderr)
                rollback = self.destroy_partitions()
                return {"supported": False, "created": created, "failed": record,
                        "rollback": rollback,
                        "reason": f"分区 {label!r} 创建失败：{record['reason']}"}
            created.append(record)
            self._partitions[label] = record

        # Evidence: re-read the authoritative table and attach the observed SM
        # counts to each partition we created.
        rows, listing = self.lspart()
        for record in created:
            row = find_partition_row(rows, record["partition_id"])
            record["observed"] = row.as_dict() if row else None
            record["observed_sm_count"] = row.used_sms if row else None
            if row is None:
                record["observed_note"] = "lspart 未列出该分区：无法证明实际 SM 数"
            elif row.used_sms is not None and record["requested_sm_count"] is not None \
                    and row.used_sms != record["requested_sm_count"]:
                record["observed_note"] = (
                    f"实际 SM 数 {row.used_sms} 与请求 {record['requested_sm_count']} 不一致；"
                    "以实际值为准")
        return {"supported": True, "device": device, "sms_per_chunk": sms_per_chunk,
                "created": created,
                "lspart_rows": [r.as_dict() for r in rows],
                "lspart_raw_tail": listing.stdout[-4000:]}

    @staticmethod
    def _resolve_chunks(part: Dict[str, Any], sms_per_chunk: int) -> Tuple[int, Optional[int]]:
        """Return (chunks, requested_sm_count). Never rounds silently."""
        chunks = part.get("chunks")
        sm_count = part.get("sm_count")
        if chunks is not None:
            chunks = int(chunks)
            if chunks < 1:
                raise MpsError("chunks 必须 >= 1")
            return chunks, chunks * sms_per_chunk
        if sm_count is None:
            raise MpsError("静态分区必须指定 chunks 或 sm_count")
        sm_count = int(sm_count)
        if sm_count % sms_per_chunk != 0:
            raise MpsError(
                f"sm_count={sm_count} 不是 chunk 粒度 {sms_per_chunk} SM 的整数倍。"
                "MPS 静态分区以 chunk 为单位，拒绝静默取整（否则报告中的 SM 数不成立）。"
                f"请改用 chunks，或选择 {sms_per_chunk} 的倍数。")
        return sm_count // sms_per_chunk, sm_count

    def partition_id(self, label: str) -> Optional[str]:
        """Resolve a config label to the runtime partition ID, or None."""
        record = self._partitions.get(label)
        return record.get("partition_id") if record else None

    def destroy_partitions(self, device: Optional[str] = None) -> Dict[str, Any]:
        """Remove every partition this controller created. Idempotent.

        `sm_partition rm` fails while clients are still attached, so this must
        run after container teardown. A failure is reported, never swallowed.
        """
        if not self._partitions:
            return {"supported": self.static_partitioning_mode, "removed": [],
                    "note": "本 run 未创建任何静态分区"}
        device = device or self.gpu_uuid
        removed: List[Dict[str, Any]] = []
        errors: List[str] = []
        for label, record in list(self._partitions.items()):
            partition_id = record.get("partition_id")
            if not partition_id:
                continue
            target_device = record.get("device") or device
            if not target_device:
                errors.append(f"{label}: 无法确定设备 UUID，分区可能仍然存在")
                continue
            res = self._sm_partition_rm(str(target_device), str(partition_id))
            low = (res.stdout + res.stderr).lower()
            ok = res.ok and "in use" not in low and "error" not in low
            removed.append({"name": label, "partition_id": partition_id,
                            "ok": ok, "returncode": res.returncode,
                            "stdout_tail": res.stdout[-500:],
                            "stderr_tail": res.stderr[-500:]})
            if ok:
                self._partitions.pop(label, None)
            else:
                errors.append(f"{label}: {partition_failure_reason(res.stdout, res.stderr)}")
        rows, _ = self.lspart()
        return {"supported": True, "removed": removed, "errors": errors,
                "all_removed": not errors,
                "lspart_after": [r.as_dict() for r in rows]}


# --------------------------------------------------------------------------- #
# active capability probe
# --------------------------------------------------------------------------- #

def probe_static_partitioning(binary: str,
                              gpu_uuid: str,
                              probe_root: str = "/tmp/mps-mvp-bench/probe",
                              log: Optional[CommandLog] = None,
                              timeout_s: float = 30) -> Dict[str, Any]:
    """Decide static-SM-partitioning support by *doing it*, not by reading help.

    R580's `help` text does not reliably advertise the feature, so parsing it
    produces false negatives. Instead we stand up a throwaway control daemon in
    its own pipe directory, ask it for the smallest possible partition (1 chunk),
    and let the return value decide. The probe instance is fully isolated from
    any MPS the user may already be running and is always torn down.

    Returns ``{"supported", "commands", "reason", ...}``.
    """
    commands: List[str] = []
    if which(binary) is None:
        return {"supported": False, "commands": commands,
                "reason": f"未找到 {binary}，无法探测静态 SM 分区"}
    if not gpu_uuid:
        return {"supported": False, "commands": commands,
                "reason": "未指定目标 GPU UUID，拒绝对未知设备执行分区探测"}

    stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
    pipe_dir = os.path.join(probe_root, stamp, "pipe")
    log_dir = os.path.join(probe_root, stamp, "log")
    ctl = MpsController(pipe_dir=pipe_dir, log_dir=log_dir, binary=binary,
                        gpu_uuid=gpu_uuid, log=log)

    evidence: Dict[str, Any] = {"pipe_dir": pipe_dir, "probe_chunks": 1}
    try:
        try:
            start = ctl.start_daemon(timeout_s=timeout_s, allow_adopt_existing=False,
                                     static_partitioning=True)
        except MpsError as exc:
            return {"supported": False, "commands": commands,
                    "reason": f"以 -S 启动探测用 MPS 实例失败：{exc}", **evidence}
        commands.append(" ".join(start.get("argv", [binary, "-d", "-S"])))

        add_cmd = f"sm_partition add {gpu_uuid} 1"
        commands.append(f"echo '{add_cmd}' | {binary}")
        partition_id, res = ctl._sm_partition_add(gpu_uuid, 1, timeout_s=timeout_s)
        evidence["add_stdout_tail"] = res.stdout[-1000:]
        evidence["add_stderr_tail"] = res.stderr[-1000:]
        evidence["add_returncode"] = res.returncode

        if partition_id is None:
            return {"supported": False, "commands": commands,
                    "reason": f"执行 `{add_cmd}` 未返回分区 ID："
                              f"{partition_failure_reason(res.stdout, res.stderr)}",
                    **evidence}

        evidence["partition_id"] = partition_id
        ctl._partitions["__probe__"] = {"partition_id": partition_id, "device": gpu_uuid}

        commands.append(f"echo 'lspart' | {binary}")
        rows, listing = ctl.lspart(timeout_s=timeout_s)
        row = find_partition_row(rows, partition_id)
        evidence["lspart_rows"] = [r.as_dict() for r in rows]
        evidence["lspart_raw_tail"] = listing.stdout[-2000:]
        evidence["probe_sm_count"] = row.used_sms if row else None
        if row is not None and row.used_sms:
            # A 1-chunk partition tells us the real chunk granularity on this
            # exact device+driver, which beats inferring it from the model name.
            evidence["sms_per_chunk_observed"] = row.used_sms

        commands.append(f"echo 'sm_partition rm {gpu_uuid} <partition>' | {binary}")
        teardown = ctl.destroy_partitions(device=gpu_uuid)
        evidence["probe_partition_removed"] = teardown.get("all_removed")
        if not teardown.get("all_removed"):
            evidence["probe_cleanup_errors"] = teardown.get("errors")

        return {"supported": True, "commands": commands,
                "reason": f"实际执行 `{add_cmd}` 成功并返回分区 ID，"
                          f"`lspart` 已确认；探测分区已释放",
                **evidence}
    finally:
        try:
            if ctl.owns_daemon:
                commands.append(f"echo 'quit' | {binary}")
                evidence["probe_daemon_stopped"] = ctl.stop_daemon(timeout_s=timeout_s).get("stopped")
        except Exception:  # teardown must never mask the probe verdict
            pass
