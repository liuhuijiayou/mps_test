"""Safety gates: GPU-UUID lock, foreign-process refusal, disruptive confirmation.

Everything here is intentionally conservative: when we cannot *prove* that the
target GPU only carries this experiment's work, we refuse to run rather than
risk someone else's job (e.g. a long-running inference server sharing the host).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .gpu import GpuInfo, GpuProcess


class SafetyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# GPU UUID lock
# --------------------------------------------------------------------------- #

@dataclass
class GpuLock:
    """Advisory exclusive lock keyed by GPU UUID, held for the whole run."""
    lock_dir: str
    gpu_uuid: str
    run_id: str
    _path: Optional[str] = None

    @property
    def path(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.gpu_uuid)
        return os.path.join(self.lock_dir, f"{safe}.lock")

    def acquire(self) -> None:
        os.makedirs(self.lock_dir, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            holder = self._read_holder()
            raise SafetyError(
                f"GPU {self.gpu_uuid} 已被本项目另一个 run 持有: {holder}. "
                f"确认其已结束后删除 {self.path}") from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"run_id": self.run_id, "pid": os.getpid(), "ts": time.time()}, fh)
        self._path = self.path

    def _read_holder(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {"unknown": True}

    def release(self) -> None:
        """Idempotent: releasing twice, or releasing a lock we never took, is fine."""
        if self._path is None:
            return
        holder = self._read_holder()
        if holder.get("run_id") != self.run_id:
            return  # never delete someone else's lock
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        self._path = None

    def __enter__(self) -> "GpuLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# foreign process detection
# --------------------------------------------------------------------------- #

@dataclass
class OwnershipCheck:
    ok: bool
    foreign: List[Dict[str, Any]] = field(default_factory=list)
    owned: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "foreign_processes": self.foreign,
                "owned_processes": self.owned, "reason": self.reason}


def check_gpu_processes(processes: Sequence[GpuProcess],
                        owned_pids: Iterable[int],
                        gpu_uuid: str) -> OwnershipCheck:
    """Refuse to run if the target GPU carries any process we do not own.

    The initial screenshot is not runtime state; this must be re-evaluated before
    every round.
    """
    owned = set(int(p) for p in owned_pids)
    foreign, mine = [], []
    for proc in processes:
        if proc.uuid != gpu_uuid:
            continue
        (mine if proc.pid in owned else foreign).append(proc.as_dict())
    if foreign:
        return OwnershipCheck(
            ok=False, foreign=foreign, owned=mine,
            reason=f"目标 GPU {gpu_uuid} 上存在非本项目 GPU 进程，拒绝运行（每轮前均重新检查）")
    return OwnershipCheck(ok=True, foreign=[], owned=mine, reason="目标 GPU 上仅有本项目进程")


def check_compute_mode(gpu: GpuInfo, mps_mode: bool,
                       require_default_for_nonmps: bool = True) -> "tuple[bool, str]":
    """A non-MPS two-process baseline needs a compute mode that allows multiple
    processes. If the GPU is EXCLUSIVE_PROCESS, a failed baseline must NOT be
    reported as an MPS win."""
    mode = (getattr(gpu, "compute_mode", None) or "unknown").strip()
    normalized = mode.replace(" ", "_").upper()
    multi_ok = normalized in ("DEFAULT", "0", "DEFAULT_(0)")
    if mps_mode or not require_default_for_nonmps:
        return True, ""
    if not multi_ok:
        return False, (f"当前 compute mode={mode}，非 MPS 双进程基线要求 Default（允许多进程）。"
                       "当前模式下基线失败不得记为 MPS 收益；"
                       "如需切换请设置 compute_mode.allow_change=true 并由用户授权。")
    return True, ""


# --------------------------------------------------------------------------- #
# disruptive gate
# --------------------------------------------------------------------------- #

DISRUPTIVE_FAULTS = ("F4", "F5", "F6", "F7", "F8", "F9")


def gate_disruptive(fault_id: Optional[str], cli_flag: bool,
                    config_flag: bool, confirm_uuid: Optional[str],
                    gpu_uuid: Optional[str]) -> None:
    if fault_id not in DISRUPTIVE_FAULTS:
        return
    if not (cli_flag and config_flag):
        raise SafetyError(f"{fault_id} 为破坏性用例：需要 CLI --allow-disruptive 与配置 "
                          "faults.allow_disruptive=true 同时开启")
    if not gpu_uuid or confirm_uuid != gpu_uuid:
        raise SafetyError(f"{fault_id} 需要 --confirm-gpu-uuid <目标 UUID> 与 gpu.uuid 完全一致，"
                          "以确认在可承受故障的专用验证节点上执行")
