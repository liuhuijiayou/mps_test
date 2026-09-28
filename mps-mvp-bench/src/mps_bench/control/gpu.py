"""Read-only GPU inventory via nvidia-smi.

We deliberately do NOT use pynvml on the host: the host orchestrator must stay
import-light and must not link a CUDA python binding. nvidia-smi query output is
parsed defensively; anything unparseable becomes None (never 0).

Important: the `CUDA Version` shown by nvidia-smi is a *driver* capability
field. It does not prove which CUDA runtime/toolkit exists inside a container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .exec import CommandLog, run


@dataclass
class GpuInfo:
    index: Optional[int]
    uuid: str
    name: str
    memory_total_mib: Optional[int]
    memory_used_mib: Optional[int]
    memory_free_mib: Optional[int]
    compute_mode: Optional[str]
    mig_mode: Optional[str]
    driver_version: Optional[str]
    sm_count: Optional[int] = None
    raw: str = ""

    @property
    def memory_total_bytes(self) -> Optional[int]:
        return None if self.memory_total_mib is None else self.memory_total_mib * 1024 * 1024


@dataclass
class GpuProcess:
    pid: int
    uuid: str
    used_memory_mib: Optional[int]
    process_name: str

    def __init__(self, pid: int, uuid: str = "", used_memory_mib: Optional[int] = None,
                 process_name: str = "", *, gpu_uuid: Optional[str] = None,
                 memory_mib: Optional[int] = None, name: Optional[str] = None):
        # Accept both the nvidia-smi column names and the shorter aliases used by
        # callers/tests; there is exactly one underlying field per concept.
        self.pid = int(pid)
        self.uuid = gpu_uuid if gpu_uuid is not None else uuid
        self.used_memory_mib = memory_mib if memory_mib is not None else used_memory_mib
        self.process_name = name if name is not None else process_name

    @property
    def gpu_uuid(self) -> str:
        return self.uuid

    @property
    def memory_mib(self) -> Optional[int]:
        return self.used_memory_mib

    @property
    def name(self) -> str:
        return self.process_name

    def as_dict(self) -> Dict[str, Any]:
        return {"pid": self.pid, "gpu_uuid": self.uuid,
                "used_memory_mib": self.used_memory_mib, "process_name": self.process_name}


_QUERY_FIELDS = ("index,uuid,name,memory.total,memory.used,memory.free,"
                 "compute_mode,mig.mode.current,driver_version")


def _to_int(token: str) -> Optional[int]:
    token = token.strip()
    if not token or token.lower() in ("[n/a]", "n/a", "[not supported]", "not supported", "unknown"):
        return None
    token = token.split()[0]
    try:
        return int(float(token))
    except ValueError:
        return None


def _clean(token: str) -> Optional[str]:
    token = token.strip()
    if not token or token.lower() in ("[n/a]", "n/a", "[not supported]", "not supported"):
        return None
    return token


def parse_gpu_query(stdout: str) -> Any:
    """Parse `nvidia-smi --query-gpu=...` CSV.

    Returns a `GpuList`: a list of GpuInfo that also proxies attribute access to
    its single element, so single-GPU callers can use it directly.
    """
    gpus: List[GpuInfo] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 9:
            continue
        gpus.append(GpuInfo(
            index=_to_int(cols[0]),
            uuid=cols[1],
            name=cols[2],
            memory_total_mib=_to_int(cols[3]),
            memory_used_mib=_to_int(cols[4]),
            memory_free_mib=_to_int(cols[5]),
            compute_mode=_clean(cols[6]),
            mig_mode=_clean(cols[7]),
            driver_version=_clean(cols[8]),
            raw=line,
        ))
    return GpuList(gpus)


class GpuList(list):
    """List of GpuInfo; when exactly one GPU was parsed, attribute access is
    forwarded to it. With 0 or 2+ GPUs this raises instead of guessing."""

    def __getattr__(self, name: str) -> Any:
        if len(self) == 1:
            return getattr(list.__getitem__(self, 0), name)
        raise AttributeError(
            f"解析出 {len(self)} 张 GPU，不能直接访问 {name!r}；请先按 UUID 选定")


def parse_apps_query(stdout: str) -> List[GpuProcess]:
    procs: List[GpuProcess] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 3:
            continue
        pid = _to_int(cols[0])
        if pid is None:
            continue
        procs.append(GpuProcess(pid=pid, uuid=cols[1], used_memory_mib=_to_int(cols[2]),
                                process_name=cols[3] if len(cols) > 3 else ""))
    return procs


@dataclass
class GpuQuery:
    binary: str = "nvidia-smi"
    log: Optional[CommandLog] = None
    errors: List[str] = field(default_factory=list)

    def list_gpus(self) -> List[GpuInfo]:
        res = run([self.binary, f"--query-gpu={_QUERY_FIELDS}", "--format=csv,noheader,nounits"],
                  timeout_s=30, log=self.log, note="query-gpu")
        if not res.ok:
            self.errors.append(f"nvidia-smi query-gpu 失败: {res.stderr.strip() or res.returncode}")
            return []
        return parse_gpu_query(res.stdout)

    def get(self, uuid: str) -> Optional[GpuInfo]:
        for gpu in self.list_gpus():
            if gpu.uuid == uuid:
                return gpu
        return None

    def compute_processes(self, uuid: Optional[str] = None) -> List[GpuProcess]:
        res = run([self.binary, "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                   "--format=csv,noheader,nounits"], timeout_s=30, log=self.log,
                  note="query-compute-apps")
        if not res.ok:
            self.errors.append("nvidia-smi query-compute-apps 失败")
            return []
        procs = parse_apps_query(res.stdout)
        return [p for p in procs if uuid is None or p.uuid == uuid]

    def raw_dump(self, uuid: str) -> str:
        res = run([self.binary, "-q", "-i", uuid], timeout_s=40, log=self.log, note="nvidia-smi -q")
        return res.stdout if res.ok else res.stderr

    def sm_count(self, uuid: str) -> Optional[int]:
        """SM count is not in --query-gpu; parse `nvidia-smi -q` if present,
        otherwise leave unknown rather than guessing from the model name."""
        dump = self.raw_dump(uuid)
        for line in dump.splitlines():
            low = line.lower()
            if "multiprocessor" in low or "number of sms" in low:
                value = _to_int(line.split(":")[-1])
                if value:
                    return value
        return None

    def set_compute_mode(self, uuid: str, mode: str) -> bool:
        """Only called after explicit user authorization; scoped to one UUID."""
        res = run([self.binary, "-i", uuid, "-c", mode], timeout_s=30, log=self.log,
                  note=f"set compute mode {mode} on {uuid}")
        return res.ok
