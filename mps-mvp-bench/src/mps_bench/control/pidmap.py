"""Container PID -> host PID mapping.

`terminate_client` needs the client PID in the MPS control process' namespace.
Docker PID 1 is usually the entrypoint shell, not the CUDA worker, so we
cross-check three sources:

  1. `docker top` (host-namespace PIDs of everything in the container)
  2. the worker's self-reported in-container PID (written to its status file)
  3. /proc/<host_pid>/status NSpid, which lists the PID in each nested namespace

Only a PID confirmed by NSpid *and* present in the MPS client list is used.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

_NSPID_RE = re.compile(r"^NSpid:\s*(.+)$", re.MULTILINE)


def parse_nspid(status_text: str) -> List[int]:
    """Parse the NSpid line of /proc/<pid>/status -> [host_pid, ..., innermost]."""
    match = _NSPID_RE.search(status_text or "")
    if not match:
        return []
    return [int(tok) for tok in match.group(1).split() if tok.strip().isdigit()]


def read_nspid(host_pid: int, proc_root: str = "/proc") -> List[int]:
    path = os.path.join(proc_root, str(host_pid), "status")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return parse_nspid(fh.read())
    except OSError:
        return []


@dataclass
class PidResolution:
    host_pid: Optional[int]
    container_pid: Optional[int]
    method: str
    candidates: List[int]
    confirmed_by_mps_client_list: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {"host_pid": self.host_pid, "container_pid": self.container_pid,
                "method": self.method, "candidates": self.candidates,
                "confirmed_by_mps_client_list": self.confirmed_by_mps_client_list}


def resolve_worker_host_pid(container_host_pids: Sequence[int],
                            worker_container_pid: Optional[int],
                            mps_client_pids: Sequence[int] = (),
                            proc_root: str = "/proc") -> PidResolution:
    """Map the in-container CUDA worker PID to its host PID.

    Preference order:
      1. NSpid match (strongest: kernel-provided namespace mapping)
      2. intersection with the MPS client list
    A single container PID is never assumed to be PID 1.
    """
    host_pids = [int(p) for p in container_host_pids]
    mps_set = set(int(p) for p in mps_client_pids)

    if worker_container_pid is not None:
        for host_pid in host_pids:
            ns = read_nspid(host_pid, proc_root=proc_root)
            if ns and ns[-1] == int(worker_container_pid):
                return PidResolution(host_pid=host_pid, container_pid=int(worker_container_pid),
                                     method="nspid", candidates=host_pids,
                                     confirmed_by_mps_client_list=host_pid in mps_set)

    overlap = [p for p in host_pids if p in mps_set]
    if len(overlap) == 1:
        return PidResolution(host_pid=overlap[0], container_pid=worker_container_pid,
                             method="mps_client_list", candidates=host_pids,
                             confirmed_by_mps_client_list=True)

    return PidResolution(host_pid=None, container_pid=worker_container_pid,
                         method="unresolved", candidates=host_pids,
                         confirmed_by_mps_client_list=False)
