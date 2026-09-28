"""Subprocess helper.

Hard rules enforced here:
* argv lists only -- never `shell=True`, never `eval`, never string concatenation
  of user-supplied runtime/device parameters.
* every command is recorded so manifest.json can list what actually ran.
* nothing that looks like a credential is echoed into the record.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_SECRET_HINTS = ("token", "password", "passwd", "secret", "apikey", "api_key", "credential")


def redact_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not env:
        return {}
    out = {}
    for key, value in env.items():
        out[key] = "<redacted>" if any(h in key.lower() for h in _SECRET_HINTS) else value
    return out


@dataclass
class CommandResult:
    argv: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_record(self) -> Dict[str, object]:
        return {
            "argv": list(self.argv),
            "cmdline": " ".join(shlex.quote(a) for a in self.argv),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 4),
            "stdout_tail": self.stdout[-4000:],
            "stderr_tail": self.stderr[-4000:],
        }


@dataclass
class CommandLog:
    """Append-only audit trail of every executed command."""
    records: List[Dict[str, object]] = field(default_factory=list)

    def add(self, result: CommandResult, env: Optional[Dict[str, str]] = None,
            note: str = "") -> CommandResult:
        rec = result.as_record()
        rec["env"] = redact_env(env)
        rec["note"] = note
        rec["wall_ts"] = time.time()
        self.records.append(rec)
        return result


def run(argv: Sequence[str],
        timeout_s: Optional[float] = 60,
        env: Optional[Dict[str, str]] = None,
        inherit_env: bool = True,
        cwd: Optional[str] = None,
        log: Optional[CommandLog] = None,
        note: str = "") -> CommandResult:
    argv = [str(a) for a in argv]
    if not argv:
        raise ValueError("argv 不能为空")
    full_env: Optional[Dict[str, str]] = None
    if env is not None:
        full_env = dict(os.environ) if inherit_env else {}
        full_env.update(env)

    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              env=full_env, cwd=cwd, shell=False)
        rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = -9
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    except FileNotFoundError as exc:
        rc, out, err = 127, "", f"{exc}"

    result = CommandResult(argv=argv, returncode=rc, stdout=out, stderr=err,
                           duration_s=time.monotonic() - start, timed_out=timed_out)
    if log is not None:
        log.add(result, env=env, note=note)
    return result


def which(binary: str) -> Optional[str]:
    from shutil import which as _which
    return _which(binary)
