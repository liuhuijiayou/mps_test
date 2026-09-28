"""Per-run result directory layout + manifest.

results/<run_id>/
  environment.json  capability.json  effective_config.yaml  manifest.json
  requests.jsonl    gpu_metrics.csv  events.jsonl
  workload_logs/    mps_logs/        summary.csv            report.html
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import string
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def new_run_id(prefix: str = "") -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
    return f"{prefix}{stamp}-{suffix}"


def sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def source_commit(repo_dir: str) -> Dict[str, Any]:
    try:
        rev = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=15)
        if rev.returncode == 0:
            return {"commit": rev.stdout.strip(),
                    "dirty": bool(dirty.stdout.strip()),
                    "dirty_files": dirty.stdout.strip().splitlines()[:50]}
    except Exception:
        pass
    return {"commit": None, "dirty": None,
            "note": "无法获取 git commit（非 git 仓库或 git 不可用）"}


@dataclass
class RunDirectory:
    root: str
    run_id: str

    @property
    def path(self) -> str:
        return os.path.join(self.root, self.run_id)

    def create(self) -> "RunDirectory":
        for sub in ("", "workload_logs", "mps_logs", "raw"):
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        return self

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)

    def write_json(self, name: str, payload: Any) -> str:
        path = self.file(name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        return path

    def write_text(self, name: str, text: str) -> str:
        path = self.file(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def append_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        with open(self.file(name), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def copy_mps_logs(self, log_dir: str) -> List[str]:
        copied = []
        if not os.path.isdir(log_dir):
            return copied
        for name in os.listdir(log_dir):
            src = os.path.join(log_dir, name)
            if os.path.isfile(src):
                dst = os.path.join(self.path, "mps_logs", name)
                try:
                    shutil.copy2(src, dst)
                    copied.append(dst)
                except OSError:
                    pass
        return copied


@dataclass
class Manifest:
    run_id: str
    run_dir: str
    config_hash: str
    seed: int
    started_wall: float = field(default_factory=time.time)
    source: Dict[str, Any] = field(default_factory=dict)
    image: Dict[str, Any] = field(default_factory=dict)
    assets: Dict[str, Any] = field(default_factory=dict)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    files: Dict[str, Any] = field(default_factory=dict)
    finished_wall: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "run_dir": self.run_dir,
            "config_hash": self.config_hash, "seed": self.seed,
            "started_wall": self.started_wall, "finished_wall": self.finished_wall,
            "source": self.source, "image": self.image, "assets": self.assets,
            "executed_commands": self.commands, "cases": self.cases,
            "files": self.files, "notes": self.notes,
        }


def image_identity(docker_binary: str, image: str) -> Dict[str, Any]:
    """Record the digest/ID actually used; `latest` tags are not acceptable
    provenance."""
    try:
        res = subprocess.run([docker_binary, "image", "inspect", image, "--format",
                              "{{.Id}}|{{json .RepoDigests}}"],
                             capture_output=True, text=True, timeout=30)
        if res.returncode == 0:
            idpart, _, digests = res.stdout.strip().partition("|")
            return {"image": image, "id": idpart,
                    "repo_digests": json.loads(digests or "[]")}
        return {"image": image, "error": res.stderr.strip()[:500]}
    except Exception as exc:
        return {"image": image, "error": str(exc)[:500]}
