"""Idempotent cleanup: clients first, then only our own MPS, logs always saved.

Order matters: we stop clients (cooperative drain, then graceful stop) before
touching MPS, and we persist logs *before* removing anything. Every step is
safe to call twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .docker import DockerClient
from .mps import MpsController


@dataclass
class CleanupReport:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def record(self, name: str, **info: Any) -> None:
        self.steps.append({"step": name, **info})

    def as_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "removed": self.removed, "errors": self.errors,
                "notes": self.notes}


def cleanup(docker: Optional[DockerClient],
            containers: List[str],
            mps: Optional[MpsController],
            save_logs: Optional[Callable[[str, str], None]] = None,
            stop_timeout_s: int = 30,
            remove_containers: bool = True) -> CleanupReport:
    """Idempotent teardown. Never touches containers/MPS we do not own."""
    report = CleanupReport()

    for name in containers:
        if docker is None:
            report.record("skip_container", container=name, reason="docker 客户端不可用")
            report.notes.append(f"{name}: docker 客户端不可用，跳过")
            continue
        try:
            docker.assert_owned(name)
        except Exception as exc:
            # A container we do not own is left alone. Not an error for us, and
            # certainly never escalated to a force-kill.
            report.record("container_not_owned", container=name, error=str(exc))
            report.notes.append(f"{name}: 非本项目容器或已不存在，不做任何处理")
            continue
        # logs first: teardown must not lose evidence
        if save_logs is not None:
            try:
                save_logs(name, docker.logs(name))
                report.record("logs_saved", container=name)
            except Exception as exc:
                report.record("logs_failed", container=name, error=str(exc))
                report.errors.append(f"{name}: 保存日志失败 {exc}")
        res = docker.stop(name, timeout_s=stop_timeout_s)
        stop_ok = res.get("ok") if isinstance(res, dict) else bool(getattr(res, "ok", False))
        report.record("container_stopped", container=name, ok=stop_ok)
        if remove_containers:
            res = docker.remove(name)
            rm_ok = res.get("ok") if isinstance(res, dict) else bool(getattr(res, "ok", False))
            already = res.get("already_absent") if isinstance(res, dict) else False
            report.record("container_removed", container=name, ok=rm_ok,
                          already_absent=bool(already))
            if rm_ok and not already:
                report.removed.append(name)
            elif already:
                # Second cleanup run: absence is the desired end state, never an
                # error and never a false "removed" claim.
                report.notes.append(f"{name}: 已不存在（幂等清理）")
            elif not rm_ok:
                report.errors.append(f"{name}: 删除失败")

    if mps is not None:
        # Only our own instance; no global `quit`.
        if getattr(mps, "owns_daemon", False):
            result = mps.stop_daemon()
            report.record("mps_stop", **(result if isinstance(result, dict) else {}))
        else:
            report.record("mps_stop_skipped",
                          reason="本实验不拥有该 MPS 实例，不执行 quit")
            report.notes.append("MPS 实例未由本 run 启动，不停止（避免影响他人）")
        try:
            destroyed = mps.destroy_partitions()
            report.record("mps_partitions", **(destroyed if isinstance(destroyed, dict) else {}))
        except Exception as exc:
            report.record("mps_partitions_failed", error=str(exc))

    return report


def restore_compute_mode(gpu_query, gpu_uuid: Optional[str], original_mode: Optional[str],
                         changed: bool) -> Dict[str, Any]:
    """Restore compute mode only if this run changed it, and only for one UUID."""
    if not changed or not gpu_uuid or not original_mode:
        return {"restored": False, "reason": "本 run 未修改 compute mode"}
    mapping = {"Default": "DEFAULT", "Exclusive_Process": "EXCLUSIVE_PROCESS",
               "Prohibited": "PROHIBITED"}
    target = mapping.get(original_mode, original_mode.upper())
    ok = gpu_query.set_compute_mode(gpu_uuid, target)
    return {"restored": ok, "target_mode": target, "gpu_uuid": gpu_uuid}
