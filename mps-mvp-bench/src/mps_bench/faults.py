"""Fault injection, safe-exit verification and layered recovery (R1/R2/R3).

Two independent dimensions are recorded for every injection:
  injection_status -- did the injection actually take effect?
  peer_outcome     -- normal / degraded / cuda_error / exited / hang / output_error

A successfully injected *expected* error is not a test-infrastructure failure,
and "peer unaffected" is not an injection failure.

We never claim SIGKILL can be caught, and we never assert that SIGKILL will (or
will not) affect the peer -- that is exactly what the experiment measures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# injection_status
INJ_APPLIED = "applied"
INJ_NOT_APPLIED = "not_applied"
INJ_UNKNOWN = "unknown"

# peer_outcome
PEER_NORMAL = "normal"
PEER_DEGRADED = "degraded"
PEER_CUDA_ERROR = "cuda_error"
PEER_EXITED = "exited"
PEER_HANG = "hang"
PEER_OUTPUT_ERROR = "output_error"

FAULTS: Dict[str, Dict[str, Any]] = {
    "F1": {"name": "低优配额 OOM", "disruptive": False,
           "how": "原生显存申请越过 MPS client 配额；记录 CUDA 错误码与对端 QoS/正确性"},
    "F2": {"name": "非致命 CUDA API 参数错误", "disruptive": False,
           "how": "在 CPU/API 校验阶段触发可识别错误，确认实际错误类型与对端影响"},
    "F3": {"name": "SIGTERM/SIGINT 安全退出", "disruptive": False,
           "how": "Fence -> Drain -> Exit；记录有无在途 GPU 工作及对端影响"},
    "F4": {"name": "terminate_client 后 SIGKILL", "disruptive": True,
           "how": "控制端确认 CUDA contexts 终止成功后才 kill OS 进程"},
    "F5": {"name": "空闲状态直接 SIGKILL", "disruptive": True,
           "how": "确认已停止提交且已 drain 的对照"},
    "F6": {"name": "GPU work 在途时直接 SIGKILL", "disruptive": True,
           "how": "在有界 kernel 或持续业务执行中注入；需设备进度标记证明已提交未完成"},
    "F7": {"name": "Device illegal memory access", "disruptive": True,
           "how": "独立 CUDA helper 触发真实设备错误，不以 CPU 异常冒充"},
    "F8": {"name": "Device assert", "disruptive": True,
           "how": "独立真实 device assert；构建时确认断言未被禁用(未使用 -DNDEBUG)"},
    "F9": {"name": "有界长 kernel", "disruptive": True,
           "how": "延迟/超时干扰而非硬件故障；kernel 与实验均设时限"},
}


@dataclass
class InjectionRecord:
    fault_id: str
    attempt: int
    injection_status: str
    # The two recording dimensions are kept adjacent and positional: whether the
    # injection actually took effect, and what happened to the peer. Evidence
    # dicts follow, so a positional call can never silently land in evidence.
    peer_outcome: str = PEER_NORMAL
    injection_evidence: Dict[str, Any] = field(default_factory=dict)
    peer_evidence: Dict[str, Any] = field(default_factory=dict)
    detect_s: Optional[float] = None
    first_success_s: Optional[float] = None
    stable_recovery_s: Optional[float] = None
    failed_requests: int = 0
    timeout_requests: int = 0
    recovery_level: Optional[str] = None
    recovery_detail: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def propagated(self) -> bool:
        # "unknown" peer state is not evidence of propagation, and a failed
        # injection is not evidence of isolation either -- both dimensions are
        # reported separately.
        return self.peer_outcome not in (PEER_NORMAL, "unaffected", "unknown")

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["propagated"] = self.propagated
        return out


def summarize(fault_id: str, records: List[InjectionRecord]) -> Dict[str, Any]:
    """Aggregate; 'no propagation observed' is reported as 0/n, never as
    '完全隔离'."""
    n = len(records)
    prop = sum(1 for r in records if r.propagated)
    applied = sum(1 for r in records if r.injection_status in (INJ_APPLIED, "injected"))
    not_applied = sum(1 for r in records
                      if r.injection_status in (INJ_NOT_APPLIED, "not_injected"))
    detects = [r.detect_s for r in records if r.detect_s is not None]
    firsts = [r.first_success_s for r in records if r.first_success_s is not None]
    stables = [r.stable_recovery_s for r in records if r.stable_recovery_s is not None]
    from .reporting.stats import t_ci95_block
    return {
        "fault_id": fault_id, "name": FAULTS.get(fault_id, {}).get("name"),
        "injections": n, "injected": applied, "not_injected": not_applied,
        "injections_applied": applied, "propagations": prop, "propagated": prop,
        "propagation_statement": (
            f"{prop}/{n} 次观察到传播"
            + ("（不等于完全隔离）" if prop == 0 else "")
            + (f"；其中 {not_applied} 次注入未生效，该部分不构成隔离证据"
               if not_applied else "")),
        "injection_status": (INJ_APPLIED if applied == n and n else
                             INJ_NOT_APPLIED if applied == 0 else "partial"),
        "peer_outcome": _dominant_peer_outcome(records),
        "failed_requests": sum(r.failed_requests for r in records),
        "timeout_requests": sum(r.timeout_requests for r in records),
        "detect_s": _avg(detects), "detect_ci95": t_ci95_block(detects),
        "first_success_s": _avg(firsts), "first_success_ci95": t_ci95_block(firsts),
        "stable_recovery_s": _avg(stables), "stable_recovery_ci95": t_ci95_block(stables),
        "recovery_level": ",".join(sorted({r.recovery_level for r in records
                                           if r.recovery_level})) or None,
        "records": [r.as_dict() for r in records],
    }


def _avg(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _dominant_peer_outcome(records: List[InjectionRecord]) -> str:
    severity = [PEER_HANG, PEER_EXITED, PEER_CUDA_ERROR, PEER_OUTPUT_ERROR,
                PEER_DEGRADED, PEER_NORMAL]
    seen = {r.peer_outcome for r in records}
    for level in severity:
        if level in seen:
            return level
    return PEER_NORMAL


# --------------------------------------------------------------------------- #
# safe exit protocol (F3) -- cooperative
# --------------------------------------------------------------------------- #

@dataclass
class SafeExitResult:
    signal_sent: str
    drained: bool
    drain_s: Optional[float]
    inflight_at_signal: Optional[int]
    terminate_client_used: bool
    terminate_result: Optional[Dict[str, Any]]
    exit_status: str  # safe_exit / drain_timeout_escalated / failed
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def exited_cleanly(self) -> bool:
        return self.drained and self.exit_status == "safe_exit"

    @property
    def escalated_to_terminate_client(self) -> bool:
        return self.terminate_client_used and not self.drained

    @property
    def inflight_at_exit(self) -> Optional[int]:
        return self.inflight_at_signal

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["exited_cleanly"] = self.exited_cleanly
        out["escalated_to_terminate_client"] = self.escalated_to_terminate_client
        out["inflight_at_exit"] = self.inflight_at_exit
        return out


def safe_exit(send_signal: Callable[[str], None],
              read_status: Callable[[], Optional[Dict[str, Any]]],
              drain_timeout_s: float,
              terminate_client: Optional[Callable[[], Dict[str, Any]]] = None,
              signal_name: str = "SIGTERM",
              poll_interval_s: float = 0.25,
              watchdog_s: Optional[float] = None) -> SafeExitResult:
    """Cooperative safe exit.

    The worker's handler only flips a flag; its control flow does fence -> drain
    -> exit. If drain does not finish in time we first ask MPS to
    `terminate_client` and check the CUDA status, and only then decide what to do
    with the OS process. This verifies *this project's* cooperative protocol; it
    is not a generic signal hijack for arbitrary third-party workloads.

    `watchdog_s` bounds the whole procedure so the external controller is never
    dragged down by a blocked CUDA call inside the worker.
    """
    start = time.monotonic()
    send_signal(signal_name)
    inflight_at_signal = None
    deadline = start + drain_timeout_s
    hard_deadline = start + (watchdog_s if watchdog_s else drain_timeout_s * 2 + 30)
    drained = False
    last_status: Optional[Dict[str, Any]] = None

    while time.monotonic() < deadline and time.monotonic() < hard_deadline:
        status = read_status()
        if status:
            last_status = status
            if inflight_at_signal is None:
                inflight_at_signal = status.get("inflight_at_signal", status.get("inflight"))
            state = status.get("state")
            if status.get("exit") == "safe_exit" or state in ("exiting", "exited"):
                # "drained" means nothing was left in flight; default to the
                # reported inflight count rather than optimistically assuming 0.
                drained = bool(status.get("drained", (status.get("inflight") or 0) == 0))
                inflight_at_signal = status.get("inflight", inflight_at_signal)
                break
        time.sleep(poll_interval_s)

    drain_s = time.monotonic() - start
    if drained:
        return SafeExitResult(signal_sent=signal_name, drained=True, drain_s=drain_s,
                              inflight_at_signal=inflight_at_signal,
                              terminate_client_used=False, terminate_result=None,
                              exit_status="safe_exit",
                              evidence={"worker_status": last_status})

    term: Optional[Dict[str, Any]] = None
    if terminate_client is not None:
        term = terminate_client()
    status = "drain_timeout_escalated" if (term and term.get("ok")) else "failed"
    return SafeExitResult(signal_sent=signal_name, drained=False, drain_s=drain_s,
                          inflight_at_signal=inflight_at_signal,
                          terminate_client_used=terminate_client is not None,
                          terminate_result=term, exit_status=status,
                          evidence={"worker_status": last_status,
                                    "note": "drain 超时：先请求 terminate_client 并确认返回的 "
                                            "CUDA 状态，再决定 OS 进程退出动作；"
                                            "超时/失败不得记为安全退出成功"})


# --------------------------------------------------------------------------- #
# recovery ladder
# --------------------------------------------------------------------------- #

@dataclass
class RecoveryStep:
    level: str
    attempted: bool
    succeeded: Optional[bool]
    detail: Dict[str, Any] = field(default_factory=dict)


def _as_result(value: Any) -> Dict[str, Any]:
    """Normalize a bool / dict callback result to {'ok': bool, ...}."""
    if isinstance(value, dict):
        return value
    return {"ok": bool(value)}


def _as_health(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"healthy": bool(value)}


def recover(levels: List[str],
            rebuild_client: Callable[[], Any],
            mps_healthy: Callable[[], Any],
            witness_healthy: Callable[[], Any],
            restart_mps: Callable[[], Any],
            diagnostics: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Escalate along the smallest fault domain first.

    R1: rebuild only the affected client -- requires MPS and witness still healthy.
    R2: clean up affected clients, FIRST observe whether MPS recovers from FAULT
        to ACTIVE on its own; only then restart this experiment's own MPS.
    R3: stop and emit human diagnostics. We never auto-reset the GPU and never
        treat every Xid as "must reset".
    """
    steps: List[RecoveryStep] = []

    if "R1" in levels:
        mps_state = _as_health(mps_healthy())
        witness_ok = bool(witness_healthy())
        if mps_state.get("healthy") and witness_ok:
            result = _as_result(rebuild_client())
            steps.append(RecoveryStep("R1", True, bool(result.get("ok")),
                                      {"mps_state": mps_state, "witness_ok": witness_ok,
                                       "rebuild": result}))
            if result.get("ok"):
                return {"recovered": True, "level": "R1",
                        "steps": [s.__dict__ for s in steps]}
        else:
            steps.append(RecoveryStep("R1", False, None,
                                      {"reason": "MPS 或 witness 不健康，R1 前提不成立",
                                       "mps_state": mps_state, "witness_ok": witness_ok}))

    if "R2" in levels:
        observed = _as_health(mps_healthy())
        if observed.get("state") == "FAULT":
            # give MPS a chance to return to ACTIVE by itself before restarting
            time.sleep(2.0)
            observed = _as_health(mps_healthy())
        if observed.get("healthy"):
            result = _as_result(rebuild_client())
            steps.append(RecoveryStep("R2", True, bool(result.get("ok")),
                                      {"mps_self_recovered": True, "observed": observed,
                                       "rebuild": result}))
            if result.get("ok"):
                return {"recovered": True, "level": "R2",
                        "mps_restart_needed": False,
                        "steps": [s.__dict__ for s in steps]}
        restart = _as_result(restart_mps())
        rebuilt = _as_result(rebuild_client()) if restart.get("ok") else {
            "ok": False, "reason": "MPS 重启失败"}
        steps.append(RecoveryStep("R2", True, bool(rebuilt.get("ok")),
                                  {"mps_self_recovered": False, "observed": observed,
                                   "restart_mps": restart, "rebuild": rebuilt,
                                   "note": "仅重启本实验拥有的 MPS，不做全机 kill"}))
        if rebuilt.get("ok"):
            return {"recovered": True, "level": "R2", "steps": [s.__dict__ for s in steps]}

    diag = diagnostics()
    steps.append(RecoveryStep("R3", True, False,
                             {"diagnostics": diag,
                              "note": "停止并输出人工诊断/维护建议，保留 Xid/NVML/MPS 日志；"
                                      "本项目不自动 GPU reset，也不把所有 Xid 一律判定为必须 reset"}))
    return {"recovered": False, "level": "R3", "steps": [s.__dict__ for s in steps],
            "diagnostics": diag}
