"""Memory quota专项 orchestration (native allocation probe driven).

Why a native probe: PyTorch's caching allocator makes it impossible to attribute
an OOM to the MPS client quota. `memory_probe` allocates with the driver API in
fixed steps, touches each block, and reports the exact CUDA error code, so we can
tell an expected quota OOM from an unexpected one.

Honest limits encoded here:
* 50% is computed from the *runtime-queried* physical total; raw bytes, the env
  var string and the unit conversion are all recorded.
* the probe never needs to fill exactly 50% of user buffers -- context and CUDA
  internal allocations consume part of the quota, and we say so.
* the non-MPS control only crosses the same ratio when physical headroom is
  proven; we never exhaust a whole card to produce an OOM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

EXPECTED_QUOTA_OOM = "expected_quota_oom"
UNEXPECTED_OOM = "unexpected_oom"
NO_OOM = "no_oom"
OTHER_ERROR = "other_error"


@dataclass
class ProbeOutcome:
    client: str
    quota_bytes: Optional[int]
    quota_env_value: Optional[str]
    requested_mib: int
    allocated_mib: int
    failed_request_mib: Optional[int]
    cuda_error: Optional[str]
    oom_class: Optional[str]
    released: bool
    recovered_compute: bool
    oom_reason: str = ""
    framework_allocated_mib: Optional[int] = None
    framework_reserved_mib: Optional[int] = None
    device_used_mib: Optional[int] = None
    device_free_mib: Optional[int] = None
    peer_qos: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["expected_quota_oom"] = self.oom_class == EXPECTED_QUOTA_OOM
        out["unexpected_oom"] = self.oom_class == UNEXPECTED_OOM
        return out


def classify_oom(allocated_bytes: int, quota_bytes: Optional[int],
                 cuda_error: Optional[str],
                 device_free_bytes: Optional[int],
                 tolerance_ratio: float = 0.15) -> "tuple[str, str]":
    """Separate an expected quota OOM from an unexpected one.

    A quota OOM is expected when the client hit roughly its own limit while the
    *device* still had free memory. If the device itself was nearly full, the OOM
    is a physical one and must not be credited to the quota mechanism.

    An allocation OOM is NOT automatically a whole-card fatal fault; that
    conclusion requires separate evidence.
    """
    if cuda_error is None:
        return NO_OOM, "未发生 CUDA 错误"
    if "OUT_OF_MEMORY" not in cuda_error.upper():
        # An illegal address / launch failure is a different failure mode and must
        # never be credited to (or blamed on) the memory quota.
        return OTHER_ERROR, f"非 OOM 类 CUDA 错误: {cuda_error}"
    if quota_bytes is None:
        return UNEXPECTED_OOM, "该 client 未配置 MPS 显存配额，OOM 不能记为配额生效"
    near_quota = allocated_bytes >= quota_bytes * (1.0 - tolerance_ratio)
    device_has_room = (device_free_bytes is not None
                       and device_free_bytes > quota_bytes * tolerance_ratio)
    if not near_quota:
        return UNEXPECTED_OOM, (f"远未达到配额即 OOM（已分配 {allocated_bytes}B / "
                                f"配额 {quota_bytes}B）")
    if device_free_bytes is None:
        # cannot prove device headroom -> stay honest
        return UNEXPECTED_OOM, "无法证明设备仍有物理余量，不能判定为配额 OOM"
    if not device_has_room:
        return UNEXPECTED_OOM, "设备物理显存本身已接近耗尽，属物理 OOM 而非配额生效"
    return EXPECTED_QUOTA_OOM, "达到本 client 配额边界且设备仍有物理余量：预期的配额 OOM"


def parse_probe_output(stdout: str) -> Optional[Dict[str, Any]]:
    """memory_probe prints one JSON object on its last non-empty line.

    Returns None when no JSON could be parsed -- an unparsable probe is missing
    evidence, not a zero result.
    """
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def probe_argv(step_mib: int, limit_mib: Optional[int], touch: bool,
               binary: str = "/opt/mps-bench/bin/memory_probe",
               hold_s: float = 2.0) -> List[str]:
    """argv list only -- never a shell string."""
    argv = [binary, "--step-mib", str(int(step_mib)), "--hold-s", str(hold_s), "--json"]
    if limit_mib is not None:
        argv += ["--max-mib", str(int(limit_mib))]
    if touch:
        argv.append("--touch")
    return argv


def build_outcome(client: str, quota_bytes: Optional[int], quota_env_value: Optional[str],
                  probe_json: Dict[str, Any],
                  device_used_mib: Optional[int] = None,
                  device_free_mib: Optional[int] = None,
                  peer_qos: Optional[Dict[str, Any]] = None,
                  framework: Optional[Dict[str, Any]] = None) -> ProbeOutcome:
    probe_json = probe_json or {}
    allocated_mib = int(probe_json.get("allocated_mib") or 0)
    cuda_error = probe_json.get("cuda_error")
    free_bytes = None if device_free_mib is None else device_free_mib * 1024 * 1024
    oom_class, oom_reason = classify_oom(allocated_mib * 1024 * 1024, quota_bytes,
                                         cuda_error, free_bytes)
    return ProbeOutcome(
        client=client, quota_bytes=quota_bytes, quota_env_value=quota_env_value,
        requested_mib=int(probe_json.get("requested_mib") or 0),
        allocated_mib=allocated_mib,
        failed_request_mib=probe_json.get("failed_request_mib"),
        cuda_error=cuda_error, oom_class=oom_class, oom_reason=oom_reason,
        released=bool(probe_json.get("released")),
        recovered_compute=bool(probe_json.get("recovered_compute")),
        framework_allocated_mib=(framework or {}).get("allocated_mib"),
        framework_reserved_mib=(framework or {}).get("reserved_mib"),
        device_used_mib=device_used_mib, device_free_mib=device_free_mib,
        peer_qos=peer_qos or {}, raw=probe_json)


def quota_report_row(client: str, quota_bytes: Optional[int], env_value: Optional[str],
                     total_bytes: Optional[int],
                     outcome: Any, case_id: Optional[str] = None) -> Dict[str, Any]:
    """One row of the memory-quota table.

    Raw bytes, the literal env string and the unit conversion are all kept so a
    reader can re-derive the 50% themselves instead of trusting a rounded number.
    """
    data = outcome.as_dict() if hasattr(outcome, "as_dict") else dict(outcome or {})
    row: Dict[str, Any] = {"case_id": case_id, "client": client,
                           "quota_bytes": quota_bytes, "quota_env": env_value,
                           "device_total_bytes": total_bytes}
    row["quota_mib"] = None if quota_bytes is None else quota_bytes // (1024 * 1024)
    row["quota_fraction_of_total"] = (None if not total_bytes or quota_bytes is None
                                      else quota_bytes / total_bytes)
    row.update(data)
    row["verdict"] = data.get("classification") or data.get("oom_class")
    row["verdict_reason"] = data.get("oom_reason", "")
    return row


def verify_expectations(case_expect: Dict[str, Any],
                        outcomes: Any) -> Dict[str, Any]:
    """Case-declared expectations -> PASS / FAIL / INCONCLUSIVE.

    A case that expected a quota OOM and did not get one is a FAIL, not a silent
    pass; a case whose evidence is incomplete is INCONCLUSIVE.

    `outcomes` may be a list of ProbeOutcome or a {client: dict} mapping.
    """
    if isinstance(outcomes, dict):
        by_client: Dict[str, Any] = dict(outcomes)
    else:
        by_client = {o.client: o for o in (outcomes or [])}
    checks: List[Dict[str, Any]] = []

    def field_of(entry: Any, name: str) -> Any:
        if entry is None:
            return None
        if isinstance(entry, dict):
            return entry.get(name)
        return getattr(entry, name, None)

    def record(name: str, ok: Optional[bool], detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    for client, expect_oom in (("a", case_expect.get("client_a_quota_oom")),
                              ("b", case_expect.get("client_b_quota_oom"))):
        if expect_oom is None:
            continue
        entry = by_client.get(client)
        if entry is None:
            record(f"client_{client}_quota_oom", None, "缺少该 client 的 probe 结果")
            continue
        observed = field_of(entry, "classification") or field_of(entry, "oom_class")
        got = observed == EXPECTED_QUOTA_OOM
        record(f"client_{client}_quota_oom", got is bool(expect_oom),
               f"expected={expect_oom} observed_oom_class={observed} "
               f"allocated={field_of(entry, 'allocated_mib')}MiB "
               f"error={field_of(entry, 'cuda_error')}")

    if case_expect.get("high_exceeds_half") is not None:
        entry = by_client.get("a")
        if entry is None or field_of(entry, "quota_bytes") is not None:
            record("high_exceeds_half", None,
                   "高优被施加了配额或缺少结果，无法验证其可超过 50%")
        else:
            half_mib = (case_expect.get("half_mib") or 0)
            allocated = field_of(entry, "allocated_mib") or 0
            got = allocated > half_mib if half_mib else None
            record("high_exceeds_half", got,
                   f"高优分配 {allocated}MiB vs 50%={half_mib}MiB"
                   "（需物理空闲足够；不保证低优驻留时必然可申请整卡）")

    if case_expect.get("peer_correctness") is not None:
        peers = [(c, e) for c, e in by_client.items()
                 if field_of(e, "peer_qos") or field_of(e, "peer_correctness") is not None]
        if not peers:
            record("peer_correctness", None, "未采集到对端 QoS/正确性证据")
        else:
            def peer_ok(entry: Any) -> bool:
                explicit = field_of(entry, "peer_correctness")
                if explicit is not None:
                    return bool(explicit)
                return (field_of(entry, "peer_qos") or {}).get("checksum_ok") is True
            ok = all(peer_ok(e) for _, e in peers)
            record("peer_correctness", ok,
                   json.dumps([str(field_of(e, "peer_qos")) for _, e in peers],
                              ensure_ascii=False))

    if not checks:
        return {"verdict": "INCONCLUSIVE", "status": "INCONCLUSIVE", "checks": [],
                "reason": "用例未声明期望"}
    if any(c["ok"] is None for c in checks):
        verdict = "INCONCLUSIVE"
        reason = "部分检查证据不足"
    elif all(c["ok"] for c in checks):
        verdict, reason = "PASS", ""
    else:
        verdict = "FAIL"
        reason = "; ".join(c["detail"] for c in checks if not c["ok"])
    return {"verdict": verdict, "status": verdict, "checks": checks, "reason": reason}
