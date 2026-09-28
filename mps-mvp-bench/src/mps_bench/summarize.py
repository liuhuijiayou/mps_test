"""Build summary rows, MVP status and findings from a results directory.

Status vocabulary (never collapsed into a green tick):
  PASS          expectation met with evidence
  FAIL          expectation not met, or evidence contradicts it
  SKIP          capability unsupported on this machine (e.g. static partitioning)
  INCONCLUSIVE  ran, but evidence insufficient (e.g. MPS attachment unproven,
                SM active unavailable while required)
  NOT_RUN       not executed in this run
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .reporting.stats import latency_delta, throughput_delta

MVP_ITEMS = [
    {"key": "hh_mem_50", "mvp_item": "高优半卡&高优半卡：显存各 50% 上限",
     "cases": ["MEM-HH-50"]},
    {"key": "hh_sm_70", "mvp_item": "高优半卡&高优半卡：SM 不隔离，上限各 70%",
     "cases": ["HH70"]},
    {"key": "hh_sm_50", "mvp_item": "高优半卡&高优半卡：SM 不隔离，上限各 50%",
     "cases": ["HH50"]},
    {"key": "hh_sm_static", "mvp_item": "高优半卡&高优半卡：SM 隔离，各 50%",
     "cases": ["HH-S"]},
    {"key": "hl_mem_50", "mvp_item": "高优整卡&低优半卡：低优 50% 显存上限",
     "cases": ["MEM-HL-50"]},
    {"key": "hl_priority", "mvp_item": "高优整卡&低优半卡：调度优先级",
     "cases": ["HL-P"]},
    {"key": "hl_reserve", "mvp_item": "高优整卡&低优半卡：预留空间（低优执行资源上限）",
     "cases": ["HL-C", "HL-PC"]},
]


def _read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _metric_means(csv_path: str) -> Dict[str, Optional[float]]:
    if not os.path.exists(csv_path):
        return {}
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    max_used: Optional[float] = None
    min_free: Optional[float] = None
    with open(csv_path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in ("sm_active", "gpu_util_device_busy_pct", "dram_active"):
                raw = row.get(key)
                if raw in (None, "", "None"):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
            for key, agg in (("memory_used_mib", "max"), ("memory_free_mib", "min")):
                raw = row.get(key)
                if raw in (None, "", "None"):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if agg == "max":
                    max_used = value if max_used is None else max(max_used, value)
                else:
                    min_free = value if min_free is None else min(min_free, value)
    out: Dict[str, Optional[float]] = {
        key: (sums[key] / counts[key]) if counts.get(key) else None
        for key in ("sm_active", "gpu_util_device_busy_pct", "dram_active")}
    out["memory_used_mib_max"] = max_used
    out["memory_free_mib_min"] = min_free
    return out


def _baseline_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Map baseline case -> worker slot -> row, for B0 and B2."""
    index: Dict[str, Dict[str, Dict[str, Any]]] = {"B0": {}, "B2": {}}
    for row in rows:
        case = str(row.get("case_id") or "")
        base = case.split("[")[0]
        if base in index:
            index[base][str(row.get("worker_id"))] = row
    return index


def build_summary(run_dir: str) -> Dict[str, Any]:
    case_results = _read_json(os.path.join(run_dir, "case_results.json"), []) or []
    manifest = _read_json(os.path.join(run_dir, "manifest.json"), {}) or {}
    capability = _read_json(os.path.join(run_dir, "capability.json"), {}) or {}
    environment = _read_json(os.path.join(run_dir, "environment.json"), {}) or {}
    telemetry_status = _read_json(os.path.join(run_dir, "telemetry_status.json"), {}) or {}
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    metric_means = _metric_means(os.path.join(run_dir, "gpu_metrics.csv"))
    memory_summary = _read_json(os.path.join(run_dir, "memory_results.json"), []) or []
    fault_summary = _read_json(os.path.join(run_dir, "fault_results.json"), []) or []
    findings = _read_json(os.path.join(run_dir, "findings.json"),
                          {"positive": [], "negative": [], "not_reproduced": [],
                           "sweeps": []}) or {}

    rows: List[Dict[str, Any]] = []
    for case in case_results:
        case_id = case.get("case")
        for rnd in case.get("rounds", []):
            for slot, st in (rnd.get("stats") or {}).items():
                evidence = (rnd.get("mps_evidence") or {}).get(f"client_{slot}", {})
                status, reason = _row_status(rnd, evidence, telemetry_status)
                rows.append({
                    "run_id": manifest.get("run_id"), "case_id": case_id,
                    "round_index": rnd.get("round"), "worker_id": slot,
                    "role": st.get("role"), "unit": st.get("unit"),
                    "window_s": st.get("window_s"), "cohort_size": st.get("cohort_size"),
                    "success": st.get("success"), "errors": st.get("errors"),
                    "timeouts": st.get("timeouts"), "rejected": st.get("rejected"),
                    "not_sent": st.get("not_sent"), "incomplete": st.get("incomplete"),
                    "output_errors": st.get("output_errors"),
                    "throughput_req_s": st.get("throughput_req_s"),
                    "throughput_units_s": st.get("throughput_units_s"),
                    "slo_throughput_req_s": st.get("slo_throughput_req_s"),
                    "success_rate": _ratio(st.get("success"), st.get("cohort_size")),
                    "error_rate": _ratio((st.get("errors") or 0) + (st.get("output_errors") or 0),
                                         st.get("cohort_size")),
                    "timeout_rate": _ratio(st.get("timeouts"), st.get("cohort_size")),
                    "reject_rate": _ratio(st.get("rejected"), st.get("cohort_size")),
                    "latency_p50_ms": (st.get("latency") or {}).get("p50"),
                    "latency_p95_ms": (st.get("latency") or {}).get("p95"),
                    "latency_p99_ms": (st.get("latency") or {}).get("p99"),
                    "latency_p999_ms": (st.get("latency") or {}).get("p999"),
                    "latency_p999_status": (st.get("latency") or {}).get("p999_status"),
                    "latency_samples": (st.get("latency") or {}).get("n"),
                    "gpu_ms_p50": (st.get("gpu_ms") or {}).get("p50"),
                    "gpu_ms_p99": (st.get("gpu_ms") or {}).get("p99"),
                    "queue_p99_ms": (st.get("queue_s") or {}).get("p99"),
                    "send_delay_p99_ms": (st.get("send_delay_s") or {}).get("p99"),
                    "mps_active_thread_pct": (evidence.get("worker_runtime") or {}).get(
                        "mps_active_thread_percentage"),
                    "mps_priority": (evidence.get("worker_runtime") or {}).get(
                        "mps_client_priority"),
                    "mps_mem_limit_bytes": (evidence.get("worker_runtime") or {}).get(
                        "mps_pinned_device_mem_limit"),
                    "static_partition": (evidence.get("worker_runtime") or {}).get(
                        "static_partition"),
                    "sm_active_mean": metric_means.get("sm_active"),
                    "sm_active_available": telemetry_status.get("sm_active_available"),
                    "gpu_util_mean": metric_means.get("gpu_util_device_busy_pct"),
                    "dram_active_mean": metric_means.get("dram_active"),
                    "memory_used_mib_max": metric_means.get("memory_used_mib_max"),
                    "memory_free_mib_min": metric_means.get("memory_free_mib_min"),
                    "status": status, "status_reason": reason,
                    "synthetic": True,
                })

    baselines = _baseline_index(rows)
    for row in rows:
        for ref in ("B0", "B2"):
            base_row = baselines.get(ref, {}).get(str(row["worker_id"]))
            row[f"throughput_delta_vs_{ref}"] = throughput_delta(
                row.get("throughput_units_s"),
                (base_row or {}).get("throughput_units_s"))
            row[f"latency_delta_vs_{ref}"] = latency_delta(
                row.get("latency_p99_ms"), (base_row or {}).get("latency_p99_ms"))

    executed = {str(r.get("case_id") or "").split("[")[0] for r in rows}
    mvp_status = _mvp_status(rows, executed, capability, memory_summary, telemetry_status)
    not_verified = _not_verified(capability, telemetry_status, executed)

    return {"run_id": manifest.get("run_id") or os.path.basename(run_dir.rstrip("/")),
            "summary_rows": rows, "mvp_status": mvp_status, "events": events,
            "capability": capability, "environment": environment,
            "telemetry_status": telemetry_status, "findings": findings,
            "fault_summary": fault_summary, "memory_summary": memory_summary,
            "not_verified": not_verified}


def _ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if not denominator:
        return None
    return (numerator or 0) / denominator


def _row_status(rnd: Dict[str, Any], evidence: Dict[str, Any],
                telemetry_status: Dict[str, Any]) -> "tuple[str, str]":
    if rnd.get("notes"):
        return "INCONCLUSIVE", "; ".join(rnd["notes"])
    if evidence.get("mps_attachment_proven") is False:
        return "INCONCLUSIVE", "未能证明业务已接入 MPS"
    if not rnd.get("healthy", True):
        return "FAIL", "轮次结束后健康检查未通过"
    if telemetry_status and not telemetry_status.get("sm_active_available"):
        return "INCONCLUSIVE", ("SM active 不可用："
                                + str(telemetry_status.get("dcgm_reason") or ""))
    return "PASS", ""


def _mvp_status(rows: Sequence[Dict[str, Any]], executed: set,
                capability: Dict[str, Any], memory_summary: Sequence[Dict[str, Any]],
                telemetry_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    caps = {c["name"]: c for c in capability.get("capabilities", [])}
    static_supported = caps.get("mps_static_partitioning", {}).get("status") == "supported"
    out: List[Dict[str, Any]] = []
    for item in MVP_ITEMS:
        matching = [r for r in rows
                    if str(r.get("case_id") or "").split("[")[0] in item["cases"]]
        if item["key"] == "hh_sm_static" and not static_supported and caps:
            # Only report SKIP when the capability was actually probed and came
            # back unsupported. With no capability evidence at all this is
            # NOT_RUN -- "we never looked" is not "the driver cannot do it".
            out.append({**item, "case_id": ",".join(item["cases"]), "status": "SKIP",
                        "config_summary": "静态 SM 分区",
                        "reason": "本机 MPS 控制接口不提供静态 SM 分区能力；"
                                  "禁止用 ACTIVE_THREAD_PERCENTAGE / 亲和性 / MIG 冒充",
                        "evidence": json.dumps(
                            caps.get("mps_static_partitioning", {}).get("evidence"),
                            ensure_ascii=False)[:300]})
            continue
        if not matching:
            out.append({**item, "case_id": ",".join(item["cases"]), "status": "NOT_RUN",
                        "config_summary": "-", "reason": "本次未执行该用例", "evidence": "-"})
            continue
        if item["key"].endswith("mem_50"):
            mem_rows = [m for m in memory_summary
                        if str(m.get("case_id") or "").split("[")[0] in item["cases"]]
            status = "INCONCLUSIVE" if not mem_rows else (
                "PASS" if all(m.get("verdict") == "PASS" for m in mem_rows) else "FAIL")
            reason = "" if status == "PASS" else "; ".join(
                str(m.get("verdict_reason") or "") for m in mem_rows) or "缺少显存 probe 证据"
            out.append({**item, "case_id": ",".join(sorted({str(m.get("case_id"))
                                                            for m in mem_rows})) or
                                 ",".join(item["cases"]),
                        "status": status,
                        "config_summary": "; ".join(f"{m.get('client')}:quota="
                                                    f"{m.get('quota_bytes')}B"
                                                    for m in mem_rows) or "-",
                        "reason": reason,
                        "evidence": "memory_results.json"})
            continue
        statuses = {r["status"] for r in matching}
        status = ("FAIL" if "FAIL" in statuses else
                  "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "PASS")
        reason = "; ".join(sorted({r["status_reason"] for r in matching if r["status_reason"]}))
        config_summary = "; ".join(sorted({
            f"{r['worker_id']}:ATP={r.get('mps_active_thread_pct')},"
            f"prio={r.get('mps_priority')}" for r in matching}))
        out.append({**item, "case_id": ",".join(sorted({str(r["case_id"]) for r in matching})),
                    "status": status, "config_summary": config_summary,
                    "reason": reason, "evidence": "summary.csv / requests.jsonl"})
    return out


def _not_verified(capability: Dict[str, Any], telemetry_status: Dict[str, Any],
                  executed: set) -> List[str]:
    out: List[str] = []
    for cap in capability.get("capabilities", []):
        if cap.get("status") == "unknown":
            out.append(f"{cap['name']}: {cap.get('detail')}"
                       + (f"（需 {cap.get('verify_by')} 验证）" if cap.get("verify_by") else ""))
        elif cap.get("status") == "unsupported":
            out.append(f"{cap['name']}: 不支持 -- {cap.get('detail')}")
    if telemetry_status and not telemetry_status.get("sm_active_available"):
        out.append("DCGM_FI_PROF_SM_ACTIVE 不可用：SM 可观测验收未完成")
    if not executed:
        out.append("本次未执行任何用例：所有 MVP 项状态为 NOT_RUN")
    return out
