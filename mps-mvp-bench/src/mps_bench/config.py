"""Configuration schema, defaults, validation and override precedence.

Precedence (lowest -> highest):
    1. built-in DEFAULTS
    2. runtime.yaml (--config, may be repeated; later files win)
    3. profile file (--profile -> configs/profiles/<name>.yaml)
    4. case file (cases/**.yaml)
    5. --set key=value CLI overrides

Unknown keys are rejected. Every leaf is declared in SCHEMA with a type, a unit
and (where meaningful) a range, so `docs/CONFIG.zh-CN.md` can be checked against
the code and so typos never silently change an experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # PyYAML is the single host dependency.
    import yaml
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise SystemExit("PyYAML is required for the host orchestrator: pip install PyYAML") from exc


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types, out-of-range or inconsistent values."""


@dataclass(frozen=True)
class Field:
    type: Any
    default: Any
    unit: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    stage: str = "run"  # when the value takes effect
    doc: str = ""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def _f(type_, default, **kw) -> Field:
    return Field(type=type_, default=default, **kw)


SCHEMA: Dict[str, Field] = {
    # ---------------- identity / paths ----------------
    "meta.project": _f(str, "mps-mvp-bench", doc="项目标识，用于容器 label 与结果目录前缀"),
    "meta.run_id": _f((str, type(None)), None, stage="run",
                      doc="留空则自动生成 <utc时间戳>-<随机后缀>"),
    "meta.notes": _f(str, "", doc="自由文本，写入 manifest"),
    "paths.results_dir": _f(str, "results", doc="结果根目录，run 会在其下建 <run_id>/"),
    "paths.mps_pipe_dir": _f(str, "/tmp/mps-mvp-bench/pipe",
                             stage="mps-start", doc="本实验专属 MPS pipe 目录，禁止用默认 /tmp/nvidia-mps"),
    "paths.mps_log_dir": _f(str, "/tmp/mps-mvp-bench/log", stage="mps-start"),
    "paths.model_mount": _f((str, type(None)), None,
                            doc="宿主机权重/输入目录，容器内挂到 /assets（只读）"),

    # ---------------- GPU / docker runtime (user supplied) ----------------
    "gpu.uuid": _f((str, type(None)), None, stage="preflight",
                   doc="目标 GPU UUID，形如 GPU-xxxxxxxx-....。真实 GPU 执行前必填"),
    "gpu.expect_total_memory_mib": _f((int, type(None)), None, unit="MiB", min=1,
                                      doc="可选断言：运行时查询到的总显存必须等于该值"),
    "docker.binary": _f(str, "docker", stage="run"),
    "docker.image": _f(str, "mps-mvp-bench:0.1.0", stage="run"),
    "docker.runtime_args": _f(list, [], stage="run",
                              doc="用户指定的 GPU runtime/设备透传参数 argv 片段，例如 "
                                  "['--runtime=nvidia','--gpus','device=GPU-...']。不写死、不默认 --gpus all"),
    "docker.extra_args": _f(list, [], stage="run", doc="附加 docker argv 片段（如 --cpuset-cpus）"),
    "docker.network": _f(str, "none", stage="run",
                         doc="业务容器网络。在线入口默认只对实验网络开放"),
    "docker.user": _f((str, type(None)), None, stage="run",
                      doc="容器 UID[:GID]，需与宿主机 MPS pipe 所属 UID 兼容"),
    "docker.ipc": _f((str, type(None)), None, stage="run"),
    "docker.shm_size": _f(str, "1g", stage="run"),
    "docker.privileged": _f(bool, False, stage="run", doc="默认 False；置 True 需在 RUNBOOK 中说明理由"),
    "docker.mount_docker_socket": _f(bool, False, stage="run", doc="默认 False，禁止把 docker socket 挂进业务容器"),
    "docker.pull": _f(bool, False, stage="run", doc="是否允许 docker pull；离线环境保持 False"),

    # ---------------- MPS control ----------------
    "mps.enabled": _f(bool, True, stage="mps-start", doc="由 case.mode 决定，通常不手工设置"),
    "mps.control_binary": _f(str, "nvidia-cuda-mps-control", stage="mps-start"),
    "mps.manage_on_host": _f(bool, True, stage="mps-start",
                             doc="True=宿主机启动 MPS daemon（驱动版本匹配），容器共享 pipe 路径"),
    "mps.allow_adopt_existing": _f(bool, False, stage="mps-start",
                                   doc="默认 False：发现非本实验的 MPS 实例则停止并报告，不接管"),
    "mps.start_timeout_s": _f((int, float), 30, unit="s", min=1, max=600, stage="mps-start"),
    "mps.stop_timeout_s": _f((int, float), 60, unit="s", min=1, max=600, stage="cleanup"),
    "mps.static_partitioning.enabled": _f(bool, False, stage="mps-start",
                                          doc="静态 SM 分区用例专用；与 active_thread_percentage 分开配置"),
    "mps.static_partitioning.partitions": _f(list, [], stage="mps-start",
                                             doc="每个元素 {name, sm_count}；不支持则用例 SKIP"),
    "compute_mode.require_default_for_nonmps": _f(bool, True, stage="preflight",
                                                  doc="非 MPS 双进程基线要求允许多进程；EXCLUSIVE_PROCESS 下基线失败不得算作 MPS 收益"),
    "compute_mode.allow_change": _f(bool, False, stage="preflight",
                                    doc="默认不改 compute mode；置 True 需用户授权，仅针对目标 UUID 并在 cleanup 恢复"),

    # ---------------- clients ----------------
    # Per-client knobs live under clients.a.* / clients.b.*; declared via _client_schema().
    # ---------------- measurement ----------------
    "measure.profile": _f(str, "standard", choices=("smoke", "standard"), stage="run"),
    "measure.warmup_s": _f((int, float), 30, unit="s", min=0, max=3600, stage="run"),
    "measure.duration_s": _f((int, float), 120, unit="s", min=1, max=7200, stage="run"),
    "measure.repeats": _f(int, 5, unit="rounds", min=1, max=100, stage="run"),
    "measure.drain_timeout_s": _f((int, float), 30, unit="s", min=0, max=600, stage="run",
                                  doc="测量窗口结束后等待在途请求 drain 的时间；超时请求单独计数"),
    "measure.interleave": _f(str, "ABBA", choices=("none", "ABBA", "random"), stage="run",
                             doc="比较模式的轮次顺序，降低时间漂移偏差"),
    "measure.seed": _f(int, 20251001, min=0, stage="run",
                       doc="到达 trace / 合成输入 / 权重初始化共用的主种子"),
    "measure.total_budget_s": _f((int, float, type(None)), None, unit="s", min=1, stage="run",
                                 doc="总时长上限；展开用例超预算时拒绝执行并提示缩小子集"),
    "measure.request_timeout_s": _f((int, float), 5.0, unit="s", min=0.001, max=600, stage="run"),
    "measure.min_samples_p999": _f(int, 100000, unit="requests", min=1000, stage="report",
                                   doc="低于该样本数时 P99.9 标记为样本不足而不是给出数值"),
    "measure.tolerance": _f((int, float), 1e-3, stage="run",
                            doc="输出一致性比较容差，固定值；不得因 MPS 结果不同而放宽"),

    # ---------------- SLO / report thresholds (never hardcoded in code) ----------------
    "slo.latency_p99_ms": _f((int, float, type(None)), None, unit="ms", min=0, stage="report"),
    "slo.max_latency_delta": _f((int, float, type(None)), None, stage="report",
                                doc="高优 P99 劣化上限，例如 0.2 表示 +20%；留空则只报告不判定"),
    "slo.min_throughput_delta": _f((int, float, type(None)), None, stage="report",
                                   doc="吞吐收益阈值；不得在代码里硬编码"),
    "slo.max_error_rate": _f((int, float), 0.0, min=0, max=1, stage="report"),
    "slo.reference": _f(str, "B2", choices=("B0", "B2"), stage="report",
                        doc="判定所用参考基线；报告始终同时展示相对 B0 与 B2"),

    # ---------------- telemetry ----------------
    "telemetry.enabled": _f(bool, True, stage="run"),
    "telemetry.sample_interval_s": _f((int, float), 1.0, unit="s", min=0.05, max=60, stage="run"),
    "telemetry.nvidia_smi_binary": _f(str, "nvidia-smi", stage="run"),
    "telemetry.dcgmi_binary": _f(str, "dcgmi", stage="run"),
    "telemetry.require_sm_metrics": _f(bool, True, stage="run",
                                       doc="标准验收默认 True；SM active 不可用时不得宣称完成可观测验收"),
    "telemetry.dcgm_fields": _f(list, [1002, 1003, 1005], stage="run",
                                doc="1002 SM_ACTIVE(必做) / 1003 SM_OCCUPANCY / 1005 DRAM_ACTIVE"),
    "telemetry.allow_degraded": _f(bool, False, stage="run",
                                   doc="显式降级采集；报告必须显著标记缺项"),
    "telemetry.nsight.enabled": _f(bool, False, stage="run",
                                   doc="Nsight 仅 opt-in 单独运行，不与正式性能结果混用"),

    # ---------------- memory quota专项 ----------------
    "memory.total_bytes_source": _f(str, "runtime", choices=("runtime",), stage="run",
                                    doc="50% 一律按运行时查询到的物理总显存计算"),
    "memory.probe_step_mib": _f(int, 256, unit="MiB", min=1, max=16384, stage="run"),
    "memory.probe_touch": _f(bool, True, stage="run", doc="分配后触碰内存，避免只记账不落地"),
    "memory.safety_margin_mib": _f(int, 1024, unit="MiB", min=0, stage="run",
                                   doc="业务自身安全工作集预算，不把测试工作集压到配额边缘"),
    "memory.expect_nonmps_no_quota": _f(bool, True, stage="run",
                                        doc="非 MPS 对照只在物理余量足够时验证不被配额拦截，不为测 OOM 耗尽整卡"),

    # ---------------- faults ----------------
    "faults.enabled": _f(bool, False, stage="run"),
    "faults.allow_disruptive": _f(bool, False, stage="run",
                                  doc="必须同时通过 CLI --allow-disruptive 与目标 UUID 确认"),
    "faults.inject_after_s": _f((int, float), 20, unit="s", min=0, max=3600, stage="run"),
    "faults.observe_window_s": _f((int, float), 30, unit="s", min=1, max=3600, stage="run",
                                  doc="先观测故障影响再恢复，避免自动重启掩盖传播"),
    "faults.repeats": _f(int, 3, unit="times", min=1, max=100, stage="run"),
    "faults.drain_timeout_s": _f((int, float), 15, unit="s", min=0, max=600, stage="run"),
    "faults.recovery_timeout_s": _f((int, float), 120, unit="s", min=1, max=3600, stage="run"),
    "faults.long_kernel_ms": _f((int, float), 2000, unit="ms", min=1, max=60000, stage="run",
                                doc="有界长 kernel 时限；禁止无限循环"),
    "faults.recovery_levels": _f(list, ["R1", "R2"], stage="run",
                                 doc="按最小故障域推进；R3 只输出人工诊断建议，不自动 GPU reset"),

    # ---------------- safety ----------------
    "safety.abort_on_foreign_gpu_process": _f(bool, True, stage="preflight",
                                              doc="目标卡存在非本项目 GPU 进程则拒绝；每轮前重新检查"),
    "safety.health_check_between_rounds": _f(bool, True, stage="run"),
    "safety.stop_on_unhealthy": _f(bool, True, stage="run",
                                   doc="异常后无法确认 GPU 健康则停止后续实验，不自动重试掩盖污染"),
    "safety.lock_dir": _f(str, "/tmp/mps-mvp-bench/locks", stage="run",
                          doc="GPU UUID 级互斥锁目录"),
}

_ARRIVAL_MODES = ("closed_loop", "fixed_interval", "poisson", "burst_trace")
_PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
_WORKLOADS = ("image_inference", "recsys_scoring")


def _client_schema(slot: str) -> Dict[str, Field]:
    p = f"clients.{slot}"
    return {
        f"{p}.role": _f(str, "high" if slot == "a" else "high", choices=("high", "low"),
                        doc="high=高优, low=低优；决定报告中的 QoS 归类"),
        f"{p}.workload": _f(str, "image_inference", choices=_WORKLOADS, stage="container-start"),
        f"{p}.mode": _f(str, "online", choices=("online", "offline"), stage="container-start",
                        doc="online=开环固定到达率(测延迟), offline=闭环饱和(测容量)"),
        # --- MPS per-client knobs: must be set before CUDA init, so they are
        # container env vars and require a NEW client to take effect.
        f"{p}.mps.active_thread_percentage": _f((int, type(None)), None, unit="%", min=1, max=100,
                                                stage="client-start",
                                                doc="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE；执行资源上限，"
                                                    "不是固定 SM 预留，也不保证抢占"),
        f"{p}.mps.priority": _f((str, type(None)), None, choices=("NORMAL", "BELOW_NORMAL", None),
                                stage="client-start",
                                doc="CUDA_MPS_CLIENT_PRIORITY: NORMAL=0 / BELOW_NORMAL=1"),
        f"{p}.mps.pinned_device_mem_limit_fraction": _f((int, float, type(None)), None, min=0.01, max=1.0,
                                                        stage="client-start",
                                                        doc="按运行时物理总显存的比例生成 "
                                                            "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"),
        f"{p}.mps.static_partition": _f((str, type(None)), None, stage="client-start",
                                        doc="绑定到的静态分区名；需 mps.static_partitioning.enabled"),
        # --- workload knobs ---
        f"{p}.model.name": _f(str, "resnet18", stage="container-start",
                              doc="image_inference: resnet18/resnet50"),
        f"{p}.model.input_size": _f(int, 224, unit="px", min=32, max=1024, stage="container-start"),
        f"{p}.model.precision": _f(str, "fp16", choices=_PRECISIONS, stage="container-start"),
        f"{p}.model.batch_size": _f(int, 8, unit="samples", min=1, max=4096, stage="container-start"),
        f"{p}.model.weights_path": _f((str, type(None)), None, stage="container-start",
                                      doc="留空=离线固定种子合成权重，结果标注 synthetic"),
        f"{p}.model.input_pool_size": _f(int, 64, unit="batches", min=1, max=100000,
                                         stage="container-start",
                                         doc="输入池大小；过小会导致只压 L2 而非 HBM"),
        f"{p}.model.cpu_threads": _f(int, 4, unit="threads", min=1, max=256, stage="container-start"),
        # recsys (DLRM-style) knobs
        f"{p}.recsys.num_tables": _f(int, 16, min=1, max=1024, stage="container-start"),
        f"{p}.recsys.rows_per_table": _f(int, 200000, unit="rows", min=16, stage="container-start"),
        f"{p}.recsys.embedding_dim": _f(int, 64, min=1, max=1024, stage="container-start"),
        f"{p}.recsys.indices_per_sample": _f(int, 32, min=1, max=4096, stage="container-start"),
        f"{p}.recsys.dense_features": _f(int, 128, min=1, max=8192, stage="container-start"),
        f"{p}.recsys.bottom_mlp": _f(list, [512, 256, 64], stage="container-start"),
        f"{p}.recsys.top_mlp": _f(list, [512, 256, 1], stage="container-start"),
        f"{p}.recsys.index_distribution": _f(str, "zipf", choices=("uniform", "zipf"),
                                             stage="container-start"),
        f"{p}.recsys.zipf_s": _f((int, float), 1.1, min=0.0, max=5.0, stage="container-start"),
        f"{p}.recsys.working_set_rows": _f((int, type(None)), None, unit="rows", min=1,
                                           stage="container-start",
                                           doc="索引工作集；必须可配，避免反复命中极小固定集合"),
        # --- load generation (CPU-side process, no CUDA context) ---
        f"{p}.load.arrival": _f(str, "poisson", choices=_ARRIVAL_MODES, stage="run"),
        f"{p}.load.target_qps": _f((int, float, type(None)), 50.0, unit="req/s", min=0.01,
                                   stage="run", doc="online 模式必填；offline 模式忽略"),
        f"{p}.load.concurrency": _f(int, 8, min=1, max=4096, stage="run",
                                    doc="offline 模式的闭环并发；online 模式为发送端上限"),
        f"{p}.load.burst_trace": _f(list, [], stage="run",
                                    doc="可重复突发 trace: [{t, qps}]；arrival=burst_trace 时必填"),
        f"{p}.load.queue_limit": _f(int, 1024, min=1, max=1000000, stage="run",
                                    doc="服务端队列上限，超出记为 rejected"),
        f"{p}.load.port": _f(int, 18801 if slot == "a" else 18802, min=1024, max=65535, stage="run"),
        # --- container-side resources ---
        f"{p}.container.name_suffix": _f(str, slot, stage="run"),
        f"{p}.container.cpuset": _f((str, type(None)), None, stage="run",
                                    doc="固定 CPU 亲和性，保持跨模式一致"),
        f"{p}.container.memory": _f((str, type(None)), None, stage="run"),
    }


for _slot in ("a", "b"):
    SCHEMA.update(_client_schema(_slot))

# Case-level keys (present in cases/**.yaml, merged into the same tree).
SCHEMA.update({
    "case.id": _f((str, type(None)), None, doc="用例 ID，例如 B0/B2/M0/HH70/HL-PC"),
    "case.family": _f(str, "baseline",
                      choices=("baseline", "high_high", "high_low", "memory", "negative", "fault"),
                      doc="决定结果归类与报告章节"),
    "case.mode": _f(str, "mps_concurrent",
                    choices=("nonmps_solo", "nonmps_concurrent", "mps_solo", "mps_concurrent"),
                    doc="是否启用 MPS、单跑还是同卡并发"),
    "case.description": _f(str, ""),
    "case.solo_client": _f((str, type(None)), None, choices=("a", "b", None),
                           doc="*_solo 模式下实际运行的 client"),
    "case.reference": _f(list, ["B0", "B2"], stage="report",
                         doc="报告必须同时展示相对 B0 与 B2"),
    "case.expect": _f(dict, {}, stage="report",
                      doc="用例声明的期望，例如 {low_priority_oom: true}；不满足即 FAIL 而非静默通过"),
    "case.sweep": _f(list, [], doc="可审计的参数扫描；不展开全参数笛卡尔积"),
    "case.fault": _f((str, type(None)), None,
                     choices=("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", None)),
    "case.enabled": _f(bool, True),
})

# Profile presets. `smoke` only proves the链路 works; it is never valid evidence.
PROFILES: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "measure.warmup_s": 5,
        "measure.duration_s": 15,
        "measure.repeats": 1,
        "measure.drain_timeout_s": 10,
        "telemetry.require_sm_metrics": False,
    },
    "standard": {
        "measure.warmup_s": 30,
        "measure.duration_s": 120,
        "measure.repeats": 5,
        "measure.drain_timeout_s": 30,
        "telemetry.require_sm_metrics": True,
    },
}


# --------------------------------------------------------------------------- #
# flatten / unflatten helpers
# --------------------------------------------------------------------------- #

_LEAF_CONTAINER_KEYS = {k for k, v in SCHEMA.items() if v.type in (list, dict) or v.type is list}


def flatten(tree: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict to dotted keys, stopping at declared leaf containers."""
    out: Dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and path not in SCHEMA:
            out.update(flatten(value, prefix=f"{path}."))
        else:
            out[path] = value
    return out


def unflatten(flat: Dict[str, Any]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for path, value in sorted(flat.items()):
        parts = path.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def defaults() -> Dict[str, Any]:
    return {key: copy.deepcopy(field.default) for key, field in SCHEMA.items()}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def _coerce(key: str, field: Field, value: Any) -> Any:
    types = field.type if isinstance(field.type, tuple) else (field.type,)
    if value is None:
        if type(None) in types:
            return None
        raise ConfigError(f"{key}: null 不被允许（期望 {_type_names(types)}）")
    # bool must be checked before int: bool is a subclass of int in Python.
    if bool in types and isinstance(value, bool):
        return value
    if isinstance(value, bool) and bool not in types:
        raise ConfigError(f"{key}: 期望 {_type_names(types)}，收到 bool")
    if isinstance(value, types):
        return value
    # Allow int -> float widening only.
    if float in types and isinstance(value, int):
        return float(value)
    raise ConfigError(f"{key}: 期望 {_type_names(types)}，收到 {type(value).__name__}")


def _type_names(types: Iterable[Any]) -> str:
    return "/".join("null" if t is type(None) else t.__name__ for t in types)


def validate(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a fully-merged flat config. Returns it unchanged (or coerced)."""
    unknown = sorted(set(flat) - set(SCHEMA))
    if unknown:
        raise ConfigError("未知配置项（拒绝执行）: " + ", ".join(unknown))

    out: Dict[str, Any] = {}
    for key, field in SCHEMA.items():
        value = _coerce(key, field, flat.get(key, copy.deepcopy(field.default)))
        if value is not None and field.choices is not None and value not in field.choices:
            raise ConfigError(f"{key}: 取值必须属于 {field.choices}，收到 {value!r}")
        if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            if field.min is not None and value < field.min:
                raise ConfigError(f"{key}: {value} < 最小值 {field.min} {field.unit}".rstrip())
            if field.max is not None and value > field.max:
                raise ConfigError(f"{key}: {value} > 最大值 {field.max} {field.unit}".rstrip())
        out[key] = value

    _validate_cross(out)
    return out


def _validate_cross(cfg: Dict[str, Any]) -> None:
    mode = cfg["case.mode"]
    solo = mode.endswith("_solo")
    if solo and cfg["case.solo_client"] not in ("a", "b"):
        raise ConfigError("case.mode 为 *_solo 时必须指定 case.solo_client=a|b")
    if not solo and cfg["case.solo_client"] is not None:
        raise ConfigError("case.solo_client 只能用于 *_solo 模式")

    mps_mode = mode.startswith("mps_")
    for slot in ("a", "b"):
        p = f"clients.{slot}"
        if not mps_mode:
            for suffix, label in (("mps.active_thread_percentage", "执行资源上限"),
                                  ("mps.priority", "优先级"),
                                  ("mps.pinned_device_mem_limit_fraction", "显存配额"),
                                  ("mps.static_partition", "静态分区")):
                if cfg[f"{p}.{suffix}"] is not None:
                    raise ConfigError(
                        f"{p}.{suffix}: 非 MPS 模式不得配置 {label}（不能伪装存在 MPS 配额）")
        if cfg[f"{p}.mode"] == "online" and cfg[f"{p}.load.target_qps"] is None:
            raise ConfigError(f"{p}.load.target_qps: online 模式必须指定到达率")
        if cfg[f"{p}.load.arrival"] == "burst_trace" and not cfg[f"{p}.load.burst_trace"]:
            raise ConfigError(f"{p}.load.burst_trace: arrival=burst_trace 时必须提供 trace")
        if cfg[f"{p}.mode"] == "offline" and cfg[f"{p}.load.arrival"] != "closed_loop":
            raise ConfigError(f"{p}.load.arrival: offline 容量测试必须使用 closed_loop")
        if cfg[f"{p}.mode"] == "online" and cfg[f"{p}.load.arrival"] == "closed_loop":
            raise ConfigError(f"{p}.load.arrival: online 延迟测试必须使用开环到达模式")
        part = cfg[f"{p}.mps.static_partition"]
        if part is not None:
            if not cfg["mps.static_partitioning.enabled"]:
                raise ConfigError(f"{p}.mps.static_partition: 需先开启 mps.static_partitioning.enabled")
            names = {d.get("name") for d in cfg["mps.static_partitioning.partitions"]
                     if isinstance(d, dict)}
            if part not in names:
                raise ConfigError(f"{p}.mps.static_partition={part!r} 未在 "
                                  f"mps.static_partitioning.partitions 中声明")
            if cfg[f"{p}.mps.active_thread_percentage"] is not None:
                raise ConfigError(
                    f"{p}: 静态分区用例与 active_thread_percentage 必须分开配置，"
                    "避免把被忽略的设置当成生效")

    if cfg["clients.a.load.port"] == cfg["clients.b.load.port"]:
        raise ConfigError("clients.a.load.port 与 clients.b.load.port 不能相同")

    if cfg["faults.enabled"] and cfg["case.fault"] is None:
        raise ConfigError("faults.enabled=true 时必须指定 case.fault")
    if cfg["case.fault"] in ("F4", "F5", "F6", "F7", "F8", "F9") and not cfg["faults.allow_disruptive"]:
        # A disabled case is never executed, so it must still be loadable (for
        # `plan`/listing). The gate applies to cases that would actually run.
        if cfg.get("case.enabled", True):
            raise ConfigError(f"{cfg['case.fault']} 属于破坏性用例，需 faults.allow_disruptive=true "
                              "且 CLI 传 --allow-disruptive 并确认目标 UUID")
    if cfg["docker.mount_docker_socket"]:
        raise ConfigError("docker.mount_docker_socket 必须为 false：禁止把 docker socket 挂入业务容器")
    if cfg["telemetry.require_sm_metrics"] and cfg["telemetry.allow_degraded"]:
        raise ConfigError("telemetry.require_sm_metrics 与 telemetry.allow_degraded 互斥")
    if 1002 not in cfg["telemetry.dcgm_fields"] and cfg["telemetry.require_sm_metrics"]:
        raise ConfigError("telemetry.dcgm_fields 必须包含 1002 (DCGM_FI_PROF_SM_ACTIVE)")


# --------------------------------------------------------------------------- #
# loading / merging
# --------------------------------------------------------------------------- #

def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: 顶层必须是 mapping")
    return data


def parse_set_value(raw: str) -> Any:
    """Parse a --set value using YAML scalar rules (so 50, true, null, [1,2] work)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_overrides(flat: Dict[str, Any], overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set 需要 key=value 形式，收到 {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in SCHEMA:
            raise ConfigError(f"--set 未知配置项: {key}")
        flat[key] = parse_set_value(raw)


@dataclass
class LoadedConfig:
    values: Dict[str, Any]
    sources: List[str]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @property
    def tree(self) -> Dict[str, Any]:
        return unflatten(self.values)

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.tree, allow_unicode=True, sort_keys=True, default_flow_style=False)

    def hash(self) -> str:
        blob = json.dumps(self.values, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load(config_files: Optional[Iterable[str]] = None,
         profile: Optional[str] = None,
         case_file: Optional[str] = None,
         overrides: Optional[Iterable[str]] = None,
         profiles_dir: Optional[str] = None) -> LoadedConfig:
    """Merge every layer, then validate. See module docstring for precedence."""
    flat = defaults()
    sources: List[str] = ["<defaults>"]

    for path in config_files or []:
        flat.update(flatten(_read_yaml(path)))
        sources.append(path)

    if profile:
        if profile not in PROFILES and profiles_dir is None:
            raise ConfigError(f"未知 profile: {profile}（内置: {sorted(PROFILES)}）")
        if profile in PROFILES:
            flat.update(PROFILES[profile])
            flat["measure.profile"] = profile
            sources.append(f"<profile:{profile}>")
        if profiles_dir:
            candidate = os.path.join(profiles_dir, f"{profile}.yaml")
            if os.path.exists(candidate):
                flat.update(flatten(_read_yaml(candidate)))
                sources.append(candidate)

    if case_file:
        flat.update(flatten(_read_yaml(case_file)))
        sources.append(case_file)

    if overrides:
        _apply_overrides(flat, overrides)
        sources.append("<cli --set>")

    return LoadedConfig(values=validate(flat), sources=sources)


def describe_schema() -> List[Dict[str, Any]]:
    """Machine-readable schema dump, used to keep docs/CONFIG.zh-CN.md honest."""
    rows = []
    for key, field in sorted(SCHEMA.items()):
        rows.append({
            "key": key,
            "type": _type_names(field.type if isinstance(field.type, tuple) else (field.type,)),
            "default": field.default,
            "unit": field.unit,
            "min": field.min,
            "max": field.max,
            "choices": list(field.choices) if field.choices else None,
            "stage": field.stage,
            "doc": field.doc,
        })
    return rows
