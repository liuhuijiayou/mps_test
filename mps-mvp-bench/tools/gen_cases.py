#!/usr/bin/env python3
"""Generate the case matrix YAML files (cases/**).

Kept as a generator so the matrix stays consistent and reviewable; the produced
YAML files are the actual inputs and are committed.
"""
from __future__ import annotations

import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cases")

IMG_ONLINE = {"workload": "image_inference", "mode": "online",
              "model": {"name": "resnet18", "input_size": 224, "precision": "fp16",
                        "batch_size": 8, "input_pool_size": 64, "cpu_threads": 4},
              "load": {"arrival": "poisson", "target_qps": 120.0, "concurrency": 32,
                       "queue_limit": 512}}
IMG_OFFLINE = {"workload": "image_inference", "mode": "offline",
               "model": {"name": "resnet50", "input_size": 224, "precision": "fp16",
                         "batch_size": 64, "input_pool_size": 32, "cpu_threads": 4},
               "load": {"arrival": "closed_loop", "target_qps": None, "concurrency": 4,
                        "queue_limit": 64}}
REC_ONLINE = {"workload": "recsys_scoring", "mode": "online",
              "model": {"precision": "fp16", "batch_size": 512, "input_pool_size": 32,
                        "cpu_threads": 4},
              "recsys": {"num_tables": 16, "rows_per_table": 400000, "embedding_dim": 64,
                         "indices_per_sample": 32, "dense_features": 128,
                         "bottom_mlp": [512, 256, 64], "top_mlp": [512, 256, 1],
                         "index_distribution": "zipf", "zipf_s": 1.1,
                         "working_set_rows": 400000},
              "load": {"arrival": "poisson", "target_qps": 60.0, "concurrency": 16,
                       "queue_limit": 256}}
REC_OFFLINE = {"workload": "recsys_scoring", "mode": "offline",
               "model": {"precision": "fp16", "batch_size": 4096, "input_pool_size": 16,
                         "cpu_threads": 4},
               "recsys": {"num_tables": 24, "rows_per_table": 800000, "embedding_dim": 128,
                          "indices_per_sample": 48, "dense_features": 256,
                          "bottom_mlp": [1024, 512, 128], "top_mlp": [1024, 512, 1],
                          "index_distribution": "zipf", "zipf_s": 1.0,
                          "working_set_rows": 800000},
               "load": {"arrival": "closed_loop", "target_qps": None, "concurrency": 4,
                        "queue_limit": 32}}


def merge(*blocks):
    out = {}
    for block in blocks:
        for key, value in block.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value)
            else:
                out[key] = value
    return out


def mps(atp=None, prio=None, mem=None, part=None):
    block = {}
    if atp is not None:
        block["active_thread_percentage"] = atp
    if prio is not None:
        block["priority"] = prio
    if mem is not None:
        block["pinned_device_mem_limit_fraction"] = mem
    if part is not None:
        block["static_partition"] = part
    return {"mps": block} if block else {}


def write(family, name, doc):
    path = os.path.join(ROOT, family, f"{name}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# 由 tools/gen_cases.py 生成，可手工微调；字段含义见 docs/CONFIG.zh-CN.md\n")
        yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return path


def case(cid, family, mode, description, a, b, solo=None, expect=None, fault=None,
         extra=None, sweep=None, reference=("B0", "B2")):
    doc = {"case": {"id": cid, "family": family, "mode": mode,
                    "description": description, "reference": list(reference)}}
    if solo:
        doc["case"]["solo_client"] = solo
    if expect:
        doc["case"]["expect"] = expect
    if fault:
        doc["case"]["fault"] = fault
    if sweep:
        doc["case"]["sweep"] = sweep
    doc["clients"] = {"a": a, "b": b}
    if extra:
        doc = merge(doc, extra)
    return doc


HIGH = {"role": "high"}
LOW = {"role": "low"}


def main():
    # ---------------- baselines ----------------
    write("baselines", "B0-solo-a", case(
        "B0", "baseline", "nonmps_solo", "非 MPS，A 单跑：整卡独占基线",
        merge(HIGH, IMG_ONLINE), merge(HIGH, IMG_OFFLINE), solo="a", reference=("B0",)))
    write("baselines", "B0-solo-b", case(
        "B0", "baseline", "nonmps_solo", "非 MPS，B 单跑：整卡独占基线",
        merge(HIGH, IMG_ONLINE), merge(HIGH, IMG_OFFLINE), solo="b", reference=("B0",)))
    write("baselines", "B1-mps-solo-a-100", case(
        "B1", "baseline", "mps_solo", "MPS，A 单跑 100%：区分 MPS 本身开销",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
        merge(HIGH, IMG_OFFLINE, mps(100, "NORMAL")), solo="a"))
    write("baselines", "B1-mps-solo-a-50", case(
        "B1", "baseline", "mps_solo", "MPS，A 单跑 50%：被测配额下的受限单跑，用于解释限额成本",
        merge(HIGH, IMG_ONLINE, mps(50, "NORMAL")),
        merge(HIGH, IMG_OFFLINE, mps(50, "NORMAL")), solo="a"))
    write("baselines", "B1-mps-solo-a-70", case(
        "B1", "baseline", "mps_solo", "MPS，A 单跑 70%：被测配额下的受限单跑",
        merge(HIGH, IMG_ONLINE, mps(70, "NORMAL")),
        merge(HIGH, IMG_OFFLINE, mps(70, "NORMAL")), solo="a"))
    write("baselines", "B2-nonmps-concurrent", case(
        "B2", "baseline", "nonmps_concurrent", "非 MPS，A+B 同卡并发：同卡并发主对照",
        merge(HIGH, IMG_ONLINE), merge(HIGH, IMG_OFFLINE)))
    write("baselines", "M0-mps-concurrent", case(
        "M0", "baseline", "mps_concurrent", "MPS，A+B 同卡并发，各 100%：无算力限制的 MPS 对照",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
        merge(HIGH, IMG_OFFLINE, mps(100, "NORMAL"))))

    # ---------------- high + high ----------------
    for cid, atp in (("HH70", 70), ("HH50", 50)):
        write("high_high", f"{cid}-image-image", case(
            cid, "high_high", "mps_concurrent",
            f"高优+高优，SM 不隔离，执行资源上限各 {atp}%，显存各 50% 上限（图像+图像）",
            merge(HIGH, IMG_ONLINE, mps(atp, "NORMAL", 0.5)),
            merge(HIGH, IMG_OFFLINE, mps(atp, "NORMAL", 0.5))))
        write("high_high", f"{cid}-recsys-recsys", case(
            cid, "high_high", "mps_concurrent",
            f"高优+高优，各 {atp}%，显存各 50%（推荐+推荐）",
            merge(HIGH, REC_ONLINE, mps(atp, "NORMAL", 0.5)),
            merge(HIGH, REC_OFFLINE, mps(atp, "NORMAL", 0.5))))
        write("high_high", f"{cid}-online-offline", case(
            cid, "high_high", "mps_concurrent",
            f"高优+高优，各 {atp}%，在线小 batch + 离线大 batch（异构，吞吐不合计）",
            merge(HIGH, IMG_ONLINE, mps(atp, "NORMAL", 0.5)),
            merge(HIGH, REC_OFFLINE, mps(atp, "NORMAL", 0.5))))
    write("high_high", "HH-S-static-partition", case(
        "HH-S", "high_high", "mps_concurrent",
        "高优+高优，静态 SM 分区各约 50% 不重叠物理 SM。能力不支持则 SKIP，"
        "禁止用 ACTIVE_THREAD_PERCENTAGE/亲和性/MIG 冒充",
        merge(HIGH, IMG_ONLINE, mps(prio="NORMAL", mem=0.5, part="partA")),
        merge(HIGH, IMG_OFFLINE, mps(prio="NORMAL", mem=0.5, part="partB")),
        extra={"mps": {"static_partitioning": {
            "enabled": True,
            # sm_count 需按 preflight 查到的实际 SM 数与 chunk 粒度填写；
            # 不能精确对半时如实展示实际 SM 数与分区 ID
            "partitions": [{"name": "partA", "sm_count": 28},
                           {"name": "partB", "sm_count": 28}]}}}))

    # ---------------- high + low ----------------
    write("high_low", "HL-P-priority-only", case(
        "HL-P", "high_low", "mps_concurrent",
        "高优+低优，仅优先级（均 100%）：优先级单因素。"
        "CLIENT_PRIORITY 提供软优先级，不抢占已运行 Block",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
        merge(LOW, IMG_OFFLINE, mps(100, "BELOW_NORMAL", 0.5))))
    write("high_low", "HL-C-cap-only", case(
        "HL-C", "high_low", "mps_concurrent",
        "高优+低优，仅执行资源上限：高 100% / 低 50%，均 NORMAL",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
        merge(LOW, IMG_OFFLINE, mps(50, "NORMAL", 0.5)),
        sweep=[{"name": "low70", "set": {"clients.b.mps.active_thread_percentage": 70}}]))
    write("high_low", "HL-PC-combined", case(
        "HL-PC", "high_low", "mps_concurrent",
        "高优+低优，优先级 + 预留空间组合：高 100%/NORMAL，低 50%/BELOW_NORMAL",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
        merge(LOW, IMG_OFFLINE, mps(50, "BELOW_NORMAL", 0.5)),
        sweep=[{"name": "low70", "set": {"clients.b.mps.active_thread_percentage": 70}}]))
    write("high_low", "HL-PC-recsys", case(
        "HL-PC", "high_low", "mps_concurrent",
        "高优在线推荐 + 低优饱和推荐，优先级+上限组合",
        merge(HIGH, REC_ONLINE, mps(100, "NORMAL")),
        merge(LOW, REC_OFFLINE, mps(50, "BELOW_NORMAL", 0.5))))

    # ---------------- memory quota专项 ----------------
    write("memory", "MEM-HH-50", case(
        "MEM-HH-50", "memory", "mps_concurrent",
        "高高各 50% 显存上限：分别触达配额边界，一方超额时另一方继续真实推理并做正确性检查",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL", 0.5)),
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL", 0.5)),
        expect={"client_a_quota_oom": True, "client_b_quota_oom": True,
                "peer_correctness": True}))
    write("memory", "MEM-HL-50", case(
        "MEM-HL-50", "memory", "mps_concurrent",
        "仅低优 50% 显存上限：低优在边界失败；高优在物理空闲足够时可超过 50%，"
        "证明未被同样限额（不宣称低优驻留时高优必然可申请整卡）",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
        merge(LOW, IMG_ONLINE, mps(50, "BELOW_NORMAL", 0.5)),
        expect={"client_b_quota_oom": True, "high_exceeds_half": True,
                "peer_correctness": True}))
    write("memory", "MEM-NONMPS-control", case(
        "MEM-NONMPS-control", "memory", "nonmps_concurrent",
        "非 MPS 对照：仅在物理余量足够时验证跨过相同比例不会被 MPS 配额拦截，"
        "不为测 OOM 耗尽整卡",
        merge(HIGH, IMG_ONLINE), merge(HIGH, IMG_ONLINE),
        expect={"client_a_quota_oom": False}))

    # ---------------- negative candidates ----------------
    write("negative", "NEG-compute-contention", case(
        "NEG-compute", "negative", "mps_concurrent",
        "计算争用：两个大 batch 图像推理，看单任务尾延迟是否变差、合计吞吐是否收益不足/退化",
        merge(HIGH, IMG_OFFLINE, mps(100, "NORMAL")),
        merge(HIGH, IMG_OFFLINE, mps(100, "NORMAL")),
        sweep=[{"name": "batch128", "set": {"clients.a.model.batch_size": 128,
                                            "clients.b.model.batch_size": 128}},
               {"name": "batch256", "set": {"clients.a.model.batch_size": 256,
                                            "clients.b.model.batch_size": 256}}]))
    write("negative", "NEG-bandwidth-contention", case(
        "NEG-bandwidth", "negative", "mps_concurrent",
        "显存带宽/L2 争用：两个较大 embedding 推荐任务，观察 SM active 上升时是否仍有业务退化",
        merge(HIGH, REC_OFFLINE, mps(100, "NORMAL")),
        merge(HIGH, REC_OFFLINE, mps(100, "NORMAL")),
        sweep=[{"name": "ws-2x", "set": {"clients.a.recsys.working_set_rows": 1600000,
                                         "clients.b.recsys.working_set_rows": 1600000,
                                         "clients.a.recsys.rows_per_table": 1600000,
                                         "clients.b.recsys.rows_per_table": 1600000}},
               {"name": "uniform-idx", "set": {"clients.a.recsys.index_distribution": "uniform",
                                               "clients.b.recsys.index_distribution": "uniform"}}]))
    write("negative", "NEG-high-low-interference", case(
        "NEG-high-low", "negative", "mps_concurrent",
        "高低干扰：高优在线小 batch + 低优饱和大 batch；对比无保护/仅优先级/仅上限/组合",
        merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
        merge(LOW, IMG_OFFLINE, mps(100, "NORMAL")),
        sweep=[{"name": "prio-only", "set": {"clients.b.mps.priority": "BELOW_NORMAL"}},
               {"name": "cap-only", "set": {"clients.b.mps.active_thread_percentage": 50}},
               {"name": "combined", "set": {"clients.b.mps.priority": "BELOW_NORMAL",
                                            "clients.b.mps.active_thread_percentage": 50}},
               {"name": "qps240", "set": {"clients.a.load.target_qps": 240.0}}]))
    write("negative", "NEG-quota-cost", case(
        "NEG-quota-cost", "negative", "mps_concurrent",
        "配额成本：一方活跃、一方低负载；50% 上限是否限制活跃方使用闲置能力。"
        "相对整卡变慢应标记为配额成本，而不是 MPS 本身退化",
        merge(HIGH, IMG_OFFLINE, mps(50, "NORMAL")),
        merge(LOW, IMG_ONLINE, mps(50, "NORMAL")),
        extra={"clients": {"b": {"load": {"target_qps": 2.0}}}}))

    # ---------------- fault cases ----------------
    faults = [
        ("F1", "低优配额 OOM：原生显存申请越过配额，记录错误码与对端 QoS/正确性", False),
        ("F2", "非致命 CUDA API 参数错误：确认实际错误类型与对端影响", False),
        ("F3", "SIGTERM/SIGINT 安全退出：Fence -> Drain -> Exit", False),
        ("F4", "terminate_client 确认成功后再 SIGKILL（单独显式执行）", True),
        ("F5", "空闲状态直接 SIGKILL（默认关闭）", True),
        ("F6", "GPU work 在途时直接 SIGKILL（默认关闭）", True),
        ("F7", "Device illegal memory access（默认关闭）", True),
        ("F8", "Device assert（默认关闭）", True),
        ("F9", "有界长 kernel 干扰（默认关闭）", True),
    ]
    for fid, desc, disruptive in faults:
        doc = case(fid, "fault", "mps_concurrent",
                   f"{fid} {desc}。witness=A 真实在线推理，被注入=B",
                   merge(HIGH, IMG_ONLINE, mps(100, "NORMAL")),
                   merge(LOW, IMG_OFFLINE, mps(50, "BELOW_NORMAL", 0.5)),
                   fault=fid)
        doc["case"]["enabled"] = not disruptive
        doc["faults"] = {"enabled": True, "allow_disruptive": False,
                         "inject_after_s": 20, "observe_window_s": 30, "repeats": 3,
                         "recovery_levels": ["R1", "R2"]}
        write("faults", f"{fid}", doc)
        # non-MPS counterpart so we can compare propagation with and without MPS
        doc_nonmps = case(f"{fid}-NONMPS", "fault", "nonmps_concurrent",
                          f"{fid} 非 MPS 对照：同一注入在无 MPS 下的对端影响",
                          merge(HIGH, IMG_ONLINE), merge(LOW, IMG_OFFLINE), fault=fid)
        doc_nonmps["case"]["enabled"] = not disruptive
        doc_nonmps["faults"] = {"enabled": True, "allow_disruptive": False,
                                "inject_after_s": 20, "observe_window_s": 30, "repeats": 3,
                                "recovery_levels": ["R1"]}
        write("faults", f"{fid}-nonmps", doc_nonmps)

    print(f"cases 已生成到 {ROOT}")


if __name__ == "__main__":
    main()
