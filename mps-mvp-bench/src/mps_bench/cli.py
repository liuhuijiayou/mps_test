"""CLI entry point.

Subcommands:
  preflight  read-only capability probe -> capability.json
  plan       expand cases, print counts + estimated time, execute nothing
  smoke      link check only (short profile); never valid performance evidence
  run        execute selected cases
  memory     memory-quota专项
  faults     explicit fault cases (disruptive ones need extra confirmation)
  report     (re)build summary.csv + report.html from a results dir
  cleanup    idempotent teardown of this project's leftovers
  schema     dump the config schema (keeps docs/CONFIG.zh-CN.md honest)

This module must import cleanly without torch / CUDA packages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import config as cfgmod
from . import matrix as matrixmod
from .control.cleanup import cleanup as do_cleanup
from .control.docker import DockerClient
from .control.exec import CommandLog
from .control.mps import MpsController
from .control.preflight import run_preflight
from .control.safety import GpuLock, SafetyError, gate_disruptive
from .reporting.results import Manifest, RunDirectory, image_identity, new_run_id, source_commit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CASES_DIR = os.path.join(REPO_ROOT, "cases")
DEFAULT_PROFILES_DIR = os.path.join(REPO_ROOT, "configs", "profiles")


def _common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", action="append", default=[],
                    help="runtime yaml，可重复；后者覆盖前者")
    ap.add_argument("--profile", default=None, choices=("smoke", "standard"))
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    help="key=value 覆盖，优先级最高")
    ap.add_argument("--cases-dir", default=DEFAULT_CASES_DIR)
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--family", action="append", default=[])


def _load_cfg(args: argparse.Namespace, case_file: Optional[str] = None) -> cfgmod.LoadedConfig:
    return cfgmod.load(config_files=args.config, profile=getattr(args, "profile", None),
                       case_file=case_file, overrides=args.overrides,
                       profiles_dir=DEFAULT_PROFILES_DIR)


def _expand(args: argparse.Namespace) -> List[matrixmod.ExpandedCase]:
    files = matrixmod.discover_case_files(args.cases_dir,
                                          families=args.family or None,
                                          ids=args.case_id or None)
    if not files:
        raise SystemExit(f"在 {args.cases_dir} 未找到匹配用例")
    return matrixmod.expand(files, base_configs=args.config, profile=args.profile,
                            overrides=args.overrides, profiles_dir=DEFAULT_PROFILES_DIR)


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #

_STAGE_LABEL = {
    "preflight": "预检阶段读取",
    "container-start": "容器启动时生效",
    "mps-start": "MPS 启动时生效",
    "client-start": "client 启动时生效（改动后必须新建 client）",
    "run": "测量过程中读取",
    "report": "只影响报告/判定",
    "cleanup": "清理阶段读取",
}


def _schema_markdown(items: List[Dict[str, Any]]) -> str:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["key"].split(".")[0], []).append(item)
    lines = ["<!-- 由 `mps-bench schema --markdown` 生成，请勿手改 -->", ""]
    for group in sorted(groups):
        lines.append(f"### `{group}.*`")
        lines.append("")
        lines.append("| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in sorted(groups[group], key=lambda d: d["key"]):
            default = item.get("default")
            default_s = "无" if default is None else f"`{default}`"
            if item.get("choices"):
                rng = " \\| ".join(f"`{c}`" for c in item["choices"])
            elif item.get("min") is not None or item.get("max") is not None:
                rng = f"{item.get('min', '')} ~ {item.get('max', '')}"
            else:
                rng = "—"
            unit = item.get("unit") or ""
            doc = (item.get("doc") or "").replace("|", "\\|")
            if unit:
                doc = f"[{unit}] {doc}".strip()
            lines.append(f"| `{item['key']}` | {item['type']} | {default_s} | {rng} | "
                         f"{_STAGE_LABEL.get(item['stage'], item['stage'])} | {doc or '—'} |")
        lines.append("")
    return "\n".join(lines)


def cmd_schema(args: argparse.Namespace) -> int:
    items = cfgmod.describe_schema()
    if getattr(args, "markdown", False):
        print(_schema_markdown(items))
    else:
        print(json.dumps(items, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    log = CommandLog()
    report = run_preflight(cfg, log=log)
    payload = report.as_dict()
    out_dir = args.out or cfg["paths.results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cap_path = os.path.join(out_dir, "capability.json")
    with open(cap_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    env_path = os.path.join(out_dir, "environment.json")
    with open(env_path, "w", encoding="utf-8") as fh:
        json.dump({"environment": report.environment,
                   "executed_commands": log.records}, fh, ensure_ascii=False, indent=2,
                  default=str)

    print(f"capability.json -> {cap_path}")
    for cap in report.capabilities:
        mark = {"supported": "OK ", "unsupported": "NO ", "unknown": "?? "}[cap.status]
        extra = f" [需 {cap.verify_by} 验证]" if cap.verify_by else ""
        print(f"  {mark}{cap.name}: {cap.detail}{extra}")
    if report.blocking:
        print("\n阻塞项（必须解决后才能进行真实 GPU 验证）:")
        for item in report.blocking:
            print(f"  - {item}")
        return 1
    print("\npreflight 完成：静态 SM 分区已通过实际执行分区命令判定（非 help 文本）；"
          "容器设备透传 / MPS 接入等能力标记为 unknown，需由 smoke 执行证明。")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cases = _expand(args)
    estimate = matrixmod.estimate_duration_s(cases)
    base = cases[0].config if cases else _load_cfg(args)
    budget_msg = matrixmod.check_budget(estimate, base["measure.total_budget_s"])
    print(f"展开用例数: {estimate['case_count']}")
    print(f"预计总时长: {estimate['estimated_total_min']} 分钟 "
          f"({estimate['estimated_total_s']} 秒)")
    print(f"{'case':28} {'family':11} {'mode':20} {'rounds':>6} {'est_s':>8}")
    for row in estimate["cases"]:
        print(f"{row['case']:28} {row['family']:11} {row['mode']:20} "
              f"{row['rounds']:>6} {row['estimated_s']:>8}")
    if args.print_config and cases:
        print("\n--- 首个用例的最终展开配置 ---")
        print(cases[0].config.dump_yaml())
        print(f"config_hash = {cases[0].config.hash()}")
    if budget_msg:
        print("\n" + budget_msg)
        return 1
    print("\nplan 不执行任何操作。")
    return 0


def _prepare_run(args: argparse.Namespace, cfg: cfgmod.LoadedConfig,
                 log: CommandLog) -> "tuple[RunDirectory, Manifest]":
    run_id = cfg["meta.run_id"] or new_run_id()
    rd = RunDirectory(root=cfg["paths.results_dir"], run_id=run_id).create()
    rd.write_text("effective_config.yaml", cfg.dump_yaml())
    manifest = Manifest(run_id=run_id, run_dir=rd.path, config_hash=cfg.hash(),
                        seed=cfg["measure.seed"],
                        source=source_commit(REPO_ROOT),
                        image=image_identity(cfg["docker.binary"], cfg["docker.image"]))
    return rd, manifest


def _static_partition_probe(pre: Any) -> Dict[str, Any]:
    """Pull the active-probe evidence out of the preflight report."""
    for cap in pre.capabilities:
        if cap.name == "mps_static_partitioning":
            evidence = cap.evidence if isinstance(cap.evidence, dict) else {}
            raw = evidence.get("probe")
            probe: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
            probe["status"] = cap.status
            probe["detail"] = cap.detail
            return probe
    return {}


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import Runner, StaticPartitioningUnavailable  # local import: keeps `schema`/`plan` light

    cases = _expand(args)
    if not cases:
        raise SystemExit("没有可执行的用例")
    base = cases[0].config
    estimate = matrixmod.estimate_duration_s(cases)
    budget_msg = matrixmod.check_budget(estimate, base["measure.total_budget_s"])
    print(f"将执行 {estimate['case_count']} 个用例，预计 {estimate['estimated_total_min']} 分钟")
    if budget_msg:
        print(budget_msg)
        return 1
    if args.dry_run:
        print("dry-run：仅打印将要执行的 docker argv，不启动任何容器")

    log = CommandLog()
    rd, manifest = _prepare_run(args, base, log)
    print(f"结果目录: {rd.path}")

    if not base["gpu.uuid"] and not args.dry_run:
        raise SystemExit("gpu.uuid 未指定：真实 GPU 执行前必须由用户显式指定目标 GPU UUID")

    # capability + environment snapshot for this run
    pre = run_preflight(base, log=log)
    rd.write_json("capability.json", pre.as_dict())
    rd.write_json("environment.json", {"environment": pre.environment})
    if pre.blocking and not args.dry_run and not args.ignore_preflight_blocking:
        print("preflight 存在阻塞项，拒绝执行：")
        for item in pre.blocking:
            print(f"  - {item}")
        return 1
    static_probe = _static_partition_probe(pre)

    collector = None
    lock = None
    results: List[Dict[str, Any]] = []
    try:
        if not args.dry_run:
            lock = GpuLock(lock_dir=base["safety.lock_dir"], gpu_uuid=base["gpu.uuid"],
                           run_id=rd.run_id)
            lock.acquire()
        if base["telemetry.enabled"] and not args.dry_run:
            from .telemetry.collector import Collector
            collector = Collector(gpu_uuid=base["gpu.uuid"],
                                  csv_path=rd.file("gpu_metrics.csv"),
                                  interval_s=base["telemetry.sample_interval_s"],
                                  nvidia_smi=base["telemetry.nvidia_smi_binary"],
                                  dcgmi=base["telemetry.dcgmi_binary"],
                                  dcgm_fields=base["telemetry.dcgm_fields"],
                                  require_sm_metrics=base["telemetry.require_sm_metrics"],
                                  allow_degraded=base["telemetry.allow_degraded"],
                                  log=log)
            collector.probe_dcgm()
            collector.start()

        for case in cases:
            cfg = case.config
            gate_disruptive(cfg["case.fault"], args.allow_disruptive,
                            cfg["faults.allow_disruptive"], args.confirm_gpu_uuid,
                            cfg["gpu.uuid"])
            runner = Runner(cfg, rd, log=log, dry_run=args.dry_run,
                            capability_report=static_probe)
            total_bytes = None
            if not args.dry_run:
                gpu = runner.gpu.get(cfg["gpu.uuid"])
                total_bytes = gpu.memory_total_bytes if gpu else None

            # A case that needs static SM partitioning on a machine that cannot
            # provide it is SKIPped with the concrete reason. It is never
            # downgraded to ACTIVE_THREAD_PERCENTAGE / affinity / MIG.
            if cfg["mps.static_partitioning.enabled"] and not args.dry_run \
                    and static_probe.get("status") == "unsupported":
                reason = static_probe.get("detail") or "静态 SM 分区不可用"
                print(f"[{case.full_id}] SKIP：{reason}")
                results.append({"case": case.full_id, "case_file": case.case_file,
                                "overrides": case.overrides,
                                "config_hash": cfg.hash(),
                                "status": "SKIP", "skip_reason": reason,
                                "static_partition_probe": static_probe,
                                "rounds": []})
                continue

            case_rounds = []
            try:
                for round_index in range(cfg["measure.repeats"]):
                    print(f"[{case.full_id}] round {round_index + 1}/{cfg['measure.repeats']}")
                    round_result = runner.run_round(case.full_id, round_index, collector,
                                                    total_bytes)
                    case_rounds.append(round_result)
                    if args.dry_run:
                        break
            except StaticPartitioningUnavailable as exc:
                # Discovered at run time (e.g. the partition could not be
                # carved). Report it, keep whatever rounds already completed.
                print(f"[{case.full_id}] SKIP：{exc}")
                results.append({"case": case.full_id, "case_file": case.case_file,
                                "overrides": case.overrides,
                                "config_hash": cfg.hash(),
                                "status": "SKIP", "skip_reason": str(exc),
                                "static_partition_probe": static_probe,
                                "rounds": []})
                continue
            results.append({"case": case.full_id, "case_file": case.case_file,
                            "overrides": case.overrides,
                            "config_hash": cfg.hash(),
                            "rounds": [{"round": r.round_index,
                                        "notes": r.notes,
                                        "healthy": r.healthy,
                                        "mps_evidence": r.mps_evidence,
                                        "stats": {k: v.as_dict() for k, v in
                                                  r.per_client_stats.items()}}
                                       for r in case_rounds]})
            for r in case_rounds:
                for rec in r.records:
                    rd.append_jsonl("requests.jsonl", rec)
    finally:
        telemetry_status = {}
        if collector is not None:
            telemetry_status = collector.stop().as_dict()
        if lock is not None:
            lock.release()
        manifest.commands = log.records
        manifest.cases = results
        manifest.files = {"requests": "requests.jsonl", "metrics": "gpu_metrics.csv",
                          "events": "events.jsonl"}
        manifest.notes.append("smoke profile 仅验链路，不作为性能证据")
        rd.write_json("manifest.json", manifest.as_dict())
        rd.write_json("telemetry_status.json", telemetry_status)
        rd.write_json("case_results.json", results)

    print(f"\n完成。运行 `mps-bench report --run-dir {rd.path}` 生成报告。")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .reporting.report import render_html, write_summary_csv
    from .summarize import build_summary

    run_dir = args.run_dir
    built = build_summary(run_dir)
    write_summary_csv(os.path.join(run_dir, "summary.csv"), built["summary_rows"])
    path = render_html(run_dir=run_dir, run_id=built["run_id"],
                       summary_rows=built["summary_rows"],
                       mvp_status=built["mvp_status"], events=built["events"],
                       capability=built["capability"], environment=built["environment"],
                       telemetry_status=built["telemetry_status"],
                       findings=built["findings"],
                       fault_summary=built["fault_summary"],
                       memory_summary=built["memory_summary"],
                       not_verified=built["not_verified"])
    print(f"summary.csv -> {os.path.join(run_dir, 'summary.csv')}")
    print(f"report.html -> {path}")
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    log = CommandLog()
    docker = DockerClient(binary=cfg["docker.binary"], project=cfg["meta.project"], log=log)
    containers = docker.owned_containers(run_id=args.run_id)
    print(f"本项目容器: {containers or '无'}")
    mps = None
    if args.stop_mps:
        mps = MpsController(pipe_dir=cfg["paths.mps_pipe_dir"],
                            log_dir=cfg["paths.mps_log_dir"],
                            binary=cfg["mps.control_binary"], log=log)
        # `owns_daemon` is False here, so stop_daemon will refuse: stopping an MPS
        # instance we did not start requires an explicit, separate decision.
        print("注意：cleanup 不会停止非本 run 启动的 MPS 实例")
    report = do_cleanup(docker, containers, mps, stop_timeout_s=30)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    lock_path = None
    if cfg["gpu.uuid"]:
        lock = GpuLock(lock_dir=cfg["safety.lock_dir"], gpu_uuid=cfg["gpu.uuid"],
                       run_id=args.run_id or "")
        lock_path = lock.path
    print(f"如确认无运行中实验，可手动删除锁文件: {lock_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mps-bench",
                                 description="NVIDIA MPS MVP 验证工具（宿主机编排器）")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("schema", help="导出配置 schema")
    p.add_argument("--markdown", action="store_true",
                   help="输出 Markdown 表格（用于生成 docs/CONFIG.zh-CN.md）")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("preflight", help="只读预检查，输出 capability.json")
    _common_args(p)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("plan", help="展开用例并打印预计时长，不执行")
    _common_args(p)
    p.add_argument("--print-config", action="store_true")
    p.set_defaults(func=cmd_plan)

    for name, help_text in (("run", "执行用例"),
                            ("smoke", "仅验链路（等价 run --profile smoke）"),
                            ("memory", "显存配额专项（等价 run --family memory）"),
                            ("faults", "故障用例（破坏性用例需额外确认）")):
        p = sub.add_parser(name, help=help_text)
        _common_args(p)
        p.add_argument("--dry-run", action="store_true",
                       help="只打印将执行的 docker argv，不启动容器")
        p.add_argument("--allow-disruptive", action="store_true")
        p.add_argument("--confirm-gpu-uuid", default=None,
                       help="破坏性用例必须与 gpu.uuid 完全一致")
        p.add_argument("--ignore-preflight-blocking", action="store_true",
                       help="仅用于排查；会在报告中标记证据不足")
        p.set_defaults(func=cmd_run, subcommand=name)

    p = sub.add_parser("report", help="从结果目录生成 summary.csv 与 report.html")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("cleanup", help="幂等清理本项目容器/分区")
    _common_args(p)
    p.add_argument("--run-id", default=None)
    p.add_argument("--stop-mps", action="store_true")
    p.set_defaults(func=cmd_cleanup)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    # subcommand sugar: smoke/memory/faults are `run` with a preset filter
    sub = getattr(args, "subcommand", None)
    if sub == "smoke":
        args.profile = "smoke"
    elif sub == "memory" and not args.family:
        args.family = ["memory"]
    elif sub == "faults" and not args.family:
        args.family = ["faults"]
    try:
        return int(args.func(args))
    except (cfgmod.ConfigError, SafetyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断：请运行 `mps-bench cleanup` 确认无残留容器/MPS", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
