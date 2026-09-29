"""Experiment runner: one case -> N rounds -> per-round records + telemetry.

Execution shape per round (concurrent case):
  ensure ownership -> (MPS start if needed) -> start 2 containers (1 CUDA worker
  each) -> verify MPS attachment evidence -> warmup -> aligned measurement window
  with 2 separate CPU load generators -> drain -> stop -> health check.

Everything that cannot be proven is recorded as unknown. No performance number is
ever synthesized here: if a step did not run, the field stays null.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfgmod
from . import faults as faultmod
from .control import docker as dockermod
from .control.cleanup import cleanup as do_cleanup
from .control.exec import CommandLog, run
from .control.gpu import GpuQuery
from .control.mps import MpsController, client_env, memory_limit_bytes
from .control.pidmap import resolve_worker_host_pid
from .control.safety import SafetyError, check_gpu_processes
from .reporting import stats as statsmod
from .reporting.results import RunDirectory
from .telemetry.collector import Collector, rolling_summary

CONTAINER_RESULTS_MOUNT = "/results"
ASSETS_MOUNT = "/assets"


@dataclass
class ClientHandle:
    slot: str
    role: str
    container: str
    port: int
    workload: str
    unit: str
    env: Dict[str, str]
    host_results_dir: str
    worker_container_pid: Optional[int] = None
    worker_host_pid: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/infer"


@dataclass
class RoundResult:
    case_id: str
    round_index: int
    records: List[Dict[str, Any]] = field(default_factory=list)
    per_client_stats: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    mps_evidence: Dict[str, Any] = field(default_factory=dict)
    healthy: bool = True
    notes: List[str] = field(default_factory=list)


class Runner:
    def __init__(self, cfg: cfgmod.LoadedConfig, run_dir: RunDirectory,
                 log: Optional[CommandLog] = None, dry_run: bool = False):
        self.cfg = cfg
        self.run_dir = run_dir
        self.log = log or CommandLog()
        self.dry_run = dry_run
        self.run_id = run_dir.run_id
        self.gpu_uuid = cfg["gpu.uuid"]
        self.docker = dockermod.DockerClient(binary=cfg["docker.binary"],
                                            project=cfg["meta.project"], log=self.log)
        self.gpu = GpuQuery(binary=cfg["telemetry.nvidia_smi_binary"], log=self.log)
        self.mps: Optional[MpsController] = None
        self.started_containers: List[str] = []
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #
    def event(self, name: str, label: str = "", **payload: Any) -> Dict[str, Any]:
        rec = {"event": name, "label": label or name,
               "monotonic_s": round(time.monotonic() - self._t0, 4),
               "wall_ts": time.time(), **payload}
        self.run_dir.append_jsonl("events.jsonl", rec)
        return rec

    # ------------------------------------------------------------------ #
    # safety
    # ------------------------------------------------------------------ #
    def assert_gpu_ownership(self) -> Dict[str, Any]:
        """Re-checked before EVERY round: the initial screenshot is not runtime
        state."""
        if not self.gpu_uuid:
            raise SafetyError("gpu.uuid 未指定，拒绝执行真实 GPU 操作")
        owned = self._owned_host_pids()
        procs = self.gpu.compute_processes(self.gpu_uuid)
        check = check_gpu_processes(procs, owned_pids=owned, gpu_uuid=self.gpu_uuid)
        self.event("gpu_ownership_check", "ownership", **check.as_dict())
        if not check.ok and self.cfg["safety.abort_on_foreign_gpu_process"]:
            raise SafetyError(check.reason + f" 详情: {check.foreign}")
        return check.as_dict()

    def _owned_host_pids(self) -> List[int]:
        pids: List[int] = []
        for name in self.started_containers:
            try:
                pids += self.docker.top_pids(name)
            except Exception:
                continue
        return pids

    # ------------------------------------------------------------------ #
    # MPS lifecycle
    # ------------------------------------------------------------------ #
    def start_mps_if_needed(self, mps_mode: bool) -> Dict[str, Any]:
        if not mps_mode:
            self.event("mps_skipped", "非 MPS 模式")
            return {"enabled": False,
                    "note": "非 MPS 对照：客户端不设置 CUDA_MPS_PIPE_DIRECTORY，"
                            "避免误连默认 pipe；也不停止其他实验/业务的 MPS"}
        ctl = MpsController(pipe_dir=self.cfg["paths.mps_pipe_dir"],
                            log_dir=self.cfg["paths.mps_log_dir"],
                            binary=self.cfg["mps.control_binary"],
                            gpu_uuid=self.gpu_uuid, log=self.log)
        info = ctl.start_daemon(timeout_s=self.cfg["mps.start_timeout_s"],
                                allow_adopt_existing=self.cfg["mps.allow_adopt_existing"])
        self.mps = ctl
        partitions: Dict[str, Any] = {"requested": False}
        if self.cfg["mps.static_partitioning.enabled"]:
            partitions = ctl.create_partitions(self.cfg["mps.static_partitioning.partitions"])
            partitions["requested"] = True
        self.event("mps_started", "MPS start", **{"daemon": info, "partitions": partitions})
        return {"enabled": True, "daemon": info, "partitions": partitions,
                "capabilities": ctl.capabilities().commands}

    # ------------------------------------------------------------------ #
    # container launch
    # ------------------------------------------------------------------ #
    def _client_env(self, slot: str, mps_mode: bool, total_bytes: Optional[int]) -> Tuple[Dict[str, str], Optional[int]]:
        cfg, p = self.cfg, f"clients.{slot}"
        if not mps_mode:
            # Non-MPS control must PROVE it is not connected to MPS: we neither
            # set a pipe directory nor any MPS client env var.
            return {"MPS_BENCH_MPS_MODE": "disabled"}, None
        fraction = cfg[f"{p}.mps.pinned_device_mem_limit_fraction"]
        limit_bytes = None
        if fraction is not None:
            if not total_bytes:
                raise SafetyError("无法获取运行时物理总显存，拒绝按比例生成显存配额")
            limit_bytes = memory_limit_bytes(total_bytes, float(fraction))
        env = client_env(pipe_dir=cfg["paths.mps_pipe_dir"],
                         log_dir=cfg["paths.mps_log_dir"],
                         active_thread_percentage=cfg[f"{p}.mps.active_thread_percentage"],
                         priority=cfg[f"{p}.mps.priority"],
                         pinned_device_mem_limit_bytes=limit_bytes,
                         gpu_uuid=self.gpu_uuid,
                         static_partition=cfg[f"{p}.mps.static_partition"])
        env["MPS_BENCH_MPS_MODE"] = "enabled"
        return env, limit_bytes

    def _worker_command(self, slot: str) -> List[str]:
        cfg, p = self.cfg, f"clients.{slot}"
        cmd = ["python", "-m", "mps_bench.workloads.worker_main",
               "--workload", cfg[f"{p}.workload"],
               "--port", str(cfg[f"{p}.load.port"]),
               "--queue-limit", str(cfg[f"{p}.load.queue_limit"]),
               "--request-timeout-s", str(cfg["measure.request_timeout_s"]),
               "--tolerance", str(cfg["measure.tolerance"]),
               "--drain-timeout-s", str(cfg["faults.drain_timeout_s"]),
               "--seed", str(cfg["measure.seed"]),
               "--batch-size", str(cfg[f"{p}.model.batch_size"]),
               "--precision", cfg[f"{p}.model.precision"],
               "--input-pool-size", str(cfg[f"{p}.model.input_pool_size"]),
               "--cpu-threads", str(cfg[f"{p}.model.cpu_threads"]),
               "--status-path", f"{CONTAINER_RESULTS_MOUNT}/worker_status.json",
               "--evidence-path", f"{CONTAINER_RESULTS_MOUNT}/worker_evidence.json"]
        if cfg[f"{p}.workload"] == "image_inference":
            cmd += ["--model-name", cfg[f"{p}.model.name"],
                    "--input-size", str(cfg[f"{p}.model.input_size"])]
            if cfg[f"{p}.model.weights_path"]:
                cmd += ["--weights-path", cfg[f"{p}.model.weights_path"]]
        else:
            cmd += ["--num-tables", str(cfg[f"{p}.recsys.num_tables"]),
                    "--rows-per-table", str(cfg[f"{p}.recsys.rows_per_table"]),
                    "--embedding-dim", str(cfg[f"{p}.recsys.embedding_dim"]),
                    "--indices-per-sample", str(cfg[f"{p}.recsys.indices_per_sample"]),
                    "--dense-features", str(cfg[f"{p}.recsys.dense_features"]),
                    "--bottom-mlp", json.dumps(cfg[f"{p}.recsys.bottom_mlp"]),
                    "--top-mlp", json.dumps(cfg[f"{p}.recsys.top_mlp"]),
                    "--index-distribution", cfg[f"{p}.recsys.index_distribution"],
                    "--zipf-s", str(cfg[f"{p}.recsys.zipf_s"])]
            if cfg[f"{p}.recsys.working_set_rows"]:
                cmd += ["--working-set-rows", str(cfg[f"{p}.recsys.working_set_rows"])]
        return cmd

    def start_client(self, slot: str, mps_mode: bool, round_index: int,
                     total_bytes: Optional[int]) -> ClientHandle:
        cfg, p = self.cfg, f"clients.{slot}"
        dockermod.requires_explicit_device(cfg["docker.runtime_args"], self.gpu_uuid)
        env, limit_bytes = self._client_env(slot, mps_mode, total_bytes)

        host_results = os.path.abspath(
            os.path.join(self.run_dir.path, "workload_logs", f"r{round_index}-{slot}")
        )
        os.makedirs(host_results, exist_ok=True)
        mounts = [{"source": host_results, "target": CONTAINER_RESULTS_MOUNT}]
        if mps_mode:
            # the container shares this experiment's pipe path (not the default)
            mounts.append({"source": cfg["paths.mps_pipe_dir"],
                           "target": cfg["paths.mps_pipe_dir"]})
            mounts.append({"source": cfg["paths.mps_log_dir"],
                           "target": cfg["paths.mps_log_dir"]})
        if cfg["paths.model_mount"]:
            mounts.append({"source": cfg["paths.model_mount"], "target": ASSETS_MOUNT,
                           "ro": True})

        name = f"{cfg['meta.project']}-{self.run_id}-r{round_index}-{slot}"
        spec = dockermod.ContainerSpec(
            name=name, image=cfg["docker.image"], slot=slot, role=cfg[f"{p}.role"],
            command=self._worker_command(slot), env=env, mounts=mounts,
            ports=[{"host": cfg[f"{p}.load.port"], "container": cfg[f"{p}.load.port"],
                    "bind": "127.0.0.1"}],
            cpuset=cfg[f"{p}.container.cpuset"], memory=cfg[f"{p}.container.memory"],
            # online entry point is bound to loopback only (experiment network)
            network="bridge", user=cfg["docker.user"], ipc=cfg["docker.ipc"],
            shm_size=cfg["docker.shm_size"],
            runtime_args=cfg["docker.runtime_args"], extra_args=cfg["docker.extra_args"],
            labels=dockermod.make_labels(cfg["meta.project"], self.run_id, slot,
                                        cfg[f"{p}.role"], self.gpu_uuid))
        argv = dockermod.build_run_argv(spec, cfg["docker.binary"])
        if self.dry_run:
            return ClientHandle(slot=slot, role=cfg[f"{p}.role"], container=name,
                                port=cfg[f"{p}.load.port"], workload=cfg[f"{p}.workload"],
                                unit="images" if cfg[f"{p}.workload"] == "image_inference"
                                else "samples",
                                env=env, host_results_dir=host_results,
                                evidence={"dry_run_argv": argv,
                                          "mps_mem_limit_bytes": limit_bytes})
        res = self.docker.start(spec)
        if not res.ok:
            raise SafetyError(f"容器 {name} 启动失败: {res.stderr.strip()[:800]}")
        self.started_containers.append(name)
        self.event("container_started", f"start {slot}", container=name, slot=slot,
                   mps_mode=mps_mode, mps_mem_limit_bytes=limit_bytes)
        return ClientHandle(slot=slot, role=cfg[f"{p}.role"], container=name,
                            port=cfg[f"{p}.load.port"], workload=cfg[f"{p}.workload"],
                            unit="images" if cfg[f"{p}.workload"] == "image_inference"
                            else "samples",
                            env=env, host_results_dir=host_results,
                            evidence={"mps_mem_limit_bytes": limit_bytes,
                                      "docker_argv": argv})

    # ------------------------------------------------------------------ #
    # readiness + evidence
    # ------------------------------------------------------------------ #
    def wait_ready(self, handle: ClientHandle, timeout_s: float) -> Dict[str, Any]:
        import urllib.request
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{handle.port}/health"
        last = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    last = json.loads(resp.read() or b"{}")
                    if last.get("state") == "ready":
                        return {"ready": True, "health": last}
            except Exception as exc:
                last = {"error": str(exc)[:200]}
            time.sleep(1.0)
        return {"ready": False, "last": last}

    def collect_client_evidence(self, handle: ClientHandle, mps_mode: bool) -> Dict[str, Any]:
        """Prove (or fail to prove) that this worker is really an MPS client.

        Setting env vars, seeing the daemon alive, or seeing the GPU busy are all
        insufficient. We require: the container's visible device matches the
        target physical UUID, and the worker's host PID appears in the MPS client
        list for a server we started.
        """
        evidence: Dict[str, Any] = {"container": handle.container}
        path = os.path.join(handle.host_results_dir, "worker_evidence.json")
        worker_ev: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    worker_ev = json.load(fh)
            except (OSError, json.JSONDecodeError):
                worker_ev = {}
        evidence["worker_runtime"] = worker_ev
        handle.worker_container_pid = worker_ev.get("worker_pid")

        device_uuid = ((worker_ev.get("device") or {}).get("uuid") or "")
        evidence["visible_device_uuid"] = device_uuid or None
        if self.gpu_uuid and device_uuid:
            normalized = device_uuid.replace("UUID('", "").replace("')", "")
            match = self.gpu_uuid.replace("GPU-", "") in normalized.replace("GPU-", "")
            evidence["device_uuid_matches_target"] = match
        else:
            evidence["device_uuid_matches_target"] = None
            evidence["device_uuid_note"] = ("torch 未暴露设备 UUID：无法在容器内直接核对物理 UUID，"
                                            "以宿主机透传参数 + 单设备可见性为间接证据")

        host_pids: List[int] = []
        try:
            host_pids = self.docker.top_pids(handle.container)
        except Exception as exc:
            evidence["docker_top_error"] = str(exc)[:200]

        if mps_mode and self.mps is not None:
            servers = self.mps.server_pids()
            client_pids = [pid for s in servers for pid in self.mps.client_pids(s)]
            resolution = resolve_worker_host_pid(host_pids, handle.worker_container_pid,
                                                 client_pids)
            handle.worker_host_pid = resolution.host_pid
            evidence["mps"] = {"server_pids": servers, "client_pids": client_pids,
                               "pid_resolution": resolution.as_dict(),
                               "server_state": self.mps.server_state()}
            evidence["mps_attachment_proven"] = bool(
                resolution.host_pid is not None and resolution.host_pid in set(client_pids))
            if not evidence["mps_attachment_proven"]:
                evidence["mps_attachment_note"] = (
                    "未能证明该 worker 已接入 MPS：仅环境变量/daemon 存在/GPU busy 均不算证据")
        else:
            evidence["mps"] = {"enabled": False}
            # A non-MPS control must demonstrate the absence of MPS env vars.
            evidence["nonmps_no_mps_env"] = all(
                not worker_ev.get(key) for key in
                ("mps_pipe_directory", "mps_active_thread_percentage",
                 "mps_client_priority", "mps_pinned_device_mem_limit"))
        handle.evidence.update(evidence)
        return evidence

    # ------------------------------------------------------------------ #
    # load generation
    # ------------------------------------------------------------------ #
    def _loadgen_argv(self, handle: ClientHandle, round_index: int, case_id: str,
                      duration_s: float, start_at_wall: float) -> List[str]:
        cfg, p = self.cfg, f"clients.{handle.slot}"
        out = os.path.join(handle.host_results_dir, f"requests-{round_index}.jsonl")
        summary = os.path.join(handle.host_results_dir, f"loadgen-{round_index}.json")
        argv = ["python", "-m", "mps_bench.loadgen.main",
                "--run-id", self.run_id, "--case-id", case_id,
                "--round-index", str(round_index), "--worker-id", handle.slot,
                "--url", handle.url, "--mode", cfg[f"{p}.mode"],
                "--arrival", cfg[f"{p}.load.arrival"],
                "--duration-s", str(duration_s),
                "--concurrency", str(cfg[f"{p}.load.concurrency"]),
                "--request-timeout-s", str(cfg["measure.request_timeout_s"]),
                "--drain-timeout-s", str(cfg["measure.drain_timeout_s"]),
                "--unit", handle.unit,
                # Same seed per (case, round, slot) -> identical frozen trace
                # across modes. A slower service can never plan fewer requests.
                "--seed", str(cfg["measure.seed"] + round_index * 1000 +
                              (0 if handle.slot == "a" else 1)),
                "--out", out, "--summary-out", summary,
                "--start-at-wall", str(start_at_wall)]
        if cfg[f"{p}.load.target_qps"] is not None:
            argv += ["--qps", str(cfg[f"{p}.load.target_qps"])]
        if cfg[f"{p}.load.burst_trace"]:
            argv += ["--burst-trace", json.dumps(cfg[f"{p}.load.burst_trace"])]
        return argv

    def run_load(self, handles: List[ClientHandle], round_index: int, case_id: str,
                 duration_s: float) -> Dict[str, Any]:
        """Start all load generators aligned on a common wall-clock start."""
        import subprocess
        start_at_wall = time.time() + 3.0
        procs = []
        for handle in handles:
            argv = self._loadgen_argv(handle, round_index, case_id, duration_s, start_at_wall)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] +
                ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            procs.append((handle, subprocess.Popen(argv, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE, text=True, env=env)))
        self.event("measurement_start", "measure", round_index=round_index, case_id=case_id,
                   start_at_wall=start_at_wall, duration_s=duration_s)
        outcomes: Dict[str, Any] = {}
        timeout = duration_s + self.cfg["measure.drain_timeout_s"] + 60
        for handle, proc in procs:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            outcomes[handle.slot] = {"returncode": proc.returncode,
                                     "stdout_tail": (out or "")[-2000:],
                                     "stderr_tail": (err or "")[-2000:]}
        self.event("measurement_end", "measure end", round_index=round_index)
        return outcomes

    def read_records(self, handles: List[ClientHandle], round_index: int) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for handle in handles:
            path = os.path.join(handle.host_results_dir, f"requests-{round_index}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    # ------------------------------------------------------------------ #
    # per-round driver
    # ------------------------------------------------------------------ #
    def run_round(self, case_id: str, round_index: int, collector: Optional[Collector],
                  total_bytes: Optional[int]) -> RoundResult:
        cfg = self.cfg
        mps_mode = cfg["case.mode"].startswith("mps_")
        solo = cfg["case.mode"].endswith("_solo")
        slots = [cfg["case.solo_client"]] if solo else ["a", "b"]
        result = RoundResult(case_id=case_id, round_index=round_index)

        self.assert_gpu_ownership()
        mps_info = self.start_mps_if_needed(mps_mode)
        result.mps_evidence = mps_info

        handles: List[ClientHandle] = []
        try:
            for slot in slots:
                handles.append(self.start_client(slot, mps_mode, round_index, total_bytes))

            if collector is not None:
                collector.set_phase("warmup")
            for handle in handles:
                ready = self.wait_ready(handle, timeout_s=max(120.0, cfg["measure.warmup_s"] * 3))
                if not ready["ready"]:
                    raise SafetyError(f"{handle.container} 未在超时内就绪: {ready.get('last')}")
                self.event("worker_ready", f"ready {handle.slot}", container=handle.container)

            for handle in handles:
                ev = self.collect_client_evidence(handle, mps_mode)
                result.mps_evidence[f"client_{handle.slot}"] = ev
                if mps_mode and ev.get("mps_attachment_proven") is False:
                    result.notes.append(
                        f"client {handle.slot} 未能证明已接入 MPS：本轮结论标记为 INCONCLUSIVE")
                if not mps_mode and ev.get("nonmps_no_mps_env") is False:
                    result.notes.append(
                        f"client {handle.slot} 在非 MPS 对照中检测到 MPS 环境变量：本轮无效")

            # warmup is driven by the workload itself (model load, pool, autotune)
            # plus a short traffic warmup at the same offered load
            if cfg["measure.warmup_s"] > 0:
                self.event("warmup_start", "warmup", round_index=round_index)
                self.run_load(handles, round_index=-round_index - 1, case_id=f"{case_id}-warmup",
                              duration_s=cfg["measure.warmup_s"])

            if collector is not None:
                collector.set_phase("measure")
            self.run_load(handles, round_index, case_id, cfg["measure.duration_s"])
            result.records = self.read_records(handles, round_index)

            for handle in handles:
                st = statsmod.summarize_workload(
                    result.records, worker_id=handle.slot, role=handle.role,
                    unit=handle.unit, window_s=cfg["measure.duration_s"],
                    min_samples_p999=cfg["measure.min_samples_p999"],
                    slo_latency_ms=cfg["slo.latency_p99_ms"])
                result.per_client_stats[handle.slot] = st
        finally:
            if collector is not None:
                collector.set_phase("teardown")
            report = do_cleanup(self.docker, [h.container for h in handles], self.mps,
                                save_logs=lambda name, text: self.run_dir.write_text(
                                    os.path.join("workload_logs", f"{name}.log"), text),
                                stop_timeout_s=int(cfg["measure.drain_timeout_s"]) + 10)
            self.event("round_cleanup", "cleanup", round_index=round_index, **report.as_dict())
            self.run_dir.copy_mps_logs(cfg["paths.mps_log_dir"])
            self.started_containers = [c for c in self.started_containers
                                       if c not in [h.container for h in handles]]
            self.mps = None

        if cfg["safety.health_check_between_rounds"]:
            health = self.health_check()
            result.healthy = health["healthy"]
            self.event("health_check", "health", round_index=round_index, **health)
            if not health["healthy"] and cfg["safety.stop_on_unhealthy"]:
                raise SafetyError(f"轮次结束后 GPU 健康检查未通过，停止后续实验: {health}")
        return result

    def health_check(self) -> Dict[str, Any]:
        """Independent health check between rounds. Not being able to prove
        health stops the experiment rather than auto-retrying over a polluted
        state."""
        if not self.gpu_uuid:
            return {"healthy": False, "reason": "无目标 GPU"}
        gpu = self.gpu.get(self.gpu_uuid)
        if gpu is None:
            return {"healthy": False, "reason": "无法查询目标 GPU"}
        procs = self.gpu.compute_processes(self.gpu_uuid)
        leftovers = [p.as_dict() for p in procs]
        return {"healthy": not leftovers,
                "reason": "" if not leftovers else "目标 GPU 仍存在残留计算进程",
                "memory_used_mib": gpu.memory_used_mib,
                "memory_free_mib": gpu.memory_free_mib,
                "compute_mode": gpu.compute_mode,
                "leftover_processes": leftovers}
