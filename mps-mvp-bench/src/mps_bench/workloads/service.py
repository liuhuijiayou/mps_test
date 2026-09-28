"""Single-worker GPU service (runs inside the workload container).

Concurrency model -- deliberately minimal:
* one HTTP thread pool accepts requests and enqueues them (bounded queue),
* exactly ONE executor thread owns the CUDA context and runs batches.
  This guarantees "one CUDA worker / one primary context per container", which is
  what the MPS per-client quota is mapped onto in this experiment.

Safe-exit protocol (F3):
* the signal handler only sets a flag (no CUDA calls from an async handler),
* the control flow then: fence (stop accepting) -> drain (finish in-flight GPU
  work) -> exit. Normal cleanup never relies on docker's post-timeout SIGKILL.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

STATE_STARTING = "starting"
STATE_WARMUP = "warmup"
STATE_READY = "ready"
STATE_FENCED = "fenced"      # no new work accepted
STATE_DRAINING = "draining"
STATE_EXITING = "exiting"


@dataclass
class Job:
    request_id: str
    sent_monotonic: float
    received_monotonic: float
    enqueued_monotonic: float
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None


class WorkerState:
    def __init__(self) -> None:
        self.state = STATE_STARTING
        self.stop_requested = False      # set by signal handler ONLY
        self.stop_signal: Optional[int] = None
        self.lock = threading.Lock()
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.rejected = 0
        self.inflight = 0
        self.last_error: Optional[str] = None
        self.checksum_mismatches = 0

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {"state": self.state, "submitted": self.submitted,
                    "completed": self.completed, "failed": self.failed,
                    "rejected": self.rejected, "inflight": self.inflight,
                    "stop_requested": self.stop_requested,
                    "stop_signal": self.stop_signal,
                    "checksum_mismatches": self.checksum_mismatches,
                    "last_error": self.last_error}


class GpuExecutor(threading.Thread):
    """The only thread that touches CUDA."""

    def __init__(self, workload, state: WorkerState, jobs: "queue.Queue[Optional[Job]]",
                 tolerance: float):
        super().__init__(name="gpu-executor", daemon=True)
        self.workload = workload
        self.state = state
        self.jobs = jobs
        self.tolerance = tolerance
        self._batch_index = 0

    def run(self) -> None:
        from .base import checksum_tolerance_ok
        reference = self.workload.reference_checksum()
        while True:
            job = self.jobs.get()
            if job is None:  # drain sentinel
                self.jobs.task_done()
                return
            with self.state.lock:
                self.state.inflight += 1
            exec_start = time.monotonic()
            try:
                res = self.workload.run_batch(self._batch_index)
                self._batch_index += 1
                gpu_done = time.monotonic()
                ok_shape = len(res.output_shape) >= 1
                ok_value = checksum_tolerance_ok(reference, res.checksum, self.tolerance)
                if not ok_value:
                    with self.state.lock:
                        self.state.checksum_mismatches += 1
                job.result = {
                    "status": "ok" if (res.finite and ok_shape and ok_value) else "output_error",
                    "exec_start_monotonic": exec_start,
                    "gpu_done_monotonic": gpu_done,
                    "gpu_ms": res.gpu_ms,
                    "samples": res.samples,
                    "batch_size": res.batch_size,
                    "finite": res.finite,
                    "checksum_ok": ok_value,
                    "output_shape": list(res.output_shape),
                }
                with self.state.lock:
                    if job.result["status"] == "ok":
                        self.state.completed += 1
                    else:
                        self.state.failed += 1
            except Exception as exc:  # CUDA or framework error: record the real type
                job.result = {"status": "error", "error_type": type(exc).__name__,
                              "error": str(exc)[:2000],
                              "exec_start_monotonic": exec_start,
                              "gpu_done_monotonic": None, "gpu_ms": None, "samples": 0}
                with self.state.lock:
                    self.state.failed += 1
                    self.state.last_error = f"{type(exc).__name__}: {exc}"[:500]
            finally:
                with self.state.lock:
                    self.state.inflight -= 1
                job.done.set()
                self.jobs.task_done()


def _make_handler(state: WorkerState, jobs: "queue.Queue[Optional[Job]]",
                  info: Dict[str, Any], queue_limit: int, request_timeout_s: float):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence access log
            return

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": state.state in (STATE_READY, STATE_WARMUP),
                                 **state.snapshot()})
            elif self.path == "/info":
                self._send(200, {"info": info, "worker_pid": os.getpid()})
            elif self.path == "/stats":
                self._send(200, state.snapshot())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/infer":
                self._send(404, {"error": "not found"})
                return
            received = time.monotonic()
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"status": "bad_request"})
                return
            if state.state != STATE_READY:
                with state.lock:
                    state.rejected += 1
                self._send(503, {"status": "rejected", "reason": f"worker state={state.state}",
                                 "service_received_monotonic": received})
                return
            job = Job(request_id=str(body.get("request_id", "")),
                      sent_monotonic=float(body.get("sent_monotonic", 0.0)),
                      received_monotonic=received,
                      enqueued_monotonic=time.monotonic())
            try:
                if jobs.qsize() >= queue_limit:
                    raise queue.Full
                jobs.put_nowait(job)
                with state.lock:
                    state.submitted += 1
            except queue.Full:
                with state.lock:
                    state.rejected += 1
                self._send(503, {"status": "rejected", "reason": "queue_full",
                                 "service_received_monotonic": received})
                return
            if not job.done.wait(timeout=request_timeout_s):
                # Timeout is reported, never silently dropped: the request stays
                # in flight on the GPU and is counted as a timeout cohort member.
                self._send(504, {"status": "timeout", "request_id": job.request_id,
                                 "service_received_monotonic": received,
                                 "enqueued_monotonic": job.enqueued_monotonic})
                return
            payload = dict(job.result or {})
            payload.update({"request_id": job.request_id,
                            "service_received_monotonic": received,
                            "enqueued_monotonic": job.enqueued_monotonic,
                            "response_monotonic": time.monotonic(),
                            "wall_ts": time.time()})
            self._send(200, payload)

    return Handler


def serve(workload, host: str, port: int, queue_limit: int, request_timeout_s: float,
          warmup_iterations: int, tolerance: float,
          status_path: Optional[str] = None,
          drain_timeout_s: float = 30.0) -> Dict[str, Any]:
    state = WorkerState()
    jobs: "queue.Queue[Optional[Job]]" = queue.Queue(maxsize=max(queue_limit * 2, 8))

    def _handler(signum, _frame):
        # Async-signal-safe: set a flag only. No CUDA, no logging, no locks.
        state.stop_requested = True
        state.stop_signal = signum

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    state.state = STATE_WARMUP
    workload.warmup(iterations=warmup_iterations)
    info = workload.info().as_dict()
    info["worker_pid"] = os.getpid()

    executor = GpuExecutor(workload, state, jobs, tolerance)
    executor.start()
    state.state = STATE_READY

    server = ThreadingHTTPServer((host, port),
                                 _make_handler(state, jobs, info, queue_limit, request_timeout_s))
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, name="http", daemon=True)
    server_thread.start()

    if status_path:
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump({"worker_pid": os.getpid(), "port": port, "info": info}, fh)

    # main loop: watch the flag set by the signal handler
    while not state.stop_requested:
        time.sleep(0.2)

    # Fence -> Drain -> Exit
    state.state = STATE_FENCED
    server.shutdown()
    state.state = STATE_DRAINING
    drain_deadline = time.monotonic() + drain_timeout_s
    inflight_at_signal = state.snapshot()["inflight"]
    while time.monotonic() < drain_deadline:
        snap = state.snapshot()
        if snap["inflight"] == 0 and jobs.empty():
            break
        time.sleep(0.05)
    drained = state.snapshot()["inflight"] == 0 and jobs.empty()
    jobs.put(None)
    state.state = STATE_EXITING
    summary = {"exit": "safe_exit", "signal": state.stop_signal,
               "inflight_at_signal": inflight_at_signal, "drained": drained,
               **state.snapshot()}
    if status_path:
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump({"worker_pid": os.getpid(), "port": port, "info": info,
                       "exit_summary": summary}, fh)
    return summary
