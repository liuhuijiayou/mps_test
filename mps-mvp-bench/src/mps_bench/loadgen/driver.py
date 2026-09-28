"""Load driver: open-loop (latency) and closed-loop (capacity).

Runs in a separate CPU process; no CUDA context is ever created here.

Open loop: requests are issued at their planned times from a frozen trace. If the
sender is saturated, the request is still recorded (with its send delay, or as
`not_sent`) instead of silently reducing the offered load.

Closed loop: N workers keep the service saturated for the whole window. Latency
from this mode is NOT comparable to open-loop latency and is reported separately.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .arrival import ArrivalTrace
from .record import (STATUS_ERROR, STATUS_INCOMPLETE, STATUS_NOT_SENT, STATUS_OK,
                     STATUS_OUTPUT_ERROR, STATUS_REJECTED, STATUS_TIMEOUT, RequestRecord)


@dataclass
class DriverConfig:
    run_id: str
    case_id: str
    round_index: int
    worker_id: str
    url: str
    unit: str = "samples"
    request_timeout_s: float = 5.0
    concurrency: int = 8
    drain_timeout_s: float = 30.0


@dataclass
class DriverOutcome:
    records: List[RequestRecord] = field(default_factory=list)
    measurement_start_wall: float = 0.0
    measurement_start_monotonic: float = 0.0
    measurement_end_monotonic: float = 0.0
    planned_count: int = 0
    notes: List[str] = field(default_factory=list)


def _post(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return {"http_status": resp.status, "body": json.loads(resp.read() or b"{}")}
    except urllib.error.HTTPError as exc:
        body = {}
        try:
            body = json.loads(exc.read() or b"{}")
        except Exception:
            pass
        return {"http_status": exc.code, "body": body}
    except Exception as exc:  # socket timeout, connection refused, ...
        return {"http_status": None, "error_type": type(exc).__name__, "error": str(exc)[:500]}


def _fill_from_response(rec: RequestRecord, resp: Dict[str, Any], t0: float) -> None:
    body = resp.get("body") or {}
    status_code = resp.get("http_status")
    rec.wall_ts = body.get("wall_ts")
    for src, dst in (("service_received_monotonic", "service_received_s"),
                     ("enqueued_monotonic", "enqueued_s"),
                     ("exec_start_monotonic", "exec_start_s"),
                     ("gpu_done_monotonic", "gpu_done_s"),
                     ("response_monotonic", "response_done_s")):
        value = body.get(src)
        # Service-side monotonic clocks come from the same host, so they share a
        # base; we still normalize against this round's t0.
        setattr(rec, dst, None if value is None else value - t0)
    rec.gpu_ms = body.get("gpu_ms")
    rec.samples = int(body.get("samples") or 0)
    rec.batch_size = int(body.get("batch_size") or 0)

    if status_code is None:
        rec.status = STATUS_ERROR
        rec.error_type = resp.get("error_type")
        rec.error = resp.get("error")
    elif status_code == 200:
        body_status = body.get("status", STATUS_OK)
        rec.status = STATUS_OK if body_status == "ok" else STATUS_OUTPUT_ERROR
        if body_status not in ("ok", "output_error"):
            rec.status = STATUS_ERROR
            rec.error = str(body_status)
    elif status_code == 503:
        rec.status = STATUS_REJECTED
        rec.error = str(body.get("reason"))
    elif status_code == 504:
        rec.status = STATUS_TIMEOUT
    else:
        rec.status = STATUS_ERROR
        rec.error = f"http {status_code}"
    if rec.response_done_s is None and rec.status != STATUS_NOT_SENT:
        rec.response_done_s = time.monotonic() - t0


def run_open_loop(cfg: DriverConfig, trace: ArrivalTrace,
                  measurement_start_monotonic: Optional[float] = None) -> DriverOutcome:
    """Issue the frozen trace. Never reduces the planned load."""
    out = DriverOutcome(planned_count=len(trace.planned))
    t0 = measurement_start_monotonic if measurement_start_monotonic is not None else time.monotonic()
    out.measurement_start_monotonic = t0
    out.measurement_start_wall = time.time()

    lock = threading.Lock()
    slots = threading.Semaphore(max(1, cfg.concurrency))
    threads: List[threading.Thread] = []

    def _issue(index: int, planned: float) -> None:
        rec = RequestRecord(run_id=cfg.run_id, case_id=cfg.case_id, round_index=cfg.round_index,
                            worker_id=cfg.worker_id, request_id=f"{cfg.worker_id}-{index}",
                            planned_arrival_s=planned, unit=cfg.unit)
        rec.actual_send_s = time.monotonic() - t0
        resp = _post(cfg.url, {"request_id": rec.request_id, "sent_monotonic": time.monotonic()},
                     cfg.request_timeout_s)
        _fill_from_response(rec, resp, t0)
        with lock:
            out.records.append(rec)
        slots.release()

    for index, planned in enumerate(trace.planned):
        now = time.monotonic() - t0
        if planned > now:
            time.sleep(planned - now)
        # Bounded sender: if every slot is busy we wait, and the resulting
        # lateness shows up in send_delay_s rather than disappearing.
        acquired = slots.acquire(timeout=cfg.request_timeout_s)
        if not acquired:
            rec = RequestRecord(run_id=cfg.run_id, case_id=cfg.case_id,
                                round_index=cfg.round_index, worker_id=cfg.worker_id,
                                request_id=f"{cfg.worker_id}-{index}",
                                planned_arrival_s=planned, unit=cfg.unit,
                                status=STATUS_NOT_SENT,
                                error="压测端并发槽位耗尽，未能按计划发送（记录而不静默降载）")
            with lock:
                out.records.append(rec)
            continue
        th = threading.Thread(target=_issue, args=(index, planned), daemon=True)
        th.start()
        threads.append(th)

    out.measurement_end_monotonic = time.monotonic()
    deadline = time.monotonic() + cfg.drain_timeout_s
    for th in threads:
        remaining = deadline - time.monotonic()
        th.join(timeout=max(0.0, remaining))
    still_running = sum(1 for th in threads if th.is_alive())
    if still_running:
        out.notes.append(f"drain 超时：{still_running} 个请求在 {cfg.drain_timeout_s}s 后仍未完成，"
                         "按 incomplete 计数")
        for _ in range(still_running):
            out.records.append(RequestRecord(
                run_id=cfg.run_id, case_id=cfg.case_id, round_index=cfg.round_index,
                worker_id=cfg.worker_id, request_id=f"{cfg.worker_id}-incomplete",
                unit=cfg.unit, status=STATUS_INCOMPLETE))
    return out


def run_closed_loop(cfg: DriverConfig, duration_s: float,
                    measurement_start_monotonic: Optional[float] = None) -> DriverOutcome:
    """Saturate the service for `duration_s` with `concurrency` senders."""
    out = DriverOutcome()
    t0 = measurement_start_monotonic if measurement_start_monotonic is not None else time.monotonic()
    out.measurement_start_monotonic = t0
    out.measurement_start_wall = time.time()
    stop_at = t0 + duration_s
    lock = threading.Lock()
    counter = {"n": 0}

    def _worker(slot: int) -> None:
        while time.monotonic() < stop_at:
            with lock:
                index = counter["n"]
                counter["n"] += 1
            rec = RequestRecord(run_id=cfg.run_id, case_id=cfg.case_id,
                                round_index=cfg.round_index, worker_id=cfg.worker_id,
                                request_id=f"{cfg.worker_id}-cl-{index}", unit=cfg.unit)
            rec.actual_send_s = time.monotonic() - t0
            resp = _post(cfg.url, {"request_id": rec.request_id,
                                   "sent_monotonic": time.monotonic()}, cfg.request_timeout_s)
            _fill_from_response(rec, resp, t0)
            rec.in_measurement_window = rec.actual_send_s <= duration_s
            with lock:
                out.records.append(rec)

    threads = [threading.Thread(target=_worker, args=(i,), daemon=True)
               for i in range(max(1, cfg.concurrency))]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=duration_s + cfg.drain_timeout_s + 5)
    out.measurement_end_monotonic = time.monotonic()
    out.planned_count = counter["n"]
    return out
