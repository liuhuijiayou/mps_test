"""Unified request record.

Every timestamp uses a monotonic clock for durations; `wall_ts` is recorded
separately so the record can be aligned with device logs. Missing timestamps stay
null -- they are never back-filled with a plausible value.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, TextIO

# Status vocabulary. `late` is not a status: lateness is recorded numerically so
# a silently de-rated load generator cannot hide coordinated omission.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
STATUS_REJECTED = "rejected"
STATUS_OUTPUT_ERROR = "output_error"
STATUS_NOT_SENT = "not_sent"      # planned but the generator could not send it
STATUS_INCOMPLETE = "incomplete"  # still in flight when drain timeout expired

STATUSES = (STATUS_OK, STATUS_ERROR, STATUS_TIMEOUT, STATUS_REJECTED,
            STATUS_OUTPUT_ERROR, STATUS_NOT_SENT, STATUS_INCOMPLETE)


@dataclass
class RequestRecord:
    run_id: str
    case_id: str
    worker_id: str
    request_id: Any
    round_index: int = 0
    # timings (monotonic seconds, relative to measurement start unless noted)
    planned_arrival_s: Optional[float] = None
    actual_send_s: Optional[float] = None
    service_received_s: Optional[float] = None
    enqueued_s: Optional[float] = None
    exec_start_s: Optional[float] = None
    gpu_done_s: Optional[float] = None
    response_done_s: Optional[float] = None
    gpu_ms: Optional[float] = None
    # Short aliases accepted at construction time (see __post_init__); they are
    # the names used in the HTTP payloads and in tests.
    send_s: Optional[float] = None
    service_recv_s: Optional[float] = None
    enqueue_s: Optional[float] = None
    response_s: Optional[float] = None
    # bookkeeping
    status: str = STATUS_OK
    error_type: Optional[str] = None
    error: Optional[str] = None
    samples: int = 0
    batch_size: int = 0
    unit: str = "samples"
    in_measurement_window: bool = True
    wall_ts: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"未知请求状态 {self.status!r}，允许值: {STATUSES}")
        for alias, canonical in (("send_s", "actual_send_s"),
                                 ("service_recv_s", "service_received_s"),
                                 ("enqueue_s", "enqueued_s"),
                                 ("response_s", "response_done_s")):
            value = getattr(self, alias)
            if value is not None and getattr(self, canonical) is None:
                setattr(self, canonical, value)
            elif value is None:
                setattr(self, alias, getattr(self, canonical))

    @property
    def send_delay_s(self) -> Optional[float]:
        """Positive means the generator was late (queueing on the client side)."""
        if self.planned_arrival_s is None or self.actual_send_s is None:
            return None
        return self.actual_send_s - self.planned_arrival_s

    @property
    def e2e_s(self) -> Optional[float]:
        """End-to-end latency measured from the PLANNED arrival time when one
        exists (open loop), otherwise from the actual send time (closed loop).
        Using planned arrival is what avoids coordinated omission."""
        start = self.planned_arrival_s if self.planned_arrival_s is not None else self.actual_send_s
        if start is None or self.response_done_s is None:
            return None
        return self.response_done_s - start

    @property
    def service_s(self) -> Optional[float]:
        if self.actual_send_s is None or self.response_done_s is None:
            return None
        return self.response_done_s - self.actual_send_s

    @property
    def queue_s(self) -> Optional[float]:
        if self.enqueued_s is None or self.exec_start_s is None:
            return None
        return self.exec_start_s - self.enqueued_s

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["send_delay_s"] = self.send_delay_s
        out["e2e_s"] = self.e2e_s
        out["service_s"] = self.service_s
        out["queue_s"] = self.queue_s
        return out


class RequestWriter:
    """Append-only jsonl writer. Accepts a path or an open handle."""

    def __init__(self, target: Any):
        self._owns_handle = isinstance(target, str)
        self.handle: TextIO = (open(target, "a", encoding="utf-8")
                               if self._owns_handle else target)
        self.count = 0

    def write(self, record: RequestRecord) -> None:
        self.handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
        self.count += 1

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> None:
        self.flush()
        if self._owns_handle:
            self.handle.close()

    def __enter__(self) -> "RequestWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
