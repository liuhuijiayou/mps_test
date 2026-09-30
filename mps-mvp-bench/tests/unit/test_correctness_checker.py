"""Correctness checker must compare each batch against its OWN reference.

Regression guard for the bug where every request was validated against the
checksum of pool entry 0, which turned every legitimate non-zero batch into an
`output_error` (observed success rate ~4%).
"""

from __future__ import annotations

import queue

import pytest

from mps_bench.workloads.base import BatchResult, checksum_tolerance_ok
from mps_bench.workloads.service import GpuExecutor, Job, WorkerState


class FakeWorkload:
    """Minimal pool-backed workload: batch i deterministically yields i*100.0."""

    def __init__(self, pool_size: int = 4) -> None:
        self.pool = [float(i) * 100.0 for i in range(pool_size)]
        self._references = None
        self.compute_calls = 0

    def _compute_checksum(self, batch_index: int) -> float:
        self.compute_calls += 1
        return self.pool[batch_index % len(self.pool)]

    def warmup(self, iterations: int = 3) -> None:
        # Pre-compute the whole pool BEFORE serving traffic.
        self._references = [self._compute_checksum(i) for i in range(len(self.pool))]

    def run_batch(self, batch_index: int) -> BatchResult:
        return BatchResult(batch_size=1, gpu_ms=1.0,
                           checksum=self.pool[batch_index % len(self.pool)],
                           output_shape=(1, 10), finite=True, samples=1)

    def reference_checksum(self, batch_index: int = 0) -> float:
        if self._references is None:
            self.warmup()
        return self._references[batch_index % len(self._references)]


def test_reference_differs_per_batch_index():
    wl = FakeWorkload(pool_size=4)
    wl.warmup()
    refs = [wl.reference_checksum(i) for i in range(4)]
    assert refs == [0.0, 100.0, 200.0, 300.0]
    # distinct batches must not share one reference
    assert len(set(refs)) == 4
    # and the index wraps with the pool
    assert wl.reference_checksum(5) == wl.reference_checksum(1)


def test_reference_lookup_is_o1_after_warmup():
    """The request path must never trigger an extra model forward."""
    wl = FakeWorkload(pool_size=4)
    wl.warmup()
    calls_after_warmup = wl.compute_calls
    assert calls_after_warmup == 4
    for i in range(50):
        wl.reference_checksum(i)
    assert wl.compute_calls == calls_after_warmup


def _drain(wl, n: int):
    state = WorkerState()
    jobs: "queue.Queue" = queue.Queue()
    ex = GpuExecutor(wl, state, jobs, tolerance=1e-3)
    ex.start()
    results = []
    for i in range(n):
        job = Job(request_id=str(i), sent_monotonic=0.0,
                  received_monotonic=0.0, enqueued_monotonic=0.0)
        jobs.put(job)
        assert job.done.wait(timeout=5), "executor did not finish the job"
        results.append(job.result)
    jobs.put(None)
    ex.join(timeout=5)
    return state, results


def test_executor_has_no_output_errors_across_the_whole_pool():
    wl = FakeWorkload(pool_size=4)
    wl.warmup()
    # more requests than pool entries: every pool entry is exercised twice
    state, results = _drain(wl, 8)
    assert [r["status"] for r in results] == ["ok"] * 8
    assert state.checksum_mismatches == 0
    assert state.failed == 0
    assert state.completed == 8


def test_executor_still_flags_a_genuinely_wrong_output():
    """Tolerance is untouched: real numeric drift must still be caught."""

    class Drifting(FakeWorkload):
        def run_batch(self, batch_index: int) -> BatchResult:
            res = super().run_batch(batch_index)
            if batch_index == 2:
                res.checksum += 50.0  # far outside tolerance
            return res

    wl = Drifting(pool_size=4)
    wl.warmup()
    state, results = _drain(wl, 4)
    assert [r["status"] for r in results] == ["ok", "ok", "output_error", "ok"]
    assert state.checksum_mismatches == 1


def test_tolerance_comparison_is_relative_and_unchanged():
    assert checksum_tolerance_ok(100.0, 100.05, 0.001)
    assert not checksum_tolerance_ok(100.0, 101.0, 0.001)
    # a zero reference falls back to an absolute comparison
    assert checksum_tolerance_ok(0.0, 0.0005, 0.001)
    assert not checksum_tolerance_ok(0.0, 0.5, 0.001)


@pytest.mark.parametrize("module_name,class_name", [
    ("mps_bench.workloads.image_inference", "ImageInferenceWorkload"),
    ("mps_bench.workloads.recsys_scoring", "RecsysScoringWorkload"),
])
def test_real_workloads_accept_a_batch_index(module_name, class_name):
    """Both adapters must expose `reference_checksum(batch_index)`.

    Signature-only check: importing the class needs no torch runtime.
    """
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    params = inspect.signature(cls.reference_checksum).parameters
    assert "batch_index" in params
    assert params["batch_index"].default == 0
