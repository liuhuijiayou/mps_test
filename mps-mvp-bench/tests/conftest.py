"""pytest configuration.

GPU tests are marked `gpu` and skipped unless MPS_BENCH_GPU_TESTS=1 is set.
The CPU suite therefore never depends on a GPU, a driver, docker, or MPS.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: requires a real GPU + docker + MPS on the target host")


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if os.environ.get("MPS_BENCH_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(
        reason="GPU 集成测试未启用：需在目标机设置 MPS_BENCH_GPU_TESTS=1 "
               "并提供 MPS_BENCH_GPU_UUID / MPS_BENCH_IMAGE")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
