"""MPS MVP verification harness.

Two independent halves live in this package:

* host orchestrator (cli/config/control/telemetry/reporting) -- must import
  cleanly WITHOUT torch or any CUDA python binding, so CPU-only machines and CI
  can run the unit tests and `plan`.
* in-container workload (workloads/, loadgen client helpers) -- imports torch
  lazily, only inside the container entrypoint.

Never add a top-level `import torch` to any module reachable from `cli.py`.
"""

__version__ = "0.1.0"
