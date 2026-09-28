# mps_test

Containerized verification harness for **NVIDIA MPS** (Multi-Process Service)
colocation on a single GPU.

The project lives in [`mps-mvp-bench/`](mps-mvp-bench/).
Full documentation is in Chinese: [`mps-mvp-bench/README.zh-CN.md`](mps-mvp-bench/README.zh-CN.md).

## What it verifies

Two colocation scenarios on one physical GPU:

**A. Two high-priority clients, half card each**

| Variant | Memory | SM |
| --- | --- | --- |
| 50% memory cap each | 50% per client | — |
| SM unpartitioned, 70% each | 50% | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70` |
| SM unpartitioned, 50% each | 50% | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` |
| SM statically partitioned, 50% each | 50% | static partition (probed; SKIP if unsupported) |

**B. One high-priority full-card client + one low-priority half-card client**

Only the low-priority client gets a memory cap. Scheduling priority
(`CUDA_MPS_CLIENT_PRIORITY`) and execution resource limits are verified
separately and combined.

Metrics: throughput, latency, failure rate.

## Explicitly out of scope

Not implemented, not verified, not claimed: Launch Hook, PID-level throttling,
dynamic memory reclamation/eviction, QoS auto-controller, Kubernetes Operator.

`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` is a **fixed** quota. This project never
describes it as dynamic reclamation.

## Status

| | |
| --- | --- |
| Control / workload / telemetry / stats / reporting code | implemented |
| CPU unit tests | **133 passing** |
| Case matrix (41 expanded cases) validated | **yes** |
| Docker image build | **not executed** (no GPU or docker daemon in the dev environment) |
| Any GPU measurement | **not executed** — every MVP item reports `NOT_RUN` |

**This repository contains no performance numbers.** All performance claims must
come from an actual run on your own hardware.

## Design stance

The harness is built around one idea: *a result is only as good as the evidence
behind it.* Concretely:

- Setting the MPS env vars is **not** proof of attachment. The worker's host PID
  must appear in `get_client_list` for a server this run started, or the round is
  recorded `INCONCLUSIVE`.
- `nvidia-smi`'s CUDA version says nothing about the container toolchain. The
  image records its own versions.
- GPU utilization (device busy) is **not** SM active. When
  `DCGM_FI_PROF_SM_ACTIVE` is unavailable the value is `null` with a reason —
  never `0`, never substituted.
- The mean of per-round P99s is **not** the overall P99. Both are reported, labeled.
- Throughput in different units is never summed.
- An unreproduced negative case is reported as "not observed in this
  configuration", not as "does not occur".
- No observed fault propagation is stated as "0/n", not as "fully isolated".
- An OOM below the quota is **not** evidence the quota works. Device headroom
  must also be provable.

## Safety

Enforced in code, not left to operator discipline: refuses to run if the target
GPU carries any foreign process (re-checked every round); GPU locked by UUID;
containers scoped by run-id label; no global `pkill`, no
`docker rm -f $(docker ps -aq)`, no global MPS `quit`, no driver unload, no host
reboot, no automatic GPU reset; never adopts an existing MPS instance; no
`--privileged`, no docker socket mount; compute mode preserved; cleanup
idempotent; destructive fault cases off by default behind a CLI flag, a config
flag, and a UUID confirmation.

## Quick start

See [`mps-mvp-bench/docs/RUNBOOK.zh-CN.md`](mps-mvp-bench/docs/RUNBOOK.zh-CN.md).

```bash
cd mps-mvp-bench
docker build -f docker/Dockerfile -t mps-mvp-bench:local .
cp configs/runtime.example.yaml configs/runtime.yaml   # fill gpu.uuid + docker.runtime_args
mps-bench preflight --config configs/runtime.yaml      # read-only
mps-bench smoke     --config configs/runtime.yaml      # link check only, not evidence
mps-bench run       --config configs/runtime.yaml --profile standard --family baselines
mps-bench report    --run-dir results/<run_id>
mps-bench cleanup   --config configs/runtime.yaml
```

Run `baselines` first — without B0 (solo) and B2 (non-MPS concurrent) every
delta is missing its denominator, and B2 is what separates "MPS helped" from
"two tasks were just taking turns".

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r mps-mvp-bench/requirements.lock
cd mps-mvp-bench && pip install -e .

PYTHONPATH=src pytest tests/unit -q          # no GPU / docker / MPS needed
```

GPU integration tests are marked `gpu` and skipped unless
`MPS_BENCH_GPU_TESTS=1` is set.

## Docs

- [`README.zh-CN.md`](mps-mvp-bench/README.zh-CN.md) — overview
- [`docs/CONFIG.zh-CN.md`](mps-mvp-bench/docs/CONFIG.zh-CN.md) — all 148 config keys (generated from the schema)
- [`docs/RUNBOOK.zh-CN.md`](mps-mvp-bench/docs/RUNBOOK.zh-CN.md) — operations and troubleshooting
- [`docs/METHODOLOGY.zh-CN.md`](mps-mvp-bench/docs/METHODOLOGY.zh-CN.md) — measurement and statistics
