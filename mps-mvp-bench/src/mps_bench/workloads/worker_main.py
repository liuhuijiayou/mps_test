"""Container-side worker entrypoint: `python -m mps_bench.workloads.worker_main`.

Runs exactly one CUDA worker. Collects the runtime version evidence that
nvidia-smi cannot provide (torch build, CUDA runtime, visible device UUID) and
writes it next to the results so the report can cite it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def _runtime_evidence() -> Dict[str, Any]:
    import torch
    ev: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "mps_active_thread_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
        "mps_pinned_device_mem_limit": os.environ.get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"),
        "worker_pid": os.getpid(),
    }
    try:
        import torchvision
        ev["torchvision_version"] = torchvision.__version__
    except Exception:
        ev["torchvision_version"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        ev["device"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
            # uuid is exposed on recent torch builds; absent -> null, never faked
            "uuid": str(getattr(props, "uuid", "")) or None,
        }
        ev["driver_reported_runtime_version"] = torch.version.cuda
    return ev


def build_workload(args: argparse.Namespace):
    if args.workload == "image_inference":
        from .image_inference import ImageInferenceWorkload
        return ImageInferenceWorkload(
            model_name=args.model_name, input_size=args.input_size,
            batch_size=args.batch_size, precision=args.precision,
            input_pool_size=args.input_pool_size, cpu_threads=args.cpu_threads,
            seed=args.seed, weights_path=args.weights_path)
    if args.workload == "recsys_scoring":
        from .recsys_scoring import RecsysScoringWorkload
        return RecsysScoringWorkload(
            num_tables=args.num_tables, rows_per_table=args.rows_per_table,
            embedding_dim=args.embedding_dim, indices_per_sample=args.indices_per_sample,
            dense_features=args.dense_features, bottom_mlp=json.loads(args.bottom_mlp),
            top_mlp=json.loads(args.top_mlp), batch_size=args.batch_size,
            precision=args.precision, index_distribution=args.index_distribution,
            zipf_s=args.zipf_s, working_set_rows=args.working_set_rows,
            input_pool_size=args.input_pool_size, cpu_threads=args.cpu_threads, seed=args.seed)
    raise SystemExit(f"未知 workload: {args.workload}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MPS bench GPU worker (one CUDA context)")
    ap.add_argument("--workload", required=True, choices=("image_inference", "recsys_scoring"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--queue-limit", type=int, default=1024)
    ap.add_argument("--request-timeout-s", type=float, default=5.0)
    ap.add_argument("--warmup-iterations", type=int, default=20)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--drain-timeout-s", type=float, default=30.0)
    ap.add_argument("--status-path", default="/results/worker_status.json")
    ap.add_argument("--evidence-path", default="/results/worker_evidence.json")
    ap.add_argument("--seed", type=int, default=20251001)
    # image
    ap.add_argument("--model-name", default="resnet18")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--weights-path", default=None)
    # shared
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--input-pool-size", type=int, default=64)
    ap.add_argument("--cpu-threads", type=int, default=4)
    # recsys
    ap.add_argument("--num-tables", type=int, default=16)
    ap.add_argument("--rows-per-table", type=int, default=200000)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--indices-per-sample", type=int, default=32)
    ap.add_argument("--dense-features", type=int, default=128)
    ap.add_argument("--bottom-mlp", default="[512,256,64]")
    ap.add_argument("--top-mlp", default="[512,256,1]")
    ap.add_argument("--index-distribution", default="zipf")
    ap.add_argument("--zipf-s", type=float, default=1.1)
    ap.add_argument("--working-set-rows", type=int, default=None)
    args = ap.parse_args(argv)

    evidence = _runtime_evidence()
    try:
        with open(args.evidence_path, "w", encoding="utf-8") as fh:
            json.dump(evidence, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[worker] 无法写入 evidence: {exc}", file=sys.stderr)

    if not evidence.get("cuda_available"):
        print("[worker] CUDA 不可用，拒绝启动（不得以 CPU 结果冒充 GPU 验证）", file=sys.stderr)
        return 2

    workload = build_workload(args)
    from .service import serve
    summary = serve(workload, host=args.host, port=args.port,
                    queue_limit=args.queue_limit,
                    request_timeout_s=args.request_timeout_s,
                    warmup_iterations=args.warmup_iterations,
                    tolerance=args.tolerance,
                    status_path=args.status_path,
                    drain_timeout_s=args.drain_timeout_s)
    print(json.dumps({"worker_exit_summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
