"""Recommendation scoring workload -- DLRM-style (explicitly labeled).

Real compute graph, not a placeholder:
    sparse indices -> EmbeddingBag(sum) per table
    dense features -> bottom MLP
    feature interaction -> pairwise dot products of [dense_emb | table embs]
    concat(dense_emb, interactions) -> top MLP -> score

The index working set is configurable (`working_set_rows`) precisely so we do not
hammer a tiny fixed row set and then claim we exercised HBM.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from .base import BatchResult, Workload, WorkloadInfo

_DTYPES = {"fp32": "float32", "tf32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def _zipf_weights(n: int, s: float):
    import torch
    ranks = torch.arange(1, n + 1, dtype=torch.float64)
    w = 1.0 / ranks.pow(s)
    return (w / w.sum()).float()


class RecsysScoringWorkload(Workload):
    unit = "samples"
    kind = "recsys_scoring"

    def __init__(self,
                 num_tables: int = 16,
                 rows_per_table: int = 200000,
                 embedding_dim: int = 64,
                 indices_per_sample: int = 32,
                 dense_features: int = 128,
                 bottom_mlp: Optional[List[int]] = None,
                 top_mlp: Optional[List[int]] = None,
                 batch_size: int = 1024,
                 precision: str = "fp16",
                 index_distribution: str = "zipf",
                 zipf_s: float = 1.1,
                 working_set_rows: Optional[int] = None,
                 input_pool_size: int = 32,
                 cpu_threads: int = 4,
                 seed: int = 20251001,
                 device: str = "cuda"):
        import torch
        import torch.nn as nn

        if precision not in _DTYPES:
            raise ValueError(f"未知 precision {precision}")
        self.torch = torch
        self.num_tables = int(num_tables)
        self.rows_per_table = int(rows_per_table)
        self.embedding_dim = int(embedding_dim)
        self.indices_per_sample = int(indices_per_sample)
        self.dense_features = int(dense_features)
        self.bottom_mlp = list(bottom_mlp or [512, 256, embedding_dim])
        self.top_mlp = list(top_mlp or [512, 256, 1])
        self.batch_size = int(batch_size)
        self.precision = precision
        self.index_distribution = index_distribution
        self.zipf_s = float(zipf_s)
        self.working_set_rows = int(working_set_rows or rows_per_table)
        if self.working_set_rows > self.rows_per_table:
            raise ValueError("working_set_rows 不能大于 rows_per_table")
        self.input_pool_size = int(input_pool_size)
        self.seed = int(seed)
        self.device = torch.device(device)

        torch.set_num_threads(int(cpu_threads))
        torch.backends.cuda.matmul.allow_tf32 = (precision == "tf32")
        torch.manual_seed(self.seed)
        self.dtype = getattr(torch, _DTYPES[precision])

        # bottom MLP must project dense features to embedding_dim so the
        # interaction stage has uniform vector width.
        bottom_dims = [self.dense_features] + self.bottom_mlp
        if bottom_dims[-1] != self.embedding_dim:
            bottom_dims.append(self.embedding_dim)

        class _Model(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.embeddings = nn.ModuleList([
                    nn.EmbeddingBag(rows_per_table, embedding_dim, mode="sum", sparse=False)
                    for _ in range(num_tables)
                ])
                layers: List[nn.Module] = []
                for i in range(len(bottom_dims) - 1):
                    layers += [nn.Linear(bottom_dims[i], bottom_dims[i + 1]), nn.ReLU()]
                inner.bottom = nn.Sequential(*layers)
                n_vec = num_tables + 1
                n_inter = n_vec * (n_vec - 1) // 2
                top_dims = [embedding_dim + n_inter] + list(top_mlp or [512, 256, 1])
                tlayers: List[nn.Module] = []
                for i in range(len(top_dims) - 2):
                    tlayers += [nn.Linear(top_dims[i], top_dims[i + 1]), nn.ReLU()]
                tlayers += [nn.Linear(top_dims[-2], top_dims[-1])]
                inner.top = nn.Sequential(*tlayers)
                inner.n_vec = n_vec

            def forward(inner, dense, indices, offsets):
                embs = [emb(indices[i], offsets) for i, emb in enumerate(inner.embeddings)]
                dense_emb = inner.bottom(dense)
                stacked = inner.torch_stack([dense_emb] + embs)
                # pairwise dot products, upper triangle without diagonal
                inter = stacked.matmul(stacked.transpose(1, 2))
                idx_r, idx_c = inner._tri_idx
                flat = inter[:, idx_r, idx_c]
                return inner.top(inner.torch_cat([dense_emb, flat]))

            # tiny indirections keep forward readable while avoiding closures
            def torch_stack(inner, vecs):
                import torch as _t
                return _t.stack(vecs, dim=1)

            def torch_cat(inner, vecs):
                import torch as _t
                return _t.cat(vecs, dim=1)

        model = _Model()
        n_vec = model.n_vec
        tri = torch.triu_indices(n_vec, n_vec, offset=1)
        model._tri_idx = (tri[0].to(self.device), tri[1].to(self.device))
        self.model = model.eval().to(self.device)
        if precision in ("fp16", "bf16"):
            self.model = self.model.to(self.dtype)

        # ---- input pool (created during warmup phase) ----
        gen = torch.Generator(device="cpu").manual_seed(self.seed + 11)
        if index_distribution == "zipf":
            probs = _zipf_weights(self.working_set_rows, self.zipf_s)
        else:
            probs = None
        total = self.batch_size * self.indices_per_sample
        self.pool: List[Dict[str, Any]] = []
        for _ in range(self.input_pool_size):
            dense = torch.randn(self.batch_size, self.dense_features, generator=gen,
                                dtype=torch.float32).to(self.device, dtype=self.dtype)
            per_table = []
            for _t in range(self.num_tables):
                if probs is not None:
                    idx = torch.multinomial(probs, total, replacement=True, generator=gen)
                else:
                    idx = torch.randint(0, self.working_set_rows, (total,), generator=gen)
                per_table.append(idx.to(self.device, dtype=torch.long))
            offsets = torch.arange(0, total, self.indices_per_sample,
                                   dtype=torch.long, device=self.device)
            self.pool.append({"dense": dense, "indices": per_table, "offsets": offsets})
        self._reference: Optional[float] = None

    # ------------------------------------------------------------------ #
    def info(self) -> WorkloadInfo:
        return WorkloadInfo(
            kind=self.kind, unit=self.unit, synthetic_inputs=True, synthetic_weights=True,
            detail={"style": "DLRM-style (EmbeddingBag + dense MLP + feature interaction + top MLP)",
                    "num_tables": self.num_tables, "rows_per_table": self.rows_per_table,
                    "embedding_dim": self.embedding_dim,
                    "indices_per_sample": self.indices_per_sample,
                    "dense_features": self.dense_features,
                    "bottom_mlp": self.bottom_mlp, "top_mlp": self.top_mlp,
                    "batch_size": self.batch_size, "precision": self.precision,
                    "index_distribution": self.index_distribution, "zipf_s": self.zipf_s,
                    "working_set_rows": self.working_set_rows,
                    "embedding_table_bytes": self.num_tables * self.rows_per_table
                                             * self.embedding_dim
                                             * (2 if self.precision in ("fp16", "bf16") else 4),
                    "seed": self.seed},
        )

    def warmup(self, iterations: int = 3) -> None:
        torch = self.torch
        with torch.inference_mode():
            for i in range(max(1, iterations)):
                item = self.pool[i % len(self.pool)]
                self.model(item["dense"], item["indices"], item["offsets"])
        torch.cuda.synchronize(self.device)
        self._reference = self._compute_checksum(0)

    def _compute_checksum(self, batch_index: int) -> float:
        torch = self.torch
        item = self.pool[batch_index % len(self.pool)]
        with torch.inference_mode():
            out = self.model(item["dense"], item["indices"], item["offsets"])
        torch.cuda.synchronize(self.device)
        return float(out.float().sum().item())

    def run_batch(self, batch_index: int) -> BatchResult:
        torch = self.torch
        item = self.pool[batch_index % len(self.pool)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            start.record()
            out = self.model(item["dense"], item["indices"], item["offsets"])
            end.record()
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
        flat = out.float()
        return BatchResult(batch_size=self.batch_size, gpu_ms=gpu_ms,
                           checksum=float(flat.sum().item()),
                           output_shape=tuple(out.shape),
                           finite=bool(torch.isfinite(flat).all().item()),
                           samples=self.batch_size)

    def reference_checksum(self) -> float:
        if self._reference is None:
            self._reference = self._compute_checksum(0)
        return self._reference
