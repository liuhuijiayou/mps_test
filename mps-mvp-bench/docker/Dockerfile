# syntax=docker/dockerfile:1.6
#
# Workload image: PyTorch + torchvision + the native CUDA probes.
#
# Version pinning rationale (verified against NVIDIA/PyTorch official release
# notes; no `latest` anywhere):
#   * CUDA 12.4.1 devel base on Ubuntu 22.04 -- CUDA 12.x requires driver >= 525,
#     and the target host runs R580, so it is forward-compatible. We deliberately
#     do NOT assume the "CUDA 13.0" string from nvidia-smi says anything about the
#     toolkit inside this image.
#   * torch 2.4.1 + torchvision 0.19.1 are an officially matched pair, published
#     with cu121/cu124 wheels. We use the cu121 wheel index, which is supported on
#     a CUDA 12.4 runtime via minor-version compatibility.
#   * The build never needs a GPU and never starts MPS.
#
# Both the base image and the package mirrors are configurable so the image can be
# built on an offline / internal-mirror host.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG APT_MIRROR=""
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG CUDA_ARCH=80

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=8.0

# Optional internal apt mirror.
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g; \
              s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        cmake ninja-build build-essential \
        ca-certificates tini procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- native probes (explicit sm_80; no GPU needed at build time) ----
COPY native /src/native
RUN cmake -S /src/native -B /src/native/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DCMAKE_INSTALL_PREFIX=/opt/mps-bench \
    && cmake --build /src/native/build --parallel \
    && cmake --install /src/native/build \
    && /opt/mps-bench/bin/fault_injector --mode api-error --verify-assert-enabled \
         > /opt/mps-bench/build-assert-check.txt 2>&1 || true

# ---- python deps ----
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade "pip==24.2" --index-url "${PIP_INDEX_URL}" \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" \
         "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    && python -m pip install --index-url "${PIP_INDEX_URL}" -r /tmp/requirements.lock

# ---- harness ----
COPY pyproject.toml /src/pyproject.toml
COPY src /src/src
RUN python -m pip install --no-deps --index-url "${PIP_INDEX_URL}" /src

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Record what was actually installed so the manifest can cite real versions
# instead of trusting the nvidia-smi CUDA field.
RUN python -c 'import json,sys,torch,torchvision; print(json.dumps({"python":sys.version,"torch":torch.__version__,"torch_cuda_build":torch.version.cuda,"torchvision":torchvision.__version__}, indent=2))' > /opt/mps-bench/build-versions.json
ENV PATH=/opt/mps-bench/bin:$PATH \
    MPS_BENCH_NATIVE_BIN=/opt/mps-bench/bin

# tini reaps zombies; entrypoint.sh forwards signals to the worker so the F3
# safe-exit test actually delivers SIGTERM/SIGINT to the CUDA process.
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "mps_bench.workloads.worker_main", "--help"]
