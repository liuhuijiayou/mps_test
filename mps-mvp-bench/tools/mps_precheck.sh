#!/bin/bash
set -e

GPU=${1:-0}

echo "===== GPU CHECK ====="

nvidia-smi -i ${GPU}

GPU_UUID=$(nvidia-smi -i ${GPU} \
    --query-gpu=uuid \
    --format=csv,noheader)

echo "GPU UUID=${GPU_UUID}"


echo
echo "===== MPS CHECK ====="

which nvidia-cuda-mps-control
which nvidia-cuda-mps-server


echo
echo "===== DCGM CHECK ====="

if ! pgrep nv-hostengine >/dev/null; then
    echo "start nv-hostengine"
    nohup nv-hostengine >/tmp/nv-hostengine.log 2>&1 &
    sleep 3
fi

dcgmi discovery -l


echo
echo "===== DOCKER CHECK ====="

docker version


echo
echo "===== IMAGE CHECK ====="

docker image inspect mps-mvp-bench:local >/dev/null \
    && echo "image OK" \
    || echo "image missing"


echo
echo "===== CLEAN OLD MPS ====="

pkill -f nvidia-cuda-mps-server || true


echo
echo "===== RESULT DIR ====="

mkdir -p results


echo
echo "===== FINISH ====="

echo "GPU=${GPU}"
echo "UUID=${GPU_UUID}"
