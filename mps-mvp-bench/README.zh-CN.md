# mps-mvp-bench

容器化的 NVIDIA MPS MVP 验证工程。只验证两类混部场景，并把每条结论都绑定到
可回溯的证据上。

## 这个项目验证什么

**场景 A：高优半卡 & 高优半卡混部**

| 子项 | 显存 | SM |
| --- | --- | --- |
| 各 50% 显存上限 | 每 client 50% | — |
| SM 不隔离各 70% | 50% | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70` |
| SM 不隔离各 50% | 50% | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` |
| SM 静态隔离各 50% | 50% | 静态分区（能力探测，不支持则 SKIP） |

**场景 B：高优整卡 & 低优半卡混部**

只给低优加 50% 显存上限；调度优先级（`CUDA_MPS_CLIENT_PRIORITY`）
与执行资源上限（高 100% / 低 50%、70%）分别与组合验证。

指标：吞吐、延迟、故障率。

## 明确不在范围内

以下能力本工程**不实现、不验证、也不声称**：

- Launch Hook
- PID 级限流
- 动态显存回收 / 驱逐
- QoS 自动调控器
- Kubernetes Operator

`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 是**固定显存配额**。
本工程不会把它表述为动态回收能力。

## 当前交付状态

| 项 | 状态 |
| --- | --- |
| 控制层 / 负载 / 采集 / 统计 / 报告代码 | 已实现 |
| CPU 单元测试 | **已实际运行通过（133 passed）** |
| 用例矩阵展开与校验（41 个展开用例） | **已实际运行通过** |
| 镜像构建 | **未执行**（Agent 环境无 GPU、无 docker daemon） |
| 任何 GPU 上的性能/故障数据 | **未执行**，报告中一律为 `NOT_RUN` |

本仓库**不包含任何性能数字**。所有性能结论必须由你在目标机实际跑出。

## 目标机最短命令链

```bash
# 0. 选定一张空闲卡的 UUID（切勿使用已被其他服务占用的那张）
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv

# 1. 构建镜像
cd mps-mvp-bench
docker build -f docker/Dockerfile -t mps-mvp-bench:local .

# 2. 写入本机专属配置（UUID 与设备传递参数必须由你填，工程不猜）
cp configs/runtime.example.yaml configs/runtime.yaml
$EDITOR configs/runtime.yaml     # 填 gpu.uuid 与 docker.runtime_args

# 3. 只读预检，产出 capability.json
mps-bench preflight --config configs/runtime.yaml

# 4. 链路冒烟（不产生性能证据）
mps-bench smoke --config configs/runtime.yaml

# 5. 正式测量（按家族分批，避免一次超预算）
mps-bench run --config configs/runtime.yaml --profile standard --family baselines
mps-bench run --config configs/runtime.yaml --profile standard --family high_high
mps-bench run --config configs/runtime.yaml --profile standard --family high_low
mps-bench run --config configs/runtime.yaml --profile standard --family memory

# 6. 生成报告
mps-bench report --run-dir results/<run_id>

# 7. 清理（幂等，只动本 run 自己的资源）
mps-bench cleanup --config configs/runtime.yaml
```

详细步骤与排障见 [`docs/RUNBOOK.zh-CN.md`](docs/RUNBOOK.zh-CN.md)。

## 安全边界

工程在代码层面拒绝以下行为，不依赖操作者自觉：

1. **目标卡上有非本实验进程就拒绝运行**，且每轮重新检查，而不是只在开始时看一眼。
2. GPU 由 UUID 锁定；容器带 `run_id` 标签，只操作自己标签的容器。
3. 不执行全局 `pkill`、`docker rm -f $(docker ps -aq)`、全局 MPS `quit`、
   驱动卸载、主机重启、自动 `nvidia-smi --gpu-reset`。
4. 不接管已存在的 MPS 实例。
5. 不使用 `--privileged`，不挂载 docker socket。
6. 默认不修改 compute mode。
7. 清理幂等。
8. 破坏性故障用例（F4–F9）默认关闭，需要 CLI 与配置双开关加 UUID 确认。

## 证据优先

几条贯穿实现的规则：

- **环境变量设置成功 ≠ 已接入 MPS。** 必须在我们自己启动的 server 的
  `get_client_list` 里看到 worker 的 host PID，否则该轮记为 `INCONCLUSIVE`。
- **`nvidia-smi` 显示的 CUDA 版本不能证明容器内的工具链版本。** 镜像自己记录
  `build-versions.json`。
- **GPU util（device busy）不是 SM active。** `DCGM_FI_PROF_SM_ACTIVE` 不可用时记
  `null` 加原因，绝不填 0，绝不用 GPU util 顶替。
- **各轮 P99 的均值不是整体 P99。** 两者分别输出并标注。
- **不同单位的吞吐不相加。** images/s 与 samples/s 各自呈现。
- **未复现的负例表述为"本配置下未观察到"**，不表述为"不存在"。
- **未观察到故障传播表述为"0/n 次"**，不表述为"完全隔离"。
- **达不到配额就 OOM 不算配额生效。** 必须同时证明设备仍有物理余量。

## 文档

- [`docs/CONFIG.zh-CN.md`](docs/CONFIG.zh-CN.md) — 配置项、默认值、覆盖优先级
- [`docs/RUNBOOK.zh-CN.md`](docs/RUNBOOK.zh-CN.md) — 目标机操作手册与排障
- [`docs/METHODOLOGY.zh-CN.md`](docs/METHODOLOGY.zh-CN.md) — 测量与统计口径

## 开发

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock
pip install -e .

# CPU 单元测试（不需要 GPU / docker / MPS）
PYTHONPATH=src pytest tests/unit -q

# GPU 集成测试（仅在目标机启用）
MPS_BENCH_GPU_TESTS=1 \
MPS_BENCH_GPU_UUID=GPU-xxxx \
MPS_BENCH_IMAGE=mps-mvp-bench:local \
MPS_BENCH_DOCKER_RUNTIME_ARGS='--gpus device=GPU-xxxx' \
  pytest tests/integration -q
```
