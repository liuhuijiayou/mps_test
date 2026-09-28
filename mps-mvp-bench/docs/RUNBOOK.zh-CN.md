# 目标机操作手册

面向在 8×A30 验证机上实际执行的人。每一步都说明**它证明了什么**，
以及**失败时代表什么**——因为这里最容易出现的不是跑不起来，而是跑起来了
但结论不成立。

目标机基线（由你提供，工程不去探测）。本文以开发时的参考机型举例：

- 8 × NVIDIA A30，单卡 24576 MiB
- Driver 580.105.08
- MIG Disabled，Compute Mode Default
- **其中一张卡已被其他长期推理服务占用（约 22 GiB），不可触碰**

换成别的机型时，除了 `native/CMakeLists.txt` 里的 SM 架构（A30 为 `sm_80`）之外
不需要改代码；卡的数量、显存大小、驱动版本都从运行时查询。

> `nvidia-smi` 顶部显示的 `CUDA Version: 13.0` 是驱动支持的最高版本，
> **既不代表主机装了 CUDA 13，也不代表容器里的工具链版本**。
> 容器内的真实版本读镜像自己记录的 `/opt/mps-bench/build-versions.json`。

## 步骤 0：选卡

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,compute_mode \
           --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv
```

选一张 `memory.used` 接近 0 且没有任何 compute app 的卡，记下它的 UUID。

**不要选已被其他服务占用的那张。** 工程会在每一轮开始前检查目标卡上是否有
非本实验进程，发现就拒绝继续；但把 UUID 填错仍然可能让你白跑一轮，
所以这一步自己核对。

## 步骤 1：构建镜像

```bash
cd mps-mvp-bench
docker build -f docker/Dockerfile -t mps-mvp-bench:local .
```

镜像内容：CUDA 12.4.1 devel（Ubuntu 22.04）、PyTorch 2.4.1 + torchvision 0.19.1
（cu121 wheel）、编译好的原生 CUDA 探针、业务代码。版本全部固定，没有 `latest`。

原生探针按 `sm_80`（A30 = Ampere 8.0）显式编译，且构建时**剥掉了 `-DNDEBUG`**，
否则 F8 用例里的 device 侧 `assert` 会被编译器直接优化掉，变成一个什么都不测的用例。

内网机器没有外网时，用构建参数走镜像源：

```bash
docker build -f docker/Dockerfile \
  --build-arg APT_MIRROR=http://your-mirror/ubuntu \
  --build-arg PIP_INDEX_URL=http://your-pypi/simple \
  -t mps-mvp-bench:local .
```

构建完成后核对镜像自己记录的版本：

```bash
docker run --rm mps-mvp-bench:local cat /opt/mps-bench/build-versions.json
```

## 步骤 2：写配置

```bash
cp configs/runtime.example.yaml configs/runtime.yaml
$EDITOR configs/runtime.yaml
```

必须填的两项：

```yaml
gpu:
  uuid: "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"   # 步骤 0 选的那张

docker:
  image: "mps-mvp-bench:local"
  runtime_args: ["--gpus", "device=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"]
```

`runtime_args` 的正确写法取决于你这台机器的容器运行时配置。工程不提供默认值，
也会拒绝 `--gpus all`——把整机所有卡都暴露给容器，意味着一次越界分配就能
影响到别人正在跑的服务。

## 步骤 3：预检（只读）

```bash
mps-bench preflight --config configs/runtime.yaml --out results/preflight
```

产出 `results/preflight/capability.json`，每项能力标注
`supported` / `unsupported` / `unknown` 并附原始证据输出。

这一步**不启动任何容器、不启动 MPS、不修改任何状态**。

重点关注：

| 能力 | 不支持时的后果 |
| --- | --- |
| `dcgm_sm_active` | 标准 profile 会拒绝启动（见下方排障） |
| `mps_static_partitioning` | `HH-S` 用例记为 SKIP，不会被伪造 |
| `mps_client_priority` | `HL-P` / `HL-PC` 的优先级维度不成立 |
| `compute_mode_default` | 非 MPS 基线 B2 不成立，MPS 的"收益"无从对比 |

## 步骤 4：确认工作量

```bash
mps-bench plan --config configs/runtime.yaml --profile standard
```

打印展开后的用例数与预计总时长。全量 41 个展开用例在 standard profile 下约
686 分钟，会超过默认预算 `measure.total_budget_s=21600`（6 小时）并被拦下。

这是预期行为。按家族分批跑：

```bash
mps-bench plan --config configs/runtime.yaml --profile standard --family baselines
```

> `--family` 按 `cases/` 下的**目录名**过滤，合法值是：
> `baselines` `high_high` `high_low` `memory` `negative` `faults`。
> 注意目录名与 YAML 里的 `case.family` 字段不完全一致
> （目录 `baselines` 对应字段 `baseline`），以目录名为准。

## 步骤 5：冒烟

```bash
mps-bench smoke --config configs/runtime.yaml
```

等价于 `run --profile smoke`：5s warmup / 15s / 1 轮。

**它只证明链路打通——容器能起、MPS 能接、请求能通、指标能采。
它的性能数字没有任何意义，不要用它下结论。**

## 步骤 6：正式测量

基线必须先跑，否则后面所有 delta 都没有分母：

```bash
mps-bench run --config configs/runtime.yaml --profile standard --family baselines
mps-bench run --config configs/runtime.yaml --profile standard --family high_high
mps-bench run --config configs/runtime.yaml --profile standard --family high_low
mps-bench run --config configs/runtime.yaml --profile standard --family memory
mps-bench run --config configs/runtime.yaml --profile standard --family negative
```

也可以单独跑某个用例：

```bash
mps-bench run --config configs/runtime.yaml --profile standard --case-id HH70
```

运行期间终端有滚动摘要。每轮开始前工程会重新检查目标卡的进程归属，
中途有别人占卡会停下来而不是把污染的数据算进结果。

## 步骤 7：故障用例

非破坏性的 F1–F3 直接跑：

```bash
mps-bench faults --config configs/runtime.yaml --case-id F1 --case-id F2 --case-id F3
```

破坏性的 F4–F9 需要三重确认，且**只应在可承受故障的专用节点上执行**：

```bash
# 1) 在用例 YAML 里把 case.enabled 改为 true
# 2) 配置里 faults.allow_disruptive: true
# 3) CLI 双参数
mps-bench faults --config configs/runtime.yaml --case-id F7 \
  --allow-disruptive --confirm-gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

工程不会执行 `nvidia-smi --gpu-reset`。若 R1/R2/R3 恢复阶梯都没能恢复，
会停下来输出诊断信息，由你决定下一步。

## 步骤 8：报告

```bash
mps-bench report --run-dir results/<run_id>
```

产出 `results/<run_id>/report.html`（自包含，内联 SVG，不需要 Prometheus/Grafana）
与 `summary.csv`。

`results/<run_id>/` 下的完整产物：

| 文件 | 内容 |
| --- | --- |
| `environment.json` | 驱动、GPU、容器运行时、镜像版本 |
| `capability.json` | 能力探测结果与原始证据 |
| `effective_config.yaml` | 本次真正生效的完整配置 |
| `manifest.json` | run 级元信息与各用例索引 |
| `requests.jsonl` | 逐请求记录（统一 schema） |
| `gpu_metrics.csv` | 原始时间序列 |
| `events.jsonl` | 事件时间线 |
| `workload_logs/` | 各 worker 容器日志 |
| `mps_logs/` | MPS server / control 日志 |
| `summary.csv` | 汇总表 |
| `report.html` | 报告 |

报告里的状态词只有五个：`PASS` / `FAIL` / `SKIP` / `INCONCLUSIVE` / `NOT_RUN`。
没跑的项显式写 `NOT_RUN`，不会从表里消失。

## 步骤 9：清理

```bash
mps-bench cleanup --config configs/runtime.yaml
```

顺序是：先保存日志 → 停 client → 再处理 MPS。可以重复执行。

只会动带本项目标签的容器；**不会**停止不是本 run 启动的 MPS 实例。

## 排障

### `require_sm_metrics=true 但 DCGM_FI_PROF_SM_ACTIVE 不可用`

这是设计上的拒绝，不是 bug。先确认 DCGM 能不能采：

```bash
dcgmi discovery -l
dcgmi dmon -e 1002,1003,1005 -c 1
```

常见原因：没装 DCGM、权限不足、profiling 分组被占、或机器上已有别的 profiler。
工程**不会**去停止别人的采集器。

确认修不了、且接受结果不构成 SM 可观测验收时，显式降级：

```bash
mps-bench run --config configs/runtime.yaml --profile standard \
  --set telemetry.allow_degraded=true --family baselines
```

报告顶部会带一条显著的缺项声明。

### `目标卡上存在非本实验进程`

有别人在用这张卡。换卡或等对方结束。不要绕过这个检查——两个互不知情的
负载同卡跑出来的数字解释不了任何事情。

### `无法证明已接入 MPS` / 该轮记为 INCONCLUSIVE

环境变量设置成功不等于接入成功。工程要求在**我们自己启动的** MPS server 的
`get_client_list` 里看到 worker 的 host PID。

自查：

```bash
echo get_server_list | CUDA_MPS_PIPE_DIRECTORY=<pipe_dir> nvidia-cuda-mps-control
echo get_client_list <server_pid> | CUDA_MPS_PIPE_DIRECTORY=<pipe_dir> nvidia-cuda-mps-control
```

常见原因：pipe 目录没挂进容器、容器内进程没继承到环境变量、
或者 MPS server 在 worker 起来之前就退了（看 `mps_logs/`）。

### `已存在 MPS 实例：本项目不接管他人的 MPS`

设计上拒绝。接管一个别人配置的 MPS server 意味着你不知道它的
`ACTIVE_THREAD_PERCENTAGE` 和配额是什么，测出来的东西无法归因。

用一个独立的 pipe 目录（`mps.pipe_dir`），或先确认那个实例确实可以停掉。

### 静态分区用例被记为 SKIP

R580 的 MPS 控制接口不一定提供 `create_device_partition`。
这条 SKIP 是正确结论，不是缺陷。

不要用 `ACTIVE_THREAD_PERCENTAGE=50` 替代——那是执行资源上限，不是静态 SM 分区，
两者的隔离性质不同。

### 显存配额用例判为 `unexpected_oom`

说明 OOM 发生了，但证明不了是配额起的作用。可能的情况：

- 远未达到配额就 OOM → 配额之外还有别的限制，或配额换算错了
- 设备物理显存本身已接近耗尽 → 这是物理 OOM，与配额无关
- 拿不到设备剩余量 → 缺证据，不给"配额生效"的结论

先确认 `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 的字面值和设备剩余量：

```bash
grep -h pinned results/<run_id>/events.jsonl
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -i <uuid>
```

注意 context 与 CUDA 内部分配也计入 client 配额，所以用户缓冲区达不到
配额数值是正常的——工程的判定留了容差，不要求恰好占满。

### 容器起不来 / 看不到设备

```bash
docker run --rm <你的 runtime_args> mps-mvp-bench:local \
  /opt/mps-bench/bin/device_probe --json
```

`devices` 里的 UUID 必须正好是目标卡那一个。看到多张卡说明设备传递参数
限定得不够，此时任何隔离结论都不成立。
