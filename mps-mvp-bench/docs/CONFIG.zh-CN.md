# 配置说明

本文前半部分是人工说明；末尾的完整配置项表格由
`mps-bench schema --markdown` 从代码里的 schema 直接生成，因此不会与实现脱节。

重新生成方式见文末。

## 覆盖优先级

后者覆盖前者：

```
内置默认值  →  --config 的 YAML（可多次传，按顺序）  →  --profile 预设
            →  用例 YAML（cases/**/*.yaml）  →  --set key=value（可多次传）
```

两条硬性约束：

- **未知配置项直接报错**，不静默忽略。写错一个键名不会让你拿着默认值跑完一整轮。
- 类型、范围、枚举值在加载时校验，而不是等到运行到一半才崩。

查看某个用例最终生效的完整配置：

```bash
mps-bench plan --config configs/runtime.yaml --case-id HH70 --print-config
```

## 必须由你填写的两项

工程刻意不为这两项提供默认值，因为猜错的代价是动到别人的卡：

| 配置项 | 说明 |
| --- | --- |
| `gpu.uuid` | 目标卡 UUID。**不要填已被其他服务占用的那张。** |
| `docker.runtime_args` | 设备传递参数，argv 列表形式。 |

`docker.runtime_args` 的取值取决于你这台机器的容器运行时配置，例如：

```yaml
docker:
  runtime_args: ["--gpus", "device=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"]
```

工程会拒绝以下写法：

- `--gpus all` 或 `--gpus=all` — 会把所有卡暴露给容器，包括别人在用的
- `--privileged`
- 挂载 `/var/run/docker.sock`
- 含 shell 元字符的参数（所有命令以 argv 列表执行，不经过 shell）

## profile 预设

| profile | warmup | 测量时长 | 轮数 | 用途 |
| --- | --- | --- | --- | --- |
| `smoke` | 5s | 15s | 1 | **只验链路是否打通，不作为任何性能证据。** 该 profile 下 `require_sm_metrics=false`、`allow_degraded=true`。 |
| `standard` | 30s | 120s | 5 | 正式测量，ABBA 交替，`require_sm_metrics=true`。 |

## 几组容易搞错的配置

### `telemetry.require_sm_metrics` 与 `telemetry.allow_degraded`

`DCGM_FI_PROF_SM_ACTIVE`（field 1002）是强制验收指标。

- `require_sm_metrics=true`（默认）且该指标不可用 → **直接拒绝启动**，
  并给出不可用原因。不会用 GPU util 顶替，也不会填 0。
- 确实需要在缺该指标的机器上采集时，必须显式设置 `allow_degraded=true`。
  此时报告会在顶部显著标注"可观测验收缺项"，并声明本次结果不构成 SM 可观测验收。

三个语义不同的指标，工程始终分开记录，不互相替代：

| 指标 | 来源 | 含义 |
| --- | --- | --- |
| `gpu_util_device_busy_pct` | `nvidia-smi utilization.gpu` | 设备 busy 时间占比 |
| `sm_active` | DCGM 1002 | SM 上有 warp 驻留的时间占比 |
| `sm_occupancy` | DCGM 1003 | 驻留 warp 占最大值的比例 |
| `dram_active` | DCGM 1005 | DRAM 读写活动占比 |

`sm_active` 高不等于算术单元满载，也不证明吞吐提升。

### `clients.*.mps.*` 只在 MPS 模式下允许配置

`case.mode` 为 `nonmps_*` 时，配置任何 `mps.active_thread_percentage`、
`mps.priority`、`mps.pinned_device_mem_limit_fraction`、`mps.static_partition`
都会直接报错。原因是这些环境变量在非 MPS 模式下无效，**留着它们会让人误以为
"非 MPS 基线也带了配额"**，从而把对比做废。

### 显存配额用的是 MPS 机制，不是框架接口

配额通过 `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 施加，格式为 `0=<MiB>M`。

**不使用** `torch.cuda.set_per_process_memory_fraction`——那只约束 PyTorch 的
缓存分配器，不约束 CUDA 驱动层分配，证明不了 MPS 的配额能力。

50% 由运行时查询到的物理总量换算，报告同时记录原始字节数、
环境变量字面值和换算过程，不只记一个百分比。

注意 context 与 CUDA 内部分配也计入 client 配额，所以用户缓冲区不会恰好
占满配额值。

### 静态 SM 分区是独立探测的能力

`mps.static_partitioning.enabled=true` 只表示"希望使用"。工程会探测控制接口
是否真的提供该命令；不提供时 `HH-S` 用例记为 **SKIP** 并写明原因。

**不会**用 `ACTIVE_THREAD_PERCENTAGE=50`、context 亲和性或 MIG 冒充静态分区。

### 在线与离线的到达模式互斥

- `mode: online` 必须用开环到达（`fixed_interval` / `poisson` / `burst_trace`），
  且必须指定 `target_qps`。闭环压测测不出真实排队延迟。
- `mode: offline` 必须用 `closed_loop`。

### 破坏性故障用例的双开关

`F4`–`F9` 需要同时满足：

1. 配置 `faults.allow_disruptive=true`
2. CLI 传 `--allow-disruptive`
3. CLI 传 `--confirm-gpu-uuid <UUID>` 且与 `gpu.uuid` 完全一致

三者缺一即拒绝执行。仓库内这些用例默认 `case.enabled: false`。

### 判定阈值来自配置

`slo.*` 与 `case.expect` 决定 PASS/FAIL，阈值不硬编码在代码里。
`case.expect` 声明的期望若不满足则记 FAIL，不会静默通过。

## 完整配置项表格

下表由 schema 生成。`生效阶段`说明该项在什么时候被读取——尤其注意
`client 启动时生效`的项：MPS 相关环境变量必须在 CUDA 初始化前设置，
改动后必须新建 client 才会生效，对运行中的 client 改这些值没有作用。

重新生成：

```bash
mps-bench schema --markdown
```

<!-- BEGIN GENERATED SCHEMA -->
<!-- 以下内容由 `mps-bench schema --markdown` 生成 -->

### `case.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `case.description` | str | `` | — | 测量过程中读取 | — |
| `case.enabled` | bool | `True` | — | 测量过程中读取 | — |
| `case.expect` | dict | `{}` | — | 只影响报告/判定 | 用例声明的期望，例如 {low_priority_oom: true}；不满足即 FAIL 而非静默通过 |
| `case.family` | str | `baseline` | `baseline` \| `high_high` \| `high_low` \| `memory` \| `negative` \| `fault` | 测量过程中读取 | 决定结果归类与报告章节 |
| `case.fault` | str/null | 无 | `F1` \| `F2` \| `F3` \| `F4` \| `F5` \| `F6` \| `F7` \| `F8` \| `F9` \| `None` | 测量过程中读取 | — |
| `case.id` | str/null | 无 | — | 测量过程中读取 | 用例 ID，例如 B0/B2/M0/HH70/HL-PC |
| `case.mode` | str | `mps_concurrent` | `nonmps_solo` \| `nonmps_concurrent` \| `mps_solo` \| `mps_concurrent` | 测量过程中读取 | 是否启用 MPS、单跑还是同卡并发 |
| `case.reference` | list | `['B0', 'B2']` | — | 只影响报告/判定 | 报告必须同时展示相对 B0 与 B2 |
| `case.solo_client` | str/null | 无 | `a` \| `b` \| `None` | 测量过程中读取 | *_solo 模式下实际运行的 client |
| `case.sweep` | list | `[]` | — | 测量过程中读取 | 可审计的参数扫描；不展开全参数笛卡尔积 |

### `clients.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `clients.a.container.cpuset` | str/null | 无 | — | 测量过程中读取 | 固定 CPU 亲和性，保持跨模式一致 |
| `clients.a.container.memory` | str/null | 无 | — | 测量过程中读取 | — |
| `clients.a.container.name_suffix` | str | `a` | — | 测量过程中读取 | — |
| `clients.a.load.arrival` | str | `poisson` | `closed_loop` \| `fixed_interval` \| `poisson` \| `burst_trace` | 测量过程中读取 | — |
| `clients.a.load.burst_trace` | list | `[]` | — | 测量过程中读取 | 可重复突发 trace: [{t, qps}]；arrival=burst_trace 时必填 |
| `clients.a.load.concurrency` | int | `8` | 1 ~ 4096 | 测量过程中读取 | offline 模式的闭环并发；online 模式为发送端上限 |
| `clients.a.load.port` | int | `18801` | 1024 ~ 65535 | 测量过程中读取 | — |
| `clients.a.load.queue_limit` | int | `1024` | 1 ~ 1000000 | 测量过程中读取 | 服务端队列上限，超出记为 rejected |
| `clients.a.load.target_qps` | int/float/null | `50.0` | 0.01 ~ None | 测量过程中读取 | [req/s] online 模式必填；offline 模式忽略 |
| `clients.a.mode` | str | `online` | `online` \| `offline` | 容器启动时生效 | online=开环固定到达率(测延迟), offline=闭环饱和(测容量) |
| `clients.a.model.batch_size` | int | `8` | 1 ~ 4096 | 容器启动时生效 | [samples] |
| `clients.a.model.cpu_threads` | int | `4` | 1 ~ 256 | 容器启动时生效 | [threads] |
| `clients.a.model.input_pool_size` | int | `64` | 1 ~ 100000 | 容器启动时生效 | [batches] 输入池大小；过小会导致只压 L2 而非 HBM |
| `clients.a.model.input_size` | int | `224` | 32 ~ 1024 | 容器启动时生效 | [px] |
| `clients.a.model.name` | str | `resnet18` | — | 容器启动时生效 | image_inference: resnet18/resnet50 |
| `clients.a.model.precision` | str | `fp16` | `fp32` \| `tf32` \| `fp16` \| `bf16` | 容器启动时生效 | — |
| `clients.a.model.weights_path` | str/null | 无 | — | 容器启动时生效 | 留空=离线固定种子合成权重，结果标注 synthetic |
| `clients.a.mps.active_thread_percentage` | int/null | 无 | 1 ~ 100 | client 启动时生效（改动后必须新建 client） | [%] CUDA_MPS_ACTIVE_THREAD_PERCENTAGE；执行资源上限，不是固定 SM 预留，也不保证抢占 |
| `clients.a.mps.pinned_device_mem_limit_fraction` | int/float/null | 无 | 0.01 ~ 1.0 | client 启动时生效（改动后必须新建 client） | 按运行时物理总显存的比例生成 CUDA_MPS_PINNED_DEVICE_MEM_LIMIT |
| `clients.a.mps.priority` | str/null | 无 | `NORMAL` \| `BELOW_NORMAL` \| `None` | client 启动时生效（改动后必须新建 client） | CUDA_MPS_CLIENT_PRIORITY: NORMAL=0 / BELOW_NORMAL=1 |
| `clients.a.mps.static_partition` | str/null | 无 | — | client 启动时生效（改动后必须新建 client） | 绑定到的静态分区名；需 mps.static_partitioning.enabled |
| `clients.a.recsys.bottom_mlp` | list | `[512, 256, 64]` | — | 容器启动时生效 | — |
| `clients.a.recsys.dense_features` | int | `128` | 1 ~ 8192 | 容器启动时生效 | — |
| `clients.a.recsys.embedding_dim` | int | `64` | 1 ~ 1024 | 容器启动时生效 | — |
| `clients.a.recsys.index_distribution` | str | `zipf` | `uniform` \| `zipf` | 容器启动时生效 | — |
| `clients.a.recsys.indices_per_sample` | int | `32` | 1 ~ 4096 | 容器启动时生效 | — |
| `clients.a.recsys.num_tables` | int | `16` | 1 ~ 1024 | 容器启动时生效 | — |
| `clients.a.recsys.rows_per_table` | int | `200000` | 16 ~ None | 容器启动时生效 | [rows] |
| `clients.a.recsys.top_mlp` | list | `[512, 256, 1]` | — | 容器启动时生效 | — |
| `clients.a.recsys.working_set_rows` | int/null | 无 | 1 ~ None | 容器启动时生效 | [rows] 索引工作集；必须可配，避免反复命中极小固定集合 |
| `clients.a.recsys.zipf_s` | int/float | `1.1` | 0.0 ~ 5.0 | 容器启动时生效 | — |
| `clients.a.role` | str | `high` | `high` \| `low` | 测量过程中读取 | high=高优, low=低优；决定报告中的 QoS 归类 |
| `clients.a.workload` | str | `image_inference` | `image_inference` \| `recsys_scoring` | 容器启动时生效 | — |
| `clients.b.container.cpuset` | str/null | 无 | — | 测量过程中读取 | 固定 CPU 亲和性，保持跨模式一致 |
| `clients.b.container.memory` | str/null | 无 | — | 测量过程中读取 | — |
| `clients.b.container.name_suffix` | str | `b` | — | 测量过程中读取 | — |
| `clients.b.load.arrival` | str | `poisson` | `closed_loop` \| `fixed_interval` \| `poisson` \| `burst_trace` | 测量过程中读取 | — |
| `clients.b.load.burst_trace` | list | `[]` | — | 测量过程中读取 | 可重复突发 trace: [{t, qps}]；arrival=burst_trace 时必填 |
| `clients.b.load.concurrency` | int | `8` | 1 ~ 4096 | 测量过程中读取 | offline 模式的闭环并发；online 模式为发送端上限 |
| `clients.b.load.port` | int | `18802` | 1024 ~ 65535 | 测量过程中读取 | — |
| `clients.b.load.queue_limit` | int | `1024` | 1 ~ 1000000 | 测量过程中读取 | 服务端队列上限，超出记为 rejected |
| `clients.b.load.target_qps` | int/float/null | `50.0` | 0.01 ~ None | 测量过程中读取 | [req/s] online 模式必填；offline 模式忽略 |
| `clients.b.mode` | str | `online` | `online` \| `offline` | 容器启动时生效 | online=开环固定到达率(测延迟), offline=闭环饱和(测容量) |
| `clients.b.model.batch_size` | int | `8` | 1 ~ 4096 | 容器启动时生效 | [samples] |
| `clients.b.model.cpu_threads` | int | `4` | 1 ~ 256 | 容器启动时生效 | [threads] |
| `clients.b.model.input_pool_size` | int | `64` | 1 ~ 100000 | 容器启动时生效 | [batches] 输入池大小；过小会导致只压 L2 而非 HBM |
| `clients.b.model.input_size` | int | `224` | 32 ~ 1024 | 容器启动时生效 | [px] |
| `clients.b.model.name` | str | `resnet18` | — | 容器启动时生效 | image_inference: resnet18/resnet50 |
| `clients.b.model.precision` | str | `fp16` | `fp32` \| `tf32` \| `fp16` \| `bf16` | 容器启动时生效 | — |
| `clients.b.model.weights_path` | str/null | 无 | — | 容器启动时生效 | 留空=离线固定种子合成权重，结果标注 synthetic |
| `clients.b.mps.active_thread_percentage` | int/null | 无 | 1 ~ 100 | client 启动时生效（改动后必须新建 client） | [%] CUDA_MPS_ACTIVE_THREAD_PERCENTAGE；执行资源上限，不是固定 SM 预留，也不保证抢占 |
| `clients.b.mps.pinned_device_mem_limit_fraction` | int/float/null | 无 | 0.01 ~ 1.0 | client 启动时生效（改动后必须新建 client） | 按运行时物理总显存的比例生成 CUDA_MPS_PINNED_DEVICE_MEM_LIMIT |
| `clients.b.mps.priority` | str/null | 无 | `NORMAL` \| `BELOW_NORMAL` \| `None` | client 启动时生效（改动后必须新建 client） | CUDA_MPS_CLIENT_PRIORITY: NORMAL=0 / BELOW_NORMAL=1 |
| `clients.b.mps.static_partition` | str/null | 无 | — | client 启动时生效（改动后必须新建 client） | 绑定到的静态分区名；需 mps.static_partitioning.enabled |
| `clients.b.recsys.bottom_mlp` | list | `[512, 256, 64]` | — | 容器启动时生效 | — |
| `clients.b.recsys.dense_features` | int | `128` | 1 ~ 8192 | 容器启动时生效 | — |
| `clients.b.recsys.embedding_dim` | int | `64` | 1 ~ 1024 | 容器启动时生效 | — |
| `clients.b.recsys.index_distribution` | str | `zipf` | `uniform` \| `zipf` | 容器启动时生效 | — |
| `clients.b.recsys.indices_per_sample` | int | `32` | 1 ~ 4096 | 容器启动时生效 | — |
| `clients.b.recsys.num_tables` | int | `16` | 1 ~ 1024 | 容器启动时生效 | — |
| `clients.b.recsys.rows_per_table` | int | `200000` | 16 ~ None | 容器启动时生效 | [rows] |
| `clients.b.recsys.top_mlp` | list | `[512, 256, 1]` | — | 容器启动时生效 | — |
| `clients.b.recsys.working_set_rows` | int/null | 无 | 1 ~ None | 容器启动时生效 | [rows] 索引工作集；必须可配，避免反复命中极小固定集合 |
| `clients.b.recsys.zipf_s` | int/float | `1.1` | 0.0 ~ 5.0 | 容器启动时生效 | — |
| `clients.b.role` | str | `high` | `high` \| `low` | 测量过程中读取 | high=高优, low=低优；决定报告中的 QoS 归类 |
| `clients.b.workload` | str | `image_inference` | `image_inference` \| `recsys_scoring` | 容器启动时生效 | — |

### `compute_mode.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `compute_mode.allow_change` | bool | `False` | — | 预检阶段读取 | 默认不改 compute mode；置 True 需用户授权，仅针对目标 UUID 并在 cleanup 恢复 |
| `compute_mode.require_default_for_nonmps` | bool | `True` | — | 预检阶段读取 | 非 MPS 双进程基线要求允许多进程；EXCLUSIVE_PROCESS 下基线失败不得算作 MPS 收益 |

### `docker.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `docker.binary` | str | `docker` | — | 测量过程中读取 | — |
| `docker.extra_args` | list | `[]` | — | 测量过程中读取 | 附加 docker argv 片段（如 --cpuset-cpus） |
| `docker.image` | str | `mps-mvp-bench:0.1.0` | — | 测量过程中读取 | — |
| `docker.ipc` | str/null | 无 | — | 测量过程中读取 | — |
| `docker.mount_docker_socket` | bool | `False` | — | 测量过程中读取 | 默认 False，禁止把 docker socket 挂进业务容器 |
| `docker.network` | str | `none` | — | 测量过程中读取 | 业务容器网络。在线入口默认只对实验网络开放 |
| `docker.privileged` | bool | `False` | — | 测量过程中读取 | 默认 False；置 True 需在 RUNBOOK 中说明理由 |
| `docker.pull` | bool | `False` | — | 测量过程中读取 | 是否允许 docker pull；离线环境保持 False |
| `docker.runtime_args` | list | `[]` | — | 测量过程中读取 | 用户指定的 GPU runtime/设备透传参数 argv 片段，例如 ['--runtime=nvidia','--gpus','device=GPU-...']。不写死、不默认 --gpus all |
| `docker.shm_size` | str | `1g` | — | 测量过程中读取 | — |
| `docker.user` | str/null | 无 | — | 测量过程中读取 | 容器 UID[:GID]，需与宿主机 MPS pipe 所属 UID 兼容 |

### `faults.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `faults.allow_disruptive` | bool | `False` | — | 测量过程中读取 | 必须同时通过 CLI --allow-disruptive 与目标 UUID 确认 |
| `faults.drain_timeout_s` | int/float | `15` | 0 ~ 600 | 测量过程中读取 | [s] |
| `faults.enabled` | bool | `False` | — | 测量过程中读取 | — |
| `faults.inject_after_s` | int/float | `20` | 0 ~ 3600 | 测量过程中读取 | [s] |
| `faults.long_kernel_ms` | int/float | `2000` | 1 ~ 60000 | 测量过程中读取 | [ms] 有界长 kernel 时限；禁止无限循环 |
| `faults.observe_window_s` | int/float | `30` | 1 ~ 3600 | 测量过程中读取 | [s] 先观测故障影响再恢复，避免自动重启掩盖传播 |
| `faults.recovery_levels` | list | `['R1', 'R2']` | — | 测量过程中读取 | 按最小故障域推进；R3 只输出人工诊断建议，不自动 GPU reset |
| `faults.recovery_timeout_s` | int/float | `120` | 1 ~ 3600 | 测量过程中读取 | [s] |
| `faults.repeats` | int | `3` | 1 ~ 100 | 测量过程中读取 | [times] |

### `gpu.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `gpu.expect_total_memory_mib` | int/null | 无 | 1 ~ None | 测量过程中读取 | [MiB] 可选断言：运行时查询到的总显存必须等于该值 |
| `gpu.uuid` | str/null | 无 | — | 预检阶段读取 | 目标 GPU UUID，形如 GPU-xxxxxxxx-....。真实 GPU 执行前必填 |

### `measure.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `measure.drain_timeout_s` | int/float | `30` | 0 ~ 600 | 测量过程中读取 | [s] 测量窗口结束后等待在途请求 drain 的时间；超时请求单独计数 |
| `measure.duration_s` | int/float | `120` | 1 ~ 7200 | 测量过程中读取 | [s] |
| `measure.interleave` | str | `ABBA` | `none` \| `ABBA` \| `random` | 测量过程中读取 | 比较模式的轮次顺序，降低时间漂移偏差 |
| `measure.min_samples_p999` | int | `100000` | 1000 ~ None | 只影响报告/判定 | [requests] 低于该样本数时 P99.9 标记为样本不足而不是给出数值 |
| `measure.profile` | str | `standard` | `smoke` \| `standard` | 测量过程中读取 | — |
| `measure.repeats` | int | `5` | 1 ~ 100 | 测量过程中读取 | [rounds] |
| `measure.request_timeout_s` | int/float | `5.0` | 0.001 ~ 600 | 测量过程中读取 | [s] |
| `measure.seed` | int | `20251001` | 0 ~ None | 测量过程中读取 | 到达 trace / 合成输入 / 权重初始化共用的主种子 |
| `measure.tolerance` | int/float | `0.001` | — | 测量过程中读取 | 输出一致性比较容差，固定值；不得因 MPS 结果不同而放宽 |
| `measure.total_budget_s` | int/float/null | 无 | 1 ~ None | 测量过程中读取 | [s] 总时长上限；展开用例超预算时拒绝执行并提示缩小子集 |
| `measure.warmup_s` | int/float | `30` | 0 ~ 3600 | 测量过程中读取 | [s] |

### `memory.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `memory.expect_nonmps_no_quota` | bool | `True` | — | 测量过程中读取 | 非 MPS 对照只在物理余量足够时验证不被配额拦截，不为测 OOM 耗尽整卡 |
| `memory.probe_step_mib` | int | `256` | 1 ~ 16384 | 测量过程中读取 | [MiB] |
| `memory.probe_touch` | bool | `True` | — | 测量过程中读取 | 分配后触碰内存，避免只记账不落地 |
| `memory.safety_margin_mib` | int | `1024` | 0 ~ None | 测量过程中读取 | [MiB] 业务自身安全工作集预算，不把测试工作集压到配额边缘 |
| `memory.total_bytes_source` | str | `runtime` | `runtime` | 测量过程中读取 | 50% 一律按运行时查询到的物理总显存计算 |

### `meta.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `meta.notes` | str | `` | — | 测量过程中读取 | 自由文本，写入 manifest |
| `meta.project` | str | `mps-mvp-bench` | — | 测量过程中读取 | 项目标识，用于容器 label 与结果目录前缀 |
| `meta.run_id` | str/null | 无 | — | 测量过程中读取 | 留空则自动生成 <utc时间戳>-<随机后缀> |

### `mps.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `mps.allow_adopt_existing` | bool | `False` | — | MPS 启动时生效 | 默认 False：发现非本实验的 MPS 实例则停止并报告，不接管 |
| `mps.control_binary` | str | `nvidia-cuda-mps-control` | — | MPS 启动时生效 | — |
| `mps.enabled` | bool | `True` | — | MPS 启动时生效 | 由 case.mode 决定，通常不手工设置 |
| `mps.manage_on_host` | bool | `True` | — | MPS 启动时生效 | True=宿主机启动 MPS daemon（驱动版本匹配），容器共享 pipe 路径 |
| `mps.start_timeout_s` | int/float | `30` | 1 ~ 600 | MPS 启动时生效 | [s] |
| `mps.static_partitioning.enabled` | bool | `False` | — | MPS 启动时生效 | 静态 SM 分区用例专用；与 active_thread_percentage 分开配置 |
| `mps.static_partitioning.partitions` | list | `[]` | — | MPS 启动时生效 | 每个元素 {name, sm_count}；不支持则用例 SKIP |
| `mps.stop_timeout_s` | int/float | `60` | 1 ~ 600 | 清理阶段读取 | [s] |

### `paths.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `paths.model_mount` | str/null | 无 | — | 测量过程中读取 | 宿主机权重/输入目录，容器内挂到 /assets（只读） |
| `paths.mps_log_dir` | str | `/tmp/mps-mvp-bench/log` | — | MPS 启动时生效 | — |
| `paths.mps_pipe_dir` | str | `/tmp/mps-mvp-bench/pipe` | — | MPS 启动时生效 | 本实验专属 MPS pipe 目录，禁止用默认 /tmp/nvidia-mps |
| `paths.results_dir` | str | `results` | — | 测量过程中读取 | 结果根目录，run 会在其下建 <run_id>/ |

### `safety.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `safety.abort_on_foreign_gpu_process` | bool | `True` | — | 预检阶段读取 | 目标卡存在非本项目 GPU 进程则拒绝；每轮前重新检查 |
| `safety.health_check_between_rounds` | bool | `True` | — | 测量过程中读取 | — |
| `safety.lock_dir` | str | `/tmp/mps-mvp-bench/locks` | — | 测量过程中读取 | GPU UUID 级互斥锁目录 |
| `safety.stop_on_unhealthy` | bool | `True` | — | 测量过程中读取 | 异常后无法确认 GPU 健康则停止后续实验，不自动重试掩盖污染 |

### `slo.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `slo.latency_p99_ms` | int/float/null | 无 | 0 ~ None | 只影响报告/判定 | [ms] |
| `slo.max_error_rate` | int/float | `0.0` | 0 ~ 1 | 只影响报告/判定 | — |
| `slo.max_latency_delta` | int/float/null | 无 | — | 只影响报告/判定 | 高优 P99 劣化上限，例如 0.2 表示 +20%；留空则只报告不判定 |
| `slo.min_throughput_delta` | int/float/null | 无 | — | 只影响报告/判定 | 吞吐收益阈值；不得在代码里硬编码 |
| `slo.reference` | str | `B2` | `B0` \| `B2` | 只影响报告/判定 | 判定所用参考基线；报告始终同时展示相对 B0 与 B2 |

### `telemetry.*`

| 配置项 | 类型 | 默认值 | 取值范围 | 生效阶段 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `telemetry.allow_degraded` | bool | `False` | — | 测量过程中读取 | 显式降级采集；报告必须显著标记缺项 |
| `telemetry.dcgm_fields` | list | `[1002, 1003, 1005]` | — | 测量过程中读取 | 1002 SM_ACTIVE(必做) / 1003 SM_OCCUPANCY / 1005 DRAM_ACTIVE |
| `telemetry.dcgmi_binary` | str | `dcgmi` | — | 测量过程中读取 | — |
| `telemetry.enabled` | bool | `True` | — | 测量过程中读取 | — |
| `telemetry.nsight.enabled` | bool | `False` | — | 测量过程中读取 | Nsight 仅 opt-in 单独运行，不与正式性能结果混用 |
| `telemetry.nvidia_smi_binary` | str | `nvidia-smi` | — | 测量过程中读取 | — |
| `telemetry.require_sm_metrics` | bool | `True` | — | 测量过程中读取 | 标准验收默认 True；SM active 不可用时不得宣称完成可观测验收 |
| `telemetry.sample_interval_s` | int/float | `1.0` | 0.05 ~ 60 | 测量过程中读取 | [s] |

<!-- END GENERATED SCHEMA -->
