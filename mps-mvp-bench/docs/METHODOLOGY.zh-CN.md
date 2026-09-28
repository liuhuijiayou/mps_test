# 测量与统计口径

本文说明本工程为什么这样测，以及刻意**不**这样测的地方。
大部分规则的存在理由是：按另一种更省事的做法，会得到一个看起来更好但站不住的结论。

## 一、负载模型

### 开环 vs 闭环

| 目的 | 加载方式 | 原因 |
| --- | --- | --- |
| 测延迟 | 开环（固定间隔 / 泊松 / 可重放突发 trace） | 请求按预定时刻发出，与系统当前是否忙无关 |
| 测容量 | 闭环（固定并发持续压满） | 要的是饱和吞吐 |

在线用例强制开环，离线用例强制闭环，配置层面就拒绝反过来配。

### 协调遗漏（coordinated omission）

端到端延迟从**计划到达时刻**算起，而不是从实际发出时刻算起：

```
e2e = response_done - planned_arrival        （而不是 response_done - actual_send）
```

闭环压测或"发送端自己也被拖慢"的情况下，实际发送时刻会被系统的拥塞推后。
以实际发送时刻为起点，等于把排队时间从统计里扣掉了——系统越过载，
测出来的延迟反而越好看。

`send_delay = actual_send - planned_arrival` 单独记录并出 P99，
它本身就是负载生成器是否跟得上的证据。

### trace 冻结

同一组对比里，到达序列由固定随机种子生成一次后复用。
两个配置如果用了不同的到达序列，测出来的差异里混了序列差异。

## 二、请求记录

每条请求记录统一字段，全程单调时钟加墙上时间戳：

```
run_id, case_id, round_index, worker_id, request_id,
planned_arrival_s,      # 计划到达
actual_send_s,          # 实际发出
service_received_s,     # 服务端收到
enqueued_s,             # 入队
exec_start_s,           # 开始执行
gpu_done_s,             # GPU 完成
response_done_s,        # 响应完成
gpu_ms,                 # CUDA event 测得的 GPU 时间
status, error, unit, samples
```

CPU 侧计时必须覆盖到 GPU 完成。异步 launch 返回不等于 kernel 跑完，
所以 worker 在计时窗口内做阻塞同步；GPU 侧时间另用 CUDA event 单独测量。

状态词固定为 `ok` / `error` / `timeout` / `rejected` / `output_error` /
`not_sent` / `incomplete`，构造时校验，写错会报错而不是产生一个没人统计的新状态。

## 三、统计规则

### 分位数

- 最近秩（nearest-rank）法，不做插值
- 样本为空返回 `null`，**不返回 0**
- `P99.9` 只在样本量达到 `measure.min_samples_p999` 时给出，
  否则标注 `insufficient_samples`
- 每个分位数附带其样本量

### 整体 P99 vs 各轮 P99 的均值

这两个数不一样，工程同时输出并明确标注：

| 字段 | 含义 |
| --- | --- |
| `pooled_p99_ms` | 全部轮次所有请求汇总后的 P99 ← **这才是整体 P99** |
| `per_round_p99_mean_ms` | 各轮 P99 的算术平均，仅作参考 |

对 P99 取平均会系统性低估尾延迟。报告里不会把后者当成整体 P99。

### 轮次是重复单位

置信区间在**轮次**层面计算（小样本 t 分布，95%）。
一轮里的一百万个请求不是一百万次独立实验。

**所有轮次都保留，包括退化的那一轮。** 悄悄丢掉表现差的轮次，
是混部结论被美化的最常见方式。

### 成功率与错误分类

分开统计：成功率、错误率、超时率、拒绝率。
超时计入请求总数但**不计入延迟分位数**——把超时当成一个"很慢但成功"的样本，
会让分位数变成一个混合了两种语义的数。

### 吞吐不跨单位相加

images/s 与 samples/s 不相加。单位不同时：

- 各任务吞吐分别呈现
- 需要单一数字时用归一化的 `weighted_speedup`：
  `Σᵢ (混部吞吐ᵢ / 独占吞吐ᵢ)`，并同时说明独占基线是哪个配置

`combine_throughput` 在单位不一致时直接返回 `combinable: false` 加原因，
而不是相加出一个没有物理意义的总数。

### SLO 吞吐

`slo_throughput` 只计延迟满足 SLO 阈值的成功请求。
阈值来自配置（`slo.*`），不硬编码。

### delta 定义

```
latency_delta    = P99_test / P99_ref - 1        # 正数 = 更差
throughput_delta = Q_test  / Q_ref  - 1          # 正数 = 更好
```

**每个结果同时给出相对 B0 与相对 B2 的 delta。** 只给相对 B0 会把
"同卡并发本身的代价"算成 MPS 的代价；只给相对 B2 会掩盖独占能力的损失。

## 四、基线设计

| 用例 | 含义 | 作用 |
| --- | --- | --- |
| `B0` | 非 MPS，单任务独占整卡 | 绝对能力上限 |
| `B1` | MPS，单任务但带限额 | 分离"限额成本"与"并发干扰成本" |
| `B2` | 非 MPS，同卡两任务并发 | **时间片竞争的对照** |
| `M0` | MPS，同卡两任务并发无限额 | MPS 本身的开销/收益 |

`B2` 是最容易被省掉也最不该省的一个。没有它，MPS 的任何"收益"都无法区分
是来自 MPS 还是来自"本来就只是两个任务排队"。

`B1` 是受限单跑，用于解释限额成本，**不等同于完整 GPU 独占能力**，
报告里会这样标注。

### 顺序效应

standard profile 用 ABBA 交替（`B2, M0, M0, B2`）而不是先跑完 A 再跑完 B。
GPU 温度、时钟、缓存状态都会随时间漂移；固定顺序会把漂移算成配置差异。

## 五、可观测口径

### 三个 SM 相关指标不可互换

| 指标 | 来源 | 含义 |
| --- | --- | --- |
| GPU util（device busy） | `nvidia-smi utilization.gpu` | 设备有活的时间占比 |
| SM active | DCGM 1002 | SM 上有 warp 驻留的时间占比 |
| SM occupancy | DCGM 1003 | 驻留 warp 占硬件上限的比例 |
| DRAM active | DCGM 1005 | DRAM 读写活动占比 |

一个只用了一个 SM 的 kernel 也能让 GPU util 显示 100%。
所以 GPU util 高**不能**用来论证算力被用满，更不能用来替代 SM active。

`sm_active` 高也不直接等于吞吐提升——它只说明 SM 上有 warp 驻留，
不说明这些 warp 在做有效工作。

### 不可用就是不可用

指标缺失（DCGM 返回 BLANK、`N/A`、`Not Supported`、NVML 数值哨兵如
`9223372036854775794`、NaN）时记 `null` 并附不可用原因。

- **不填 0**——0 是一个有意义的测量值，缺失不是
- **不用其他指标顶替**
- 图表在缺口处断开，不跨缺口连线

`DCGM_FI_PROF_SM_ACTIVE` 是强制验收项。默认 `require_sm_metrics=true`，
不可用时拒绝启动；显式降级后报告顶部标注本次不构成 SM 可观测验收。

### 采集开销对称

采集器按 run 启动一个，不是每 client 一个，保证对比双方承受相同的采集开销。

## 六、显存配额

### 机制

用 `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`（格式 `0=<MiB>M`）。

**不用** `torch.cuda.set_per_process_memory_fraction`——那只约束 PyTorch
缓存分配器，约束不到驱动层，证明不了 MPS 的配额能力。

50% 由运行时查询的物理总量换算；记录原始字节数、环境变量字面值与换算过程。

### 独立的原生探针

配额验证用独立的 CUDA driver API 探针（`cuMemAlloc`），而不是通过 PyTorch。
框架的缓存分配器会把分配行为包一层，看不清真实边界。

探针逐步分配、写入并校验 checksum（只分配不写入不能证明内存真的可用），
记录成功量、失败点、CUDA 错误码、释放结果与释放后能否继续计算。

### OOM 分类

OOM 本身不说明配额生效。判定分六类：

| 分类 | 条件 |
| --- | --- |
| `no_oom` | 没有 CUDA 错误 |
| `other_error` | 有 CUDA 错误但不是 OOM |
| `unexpected_oom` | 该 client 没配配额 |
| `unexpected_oom` | 远未达到配额就 OOM |
| `unexpected_oom` | 拿不到设备剩余量，无法证明还有物理余量 |
| `unexpected_oom` | 设备物理显存本身已耗尽 → 物理 OOM |
| `expected_quota_oom` | 达到配额边界**且**设备仍有物理余量 |

只有最后一类才算配额机制生效。"设备仍有余量但这个 client 分不到"——
这才是配额存在的证据。

context 与 CUDA 内部分配也计入 client 配额，所以用户缓冲区不会恰好占满配额值，
判定留了容差。

### 固定配额 ≠ 动态回收

`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 是固定上限。
本工程不会把它表述为动态显存回收或驱逐能力。

## 七、负例

五个候选：算力争抢、显存带宽/L2 争抢、高低优互扰、配额成本、有界长 kernel。

每个负例都要有**同样双任务的非 MPS 对照**，否则无法区分"MPS 导致的干扰"
和"两个任务同卡必然产生的干扰"。

没复现出来时，报告写**"本配置下未观察到"**，并列出扫过的参数范围。
不写成"不存在"——没观察到和不存在是两件事，尤其在只扫了有限参数的情况下。

## 八、故障

### 两个维度分开记

| 维度 | 取值 |
| --- | --- |
| `injection_status` | `applied` / `not_applied` |
| `peer_outcome` | `normal` / `degraded` / `failed` / `unknown` |

注入没生效的那次，**不能算作"隔离住了"**。
`propagation_statement` 会显式区分：`0/3 次观察到传播（不等于完全隔离）；
其中 1 次注入未生效，该部分不构成隔离证据`。

`peer_outcome=unknown` 也不算传播证据——不知道不等于没事。

### 传播表述

未观察到传播表述为 **"0/n 次"**，不表述为"完全隔离"。

驱动版本差异在这里很关键：R580 与 r610 的部分错误隔离能力不同，
在 R580 上观察到的隔离行为不能外推。

### 协作式安全退出

信号处理函数只置标志位（async-signal-safe），不在 handler 里做清理。
主循环看到标志后：**fence → drain → exit**。

记录 drain 时长、收到信号时的在途请求数、是否升级到 `terminate_client`。
`exited_cleanly` 要求 drain 成功**且**退出状态为 `safe_exit`。

`terminate_client` 用真实的 PID namespace 映射后的 host PID（`/proc/<pid>/status`
的 NSpid 加 `docker top` 交叉验证，绝不假设容器主进程是 PID 1），
并解析返回里的 CUDA 状态——返回码 0 但 CUDA 报错不算成功，超时也不算成功。

### 恢复阶梯

| 层级 | 动作 |
| --- | --- |
| R1 | 重建 client（MPS 与旁证任务都健康时） |
| R2 | 给 MPS 自恢复的机会，仍不行才重启我们自己的 MPS，再重建 client |
| R3 | 输出诊断，交人处理 |

**不执行自动 GPU reset。** 容器 RUNNING、MPS ACTIVE、业务恢复是三个不同事件，
分别记录时间点——容器起来了不等于业务恢复了。

## 九、synthetic 标注

离线负载可以用固定种子的合成权重与输入，但必须标注 `synthetic`。
报告顶部会声明：这类结果仅验证系统行为与相对性能，
**不代表真实模型精度或生产收益**。

## 十、结论词表

只有五个：

| 状态 | 含义 |
| --- | --- |
| `PASS` | 有证据且满足期望 |
| `FAIL` | 有证据且不满足期望 |
| `SKIP` | 本机不具备该能力（附探测证据） |
| `INCONCLUSIVE` | 跑了，但证据不足以支撑结论（如未能证明接入 MPS、SM 指标缺失） |
| `NOT_RUN` | 本次未执行 |

没有"大概可以"这一档。证据不足就是 `INCONCLUSIVE`，不会降级成 `PASS`。
