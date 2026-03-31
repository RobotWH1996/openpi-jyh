# Real-Time Chunking (RTC) 模块

基于论文 [Real-Time Chunking](https://www.physicalintelligence.company/research/real_time_chunking)，

RTC 解决的核心问题：action chunking 策略在 chunk 边界处会产生动作不连续（抖动），
RTC 通过在 flow matching 去噪过程中加入 **引导修正**，让新生成的 chunk 与上一个 chunk 的剩余动作保持平滑过渡。

---

## 目录结构

```
src/openpi/rtc
├── __init__.py
├── README.md                 # 本文件
├── configuration_rtc.py      # RTCConfig 配置类 + RTCAttentionSchedule 枚举
├── modeling_rtc_jax.py       # RTCProcessorJax —— 核心 RTC 处理器 (JAX)
├── debug_tracker_jax.py      # 调试追踪器，记录每步去噪的中间状态
├── latency_tracker.py        # 推理延迟记录器，用于估计 inference_delay
└── action_queue.py           # RTC 动作队列 (NumPy)，客户端侧使用
```

---

## 各文件功能

### `configuration_rtc.py`

定义 `RTCConfig` 数据类和 `RTCAttentionSchedule` 枚举。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `enabled` | `False` | 是否启用 RTC |
| `prefix_attention_schedule` | `LINEAR` | 软 masking 衰减方式：`LINEAR`（线性衰减）、`EXP`（指数衰减）、`ZEROS`（仅冻结区）、`ONES`（全约束） |
| `max_guidance_weight` | `10.0` | guidance weight 的上限，防止去噪早期阶段修正量过大 |
| `execution_horizon` | `10` | 执行窗口大小，即新旧 chunk 重叠区间的终点 |
| `debug` | `False` | 是否启用调试追踪 |


### `modeling_rtc_jax.py`

核心类 `RTCProcessorJax`，包含两个关键方法：

#### `get_prefix_weights(start, end, total)`

生成长度为 `total` 的权重向量，用于软 masking：

```
索引:    [0, ..., start-1]  [start, ..., end-1]  [end, ..., total-1]
权重:    [  1.0 (冻结区)  ]  [  衰减 (中间区)  ]  [ 0.0 (全新生成) ]
```

- `start` = `inference_delay`（推理延迟步数，这些动作在推理完成前已执行，必须冻结）
- `end` = `execution_horizon`（执行窗口终点，超出此范围的是全新生成的动作）

#### `compute_guidance(x_t, time, prev_chunk_left_over, inference_delay, execution_horizon, model_fn)`

在 flow matching 去噪的每一步中，替代原始的 `v_t = model_fn(x_t)`:

1. 正向传播得到 `v_t` 和预测目标 `x1 = x_t - time * v_t`
2. 计算误差 `err = (prev_chunk_left_over - x1) * weights`
3. 通过 VJP (反向传播) 计算修正量 `correction`
4. 计算时间相关的 `guidance_weight`（去噪早期大、后期小）
5. 返回修正后的速度场：`v_t_guided = v_t - guidance_weight * correction`


### `action_queue.py`

线程安全的动作队列 `ActionQueue`，客户端（`cr100_ros2_client.py`）使用。

核心方法：
- `get()` — 消费一个动作
- `get_left_over()` — 获取未消费的原始动作（传给下一次推理作为 `prev_chunk_left_over`）
- `merge(new_original, new_processed, estimated_delay, action_index_before)` — 将新 chunk 合并到队列
  - RTC 模式：基于真实延迟 `real_delay` 截断新 chunk，丢弃已执行的帧，用新动作替换整个队列
  - 非 RTC 模式：将新动作追加到队列尾部
  - 可选 `blend_steps`：在截断点附近做线性插值混合


### `latency_tracker.py`

记录最近 N 次推理延迟，提供 `max()` 和 `percentile()` 查询，用于估计下一次推理的 `inference_delay`。


### `debug_tracker_jax.py`

调试用追踪器，记录 RTC 去噪过程中每一步的 `x_t`, `v_t`, `correction`, `err`, `weights`, `guidance_weight` 等。
通过 `RTCConfig(debug=True)` 启用。

---

## 对现有代码的改动

所有改动位置都用 `[RTC]` 注释标记，可以全局搜索 `[RTC]` 快速定位。

### 1. `src/openpi/models/pi0.py` — 去噪循环中注入 RTC 引导

| 改动 | 说明 |
|------|------|
| `__init__` 签名 | 新增 `rtc_processor` 参数，保存为 `self.rtc_processor` |
| `sample_actions` 签名 | 新增 `**kwargs`，接收 `inference_delay`, `prev_chunk_left_over`, `execution_horizon` |
| `step()` 函数 | 原始内联的模型前向代码被抽取为 `get_v_t()` 函数；`step()` 内根据 `self._rtc_enabled` 决定走 `compute_guidance` 还是直接 `get_v_t` |

原始 `step()` 中的模型前向逻辑完全保留在 `get_v_t()` 中，语义不变。

### 2. `src/openpi/models/pi0_config.py` — 配置层支持 RTC

| 改动 | 说明 |
|------|------|
| 新增字段 | `rtc_config: RTCConfig | None = None` |
| `create()` 方法 | 若 `rtc_config` 不为 None，创建 `RTCProcessorJax` 传入 `Pi0()` |

### 3. `src/openpi/policies/policy.py` — 透传 RTC 参数

| 改动 | 说明 |
|------|------|
| `infer()` 签名 | 新增 `**kwargs`，将 int/float 转为 `jnp.asarray`，ndarray 转为 JAX array 并加 batch 维度 |
| 返回值 | 在 `output_transform` 之前保存 `actions_original`，结果中多返回此字段（RTC 需要未经变换的原始动作作 `prev_chunk_left_over`） |

### 4. `src/openpi/serving/websocket_policy_server.py` — 服务端提取 RTC 参数

| 改动 | 说明 |
|------|------|
| `_handler()` | 从 `obs` 中提取 `__rtc_kwargs__`，透传给 `policy.infer(obs, **rtc_kwargs)` |

### 5. `scripts/serve_policy.py` — 启动参数

| 改动 | 说明 |
|------|------|
| `Args` 新增 | `--rtc`, `--rtc-execution-horizon`, `--rtc-max-guidance-weight`, `--rtc-schedule` |
| `create_policy()` | 若 `--rtc` 启用，用 `_apply_rtc_config()` 把 `RTCConfig` 注入到 `Pi0Config` 中 |

### 6. `examples/cr100_open_door/cr100_ros2_client.py` — 客户端 RTC 支持

| 改动 | 说明 |
|------|------|
| `send_observation()` | 新增 `**kwargs`，打包为 `obs["__rtc_kwargs__"]` 发送到服务端 |
| `CR100ROS2ClientNode.__init__` | 新增 `rtc_enabled`, `execution_horizon`, `rtc_blend_steps` 参数 |
| 新增 `_rtc_inference_loop()` | 后台线程，当队列剩余动作不足时异步发起推理 |
| 新增 `_control_callback_rtc()` | RTC 模式的控制回调：只从 `ActionQueue` 取动作发布，推理由后台线程完成 |
| 新增 CLI 参数 | `--rtc`, `--execution-horizon`, `--rtc-blend-steps` |

---

## 运行方法

### 前提条件

- 已训练好 checkpoint（比如 `pi05_cr100_open_door_lora`）
- 服务端和客户端可以是同一台机器或不同机器

### 第一步：启动策略服务（服务端）

#### 不使用 RTC（传统模式，与之前完全一样）

```bash
cd /home/x100/wh/openpi

uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_cr100_open_door_lora \
    --policy.dir /home/x100/wh/openpi/checkpoints/pi05_cr100_open_door_lora/pi05_cr100_open_door_lora/5000
```

#### 使用 RTC（新增）

```bash
cd /home/x100/wh/openpi

uv run scripts/serve_policy.py --port 8000 \
    --rtc \
    --rtc-execution-horizon 10 \
    --rtc-schedule LINEAR \
    --rtc-max-guidance-weight 10.0 \
    policy:checkpoint \
    --policy.config pi05_cr100_open_door_lora \
    --policy.dir /home/x100/wh/openpi/checkpoints/pi05_cr100_open_door_lora/pi05_cr100_open_door_lora/5000
```

服务端 RTC 参数说明：

| 参数 | 说明 |
|------|------|
| `--rtc` | 启用 RTC（不加则完全兼容原有行为） |
| `--rtc-execution-horizon 10` | 执行窗口 = 10 步（与 `action_horizon` 一致时效果最佳） |
| `--rtc-schedule LINEAR` | 软 masking 衰减方式，可选 `LINEAR` / `EXP` / `ZEROS` / `ONES` |
| `--rtc-max-guidance-weight 10.0` | guidance weight 上限 |

> 也可以用 `pi0_cr100_action_expert_lora_freeze_vlm` 等其他配置名，只要 checkpoint 对应即可。


### 第二步：启动客户端（需要 ROS2 环境）

#### 不使用 RTC（传统模式，与之前完全一样）

```bash
python3 examples/cr100_open_door/cr100_ros2_client.py \
    --host localhost --port 8000 \
    --frequency 10.0 \
    --prompt "open the door" \
    --record
```

#### 使用 RTC（新增）

```bash
python3 examples/cr100_open_door/cr100_ros2_client.py \
    --host localhost --port 8000 \
    --frequency 10.0 \
    --prompt "open the door" \
    --rtc \
    --execution-horizon 10 \
    --record
```

客户端 RTC 参数说明：

| 参数 | 说明 |
|------|------|
| `--rtc` | 启用 RTC 模式（不加则完全兼容原有行为） |
| `--execution-horizon 10` | 传给服务端的执行窗口大小 |
| `--rtc-blend-steps 0` | 新旧 chunk 合并时的线性混合步数（0=不混合，纯 RTC 引导） |


### 第三步：测试连接（无需 ROS2）

```bash
python3 examples/cr100_open_door/cr100_ros2_client.py \
    --host localhost --port 8000 \
    --test
```

---

## 原理简述

### 传统 action chunking 的问题

```
时间线:  ───────────────────────────────────────→
Chunk A: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
                              ↑ 执行到这里时请求新 chunk
Chunk B:                      [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9]
                              ↑ a4→b0 可能出现跳变！
```

### RTC 的解决方式

```
时间线:  ───────────────────────────────────────→
Chunk A: [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
         ↑ 推理开始       ↑ 推理完成(d=4)
                          ↑ 新chunk可用

Chunk B 生成时受 A 的 guidance 引导:
  [0..3]  冻结区: weight=1.0, 强制 b[0:4] ≈ a[0:4] (已执行，不可改)
  [4..9]  中间区: weight 从 1.0 衰减到 0.0 (新旧过渡)
  [10+]   全新区: weight=0.0 (自由生成)
```

在 flow matching 去噪的每一步中，`compute_guidance` 计算引导速度场：

```
v_guided = v_original - guidance_weight × VJP(err)
```

其中 `err = (old_chunk - predicted_x0) × weights`，通过 VJP 反传修正速度场方向，
使得生成的新 chunk 在冻结区和过渡区与旧 chunk 保持连续。

---

## 数据流总览

```
┌────────────────────────────────────────────────────────────────┐
│                    客户端 (cr100_ros2_client.py)                 │
│                                                                │
│  [后台线程 _rtc_inference_loop]                                  │
│    ① 检查 ActionQueue 剩余量 ≤ num_action_steps                  │
│    ② get_left_over() → prev_chunk_left_over                    │
│    ③ 估计 inference_delay (基于上次真实延迟)                       │
│    ④ send_observation(obs,                                     │
│         prev_chunk_left_over=...,                              │
│         inference_delay=...,                                   │
│         execution_horizon=...)                                 │
│         ↓ 参数打包为 obs["__rtc_kwargs__"]                       │
│         ↓ msgpack 发送                                         │
│    ⑤ 收到 result (actions + actions_original)                   │
│    ⑥ ActionQueue.merge(original, processed, delay)             │
│                                                                │
│  [主线程 control_callback]                                      │
│    ⑦ ActionQueue.get() → 单步 action                            │
│    ⑧ publish_action(action) → ROS2 JointState                 │
└────────────────────────────────────────────────────────────────┘
                              ↕ WebSocket
┌────────────────────────────────────────────────────────────────┐
│                    服务端 (serve_policy.py)                      │
│                                                                │
│  websocket_policy_server._handler:                             │
│    ① obs = unpack(recv())                                      │
│    ② rtc_kwargs = obs.pop("__rtc_kwargs__")                    │
│    ③ policy.infer(obs, **rtc_kwargs)                           │
│         ↓                                                      │
│  policy.py (Policy.infer):                                     │
│    ④ kwargs 中 int/ndarray → jnp.asarray (加 batch 维度)         │
│    ⑤ model.sample_actions(rng, obs, **sample_kwargs)           │
│         ↓                                                      │
│  pi0.py (Pi0.sample_actions):                                  │
│    ⑥ 去噪循环 step() 中:                                        │
│       if rtc_enabled:                                          │
│         v_t = rtc_processor.compute_guidance(                  │
│           x_t, time, prev_chunk_left_over,                     │
│           inference_delay, execution_horizon,                  │
│           model_fn=get_v_t)                                    │
│       else:                                                    │
│         v_t = get_v_t(x_t, time, obs)    ← 原始路径不变          │
│    ⑦ 返回 actions                                              │
│                                                                │
│  policy.py:                                                    │
│    ⑧ 保存 actions_original (output_transform 前)                │
│    ⑨ 返回 {actions, actions_original, policy_timing}            │
└────────────────────────────────────────────────────────────────┘
```

---

## 注意事项

1. **服务端和客户端必须同时启用或同时不启用 RTC**
   - 服务端 `--rtc` 控制模型是否在去噪中使用 `compute_guidance`
   - 客户端 `--rtc` 控制是否发送 `prev_chunk_left_over` 和使用 `ActionQueue`
   - 如果只有一端启用，另一端不启用，功能不会出错但 RTC 不会生效

2. **首次推理会比较慢**（JAX JIT 编译），客户端会自动等待首次推理完成

3. **`execution_horizon` 建议设为与 `action_horizon` 相同**（当前模型 `action_horizon=10`）

4. **不加 `--rtc` 时行为与修改前完全一致**，所有改动都在 `if rtc_enabled` 分支内
