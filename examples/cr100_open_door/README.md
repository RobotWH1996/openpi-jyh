# CR100 Open Door - ROS2 Policy Inference

**WebSocket 客户端模式：ROS2 节点 + 策略服务器分离部署，支持 RTC (Real-Time Chunking) 平滑控制**

## 文件结构

```
cr100_open_door/
├── cr100_ros2_client.py     # 主控逻辑：经典/RTC 模式调度、CLI 入口
├── msgpack_utils.py         # msgpack numpy 序列化 + CR100WebSocketClient
├── ros2_bridge.py           # ROS2 订阅/发布/observation 构建 (ROS2SensorBridge)
├── recorder.py              # 数据记录 + .npz 保存 + matplotlib 可视化 (InferenceRecorder)
├── contro.cpp               # C++ TensorRT 本地推理节点（独立方案，不走 WebSocket）
└── README.md
```

## 架构说明

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          本机 (GPU Server)                                │
│                                                                          │
│  ┌─────────────────────────┐       ┌──────────────────────────────────┐  │
│  │  serve_policy.py        │       │  cr100_ros2_client.py            │  │
│  │  (uv run, Python 3.11)  │◄─────►│  (system Python 3.10)            │  │
│  │                         │  WS   │                                  │  │
│  │  - 模型推理 (GPU)       │       │  msgpack_utils.py  WebSocket通信  │  │
│  │  - RTC guided denoise   │       │  ros2_bridge.py    ROS2收发      │  │
│  │  - 策略服务             │       │  recorder.py       数据记录+画图  │  │
│  └─────────────────────────┘       └──────────────┬───────────────────┘  │
│                                                    │ ROS2                 │
└────────────────────────────────────────────────────┼─────────────────────┘
                                                     │
                                                     ▼
                                    ┌──────────────────────────────────┐
                                    │  远端机器人 (Robot PC)            │
                                    │                                  │
                                    │  - 相机图像发布                   │
                                    │  - 关节状态发布                   │
                                    │  - 接收关节命令                   │
                                    └──────────────────────────────────┘
```

## 快速开始

### 步骤 1: 启动策略服务器 (终端 1)

```bash
cd /home/x100/wh/openpi

# 普通模式
uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_cr100_open_door_lora \
    --policy.dir ./checkpoints/pi05_cr100_open_door_lora/your_exp/10000 \
    --port 8000

# 启用 RTC（服务端）
uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config pi05_cr100_open_door_lora \
    --policy.dir ./checkpoints/... \
    --port 8000 \
    --rtc --rtc-execution-horizon 10
```

### 步骤 2: 测试连接 (可选)

```bash
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages

/usr/bin/python3 examples/cr100_open_door/cr100_ros2_client.py \
    --test --host localhost --port 8000
```

### 步骤 3: 启动 ROS2 客户端节点 (终端 2)

```bash
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages

# 经典模式
/usr/bin/python3 examples/cr100_open_door/cr100_ros2_client.py \
    --host localhost --port 8000 \
    --prompt "open the door"

# 经典模式 + 录制
/usr/bin/python3 examples/cr100_open_door/cr100_ros2_client.py \
    --host localhost --port 8000 \
    --prompt "open the door" --record

# RTC 模式 + 录制
/usr/bin/python3 examples/cr100_open_door/cr100_ros2_client.py \
    --host localhost --port 8000 \
    --prompt "open the door" --rtc --execution-horizon 10 --record
```

---

## 命令行参数

### serve_policy.py（服务端 RTC 相关）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8000` | WebSocket 服务端口 |
| `--rtc` | `False` | 启用 RTC guided denoising |
| `--rtc-execution-horizon` | `10` | RTC 执行步长 |
| `--rtc-max-guidance-weight` | `10.0` | RTC 最大引导权重 |
| `--rtc-schedule` | `LINEAR` | 注意力衰减方式 (ZEROS/ONES/LINEAR/EXP) |

### cr100_ros2_client.py（客户端）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `localhost` | 策略服务器地址 |
| `--port` | `8000` | 策略服务器端口 |
| `--image-topics` | 3 个压缩话题 | 图像话题（left, right, high） |
| `--joint-state-topics` | 4 个关节话题 | 关节状态话题 |
| `--action-topics` | 2 个命令话题 | 动作发布话题 |
| `--frequency` | `10.0` | 控制频率 (Hz) |
| `--prompt` | `open the door` | 任务提示语 |
| `--record` | `False` | 录制数据并在退出时保存图表 |
| `--test` | `False` | 测试 WebSocket 连接（无需 ROS2） |
| `--rtc` | `False` | 启用 RTC 异步推理模式 |
| `--execution-horizon` | `10` | RTC 执行步长 |
| `--rtc-blend-steps` | `0` | RTC merge 时线性插值步数 |

---

## 两种运行模式

### 经典模式（默认）

同步推理：当 action 队列不足时触发推理，获取新 chunk 后按顺序执行。

```
控制回调 → 队列不足? → 发送 obs → 等待推理 → 入队 → 逐步发布
```

### RTC 模式（`--rtc`）

异步推理：后台线程持续推理新 chunk，主线程按固定频率从队列取动作。
新旧 chunk 通过 soft masking 平滑融合，消除 chunk 边界的动作跳变。

```
后台线程：持续推理 → merge 新 chunk（带 guidance 引导）→ 入队
控制回调：从队列取一个 action → 发布到 ROS2
```

服务端和客户端都需要加 `--rtc` 参数。

---

## ROS2 话题

### 订阅话题

| 话题 | 类型 | 说明 | 映射 |
|------|------|------|------|
| `/camera/left/image_raw/compressed` | `CompressedImage` | 左腕相机 | `cam_left_wrist` |
| `/camera/right/image_raw/compressed` | `CompressedImage` | 右腕相机 | `cam_right_wrist` |
| `/camera_high/color/image_raw/compressed` | `CompressedImage` | 顶部相机 | `cam_high` |
| `/cr100/left_arm_state` | `JointState` | 左臂 (7维) | state[0:7] |
| `/cr100/left_hand_state` | `JointState` | 左手 (6维) | state[7:13] |
| `/cr100/right_arm_state` | `JointState` | 右臂 (7维) | state[13:20] |
| `/cr100/right_hand_state` | `JointState` | 右手 (6维) | state[20:26] |

### 发布话题

| 话题 | 类型 | 维度 |
|------|------|------|
| `/cr100/left_arm/online_joint_command` | `JointState` | 7维 (action[0:7]) |
| `/cr100/left_dexterous_hand_command` | `JointState` | 6维 (action[7:13]) |
| `/cr100/right_arm/online_joint_command` | `JointState` | 7维 (action[13:20]) |
| `/cr100/right_dexterous_hand_command` | `JointState` | 6维 (action[20:26]) |

**State/Action 维度：** `left_arm(7) + left_hand(6) + right_arm(7) + right_hand(6) = 26维`

---

## 录制功能（`--record`）

加 `--record` 后，退出时（Ctrl+C）自动保存到 `logs/<timestamp>/`：

```
logs/20260304_153000/
├── data.npz                        # obs_states, published_actions, inference_times
├── images/                         # 每次推理时的相机截图
│   ├── inf0001_cam_high.jpg
│   ├── inf0001_cam_left_wrist.jpg
│   └── ...
├── obs_vs_action_per_inference.png  # 观测 vs 首步动作对比图
├── action_trajectory.png           # 连续发布动作轨迹
└── observation_trajectory.png      # 观测状态轨迹 + 推理耗时
```

---

## 依赖安装

### 策略服务器 (uv 环境)

```bash
cd /home/x100/wh/openpi
uv sync
```

### ROS2 客户端 (系统 Python)

```bash
pip3 install websocket-client msgpack --user
pip3 install 'numpy<2' 'opencv-python<4.10' --user  # 兼容 cv_bridge
```

---

## 远程部署

```bash
# GPU 服务器端
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_cr100_open_door_lora \
    --policy.dir ... --port 8000

# 机器人端
/usr/bin/python3 examples/cr100_open_door/cr100_ros2_client.py \
    --host 192.168.1.100 --port 8000
```

---

## 故障排除

### WebSocket 连接失败

```bash
curl -I http://localhost:8000
sudo ufw allow 8000
```

### ROS2 不可用

```bash
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages
/usr/bin/python3 -c "import rclpy; print('ROS2 OK')"
```

### NumPy 版本冲突

```bash
pip3 install 'numpy<2' 'opencv-python<4.10' --user
```
