#!/usr/bin/env python3
"""
CR100 ROS2 WebSocket Client Node

This node:
1. Subscribes to ROS2 topics (images, joint states) from remote robot
2. Sends observations to serve_policy.py via WebSocket (msgpack protocol)
3. Receives actions and publishes them as ROS2 messages

Run serve_policy.py first (in uv environment):
    cd /home/x100/wh/openpi && uv run scripts/serve_policy.py --port 8000 \\
        policy:checkpoint \\
        --policy.config pi05_cr100_open_door_lora \\
        --policy.dir /path/to/checkpoint

Then run this node (with system Python 3.10):
    export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages
    python3 examples/cr100_open_door/cr100_ros2_client.py
"""

import argparse
import sys
from pathlib import Path

# Repo layout: openpi/src/openpi/... — allow `import openpi` when using system Python + ROS
# (not only `uv run`, which installs the package into the venv).
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
import logging
import threading
import time
from collections import deque

import numpy as np

from msgpack_utils import CR100WebSocketClient, MSGPACK_AVAILABLE, WEBSOCKET_AVAILABLE

try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("ROS2 not available")

from ros2_bridge import (
    ROS2SensorBridge,
    DEFAULT_IMAGE_TOPICS,
    DEFAULT_JOINT_STATE_TOPICS,
    DEFAULT_ACTION_TOPICS,
)
from recorder import InferenceRecorder


class CR100ROS2ClientNode(Node):
    """ROS2 node that bridges ROS2 topics to WebSocket policy server."""

    def __init__(
        self,
        # --- WebSocket policy server (serve_policy.py) ---
        server_host: str = "localhost",  # Hostname or IP of the machine running the policy server.
        server_port: int = 8000,  # TCP port passed to serve_policy.py --port (default 8000).
        # --- ROS2 I/O (see ros2_bridge.ROS2SensorBridge; None → module defaults) ---
        image_topics: list = None,  # 3 compressed image topic names (high + left_wrist + right_wrist).
        joint_state_topics: list = None,  # 4 topics: left_arm, left_hand, right_arm, right_hand states.
        action_topics: list = None,  # 2 publish topics: left arm cmd + left dexterous hand cmd (CR100 layout).
        # --- Control loop timing ---
        control_frequency: float = 30.0,  # Timer Hz: how often control_callback runs (publish + optional infer).
        # --- Client-side action buffering (independent of model action_horizon in the checkpoint) ---
        action_horizon: int = 50,  # Max steps stored in classic deque; oldest dropped if a chunk is longer.
        num_action_steps: int = 30,  # Replenish threshold: fetch new chunk when queue has < this (classic) or ≤ this (RTC).
        # --- Policy conditioning ---
        prompt: str = "open the door",  # Task string sent in observation; must match training-style language.
        record: bool = False,  # If True, log obs/actions for plots via InferenceRecorder.
        # --- Real-Time Chunking (RTC); server must load an RTC-capable policy if you enable this ---
        rtc_enabled: bool = False,  # Use background infer thread + ActionQueue merge instead of simple deque.
        execution_horizon: int = 10,  # RTC: overlap length (steps) for prev_chunk guidance; passed to server infer kwargs.
        rtc_blend_steps: int = 0,  # RTC: linear blend length at merge between old tail and new chunk head (0 = off).
    ):
        super().__init__('cr100_policy_client')

        self.control_frequency = control_frequency
        self.action_horizon = action_horizon
        self.num_action_steps = num_action_steps
        self.prompt = prompt
        self.record = record
        self.rtc_enabled = rtc_enabled
        self.execution_horizon = execution_horizon

        # ROS2 sensor bridge (subscriptions + publishers)
        self.sensor = ROS2SensorBridge(
            node=self,
            image_topics=image_topics,
            joint_state_topics=joint_state_topics,
            action_topics=action_topics,
        )

        # WebSocket client
        self.ws_client = CR100WebSocketClient(server_host, server_port)

        # Action queue (classic mode)
        self.action_queue = deque(maxlen=action_horizon)

        # RTC state
        if self.rtc_enabled:
            from openpi.rtc.action_queue import ActionQueue as RTCActionQueue
            from openpi.rtc.latency_tracker import LatencyTracker
            self.rtc_action_queue = RTCActionQueue(rtc_enabled=True, blend_steps=rtc_blend_steps)
            self.rtc_latency_tracker = LatencyTracker()
            self.rtc_last_real_delay = None
            self.rtc_first_inference_done = threading.Event()
            self.rtc_stop_event = threading.Event()
            self.rtc_latest_obs = None
            self.rtc_latest_obs_lock = threading.Lock()
            self.rtc_thread = None

        # Recorder
        self.recorder = InferenceRecorder(logger=self.get_logger()) if record else None

        # Statistics
        self.inference_count = 0
        self.last_inference_time = 0.0

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / control_frequency,
            self.control_callback,
        )

        self.get_logger().info("CR100 ROS2 Client Node initialized")
        self.get_logger().info(f"  Recording: {'ENABLED' if record else 'disabled'}")
        self.get_logger().info(f"  RTC: {'ENABLED' if rtc_enabled else 'disabled'}")
        if rtc_enabled:
            self.get_logger().info(f"  RTC execution_horizon: {execution_horizon}")
            self.get_logger().info(f"  RTC blend_steps: {rtc_blend_steps}")
        self.get_logger().info(f"  Server: {server_host}:{server_port}")
        self.get_logger().info(f"  Control frequency: {control_frequency} Hz")
        self.get_logger().info(f"  Prompt: {prompt}")

        self.connect_to_server()

    # ---- connection ----

    def connect_to_server(self):
        max_retries = 10
        retry_delay = 2.0
        for i in range(max_retries):
            self.get_logger().info(f"Connecting to server (attempt {i+1}/{max_retries})...")
            if self.ws_client.connect():
                self.get_logger().info("Connected to policy server!")
                return True
            time.sleep(retry_delay)
        self.get_logger().error("Failed to connect to policy server after max retries")
        return False

    # ---- control loop ----

    def control_callback(self):
        if not self.ws_client.connected:
            if not self.connect_to_server():
                return

        if self.rtc_enabled:
            self._control_callback_rtc()
        else:
            self._control_callback_classic()

    def _control_callback_classic(self):
        if len(self.action_queue) < self.num_action_steps:
            time.sleep(1)
            observation = self.sensor.build_observation(self.prompt)
            if observation is None:
                return

            start_time = time.time()
            result = self.ws_client.send_observation(observation)
            inference_time = time.time() - start_time
            

            if result is not None:
                action_chunk = np.array(result["actions"])
                self.inference_count += 1
                self.last_inference_time = inference_time

                if len(action_chunk.shape) == 1:
                    actions_list = [action_chunk.copy()]
                    self.action_queue.append(action_chunk)
                else:
                    actions_list = [a.copy() for a in action_chunk]
                    for action in action_chunk:
                        self.action_queue.append(action)

                if self.recorder:
                    self.recorder.add_inference(
                        inference_id=self.inference_count,
                        timestamp=time.time(),
                        inference_time=inference_time,
                        obs_state=observation["state"],
                        obs_images=observation["images"],
                        action_chunk=np.array(actions_list),
                    )

                self.get_logger().info(
                    f"Inference #{self.inference_count}: {inference_time:.3f}s, "
                    f"queue size: {len(self.action_queue)}, "
                    f"action_chunk shape: {np.array(actions_list).shape}"
                )

        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            if self.recorder:
                self.recorder.add_publish(action)
            self.sensor.publish_action(action)

    # ---- RTC ----

    def _start_rtc_thread(self):
        if self.rtc_thread is not None:
            return
        self.rtc_thread = threading.Thread(target=self._rtc_inference_loop, daemon=True)
        self.rtc_thread.start()
        self.get_logger().info("RTC background inference thread started")

    def _warmup_jit(self):
        """Pre-compile both JAX JIT paths (non-RTC and RTC) before real control starts.

        JAX traces lazily: the first call with a new kwarg signature triggers a
        full recompilation that takes 5-10 seconds.  During that time the action
        queue drains, real_delay explodes, and RTC merge corrupts everything.

        By sending two dummy inferences here — one without and one with
        prev_chunk_left_over — we force both compilation paths to complete
        *before* any actions are published to the robot.
        """
        self.get_logger().info("JIT warm-up: waiting for first observation...")
        obs = None
        while obs is None:
            with self.rtc_latest_obs_lock:
                obs = self.rtc_latest_obs
            if obs is None:
                time.sleep(0.05)

        self.get_logger().info("JIT warm-up [1/2]: compiling non-RTC path...")
        t0 = time.perf_counter()
        result1 = self.ws_client.send_observation(obs)
        t1 = time.perf_counter()
        self.get_logger().info(f"JIT warm-up [1/2]: done in {(t1-t0)*1000:.0f}ms")

        if result1 is not None:
            actions_orig = result1.get("actions_original")
            if actions_orig is not None:
                dummy_prev = np.array(actions_orig)
            else:
                dummy_prev = np.zeros((self.action_horizon, 32), dtype=np.float32)
        else:
            dummy_prev = np.zeros((self.action_horizon, 32), dtype=np.float32)

        self.get_logger().info("JIT warm-up [2/2]: compiling RTC path...")
        t0 = time.perf_counter()
        self.ws_client.send_observation(
            obs,
            prev_chunk_left_over=dummy_prev,
            inference_delay=4,
            execution_horizon=self.execution_horizon,
        )
        t1 = time.perf_counter()
        self.get_logger().info(f"JIT warm-up [2/2]: done in {(t1-t0)*1000:.0f}ms")
        self.get_logger().info("JIT warm-up complete. Starting real RTC inference loop.")

    def _rtc_inference_loop(self):
        import math
        time_per_step = 1.0 / self.control_frequency
        max_delay_steps = self.action_horizon

        self._warmup_jit()

        while not self.rtc_stop_event.is_set():
            try:
                if self.rtc_action_queue.qsize() <= self.num_action_steps:
                    with self.rtc_latest_obs_lock:
                        obs = self.rtc_latest_obs

                    if obs is None:
                        time.sleep(0.001)
                        continue

                    action_index_before = self.rtc_action_queue.get_action_index()

                    if self.rtc_last_real_delay is not None:
                        estimated_delay = self.rtc_last_real_delay
                    else:
                        latency = self.rtc_latency_tracker.max()
                        if latency and latency > 0:
                            estimated_delay = math.ceil(latency / time_per_step)
                        else:
                            estimated_delay = 4

                    prev_chunk_left_over = self.rtc_action_queue.get_left_over()

                    self.get_logger().info(
                        f"RTC: Starting inference #{self.inference_count + 1}. "
                        f"Queue={self.rtc_action_queue.qsize()}, "
                        f"est_delay={estimated_delay}, "
                        f"leftover={'None' if prev_chunk_left_over is None else prev_chunk_left_over.shape}"
                    )

                    current_time = time.perf_counter()
                    if prev_chunk_left_over is not None:
                        result = self.ws_client.send_observation(
                            obs,
                            prev_chunk_left_over=prev_chunk_left_over,
                            inference_delay=estimated_delay,
                            execution_horizon=self.execution_horizon,
                        )
                    else:
                        result = self.ws_client.send_observation(obs)
                    latency = time.perf_counter() - current_time
                    self.rtc_latency_tracker.add(latency)
                    inference_delay_steps = math.ceil(latency / time_per_step)

                    if result is None:
                        self.get_logger().error("RTC: Inference returned None")
                        time.sleep(0.1)
                        continue

                    processed_actions = result.get("actions")
                    original_actions = result.get("actions_original")

                    if processed_actions is None:
                        self.get_logger().error("RTC: No 'actions' in result")
                        continue

                    processed_actions = np.array(processed_actions)
                    if original_actions is None:
                        original_actions = processed_actions.copy()
                    else:
                        original_actions = np.array(original_actions)

                    self.inference_count += 1
                    self.last_inference_time = latency

                    if self.recorder:
                        obs_state = obs.get("state", np.zeros(26))
                        if not isinstance(obs_state, np.ndarray):
                            obs_state = np.zeros(26)
                        self.recorder.add_inference(
                            inference_id=self.inference_count,
                            timestamp=time.time(),
                            inference_time=latency,
                            obs_state=obs_state,
                            obs_images=obs.get("images", {}),
                            action_chunk=processed_actions,
                        )

                    real_delay_before = self.rtc_action_queue.get_action_index() - action_index_before
                    clamped_delay = min(real_delay_before, max_delay_steps)

                    self.rtc_action_queue.merge(
                        new_original_actions=original_actions,
                        new_processed_actions=processed_actions,
                        estimated_delay=estimated_delay,
                        real_delay=clamped_delay,
                    )

                    if real_delay_before > 0:
                        self.rtc_last_real_delay = min(real_delay_before, max_delay_steps)
                    elif inference_delay_steps <= 10:
                        self.rtc_last_real_delay = inference_delay_steps
                    else:
                        self.rtc_last_real_delay = 4

                    if not self.rtc_first_inference_done.is_set():
                        self.get_logger().info("RTC: First inference done, queue ready")
                        self.rtc_first_inference_done.set()

                    self.get_logger().info(
                        f"RTC Inference #{self.inference_count}: {latency*1000:.1f}ms, "
                        f"queue={self.rtc_action_queue.qsize()}, "
                        f"delay_steps={inference_delay_steps}, "
                        f"clamped_delay={clamped_delay}"
                    )
                else:
                    time.sleep(0.001)

            except Exception as e:
                self.get_logger().error(f"RTC thread error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _control_callback_rtc(self):
        self._start_rtc_thread()

        observation = self.sensor.build_observation(self.prompt)
        if observation is not None:
            with self.rtc_latest_obs_lock:
                self.rtc_latest_obs = observation

        if not self.rtc_first_inference_done.is_set():
            return

        action = self.rtc_action_queue.get()
        if action is None:
            return

        if self.recorder:
            self.recorder.add_publish(action)
        self.sensor.publish_action(action)

    # ---- lifecycle ----

    def destroy_node(self):
        self.get_logger().info(
            f"Shutting down. Total inferences: {self.inference_count}, "
            f"total published actions: {self.recorder and len(self.recorder.publish_log) or 0}"
        )
        if self.recorder:
            self.recorder.save()
        self.ws_client.disconnect()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Standalone test (no ROS2 required)
# ---------------------------------------------------------------------------

def test_connection(host: str, port: int):
    """Test WebSocket connection without ROS2."""
    print(f"Testing connection to ws://{host}:{port}...")

    client = CR100WebSocketClient(host, port)
    if not client.connect():
        print("Failed to connect!")
        return False

    print(f"Connected! Metadata: {client.metadata}")

    dummy_image = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    dummy_state = np.zeros(26, dtype=np.float32)

    observation = {
        "state": dummy_state,
        "images": {
            "cam_high": dummy_image,
            "cam_left_wrist": dummy_image,
            "cam_right_wrist": dummy_image,
        },
        "prompt": "cr100 open door",
    }

    print("Sending dummy observation (3 cameras CHW, 26-dim state)...")
    start = time.time()
    result = client.send_observation(observation)
    elapsed = time.time() - start

    if result is not None:
        action = np.array(result["actions"])
        print(f"Received action! Shape: {action.shape}, Time: {elapsed:.3f}s")
        print(f"Action sample: {action.flatten()[:26]}")
        client.disconnect()
        return True
    else:
        print("Failed to receive action!")
        client.disconnect()
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CR100 ROS2 WebSocket Client')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Policy server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Policy server port (default: 8000)')
    parser.add_argument('--image-topics', type=str, nargs='+',
                        default=DEFAULT_IMAGE_TOPICS,
                        help='Image topics to subscribe (3 compressed topics: left, right, high)')
    parser.add_argument('--joint-state-topics', type=str, nargs='+',
                        default=DEFAULT_JOINT_STATE_TOPICS,
                        help='Joint state topics (4 topics: left_arm, left_hand, right_arm, right_hand)')
    parser.add_argument('--action-topics', type=str, nargs='+',
                        default=['/cr100/left_arm/online_joint_command',
                                 '/cr100/left_dexterous_hand_command'],
                        help='Action topics to publish')
    parser.add_argument('--frequency', type=float, default=30.0,
                        help='Control frequency in Hz')
    parser.add_argument('--prompt', type=str, default='open the door',
                        help='Task prompt for the policy')
    parser.add_argument('--record', action='store_true',
                        help='Record observations and actions, save plots on exit')
    parser.add_argument('--rtc', action='store_true',
                        help='Enable Real-Time Chunking (RTC) for smooth action transitions')
    parser.add_argument('--execution-horizon', type=int, default=10,
                        help='RTC execution horizon (default: 10)')
    parser.add_argument('--rtc-blend-steps', type=int, default=3,
                        help='RTC blend steps for linear interpolation at merge (default: 0)')
    parser.add_argument('--test', action='store_true',
                        help='Test connection without ROS2')
    args = parser.parse_args()

    if len(args.image_topics) != 3:
        print(f"ERROR: Expected 3 image topics, got {len(args.image_topics)}")
        return 1
    if len(args.joint_state_topics) != 4:
        print(f"ERROR: Expected 4 joint state topics, got {len(args.joint_state_topics)}")
        return 1
    if len(args.action_topics) != 2:
        print(f"ERROR: Expected 2 action topics, got {len(args.action_topics)}")
        return 1

    if not WEBSOCKET_AVAILABLE:
        print("ERROR: websocket-client not installed. Run: pip3 install websocket-client --user")
        return 1
    if not MSGPACK_AVAILABLE:
        print("ERROR: msgpack not installed. Run: pip3 install msgpack --user")
        return 1

    if args.test:
        success = test_connection(args.host, args.port)
        return 0 if success else 1

    if not ROS2_AVAILABLE:
        print("ERROR: ROS2 not available. Set PYTHONPATH to include ROS2 packages.")
        print("  export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:"
              "/opt/ros/humble/lib/python3.10/site-packages")
        return 1

    rclpy.init()

    node = CR100ROS2ClientNode(
        server_host=args.host,
        server_port=args.port,
        image_topics=args.image_topics,
        joint_state_topics=args.joint_state_topics,
        action_topics=args.action_topics,
        control_frequency=args.frequency,
        prompt=args.prompt,
        record=args.record,
        rtc_enabled=args.rtc,
        execution_horizon=args.execution_horizon,
        rtc_blend_steps=args.rtc_blend_steps,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'rtc_stop_event'):
            node.rtc_stop_event.set()
            if node.rtc_thread is not None and node.rtc_thread.is_alive():
                node.rtc_thread.join(timeout=2.0)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    exit(main())
