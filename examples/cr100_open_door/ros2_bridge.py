"""
ROS2 subscription / publication / observation building for CR100 robot.

Handles:
  - Subscribing to compressed image topics and joint state topics
  - Publishing actions as JointState messages to control topics
  - Building observation dicts (images + state + prompt) for the policy server
"""

import threading

import cv2
import numpy as np

try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CompressedImage, JointState
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# Default topics
DEFAULT_IMAGE_TOPICS = [
    '/camera/left/image_raw/compressed',
    '/camera/right/image_raw/compressed',
    '/camera_high/color/image_raw/compressed',
]

DEFAULT_JOINT_STATE_TOPICS = [
    '/cr100/left_arm_state',
    '/cr100/left_hand_state',
    '/cr100/right_arm_state',
    '/cr100/right_hand_state',
]

DEFAULT_ACTION_TOPICS = [
    '/cr100/left_arm/online_joint_command',
    '/cr100/left_dexterous_hand_command',
    '/cr100/right_arm/online_joint_command',
    '/cr100/right_dexterous_hand_command',
]

# Camera name mapping (index in image_topics -> observation key)
CAMERA_NAMES = ['cam_left_wrist', 'cam_right_wrist', 'cam_high']

# Joint group name mapping (index in joint_state_topics -> key)
JOINT_NAMES = ['left_arm', 'left_hand', 'right_arm', 'right_hand']

# Action dimension layout: left_arm(7) + left_hand(6) + right_arm(7) + right_hand(6) = 26
ACTION_SLICES = [slice(0, 7), slice(7, 13), slice(13, 20), slice(20, 26)]
ACTION_DIMS = [7, 6, 7, 6]
REQUIRED_CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
REQUIRED_JOINTS = ['left_arm', 'left_hand', 'right_arm', 'right_hand']


class ROS2SensorBridge:
    """Manages ROS2 subscriptions for images and joint states, and action publishers."""

    def __init__(self, node, image_topics=None, joint_state_topics=None, action_topics=None):
        """
        Args:
            node: rclpy.node.Node instance (the parent node that owns subscriptions/publishers)
            image_topics: list of 3 compressed image topic names
            joint_state_topics: list of 4 joint state topic names
            action_topics: list of action topic names
        """
        self.node = node
        self.bridge = CvBridge()

        self.image_topics = image_topics or DEFAULT_IMAGE_TOPICS
        self.joint_state_topics = joint_state_topics or DEFAULT_JOINT_STATE_TOPICS
        self.action_topics = action_topics or DEFAULT_ACTION_TOPICS

        self.image_topic_map = {
            self.image_topics[i]: CAMERA_NAMES[i] for i in range(len(self.image_topics))
        }
        self.joint_state_topic_map = {
            self.joint_state_topics[i]: JOINT_NAMES[i] for i in range(len(self.joint_state_topics))
        }

        self.latest_images: dict[str, np.ndarray] = {}
        self.latest_joint_states: dict[str, list] = {}
        self.data_lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for topic in self.image_topics:
            cb = self._make_image_callback(topic)
            node.create_subscription(CompressedImage, topic, cb, sensor_qos)

        for topic in self.joint_state_topics:
            cb = self._make_joint_callback(topic)
            node.create_subscription(JointState, topic, cb, sensor_qos)

        self.action_pubs: dict[str, object] = {}
        for i, topic in enumerate(self.action_topics):
            self.action_pubs[topic] = node.create_publisher(JointState, topic, 10)
            node.get_logger().info(f"  Action publisher {i+1}: {topic} ({ACTION_DIMS[i]} dims)")

    # ---- callbacks ----

    def _make_image_callback(self, topic):
        def callback(msg):
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
                camera_name = self.image_topic_map.get(topic)
                if camera_name:
                    with self.data_lock:
                        self.latest_images[camera_name] = cv_image
            except Exception as e:
                self.node.get_logger().error(f"Image conversion error for {topic}: {e}")
        return callback

    def _make_joint_callback(self, topic):
        def callback(msg):
            joint_name = self.joint_state_topic_map.get(topic)
            if joint_name:
                with self.data_lock:
                    self.latest_joint_states[joint_name] = list(msg.position)
        return callback

    # ---- observation building ----

    def build_observation(self, prompt: str) -> dict | None:
        """Build observation dict for the policy server. Returns None if data is incomplete."""
        with self.data_lock:
            missing_cameras = [c for c in REQUIRED_CAMERAS if c not in self.latest_images]
            if missing_cameras:
                self.node.get_logger().warn(f"Missing image data for cameras: {missing_cameras}")
                return None

            missing_joints = [j for j in REQUIRED_JOINTS if j not in self.latest_joint_states]
            if missing_joints:
                self.node.get_logger().warn(f"Missing joint state data for: {missing_joints}")
                return None

            images = {k: v.copy() for k, v in self.latest_images.items()}
            joint_states = {k: v.copy() for k, v in self.latest_joint_states.items()}

        processed_images = {}
        for camera_name, image in images.items():
            image_resized = cv2.resize(image, (224, 224))
            image_chw = np.transpose(image_resized, (2, 0, 1)).astype(np.uint8)
            processed_images[camera_name] = image_chw

        state_parts = [
            joint_states['left_arm'],
            joint_states['left_hand'],
            joint_states['right_arm'],
            joint_states['right_hand'],
        ]
        state = np.concatenate(state_parts, axis=0).astype(np.float32)

        if len(state) != 26:
            self.node.get_logger().error(
                f"State dimension mismatch: expected 26, got {len(state)}. "
                f"Parts: left_arm={len(joint_states['left_arm'])}, "
                f"left_hand={len(joint_states['left_hand'])}, "
                f"right_arm={len(joint_states['right_arm'])}, "
                f"right_hand={len(joint_states['right_hand'])}"
            )
            return None

        return {
            "state": state,
            "images": processed_images,
            "prompt": prompt,
        }

    # ---- action publishing ----

    def publish_action(self, action: np.ndarray):
        """Publish 26-dim action to 4 ROS2 JointState topics."""
        if len(action) != 26:
            self.node.get_logger().error(f"Action dimension mismatch: expected 26, got {len(action)}")
            return

        now = self.node.get_clock().now().to_msg()
        for i, topic in enumerate(self.action_topics):
            part = action[ACTION_SLICES[i]].tolist()
            msg = JointState()
            msg.header.stamp = now
            msg.name = []
            msg.position = part
            self.action_pubs[topic].publish(msg)
