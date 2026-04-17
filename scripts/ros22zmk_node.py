#!/usr/bin/env python3
"""
ROS2 Humble -> LeRobot ZMQ camera bridge with NV12 support.

Publishes JSON strings over a ZMQ PUB socket in the format expected by LeRobot's ZMQCamera: [1](https://github.com/huggingface/lerobot/blob/main/docs/source/cameras.mdx)
{
  "timestamps": {"camera_name": float},
  "images": {"camera_name": "<base64-jpeg>"}
}

Design goals:
- Do almost nothing in ROS callback (store latest msg only).
- Convert/encode/send on a timer at max_fps to avoid backpressure.
- BEST_EFFORT QoS for sensor topics to avoid blocking publishers.
- Non-blocking ZMQ send and drop frames if busy. [2](https://discourse.openrobotics.org/t/lerobot-ros-a-lightweight-interface-for-controlling-ros-based-robotic-arms-using-lerobot/49420)
"""

import argparse
import base64
import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import numpy as np
import cv2
import zmq

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


def ros_stamp_to_float_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def encode_jpeg_base64(rgb: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return base64.b64encode(buf).decode("utf-8")


def nv12_to_rgb(msg: Image) -> np.ndarray:
    """
    Convert NV12 in sensor_msgs/Image to RGB uint8 HxWx3.

    NV12 layout: Y plane (H x stride), then interleaved UV plane (H/2 x stride).
    OpenCV expects a single-channel image shaped ((H*3)/2, W) with contiguous stride.
    We handle possible msg.step (stride) != width by copying rows.
    """
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid dimensions: {w}x{h}")
    if step < w:
        raise ValueError(f"Invalid step {step} < width {w}")

    data = np.frombuffer(msg.data, dtype=np.uint8)

    # Expected total bytes: step*h + step*(h//2) = step*h*3/2
    needed = step * h + step * (h // 2)
    if data.size < needed:
        raise ValueError(f"Buffer too small: {data.size} < {needed}")

    # Build contiguous NV12 buffer with width 'w' (crop padding if step>w)
    yuv = np.empty((h + h // 2, w), dtype=np.uint8)

    # Copy Y plane
    for row in range(h):
        start = row * step
        yuv[row, :] = data[start : start + w]

    # Copy UV plane
    uv_base = step * h
    for row in range(h // 2):
        start = uv_base + row * step
        yuv[h + row, :] = data[start : start + w]

    # Convert NV12 -> RGB
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
    return rgb


def yuv420_to_rgb_i420(msg: Image) -> np.ndarray:
    """
    Optional: handle yuv420p / i420 style if your encoding differs.
    """
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    # For I420, planes are Y (H*W), U (H/2*W/2), V (H/2*W/2), typically tightly packed.
    # Many ROS publishers set step=w for these.
    if step != w:
        # could still work but is ambiguous; prefer converting upstream or using cv_bridge
        raise ValueError(f"I420 handler expects step==width, got step={step}, width={w}")

    needed = (h * w * 3) // 2
    if data.size < needed:
        raise ValueError(f"Buffer too small: {data.size} < {needed}")

    yuv = data[:needed].reshape((h + h // 2, w))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    return rgb


@dataclass
class BridgeStats:
    last_send_t: float = 0.0
    sent_frames: int = 0
    dropped_zmq: int = 0
    dropped_throttle: int = 0


class Ros2ZmqNv12Bridge(Node):
    def __init__(
        self,
        *,
        topic: str,
        camera_name: str,
        bind: str,
        port: int,
        max_fps: float,
        jpeg_quality: int,
        resize_width: int,
        snd_hwm: int,
        drop_if_busy: bool,
    ):
        super().__init__("ros2_zmq_nv12_bridge")

        self.topic = topic
        self.camera_name = camera_name
        self.max_fps = float(max_fps)
        self.jpeg_quality = int(jpeg_quality)
        self.resize_width = int(resize_width)
        self.drop_if_busy = bool(drop_if_busy)

        # BEST_EFFORT, KEEP_LAST(1) is the safest for high-rate sensor topics
        # and avoids blocking publishers. [4](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/ros2_tutorials/tutorial_ros2_camera.html)[5](https://github.com/huggingface/lerobot/issues/1195)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._lock = Lock()
        self._latest_msg: Optional[Image] = None
        self._latest_stamp: float = 0.0

        self._sub = self.create_subscription(Image, topic, self._on_image, qos)

        # ZMQ PUB socket similar to LeRobot's image_server. [2](https://discourse.openrobotics.org/t/lerobot-ros-a-lightweight-interface-for-controlling-ros-based-robotic-arms-using-lerobot/49420)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, int(snd_hwm))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{bind}:{int(port)}")

        self.get_logger().info(
            f"Bridge started. topic='{topic}' camera_name='{camera_name}' "
            f"ZMQ=tcp://{bind}:{port} max_fps={self.max_fps} quality={self.jpeg_quality} "
            f"resize_width={self.resize_width if self.resize_width>0 else 'no'}"
        )

        self._stats = BridgeStats(last_send_t=time.time())

        # Timer controls processing rate (throttle)
        period = 1.0 / self.max_fps if self.max_fps > 0 else 0.1
        self._timer = self.create_timer(period, self._process_and_send)

    def _on_image(self, msg: Image):
        # Keep callback super fast: just store latest message
        stamp = ros_stamp_to_float_seconds(msg.header.stamp) if msg.header else time.time()
        with self._lock:
            self._latest_msg = msg
            self._latest_stamp = stamp

    def _process_and_send(self):
        with self._lock:
            msg = self._latest_msg
            stamp = self._latest_stamp
            self._latest_msg = None  # consume latest (drop intermediate frames by design)

        if msg is None:
            return

        try:
            enc = (msg.encoding or "").lower()

            if enc in ("nv12", "nv12_8", "yuv420sp", "yuv420sp_nv12"):
                rgb = nv12_to_rgb(msg)
            elif enc in ("yuv420", "i420", "yuv420p"):
                rgb = yuv420_to_rgb_i420(msg)
            elif enc in ("rgb8",):
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            elif enc in ("bgr8",):
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(
                    f"Unsupported encoding '{msg.encoding}'. "
                    "Expected nv12 (NV12/YUV420SP) or rgb8/bgr8."
                )

            # Optional downscale to reduce CPU (very helpful!)
            if self.resize_width and self.resize_width > 0 and rgb.shape[1] > self.resize_width:
                new_w = self.resize_width
                new_h = int(round(rgb.shape[0] * (new_w / rgb.shape[1])))
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            b64 = encode_jpeg_base64(rgb, quality=self.jpeg_quality)
            payload = json.dumps(
                {
                    "timestamps": {self.camera_name: float(stamp)},
                    "images": {self.camera_name: b64},
                }
            )

            if self.drop_if_busy:
                try:
                    self._sock.send_string(payload, flags=zmq.NOBLOCK)
                except zmq.Again:
                    self._stats.dropped_zmq += 1
                    return
            else:
                self._sock.send_string(payload)

            self._stats.sent_frames += 1
            now = time.time()
            if now - self._stats.last_send_t >= 2.0:
                fps = self._stats.sent_frames / (now - self._stats.last_send_t)
                self.get_logger().info(
                    f"ZMQ streaming ~{fps:.1f} FPS "
                    f"(dropped_zmq={self._stats.dropped_zmq})"
                )
                self._stats.sent_frames = 0
                self._stats.dropped_zmq = 0
                self._stats.last_send_t = now

        except Exception as e:
            self.get_logger().warn(f"process/send failed: {e}")

    def destroy_node(self):
        try:
            self._sock.close()
            self._ctx.term()
        except Exception:
            pass
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="sensor_msgs/Image topic (NV12).")
    ap.add_argument("--camera-name", default="front", help="Key name used in ZMQ JSON images/timestamps.")
    ap.add_argument("--bind", default="*", help="ZMQ bind address, default '*'.")
    ap.add_argument("--port", type=int, default=5555, help="ZMQ port.")
    ap.add_argument("--max-fps", type=float, default=10.0, help="Max FPS to encode/send (throttle).")
    ap.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality (lower reduces CPU).")
    ap.add_argument("--resize-width", type=int, default=640, help="Downscale width (0 disables).")
    ap.add_argument("--snd-hwm", type=int, default=5, help="ZMQ SNDHWM (lower drops sooner).")
    ap.add_argument("--drop-if-busy", action="store_true", help="Use non-blocking ZMQ send and drop if busy.")
    args = ap.parse_args()

    rclpy.init()
    node = Ros2ZmqNv12Bridge(
        topic=args.topic,
        camera_name=args.camera_name,
        bind=args.bind,
        port=args.port,
        max_fps=args.max_fps,
        jpeg_quality=args.jpeg_quality,
        resize_width=args.resize_width,
        snd_hwm=args.snd_hwm,
        drop_if_busy=args.drop_if_busy,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
