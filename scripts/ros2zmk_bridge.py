#!/usr/bin/env python3
"""
ROS2 Humble -> LeRobot ZMQ bridge (low-latency RGB).

- Subscribes to ONE sensor_msgs/Image topic (RGB: bgr8/rgb8/nv12).
- Encodes + publishes directly in the subscriber callback (no timer delay).
- Rate-limits via min_interval to avoid flooding.
- Vectorized NV12 decode (no Python row-loops).
- TCP_NODELAY for minimal wire latency.

Output JSON (compatible with LeRobot ZMQCamera):
{
  "timestamps": {"<name>": float},
  "images":     {"<name>": "<base64-jpeg>"}
}
"""

import argparse
import base64
import json
import time

import numpy as np
import cv2
import zmq
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


def _stamp_to_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _nv12_to_bgr(msg: Image) -> np.ndarray:
    """Vectorized NV12 -> BGR conversion (handles stride != width)."""
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if step == w:
        # No padding — direct reshape (fastest path)
        yuv = data[: h * w * 3 // 2].reshape((h * 3 // 2, w))
    else:
        # Stride differs from width — vectorized slice (no Python loop)
        total_rows = h + h // 2
        strided = data[: total_rows * step].reshape((total_rows, step))
        yuv = strided[:, :w].copy()

    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


class RgbToZmq(Node):
    def __init__(
        self,
        topic: str,
        bind: str,
        port: int,
        camera_name: str,
        max_fps: float,
        jpeg_quality: int,
        resize_width: int,
        snd_hwm: int,
        drop_if_busy: bool,
    ):
        super().__init__("rgb_to_zmq")
        self.camera_name = camera_name
        self.jpeg_quality = int(jpeg_quality)
        self.resize_width = int(resize_width)
        self.drop_if_busy = bool(drop_if_busy)

        # Rate limiting: min interval between published frames
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_pub_time: float = 0.0

        # Pre-allocate JPEG encode params (avoid list creation per frame)
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        # BEST_EFFORT + KEEP_LAST(1) — lightweight subscriber
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Optional cv_bridge
        self._cv_bridge = None
        try:
            from cv_bridge import CvBridge  # type: ignore
            self._cv_bridge = CvBridge()
        except Exception:
            self.get_logger().warn("cv_bridge not found; supporting nv12/bgr8/rgb8 manually.")

        self._seen_enc = False

        self.create_subscription(Image, topic, self._on_image, qos)

        # ZMQ PUB socket (libzmq sets TCP_NODELAY internally by default)
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, int(snd_hwm))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.IMMEDIATE, 1)
        self._sock.bind(f"tcp://{bind}:{int(port)}")
        self.get_logger().info(f"ZMQ PUB tcp://{bind}:{port} name='{camera_name}' topic='{topic}'")

        self._last_log = time.time()
        self._sent = 0
        self._dropped = 0

    def _on_image(self, msg: Image):
        """Decode, encode, and publish directly — no timer indirection."""
        now = time.monotonic()
        if (now - self._last_pub_time) < self._min_interval:
            return  # rate-limit

        if not self._seen_enc:
            self.get_logger().info(f"Encoding: {msg.encoding}, size=({msg.height},{msg.width})")
            self._seen_enc = True

        try:
            bgr = self._decode(msg)
            bgr = self._resize(bgr)

            # Encode JPEG directly in BGR (imencode expects BGR, receiver decodes fine)
            ok, buf = cv2.imencode(".jpg", bgr, self._jpeg_params)
            if not ok:
                return
            b64 = base64.b64encode(buf).decode("ascii")

            ts = _stamp_to_s(msg.header.stamp)
            payload = json.dumps({
                "timestamps": {self.camera_name: ts},
                "images": {self.camera_name: b64},
            })

            if self.drop_if_busy:
                try:
                    self._sock.send_string(payload, flags=zmq.NOBLOCK)
                except zmq.Again:
                    self._dropped += 1
                    return
            else:
                self._sock.send_string(payload)

            self._last_pub_time = now
            self._sent += 1
            wall = time.time()
            if wall - self._last_log > 2.0:
                fps = self._sent / (wall - self._last_log)
                self.get_logger().info(f"~{fps:.1f} FPS (dropped={self._dropped})")
                self._sent = 0
                self._dropped = 0
                self._last_log = wall

        except Exception as e:
            self.get_logger().warn(f"frame failed: {e}")

    def _decode(self, msg: Image) -> np.ndarray:
        enc = (msg.encoding or "").lower()
        h, w = int(msg.height), int(msg.width)

        if enc in ("nv12", "yuv420sp", "yuv420sp_nv12", "nv12_8"):
            return _nv12_to_bgr(msg)

        if self._cv_bridge is not None:
            return self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if enc == "bgr8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
        if enc == "rgb8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raise ValueError(f"Unsupported encoding '{msg.encoding}' (install cv_bridge for more)")

    def _resize(self, bgr: np.ndarray) -> np.ndarray:
        if self.resize_width > 0 and bgr.shape[1] > self.resize_width:
            new_w = self.resize_width
            new_h = int(round(bgr.shape[0] * (new_w / bgr.shape[1])))
            return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return bgr

    def destroy_node(self):
        try:
            self._sock.close()
            self._ctx.term()
        except Exception:
            pass
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/StereoNetNode/rectified_image")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--camera-name", default="front")
    ap.add_argument("--max-fps", type=float, default=30.0)
    ap.add_argument("--jpeg-quality", type=int, default=60)
    ap.add_argument("--resize-width", type=int, default=640, help="0 disables resizing")
    ap.add_argument("--snd-hwm", type=int, default=3)
    ap.add_argument("--drop-if-busy", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = RgbToZmq(
        topic=args.topic,
        bind=args.bind,
        port=args.port,
        camera_name=args.camera_name,
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
