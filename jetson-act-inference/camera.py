"""Camera capture: Intel RealSense (wrist) and MIPI CSI / USB (front)."""

import atexit
import time
import cv2
import numpy as np


def _usb_reset(device: int) -> None:
    """Reset the USB device backing /dev/videoN to simulate a replug."""
    USBDEVFS_RESET = 21780
    import os
    import subprocess
    # /sys/class/video4linux/videoN/device -> USB interface dir, parent is the USB device
    sysfs = f"/sys/class/video4linux/video{device}/device/.."
    try:
        usb_path = os.path.realpath(sysfs)
        busnum = open(f"{usb_path}/busnum").read().strip()
        devnum = open(f"{usb_path}/devnum").read().strip()
        usb_dev = f"/dev/bus/usb/{int(busnum):03d}/{int(devnum):03d}"
        # Use sudo to reset since USB device files need root
        subprocess.run(
            ["sudo", "python3", "-c",
             f"import fcntl; fd=open('{usb_dev}','wb'); fcntl.ioctl(fd,{USBDEVFS_RESET},0); fd.close()"],
            check=True, timeout=5
        )
        time.sleep(1)
    except (OSError, FileNotFoundError, subprocess.SubprocessError):
        pass  # Best effort

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class RealSenseCamera:
    """Manages an Intel RealSense camera for RGB frame capture."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        if rs is None:
            raise ImportError("pyrealsense2 is required. Install with: pip install pyrealsense2")
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._started = False

    def start(self) -> None:
        """Start the camera stream."""
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.pipeline.start(self.config)
        self._started = True
        # Allow auto-exposure to settle
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def read(self) -> np.ndarray:
        """Capture a single RGB frame.

        Returns:
            np.ndarray of shape (H, W, 3), dtype uint8, RGB order.
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame")
        return np.asanyarray(color_frame.get_data())

    def stop(self) -> None:
        """Stop the camera stream."""
        if self._started:
            self.pipeline.stop()
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class USBCamera:
    """USB camera via V4L2 with MJPG format."""

    def __init__(self, device: int = 0, width: int = 1280, height: int = 720, fps: int = 60, flip: bool = False):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip
        self.cap = None

    def start(self) -> None:
        _usb_reset(self.device)
        max_attempts = 5
        for attempt in range(max_attempts):
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not self.cap.isOpened():
                self.cap.release()
                time.sleep(2)
                continue
            # Give the camera time to initialize streaming
            time.sleep(0.5)
            ret = self.cap.grab()
            if ret:
                break
            self.cap.release()
            time.sleep(2)
        else:
            raise RuntimeError(f"Failed to open camera device {self.device} after {max_attempts} attempts")
        atexit.register(self.stop)
        # Warm up
        for _ in range(10):
            self.cap.read()

    def read(self) -> np.ndarray:
        """Capture a single RGB frame."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from USB camera")
        if self.flip:
            frame = cv2.flip(frame, -1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class CSICamera:
    """MIPI CSI camera via GStreamer on Jetson."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 352, fps: int = 30, flip: bool = False):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip
        self.cap = None

    def _gstreamer_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self.device} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"framerate={self.fps}/1, format=NV12 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )

    def start(self) -> None:
        gst = self._gstreamer_pipeline()
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open CSI camera device {self.device}")
        # Warm up
        for _ in range(10):
            self.cap.read()

    def read(self) -> np.ndarray:
        """Capture a single RGB frame.

        Returns:
            np.ndarray of shape (H, W, 3), dtype uint8, RGB order.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from CSI camera")
        if self.flip:
            frame = cv2.flip(frame, -1)  # Rotate 180 degrees
        # OpenCV captures BGR; convert to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
