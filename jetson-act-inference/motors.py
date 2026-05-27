"""Feetech STS3215 servo motor bus: read positions, write goals, calibration.

Matches the normalization logic from LeRobot's motors_bus.py exactly.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scservo_sdk import PacketHandler, PortHandler, GroupSyncRead, GroupSyncWrite

# STS3215 register addresses and lengths
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
LEN_GOAL_POSITION = 2
ADDR_PRESENT_POSITION = 56
LEN_PRESENT_POSITION = 2

# STS3215 resolution: 12-bit (0-4095)
MOTOR_RESOLUTION = 4096
MAX_RES = MOTOR_RESOLUTION - 1  # 4095

# Sign-magnitude encoding: bit 15 is sign for Present_Position
SIGN_BIT_INDEX = 15

# Feetech protocol version
PROTOCOL_VERSION = 0


@dataclass
class MotorCalibration:
    """Calibration data for a single motor, matching LeRobot's format."""
    id: int
    drive_mode: int       # 0=normal, 1=inverted
    homing_offset: int    # Applied by hardware on STS3215
    range_min: int        # Min raw position from calibration
    range_max: int        # Max raw position from calibration


@dataclass
class MotorDef:
    """Motor definition with name, ID, and normalization mode."""
    name: str
    motor_id: int
    norm_mode: str  # "degrees" or "range_0_100"


def decode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    """Decode sign-magnitude encoded servo value.

    Feetech STS3215 uses bit 15 as sign bit for position values.
    """
    direction_bit = (value >> sign_bit_index) & 1
    magnitude_mask = (1 << sign_bit_index) - 1
    magnitude = value & magnitude_mask
    return -magnitude if direction_bit else magnitude


def encode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    """Encode a signed value into sign-magnitude format for writing."""
    if value < 0:
        return (1 << sign_bit_index) | (-value)
    return value


class FeetechMotorBus:
    """Minimal Feetech STS3215 motor bus for SO-101 robot arm.

    Handles sync read/write of 6 motors with calibration-based normalization.
    """

    def __init__(self, port: str, baudrate: int, motors: list[MotorDef]):
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        self.motors = motors
        self.motor_ids = [m.motor_id for m in motors]
        self.calibration: dict[str, MotorCalibration] = {}
        self.baudrate = baudrate
        self._connected = False

    def connect(self) -> None:
        """Open serial port and enable torque on all motors."""
        if not self.port_handler.openPort():
            raise ConnectionError(f"Failed to open port {self.port_handler.port_name}")
        if not self.port_handler.setBaudRate(self.baudrate):
            raise ConnectionError(f"Failed to set baudrate {self.baudrate}")
        self._connected = True

        for motor in self.motors:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor.motor_id, ADDR_TORQUE_ENABLE, 1
            )

    def disconnect(self) -> None:
        """Disable torque and close port."""
        if not self._connected:
            return
        for motor in self.motors:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor.motor_id, ADDR_TORQUE_ENABLE, 0
            )
        self.port_handler.closePort()
        self._connected = False

    def load_calibration(self, calibration_path: Path) -> None:
        """Load calibration from JSON file (LeRobot format).

        Expected format:
        {
          "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0,
                           "range_min": 500, "range_max": 3595},
          ...
        }
        """
        with open(calibration_path) as f:
            data = json.load(f)
        self.calibration = {}
        for name, cal in data.items():
            self.calibration[name] = MotorCalibration(
                id=cal["id"],
                drive_mode=cal["drive_mode"],
                homing_offset=cal["homing_offset"],
                range_min=cal["range_min"],
                range_max=cal["range_max"],
            )

    def sync_read_positions(self) -> dict[str, int]:
        """Read raw Present_Position from all motors via sync_read.

        Returns dict of {motor_name: decoded_raw_value}.
        Note: hardware already applies homing_offset to Present_Position.
        """
        sync_read = GroupSyncRead(
            self.port_handler, self.packet_handler,
            ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for mid in self.motor_ids:
            sync_read.addParam(mid)

        result = sync_read.txRxPacket()
        if result != 0:
            raise RuntimeError(f"sync_read failed with error code {result}")

        positions = {}
        for motor in self.motors:
            raw = sync_read.getData(motor.motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            positions[motor.name] = decode_sign_magnitude(raw, SIGN_BIT_INDEX)

        sync_read.clearParam()
        return positions

    def read_normalized(self, joint_order: list[str]) -> np.ndarray:
        """Read all motor positions, apply calibration normalization.

        Returns np.ndarray of shape (n_motors,) in the given joint order,
        with values in degrees (arm) or 0-100 (gripper).
        """
        raw_positions = self.sync_read_positions()
        result = np.zeros(len(joint_order), dtype=np.float32)

        for i, name in enumerate(joint_order):
            motor = next(m for m in self.motors if m.name == name)
            cal = self.calibration[name]
            val = raw_positions[name]
            result[i] = self._normalize_value(val, motor.norm_mode, cal)

        return result

    def write_normalized(self, actions: np.ndarray, joint_order: list[str]) -> None:
        """Write goal positions to all motors from normalized values.

        Args:
            actions: np.ndarray of shape (n_motors,) with normalized values
                     (degrees for arm joints, 0-100 for gripper).
            joint_order: list of joint names matching action vector order.
        """
        sync_write = GroupSyncWrite(
            self.port_handler, self.packet_handler,
            ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )

        for i, name in enumerate(joint_order):
            motor = next(m for m in self.motors if m.name == name)
            cal = self.calibration[name]
            raw = self._unnormalize_value(float(actions[i]), motor.norm_mode, cal)
            encoded = encode_sign_magnitude(raw, SIGN_BIT_INDEX)
            # Pack as 2 bytes, little-endian
            param = [encoded & 0xFF, (encoded >> 8) & 0xFF]
            sync_write.addParam(motor.motor_id, param)

        result = sync_write.txPacket()
        if result != 0:
            raise RuntimeError(f"sync_write failed with error code {result}")
        sync_write.clearParam()

    @staticmethod
    def _normalize_value(val: int, norm_mode: str, cal: MotorCalibration) -> float:
        """Convert raw servo value to normalized value.

        Matches LeRobot's motors_bus.py _normalize method exactly.
        """
        min_ = cal.range_min
        max_ = cal.range_max

        if norm_mode == "degrees":
            mid = (min_ + max_) / 2.0
            return (val - mid) * 360.0 / MAX_RES

        elif norm_mode == "range_0_100":
            bounded = min(max_, max(min_, val))
            norm = ((bounded - min_) / (max_ - min_)) * 100.0
            return (100.0 - norm) if cal.drive_mode else norm

        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    @staticmethod
    def _unnormalize_value(val: float, norm_mode: str, cal: MotorCalibration) -> int:
        """Convert normalized value back to raw servo value.

        Matches LeRobot's motors_bus.py _unnormalize method exactly.
        """
        min_ = cal.range_min
        max_ = cal.range_max

        if norm_mode == "degrees":
            mid = (min_ + max_) / 2.0
            return int((val * MAX_RES / 360.0) + mid)

        elif norm_mode == "range_0_100":
            v = (100.0 - val) if cal.drive_mode else val
            bounded = min(100.0, max(0.0, v))
            return int((bounded / 100.0) * (max_ - min_) + min_)

        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
