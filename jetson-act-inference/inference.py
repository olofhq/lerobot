#!/usr/bin/env python3
"""Main inference loop: camera → preprocess → model → postprocess → motors @ 30Hz.

Backend priority:
  1. TensorRT engine (.engine) — GPU, preinstalled on Jetson
  2. ONNX Runtime (.onnx) — CPU fallback

Usage:
    python inference.py                    # Run with config.yaml defaults
    python inference.py --config my.yaml   # Custom config
    python inference.py --dry-run          # No motor writes (testing)
"""

import argparse
import signal
import time
from pathlib import Path

import numpy as np

from camera import CSICamera, RealSenseCamera, USBCamera
from motors import FeetechMotorBus, MotorDef, MOTOR_RESOLUTION
from postprocessing import Postprocessor, TemporalEnsembler
from preprocessing import Preprocessor
from utils import get_calibration_path, load_config, load_stats


def build_motors(cfg: dict) -> tuple[FeetechMotorBus, list[str]]:
    """Build motor bus and joint order from config."""
    motor_defs = []
    for name, info in cfg["motors"]["joints"].items():
        motor_defs.append(MotorDef(name=name, motor_id=info["id"], norm_mode=info["norm_mode"]))

    bus = FeetechMotorBus(
        port=cfg["motors"]["port"],
        baudrate=cfg["motors"]["baudrate"],
        motors=motor_defs,
    )
    joint_order = cfg["motors"]["joint_order"]
    return bus, joint_order


def get_home_position(motor_bus: FeetechMotorBus, joint_order: list[str]) -> np.ndarray:
    """Compute home position from calibration data.

    During calibration, homing_offset is set so that the physical home pose
    reads as raw value 2047 (half of 4096 resolution). We normalize that
    value through the same calibration logic to get the home in output units.
    """
    home_raw = (MOTOR_RESOLUTION - 1) // 2  # 2047: the half-turn point

    home = np.zeros(len(joint_order), dtype=np.float32)
    for i, name in enumerate(joint_order):
        motor = next(m for m in motor_bus.motors if m.name == name)
        cal = motor_bus.calibration[name]
        home[i] = motor_bus._normalize_value(home_raw, motor.norm_mode, cal)
    return home


def return_to_home(motor_bus: FeetechMotorBus, joint_order: list[str],
                   initial_pos: np.ndarray, cfg: dict) -> None:
    """Gradually move the arm back to its initial position (captured at startup)."""
    home_steps = cfg["inference"].get("home_steps", 60)
    target_dt = 1.0 / cfg["inference"]["fps"]

    current = motor_bus.read_normalized(joint_order)

    print(f"Current position: {current}")
    print(f"Target position:  {initial_pos}")
    print(f"Returning to initial position over {home_steps} steps at {cfg['inference']['fps']} Hz...")
    for i in range(1, home_steps + 1):
        alpha = i / home_steps
        interpolated = current + alpha * (initial_pos - current)
        motor_bus.write_normalized(interpolated, joint_order)
        time.sleep(target_dt)
    print("Position reached.")


def build_session(cfg: dict):
    """Build inference session: TensorRT engine if available, else ONNX Runtime."""
    engine_path = cfg["model"].get("engine_path", "")
    onnx_path = cfg["model"].get("onnx_path", "")

    if engine_path and Path(engine_path).exists():
        from trt_runtime import TRTSession
        return TRTSession(engine_path)

    if onnx_path and Path(onnx_path).exists():
        try:
            import onnxruntime as ort
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if p in ort.get_available_providers()]
            session = ort.InferenceSession(onnx_path, providers=providers)
            print(f"ONNX Runtime providers: {session.get_providers()}")
            return session
        except ImportError:
            pass

    raise FileNotFoundError(
        f"No model found. Provide either:\n"
        f"  engine_path: {engine_path}  (run: trtexec --onnx=model.onnx --saveEngine=model.engine --fp16)\n"
        f"  onnx_path:   {onnx_path}    (pip install onnxruntime)"
    )


def main():
    parser = argparse.ArgumentParser(description="ACT policy inference on Jetson Orin Nano")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Skip motor writes (camera + model only)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Load normalization stats ──────────────────────────────────────
    pre_stats = load_stats(cfg["model"]["preprocessor_path"])
    post_stats = load_stats(cfg["model"]["postprocessor_path"])

    preprocessor = Preprocessor(
        stats=pre_stats,
        image_keys=[cam["input_name"] for cam in cfg["camera"].values()],
        state_key=cfg["normalization"]["state_key"],
        state_mode=cfg["normalization"]["state_mode"],
    )

    postprocessor = Postprocessor(
        stats=post_stats,
        action_key=cfg["normalization"]["action_key"],
        action_mode=cfg["normalization"]["action_mode"],
    )

    # ── Temporal ensemble ─────────────────────────────────────────────
    use_ensemble = cfg["inference"]["temporal_ensemble"]
    ensembler = None
    if use_ensemble:
        ensembler = TemporalEnsembler(
            chunk_size=cfg["model"]["chunk_size"],
            action_dim=cfg["model"]["action_dim"],
            coeff=cfg["inference"]["temporal_ensemble_coeff"],
        )

    # ── Model session ─────────────────────────────────────────────────
    session = build_session(cfg)
    # TRTSession exposes .input_names/.output_names directly;
    # onnxruntime.InferenceSession uses .get_inputs()/.get_outputs()
    if hasattr(session, "input_names"):
        input_names = session.input_names
        output_names = session.output_names
    else:
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
    print(f"Model inputs:  {input_names}")
    print(f"Model outputs: {output_names}")

    # ── Cameras ────────────────────────────────────────────────────────
    cameras: dict[str, RealSenseCamera | CSICamera | USBCamera] = {}
    cam_configs: dict[str, dict] = {}
    for cam_name, cam_cfg in cfg["camera"].items():
        cam_configs[cam_cfg["input_name"]] = cam_cfg
        if cam_cfg["type"] == "realsense":
            cameras[cam_cfg["input_name"]] = RealSenseCamera(
                width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"],
            )
        elif cam_cfg["type"] == "csi":
            cameras[cam_cfg["input_name"]] = CSICamera(
                device=cam_cfg.get("device", 0),
                width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"],
                flip=cam_cfg.get("flip", False),
            )
        elif cam_cfg["type"] == "usb":
            cameras[cam_cfg["input_name"]] = USBCamera(
                device=cam_cfg.get("device", 0),
                width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"],
                flip=cam_cfg.get("flip", False),
            )
        else:
            raise ValueError(f"Unknown camera type: {cam_cfg['type']}")

    # ── Motors ────────────────────────────────────────────────────────
    motor_bus, joint_order = build_motors(cfg)
    calibration_path = get_calibration_path(cfg["motors"]["calibration_path"])

    # ── Graceful shutdown ─────────────────────────────────────────────
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ── Main loop ─────────────────────────────────────────────────────
    target_dt = 1.0 / cfg["inference"]["fps"]
    state_input_name = cfg["inference"]["state_input_name"]

    try:
        if not args.dry_run:
            # Start RealSense cameras first, before USB cameras (whose _usb_reset
            # can disrupt other devices on the same USB bus).
            for name, cam in sorted(
                cameras.items(),
                key=lambda x: 0 if isinstance(x[1], RealSenseCamera) else 1,
            ):
                cam.start()
            print(f"Cameras started: {list(cameras.keys())}")
            motor_bus.connect()
            motor_bus.load_calibration(calibration_path)
            print(f"Motors connected, calibration loaded from {calibration_path}")
            initial_position = motor_bus.read_normalized(joint_order)
            print(f"Initial position captured: {initial_position}")
        else:
            print("Dry-run mode: skipping camera/motor initialization")

        if ensembler is not None:
            ensembler.reset()

        step = 0
        print(f"Running inference at {cfg['inference']['fps']} Hz. Press Ctrl+C to stop.")

        while running:
            t_start = time.monotonic()

            # 1. Capture frames from all cameras
            feeds: dict[str, np.ndarray] = {}
            for input_name, cam_cfg in cam_configs.items():
                if args.dry_run:
                    frame = np.zeros((cam_cfg["height"], cam_cfg["width"], 3), dtype=np.uint8)
                else:
                    frame = cameras[input_name].read()  # (H, W, 3) uint8 RGB
                feeds[input_name] = preprocessor.preprocess_image(
                    frame, input_name, cam_cfg["height"], cam_cfg["width"],
                )

            # 2. Read motor positions
            if args.dry_run:
                state = np.zeros(cfg["model"]["action_dim"], dtype=np.float32)
            else:
                state = motor_bus.read_normalized(joint_order)

            # 3. Preprocess state
            feeds[state_input_name] = preprocessor.preprocess_state(state)

            # 4. Run model
            outputs = session.run(output_names, feeds)
            raw_actions = outputs[0]  # (1, chunk_size, action_dim)

            # 5. Denormalize actions
            actions = postprocessor.denormalize(raw_actions)  # (1, chunk_size, action_dim)

            # 6. Temporal ensemble or take first action
            if ensembler is not None:
                action = ensembler.update(actions)  # (action_dim,)
            else:
                action = actions[0, 0]  # First action from chunk

            # 7. Write to motors
            if not args.dry_run:
                motor_bus.write_normalized(action, joint_order)

            # 8. Maintain loop rate
            elapsed = time.monotonic() - t_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            actual_dt = time.monotonic() - t_start
            if step % 30 == 0:
                print(
                    f"Step {step:5d} | "
                    f"dt={actual_dt*1000:.1f}ms | "
                    f"action[0:3]={action[:3].round(2)}"
                )
            step += 1

    finally:
        if not args.dry_run:
            print("Stopping inference, returning arm to initial position...")
            try:
                return_to_home(motor_bus, joint_order, initial_position, cfg)
            except Exception as e:
                print(f"Warning: return to home failed: {e}")
            for cam in cameras.values():
                cam.stop()
            try:
                motor_bus.disconnect()
            except Exception as e:
                print(f"Warning: motor disconnect failed: {e}")
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
