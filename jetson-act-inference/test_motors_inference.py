#!/usr/bin/env python3
"""Read current joint positions, capture real camera frames, run 20 inference steps.

Usage:
    python test_motors_inference.py
"""

import argparse
from pathlib import Path

import numpy as np

from camera import CSICamera, RealSenseCamera
from motors import FeetechMotorBus, MotorDef
from postprocessing import Postprocessor, TemporalEnsembler
from preprocessing import Preprocessor
from utils import get_calibration_path, load_config, load_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    joint_order = cfg["motors"]["joint_order"]

    # --- Load model ---
    engine_path = cfg["model"].get("engine_path", "")
    if engine_path and Path(engine_path).exists():
        from trt_runtime import TRTSession
        session = TRTSession(engine_path)
        input_names = session.input_names
        output_names = session.output_names
    else:
        raise FileNotFoundError(f"No engine at {engine_path}")

    # --- Load normalization ---
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

    use_ensemble = cfg["inference"]["temporal_ensemble"]
    ensembler = None
    if use_ensemble:
        ensembler = TemporalEnsembler(
            chunk_size=cfg["model"]["chunk_size"],
            action_dim=cfg["model"]["action_dim"],
            coeff=cfg["inference"]["temporal_ensemble_coeff"],
        )
        ensembler.reset()

    # --- Setup cameras ---
    cameras = {}
    cam_configs = {}
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

    # --- Setup motors ---
    motor_defs = []
    for name, info in cfg["motors"]["joints"].items():
        motor_defs.append(MotorDef(name=name, motor_id=info["id"], norm_mode=info["norm_mode"]))

    bus = FeetechMotorBus(
        port=cfg["motors"]["port"],
        baudrate=cfg["motors"]["baudrate"],
        motors=motor_defs,
    )

    # --- Start everything ---
    for cam in cameras.values():
        cam.start()
    print("Cameras started.")

    bus.connect()
    cal_path = get_calibration_path(cfg["motors"]["calibration_path"])
    bus.load_calibration(cal_path)
    print("Motors connected.")

    # --- Read initial state ---
    state = bus.read_normalized(joint_order)

    print(f"\n{'='*70}")
    print("CURRENT JOINT STATE (normalized)")
    print(f"{'='*70}")
    for name, val in zip(joint_order, state):
        print(f"  {name:20s}: {val:8.2f}")

    # --- Run 20 inference steps with real cameras ---
    print(f"\n{'='*70}")
    print("PREDICTED ACTIONS (20 steps, real cameras + real state)")
    print(f"{'='*70}")
    header = "Step | " + " | ".join(f"{n:>14s}" for n in joint_order)
    print(header)
    print("-" * len(header))

    state_input_name = cfg["inference"]["state_input_name"]

    try:
        for step in range(20):
            # Capture real frames
            feeds = {}
            for input_name, cam_cfg in cam_configs.items():
                frame = cameras[input_name].read()
                feeds[input_name] = preprocessor.preprocess_image(
                    frame, input_name, cam_cfg["height"], cam_cfg["width"],
                )

            # Read real joint state
            state = bus.read_normalized(joint_order)
            feeds[state_input_name] = preprocessor.preprocess_state(state)

            # Inference
            outputs = session.run(output_names, feeds)
            raw_actions = outputs[0]
            actions = postprocessor.denormalize(raw_actions)

            if ensembler is not None:
                action = ensembler.update(actions)
            else:
                action = actions[0, 0]

            vals = " | ".join(f"{v:14.2f}" for v in action)
            print(f"{step:4d} | {vals}")

    finally:
        for cam in cameras.values():
            cam.stop()
        bus.disconnect()
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
