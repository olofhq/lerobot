#!/usr/bin/env python3
"""Preview both camera streams with before/after crop side by side.

Press 'q' to quit.
"""

import cv2
import numpy as np

from camera import CSICamera, RealSenseCamera, USBCamera
from utils import load_config


def main():
    cfg = load_config("config.yaml")

    # Build cameras
    cameras = {}
    for cam_name, cam_cfg in cfg["camera"].items():
        if cam_cfg["type"] == "realsense":
            cameras[cam_name] = {
                "cam": RealSenseCamera(
                    width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"]
                ),
                "model_h": cam_cfg["height"],
                "model_w": cam_cfg["width"],
            }
        elif cam_cfg["type"].lower() == "usb":
            cameras[cam_name] = {
                "cam": USBCamera(
                    device=cam_cfg.get("device", 0),
                    width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"],
                    flip=cam_cfg.get("flip", False),
                ),
                "model_h": cam_cfg["height"],
                "model_w": cam_cfg["width"],
            }
        elif cam_cfg["type"] == "csi":
            cameras[cam_name] = {
                "cam": CSICamera(
                    device=cam_cfg.get("device", 0),
                    width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"],
                    flip=cam_cfg.get("flip", False),
                ),
                "model_h": cam_cfg["height"],
                "model_w": cam_cfg["width"],
            }

    # Start all cameras
    for info in cameras.values():
        info["cam"].start()

    print("Cameras started. Press 'q' to quit.")

    try:
        while True:
            panels = []

            for cam_name, info in cameras.items():
                frame_rgb = info["cam"].read()  # (H, W, 3) RGB
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Resize original to max 480px height for display
                h, w = frame_bgr.shape[:2]
                scale = min(480 / h, 640 / w)
                display_raw = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

                # Crop/resize to model input size
                model_h, model_w = info["model_h"], info["model_w"]
                cropped = cv2.resize(frame_bgr, (model_w, model_h), interpolation=cv2.INTER_LINEAR)

                # Scale cropped to same display height
                disp_h = display_raw.shape[0]
                crop_scale = disp_h / model_h
                display_crop = cv2.resize(cropped, (int(model_w * crop_scale), disp_h))

                # Add labels
                cv2.putText(display_raw, f"{cam_name} raw ({w}x{h})",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_crop, f"{cam_name} model ({model_w}x{model_h})",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Stack raw and cropped horizontally
                pair = np.hstack([display_raw, display_crop])
                panels.append(pair)

            # Stack all camera pairs vertically
            # Pad to same width
            max_w = max(p.shape[1] for p in panels)
            padded = []
            for p in panels:
                if p.shape[1] < max_w:
                    pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                    p = np.hstack([p, pad])
                padded.append(p)

            combined = np.vstack(padded)
            cv2.imshow("Camera Preview (q to quit)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        for info in cameras.values():
            info["cam"].stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
