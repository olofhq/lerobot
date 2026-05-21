"""Image and state preprocessing: resize, normalize to match policy training stats.

Reproduces the LeRobot processor pipeline without PyTorch:
  1. Resize image to model resolution (bilinear) if camera res differs
  2. Convert to float32, scale [0,1]
  3. Rearrange HWC -> CHW
  4. Normalize with stats from policy_preprocessor.safetensors
  5. Normalize state vector with stats from policy_preprocessor.safetensors
"""

import cv2
import numpy as np


def _get_image_stats(stats: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean/std for an image key, reshaped to (C,1,1)."""
    if key in stats:
        mean = stats[key]["mean"].astype(np.float32)
        std = stats[key]["std"].astype(np.float32)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1, 1)
            std = std.reshape(-1, 1, 1)
        return mean, std
    return np.zeros((3, 1, 1), dtype=np.float32), np.ones((3, 1, 1), dtype=np.float32)


class Preprocessor:
    """Preprocesses camera frames and state vectors for inference."""

    def __init__(
        self,
        stats: dict[str, dict[str, np.ndarray]],
        image_keys: list[str],
        state_key: str = "observation.state",
        state_mode: str = "mean_std",
    ):
        self.state_mode = state_mode

        # Per-camera normalization stats
        self.image_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key in image_keys:
            self.image_stats[key] = _get_image_stats(stats, key)

        # State normalization stats
        self.state_stats = stats.get(state_key)

    def preprocess_image(self, frame: np.ndarray, input_name: str,
                         model_height: int, model_width: int) -> np.ndarray:
        """Convert a raw camera frame to model input tensor.

        Args:
            frame: uint8 RGB image, shape (H, W, 3).
            input_name: ONNX/TRT input tensor name (e.g. "observation.images.wrist").
            model_height: expected height for this input.
            model_width: expected width for this input.

        Returns:
            np.ndarray of shape (1, 3, model_height, model_width), float32.
        """
        h, w = frame.shape[:2]
        if h != model_height or w != model_width:
            frame = cv2.resize(frame, (model_width, model_height), interpolation=cv2.INTER_LINEAR)

        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.clip(img, 0.0, 1.0)

        mean, std = self.image_stats.get(
            input_name,
            (np.zeros((3, 1, 1), dtype=np.float32), np.ones((3, 1, 1), dtype=np.float32)),
        )
        img = (img - mean) / (std + 1e-8)

        return img[np.newaxis]  # (1, 3, H, W)

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state vector using training stats.

        Args:
            state: shape (action_dim,), float32, in calibrated units
                   (degrees for arm, 0-100 for gripper).

        Returns:
            np.ndarray of shape (1, action_dim), float32, normalized.
        """
        state = state.astype(np.float32)

        if self.state_stats is not None:
            if self.state_mode == "mean_std":
                eps = 1e-8
                mean = self.state_stats["mean"]
                std = self.state_stats["std"]
                state = (state - mean) / (std + eps)
            elif self.state_mode == "min_max":
                min_val = self.state_stats["min"]
                max_val = self.state_stats["max"]
                state = 2.0 * (state - min_val) / (max_val - min_val) - 1.0

        # Add batch dimension: (1, action_dim)
        return state[np.newaxis]
