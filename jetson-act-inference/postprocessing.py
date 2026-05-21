"""Action postprocessing: denormalize and temporal ensemble.

Reproduces LeRobot's postprocessor denormalization and ACTTemporalEnsembler
without any PyTorch dependency.
"""

import numpy as np


class Postprocessor:
    """Denormalize raw model action outputs using training stats."""

    def __init__(
        self,
        stats: dict[str, dict[str, np.ndarray]],
        action_key: str = "action",
        action_mode: str = "mean_std",
    ):
        self.action_mode = action_mode
        if action_key in stats:
            self.action_stats = stats[action_key]
        else:
            self.action_stats = None

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Denormalize action tensor.

        Args:
            actions: shape (1, chunk_size, action_dim) or (chunk_size, action_dim).

        Returns:
            Same shape, in original (calibrated) units.
        """
        if self.action_stats is None:
            return actions

        if self.action_mode == "mean_std":
            mean = self.action_stats["mean"]
            std = self.action_stats["std"]
            return actions * std + mean
        elif self.action_mode == "min_max":
            min_val = self.action_stats["min"]
            max_val = self.action_stats["max"]
            return ((actions + 1.0) / 2.0) * (max_val - min_val) + min_val

        return actions


class TemporalEnsembler:
    """Temporal ensemble for ACT action chunks.

    Matches LeRobot's ACTTemporalEnsembler exactly.

    Weights: w_i = exp(-coeff * i) where w_0 corresponds to the oldest action.
    Higher coeff => more weight on older predictions (smoother).
    """

    def __init__(self, chunk_size: int, action_dim: int, coeff: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.coeff = coeff

        # Precompute weights and cumulative sums
        self.weights = np.exp(-coeff * np.arange(chunk_size, dtype=np.float32))
        self.weights_cumsum = np.cumsum(self.weights)

        self.ensembled_actions: np.ndarray | None = None
        self.ensembled_actions_count: np.ndarray | None = None

    def reset(self) -> None:
        """Reset ensemble state (call at start of each episode)."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: np.ndarray) -> np.ndarray:
        """Incorporate a new action chunk and return the next action.

        Args:
            actions: shape (1, chunk_size, action_dim) — a new predicted chunk.

        Returns:
            np.ndarray of shape (action_dim,) — the next action to execute.
        """
        # Remove batch dim for internal processing: (chunk_size, action_dim)
        actions = actions[0]

        if self.ensembled_actions is None:
            # First call: initialize
            self.ensembled_actions = actions.copy()
            self.ensembled_actions_count = np.ones((self.chunk_size, 1), dtype=np.int64)
        else:
            # Online weighted average update for existing entries
            count_idx = self.ensembled_actions_count - 1  # (n, 1)
            self.ensembled_actions *= self.weights_cumsum[count_idx].reshape(-1, 1)
            self.ensembled_actions += actions[:-1] * self.weights[self.ensembled_actions_count].reshape(-1, 1)
            self.ensembled_actions /= self.weights_cumsum[self.ensembled_actions_count].reshape(-1, 1)
            self.ensembled_actions_count = np.clip(self.ensembled_actions_count + 1, a_min=None, a_max=self.chunk_size)

            # Append the newest action (no prior average)
            self.ensembled_actions = np.concatenate(
                [self.ensembled_actions, actions[-1:]], axis=0
            )
            self.ensembled_actions_count = np.concatenate(
                [self.ensembled_actions_count, np.ones((1, 1), dtype=np.int64)], axis=0
            )

        # Consume the first action
        action = self.ensembled_actions[0].copy()
        self.ensembled_actions = self.ensembled_actions[1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]

        return action
