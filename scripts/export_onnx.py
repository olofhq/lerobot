"""Export a pretrained ACT policy to ONNX for visualization in Netron."""

import torch
from pathlib import Path

MODEL_PATH = "/home/olofmattsson/lerobot/outputs/train/wrist_depth260416/checkpoints/500000/pretrained_model"
OUTPUT_PATH = "/home/olofmattsson/lerobot/outputs/train/wrist_depth260416/checkpoints/500000/model.onnx"


class ACTWrapper(torch.nn.Module):
    """Wraps ACT policy to accept flat tensor inputs instead of a dict (required for ONNX export)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state, img_front, img_wrist, img_wrist_depth):
        batch = {
            "observation.state": state,
            "observation.images.front": img_front,
            "observation.images.wrist": img_wrist,
            "observation.images.wrist_depth": img_wrist_depth,
        }
        actions = self.policy.predict_action_chunk(batch)
        return actions


def main():
    from lerobot.policies.act.modeling_act import ACTPolicy

    print(f"Loading model from {MODEL_PATH}")
    policy = ACTPolicy.from_pretrained(MODEL_PATH)
    policy.eval()

    device = "cpu"
    policy = policy.to(device)
    wrapper = ACTWrapper(policy)
    wrapper.eval()

    # Create dummy inputs matching config shapes (batch_size=1)
    dummy_state = torch.randn(1, 6, device=device)
    dummy_front = torch.randn(1, 3, 352, 640, device=device)
    dummy_wrist = torch.randn(1, 3, 480, 640, device=device)
    dummy_wrist_depth = torch.randn(1, 3, 480, 640, device=device)

    print(f"Exporting to {OUTPUT_PATH}")
    torch.onnx.export(
        wrapper,
        (dummy_state, dummy_front, dummy_wrist, dummy_wrist_depth),
        OUTPUT_PATH,
        input_names=["observation.state", "observation.images.front", "observation.images.wrist", "observation.images.wrist_depth"],
        output_names=["actions"],
        dynamic_axes={
            "observation.state": {0: "batch"},
            "observation.images.front": {0: "batch"},
            "observation.images.wrist": {0: "batch"},
            "observation.images.wrist_depth": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=18,
    )
    print(f"Done! Open {OUTPUT_PATH} in Netron (https://netron.app)")


if __name__ == "__main__":
    main()
