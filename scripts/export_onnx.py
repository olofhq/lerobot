"""Export a pretrained ACT policy to ONNX for visualization in Netron."""

import argparse
import torch

from lerobot.configs.types import FeatureType


class ACTWrapper(torch.nn.Module):
    """Wraps ACT policy to accept flat tensor inputs instead of a dict (required for ONNX export)."""

    def __init__(self, policy, input_keys):
        super().__init__()
        self.policy = policy
        self.input_keys = input_keys

    def forward(self, *args):
        batch = dict(zip(self.input_keys, args))
        actions = self.policy.predict_action_chunk(batch)
        return actions


def main():
    parser = argparse.ArgumentParser(description="Export a pretrained ACT policy to ONNX.")
    parser.add_argument("--input", required=True, help="Path to the pretrained model directory.")
    parser.add_argument("--output", required=True, help="Path for the exported ONNX file.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18).")
    args = parser.parse_args()

    from lerobot.policies.act.modeling_act import ACTPolicy

    print(f"Loading model from {args.input}")
    policy = ACTPolicy.from_pretrained(args.input)
    policy.eval()

    device = "cpu"
    policy = policy.to(device)

    # Build input keys and dummy tensors from the model's actual config
    input_keys = []
    dummy_inputs = []
    for key, feature in policy.config.input_features.items():
        if feature.type == FeatureType.ACTION:
            continue
        input_keys.append(key)
        dummy_inputs.append(torch.randn(1, *feature.shape, device=device))

    print(f"Model inputs: {input_keys}")

    wrapper = ACTWrapper(policy, input_keys)
    wrapper.eval()

    dynamic_axes = {key: {0: "batch"} for key in input_keys}
    dynamic_axes["actions"] = {0: "batch"}

    print(f"Exporting to {args.output}")
    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs),
        args.output,
        input_names=input_keys,
        output_names=["actions"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Done! Open {args.output} in Netron (https://netron.app)")


if __name__ == "__main__":
    main()
