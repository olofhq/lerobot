#!/bin/bash
# Convert an ONNX model to a TensorRT engine for Jetson GPU inference.
# trtexec is preinstalled with JetPack — no extra packages needed.
#
# Usage:
#   ./convert_engine.sh model/act_policy.onnx
#   ./convert_engine.sh model/act_policy.onnx --best  # INT8+FP16 (fastest, may lose accuracy)

set -e

ONNX_PATH="${1:?Usage: $0 <model.onnx> [trtexec flags...]}"
shift
ENGINE_PATH="${ONNX_PATH%.onnx}.engine"

echo "Converting: ${ONNX_PATH} → ${ENGINE_PATH}"
echo "Extra flags: $@"

trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --fp16 \
    "$@"

echo ""
echo "Done! Engine saved to: ${ENGINE_PATH}"
echo "Set engine_path in config.yaml to: ${ENGINE_PATH}"
