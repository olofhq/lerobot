#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --policy-path <path> [options]"
    echo ""
    echo "Required:"
    echo "  --policy-path        Path to pretrained model checkpoint"
    echo ""
    echo "Optional:"
    echo "  --robot-port         Robot serial port (default: /dev/ttyACM0)"
    echo "  --robot-id           Robot ID (default: thing)"
    echo "  --repo-id            Dataset repo ID (default: tee/eval_temp)"
    echo "  --task               Single task description (default: Pick tee)"
    echo "  --n-action-steps     Number of action steps (default: 50)"
    echo "  --zmq-address        ZMQ camera server address (default: 192.168.128.10)"
    echo "  --zmq-port           ZMQ camera port (default: 5555)"
    echo "  --realsense-serial   RealSense serial number (default: 353322270661)"
    echo "  --display-data       Display data (default: true)"
    echo "  --push-to-hub        Push to hub (default: False)"
    echo "  --play-sounds        Play sounds (default: False)"
    exit 1
}

# Defaults
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="thing"
REPO_ID="tee/eval_temp"
TASK="Pick tee"
N_ACTION_STEPS=50
ZMQ_ADDRESS="192.168.128.10"
ZMQ_PORT=5555
REALSENSE_SERIAL="353322270661"
DISPLAY_DATA="true"
PUSH_TO_HUB="False"
PLAY_SOUNDS="False"
POLICY_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-path)       POLICY_PATH="$2"; shift 2 ;;
        --robot-port)        ROBOT_PORT="$2"; shift 2 ;;
        --robot-id)          ROBOT_ID="$2"; shift 2 ;;
        --repo-id)           REPO_ID="$2"; shift 2 ;;
        --task)              TASK="$2"; shift 2 ;;
        --n-action-steps)    N_ACTION_STEPS="$2"; shift 2 ;;
        --zmq-address)       ZMQ_ADDRESS="$2"; shift 2 ;;
        --zmq-port)          ZMQ_PORT="$2"; shift 2 ;;
        --realsense-serial)  REALSENSE_SERIAL="$2"; shift 2 ;;
        --display-data)      DISPLAY_DATA="$2"; shift 2 ;;
        --push-to-hub)       PUSH_TO_HUB="$2"; shift 2 ;;
        --play-sounds)       PLAY_SOUNDS="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *)                   echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$POLICY_PATH" ]]; then
    echo "Error: --policy-path is required"
    usage
fi

CACHE_DIR="$HOME/.cache/huggingface/lerobot/${REPO_ID}"
echo "Clearing cache: $CACHE_DIR"
rm -rf "$CACHE_DIR"

exec lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.cameras="{front: {type: zmq, server_address: '$ZMQ_ADDRESS', port: $ZMQ_PORT, camera_name: 'front', width: 640, height: 352, fps: 10}, wrist: {type: intelrealsense, use_depth: true, width: 640, height: 480, fps: 30, serial_number_or_name: $REALSENSE_SERIAL}}" \
    --display_data="$DISPLAY_DATA" \
    --dataset.single_task="$TASK" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --play_sounds="$PLAY_SOUNDS" \
    --policy.path="$POLICY_PATH" \
    --dataset.repo_id="$REPO_ID" \
    --policy.n_action_steps="$N_ACTION_STEPS"
