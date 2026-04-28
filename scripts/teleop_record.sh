#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --dataset-name <name> [options]"
    echo ""
    echo "Required:"
    echo "  --dataset-name       Dataset name (e.g. wrist_depth)"
    echo ""
    echo "Optional:"
    echo "  --robot-port         Robot serial port (default: /dev/ttyACM0)"
    echo "  --robot-id           Robot ID (default: thing)"
    echo "  --teleop-port        Teleop serial port (default: /dev/ttyACM1)"
    echo "  --teleop-id          Teleop ID (default: it)"
    echo "  --task               Single task description (default: Pick tee)"
    echo "  --num-episodes       Number of episodes (default: 20)"
    echo "  --episode-time       Episode time in seconds (default: 9000)"
    echo "  --zmq-address        ZMQ camera server address (default: 192.168.128.10)"
    echo "  --zmq-port           ZMQ camera port (default: 5555)"
    echo "  --realsense-serial   RealSense serial number (default: 353322270661)"
    echo "  --display-data       Display data (default: true)"
    echo "  --push-to-hub        Push to hub (default: False)"
    echo "  --play-sounds        Play sounds (default: False)"
    echo "  --resume             Resume recording (default: True)"
    exit 1
}

# Defaults
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="thing"
TELEOP_PORT="/dev/ttyACM1"
TELEOP_ID="it"
TASK="Pick tee"
NUM_EPISODES=20
EPISODE_TIME=9000
ZMQ_ADDRESS="192.168.128.10"
ZMQ_PORT=5555
REALSENSE_SERIAL="353322270661"
DISPLAY_DATA="true"
PUSH_TO_HUB="False"
PLAY_SOUNDS="False"
RESUME="True"
DATASET_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-name)      DATASET_NAME="$2"; shift 2 ;;
        --robot-port)        ROBOT_PORT="$2"; shift 2 ;;
        --robot-id)          ROBOT_ID="$2"; shift 2 ;;
        --teleop-port)       TELEOP_PORT="$2"; shift 2 ;;
        --teleop-id)         TELEOP_ID="$2"; shift 2 ;;
        --task)              TASK="$2"; shift 2 ;;
        --num-episodes)      NUM_EPISODES="$2"; shift 2 ;;
        --episode-time)      EPISODE_TIME="$2"; shift 2 ;;
        --zmq-address)       ZMQ_ADDRESS="$2"; shift 2 ;;
        --zmq-port)          ZMQ_PORT="$2"; shift 2 ;;
        --realsense-serial)  REALSENSE_SERIAL="$2"; shift 2 ;;
        --display-data)      DISPLAY_DATA="$2"; shift 2 ;;
        --push-to-hub)       PUSH_TO_HUB="$2"; shift 2 ;;
        --play-sounds)       PLAY_SOUNDS="$2"; shift 2 ;;
        --resume)            RESUME="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *)                   EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset-name is required"
    usage
fi

REPO_ID="tee/${DATASET_NAME}"
DATASET_ROOT="$HOME/.cache/huggingface/lerobot/${REPO_ID}"

exec lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --teleop.type=so101_leader \
    --teleop.port="$TELEOP_PORT" \
    --teleop.id="$TELEOP_ID" \
    --robot.cameras="{front: {type: zmq, server_address: '$ZMQ_ADDRESS', port: $ZMQ_PORT, camera_name: 'front', width: 640, height: 352, fps: 10}, wrist: {type: intelrealsense, use_depth: true, width: 640, height: 480, fps: 30, serial_number_or_name: $REALSENSE_SERIAL}}" \
    --display_data="$DISPLAY_DATA" \
    --dataset.single_task="$TASK" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --play_sounds="$PLAY_SOUNDS" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s "$EPISODE_TIME" \
    --resume "$RESUME" \
    --dataset.root "$DATASET_ROOT" \
    "${EXTRA_ARGS[@]}"
