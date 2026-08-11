#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --policy-path <path> --dataset-name <name> [options]"
    echo ""
    echo "Human-in-the-loop (DAgger) data collection via lerobot-rollout."
    echo ""
    echo "Required:"
    echo "  --policy-path        Path to pretrained model checkpoint"
    echo "  --dataset-name       Dataset name (e.g. wrist_depth)"
    echo ""
    echo "Optional:"
    echo "  --robot-port         Robot serial port (default: /dev/follower_arm)"
    echo "  --robot-id           Robot ID (default: biggie)"
    echo "  --teleop-port        Teleop serial port (default: /dev/leader_arm)"
    echo "  --teleop-id          Teleop ID (default: it)"
    echo "  --task               Single task description (default: Pick tee)"
    echo "  --num-episodes       Number of episodes (default: 20)"
    echo "  --camera-index       Front camera device index or path (default: /dev/see3cam)"
    echo "  --camera-fps         Front camera FPS (default: 60)"
    echo "  --camera-width       Front camera width (default: 1280)"
    echo "  --camera-height      Front camera height (default: 720)"
    echo "  --realsense-serial   RealSense serial number (default: 353322270661)"
    echo "  --input-device       Input device for HIL controls: keyboard|pedal (default: pedal)"
    echo "  --display-data       Display data (default: true)"
    echo "  --push-to-hub        Push to hub (default: False)"
    echo "  --play-sounds        Play sounds (default: False)"
    echo "  --resume             Resume recording (default: False)"
    echo "  --clean              Remove existing dataset before starting"
    echo "  --n-action-steps     Policy action steps (default: 1)"
    echo "  --temporal-ensemble-coeff  Temporal ensemble coefficient (default: 0.01)"
    exit 1
}

# Defaults
ROBOT_PORT="/dev/follower_arm"
ROBOT_ID="rake2"
TELEOP_PORT="/dev/leader_arm"
TELEOP_ID="it"
TASK="Pick tee"
NUM_EPISODES=20
CAMERA_INDEX="/dev/see3cam"
CAMERA_FPS=60
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
REALSENSE_SERIAL="353322270661"
INPUT_DEVICE="pedal"
DISPLAY_DATA="true"
PUSH_TO_HUB="False"
PLAY_SOUNDS="False"
RESUME="False"
CLEAN=false
N_ACTION_STEPS=1
TEMPORAL_ENSEMBLE_COEFF=0.01
DATASET_NAME=""
POLICY_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-name)      DATASET_NAME="$2"; shift 2 ;;
        --policy-path)       POLICY_PATH="$2"; shift 2 ;;
        --robot-port)        ROBOT_PORT="$2"; shift 2 ;;
        --robot-id)          ROBOT_ID="$2"; shift 2 ;;
        --teleop-port)       TELEOP_PORT="$2"; shift 2 ;;
        --teleop-id)         TELEOP_ID="$2"; shift 2 ;;
        --task)              TASK="$2"; shift 2 ;;
        --num-episodes)      NUM_EPISODES="$2"; shift 2 ;;
        --camera-index)      CAMERA_INDEX="$2"; shift 2 ;;
        --camera-fps)        CAMERA_FPS="$2"; shift 2 ;;
        --camera-width)      CAMERA_WIDTH="$2"; shift 2 ;;
        --camera-height)     CAMERA_HEIGHT="$2"; shift 2 ;;
        --realsense-serial)  REALSENSE_SERIAL="$2"; shift 2 ;;
        --input-device)      INPUT_DEVICE="$2"; shift 2 ;;
        --display-data)      DISPLAY_DATA="$2"; shift 2 ;;
        --push-to-hub)       PUSH_TO_HUB="$2"; shift 2 ;;
        --play-sounds)       PLAY_SOUNDS="$2"; shift 2 ;;
        --resume)            RESUME="$2"; shift 2 ;;
        --clean)             CLEAN=true; shift ;;
        --n-action-steps)    N_ACTION_STEPS="$2"; shift 2 ;;
        --temporal-ensemble-coeff) TEMPORAL_ENSEMBLE_COEFF="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *)                   EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$POLICY_PATH" ]]; then
    echo "Error: --policy-path is required"
    usage
fi

if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset-name is required"
    usage
fi

if [[ "$INPUT_DEVICE" != "keyboard" && "$INPUT_DEVICE" != "pedal" ]]; then
    echo "Error: --input-device must be 'keyboard' or 'pedal'"
    usage
fi

REPO_ID="tee_usbc/${DATASET_NAME}"
DATASET_ROOT="$HOME/.cache/huggingface/lerobot/${REPO_ID}"

if [[ "$CLEAN" == true ]]; then
    echo "Removing existing dataset: $DATASET_ROOT"
    rm -rf "$DATASET_ROOT"
fi

exec lerobot-rollout \
    --strategy.type=dagger \
    --strategy.input_device="$INPUT_DEVICE" \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --teleop.type=so101_leader \
    --teleop.port="$TELEOP_PORT" \
    --teleop.id="$TELEOP_ID" \
    --robot.cameras="{front: {type: opencv, index_or_path: '$CAMERA_INDEX', width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS, fourcc: MJPG, rotation: 180}, wrist: {type: intelrealsense, use_depth: true, width: 640, height: 480, fps: 30, serial_number_or_name: $REALSENSE_SERIAL}}" \
    --display_data="$DISPLAY_DATA" \
    --dataset.single_task="$TASK" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --play_sounds="$PLAY_SOUNDS" \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --policy.temporal_ensemble_coeff="$TEMPORAL_ENSEMBLE_COEFF" \
    --dataset.repo_id="$REPO_ID" \
    --strategy.num_episodes="$NUM_EPISODES" \
    --resume "$RESUME" \
    --dataset.root "$DATASET_ROOT" \
    "${EXTRA_ARGS[@]}"
