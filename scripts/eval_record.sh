#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --policy-path <path> [options]"
    echo ""
    echo "Deploy a trained policy on a real robot and record episodes (lerobot-rollout sentry)."
    echo ""
    echo "Required:"
    echo "  --policy-path        Path to pretrained model checkpoint"
    echo ""
    echo "Optional:"
    echo "  --robot-port         Robot serial port (default: /dev/follower_arm)"
    echo "  --robot-id           Robot ID (default: rake2)"
    echo "  --repo-id            Dataset repo ID (default: tee_usbc/rollout_temp)"
    echo "  --task               Single task description (default: Pick tee)"
    echo "  --camera-index       Front camera device index or path (default: /dev/see3cam)"
    echo "  --camera-fps         Front camera FPS (default: 60)"
    echo "  --camera-width       Front camera width (default: 1280)"
    echo "  --camera-height      Front camera height (default: 720)"
    echo "  --realsense-serial   RealSense serial number (default: 353322270661)"
    echo "  --display-data       Display data (default: true)"
    echo "  --push-to-hub        Push to hub (default: False)"
    echo "  --play-sounds        Play sounds (default: False)"
    echo "  --n-action-steps     Policy action steps (default: 1)"
    echo "  --temporal-ensemble-coeff  Temporal ensemble coefficient (default: 0.01)"
    exit 1
}

# Defaults
ROBOT_PORT="/dev/follower_arm"
ROBOT_ID="rake2"
REPO_ID="tee_usbc/rollout_temp"
TASK="Pick tee"
CAMERA_INDEX="/dev/see3cam"
CAMERA_FPS=60
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
REALSENSE_SERIAL="353322270661"
DISPLAY_DATA="true"
PUSH_TO_HUB="False"
PLAY_SOUNDS="False"
N_ACTION_STEPS=1
TEMPORAL_ENSEMBLE_COEFF=0.01
POLICY_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-path)       POLICY_PATH="$2"; shift 2 ;;
        --robot-port)        ROBOT_PORT="$2"; shift 2 ;;
        --robot-id)          ROBOT_ID="$2"; shift 2 ;;
        --repo-id)           REPO_ID="$2"; shift 2 ;;
        --task)              TASK="$2"; shift 2 ;;
        --camera-index)      CAMERA_INDEX="$2"; shift 2 ;;
        --camera-fps)        CAMERA_FPS="$2"; shift 2 ;;
        --camera-width)      CAMERA_WIDTH="$2"; shift 2 ;;
        --camera-height)     CAMERA_HEIGHT="$2"; shift 2 ;;
        --realsense-serial)  REALSENSE_SERIAL="$2"; shift 2 ;;
        --display-data)      DISPLAY_DATA="$2"; shift 2 ;;
        --push-to-hub)       PUSH_TO_HUB="$2"; shift 2 ;;
        --play-sounds)       PLAY_SOUNDS="$2"; shift 2 ;;
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

CACHE_DIR="$HOME/.cache/huggingface/lerobot/${REPO_ID}"
echo "Clearing cache: $CACHE_DIR"
rm -rf "$CACHE_DIR"

exec lerobot-rollout \
    --strategy.type=sentry \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    --robot.cameras="{front: {type: opencv, index_or_path: '$CAMERA_INDEX', width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $CAMERA_FPS, fourcc: MJPG, rotation: 180}, wrist: {type: intelrealsense, use_depth: true, width: 640, height: 480, fps: 30, serial_number_or_name: $REALSENSE_SERIAL}}" \
    --display_data="$DISPLAY_DATA" \
    --dataset.single_task="$TASK" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --play_sounds="$PLAY_SOUNDS" \
    --policy.path="$POLICY_PATH" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --policy.temporal_ensemble_coeff="$TEMPORAL_ENSEMBLE_COEFF" \
    --dataset.repo_id="$REPO_ID" \
    "${EXTRA_ARGS[@]}"
