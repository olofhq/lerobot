#!/usr/bin/env bash
# Delete episodes from a LeRobot dataset, fixing timestamp/length metadata
# mismatches first if needed.
#
# Usage: ./scripts/delete_episodes.sh <repo_id> <episode_indices>
# Example: ./scripts/delete_episodes.sh tee_usbc/v1 "[18]"
#          ./scripts/delete_episodes.sh tee_usbc/v1 "[5,8,12]"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <repo_id> <episode_indices>"
    echo "  e.g. $0 tee_usbc/v1 \"[18]\""
    echo "  e.g. $0 tee_usbc/v1 \"[5,8,12]\""
    exit 1
fi

REPO_ID="$1"
EPISODE_INDICES="$2"

echo "=== Fixing any timestamp/length mismatches in $REPO_ID ==="
"$VENV_PYTHON" -c "
import pandas as pd
from pathlib import Path

root = Path.home() / '.cache/huggingface/lerobot' / '$REPO_ID'
if not root.exists():
    print(f'Dataset not found at {root}')
    exit(1)

import json
with open(root / 'meta/info.json') as f:
    info = json.load(f)
fps = info['fps']

eps_dir = root / 'meta/episodes'
fixed = False

for parquet_path in sorted(eps_dir.rglob('*.parquet')):
    df = pd.read_parquet(parquet_path)

    # Find video timestamp columns
    video_keys = set()
    for col in df.columns:
        if col.startswith('videos/') and col.endswith('/from_timestamp'):
            video_keys.add(col.replace('/from_timestamp', '').replace('videos/', ''))

    for video_key in sorted(video_keys):
        prefix = f'videos/{video_key}'
        for idx in df.index:
            row = df.loc[idx]
            from_ts = row[f'{prefix}/from_timestamp']
            to_ts = row[f'{prefix}/to_timestamp']
            length = row['length']
            frame_range = round(to_ts * fps) - round(from_ts * fps)

            if length != frame_range:
                correct_to_ts = from_ts + length / fps
                shift = correct_to_ts - to_ts
                ep_idx = row['episode_index']
                print(f'  Fixing {video_key} ep {ep_idx}: {frame_range} frames -> {length} frames (shift {shift:.6f}s)')

                df.at[idx, f'{prefix}/to_timestamp'] = correct_to_ts
                file_idx = row[f'{prefix}/file_index']

                # Shift subsequent episodes in the same file
                for j in range(idx + 1, len(df)):
                    if df.loc[j, f'{prefix}/file_index'] == file_idx:
                        df.at[j, f'{prefix}/from_timestamp'] += shift
                        df.at[j, f'{prefix}/to_timestamp'] += shift

                fixed = True

    if fixed:
        df.to_parquet(parquet_path)

if fixed:
    print('Metadata fixed.')
else:
    print('No mismatches found.')
"

echo ""
echo "=== Deleting episodes $EPISODE_INDICES from $REPO_ID ==="
"$REPO_ROOT/.venv/bin/lerobot-edit-dataset" \
    --operation.type delete_episodes \
    --repo_id "$REPO_ID" \
    --operation.episode_indices "$EPISODE_INDICES"
