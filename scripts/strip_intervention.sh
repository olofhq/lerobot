#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <dataset-name>"
    echo "Example: $0 rollout_hil"
    exit 1
fi

DATASET_ROOT="$HOME/.cache/huggingface/lerobot/tee_usbc/$1"

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Error: dataset not found at $DATASET_ROOT"
    exit 1
fi

python3 -c "
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('$DATASET_ROOT')
info_path = root / 'meta/info.json'
info = json.loads(info_path.read_text())
if 'intervention' in info['features']:
    del info['features']['intervention']
    info_path.write_text(json.dumps(info, indent=4))
    print('Removed intervention from info.json')
else:
    print('intervention not in info.json — nothing to do')

for pf in sorted(root.glob('data/**/*.parquet')):
    table = pq.read_table(pf)
    if 'intervention' in table.column_names:
        table = table.drop(['intervention'])
        pq.write_table(table, pf)
        print(f'Dropped intervention column from {pf.name}')
"
