#!/bin/bash
# run python script to check behavioral data
# This script will use the Python from your currently activated conda environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FNAME=$1
FLAG=${2:-}

if [[ "$FLAG" == '-n' ]]; then
    python3 "$SCRIPT_DIR/beh_check.py" "$FNAME" -n
else
    python3 "$SCRIPT_DIR/beh_check.py" "$FNAME"
fi
