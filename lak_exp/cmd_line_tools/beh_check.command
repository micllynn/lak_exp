#!/bin/bash
# Double-clickable macOS launcher for beh_check.py
# Opens in Terminal and prompts for input

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate lak_exp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -rp "Enter path to behaviour folder: " FNAME
read -rp "Noisy lick signal? (y/n): " NOISE_FLAG

if [[ "$NOISE_FLAG" == 'y' || "$NOISE_FLAG" == 'Y' ]]; then
    python3 "$SCRIPT_DIR/beh_check.py" "$FNAME" -n
else
    python3 "$SCRIPT_DIR/beh_check.py" "$FNAME"
fi

read -rp "Press Enter to close..."
