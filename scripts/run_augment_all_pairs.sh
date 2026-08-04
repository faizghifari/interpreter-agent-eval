#!/usr/bin/env bash
# Augment windows JSONL into MAPS-like data for all 5 language pairs in parallel.
# Reads from outputs/opensubtitles_windows/{pair}/windows.jsonl
# Writes to outputs/opensubtitles_augmented/{pair}/augmented.jsonl
# Run from WSL: bash /mnt/d/dev/mt-eval/interpreter-agent-eval/scripts/run_augment_all_pairs.sh

set -euo pipefail

PYTHON="python3"
SCRIPT="/mnt/d/dev/mt-eval/interpreter-agent-eval/scripts/augment_opensubs_maps.py"
WIN_BASE="/mnt/d/dev/mt-eval/interpreter-agent-eval/outputs/opensubtitles_windows"
OUT_BASE="/mnt/d/dev/mt-eval/interpreter-agent-eval/outputs/opensubtitles_augmented"
MODEL="gemini-3.1-pro-preview"

augment_pair() {
    local pair="$1"
    echo "[$(date +%H:%M:%S)] Starting: $pair"
    mkdir -p "$OUT_BASE/$pair"
    "$PYTHON" "$SCRIPT" \
        --input   "$WIN_BASE/$pair/windows.jsonl" \
        --output  "$OUT_BASE/$pair/augmented.jsonl" \
        --lang-pair "$pair" \
        --model "$MODEL" \
        --append \
        --sleep-s 0.2 \
        --temperature 0.2 \
        --max-output-tokens 32768 \
        --max-retries 4 \
        --retry-backoff-s 3.0
    echo "[$(date +%H:%M:%S)] Done: $pair"
}

# Launch all pairs in parallel
augment_pair ar-bn &
augment_pair ar-id &
augment_pair ar-ko &
augment_pair bn-id &
augment_pair bn-ko &

wait
echo ""
echo "All pairs augmented."
