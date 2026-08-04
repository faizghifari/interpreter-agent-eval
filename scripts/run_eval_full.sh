#!/usr/bin/env bash
# Full eval run: 2 interpreter models × all directions targeting kor + ind.
# All pairs across both target-language groups run in parallel (kor uses exaone,
# ind uses qwen-sea-lion — different LM Studio models, so no conflict).
# Within each pair, interpreter models run sequentially to avoid LM Studio overload.
#
# Run from WSL:
#   bash /mnt/d/dev/mt-eval/interpreter-agent-eval/scripts/run_eval_full.sh
#
# Optional env overrides:
#   NUM_SAMPLES   max records per run (default: all)
#   TARGET_LANGS  space-separated subset, e.g. "kor" (default: kor ind)

set -euo pipefail

ROOT="/mnt/d/dev/mt-eval/interpreter-agent-eval"
PYTHON="/home/haznitrama/.local/bin/uv run python"
EVAL="$ROOT/scripts/run_eval.py"
DATA="$ROOT/outputs/opensubs_augmented/top500"
OUT="$ROOT/outputs/eval_consolidated"

NUM_SAMPLES="${NUM_SAMPLES:-}"
TARGET_LANGS="${TARGET_LANGS:-kor ind}"

JUDGE_PROVIDER="gemini"
JUDGE_MODEL="gemini-3.1-pro-preview"
JUDGE_THINKING="high"

cd "$ROOT"

# Run all interpreter models sequentially for one (pair, target_lang) combination.
# model_order: "qwen_first" (default) or "gemini_first"
run_pair() {
    local pair="$1"
    local target_lang="$2"
    local model_order="${3:-qwen_first}"
    local input="$DATA/$pair/consolidated.jsonl"

    local samples_arg=""
    [[ -n "$NUM_SAMPLES" ]] && samples_arg="--num_samples $NUM_SAMPLES"

    local models_qwen_first=(
        "openrouter|qwen/qwen3.5-flash-02-23|qwen3.5-flash"
        "gemini|gemini-3.1-flash-lite-preview|gemini-flash-lite"
    )
    local models_gemini_first=(
        "gemini|gemini-3.1-flash-lite-preview|gemini-flash-lite"
        "openrouter|qwen/qwen3.5-flash-02-23|qwen3.5-flash"
    )

    local models=("${models_qwen_first[@]}")
    [[ "$model_order" == "gemini_first" ]] && models=("${models_gemini_first[@]}")

    for prov_model in "${models[@]}"; do
        local prov="${prov_model%%|*}"
        local rest="${prov_model#*|}"
        local model="${rest%%|*}"
        local slug="${rest#*|}"
        local output="$OUT/$pair/${slug}_${target_lang}.jsonl"

        mkdir -p "$OUT/$pair"
        echo "[$(date +%H:%M:%S)] $pair target=$target_lang  interpreter=$prov:$model"
        $PYTHON "$EVAL" \
            --data "$input" \
            --filter-target-lang "$target_lang" \
            $samples_arg \
            --resume "$output" \
            --interpreter-provider "$prov" \
            --interpreter-model   "$model" \
            --judge-provider      "$JUDGE_PROVIDER" \
            --judge-model         "$JUDGE_MODEL" \
            --judge-thinking-level "$JUDGE_THINKING"
        echo "[$(date +%H:%M:%S)] done  $pair target=$target_lang  interpreter=$prov:$model"
    done
}

echo ""
echo "============================================================"
echo " Launching all pairs  (NUM_SAMPLES=${NUM_SAMPLES:-all})"
echo "============================================================"

for target_lang in $TARGET_LANGS; do
    case "$target_lang" in
        kor)
            run_pair id-ko kor &
            run_pair ar-ko kor &
            run_pair bn-ko kor &
            ;;
        ind)
            run_pair ar-id ind &
            run_pair bn-id ind &
            run_pair id-ko ind &
            ;;
        arb)
            run_pair ar-bn arb gemini_first &
            run_pair ar-id arb gemini_first &
            run_pair ar-ko arb gemini_first &
            ;;
        ben)
            run_pair ar-bn ben gemini_first &
            run_pair bn-id ben gemini_first &
            run_pair bn-ko ben gemini_first &
            ;;
        *)
            echo "[WARN] Unknown target_lang '$target_lang' — skipping"
            ;;
    esac
done

wait
echo ""
echo "All pairs done."
