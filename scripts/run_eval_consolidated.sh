#!/usr/bin/env bash
# Evaluate all 6 language pairs using the self-consistency consolidated data.
# Pairs sharing the same target language (User B) run in parallel.
#
# Bengali target (ben) is excluded until a Bengali LM Studio model is set via
# LM_STUDIO_BN_MODEL — add "ben" to TARGET_LANGS once that is configured.
#
# Run from WSL:
#   bash /mnt/d/dev/mt-eval/interpreter-agent-eval/scripts/run_eval_consolidated.sh
#
# Optional env overrides:
#   NUM_SAMPLES   max records per (pair × target_lang) to evaluate (default: all)
#   TARGET_LANGS  space-separated subset to run, e.g. "arb kor" (default: arb kor ind)

set -euo pipefail

ROOT="/mnt/d/dev/mt-eval/interpreter-agent-eval"
PYTHON="${UV:-/home/haznitrama/.local/bin/uv} run python"
EVAL_SCRIPT="$ROOT/scripts/run_eval.py"
DATA_BASE="$ROOT/outputs/opensubs_augmented/top500"
EVAL_OUT_BASE="$ROOT/outputs/eval_consolidated"

NUM_SAMPLES="${NUM_SAMPLES:-}"
TARGET_LANGS="${TARGET_LANGS:-arb kor ind}"

# Interpreter agent
INTERP_PROVIDER="${INTERP_PROVIDER:-gemini}"
INTERP_MODEL="${INTERP_MODEL:-gemini-3.1-flash-lite-preview}"

# LLM judge
JUDGE_PROVIDER="${JUDGE_PROVIDER:-gemini}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-pro-preview}"
JUDGE_THINKING="${JUDGE_THINKING:-high}"

cd "$ROOT"

eval_pair() {
    local pair="$1"
    local target_lang="$2"
    local input="$DATA_BASE/$pair/consolidated.jsonl"
    local output="$EVAL_OUT_BASE/$pair/${target_lang}_results.jsonl"

    if [[ ! -f "$input" ]]; then
        echo "[SKIP] $pair/$target_lang — consolidated.jsonl not found"
        return 0
    fi

    mkdir -p "$EVAL_OUT_BASE/$pair"
    echo "[$(date +%H:%M:%S)] Start $pair → target=$target_lang (out: $output)"

    local samples_arg=""
    [[ -n "$NUM_SAMPLES" ]] && samples_arg="--num_samples $NUM_SAMPLES"

    $PYTHON "$EVAL_SCRIPT" \
        --data "$input" \
        --filter-target-lang "$target_lang" \
        $samples_arg \
        --resume "$output" \
        --interpreter-provider "$INTERP_PROVIDER" \
        --interpreter-model   "$INTERP_MODEL" \
        --judge-provider      "$JUDGE_PROVIDER" \
        --judge-model         "$JUDGE_MODEL" \
        --judge-thinking-level "$JUDGE_THINKING"

    echo "[$(date +%H:%M:%S)] Done  $pair → target=$target_lang"
}

for target_lang in $TARGET_LANGS; do
    echo ""
    echo "============================================================"
    echo " Target language group: $target_lang  (NUM_SAMPLES=${NUM_SAMPLES:-all})"
    echo "============================================================"

    case "$target_lang" in
        arb)
            # Arabic User B: ar-bn (ben→arb), ar-id (ind→arb), ar-ko (kor→arb)
            eval_pair ar-bn arb &
            eval_pair ar-id arb &
            eval_pair ar-ko arb &
            ;;
        kor)
            # Korean User B: ar-ko (arb→kor), bn-ko (ben→kor), id-ko (ind→kor)
            eval_pair ar-ko kor &
            eval_pair bn-ko kor &
            eval_pair id-ko kor &
            ;;
        ind)
            # Indonesian User B: ar-id (arb→ind), bn-id (ben→ind), id-ko (kor→ind)
            eval_pair ar-id ind &
            eval_pair bn-id ind &
            eval_pair id-ko ind &
            ;;
        ben)
            # Bengali User B: ar-bn (arb→ben), bn-id (ben source is User A here, skip),
            # bn-ko (ben source is User A here, skip)
            # Only directions with Bengali as TARGET:
            eval_pair ar-bn ben &
            eval_pair bn-id ben &
            eval_pair bn-ko ben &
            ;;
        *)
            echo "[WARN] Unknown target_lang '$target_lang' — skipping"
            ;;
    esac

    wait
    echo "Group '$target_lang' done."
done

echo ""
echo "All groups done."
