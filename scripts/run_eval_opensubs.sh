#!/usr/bin/env bash
# Evaluate all 5 language pairs, grouped by target language (User B).
# Pairs sharing the same target language run in PARALLEL — they all hit the same LLM.
# Bengali (ben) target is skipped — no Bengali user model available.
#
# Run from WSL:
#   bash /mnt/d/dev/mt-eval/interpreter-agent-eval/scripts/run_eval_opensubs.sh
#
# Optional env overrides:
#   NUM_SAMPLES   max records per (pair × target_lang) to evaluate (default: 100)
#   TARGET_LANGS  space-separated subset to run, e.g. "arb kor" (default: all three)

set -euo pipefail

ROOT="/mnt/d/dev/mt-eval/interpreter-agent-eval"
PYTHON="${UV:-uv} run python"
EVAL_SCRIPT="$ROOT/scripts/run_eval.py"
AUG_BASE="$ROOT/outputs/opensubtitles_augmented"
EVAL_OUT_BASE="$ROOT/outputs/opensubs_eval"

NUM_SAMPLES="${NUM_SAMPLES:-100}"
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
    local input="$AUG_BASE/$pair/augmented.jsonl"
    local output="$EVAL_OUT_BASE/$pair/${target_lang}_results.jsonl"

    if [[ ! -f "$input" ]]; then
        echo "[SKIP] $pair/$target_lang — augmented.jsonl not found"
        return 0
    fi

    mkdir -p "$EVAL_OUT_BASE/$pair"
    echo "[$(date +%H:%M:%S)] Start $pair → target=$target_lang (out: $output)"
    $PYTHON "$EVAL_SCRIPT" \
        --data "$input" \
        --filter-target-lang "$target_lang" \
        --num_samples "$NUM_SAMPLES" \
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
    echo " Target language group: $target_lang  (NUM_SAMPLES=$NUM_SAMPLES)"
    echo "============================================================"

    case "$target_lang" in
        arb)
            # Arabic User B: ar-bn (ben→arb), ar-id (ind→arb), ar-ko (kor→arb)
            eval_pair ar-bn arb &
            eval_pair ar-id arb &
            eval_pair ar-ko arb &
            ;;
        kor)
            # Korean User B: ar-ko (arb→kor), bn-ko (ben→kor)
            eval_pair ar-ko kor &
            eval_pair bn-ko kor &
            ;;
        ind)
            # Indonesian User B: ar-id (arb→ind), bn-id (ben→ind)
            eval_pair ar-id ind &
            eval_pair bn-id ind &
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
