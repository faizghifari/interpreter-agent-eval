"""Backward-compatible entry point — now a thin wrapper over the staged pipeline.

Historically this ran translate -> respond -> verify -> judge inline, per sample.
That logic now lives in ``interpreter_agent_eval.pipeline`` as independent,
resumable stages. This script preserves the original CLI surface and the final
output schema, but executes the modular pipeline underneath (sequentially,
synchronous backend). For per-stage control or batch APIs, use
``scripts/run_pipeline.py``.
"""

import argparse
import io
import os
import sys
from datetime import datetime

# Force UTF-8 stdout/stderr (Korean/Arabic/Indonesian output on Windows).
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_ROOT, "src"))

from dotenv import load_dotenv

from interpreter_agent_eval.pipeline import registry, stages
from interpreter_agent_eval.utils.language_verification import load_glotlid_model

load_dotenv()

# Re-exported for callers/tests that imported these defaults from run_eval.
DEFAULT_INTERPRETER_PROVIDER = registry.DEFAULT_INTERPRETER_PROVIDER
DEFAULT_INTERPRETER_MODEL = registry.DEFAULT_INTERPRETER_MODEL
DEFAULT_JUDGE_PROVIDER = registry.DEFAULT_JUDGE_PROVIDER
DEFAULT_JUDGE_MODEL = registry.DEFAULT_JUDGE_MODEL
DEFAULT_JUDGE_THINKING_LEVEL = registry.DEFAULT_JUDGE_THINKING_LEVEL


def main():
    default_data = os.path.join(_ROOT, "data", "enriched", "id_kor_maps.jsonl")

    parser = argparse.ArgumentParser(description="Run interpreter agent evaluation")
    parser.add_argument(
        "--data",
        nargs="+",
        default=[default_data],
        help="Path(s) to enriched .jsonl data file(s)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run per file (default: all)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a results .jsonl to resume into (reuses its stage dir)",
    )
    parser.add_argument(
        "--filter-target-lang",
        dest="filter_target_lang",
        default=None,
        help="Only evaluate records with this target_language_code",
    )
    # Interpreter
    parser.add_argument(
        "--interpreter-provider",
        dest="interpreter_provider",
        default=DEFAULT_INTERPRETER_PROVIDER,
        choices=["gemini", "openrouter", "openai"],
    )
    parser.add_argument(
        "--interpreter-model",
        dest="interpreter_model",
        default=DEFAULT_INTERPRETER_MODEL,
    )
    parser.add_argument(
        "--interpreter-thinking-level",
        dest="interpreter_thinking_level",
        default="minimal",
        choices=["none", "minimal", "low", "medium", "high"],
    )
    # Judge
    parser.add_argument(
        "--judge-provider",
        dest="judge_provider",
        default=DEFAULT_JUDGE_PROVIDER,
        choices=["gemini", "openrouter", "openai"],
    )
    parser.add_argument(
        "--judge-model", dest="judge_model", default=DEFAULT_JUDGE_MODEL
    )
    parser.add_argument(
        "--judge-thinking-level",
        dest="judge_thinking_level",
        default=DEFAULT_JUDGE_THINKING_LEVEL,
        choices=["none", "minimal", "low", "medium", "high"],
    )
    args = parser.parse_args()

    # Resolve data paths (bare filenames resolve under data/enriched).
    data_paths = []
    for p in args.data:
        if not os.path.isabs(p):
            p = os.path.join(_ROOT, "data", "enriched", p)
        if os.path.exists(p):
            data_paths.append(p)
        else:
            print(f"File not found: {p}")
    if not data_paths:
        print("No valid data files; nothing to do.")
        return

    # Final output + stage dir. --resume reuses the same locations so stages resume.
    if args.resume:
        final_output = args.resume
        run_dir = os.path.splitext(args.resume)[0] + "_stages"
    else:
        base = os.path.basename(data_paths[0]).replace(".jsonl", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = os.path.join(_ROOT, "outputs", f"eval_{base}_{ts}.jsonl")
        run_dir = os.path.join(_ROOT, "outputs", "pipeline", f"{base}_{ts}")

    # Load GlotLID once for the verify stage.
    print("\n" + "=" * 50)
    print("Initializing Language Verification")
    print("=" * 50)
    glotlid_model = load_glotlid_model()
    print(
        "[OK] GlotLID loaded"
        if glotlid_model
        else "[!] GlotLID unavailable — verification disabled"
    )

    interpreter_label = registry.label_for(
        args.interpreter_provider, args.interpreter_model
    )
    judge_label = registry.label_for(args.judge_provider, args.judge_model)
    print(
        f"\nInterpreter: {interpreter_label} (thinking={args.interpreter_thinking_level})"
    )
    print(f"Judge:       {judge_label} (thinking={args.judge_thinking_level})")
    print(f"Stage dir:   {run_dir}")

    stages.run_all(
        data_files=data_paths,
        run_dir=run_dir,
        num_samples=args.num_samples,
        filter_target_lang=args.filter_target_lang,
        interpreter_provider_type=args.interpreter_provider,
        interpreter_model=args.interpreter_model,
        interpreter_thinking=args.interpreter_thinking_level,
        judge_provider_type=args.judge_provider,
        judge_model=args.judge_model,
        judge_thinking=args.judge_thinking_level,
        min_confidence=0.8,
        glotlid_model=glotlid_model,
        final_output=final_output,
    )


if __name__ == "__main__":
    main()
