"""Modular, resumable evaluation pipeline CLI.

Run the whole thing, or one stage at a time so you can isolate a single model
(e.g. translate all directions with one interpreter, then stop):

    # Everything, chained in a run directory:
    uv run python scripts/run_pipeline.py all \
        --data data/enriched/id_kor_maps.jsonl --run-dir outputs/pipeline/id_kor

    # One stage at a time (artifacts are plain JSONL you can inspect/edit):
    uv run python scripts/run_pipeline.py prepare   --data data/enriched/id_kor_maps.jsonl --output out/00_units.jsonl
    uv run python scripts/run_pipeline.py translate  --input out/00_units.jsonl --output out/01_translated.jsonl
    uv run python scripts/run_pipeline.py respond    --input out/01_translated.jsonl --output out/02_responded.jsonl
    uv run python scripts/run_pipeline.py verify      --input out/02_responded.jsonl --output out/03_verified.jsonl
    uv run python scripts/run_pipeline.py judge       --input out/03_verified.jsonl --output out/04_judged.jsonl
    uv run python scripts/run_pipeline.py consolidate --input out/04_judged.jsonl --output out/results.jsonl

Every stage is resumable: re-running skips records already in the output file.
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

# Ensure src is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_ROOT, "src"))

from dotenv import load_dotenv

from interpreter_agent_eval.pipeline import registry, stages

load_dotenv()

_PROVIDERS = ["gemini", "openrouter", "openai"]
_THINKING = ["none", "minimal", "low", "medium", "high"]


def _add_batch_args(sp):
    """Common batch-backend flags for the translate/judge subcommands."""
    sp.add_argument("--backend", default="sync", choices=["sync", "batch"])
    sp.add_argument(
        "--no-batch-wait",
        dest="batch_wait",
        action="store_false",
        help="Submit the batch and exit; collect later with batch-collect",
    )
    sp.add_argument("--poll-interval", type=float, default=30.0)
    sp.add_argument("--batch-timeout", type=float, default=None)
    sp.add_argument(
        "--api-key-env",
        default=None,
        help="Name of an env var holding the provider API key to use for this "
        "batch job (e.g. GEMINI_API_KEY_3), to spread jobs across multiple "
        "keys/quotas. Defaults to the provider's usual env var.",
    )
    sp.set_defaults(batch_wait=True)


def _resolve_api_key(args) -> "str | None":
    key_env = getattr(args, "api_key_env", None)
    if not key_env:
        return None
    value = os.environ.get(key_env)
    if not value:
        print(f"Warning: env var {key_env} not set or empty; using provider default.")
        return None
    return value


def _resolve_data_paths(paths):
    resolved = []
    for p in paths:
        if not os.path.isabs(p):
            p = os.path.join(_ROOT, "data", "enriched", p)
        if os.path.exists(p):
            resolved.append(p)
        else:
            print(f"File not found: {p}")
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    sp = sub.add_parser("prepare", help="Flatten enriched data files into work units")
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--num-samples", type=int, default=None)
    sp.add_argument("--filter-target-lang", default=None)

    # translate
    sp = sub.add_parser("translate", help="Interpreter translates source_text")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument(
        "--provider",
        default=registry.DEFAULT_INTERPRETER_PROVIDER,
        choices=_PROVIDERS + ["nllb", "seamless", "aya", "google", "papago"],
        help="LLM providers, local GPU NMT (nllb/seamless), local chat LLM (aya), "
        "or free web MT (google/papago)",
    )
    sp.add_argument(
        "--model",
        default=registry.DEFAULT_INTERPRETER_MODEL,
        help="LLM model id, or HF checkpoint for nllb/seamless (default: largest)",
    )
    sp.add_argument("--thinking-level", default="minimal", choices=_THINKING)
    sp.add_argument("--concurrency", type=int, default=4)
    sp.add_argument(
        "--condition",
        default="cultural_context",
        choices=["cultural_context", "direct_no_context", "direct_context", "spec_aware"],
        help="Translation-brief ablation: cultural_context (default, production brief), "
        "direct_no_context, direct_context, or spec_aware",
    )
    _add_batch_args(sp)

    # respond
    sp = sub.add_parser(
        "respond", help="User B (local model) responds to the translation"
    )
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--concurrency", type=int, default=1)

    # verify
    sp = sub.add_parser("verify", help="GlotLID language verification")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--min-confidence", type=float, default=0.8)

    # judge
    sp = sub.add_parser("judge", help="LLM-as-judge evaluation")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument(
        "--provider", default=registry.DEFAULT_JUDGE_PROVIDER, choices=_PROVIDERS
    )
    sp.add_argument("--model", default=registry.DEFAULT_JUDGE_MODEL)
    sp.add_argument(
        "--thinking-level",
        default=registry.DEFAULT_JUDGE_THINKING_LEVEL,
        choices=_THINKING,
    )
    sp.add_argument("--concurrency", type=int, default=4)
    _add_batch_args(sp)

    # batch-collect — poll + collect a previously submitted batch (via sidecar)
    sp = sub.add_parser(
        "batch-collect",
        help="Collect a batch submitted earlier with --backend batch --no-batch-wait",
    )
    sp.add_argument("--stage", required=True, choices=["translate", "judge"])
    sp.add_argument("--input", required=True, help="The stage's input JSONL")
    sp.add_argument(
        "--output",
        required=True,
        help="The stage's output JSONL (+ .batch.json sidecar)",
    )
    sp.add_argument("--provider", required=True, choices=["gemini", "openai"])
    sp.add_argument("--model", default=None, help="Optional; only used if resubmitting")
    sp.add_argument("--poll-interval", type=float, default=30.0)
    sp.add_argument("--batch-timeout", type=float, default=None)
    sp.add_argument(
        "--api-key-env",
        default=None,
        help="Env var holding the API key the batch job was originally submitted "
        "under (must match, or polling/collecting will fail).",
    )

    # consolidate
    sp = sub.add_parser("consolidate", help="Emit the final legacy-schema results file")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)

    # all
    sp = sub.add_parser("all", help="Run every stage sequentially in a run directory")
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument(
        "--run-dir",
        default=None,
        help="Directory for stage artifacts (default: outputs/pipeline/<basename>_<ts>)",
    )
    sp.add_argument(
        "--final-output", default=None, help="Path for the consolidated results.jsonl"
    )
    sp.add_argument("--num-samples", type=int, default=None)
    sp.add_argument("--filter-target-lang", default=None)
    sp.add_argument(
        "--interpreter-provider",
        default=registry.DEFAULT_INTERPRETER_PROVIDER,
        choices=_PROVIDERS,
    )
    sp.add_argument("--interpreter-model", default=registry.DEFAULT_INTERPRETER_MODEL)
    sp.add_argument(
        "--interpreter-thinking-level", default="minimal", choices=_THINKING
    )
    sp.add_argument(
        "--judge-provider", default=registry.DEFAULT_JUDGE_PROVIDER, choices=_PROVIDERS
    )
    sp.add_argument("--judge-model", default=registry.DEFAULT_JUDGE_MODEL)
    sp.add_argument(
        "--judge-thinking-level",
        default=registry.DEFAULT_JUDGE_THINKING_LEVEL,
        choices=_THINKING,
    )
    sp.add_argument("--translate-concurrency", type=int, default=4)
    sp.add_argument("--respond-concurrency", type=int, default=1)
    sp.add_argument("--judge-concurrency", type=int, default=4)
    sp.add_argument("--translate-backend", default="sync", choices=["sync", "batch"])
    sp.add_argument("--judge-backend", default="sync", choices=["sync", "batch"])
    sp.add_argument("--poll-interval", type=float, default=30.0)
    sp.add_argument("--min-confidence", type=float, default=0.8)

    args = parser.parse_args()

    if args.command == "prepare":
        stages.run_prepare(
            _resolve_data_paths(args.data),
            args.output,
            args.num_samples,
            args.filter_target_lang,
        )
    elif args.command == "translate":
        stages.run_translate(
            args.input,
            args.output,
            args.provider,
            args.model,
            args.thinking_level,
            args.concurrency,
            backend=args.backend,
            batch_wait=args.batch_wait,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            condition=args.condition,
            api_key=_resolve_api_key(args),
        )
    elif args.command == "respond":
        stages.run_respond(args.input, args.output, args.concurrency)
    elif args.command == "verify":
        stages.run_verify(args.input, args.output, args.min_confidence)
    elif args.command == "judge":
        stages.run_judge(
            args.input,
            args.output,
            args.provider,
            args.model,
            args.thinking_level,
            args.concurrency,
            backend=args.backend,
            batch_wait=args.batch_wait,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=_resolve_api_key(args),
        )
    elif args.command == "batch-collect":
        runner = stages.run_translate if args.stage == "translate" else stages.run_judge
        # Re-invoking the stage in batch mode detects the sidecar and collects.
        runner(
            args.input,
            args.output,
            args.provider,
            args.model or registry.DEFAULT_JUDGE_MODEL,
            backend="batch",
            batch_wait=True,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=_resolve_api_key(args),
        )
    elif args.command == "consolidate":
        stages.run_consolidate(args.input, args.output)
    elif args.command == "all":
        data_paths = _resolve_data_paths(args.data)
        if not data_paths:
            print("No valid data files; nothing to do.")
            return
        run_dir = args.run_dir
        if run_dir is None:
            base = os.path.basename(data_paths[0]).replace(".jsonl", "")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(_ROOT, "outputs", "pipeline", f"{base}_{ts}")
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
            translate_concurrency=args.translate_concurrency,
            respond_concurrency=args.respond_concurrency,
            judge_concurrency=args.judge_concurrency,
            translate_backend=args.translate_backend,
            judge_backend=args.judge_backend,
            poll_interval=args.poll_interval,
            min_confidence=args.min_confidence,
            final_output=args.final_output,
        )


if __name__ == "__main__":
    main()
