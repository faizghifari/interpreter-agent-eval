"""Modular, resumable multi-turn evaluation pipeline CLI (docs/multiturn_expansion_plan.md).

Mirrors ``scripts/run_pipeline.py``'s shape and flags. Scripted mode flattens
pre-authored scenarios and translates them wave-by-wave; dynamic mode
live-simulates both users and converses turn-by-turn. Both converge on a
shared verify -> judge-turns -> judge-conv -> consolidate tail.

    # Scripted, everything chained in a run directory:
    uv run python scripts/run_multiturn_pipeline.py all --mode scripted \
        --data id_ko_mts_smoke.jsonl --run-dir outputs/multiturn/smoke_scripted

    # Dynamic, everything chained:
    uv run python scripts/run_multiturn_pipeline.py all --mode dynamic \
        --data id_ko_mtd_smoke.jsonl --run-dir outputs/multiturn/smoke_dynamic

    # One stage at a time (artifacts are plain JSONL you can inspect/edit):
    uv run python scripts/run_multiturn_pipeline.py prepare --data id_ko_mts_smoke.jsonl --output out/00_units.jsonl
    uv run python scripts/run_multiturn_pipeline.py translate --input out/00_units.jsonl --output out/01_translated.jsonl
    uv run python scripts/run_multiturn_pipeline.py verify --input out/01_translated.jsonl --output out/03_verified.jsonl
    uv run python scripts/run_multiturn_pipeline.py judge-turns --input out/03_verified.jsonl --output out/04_turn_judged.jsonl
    uv run python scripts/run_multiturn_pipeline.py judge-conv --data id_ko_mts_smoke.jsonl \
        --turn-judged out/04_turn_judged.jsonl --output out/05_conv_judged.jsonl
    uv run python scripts/run_multiturn_pipeline.py consolidate \
        --conv-judged out/05_conv_judged.jsonl --turn-judged out/04_turn_judged.jsonl \
        --output out/results.jsonl --turns-output out/results_turns.jsonl

Every stage is resumable: re-running skips records already in the output file
(scripted translate resumes wave-by-wave; dynamic converse resumes each
conversation from its first missing turn).
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

from interpreter_agent_eval.pipeline import registry
from interpreter_agent_eval.pipeline.multiturn import stages
from interpreter_agent_eval.providers.google_ai import get_usage_totals, reset_usage_totals

load_dotenv()

_PROVIDERS = ["gemini", "openrouter", "openai"]
_THINKING = ["none", "minimal", "low", "medium", "high"]

# Canonical artifact names within a run directory (plan's Stage flow table).
UNITS = "00_units.jsonl"
TRANSLATED = "01_translated.jsonl"
RESPONDED = "02_responded.jsonl"
CONVERSED = "01_conversed.jsonl"
CHECKLISTED = "02_checklisted.jsonl"  # dynamic + --defer-checklist only
VERIFIED = "03_verified.jsonl"
TURN_JUDGED = "04_turn_judged.jsonl"
CONV_JUDGED = "05_conv_judged.jsonl"
RESULTS = "results.jsonl"
RESULTS_TURNS = "results_turns.jsonl"


def _resolve_data_paths(paths):
    resolved = []
    for p in paths:
        if not os.path.isabs(p):
            candidate = os.path.join(_ROOT, "data", "enriched", "multiturn", p)
            p = candidate if os.path.exists(candidate) else os.path.join(_ROOT, p)
        if os.path.exists(p):
            resolved.append(p)
        else:
            print(f"File not found: {p}")
    return resolved


def _add_batch_args(sp):
    """Common batch-backend flags for the translate/judge-turns/judge-conv subcommands."""
    sp.add_argument("--backend", default="sync", choices=["sync", "batch"])
    sp.add_argument(
        "--no-batch-wait",
        dest="batch_wait",
        action="store_false",
        help="Submit the batch (job, or this wave) and exit; collect later with batch-collect "
        "(translate) or by re-running the same command (judge-turns/judge-conv).",
    )
    sp.add_argument("--poll-interval", type=float, default=30.0)
    sp.add_argument("--batch-timeout", type=float, default=None)
    sp.add_argument(
        "--api-key-env",
        default=None,
        help="Name of an env var holding the provider API key to use for this batch job "
        "(e.g. GEMINI_API_KEY_3), to spread jobs across multiple keys/quotas. Defaults to "
        "the provider's usual env var.",
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


def _add_checklist_args(sp):
    sp.add_argument(
        "--checklist-provider", default=registry.DEFAULT_JUDGE_PROVIDER, choices=_PROVIDERS
    )
    sp.add_argument("--checklist-model", default=registry.DEFAULT_JUDGE_MODEL)
    sp.add_argument(
        "--checklist-thinking-level",
        default=registry.DEFAULT_JUDGE_THINKING_LEVEL,
        choices=_THINKING,
    )
    sp.add_argument(
        "--no-function-grounding",
        action="store_true",
        help="Escape hatch for languages without a function taxonomy: checklists "
        "are still generated but with function_id=null.",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare (scripted only)
    sp = sub.add_parser("prepare", help="Flatten scripted scenario file(s) into turn units")
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--num-conversations", type=int, default=None)

    # translate (scripted only)
    sp = sub.add_parser("translate", help="Wave-loop interpreter translation (scripted)")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument(
        "--provider",
        default=registry.DEFAULT_INTERPRETER_PROVIDER,
        choices=_PROVIDERS + ["nllb", "seamless", "aya", "google", "papago"],
    )
    sp.add_argument("--model", default=registry.DEFAULT_INTERPRETER_MODEL)
    sp.add_argument("--thinking-level", default="minimal", choices=_THINKING)
    sp.add_argument("--concurrency", type=int, default=4)
    sp.add_argument("--context-mode", default="transcript", choices=["transcript", "none"])
    _add_batch_args(sp)

    # respond (scripted only, optional)
    sp = sub.add_parser("respond", help="Listener comprehension probe (scripted, optional)")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--concurrency", type=int, default=1)

    # converse (dynamic only)
    sp = sub.add_parser("converse", help="Live user-sim -> checklist -> translate (dynamic)")
    sp.add_argument("--data", nargs="+", required=True, help="Dynamic seed file(s)")
    sp.add_argument("--output", required=True)
    sp.add_argument("--interpreter-provider", default=registry.DEFAULT_INTERPRETER_PROVIDER, choices=_PROVIDERS)
    sp.add_argument("--interpreter-model", default=registry.DEFAULT_INTERPRETER_MODEL)
    sp.add_argument("--interpreter-thinking-level", default="minimal", choices=_THINKING)
    _add_checklist_args(sp)
    sp.add_argument("--context-mode", default="transcript", choices=["transcript", "none"])
    sp.add_argument("--concurrency", type=int, default=4, help="Conversations run in parallel")
    sp.add_argument(
        "--defer-checklist",
        action="store_true",
        help="Skip inline per-turn checklist-gen; leaves checklist_items empty. Run the "
        "checklist-batch subcommand afterward to fill them in as ONE batch job instead of "
        "N sync calls woven into this live loop.",
    )

    # checklist-batch (dynamic + --defer-checklist only)
    sp = sub.add_parser(
        "checklist-batch", help="Post-hoc per-turn checklist-gen for turns produced with converse --defer-checklist"
    )
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    _add_checklist_args(sp)
    sp.add_argument("--concurrency", type=int, default=4)
    _add_batch_args(sp)
    sp.set_defaults(backend="batch")  # the whole point of deferring is to batch it

    # verify (shared tail)
    sp = sub.add_parser("verify", help="GlotLID language verification")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--min-confidence", type=float, default=0.8)
    sp.add_argument("--concurrency", type=int, default=1)

    # judge-turns (shared tail)
    sp = sub.add_parser("judge-turns", help="Per-turn judge, always transcript-conditioned")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--provider", default=registry.DEFAULT_JUDGE_PROVIDER, choices=_PROVIDERS)
    sp.add_argument("--model", default=registry.DEFAULT_JUDGE_MODEL)
    sp.add_argument("--thinking-level", default=registry.DEFAULT_JUDGE_THINKING_LEVEL, choices=_THINKING)
    sp.add_argument("--concurrency", type=int, default=4)
    sp.add_argument(
        "--judge-history",
        action="store_true",
        help="Experimental (plan D5): feed prior turns' judge verdicts forward. "
        "Default off, off in every funded tier, forces a sync wave loop when on "
        "(incompatible with --backend batch).",
    )
    _add_batch_args(sp)

    # judge-conv (shared tail)
    sp = sub.add_parser("judge-conv", help="One judge call per conversation, over the full transcript")
    sp.add_argument("--data", nargs="+", required=True, help="The scenario/seed file(s) used to generate this run")
    sp.add_argument("--turn-judged", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--judge-provider", default=registry.DEFAULT_JUDGE_PROVIDER, choices=_PROVIDERS)
    sp.add_argument("--judge-model", default=registry.DEFAULT_JUDGE_MODEL)
    sp.add_argument("--judge-thinking-level", default=registry.DEFAULT_JUDGE_THINKING_LEVEL, choices=_THINKING)
    _add_checklist_args(sp)
    sp.add_argument("--concurrency", type=int, default=4)
    _add_batch_args(sp)

    # batch-collect — poll + collect a previously submitted batch (via sidecar)
    sp = sub.add_parser(
        "batch-collect",
        help="Collect a batch submitted earlier with --backend batch --no-batch-wait",
    )
    sp.add_argument("--stage", required=True, choices=["translate", "judge-turns", "judge-conv"])
    sp.add_argument("--input", required=True, help="The stage's input JSONL (or --turn-judged for judge-conv)")
    sp.add_argument(
        "--data", nargs="*", default=None, help="judge-conv only: the scenario/seed file(s)"
    )
    sp.add_argument(
        "--output",
        required=True,
        help="The stage's output JSONL (+ .batch.json / .batch.t{NN}.json sidecar(s))",
    )
    sp.add_argument("--provider", required=True, choices=["gemini", "openai"])
    sp.add_argument("--model", default=None, help="Optional; only used if resubmitting")
    sp.add_argument("--poll-interval", type=float, default=30.0)
    sp.add_argument("--batch-timeout", type=float, default=None)
    sp.add_argument(
        "--api-key-env",
        default=None,
        help="Env var holding the API key the batch job was originally submitted under "
        "(must match, or polling/collecting will fail).",
    )

    # consolidate (shared tail)
    sp = sub.add_parser("consolidate", help="Emit results.jsonl + results_turns.jsonl")
    sp.add_argument("--conv-judged", required=True)
    sp.add_argument("--turn-judged", required=True)
    sp.add_argument("--output", required=True)
    sp.add_argument("--turns-output", required=True)

    # all
    sp = sub.add_parser("all", help="Run the mode-appropriate chain end to end in a run directory")
    sp.add_argument("--mode", required=True, choices=["scripted", "dynamic"])
    sp.add_argument("--data", nargs="+", required=True)
    sp.add_argument("--run-dir", default=None, help="Default: outputs/multiturn/<basename>_<ts>")
    sp.add_argument("--num-conversations", type=int, default=None)
    sp.add_argument("--skip-respond", action="store_true", help="Scripted only: skip the listener probe stage")
    sp.add_argument("--context-mode", default="transcript", choices=["transcript", "none"])
    sp.add_argument(
        "--interpreter-provider",
        default=registry.DEFAULT_INTERPRETER_PROVIDER,
        choices=_PROVIDERS + ["nllb", "seamless", "aya", "google", "papago"],
    )
    sp.add_argument("--interpreter-model", default=registry.DEFAULT_INTERPRETER_MODEL)
    sp.add_argument("--interpreter-thinking-level", default="minimal", choices=_THINKING)
    _add_checklist_args(sp)
    sp.add_argument("--judge-provider", default=registry.DEFAULT_JUDGE_PROVIDER, choices=_PROVIDERS)
    sp.add_argument("--judge-model", default=registry.DEFAULT_JUDGE_MODEL)
    sp.add_argument("--judge-thinking-level", default=registry.DEFAULT_JUDGE_THINKING_LEVEL, choices=_THINKING)
    sp.add_argument("--judge-history", action="store_true")
    sp.add_argument("--translate-concurrency", type=int, default=4)
    sp.add_argument("--respond-concurrency", type=int, default=1)
    sp.add_argument("--converse-concurrency", type=int, default=4)
    sp.add_argument("--verify-concurrency", type=int, default=1)
    sp.add_argument("--judge-turns-concurrency", type=int, default=4)
    sp.add_argument("--judge-conv-concurrency", type=int, default=4)
    sp.add_argument("--min-confidence", type=float, default=0.8)
    sp.add_argument("--translate-backend", default="sync", choices=["sync", "batch"], help="Scripted only")
    sp.add_argument("--judge-backend", default="sync", choices=["sync", "batch"])
    sp.add_argument(
        "--defer-checklist",
        action="store_true",
        help="Dynamic only: skip inline checklist-gen in mt-converse and batch it afterward "
        "as ONE job (plan Step 8 optimization) instead of N sync calls per conversation.",
    )
    sp.add_argument("--poll-interval", type=float, default=30.0)
    sp.add_argument("--batch-timeout", type=float, default=None)
    sp.add_argument("--api-key-env", default=None)

    args = parser.parse_args()
    reset_usage_totals()
    usage_sidecar_target = None

    if args.command == "prepare":
        stages.run_mt_prepare(_resolve_data_paths(args.data), args.output, args.num_conversations)

    elif args.command == "translate":
        stages.run_mt_translate(
            args.input,
            args.output,
            provider_type=args.provider,
            model_name=args.model,
            thinking_level=args.thinking_level,
            concurrency=args.concurrency,
            context_mode=args.context_mode,
            backend=args.backend,
            batch_wait=args.batch_wait,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=_resolve_api_key(args),
        )

    elif args.command == "respond":
        stages.run_mt_respond(args.input, args.output, concurrency=args.concurrency)

    elif args.command == "converse":
        stages.run_mt_converse(
            _resolve_data_paths(args.data),
            args.output,
            interpreter_provider_type=args.interpreter_provider,
            interpreter_model=args.interpreter_model,
            interpreter_thinking=args.interpreter_thinking_level,
            checklist_provider_type=args.checklist_provider,
            checklist_model=args.checklist_model,
            checklist_thinking=args.checklist_thinking_level,
            context_mode=args.context_mode,
            concurrency=args.concurrency,
            no_function_grounding=args.no_function_grounding,
            defer_checklist=args.defer_checklist,
        )

    elif args.command == "checklist-batch":
        stages.run_mt_checklist_batch(
            args.input,
            args.output,
            checklist_provider_type=args.checklist_provider,
            checklist_model=args.checklist_model,
            checklist_thinking=args.checklist_thinking_level,
            concurrency=args.concurrency,
            no_function_grounding=args.no_function_grounding,
            backend=args.backend,
            batch_wait=args.batch_wait,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=_resolve_api_key(args),
        )

    elif args.command == "verify":
        stages.run_mt_verify(
            args.input, args.output, min_confidence=args.min_confidence, concurrency=args.concurrency
        )

    elif args.command == "judge-turns":
        stages.run_mt_judge_turns(
            args.input,
            args.output,
            provider_type=args.provider,
            model_name=args.model,
            thinking_level=args.thinking_level,
            concurrency=args.concurrency,
            judge_history=args.judge_history,
            backend=args.backend,
            batch_wait=args.batch_wait,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=_resolve_api_key(args),
        )

    elif args.command == "judge-conv":
        stages.run_mt_judge_conv(
            _resolve_data_paths(args.data),
            args.turn_judged,
            args.output,
            judge_provider_type=args.judge_provider,
            judge_model=args.judge_model,
            judge_thinking=args.judge_thinking_level,
            checklist_provider_type=args.checklist_provider,
            checklist_model=args.checklist_model,
            checklist_thinking=args.checklist_thinking_level,
            concurrency=args.concurrency,
            no_function_grounding=args.no_function_grounding,
            backend=args.backend,
            batch_wait=args.batch_wait,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=_resolve_api_key(args),
        )

    elif args.command == "batch-collect":
        api_key = _resolve_api_key(args)
        if args.stage == "translate":
            # Collect one wave at a time: re-invoking with backend="batch" resumes
            # whichever wave has a pending sidecar, same as a normal re-run.
            stages.run_mt_translate(
                args.input,
                args.output,
                provider_type=args.provider,
                model_name=args.model or registry.DEFAULT_INTERPRETER_MODEL,
                backend="batch",
                batch_wait=True,
                poll_interval=args.poll_interval,
                batch_timeout=args.batch_timeout,
                api_key=api_key,
            )
        elif args.stage == "judge-turns":
            stages.run_mt_judge_turns(
                args.input,
                args.output,
                provider_type=args.provider,
                model_name=args.model or registry.DEFAULT_JUDGE_MODEL,
                backend="batch",
                batch_wait=True,
                poll_interval=args.poll_interval,
                batch_timeout=args.batch_timeout,
                api_key=api_key,
            )
        else:  # judge-conv
            if not args.data:
                raise SystemExit("--data (scenario/seed files) is required for --stage judge-conv")
            stages.run_mt_judge_conv(
                _resolve_data_paths(args.data),
                args.input,
                args.output,
                judge_provider_type=args.provider,
                judge_model=args.model or registry.DEFAULT_JUDGE_MODEL,
                backend="batch",
                batch_wait=True,
                poll_interval=args.poll_interval,
                batch_timeout=args.batch_timeout,
                api_key=api_key,
            )

    elif args.command == "consolidate":
        stages.run_mt_consolidate(args.conv_judged, args.turn_judged, args.output, args.turns_output)

    elif args.command == "all":
        data_paths = _resolve_data_paths(args.data)
        if not data_paths:
            print("No valid data files; nothing to do.")
            return
        run_dir = args.run_dir
        if run_dir is None:
            base = os.path.basename(data_paths[0]).replace(".jsonl", "")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(_ROOT, "outputs", "multiturn", f"{base}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        usage_sidecar_target = run_dir

        def p(name):
            return os.path.join(run_dir, name)

        api_key = _resolve_api_key(args)

        if args.mode == "scripted":
            stages.run_mt_prepare(data_paths, p(UNITS), args.num_conversations)
            stages.run_mt_translate(
                p(UNITS),
                p(TRANSLATED),
                provider_type=args.interpreter_provider,
                model_name=args.interpreter_model,
                thinking_level=args.interpreter_thinking_level,
                concurrency=args.translate_concurrency,
                context_mode=args.context_mode,
                backend=args.translate_backend,
                poll_interval=args.poll_interval,
                batch_timeout=args.batch_timeout,
                api_key=api_key,
            )
            verify_input = p(TRANSLATED)
            if not args.skip_respond:
                stages.run_mt_respond(p(TRANSLATED), p(RESPONDED), concurrency=args.respond_concurrency)
                verify_input = p(RESPONDED)
        else:
            stages.run_mt_converse(
                data_paths,
                p(CONVERSED),
                interpreter_provider_type=args.interpreter_provider,
                interpreter_model=args.interpreter_model,
                interpreter_thinking=args.interpreter_thinking_level,
                checklist_provider_type=args.checklist_provider,
                checklist_model=args.checklist_model,
                checklist_thinking=args.checklist_thinking_level,
                context_mode=args.context_mode,
                concurrency=args.converse_concurrency,
                no_function_grounding=args.no_function_grounding,
                defer_checklist=args.defer_checklist,
            )
            verify_input = p(CONVERSED)
            if args.defer_checklist:
                stages.run_mt_checklist_batch(
                    p(CONVERSED),
                    p(CHECKLISTED),
                    checklist_provider_type=args.checklist_provider,
                    checklist_model=args.checklist_model,
                    checklist_thinking=args.checklist_thinking_level,
                    no_function_grounding=args.no_function_grounding,
                    backend="batch",
                    poll_interval=args.poll_interval,
                    batch_timeout=args.batch_timeout,
                    api_key=api_key,
                )
                verify_input = p(CHECKLISTED)

        stages.run_mt_verify(
            verify_input, p(VERIFIED), min_confidence=args.min_confidence, concurrency=args.verify_concurrency
        )
        stages.run_mt_judge_turns(
            p(VERIFIED),
            p(TURN_JUDGED),
            provider_type=args.judge_provider,
            model_name=args.judge_model,
            thinking_level=args.judge_thinking_level,
            concurrency=args.judge_turns_concurrency,
            judge_history=args.judge_history,
            backend=args.judge_backend,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=api_key,
        )
        stages.run_mt_judge_conv(
            data_paths,
            p(TURN_JUDGED),
            p(CONV_JUDGED),
            judge_provider_type=args.judge_provider,
            judge_model=args.judge_model,
            judge_thinking=args.judge_thinking_level,
            checklist_provider_type=args.checklist_provider,
            checklist_model=args.checklist_model,
            checklist_thinking=args.checklist_thinking_level,
            concurrency=args.judge_conv_concurrency,
            no_function_grounding=args.no_function_grounding,
            backend=args.judge_backend,
            poll_interval=args.poll_interval,
            batch_timeout=args.batch_timeout,
            api_key=api_key,
        )
        stages.run_mt_consolidate(p(CONV_JUDGED), p(TURN_JUDGED), p(RESULTS), p(RESULTS_TURNS))
        print(f"\n[done] Final results: {p(RESULTS)} (+ {p(RESULTS_TURNS)})")

    if usage_sidecar_target is None and hasattr(args, "output"):
        usage_sidecar_target = args.output
    if usage_sidecar_target is not None:
        totals = get_usage_totals()
        if totals:
            import json

            sidecar_path = (
                os.path.join(usage_sidecar_target, f"usage_totals.{os.getpid()}.json")
                if os.path.isdir(usage_sidecar_target)
                else f"{usage_sidecar_target}.usage_totals.{os.getpid()}.json"
            )
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({f"{k[0]}|{k[1]}": v for k, v in totals.items()}, f, indent=2)
            print(f"[usage] token totals written to {sidecar_path}")


if __name__ == "__main__":
    main()
