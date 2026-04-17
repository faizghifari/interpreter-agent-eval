import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import sys
import io

# Force UTF-8 output on Windows to handle Korean/Arabic/Indonesian characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, "src")
sys.path.append(src_dir)

from interpreter_agent_eval.providers import GoogleAIProvider, OpenAIProvider, OpenRouterProvider
from interpreter_agent_eval.interpreter import InterpreterAgent
from interpreter_agent_eval.user import User
from interpreter_agent_eval.evaluator import EvaluationFramework
from interpreter_agent_eval.utils.language_verification import load_glotlid_model
from typing import Optional

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_THINKING_HIGH = {
    "thinking_config": {"include_thoughts": True, "thinking_level": "high"}
}
GOOGLE_THINKING_MINIMAL = {
    "thinking_config": {"include_thoughts": True, "thinking_level": "minimal"}
}


DEFAULT_INTERPRETER_PROVIDER = "gemini"
DEFAULT_INTERPRETER_MODEL = "gemini-3.1-flash-lite-preview"


def create_judge_provider():
    return GoogleAIProvider(model_name="gemini-3.1-pro-preview", **GOOGLE_THINKING_HIGH)


def build_interpreter_provider(provider_type: str, model_name: str):
    """Build an interpreter LLM provider from CLI-supplied type and model name."""
    if provider_type == "gemini":
        return GoogleAIProvider(model_name=model_name, **GOOGLE_THINKING_MINIMAL)
    elif provider_type == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return OpenRouterProvider(api_key=api_key, model_name=model_name, app_name="mt-eval")
    elif provider_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return OpenAIProvider(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(
            f"Unknown interpreter provider '{provider_type}'. "
            "Choose from: gemini, openrouter, openai"
        )


def create_id_model_provider():
    # SEA-LION v4 Qwen VL 8B — SEA languages only, no Korean/Arabic
    return OpenAIProvider(
        api_key="lm-studio",
        model_name=os.getenv("LM_STUDIO_ID_MODEL", "qwen-sea-lion-v4-8b-vl"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    )


def create_kr_model_provider():
    # EXAONE-3.5-7.8B — Korean/English bilingual only, no Indonesian/Arabic
    return OpenAIProvider(
        api_key="lm-studio",
        model_name=os.getenv("LM_STUDIO_KR_MODEL", "exaone-3.5-7.8b-instruct"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    )


def create_ar_model_provider():
    # Jais-2-8B-Chat — Arabic/English only, no Korean/Indonesian
    # NOTE: swap EXAONE for c4ai-command-r7b-arabic-02-2025 in LM Studio before running Arabic evals
    return OpenAIProvider(
        api_key="lm-studio",
        model_name=os.getenv("LM_STUDIO_AR_MODEL", "c4ai-command-r7b-arabic-02-2025"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    )


def run_simulation_sample(
    data_file: str,
    num_samples: Optional[int] = 1,
    glotlid_model=None,
    resume_file: Optional[str] = None,
    interpreter_provider=None,
    interpreter_label: Optional[str] = None,
):
    print(f"\n{'='*50}")
    print(f"Running Simulation for {os.path.basename(data_file)}")
    print(f"{'='*50}")

    # Output file setup
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(data_file).replace(".jsonl", "")

    # Resume: reuse existing output file and skip already-processed indices
    already_done = set()
    if resume_file and os.path.exists(resume_file):
        output_path = resume_file
        with open(resume_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    already_done.add(entry["sample_index"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming from: {output_path}")
        print(
            f"Already completed: {len(already_done)} sample(s) — skipping indices {sorted(already_done)[:5]}{'...' if len(already_done) > 5 else ''}"
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"eval_{base_name}_{timestamp}.jsonl")
        print(f"Results will be saved to: {output_path}")

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        if num_samples is None:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        else:
            for _ in range(num_samples):
                line = f.readline()
                if not line:
                    break
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    break

    if not samples:
        print(f"No samples found in {data_file}")
        return

    # Filter out already-processed samples
    remaining = [
        (i + 1, s) for i, s in enumerate(samples) if (i + 1) not in already_done
    ]
    print(f"Loaded {len(samples)} sample(s) — {len(remaining)} remaining to evaluate")

    # Setup shared providers
    try:
        if interpreter_provider is None:
            interpreter_provider = build_interpreter_provider(
                DEFAULT_INTERPRETER_PROVIDER, DEFAULT_INTERPRETER_MODEL
            )
            interpreter_label = interpreter_label or f"{DEFAULT_INTERPRETER_PROVIDER}:{DEFAULT_INTERPRETER_MODEL}"
        judge_provider = create_judge_provider()
    except Exception as e:
        print(f"Failed to initialize providers: {e}")
        return

    print(f"Interpreter: {interpreter_label}")

    if not remaining:
        print("All samples already evaluated. Nothing to do.")
        return

    # Process each remaining sample (sample_index is 1-based, matches original position)
    for sample_index, sample in remaining:
        print(f"\n--- Sample {sample_index}/{len(samples)} ---")
        print(f"Scenario: {sample.get('Category')}")
        print(f"Source: {sample.get('source_text')}")

        # Read language codes from sample (each sample can have different direction)
        source_lang = sample.get("source_language_code")
        target_lang = sample.get("target_language_code")

        if not source_lang or not target_lang:
            print(f"  Skipping sample {sample_index}: Missing language codes")
            continue

        print(f"  Direction: {source_lang} → {target_lang}")

        # Get language-specific providers and constraints
        try:
            if source_lang == "arb":
                user_a_provider = create_ar_model_provider()
                user_a_constraint = "مهم جداً: أنت تتحدث وتفهم اللغة العربية فقط. إذا تلقيت رسالة بلغة أخرى، يجب أن ترد بالعربية قائلاً إنك تفهم العربية فقط."
                user_a_lang_full = "Arabic"
            elif source_lang == "ind":
                user_a_provider = create_id_model_provider()
                user_a_constraint = "PENTING: Anda HANYA bisa berbicara dan memahami Bahasa Indonesia. Jika Anda menerima input dalam bahasa lain, Anda harus menjawab dalam Bahasa Indonesia mengatakan bahwa Anda hanya mengerti Bahasa Indonesia."
                user_a_lang_full = "Indonesian"
            elif source_lang == "kor":
                user_a_provider = create_kr_model_provider()
                user_a_constraint = "중요: 당신은 한국어만 말하고 이해할 수 있습니다. 다른 언어로 입력을 받으면 한국어로만 이해할 수 있다고 한국어로 대답해야 합니다."
                user_a_lang_full = "Korean"
            else:
                print(
                    f"  Skipping sample {sample_index}: Unknown source language {source_lang}"
                )
                continue

            if target_lang == "ind":
                user_b_provider = create_id_model_provider()
                user_b_constraint = "PENTING: Anda HANYA bisa berbicara dan memahami Bahasa Indonesia. Jika Anda menerima input dalam bahasa lain, Anda harus menjawab dalam Bahasa Indonesia mengatakan bahwa Anda hanya mengerti Bahasa Indonesia."
                user_b_lang_full = "Indonesian"
            elif target_lang == "kor":
                user_b_provider = create_kr_model_provider()
                user_b_constraint = "중요: 당신은 한국어만 말하고 이해할 수 있습니다. 다른 언어로 입력을 받으면 한국어로만 이해할 수 있다고 한국어로 대답해야 합니다."
                user_b_lang_full = "Korean"
            elif target_lang == "arb":
                user_b_provider = create_ar_model_provider()
                user_b_constraint = "مهم جداً: أنت تتحدث وتفهم اللغة العربية فقط. إذا تلقيت رسالة بلغة أخرى، يجب أن ترد بالعربية قائلاً إنك تفهم العربية فقط."
                user_b_lang_full = "Arabic"
            else:
                print(
                    f"  Skipping sample {sample_index}: Unknown target language {target_lang}"
                )
                continue
        except Exception as e:
            print(
                f"  Skipping sample {sample_index}: Provider initialization error: {e}"
            )
            continue

        # Get conversation context from sample
        conversation_context = sample.get(
            "conversation_context", "A general conversation between two users."
        )

        # Initialize Interpreter with conversation context
        interpreter = InterpreterAgent(
            llm_provider=interpreter_provider,
            source_language=source_lang,
            target_language=target_lang,
            conversation_context=conversation_context,
            name="AI Interpreter",
        )

        # Initialize Users for this sample (contexts change per sample)
        # Note: User A is set to is_llm=False to ensure they say the exact source_text
        # instead of treating it as a prompt to generate a new message.
        user_a = User(
            name=f"User A ({user_a_lang_full})",
            language=source_lang,
            is_llm=False,
            llm_provider=user_a_provider,
            context=f"{sample.get('user_a_context', '')}\n\n{user_a_constraint}",
        )

        user_b = User(
            name=f"User B ({user_b_lang_full})",
            language=target_lang,
            is_llm=True,
            llm_provider=user_b_provider,
            context=f"{sample.get('user_b_context', '')}\n\n{user_b_constraint}",
        )

        # Initialize EvaluationFramework
        evaluator = EvaluationFramework(
            user1=user_a,
            user2=user_b,
            interpreter=interpreter,
            conversation_context=conversation_context,
            name=f"eval_{source_lang}_{target_lang}_sample_{sample_index}",
        )

        # --- Execution: Run conversation through evaluator ---
        source_message = sample["source_text"]
        print(f"\n[{user_a.name}] sends: {source_message}")

        try:
            # Run a single-turn conversation (User A -> Interpreter -> User B)
            evaluator.run_conversation([source_message], from_user=1)

            # The conversation log now has the translation
            translation_result = None
            response_from_b = None

            if evaluator.conversation_log:
                turn = evaluator.conversation_log[0]
                translation_result = turn["translated_message"]
                print(f"[Interpreter] Translation: {translation_result}")

                # Now have User B generate a response to the translation
                # User B already received the translation in their history
                if user_b.conversation_history:
                    # Build prompt for User B to respond
                    response_from_b = user_b.send_message(
                        ""
                    )  # Empty message triggers LLM to generate response based on history
                    print(f"[{user_b.name}] responds: {response_from_b}")
            else:
                print("No conversation log generated")
                continue

        except Exception as e:
            print(f"Error during conversation: {e}")
            continue

        # --- Evaluation ---
        print(f"\n[Judge] Evaluating...")
        try:
            judge_eval = evaluator.evaluate_with_judge(
                judge_llm_provider=judge_provider,
                verification_prompt=sample["verification_prompt"],
                turn_index=0,  # Evaluate first (and only) turn
                glotlid_model=glotlid_model,  # Pass GlotLID model for language verification
                min_confidence=0.8,  # Reduced threshold for language verification
                response_text=response_from_b,  # Pass User B's response for verification
                response_language=target_lang,  # Expected language of response
            )

            print("\n--- Evaluation Results ---")

            # Show language verification results
            if judge_eval.translation_language_check:
                trans_check = judge_eval.translation_language_check
                trans_status = "[OK]" if trans_check.is_correct else "[FAIL]"
                print(f"{trans_status} Translation Language: {trans_check.message}")

            if judge_eval.response_language_check:
                resp_check = judge_eval.response_language_check
                resp_status = "[OK]" if resp_check.is_correct else "[FAIL]"
                print(f"{resp_status} Response Language: {resp_check.message}")

            if not judge_eval.language_check_passed:
                print(
                    "\n[!] LANGUAGE CHECK FAILED - Task automatically marked as failed\n"
                )

        except Exception as e:
            print(f"Evaluation error: {e}")
            judge_eval = None

        # Save results
        result_entry = {
            "sample_index": sample_index,
            "timestamp": datetime.now().isoformat(),
            "interpreter": interpreter_label,
            "category": sample.get("Category"),
            "conversation_context": conversation_context,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_text": source_message,
            "user_a_context": sample.get("user_a_context"),
            "user_b_context": sample.get("user_b_context"),
            "translated_text": (
                translation_result if "translation_result" in locals() else None
            ),
            "user_b_response": response_from_b,
            "verification_prompt": sample.get("verification_prompt"),
            "evaluation": judge_eval.model_dump() if judge_eval else None,
            "completion_rate": (
                judge_eval.get_completion_rate() if judge_eval else "0/0"
            ),
            "success_rate": judge_eval.get_success_rate() if judge_eval else 0.0,
        }

        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to Python types."""
                import numpy as np

                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.bool_, np.integer)):
                    return bool(obj) if isinstance(obj, np.bool_) else int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            result_entry = convert_numpy_types(result_entry)

            with open(output_path, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            print(f"Result saved to {output_path}")
        except Exception as e:
            print(f"Error saving result: {e}")


def main():
    import argparse

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(root_path, "data", "enriched", "id_kor_maps.jsonl")

    parser = argparse.ArgumentParser(description="Run interpreter agent evaluation")
    parser.add_argument(
        "--data",
        nargs="+",
        default=[default_data],
        help="Path(s) to enriched .jsonl data file(s). Defaults to id_kor_maps.jsonl",
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
        help="Path to an existing output .jsonl file to resume from (skips already-evaluated samples)",
    )
    parser.add_argument(
        "--interpreter-provider",
        dest="interpreter_provider",
        default=DEFAULT_INTERPRETER_PROVIDER,
        choices=["gemini", "openrouter", "openai"],
        help=f"Provider for the interpreter agent (default: {DEFAULT_INTERPRETER_PROVIDER})",
    )
    parser.add_argument(
        "--interpreter-model",
        dest="interpreter_model",
        default=DEFAULT_INTERPRETER_MODEL,
        help=f"Model name/ID for the interpreter agent (default: {DEFAULT_INTERPRETER_MODEL})",
    )
    args = parser.parse_args()

    # Load GlotLID model once for language verification
    print("\n" + "=" * 50)
    print("Initializing Language Verification")
    print("=" * 50)
    glotlid_model = load_glotlid_model()
    if glotlid_model:
        print("[OK] GlotLID model loaded successfully")
    else:
        print("[!] GlotLID model not available - language verification disabled")
    print()

    # Build interpreter provider once, shared across all data files
    interpreter_label = f"{args.interpreter_provider}:{args.interpreter_model}"
    print(f"Interpreter: {interpreter_label}")
    try:
        interp_provider = build_interpreter_provider(
            args.interpreter_provider, args.interpreter_model
        )
    except Exception as e:
        print(f"Failed to build interpreter provider: {e}")
        return

    for data_path in args.data:
        if not os.path.isabs(data_path):
            data_path = os.path.join(root_path, "data", "enriched", data_path)
        if os.path.exists(data_path):
            run_simulation_sample(
                data_path,
                num_samples=args.num_samples,
                glotlid_model=glotlid_model,
                resume_file=args.resume,
                interpreter_provider=interp_provider,
                interpreter_label=interpreter_label,
            )
        else:
            print(f"File not found: {data_path}")


if __name__ == "__main__":
    main()
