import argparse
import glob
import os
import json
import sys
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, "src")
sys.path.append(src_dir)

from interpreter_agent_eval.providers import GoogleAIProvider
from interpreter_agent_eval.user import User
from interpreter_agent_eval.interpreter import InterpreterAgent
from interpreter_agent_eval.evaluator import EvaluationFramework

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_THINKING_HIGH = {
    "thinking_config": {"include_thoughts": True, "thinking_level": "high"}
}


class _NoOpProvider:
    """Placeholder provider; generation is never used during re-evaluation."""

    def generate(self, *args, **kwargs):
        raise RuntimeError("NoOp provider should not be used for generation.")


def create_judge_provider():
    return GoogleAIProvider(model_name="gemini-3-pro-preview", **GOOGLE_THINKING_HIGH)


def re_evaluate_file(input_file_path: str, judge_provider):
    if not os.path.exists(input_file_path):
        print(f"File not found: {input_file_path}")
        return

    print(f"\n{'='*50}")
    print(f"Re-evaluating {os.path.basename(input_file_path)}")
    print(f"{'='*50}")

    output_dir = os.path.dirname(input_file_path)
    base_name = os.path.basename(input_file_path).replace(".jsonl", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"re_eval_{base_name}_{timestamp}.jsonl")

    samples = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(samples)} samples.")

    # We need a dummy interpreter and users to initialize EvaluationFramework
    # Their actual properties don't matter much as we hijack the conversation log
    dummy_provider = _NoOpProvider()
    interpreter = InterpreterAgent(dummy_provider, "eng", "eng", name="Interpreter")
    user_a = User("User A", "eng")
    user_b = User("User B", "eng")

    processed_count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}...")

            source_text = sample.get("source_text")
            translated_text = sample.get("translated_text")
            user_b_response = sample.get("user_b_response")
            verification_prompt = sample.get("verification_prompt")
            conversation_context = sample.get("conversation_context")
            source_lang = sample.get("source_lang", "eng")
            target_lang = sample.get("target_lang", "eng")

            # Check properly if we have the necessary data
            if not (source_text and translated_text and verification_prompt):
                print(
                    f"  Skipping sample {i+1}: Missing essential data (source, translation, or prompt)."
                )
                # Write original sample back without changes?? Or skip?
                # Let's write updated version with error note or just skip re-eval and keep old
                # User asked to re-run eval, so we assume we want new evaluations.
                # If we can't evaluate, we probably shouldn't write incomplete data effectively filtering it?
                # Or just write the original. Let's write the original.
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            # Re-initialize framework for each sample to reset context/logs
            # We use the specific context for this sample
            framework = EvaluationFramework(
                user1=user_a,
                user2=user_b,
                interpreter=interpreter,
                conversation_context=conversation_context,
            )

            # Manually populate conversation log
            # The evaluator looks at self.conversation_log[turn_index]
            # to get original and translated messages

            # We need to ensure 'to_user' matches one of the user names for clean logic in evaluate_with_judge
            # forcing user1 -> user2 direction for simplicity of the log structure

            turn_data = {
                "turn": 1,
                "timestamp": datetime.now().isoformat(),
                "from_user": user_a.name,
                "to_user": user_b.name,
                "original_message": source_text,
                "original_language": source_lang,
                "translated_message": translated_text,
                "translated_language": target_lang,
                "translation_time": 0.0,
            }
            framework.conversation_log.append(turn_data)

            try:
                # Run evaluation
                # glotlid_model=None disables language verification locally
                judge_eval = framework.evaluate_with_judge(
                    judge_llm_provider=judge_provider,
                    verification_prompt=verification_prompt,
                    turn_index=0,
                    glotlid_model=None,
                    response_text=user_b_response,
                    response_language=target_lang,
                )

                # Update sample with new evaluation results
                # We overwrite the old evaluation
                sample["evaluation"] = judge_eval.model_dump()
                sample["completion_rate"] = judge_eval.get_completion_rate()
                sample["success_rate"] = judge_eval.get_success_rate()

                # Update timestamp to show when re-eval happened? Or keep orig?
                # Let's add re_eval_timestamp
                sample["re_eval_timestamp"] = datetime.now().isoformat()

                print(f"  Success rate: {judge_eval.get_success_rate()}")

            except Exception as e:
                print(f"  Error evaluating sample {i+1}: {e}")
                # Keep original if failed?

            # Write result
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"Finished re-evaluating {input_file_path}. Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing evaluation JSONL files with the judge model."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="One or more evaluation JSONL files. If omitted, outputs/eval_*.jsonl is used.",
    )
    parser.add_argument(
        "--input-glob",
        default=str(Path(root_dir) / "outputs" / "eval_*.jsonl"),
        help="Glob pattern used when --inputs is not provided.",
    )
    args = parser.parse_args()

    print("Initializing Judge Provider...")
    try:
        judge_provider = create_judge_provider()
    except Exception as e:
        print(f"Failed to create judge provider: {e}")
        return

    files_to_process = args.inputs or sorted(glob.glob(args.input_glob))

    if not files_to_process:
        print("No input files found for re-evaluation.")
        return

    for file_path in files_to_process:
        re_evaluate_file(file_path, judge_provider)


if __name__ == "__main__":
    main()
