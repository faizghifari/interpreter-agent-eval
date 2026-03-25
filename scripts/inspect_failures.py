import argparse
import glob
import json
import os
from pathlib import Path


def load_failures(filepaths):
    failures = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Check if success_rate is less than 1.0
                    success_rate = record.get("success_rate", 1.0)
                    if isinstance(success_rate, str):
                        try:
                            success_rate = float(success_rate)
                        except ValueError:
                            success_rate = 0.0  # Treat invalid as failure

                    if success_rate < 1.0:
                        lang_pair = f"{record.get('source_lang', 'unknown')} -> {record.get('target_lang', 'unknown')}"
                        record["language_pair"] = lang_pair
                        failures.append(record)
                except json.JSONDecodeError:
                    continue
    return failures


def print_failure_details(failures):
    if not failures:
        print("No failures found (all success rates are 1.0).")
        return

    print("\n" + "=" * 80)
    print(f" DETAILED FAILURE ANALYSIS ({len(failures)} SAMPLES FOUND)")
    print("=" * 80 + "\n")

    for i, record in enumerate(failures, 1):
        print(
            f"SAMPLE {i}: [{record['language_pair']}] - Category: {record.get('category', 'N/A')}"
        )
        print("-" * 80)
        print(f"Context: {record.get('conversation_context', 'N/A')}")
        print(f"Source:  {record.get('source_text', 'N/A')}")
        print(f"Trans:   {record.get('translated_text', 'N/A')}")
        print("-" * 80)
        print("FAILED CRITERIA:")

        eval_results = record.get("evaluation", {}).get("results", [])
        failed_criteria = [r for r in eval_results if r.get("met") is False]

        if not failed_criteria and record.get("success_rate", 0) < 1.0:
            # Fallback if specific criteria aren't marked but overall isn't 1.0 (though unlikely with this schema)
            print("  (No specific criteria marked as failed, but success rate < 1.0)")

        for fc in failed_criteria:
            print(f"  [X] {fc.get('criteria')}")
            print(f"      Reason: {fc.get('reasoning')}")

        print("\n" + "=" * 80 + "\n")


def main():
    root_dir = Path(__file__).resolve().parent.parent
    default_glob = str(root_dir / "outputs" / "re_eval_*.jsonl")

    parser = argparse.ArgumentParser(
        description="Inspect failed evaluation samples from JSONL outputs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="One or more JSONL files to inspect. If omitted, outputs/re_eval_*.jsonl is used.",
    )
    parser.add_argument(
        "--input-glob",
        default=default_glob,
        help="Glob pattern used when --inputs is not provided.",
    )
    args = parser.parse_args()

    files = args.inputs or sorted(glob.glob(args.input_glob))

    existing_files = [f for f in files if os.path.exists(f)]
    if not existing_files:
        print("No input files found.")
        return

    failures = load_failures(existing_files)
    print_failure_details(failures)


if __name__ == "__main__":
    main()
