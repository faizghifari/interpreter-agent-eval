import argparse
import glob
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from tabulate import tabulate


def load_data(filepaths):
    data = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Infer language pair if not explicitly clear, though filename helps
                    # But the record has source_lang and target_lang
                    lang_pair = f"{record.get('source_lang', 'unknown')} -> {record.get('target_lang', 'unknown')}"
                    record["language_pair"] = lang_pair
                    data.append(record)
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(data)


def parse_args():
    root_dir = Path(__file__).resolve().parent.parent
    default_outputs = root_dir / "outputs"
    default_analysis = default_outputs / "analysis"

    parser = argparse.ArgumentParser(
        description="Analyze evaluation JSONL outputs and generate summary report/plots."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="One or more JSONL files to analyze. If omitted, all outputs/re_eval_*.jsonl are used.",
    )
    parser.add_argument(
        "--input-glob",
        default=str(default_outputs / "re_eval_*.jsonl"),
        help="Glob pattern used when --inputs is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_analysis),
        help="Directory where analysis artifacts (plots) are saved.",
    )
    return parser.parse_args()


def analyze_results(files, output_dir):

    # Check if files exist
    existing_files = [f for f in files if os.path.exists(f)]
    if not existing_files:
        print("No input files found.")
        return

    df = load_data(existing_files)

    # ensure success_rate is float
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")

    print("\n" + "=" * 50)
    print(" EVALUATION ANALYSIS REPORT")
    print("=" * 50 + "\n")

    # 1. Overall Statistics per Language Pair
    print("### 1. Overall Statistics per Language Pair (Success Rate)\n")
    stats = (
        df.groupby("language_pair")["success_rate"]
        .agg(["count", "mean", "max", "min", "std"])
        .reset_index()
    )
    print(tabulate(stats, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("\n")

    # 2. Category Performance
    print("### 2. Performance by Category and Language Pair\n")
    cat_stats = (
        df.groupby(["language_pair", "category"])["success_rate"]
        .agg(["count", "mean"])
        .reset_index()
    )
    cat_pivot = cat_stats.pivot(
        index="category", columns="language_pair", values="mean"
    )

    # Sort by one of the columns if possible, or just index
    print(
        tabulate(
            cat_pivot, headers="keys", tablefmt="github", floatfmt=".4f", showindex=True
        )
    )
    print("\n")

    # Highlighting Weaknesses (Success Rate < 0.6 as arbitrary threshold for "lacking")
    print("### 3. Areas for Improvement (Average Success Rate < 0.6)\n")
    weaknesses = cat_stats[cat_stats["mean"] < 0.6].sort_values("mean")
    if not weaknesses.empty:
        print(tabulate(weaknesses, headers="keys", tablefmt="github", floatfmt=".4f"))
    else:
        print("No categories with average success rate < 0.6 found.")
    print("\n")

    # Highlighting Strengths (Success Rate > 0.8)
    print("### 4. Strong Areas (Average Success Rate >= 0.8)\n")
    strengths = cat_stats[cat_stats["mean"] >= 0.8].sort_values("mean", ascending=False)
    if not strengths.empty:
        print(tabulate(strengths, headers="keys", tablefmt="github", floatfmt=".4f"))
    else:
        print("No categories with average success rate >= 0.8 found.")
    print("\n")

    # Visualization
    os.makedirs(output_dir, exist_ok=True)

    # Bar plot for Categories
    # We clear figure to avoid overlap if run multiple times in same session (though this is a script)
    plt.clf()
    plt.figure(figsize=(12, 8))

    cat_pivot.plot(kind="bar", figsize=(14, 7))
    plt.title("Average Success Rate by Category and Language Pair")
    plt.ylabel("Average Success Rate")
    plt.xlabel("Category")
    plt.ylim(0, 1.1)
    plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)
    plt.legend(title="Language Pair")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_performance.png")
    print(f"Plot saved to {output_dir}/category_performance.png")


if __name__ == "__main__":
    args = parse_args()
    files = args.inputs or sorted(glob.glob(args.input_glob))
    analyze_results(files=files, output_dir=args.output_dir)
