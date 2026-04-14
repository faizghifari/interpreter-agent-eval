"""
Shared utilities for evaluation analysis scripts.

Provides:
    project_root()            → Path to the interpreter-agent-eval package root
    load_jsonl_records(paths) → List[dict] from one or more JSONL files
    add_input_args(parser, default_glob)
    resolve_input_files(args) → List of existing file paths
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List


def project_root() -> Path:
    """Return the interpreter-agent-eval root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def load_jsonl_records(filepaths: List[str]) -> List[dict]:
    """Load all JSON-L records from the given files, skipping malformed lines."""
    records: List[dict] = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def add_input_args(parser: argparse.ArgumentParser, default_glob: str) -> None:
    """Attach --inputs and --input-glob arguments to an ArgumentParser."""
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="One or more JSONL files to process. If omitted, the glob pattern is used.",
    )
    parser.add_argument(
        "--input-glob",
        default=default_glob,
        help="Glob pattern used when --inputs is not provided.",
    )


def resolve_input_files(args: argparse.Namespace) -> List[str]:
    """Return the list of existing files from --inputs or --input-glob."""
    files = args.inputs or sorted(glob.glob(args.input_glob))
    return [f for f in files if os.path.exists(f)]
