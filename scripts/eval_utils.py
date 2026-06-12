"""
Shared utilities for evaluation analysis scripts.

The new opensubs_eval layout is:
    outputs/opensubs_eval/<lang_pair>/<target>_results[_<model_tag>].jsonl
where <lang_pair> is e.g. "ar-bn" / "bn-ko" and <target> is the ISO 639-3
code of the language being translated INTO. <model_tag> is empty for the
default interpreter and one of {gpt5mini, qwen35flash} for the variants.

Layered checklist for each record lives in
    outputs/opensubtitles_augmented/<lang_pair>/augmented.jsonl
and is concatenated into ``verification_prompt`` in L3 → L2 → L1 order
(see ``_compose_verification_prompt`` in scripts/augment_opensubs_maps.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

DEFAULT_OPENSUBS_EVAL_DIRNAME = "opensubs_eval"
DEFAULT_AUGMENTED_DIRNAME = "opensubtitles_augmented"
DEFAULT_LEGACY_OUTPUTS_DIRNAME = "outputs"
DEFAULT_ENRICHED_DIRNAME = "data/enriched"


def project_root() -> Path:
    """Return the interpreter-agent-eval root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def default_opensubs_eval_dir() -> Path:
    return project_root() / "outputs" / DEFAULT_OPENSUBS_EVAL_DIRNAME


def default_augmented_dir() -> Path:
    return project_root() / "outputs" / DEFAULT_AUGMENTED_DIRNAME


def default_legacy_outputs_dir() -> Path:
    return project_root() / "outputs"


def default_enriched_dir() -> Path:
    return project_root() / "data" / "enriched"


# ---------------------------------------------------------------------------
# Model / language display helpers
# ---------------------------------------------------------------------------

# Maps the file-name suffix (or "default") to a stable short tag.
MODEL_TAG_DEFAULT = "default"

# The interpreter string saved in records is authoritative; this map is used
# only when a record is missing it (older outputs).
INTERPRETER_TAG_FROM_FILENAME = {
    MODEL_TAG_DEFAULT: "gemini:gemini-3.1-flash-lite-preview",
    "gpt5mini": "openai:gpt-5.4-mini-2026-03-17",
    "qwen35flash": "openrouter:qwen/qwen3.5-flash-02-23",
}

INTERPRETER_LABEL = {
    "gemini:gemini-3.1-flash-lite-preview": "Gemini Flash Lite",
    "openai:gpt-5.4-mini-2026-03-17": "GPT-5.4 Mini",
    "openrouter:qwen/qwen3.5-flash-02-23": "Qwen3.5 Flash",
}

MODEL_ORDER = ["Gemini Flash Lite", "GPT-5.4 Mini", "Qwen3.5 Flash"]
MODEL_COLORS = {
    "Gemini Flash Lite": "#4285F4",
    "GPT-5.4 Mini": "#34A853",
    "Qwen3.5 Flash": "#EA4335",
}

LANG_DISPLAY = {
    "arb": "AR", "ben": "BN", "ind": "ID", "kor": "KO", "eng": "EN",
}

LAYER_ORDER = ["L1", "L2", "L3"]
LAYER_LABELS = {
    "L1": "L1: Semantic Core",
    "L2": "L2: Pragmatic Function",
    "L3": "L3: Cultural / Social",
}


def model_label(interpreter: Optional[str]) -> str:
    """Return a stable display label for an interpreter string."""
    if not interpreter:
        return "Unknown"
    return INTERPRETER_LABEL.get(interpreter, interpreter)


def lang_display(code: str) -> str:
    return LANG_DISPLAY.get(code, code.upper())


def direction_display(src: str, tgt: str) -> str:
    return f"{lang_display(src)} → {lang_display(tgt)}"


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl_records(filepaths: Iterable[str]) -> List[dict]:
    """Load all JSON-L records from the given files, skipping malformed lines."""
    records: List[dict] = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ---------------------------------------------------------------------------
# Eval file discovery (new opensubs_eval layout)
# ---------------------------------------------------------------------------

_EVAL_FILENAME_RE = re.compile(r"^([a-z]{3})_results(?:_(.+))?\.jsonl$")


def parse_eval_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Return (target_lang_code, model_tag) or None if filename does not match.

    Examples:
        "ind_results.jsonl"           -> ("ind", "default")
        "arb_results_gpt5mini.jsonl"  -> ("arb", "gpt5mini")
    """
    m = _EVAL_FILENAME_RE.match(filename)
    if not m:
        return None
    target_lang = m.group(1)
    model_tag = m.group(2) or MODEL_TAG_DEFAULT
    return target_lang, model_tag


def discover_eval_files(opensubs_eval_dir: Path) -> List[Tuple[str, str, str, str]]:
    """Walk ``opensubs_eval_dir`` and return [(lang_pair, target_lang, model_tag, path), ...]."""
    base = Path(opensubs_eval_dir)
    out: List[Tuple[str, str, str, str]] = []
    if not base.exists():
        return out
    for lp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for fp in sorted(lp_dir.glob("*.jsonl")):
            parsed = parse_eval_filename(fp.name)
            if parsed is None:
                continue
            target_lang, model_tag = parsed
            out.append((lp_dir.name, target_lang, model_tag, str(fp)))
    return out


# ---------------------------------------------------------------------------
# Legacy outputs/*.jsonl manifest (MAPS + old opensubs runs)
# ---------------------------------------------------------------------------

# Each entry: (filename) → {dataset, dataset_source, source_lang, target_lang,
# enriched_file, model_interpreter, layer_order}.
#
# layer_order indicates the order in which the verification_prompt was composed.
# Legacy runs use L1 → L2 → L3 (the enriched data file order); new opensubs
# runs use L3 → L2 → L1 (see _compose_verification_prompt in
# scripts/augment_opensubs_maps.py).

LEGACY_FILE_MANIFEST: Dict[str, Dict[str, str]] = {
    # ind → arb (MAPS proverbs)
    "eval_id_arb_maps_20260416_111707.jsonl": {
        "dataset": "id_arb_maps", "dataset_source": "MAPS",
        "source_lang": "ind", "target_lang": "arb",
        "enriched_file": "id_arb_maps.jsonl",
        "interpreter": "gemini:gemini-3.1-flash-lite-preview",
        "layer_order": "L1L2L3",
    },
    "eval_id_arb_maps_20260416_155811.jsonl": {
        "dataset": "id_arb_maps", "dataset_source": "MAPS",
        "source_lang": "ind", "target_lang": "arb",
        "enriched_file": "id_arb_maps.jsonl",
        "interpreter": "openai:gpt-5.4-mini-2026-03-17",
        "layer_order": "L1L2L3",
    },
    "eval_id_arb_maps_20260416_195434.jsonl": {
        "dataset": "id_arb_maps", "dataset_source": "MAPS",
        "source_lang": "ind", "target_lang": "arb",
        "enriched_file": "id_arb_maps.jsonl",
        "interpreter": "openrouter:qwen/qwen3.5-flash-02-23",
        "layer_order": "L1L2L3",
    },
    # ind → kor (MAPS proverbs)
    "eval_id_kor_maps_20260415_215446.jsonl": {
        "dataset": "id_kor_maps", "dataset_source": "MAPS",
        "source_lang": "ind", "target_lang": "kor",
        "enriched_file": "id_kor_maps.jsonl",
        "interpreter": "gemini:gemini-3.1-flash-lite-preview",
        "layer_order": "L1L2L3",
    },
    "eval_id_kor_maps_20260416_162759.jsonl": {
        "dataset": "id_kor_maps", "dataset_source": "MAPS",
        "source_lang": "ind", "target_lang": "kor",
        "enriched_file": "id_kor_maps.jsonl",
        "interpreter": "openai:gpt-5.4-mini-2026-03-17",
        "layer_order": "L1L2L3",
    },
    "eval_id_kor_maps_20260416_200842.jsonl": {
        "dataset": "id_kor_maps", "dataset_source": "MAPS",
        "source_lang": "ind", "target_lang": "kor",
        "enriched_file": "id_kor_maps.jsonl",
        "interpreter": "openrouter:qwen/qwen3.5-flash-02-23",
        "layer_order": "L1L2L3",
    },
    # ind → kor (older opensubs-augmented)
    "eval_id_kor_maps_from_opensubs_20260416_091605.jsonl": {
        "dataset": "id_kor_opensubs_legacy", "dataset_source": "OpenSubs",
        "source_lang": "ind", "target_lang": "kor",
        "enriched_file": "id_kor_maps_from_opensubs.jsonl",
        "interpreter": "gemini:gemini-3.1-flash-lite-preview",
        "layer_order": "L1L2L3",
    },
    "eval_id_kor_maps_from_opensubs_20260416_142034.jsonl": {
        "dataset": "id_kor_opensubs_legacy", "dataset_source": "OpenSubs",
        "source_lang": "ind", "target_lang": "kor",
        "enriched_file": "id_kor_maps_from_opensubs.jsonl",
        "interpreter": "openrouter:qwen/qwen3.5-flash-02-23",
        "layer_order": "L1L2L3",
    },
    "eval_id_kor_maps_from_opensubs_20260416_155742.jsonl": {
        "dataset": "id_kor_opensubs_legacy", "dataset_source": "OpenSubs",
        "source_lang": "ind", "target_lang": "kor",
        "enriched_file": "id_kor_maps_from_opensubs.jsonl",
        "interpreter": "openai:gpt-5.4-mini-2026-03-17",
        "layer_order": "L1L2L3",
    },
    # kor → ind (older opensubs-augmented)
    "eval_kor_id_maps_from_opensubs_20260416_092316.jsonl": {
        "dataset": "kor_id_opensubs_legacy", "dataset_source": "OpenSubs",
        "source_lang": "kor", "target_lang": "ind",
        "enriched_file": "kor_id_maps_from_opensubs.jsonl",
        "interpreter": "gemini:gemini-3.1-flash-lite-preview",
        "layer_order": "L1L2L3",
    },
    "eval_kor_id_maps_from_opensubs_20260416_120148.jsonl": {
        "dataset": "kor_id_opensubs_legacy", "dataset_source": "OpenSubs",
        "source_lang": "kor", "target_lang": "ind",
        "enriched_file": "kor_id_maps_from_opensubs.jsonl",
        "interpreter": "openai:gpt-5.4-mini-2026-03-17",
        "layer_order": "L1L2L3",
    },
    "eval_kor_id_maps_from_opensubs_20260416_124827.jsonl": {
        "dataset": "kor_id_opensubs_legacy", "dataset_source": "OpenSubs",
        "source_lang": "kor", "target_lang": "ind",
        "enriched_file": "kor_id_maps_from_opensubs.jsonl",
        "interpreter": "openrouter:qwen/qwen3.5-flash-02-23",
        "layer_order": "L1L2L3",
    },
}


def discover_legacy_eval_files(
    legacy_dir: Path,
    manifest: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Tuple[str, Dict[str, str]]]:
    """Return [(filepath, meta), ...] for legacy outputs/*.jsonl files.

    Files not in the manifest are skipped — this is intentional, since
    legacy files don't carry the ``interpreter`` field and we need the
    manifest to attribute model identity. Add new files to
    ``LEGACY_FILE_MANIFEST`` as runs are produced.
    """
    base = Path(legacy_dir)
    manifest = manifest or LEGACY_FILE_MANIFEST
    out: List[Tuple[str, Dict[str, str]]] = []
    if not base.exists():
        return out
    for fname, meta in manifest.items():
        fp = base / fname
        if fp.exists():
            out.append((str(fp), meta))
    return out


# ---------------------------------------------------------------------------
# Dataset-source helpers for the new opensubs_eval files
# ---------------------------------------------------------------------------

def opensubs_dataset_name(lang_pair: str, target_lang: str) -> str:
    """Return a canonical 'dataset' name for an opensubs_eval file.

    Encodes both the lang_pair and direction so the same lang_pair with
    two directions is split into two dataset rows.
    """
    return f"{lang_pair}__{target_lang}"


# ---------------------------------------------------------------------------
# Dual success rate (strict = LID-gated, lenient = judge-only)
# ---------------------------------------------------------------------------

def compute_dual_rates(record: dict) -> Tuple[float, float]:
    """Return (lenient_rate, strict_rate).

    * lenient = the LLM judge's success_rate as recorded (criteria met / total).
    * strict  = lenient if the GlotLID language check passed, else 0.0.
      This matches the original behaviour where a failed translation language
      check short-circuits the judge to all-failed.
    """
    sr = record.get("success_rate")
    lenient = float(sr) if sr is not None else 0.0
    ev = record.get("evaluation") or {}
    lid_passed = ev.get("language_check_passed")
    if lid_passed is None:
        # No LID gate available — treat as pass to avoid penalising old records.
        strict = lenient
    else:
        strict = lenient if bool(lid_passed) else 0.0
    return lenient, strict


def parse_record_id(record_id: str) -> Optional[Tuple[str, str, str]]:
    """Split '28739_ben_arb' → ('28739', 'ben', 'arb'). Return None if malformed."""
    m = re.match(r"^([0-9]+)_([a-z]{3})_([a-z]{3})$", record_id or "")
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


# ---------------------------------------------------------------------------
# Layer mapping (criterion id → L1/L2/L3) from augmented data
# ---------------------------------------------------------------------------

def _layer_map_from_arrays(l1: list, l2: list, l3: list, order: str) -> Dict[int, str]:
    """Build {criterion_id: layer} given the three layer arrays and the
    composition order ('L3L2L1' or 'L1L2L3')."""
    cid_to_layer: Dict[int, str] = {}
    cid = 1
    if order == "L3L2L1":
        groups = [(l3, "L3"), (l2, "L2"), (l1, "L1")]
    else:  # default: legacy enriched files
        groups = [(l1, "L1"), (l2, "L2"), (l3, "L3")]
    for items, label in groups:
        for _ in items:
            cid_to_layer[cid] = label
            cid += 1
    return cid_to_layer


def load_layer_map(augmented_dir: Path) -> Dict[str, Dict[int, str]]:
    """Build {record_id: {criterion_id: 'L1'|'L2'|'L3'}} for the new
    opensubs_eval files, indexed by ``record_id`` ('segment_id_src_tgt').

    Verification prompt is composed in **L3 → L2 → L1** order
    (see scripts/augment_opensubs_maps.py::_compose_verification_prompt).
    """
    base = Path(augmented_dir)
    out: Dict[str, Dict[int, str]] = {}
    if not base.exists():
        return out
    for lp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        ap = lp_dir / "augmented.jsonl"
        if not ap.exists():
            continue
        with open(ap, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seg = d.get("segment_id")
                src = d.get("source_language_code")
                tgt = d.get("target_language_code")
                if seg is None or not src or not tgt:
                    continue
                rid = f"{seg}_{src}_{tgt}"
                l1 = d.get("checklist_layer_1_semantic_core") or []
                l2 = d.get("checklist_layer_2_pragmatic_function") or []
                l3 = d.get("checklist_layer_3_cultural_social_constraints") or []
                out[rid] = _layer_map_from_arrays(l1, l2, l3, order="L3L2L1")
    return out


def load_legacy_layer_map(
    enriched_dir: Path,
    manifest: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[Tuple[str, int], Dict[int, str]]:
    """Build {(dataset, sample_index): {criterion_id: layer}} for legacy
    enriched ``data/enriched/*.jsonl`` files. Indexed by ``sample_index``
    (1-based file line number, matching what the legacy runs recorded).

    Legacy enriched files compose verification_prompt as L1 → L2 → L3.
    """
    manifest = manifest or LEGACY_FILE_MANIFEST
    enriched_dir = Path(enriched_dir)
    # collect unique (dataset, enriched_file) pairs to avoid re-reading
    enriched_for_dataset: Dict[str, str] = {}
    for meta in manifest.values():
        ds = meta.get("dataset")
        ef = meta.get("enriched_file")
        if ds and ef and ds not in enriched_for_dataset:
            enriched_for_dataset[ds] = ef

    out: Dict[Tuple[str, int], Dict[int, str]] = {}
    for dataset, fname in enriched_for_dataset.items():
        fp = enriched_dir / fname
        if not fp.exists():
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                l1 = d.get("checklist_layer_1_semantic_core") or []
                l2 = d.get("checklist_layer_2_pragmatic_function") or []
                l3 = d.get("checklist_layer_3_cultural_social_constraints") or []
                out[(dataset, idx)] = _layer_map_from_arrays(
                    l1, l2, l3, order="L1L2L3"
                )
    return out


# ---------------------------------------------------------------------------
# Statistics (no scipy dependency)
# ---------------------------------------------------------------------------

def wilson_ci(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion. ``k`` may be fractional
    (sum of per-record success rates). Returns (low, high)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_ci(
    values: Iterable[float],
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for the mean. Returns (mean, low, high)."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    return (float(arr.mean()), lo, hi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_ztest(s1: float, n1: float, s2: float, n2: float) -> Tuple[float, float]:
    """Two-sided z-test for difference of proportions. Returns (z, p_value)."""
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"))
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return (z, p)


def bootstrap_diff_ci(
    a: Iterable[float],
    b: Iterable[float],
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for the difference of means ``mean(a) - mean(b)``.
    The two samples are resampled independently. Returns (delta, low, high)."""
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if arr_a.size == 0 or arr_b.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, arr_a.size, size=(n_boot, arr_a.size))
    ib = rng.integers(0, arr_b.size, size=(n_boot, arr_b.size))
    diffs = arr_a[ia].mean(axis=1) - arr_b[ib].mean(axis=1)
    lo = float(np.percentile(diffs, (100 - ci) / 2))
    hi = float(np.percentile(diffs, 100 - (100 - ci) / 2))
    return (float(arr_a.mean() - arr_b.mean()), lo, hi)


def bootstrap_paired_test(
    diffs: Iterable[float],
    n_boot: int = 5000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Paired-difference bootstrap for matched samples. ``diffs`` is the list of
    per-pair differences (e.g. frontier_score − baseline_score for the same
    record). Returns (mean_diff, low, high, p_value), where p_value is the
    two-sided bootstrap p for H0: mean_diff = 0 (fraction of resample means on
    the opposite side of zero, doubled)."""
    arr = np.asarray(list(diffs), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    frac_le = float(np.mean(means <= 0.0))
    frac_ge = float(np.mean(means >= 0.0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return (float(arr.mean()), lo, hi, p)


def holm_correction(pvals: Iterable[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, preserving input order.
    NaN p-values are passed through unchanged and excluded from the family."""
    raw = list(pvals)
    idx = [i for i, p in enumerate(raw) if p == p]  # drop NaN
    m = len(idx)
    order = sorted(idx, key=lambda i: raw[i])
    adj = list(raw)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * raw[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
# Backwards-compatible argparse helpers (used by other scripts)
# ---------------------------------------------------------------------------

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
