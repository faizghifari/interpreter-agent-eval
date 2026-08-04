"""Human-annotation (Task A) validity filter, shared by both the single-turn
and multi-turn checklist generators. Optional post-dedup step: drop criteria
whose taxonomy function is Situation-specific (by human majority vote) and/or
below a meaningfulness threshold.

Moved here from research/functions/filter_by_human_validity.py so it's
importable production code, not a one-off research script. Decisions
(unchanged from the original research finding):
  - Situation-specific functions are dropped outright — case-by-case 3rd-party
    validation (e.g. a cross-check model) is a possible future step, not done here.
  - Meaningfulness: average across annotators (not per-annotator min).
  - Arabic has only 2/3 annotators completed; on A4 disagreement between the
    2, lean towards "generality" (i.e. keep/valid) rather than dropping.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "outputs" / "annotation_sheets" / "results"

LANG_2TO3 = {"ar": "arb", "bn": "ben", "id": "ind", "ko": "kor"}
DROP_LABEL = "situation-specific"

DEFAULT_MEANINGFUL_THRESHOLD = 4.0

_validity_table_cache: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None


def _norm_label(raw: Optional[str]) -> str:
    return (raw or "").strip().lower()


def _load_task_a() -> Dict[str, Dict[str, Dict[str, list]]]:
    by_target: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"generality_votes": [], "meaningful": []})
    )
    for path in sorted(RESULTS_DIR.glob("*_Task_A.csv")):
        lang2 = path.stem.split("_")[0]
        target = LANG_2TO3.get(lang2)
        if not target:
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = row.get("Function ID", "").strip()
                if not fid:
                    continue
                a1_raw = row.get("A1 Meaningful? (1-5)", "").strip()
                a4_raw = row.get("A4 Generality (General/Situation-specific/Mixed)", "").strip()
                if a1_raw:
                    by_target[target][fid]["meaningful"].append(float(a1_raw))
                if a4_raw:
                    by_target[target][fid]["generality_votes"].append(_norm_label(a4_raw))
    return by_target


def _resolve_generality(target: str, votes: List[str]) -> Tuple[Optional[str], str]:
    if not votes:
        return None, "no_votes"
    counts: Dict[str, int] = defaultdict(int)
    for v in votes:
        counts[v] += 1
    if target == "arb" and len(votes) == 2:
        if votes[0] == votes[1]:
            return votes[0], "agree_2"
        if DROP_LABEL in votes:
            other = [v for v in votes if v != DROP_LABEL][0]
            return other, "disagree_2_lean_generality"
        return votes[0], "disagree_2_both_valid"
    top_label, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count >= 2:
        return top_label, "majority_3"
    if DROP_LABEL in counts:
        non_drop = [v for v in votes if v != DROP_LABEL]
        return non_drop[0], "tie_3_lean_generality"
    return votes[0], "tie_3_both_valid"


def _build_validity_table() -> Dict[str, Dict[str, Dict[str, Any]]]:
    task_a = _load_task_a()
    table: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for target, funcs in task_a.items():
        for fid, data in funcs.items():
            generality, resolution = _resolve_generality(target, data["generality_votes"])
            meaningful_mean = statistics.fmean(data["meaningful"]) if data["meaningful"] else None
            table[target][fid] = {
                "generality": generality,
                "resolution": resolution,
                "meaningful_mean": meaningful_mean,
            }
    return table


def get_validity_table() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Cached: {target_lang: {function_id: {generality, resolution, meaningful_mean}}}."""
    global _validity_table_cache
    if _validity_table_cache is None:
        _validity_table_cache = _build_validity_table()
    return _validity_table_cache


def is_valid_by_annotation(
    target_lang: str,
    function_id: Optional[str],
    meaningful_threshold: float = DEFAULT_MEANINGFUL_THRESHOLD,
) -> bool:
    """True = keep this criterion. Ungrounded (function_id is None) or functions
    with no Task A entry are always kept (can't be judged, conservative default).

    Meaningfulness-only (Task A A1, >= meaningful_threshold) -- NOT generality
    (A4 Situation-specific/General/Mixed). Dropping Situation-specific functions
    outright was a research-only simplification (no case-by-case 3rd-party
    validation was ever done for it); the production filter only applies the
    meaningfulness threshold, which is directly validated."""
    if not function_id:
        return True
    row = get_validity_table().get(target_lang, {}).get(function_id)
    if row is None:
        return True
    return row["meaningful_mean"] is None or row["meaningful_mean"] >= meaningful_threshold


def filter_items_by_annotation(
    items: List[Any],
    target_lang: str,
    meaningful_threshold: float = DEFAULT_MEANINGFUL_THRESHOLD,
    function_id_getter=lambda item: getattr(item, "function_id", None) or item.get("function_id"),
) -> List[Any]:
    """Filter a list of checklist items (ChecklistItem objects or dicts), keeping
    only those whose function passes the human-annotation validity check."""
    return [it for it in items if is_valid_by_annotation(target_lang, function_id_getter(it), meaningful_threshold)]
