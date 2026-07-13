"""
Analyse opensubs_eval JSONL outputs and generate a multi-faceted report.

Discovers two sources of evaluation data:

* New layout — ``outputs/opensubs_eval/<lang_pair>/<target>_results[_<tag>].jsonl``
* Legacy layout — ``outputs/eval_*.jsonl`` (MAPS proverbs + older opensubs runs).
  These are matched against ``data/enriched/<file>.jsonl`` for layer info.
  Only filenames listed in ``LEGACY_FILE_MANIFEST`` (eval_utils.py) are loaded,
  because the legacy records do not carry an ``interpreter`` field.

Outputs (under ``--output-dir``):

    summary_strict.csv / summary_lenient.csv
        Per (dataset, lang_pair, direction, model) success rate (mean, std, n,
        Wilson 95% CI, bootstrap mean CI).
    direction_*.csv         Per (direction, model) collapsed across datasets.
    dataset_compare.csv     MAPS vs OpenSubs for directions present in both.
    source_lang_stats.csv   Per (source_lang, model) collapsed across targets.
    target_lang_stats.csv   Per (target_lang, model) collapsed across sources.
    layer_stats.csv         Per (direction, model, layer) criterion pass rate.
    distribution_buckets.csv  Distribution of per-record success rate into
                              [=0, (0, 0.5], (0.5, 0.8], (0.8, 1), =1] buckets.
    criteria_pass_rate.csv  Per-criterion pass rate (hardest/easiest).
    lid_failure_stats.csv   GlotLID failure rates per (direction, model).
    model_pairwise_tests.csv Two-proportion z-tests pairwise between models.
    failure_cases.csv / differentiating_cases.csv
    fig_*.png               Plots (matplotlib + seaborn).
    report.md               Markdown summary report.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from tabulate import tabulate

from eval_utils import (
    LAYER_LABELS,
    LAYER_ORDER,
    MODEL_COLORS,
    MODEL_ORDER,
    MODEL_SLUG_COLORS,
    MODEL_SLUG_LABEL,
    MODEL_SLUG_ORDER,
    bootstrap_mean_ci,
    compute_dual_rates,
    default_augmented_dir,
    default_enriched_dir,
    default_legacy_outputs_dir,
    default_opensubs_eval_dir,
    direction_display,
    discover_eval_files,
    discover_legacy_eval_files,
    lang_display,
    load_cluster_layer_map,
    load_layer_map,
    load_legacy_layer_map,
    model_label,
    model_label_from_slug,
    normalize_criteria,
    opensubs_dataset_name,
    parse_record_id,
    project_root,
    two_proportion_ztest,
    wilson_ci,
)

warnings.filterwarnings("ignore")

DATASET_SOURCE_ORDER = ["MAPS", "OpenSubs"]
DATASET_SOURCE_COLORS = {"MAPS": "#5B8DEF", "OpenSubs": "#F2A65A"}

BUCKET_EDGES = [-0.001, 0.0001, 0.5, 0.8, 0.9999, 1.001]
BUCKET_LABELS = ["= 0", "(0, 0.5]", "(0.5, 0.8]", "(0.8, 1)", "= 1"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse interpreter-agent eval results (new + legacy layouts)."
    )
    parser.add_argument(
        "--source",
        choices=["store", "legacy"],
        default="store",
        help="Data source. 'store' (default) = unified outputs/results/ "
             "(all 10 models, canonical `model` field). 'legacy' = the old "
             "opensubs_eval + outputs/eval_*.jsonl discovery.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(project_root() / "outputs" / "results"),
        help="Unified results store (used when --source store).",
    )
    parser.add_argument(
        "--clusters-dir",
        default=str(project_root() / "outputs" / "annotation_clusters"),
        help="annotation_clusters dir for layer mapping (used when --source store).",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Exclude synthetic translation-failure fills (store mode only).",
    )
    parser.add_argument(
        "--opensubs-eval-dir",
        default=str(default_opensubs_eval_dir()),
        help="Directory holding <lang_pair>/<target>_results*.jsonl files.",
    )
    parser.add_argument(
        "--augmented-dir",
        default=str(default_augmented_dir()),
        help="Directory holding <lang_pair>/augmented.jsonl source files "
             "(for L1/L2/L3 layer mapping of the new opensubs_eval files).",
    )
    parser.add_argument(
        "--legacy-dir",
        default=str(default_legacy_outputs_dir()),
        help="Directory holding legacy outputs/eval_*.jsonl files (MAPS + old "
             "opensubs runs). Only files in LEGACY_FILE_MANIFEST are loaded.",
    )
    parser.add_argument(
        "--enriched-dir",
        default=str(default_enriched_dir()),
        help="Directory with data/enriched/*.jsonl source files (for L1/L2/L3 "
             "layer mapping of the legacy runs).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "outputs" / "analysis" / "eval_report" / "full"),
        help="Directory where analysis CSVs, plots, and report.md are saved.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Bootstrap replicates for mean CIs (default 2000).",
    )
    parser.add_argument(
        "--min-models-per-record",
        type=int,
        default=2,
        help="Minimum number of models that must have evaluated a record "
             "for it to count as a 'differentiating case' candidate.",
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Skip legacy outputs/*.jsonl files (only analyse the new layout).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _record_row(
    r: dict, dataset: str, dataset_source: str, lang_pair: str,
    model_tag: str, src_default: str, tgt_default: str,
) -> Tuple[dict, list]:
    """Convert one JSONL record into (record_row, [criterion_rows])."""
    src = r.get("source_lang") or src_default
    tgt = r.get("target_lang") or tgt_default
    interp = r.get("interpreter") or ""
    model = model_label(interp)
    rid = r.get("record_id") or ""
    ev = r.get("evaluation") or {}
    results = ev.get("results") or []
    lenient, strict = compute_dual_rates(r)
    tlc = ev.get("translation_language_check") or {}
    rlc = ev.get("response_language_check") or {}

    rec = {
        "dataset": dataset,
        "dataset_source": dataset_source,
        "lang_pair": lang_pair,
        "source_lang": src,
        "target_lang": tgt,
        "direction": direction_display(src, tgt),
        "model": model,
        "model_tag": model_tag,
        "interpreter": interp,
        "judge": r.get("judge"),
        "category": r.get("category"),
        "record_id": rid,
        "sample_index": r.get("sample_index"),
        "success_rate_lenient": lenient,
        "success_rate_strict": strict,
        "lid_passed": ev.get("language_check_passed"),
        "translation_lid_correct": tlc.get("is_correct"),
        "response_lid_correct": rlc.get("is_correct"),
        "translation_lid_confidence": tlc.get("confidence"),
        "response_lid_confidence": rlc.get("confidence"),
        "n_criteria": len(results),
        "n_criteria_met": sum(1 for c in results if c.get("met")),
    }
    crit_rows: List[dict] = []
    for c in results:
        crit_rows.append({
            "dataset": dataset,
            "dataset_source": dataset_source,
            "lang_pair": lang_pair,
            "source_lang": src,
            "target_lang": tgt,
            "direction": direction_display(src, tgt),
            "model": model,
            "record_id": rid,
            "sample_index": r.get("sample_index"),
            "criterion_id": c.get("id"),
            "criteria_text": c.get("criteria") or "",
            "met": bool(c.get("met")),
            "lid_passed": ev.get("language_check_passed"),
        })
    return rec, crit_rows


def build_dataframes(
    eval_files: List[Tuple[str, str, str, str]],
    opensubs_layer_map: Dict[str, Dict[int, str]],
    legacy_files: List[Tuple[str, Dict[str, str]]],
    legacy_layer_map: Dict[Tuple[str, int], Dict[int, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rec_rows: List[dict] = []
    crit_rows: List[dict] = []

    # ---- New opensubs_eval ----
    for lang_pair, target_lang, model_tag, fp in eval_files:
        dataset = opensubs_dataset_name(lang_pair, target_lang)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec, crits = _record_row(
                    r, dataset=dataset, dataset_source="OpenSubs",
                    lang_pair=lang_pair, model_tag=model_tag,
                    src_default="", tgt_default=target_lang,
                )
                rid = rec["record_id"]
                cid_to_layer = opensubs_layer_map.get(rid, {})
                for c in crits:
                    c["layer"] = cid_to_layer.get(c["criterion_id"])
                rec_rows.append(rec)
                crit_rows.extend(crits)

    # ---- Legacy outputs/*.jsonl ----
    for fp, meta in legacy_files:
        dataset = meta["dataset"]
        dataset_source = meta["dataset_source"]
        src_default = meta["source_lang"]
        tgt_default = meta["target_lang"]
        # legacy records lack interpreter — inject from manifest
        interp_from_manifest = meta["interpreter"]
        # Reconstruct a stable lang_pair label like 'id-kor' from source/target
        lp = f"{lang_display(src_default).lower()}-{lang_display(tgt_default).lower()}"
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("interpreter"):
                    r["interpreter"] = interp_from_manifest
                rec, crits = _record_row(
                    r, dataset=dataset, dataset_source=dataset_source,
                    lang_pair=lp, model_tag="legacy",
                    src_default=src_default, tgt_default=tgt_default,
                )
                cid_to_layer = legacy_layer_map.get(
                    (dataset, rec["sample_index"]), {}
                )
                for c in crits:
                    c["layer"] = cid_to_layer.get(c["criterion_id"])
                rec_rows.append(rec)
                crit_rows.extend(crits)

    df = pd.DataFrame(rec_rows)
    df_crit = pd.DataFrame(crit_rows)
    return df, df_crit


# ---------------------------------------------------------------------------
# Unified results-store layout (outputs/results/) — the canonical source.
# Keys models off the reliable `model` slug (NOT the unreliable `interpreter`),
# layers via annotation_clusters (layer_1/2/3 -> L1/L2/L3; unmatched -> None,
# matching the legacy "criterion not in any layer" behaviour).
# ---------------------------------------------------------------------------

T3_TO_T2 = {"arb": "ar", "ben": "bn", "ind": "id", "kor": "ko", "eng": "en"}
LAYER_SHORT = {"layer_1": "L1", "layer_2": "L2", "layer_3": "L3"}


def _store_lang_pair(src: str, tgt: str) -> str:
    a, b = T3_TO_T2.get(src, src), T3_TO_T2.get(tgt, tgt)
    return "-".join(sorted([a, b]))


def build_dataframes_from_store(
    results_dir: Path,
    clusters_dir: Path,
    include_synthetic: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build (df, df_crit) from outputs/results/by_model/*.jsonl.

    All records are OpenSubs-sourced. ``model`` is the canonical slug's display
    label; ``synthetic_fill`` translation-failure records are included by default
    (legitimate 0-score failures)."""
    cmap = load_cluster_layer_map(Path(clusters_dir))
    by_model = Path(results_dir) / "by_model"
    rec_rows: List[dict] = []
    crit_rows: List[dict] = []
    for fp in sorted(by_model.glob("*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (not include_synthetic) and r.get("synthetic_fill"):
                    continue
                src = r.get("source_lang") or ""
                tgt = r.get("target_lang") or ""
                lp = _store_lang_pair(src, tgt)
                dataset = opensubs_dataset_name(lp, tgt)
                rec, crits = _record_row(
                    r, dataset=dataset, dataset_source="OpenSubs",
                    lang_pair=lp, model_tag="store",
                    src_default=src, tgt_default=tgt,
                )
                model = model_label_from_slug(r.get("model"))
                rec["model"] = model
                seg = rec["record_id"].split("_")[0]
                crit_map = cmap.get((src, tgt, seg), {})
                for c in crits:
                    c["model"] = model
                    hit = crit_map.get(normalize_criteria(c["criteria_text"]))
                    c["layer"] = LAYER_SHORT.get(hit[0]) if hit else None
                rec_rows.append(rec)
                crit_rows.extend(crits)
    return pd.DataFrame(rec_rows), pd.DataFrame(crit_rows)


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def _agg_with_ci(values: pd.Series, n_boot: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").dropna().values
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr)) if n else float("nan")
    lo_w, hi_w = wilson_ci(mean * n, n) if n else (float("nan"), float("nan"))
    if n:
        _m, lo_b, hi_b = bootstrap_mean_ci(arr, n_boot=n_boot)
    else:
        lo_b = hi_b = float("nan")
    return pd.Series({
        "n": n,
        "mean": mean,
        "std": std,
        "median": median,
        "wilson_lo": lo_w,
        "wilson_hi": hi_w,
        "boot_lo": lo_b,
        "boot_hi": hi_b,
    })


def summarise(df: pd.DataFrame, group_cols: List[str], rate_col: str, n_boot: int) -> pd.DataFrame:
    rows: List[dict] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = _agg_with_ci(grp[rate_col], n_boot=n_boot)
        rows.append({**dict(zip(group_cols, keys)), **s.to_dict()})
    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    if "n" in out.columns:
        out["n"] = out["n"].astype(int)
    return out


def lid_failure_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(["dataset_source", "direction", "model"], dropna=False):
        n = len(grp)
        lid_fail = (~grp["lid_passed"].fillna(True).astype(bool)).sum()
        tlc_fail = (~grp["translation_lid_correct"].fillna(True).astype(bool)).sum()
        rlc_fail = (~grp["response_lid_correct"].fillna(True).astype(bool)).sum()
        lo, hi = wilson_ci(lid_fail, n)
        rows.append({
            "dataset_source": keys[0],
            "direction": keys[1],
            "model": keys[2],
            "n": n,
            "lid_fail": int(lid_fail),
            "lid_fail_rate": float(lid_fail) / n if n else float("nan"),
            "lid_fail_wilson_lo": lo,
            "lid_fail_wilson_hi": hi,
            "translation_lid_fail": int(tlc_fail),
            "translation_lid_fail_rate": float(tlc_fail) / n if n else float("nan"),
            "response_lid_fail": int(rlc_fail),
            "response_lid_fail_rate": float(rlc_fail) / n if n else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_source", "direction", "model"]
    ).reset_index(drop=True)


def layer_pass_rates(df_crit: pd.DataFrame) -> pd.DataFrame:
    sub = df_crit[df_crit["layer"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["met_int"] = sub["met"].astype(int)
    rows = []
    for keys, grp in sub.groupby(
        ["dataset_source", "direction", "model", "layer"], dropna=False
    ):
        k = grp["met_int"].sum()
        n = len(grp)
        lo, hi = wilson_ci(k, n)
        rows.append({
            "dataset_source": keys[0],
            "direction": keys[1],
            "model": keys[2],
            "layer": keys[3],
            "n_criteria": int(n),
            "n_met": int(k),
            "pass_rate": float(k) / n if n else float("nan"),
            "wilson_lo": lo,
            "wilson_hi": hi,
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_source", "direction", "model", "layer"]
    ).reset_index(drop=True)


def model_pairwise_tests(df: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["dataset_source", "direction"], dropna=False):
        models = sorted(sub["model"].unique())
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                a = pd.to_numeric(sub.loc[sub["model"] == m1, rate_col],
                                  errors="coerce").dropna()
                b = pd.to_numeric(sub.loc[sub["model"] == m2, rate_col],
                                  errors="coerce").dropna()
                if len(a) == 0 or len(b) == 0:
                    continue
                z, p = two_proportion_ztest(a.sum(), len(a), b.sum(), len(b))
                rows.append({
                    "rate": rate_col,
                    "dataset_source": keys[0],
                    "direction": keys[1],
                    "model_a": m1, "n_a": int(len(a)), "mean_a": float(a.mean()),
                    "model_b": m2, "n_b": int(len(b)), "mean_b": float(b.mean()),
                    "delta": float(a.mean() - b.mean()),
                    "z": z, "p_value": p,
                })
    return pd.DataFrame(rows)


def model_overall_summary(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """One row per model with strict & lenient means + LID stats — pooled
    across ALL records."""
    rows = []
    for m, grp in df.groupby("model"):
        n = len(grp)
        s_arr = pd.to_numeric(grp["success_rate_strict"], errors="coerce").dropna()
        l_arr = pd.to_numeric(grp["success_rate_lenient"], errors="coerce").dropna()
        s_mean, s_lo, s_hi = bootstrap_mean_ci(s_arr.values, n_boot=n_boot)
        l_mean, l_lo, l_hi = bootstrap_mean_ci(l_arr.values, n_boot=n_boot)
        lid_fail = (~grp["lid_passed"].fillna(True).astype(bool)).sum()
        rows.append({
            "model": m,
            "n_records": n,
            "strict_mean": s_mean,
            "strict_boot_lo": s_lo,
            "strict_boot_hi": s_hi,
            "lenient_mean": l_mean,
            "lenient_boot_lo": l_lo,
            "lenient_boot_hi": l_hi,
            "lid_fail_rate": float(lid_fail) / n if n else float("nan"),
            "n_lid_fail": int(lid_fail),
        })
    out = pd.DataFrame(rows).sort_values("strict_mean", ascending=False).reset_index(drop=True)
    return out


def pairwise_grouped(df: pd.DataFrame, rate_col: str, group_col: str,
                     within_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Pairwise two-proportion z-tests on per-record ``rate_col``, comparing
    levels of ``group_col``. If ``within_cols`` is given, the tests are
    performed within each unique combination of those (e.g. per-model)."""
    rows = []
    if within_cols:
        outer_groups = df.groupby(within_cols, dropna=False)
    else:
        outer_groups = [(("__all__",), df)]
    for outer_key, sub in outer_groups:
        if not isinstance(outer_key, tuple):
            outer_key = (outer_key,)
        levels = sorted(sub[group_col].dropna().unique())
        for i, a_lvl in enumerate(levels):
            for b_lvl in levels[i + 1:]:
                a = pd.to_numeric(sub.loc[sub[group_col] == a_lvl, rate_col],
                                  errors="coerce").dropna()
                b = pd.to_numeric(sub.loc[sub[group_col] == b_lvl, rate_col],
                                  errors="coerce").dropna()
                if len(a) == 0 or len(b) == 0:
                    continue
                z, p = two_proportion_ztest(a.sum(), len(a), b.sum(), len(b))
                row = {
                    "rate": rate_col,
                    "compare": group_col,
                    "level_a": a_lvl, "n_a": int(len(a)), "mean_a": float(a.mean()),
                    "level_b": b_lvl, "n_b": int(len(b)), "mean_b": float(b.mean()),
                    "delta": float(a.mean() - b.mean()),
                    "z": z, "p_value": p,
                }
                if within_cols:
                    for col, val in zip(within_cols, outer_key):
                        row[col] = val
                rows.append(row)
    return pd.DataFrame(rows)


def layer_pairwise_tests(df_crit: pd.DataFrame) -> pd.DataFrame:
    """Layer comparisons (L1 vs L2 vs L3) within each
    ``(dataset_source, model)``. Each criterion is a Bernoulli observation."""
    sub = df_crit[df_crit["layer"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["met_int"] = sub["met"].astype(int)
    rows = []
    for keys, grp in sub.groupby(["dataset_source", "model"], dropna=False):
        layers = [l for l in LAYER_ORDER if l in grp["layer"].unique()]
        for i, la in enumerate(layers):
            for lb in layers[i + 1:]:
                a = grp.loc[grp["layer"] == la, "met_int"]
                b = grp.loc[grp["layer"] == lb, "met_int"]
                if len(a) == 0 or len(b) == 0:
                    continue
                z, p = two_proportion_ztest(a.sum(), len(a), b.sum(), len(b))
                rows.append({
                    "dataset_source": keys[0],
                    "model": keys[1],
                    "layer_a": la, "n_a": int(len(a)),
                    "pass_a": float(a.mean()),
                    "layer_b": lb, "n_b": int(len(b)),
                    "pass_b": float(b.mean()),
                    "delta": float(a.mean() - b.mean()),
                    "z": z, "p_value": p,
                })
    return pd.DataFrame(rows)


def collect_significant_effects(
    pairwise_models: pd.DataFrame,
    pairwise_models_overall: pd.DataFrame,
    pairwise_target_lang: pd.DataFrame,
    pairwise_source_lang: pd.DataFrame,
    layer_pairs: pd.DataFrame,
    dataset_compare: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Stack every pairwise test into a single, filtered ``p < alpha`` table."""
    frames = []

    if not pairwise_models.empty:
        f = pairwise_models[pairwise_models["p_value"] < alpha].copy()
        f["family"] = "model_within_direction"
        frames.append(f)
    if not pairwise_models_overall.empty:
        f = pairwise_models_overall[pairwise_models_overall["p_value"] < alpha].copy()
        f["family"] = "model_overall"
        frames.append(f)
    if not pairwise_target_lang.empty:
        f = pairwise_target_lang[pairwise_target_lang["p_value"] < alpha].copy()
        f["family"] = "target_lang_within_model"
        frames.append(f)
    if not pairwise_source_lang.empty:
        f = pairwise_source_lang[pairwise_source_lang["p_value"] < alpha].copy()
        f["family"] = "source_lang_within_model"
        frames.append(f)
    if not layer_pairs.empty:
        f = layer_pairs[layer_pairs["p_value"] < alpha].copy()
        f["family"] = "layer_within_dataset_model"
        frames.append(f)
    if not dataset_compare.empty:
        f = dataset_compare[dataset_compare["p_value"] < alpha].copy()
        f["family"] = "dataset_MAPS_vs_OpenSubs"
        frames.append(f)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["family", "p_value"]).reset_index(drop=True)


def dataset_compare(df: pd.DataFrame, rate_col: str, n_boot: int) -> pd.DataFrame:
    """For each (direction, model) present in BOTH MAPS and OpenSubs,
    compute means + a two-proportion z-test."""
    rows = []
    for keys, sub in df.groupby(["direction", "model"], dropna=False):
        sources = set(sub["dataset_source"].unique())
        if not {"MAPS", "OpenSubs"}.issubset(sources):
            continue
        a = pd.to_numeric(
            sub.loc[sub["dataset_source"] == "MAPS", rate_col],
            errors="coerce").dropna()
        b = pd.to_numeric(
            sub.loc[sub["dataset_source"] == "OpenSubs", rate_col],
            errors="coerce").dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        z, p = two_proportion_ztest(a.sum(), len(a), b.sum(), len(b))
        _m1, lo1, hi1 = bootstrap_mean_ci(a.values, n_boot=n_boot)
        _m2, lo2, hi2 = bootstrap_mean_ci(b.values, n_boot=n_boot)
        rows.append({
            "rate": rate_col,
            "direction": keys[0],
            "model": keys[1],
            "n_maps": int(len(a)), "mean_maps": float(a.mean()),
            "boot_lo_maps": lo1, "boot_hi_maps": hi1,
            "n_opensubs": int(len(b)), "mean_opensubs": float(b.mean()),
            "boot_lo_opensubs": lo2, "boot_hi_opensubs": hi2,
            "delta_maps_minus_opensubs": float(a.mean() - b.mean()),
            "z": z, "p_value": p,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["direction", "model"]).reset_index(drop=True)


def distribution_buckets(df: pd.DataFrame, rate_col: str,
                         group_cols: List[str]) -> pd.DataFrame:
    """Bucketed distribution of a per-record rate."""
    work = df[group_cols + [rate_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()
    work["bucket"] = pd.cut(
        work[rate_col], bins=BUCKET_EDGES, labels=BUCKET_LABELS,
        include_lowest=True,
    )
    counts = (
        work.groupby(group_cols + ["bucket"], dropna=False, observed=False)
        .size().reset_index(name="n")
    )
    totals = (
        work.groupby(group_cols, dropna=False).size().reset_index(name="total")
    )
    out = counts.merge(totals, on=group_cols, how="left")
    out["share"] = out["n"] / out["total"]
    return out.sort_values(group_cols + ["bucket"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Failure / differentiating cases
# ---------------------------------------------------------------------------

def find_failure_and_diff_cases(
    df: pd.DataFrame, rate_col: str, min_models: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = df.pivot_table(
        index=["dataset", "lang_pair", "direction", "record_id", "sample_index"],
        columns="model", values=rate_col, aggfunc="first",
    ).reset_index()
    model_cols = [c for c in pivot.columns
                  if c not in ("dataset", "lang_pair", "direction",
                                "record_id", "sample_index")]
    if not model_cols:
        return pd.DataFrame(), pd.DataFrame()

    n_per_row = pivot[model_cols].notna().sum(axis=1)
    qualified = pivot[n_per_row >= min_models].copy()
    failed_mask = (qualified[model_cols].fillna(1) < 0.5).all(axis=1)
    all_failed = qualified[failed_mask].copy()

    spread = qualified[model_cols].max(axis=1) - qualified[model_cols].min(axis=1)
    diff_mask = (
        (qualified[model_cols].fillna(0).max(axis=1) >= 0.7)
        & (qualified[model_cols].fillna(1).min(axis=1) < 0.4)
    )
    diff_cases = qualified[diff_mask].copy()
    diff_cases["score_range"] = spread[diff_mask].values
    diff_cases = diff_cases.sort_values("score_range", ascending=False)
    return all_failed, diff_cases


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _setup_plot_theme() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
    })


def _model_palette(models: List[str]) -> Dict[str, str]:
    fallback = sns.color_palette("Set2", n_colors=max(len(models), 1)).as_hex()
    return {m: MODEL_COLORS.get(m, fallback[i]) for i, m in enumerate(models)}


def _ordered_models(df: pd.DataFrame) -> List[str]:
    present = list(df["model"].dropna().unique())
    ordered = [m for m in MODEL_ORDER if m in present]
    return ordered + [m for m in present if m not in ordered]


def _ordered_directions(df: pd.DataFrame) -> List[str]:
    return sorted(df["direction"].dropna().unique())


def _ordered_dataset_sources(df: pd.DataFrame) -> List[str]:
    present = list(df["dataset_source"].dropna().unique())
    ordered = [s for s in DATASET_SOURCE_ORDER if s in present]
    return ordered + [s for s in present if s not in ordered]


def _grid_dims(n: int, max_cols: int = 4) -> Tuple[int, int]:
    if n <= 0:
        return (1, 1)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    return rows, cols


# ---------------------------------------------------------------------------
# Plots — means by direction × model, faceted by dataset_source
# ---------------------------------------------------------------------------

def _draw_means_axes(ax, sub: pd.DataFrame, models: List[str],
                     palette: Dict[str, str], rate_col: str, title: str) -> None:
    directions = sorted(sub["direction"].dropna().unique())
    x = np.arange(len(directions))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals, ns = [], []
        for d in directions:
            col = sub.loc[(sub["direction"] == d) & (sub["model"] == m), rate_col]
            col = pd.to_numeric(col, errors="coerce").dropna()
            vals.append(col.mean() if len(col) else np.nan)
            ns.append(len(col))
        offsets = x + (i - (len(models) - 1) / 2) * width
        ax.bar(offsets, vals, width, color=palette[m], alpha=0.88, label=m)
        for off, v, n in zip(offsets, vals, ns):
            if not np.isnan(v):
                ax.text(off, v + 0.012, f"{v:.2f}\n(n={n})",
                        ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(directions, fontsize=9.5)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(0.5, color="grey", ls="--", lw=0.7, alpha=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Mean success rate")


def plot_means_per_source(df: pd.DataFrame, rate_col: str, out_dir: Path,
                          prefix: str, base_title: str) -> List[Path]:
    """One figure per dataset_source."""
    saved: List[Path] = []
    sources = _ordered_dataset_sources(df)
    if not sources:
        return saved
    models = _ordered_models(df)
    palette = _model_palette(models)
    for src in sources:
        sub = df[df["dataset_source"] == src]
        directions = sorted(sub["direction"].dropna().unique())
        if not directions:
            continue
        width = max(6.0, 1.6 * len(directions) + 1.5)
        fig, ax = plt.subplots(figsize=(width, 5.0))
        _draw_means_axes(ax, sub, models, palette, rate_col,
                         title=f"{src} — {base_title}")
        ax.legend(handles=[Patch(color=palette[m], alpha=0.88, label=m)
                            for m in models],
                  title="Model", loc="lower right", fontsize=9)
        plt.tight_layout()
        path = out_dir / f"{prefix}_{src.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Plot — strict vs lenient grouped (per direction, faceted dataset_source)
# ---------------------------------------------------------------------------

def plot_lid_penalty_facet(df: pd.DataFrame, out_path: Path) -> None:
    sources = _ordered_dataset_sources(df)
    if not sources:
        return
    models = _ordered_models(df)
    palette = _model_palette(models)

    fig, axes = plt.subplots(
        len(sources), 1, figsize=(11, max(3.6, 3.0 * len(sources))),
        sharey=True, squeeze=False,
    )
    axes = axes.flatten()
    for ax, src in zip(axes, sources):
        sub = df[df["dataset_source"] == src]
        directions = sorted(sub["direction"].dropna().unique())
        if not directions:
            ax.set_visible(False)
            continue
        x = np.arange(len(directions))
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            vals = []
            for d in directions:
                a = pd.to_numeric(
                    sub.loc[(sub["direction"] == d) & (sub["model"] == m),
                             "success_rate_lenient"], errors="coerce").dropna()
                b = pd.to_numeric(
                    sub.loc[(sub["direction"] == d) & (sub["model"] == m),
                             "success_rate_strict"], errors="coerce").dropna()
                if len(a) and len(b):
                    vals.append(float(a.mean() - b.mean()))
                else:
                    vals.append(np.nan)
            offsets = x + (i - (len(models) - 1) / 2) * width
            ax.bar(offsets, vals, width, color=palette[m], alpha=0.88, label=m)
            for off, v in zip(offsets, vals):
                if not np.isnan(v):
                    ax.text(off, v + 0.003, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(directions, fontsize=9)
        ax.set_title(f"{src} — Lenient minus Strict (LID penalty)")
        ax.set_ylabel("Δ rate")
    handles = [Patch(color=palette[m], alpha=0.88, label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — distribution violins (small multiples, one panel per direction)
# ---------------------------------------------------------------------------

def plot_distribution_grid(df: pd.DataFrame, rate_col: str, out_path: Path,
                           title: str) -> None:
    directions = _ordered_directions(df)
    if not directions:
        return
    models = _ordered_models(df)
    palette = _model_palette(models)
    rows, cols = _grid_dims(len(directions), max_cols=4)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[len(directions):]:
        ax.set_visible(False)
    for ax, d in zip(axes_flat[:len(directions)], directions):
        data = []
        used_models = []
        for m in models:
            v = pd.to_numeric(
                df.loc[(df["direction"] == d) & (df["model"] == m), rate_col],
                errors="coerce").dropna().values
            if len(v):
                data.append(v)
                used_models.append(m)
        if not data:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(data, positions=range(len(used_models)),
                              showmedians=True, showextrema=True)
        for pc, m in zip(parts["bodies"], used_models):
            pc.set_facecolor(palette[m])
            pc.set_alpha(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.3)
        ns = [len(x) for x in data]
        short_models = [m.replace(" ", "\n") for m in used_models]
        ax.set_xticks(range(len(used_models)))
        ax.set_xticklabels(
            [f"{sm}\n(n={n})" for sm, n in zip(short_models, ns)],
            fontsize=7,
        )
        ax.set_ylim(-0.05, 1.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(d, fontsize=10)
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — bucketed score distribution stacked bars
# ---------------------------------------------------------------------------

BUCKET_COLORS = {
    "= 0":        "#B81D13",
    "(0, 0.5]":   "#EE8A18",
    "(0.5, 0.8]": "#EFD415",
    "(0.8, 1)":   "#86C04A",
    "= 1":        "#1B7837",
}


def _draw_bucket_axes(ax, pivot: pd.DataFrame, title: str) -> None:
    bottom = np.zeros(len(pivot.index))
    y = np.arange(len(pivot.index))
    for bucket in BUCKET_LABELS:
        if bucket not in pivot.columns:
            continue
        widths = pivot[bucket].values
        ax.barh(y, widths, left=bottom, color=BUCKET_COLORS[bucket],
                edgecolor="white", linewidth=0.4, label=bucket)
        for yi, w, b in zip(y, widths, bottom):
            if w >= 0.05:
                ax.text(b + w / 2, yi, f"{w:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="white"
                        if bucket in ("= 0", "= 1", "(0, 0.5]") else "black")
        bottom = bottom + widths
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}  /  {m}" for d, m in pivot.index], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.invert_yaxis()
    ax.set_title(title)


def plot_distribution_buckets_per_source(
    buckets_df: pd.DataFrame, out_dir: Path, prefix: str, base_title: str,
) -> List[Path]:
    """One figure per dataset_source. Returns list of saved paths."""
    saved: List[Path] = []
    if buckets_df.empty:
        return saved
    sources = sorted(buckets_df["dataset_source"].dropna().unique(),
                     key=lambda s: DATASET_SOURCE_ORDER.index(s)
                     if s in DATASET_SOURCE_ORDER else 99)
    for src in sources:
        sub = buckets_df[buckets_df["dataset_source"] == src]
        pivot = sub.pivot_table(
            index=["direction", "model"], columns="bucket",
            values="share", aggfunc="first", observed=False,
        ).reindex(columns=BUCKET_LABELS).fillna(0)
        if pivot.empty:
            continue
        fig, ax = plt.subplots(
            figsize=(11, max(3.5, 0.32 * len(pivot.index) + 2))
        )
        _draw_bucket_axes(ax, pivot, f"{src} — {base_title}")
        ax.set_xlabel("Share of records")
        ax.legend(title="Success rate bucket", loc="lower right", fontsize=8,
                  ncol=5, bbox_to_anchor=(1.0, -0.10))
        plt.tight_layout()
        path = out_dir / f"{prefix}_{src.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Plot — layer heatmap (faceted by dataset_source)
# ---------------------------------------------------------------------------

def plot_layer_heatmap_per_source(layer_df: pd.DataFrame, out_dir: Path,
                                  prefix: str) -> List[Path]:
    """One layer-heatmap figure per dataset_source."""
    saved: List[Path] = []
    if layer_df.empty:
        return saved
    sources = sorted(layer_df["dataset_source"].dropna().unique(),
                     key=lambda s: DATASET_SOURCE_ORDER.index(s)
                     if s in DATASET_SOURCE_ORDER else 99)
    for src in sources:
        sub = layer_df[layer_df["dataset_source"] == src]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index=["direction", "model"], columns="layer",
            values="pass_rate", aggfunc="mean",
        ).reindex(columns=LAYER_ORDER)
        n_lookup = sub.pivot_table(
            index=["direction", "model"], columns="layer",
            values="n_criteria", aggfunc="sum",
        ).reindex(columns=LAYER_ORDER)

        annot = pivot.copy().astype(object)
        for r in pivot.index:
            for c in pivot.columns:
                v = pivot.at[r, c]
                n = (n_lookup.at[r, c]
                     if (r in n_lookup.index and c in n_lookup.columns)
                     else None)
                if pd.isna(v):
                    annot.at[r, c] = ""
                else:
                    annot.at[r, c] = (f"{v:.2f}\n(n={int(n)})"
                                      if n and not pd.isna(n) else f"{v:.2f}")
        fig, ax = plt.subplots(
            figsize=(7.0, max(4, 0.42 * len(pivot.index) + 1.5))
        )
        sns.heatmap(
            pivot.astype(float), annot=annot.values, fmt="",
            cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pass rate"}, annot_kws={"fontsize": 8.5},
        )
        ax.set_title(f"{src} — layer pass rate", fontsize=12)
        ax.set_xticklabels([LAYER_LABELS.get(c, c) for c in pivot.columns],
                           rotation=0, fontsize=9.5)
        ax.set_yticklabels([f"{d}  /  {m}" for d, m in pivot.index],
                           rotation=0, fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        path = out_dir / f"{prefix}_{src.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Plot — LID failure rate per dataset_source
# ---------------------------------------------------------------------------

def plot_lid_failure_facet(lid_df: pd.DataFrame, out_path: Path) -> None:
    if lid_df.empty:
        return
    sources = sorted(lid_df["dataset_source"].unique(),
                     key=lambda s: DATASET_SOURCE_ORDER.index(s)
                     if s in DATASET_SOURCE_ORDER else 99)
    models = sorted(lid_df["model"].unique(),
                    key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    palette = _model_palette(models)
    fig, axes = plt.subplots(
        len(sources), 1, figsize=(11, max(3.5, 2.6 * len(sources))),
        sharey=True, squeeze=False,
    )
    axes = axes.flatten()
    for ax, src in zip(axes, sources):
        sub = lid_df[lid_df["dataset_source"] == src]
        directions = sorted(sub["direction"].unique())
        x = np.arange(len(directions))
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            vals, ns = [], []
            for d in directions:
                row = sub[(sub["direction"] == d) & (sub["model"] == m)]
                vals.append(float(row["lid_fail_rate"].iloc[0]) if not row.empty else np.nan)
                ns.append(int(row["n"].iloc[0]) if not row.empty else 0)
            offsets = x + (i - (len(models) - 1) / 2) * width
            ax.bar(offsets, vals, width, color=palette[m], alpha=0.88, label=m)
            for off, v, n in zip(offsets, vals, ns):
                if not np.isnan(v):
                    ax.text(off, v + 0.005, f"{v:.0%}\n(n={n})",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(directions, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f"{src} — GlotLID failure rate")
        ax.set_ylabel("LID failure")
    handles = [Patch(color=palette[m], alpha=0.88, label=m) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — MAPS vs OpenSubs side-by-side for shared (direction, model)
# ---------------------------------------------------------------------------

def plot_dataset_compare(compare_df: pd.DataFrame, out_path: Path,
                         title: str) -> None:
    if compare_df.empty:
        return
    rows = compare_df.copy()
    rows["label"] = rows["direction"] + "  /  " + rows["model"]
    rows = rows.sort_values(["direction", "model"]).reset_index(drop=True)
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(rows) + 1.5)))
    height = 0.4
    for offset, src, color in [
        (-height / 2, "MAPS", DATASET_SOURCE_COLORS["MAPS"]),
        (+height / 2, "OpenSubs", DATASET_SOURCE_COLORS["OpenSubs"]),
    ]:
        col = "mean_maps" if src == "MAPS" else "mean_opensubs"
        n_col = "n_maps" if src == "MAPS" else "n_opensubs"
        lo_col = "boot_lo_maps" if src == "MAPS" else "boot_lo_opensubs"
        hi_col = "boot_hi_maps" if src == "MAPS" else "boot_hi_opensubs"
        means = rows[col].values
        ns = rows[n_col].values
        lo = rows[lo_col].values
        hi = rows[hi_col].values
        ax.barh(y + offset, means, height=height, color=color, alpha=0.88,
                label=src,
                xerr=[means - lo, hi - means], error_kw={"linewidth": 0.9})
        for yi, mv, n in zip(y + offset, means, ns):
            if not np.isnan(mv):
                ax.text(mv + 0.01, yi, f"{mv:.2f}\n(n={int(n)})",
                        va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.invert_yaxis()
    ax.set_xlabel("Mean success rate")
    ax.legend(title="Dataset", loc="lower right")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — per source / target language summary
# ---------------------------------------------------------------------------

def plot_lang_summary(df: pd.DataFrame, lang_col: str, rate_col: str,
                      out_path: Path, title: str) -> None:
    if df.empty:
        return
    models = _ordered_models(df)
    palette = _model_palette(models)
    langs = sorted(df[lang_col].dropna().unique())
    if not langs:
        return
    x = np.arange(len(langs))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(langs) * len(models)), 5))
    for i, m in enumerate(models):
        vals, ns = [], []
        for lng in langs:
            v = pd.to_numeric(
                df.loc[(df[lang_col] == lng) & (df["model"] == m), rate_col],
                errors="coerce").dropna()
            vals.append(v.mean() if len(v) else np.nan)
            ns.append(len(v))
        offsets = x + (i - (len(models) - 1) / 2) * width
        ax.bar(offsets, vals, width, color=palette[m], alpha=0.88, label=m)
        for off, v, n in zip(offsets, vals, ns):
            if not np.isnan(v):
                ax.text(off, v + 0.012, f"{v:.2f}\n(n={n})",
                        ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lang_display(l) for l in langs], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title)
    ax.legend(title="Model")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — category × model
# ---------------------------------------------------------------------------

def plot_category(df: pd.DataFrame, rate_col: str, out_path: Path,
                  title: str) -> None:
    if "category" not in df.columns or df["category"].isna().all():
        return
    cats = sorted(df["category"].dropna().unique())
    if not cats:
        return
    models = _ordered_models(df)
    palette = _model_palette(models)
    x = np.arange(len(cats))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(cats) * len(models)), 5))
    for i, m in enumerate(models):
        vals, ns = [], []
        for c in cats:
            v = pd.to_numeric(df.loc[(df["category"] == c) & (df["model"] == m),
                                       rate_col], errors="coerce").dropna()
            vals.append(v.mean() if len(v) else np.nan)
            ns.append(len(v))
        offsets = x + (i - (len(models) - 1) / 2) * width
        ax.bar(offsets, vals, width, color=palette[m], alpha=0.88, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title)
    ax.legend(title="Model")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _df_table(df: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> str:
    if df.empty:
        return "_no rows_"
    cols = [c for c in df.columns if c not in (drop_cols or [])]
    return tabulate(
        df[cols], headers="keys", tablefmt="github",
        floatfmt=".4f", showindex=False,
    )


def write_report(
    out_dir: Path,
    df: pd.DataFrame,
    df_crit: pd.DataFrame,
    summary_strict: pd.DataFrame,
    summary_lenient: pd.DataFrame,
    by_dir_strict: pd.DataFrame,
    by_dir_lenient: pd.DataFrame,
    by_dir_src_strict: pd.DataFrame,
    by_dir_src_lenient: pd.DataFrame,
    src_lang_stats: pd.DataFrame,
    tgt_lang_stats: pd.DataFrame,
    layer_df: pd.DataFrame,
    lid_df: pd.DataFrame,
    pairwise: pd.DataFrame,
    compare_strict: pd.DataFrame,
    compare_lenient: pd.DataFrame,
    buckets_strict: pd.DataFrame,
    buckets_lenient: pd.DataFrame,
    failure_cases: pd.DataFrame,
    diff_cases: pd.DataFrame,
    criteria_df: pd.DataFrame,
    *,
    model_overall: pd.DataFrame,
    pairwise_model_overall: pd.DataFrame,
    pairwise_target: pd.DataFrame,
    pairwise_source: pd.DataFrame,
    pairwise_layer: pd.DataFrame,
    sig_digest: pd.DataFrame,
) -> None:
    md: List[str] = []
    md.append("# Interpreter Eval — Full Analysis Report")
    md.append("")
    md.append(f"_{len(df)} records · "
              f"{df['dataset_source'].nunique()} dataset sources · "
              f"{df['lang_pair'].nunique()} language pairs · "
              f"{df['direction'].nunique()} directions · "
              f"{df['model'].nunique()} interpreter models · "
              f"{len(df_crit)} criterion observations._")
    md.append("")
    md.append("Two complementary success rates are reported:")
    md.append("")
    md.append("- **Strict** — gated on GlotLID. A failed language check zeroes the record. "
              "This is the original framework behaviour.")
    md.append("- **Lenient** — uses the LLM-judge checklist directly, ignoring LID. "
              "This isolates checklist completion from script/language detection.")
    md.append("")
    md.append("Datasets fall into two sources:")
    md.append("")
    md.append("- **MAPS** — proverb-seeded scenarios from the MAPS-Final spreadsheets.")
    md.append("- **OpenSubs** — auto-augmented scenarios sourced from OpenSubtitles "
              "parallel corpora (combines older `eval_*_maps_from_opensubs_*.jsonl` "
              "runs and the current `outputs/opensubs_eval/<lang_pair>/...` runs).")
    md.append("")

    md.append("## 0. Model-only aggregate (3 rows)")
    md.append("")
    md.append("Pooled across **all** records (every dataset, every direction). "
              "Use this as the headline ranking; the per-direction tables below "
              "show where the gap comes from.")
    md.append("")
    md.append(_df_table(model_overall))
    md.append("")
    if not pairwise_model_overall.empty:
        md.append("**Pairwise model z-tests (strict, pooled):**")
        md.append("")
        md.append(_df_table(pairwise_model_overall))
        md.append("")

    md.append("## 1. Per-direction summary (across datasets)")
    md.append("")
    md.append("**Strict success rate (LID-gated)**")
    md.append("")
    md.append(_df_table(by_dir_strict))
    md.append("")
    md.append("**Lenient success rate (judge only)**")
    md.append("")
    md.append(_df_table(by_dir_lenient))
    md.append("")

    md.append("## 2. Per-direction × dataset_source summary")
    md.append("")
    md.append("**Strict**")
    md.append("")
    md.append(_df_table(by_dir_src_strict))
    md.append("")
    md.append("**Lenient**")
    md.append("")
    md.append(_df_table(by_dir_src_lenient))
    md.append("")

    md.append("## 3. MAPS vs OpenSubs (where both exist)")
    md.append("")
    md.append("Two-proportion z-tests on per-record mean success rates. "
              "Compare same `(direction, model)` across the two dataset sources.")
    md.append("")
    md.append("**Strict**")
    md.append("")
    md.append(_df_table(compare_strict))
    md.append("")
    md.append("**Lenient**")
    md.append("")
    md.append(_df_table(compare_lenient))
    md.append("")

    md.append("## 4. Per-language summaries")
    md.append("")
    md.append("**By source language**")
    md.append("")
    md.append(_df_table(src_lang_stats))
    md.append("")
    md.append("**By target language**")
    md.append("")
    md.append(_df_table(tgt_lang_stats))
    md.append("")
    if not pairwise_target.empty:
        md.append("**Pairwise target-language z-tests (strict, within model):**")
        md.append("")
        md.append(_df_table(pairwise_target))
        md.append("")
    if not pairwise_source.empty:
        md.append("**Pairwise source-language z-tests (strict, within model):**")
        md.append("")
        md.append(_df_table(pairwise_source))
        md.append("")

    md.append("## 5. Per (dataset, lang_pair, direction, model) — strict")
    md.append("")
    md.append(_df_table(summary_strict, drop_cols=["boot_lo", "boot_hi"]))
    md.append("")
    md.append("## 6. Per (dataset, lang_pair, direction, model) — lenient")
    md.append("")
    md.append(_df_table(summary_lenient, drop_cols=["boot_lo", "boot_hi"]))
    md.append("")

    md.append("## 7. Layer pass rates (per dataset_source × direction × model)")
    md.append("")
    md.append("L1 = Semantic Core, L2 = Pragmatic Function, L3 = Cultural / Social "
              "Constraints. Pass rates are computed over individual criteria.")
    md.append("")
    md.append(_df_table(layer_df))
    md.append("")
    if not pairwise_layer.empty:
        md.append("**Pairwise layer z-tests (within dataset_source × model):**")
        md.append("")
        md.append(_df_table(pairwise_layer))
        md.append("")

    md.append("## 8. Success rate distribution buckets")
    md.append("")
    md.append("Per-record success rate falls into one of "
              "`= 0`, `(0, 0.5]`, `(0.5, 0.8]`, `(0.8, 1)`, `= 1`.")
    md.append("")
    md.append("**Strict**")
    md.append("")
    md.append(_df_table(buckets_strict))
    md.append("")
    md.append("**Lenient**")
    md.append("")
    md.append(_df_table(buckets_lenient))
    md.append("")

    md.append("## 9. GlotLID failure rates (per dataset_source × direction × model)")
    md.append("")
    md.append(_df_table(lid_df,
                         drop_cols=["lid_fail_wilson_lo", "lid_fail_wilson_hi"]))
    md.append("")

    md.append("## 10. Pairwise model comparisons")
    md.append("")
    md.append("Two-proportion z-tests on per-record mean strict success rate, "
              "within each `(dataset_source, direction)`. p-values are uncorrected.")
    md.append("")
    md.append(_df_table(pairwise))
    md.append("")

    md.append("## 11. Hardest and easiest criteria")
    md.append("")
    if criteria_df.empty:
        md.append("_no criteria data._")
    else:
        md.append("**Hardest 10 (lowest pass rate, n ≥ 3)**")
        md.append("")
        md.append(_df_table(
            criteria_df.head(10)[["direction", "layer", "n", "pass_rate", "criteria_text"]]
        ))
        md.append("")
        md.append("**Easiest 10 (highest pass rate, n ≥ 3)**")
        md.append("")
        md.append(_df_table(
            criteria_df.tail(10)[["direction", "layer", "n", "pass_rate", "criteria_text"]]
        ))
    md.append("")

    md.append("## 12. Failure / differentiating cases")
    md.append("")
    md.append(f"- **{len(failure_cases)}** records where every model scored "
              f"strict < 0.5 (`failure_cases.csv`).")
    md.append(f"- **{len(diff_cases)}** records where the best model scored "
              f"≥ 0.7 *and* the worst < 0.4 — these are the most informative "
              f"for model comparison (`differentiating_cases.csv`).")
    md.append("")

    md.append("## 13. Significant effects (p < 0.05)")
    md.append("")
    md.append("Stacked filtered view of every pairwise z-test computed in the "
              "report. Families: model_overall, model_within_direction, "
              "target_lang_within_model, source_lang_within_model, "
              "layer_within_dataset_model, dataset_MAPS_vs_OpenSubs.")
    md.append("")
    if sig_digest.empty:
        md.append("_no effects reach p < 0.05._")
    else:
        md.append(_df_table(sig_digest))
    md.append("")

    md.append("## 14. Plots")
    md.append("")
    md.append("| File | Description |")
    md.append("|------|-------------|")
    md.append("| fig01_mean_strict_maps.png / fig01_mean_strict_opensubs.png | Mean strict success rate per direction × model — separate file per dataset source |")
    md.append("| fig02_mean_lenient_maps.png / fig02_mean_lenient_opensubs.png | Mean lenient success rate, separate per dataset source |")
    md.append("| fig03_lid_penalty.png | Lenient − Strict (LID gating cost) per direction × model |")
    md.append("| fig04_distribution_strict.png | Strict success rate violins, small multiples per direction |")
    md.append("| fig05_distribution_lenient.png | Lenient success rate violins, small multiples per direction |")
    md.append("| fig06_buckets_strict_maps.png / fig06_buckets_strict_opensubs.png | Strict score distribution buckets, separate per dataset source |")
    md.append("| fig07_buckets_lenient_maps.png / fig07_buckets_lenient_opensubs.png | Lenient score distribution buckets, separate per dataset source |")
    md.append("| fig08_layer_heatmap_maps.png / fig08_layer_heatmap_opensubs.png | Layer (L1/L2/L3) pass rate heatmap, separate per dataset source |")
    md.append("| fig09_lid_failure.png | GlotLID failure rate per direction × model, faceted by dataset_source |")
    md.append("| fig10_dataset_compare_strict.png | MAPS vs OpenSubs strict success per (direction, model) |")
    md.append("| fig11_dataset_compare_lenient.png | MAPS vs OpenSubs lenient success per (direction, model) |")
    md.append("| fig12_source_lang_strict.png | Mean strict success grouped by source language × model |")
    md.append("| fig13_target_lang_strict.png | Mean strict success grouped by target language × model |")
    md.append("| fig14_source_lang_lenient.png | Same, lenient |")
    md.append("| fig15_target_lang_lenient.png | Same, lenient |")
    md.append("| fig16_category_strict.png | Strict success per category × model |")
    md.append("| fig17_category_lenient.png | Lenient success per category × model |")
    md.append("")

    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Loading evaluation files…")
    print("=" * 70)

    # ---- Unified store source (default) ----
    if args.source == "store":
        global MODEL_ORDER, MODEL_COLORS
        MODEL_ORDER = [MODEL_SLUG_LABEL[s] for s in MODEL_SLUG_ORDER]
        MODEL_COLORS = {MODEL_SLUG_LABEL[s]: MODEL_SLUG_COLORS[s]
                        for s in MODEL_SLUG_ORDER}
        print(f"\n[results store]  {args.results_dir}")
        df, df_crit = build_dataframes_from_store(
            Path(args.results_dir), Path(args.clusters_dir),
            include_synthetic=not args.no_synthetic,
        )
        print(f"\n{len(df)} records loaded · {len(df_crit)} criterion "
              f"observations · {df['model'].nunique()} models.")
        if not df_crit.empty and df_crit["layer"].notna().any():
            cov = df_crit["layer"].notna().mean()
            print(f"  Layer coverage: {cov:.1%} of criteria mapped to a layer.")
        _run_analysis(args, out_dir, df, df_crit)
        return

    eval_files = discover_eval_files(Path(args.opensubs_eval_dir))
    if eval_files:
        print(f"\n[opensubs_eval]  {len(eval_files)} files")
        for lp, tgt, tag, fp in eval_files:
            print(f"  {lp:<8s}  target={tgt}  tag={tag:<12s}  {fp}")

    legacy_files: List[Tuple[str, Dict[str, str]]] = []
    if not args.skip_legacy:
        legacy_files = discover_legacy_eval_files(Path(args.legacy_dir))
        if legacy_files:
            print(f"\n[legacy outputs]  {len(legacy_files)} files")
            for fp, meta in legacy_files:
                print(f"  {meta['dataset']:<25s}  "
                      f"{meta['source_lang']}->{meta['target_lang']}  "
                      f"interp={meta['interpreter']}  {fp}")

    if not eval_files and not legacy_files:
        print("No files found in either source.")
        return

    print("\nLoading layer maps…")
    opensubs_layer_map = load_layer_map(Path(args.augmented_dir))
    print(f"  opensubs_eval: {len(opensubs_layer_map)} record_ids indexed.")
    legacy_layer_map = load_legacy_layer_map(Path(args.enriched_dir))
    print(f"  legacy: {len(legacy_layer_map)} (dataset, sample_index) keys indexed.")

    df, df_crit = build_dataframes(
        eval_files, opensubs_layer_map, legacy_files, legacy_layer_map,
    )
    print(f"\n{len(df)} records loaded · {len(df_crit)} criterion observations.")
    if df_crit["layer"].notna().any():
        cov = df_crit["layer"].notna().mean()
        print(f"  Layer coverage: {cov:.1%} of criteria mapped to a layer.")
    else:
        print("  WARNING: 0 criteria mapped to a layer.")

    _run_analysis(args, out_dir, df, df_crit)


def _run_analysis(args, out_dir: Path, df: pd.DataFrame,
                  df_crit: pd.DataFrame) -> None:
    # Only the new opensubs_eval files are expected to carry a record_id;
    # legacy outputs used sample_index instead. Restrict the sanity check
    # to non-legacy rows so we don't false-alarm on legacy data.
    bad_rids = df[
        df["model_tag"].ne("legacy")
        & df["record_id"].apply(lambda x: parse_record_id(x) is None)
    ]
    if not bad_rids.empty:
        print(f"  WARNING: {len(bad_rids)} new-format records have malformed "
              f"record_id (first: {bad_rids['record_id'].iloc[0]!r}).")

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Computing summary statistics…")
    print("=" * 70)
    detail_cols = ["dataset_source", "dataset", "lang_pair", "direction", "model"]
    summary_strict = summarise(df, detail_cols, "success_rate_strict",
                               n_boot=args.n_boot)
    summary_lenient = summarise(df, detail_cols, "success_rate_lenient",
                                n_boot=args.n_boot)
    summary_strict.to_csv(out_dir / "summary_strict.csv", index=False)
    summary_lenient.to_csv(out_dir / "summary_lenient.csv", index=False)

    by_dir_strict = summarise(df, ["direction", "model"], "success_rate_strict",
                              n_boot=args.n_boot)
    by_dir_lenient = summarise(df, ["direction", "model"], "success_rate_lenient",
                               n_boot=args.n_boot)
    by_dir_strict.to_csv(out_dir / "direction_strict.csv", index=False)
    by_dir_lenient.to_csv(out_dir / "direction_lenient.csv", index=False)

    by_dir_src_strict = summarise(
        df, ["dataset_source", "direction", "model"], "success_rate_strict",
        n_boot=args.n_boot,
    )
    by_dir_src_lenient = summarise(
        df, ["dataset_source", "direction", "model"], "success_rate_lenient",
        n_boot=args.n_boot,
    )
    by_dir_src_strict.to_csv(out_dir / "direction_source_strict.csv", index=False)
    by_dir_src_lenient.to_csv(out_dir / "direction_source_lenient.csv", index=False)

    src_lang_stats = summarise(df, ["source_lang", "model"], "success_rate_strict",
                               n_boot=args.n_boot)
    tgt_lang_stats = summarise(df, ["target_lang", "model"], "success_rate_strict",
                               n_boot=args.n_boot)
    src_lang_stats.to_csv(out_dir / "source_lang_stats.csv", index=False)
    tgt_lang_stats.to_csv(out_dir / "target_lang_stats.csv", index=False)

    print("\nDirection × model — STRICT")
    print(_df_table(by_dir_strict))
    print("\nDirection × model — LENIENT")
    print(_df_table(by_dir_lenient))

    # ------------------------------------------------------------------
    # Layer / LID / pairwise / criteria / dataset-compare / buckets
    # ------------------------------------------------------------------
    layer_df = layer_pass_rates(df_crit)
    layer_df.to_csv(out_dir / "layer_stats.csv", index=False)
    if not layer_df.empty:
        print("\nLayer pass rates (top 15 rows):")
        print(_df_table(layer_df.head(15)))

    lid_df = lid_failure_stats(df)
    lid_df.to_csv(out_dir / "lid_failure_stats.csv", index=False)
    print("\nLID failure rates (top 20 rows):")
    print(_df_table(lid_df.head(20)))

    pairwise = model_pairwise_tests(df, "success_rate_strict")
    pairwise.to_csv(out_dir / "model_pairwise_tests.csv", index=False)

    # Model-only aggregate (3 rows, pooled across everything)
    model_overall_strict = model_overall_summary(df, n_boot=args.n_boot)
    model_overall_strict.to_csv(out_dir / "model_overall.csv", index=False)
    print("\nModel-only aggregate (pooled across all data):")
    print(_df_table(model_overall_strict))

    # Pairwise model tests pooled across everything
    pairwise_model_overall = pairwise_grouped(
        df, "success_rate_strict", group_col="model")
    pairwise_model_overall_lenient = pairwise_grouped(
        df, "success_rate_lenient", group_col="model")
    pairwise_model_overall.to_csv(
        out_dir / "model_overall_pairwise_strict.csv", index=False)
    pairwise_model_overall_lenient.to_csv(
        out_dir / "model_overall_pairwise_lenient.csv", index=False)

    # Pairwise target language tests, within each model
    pairwise_target = pairwise_grouped(
        df, "success_rate_strict", group_col="target_lang",
        within_cols=["model"])
    pairwise_target_lenient = pairwise_grouped(
        df, "success_rate_lenient", group_col="target_lang",
        within_cols=["model"])
    pairwise_target.to_csv(out_dir / "target_lang_pairwise_strict.csv", index=False)
    pairwise_target_lenient.to_csv(
        out_dir / "target_lang_pairwise_lenient.csv", index=False)

    # Pairwise source language tests, within each model
    pairwise_source = pairwise_grouped(
        df, "success_rate_strict", group_col="source_lang",
        within_cols=["model"])
    pairwise_source_lenient = pairwise_grouped(
        df, "success_rate_lenient", group_col="source_lang",
        within_cols=["model"])
    pairwise_source.to_csv(out_dir / "source_lang_pairwise_strict.csv", index=False)
    pairwise_source_lenient.to_csv(
        out_dir / "source_lang_pairwise_lenient.csv", index=False)

    # Pairwise layer tests, within each (dataset_source, model)
    pairwise_layer = layer_pairwise_tests(df_crit)
    pairwise_layer.to_csv(out_dir / "layer_pairwise.csv", index=False)

    compare_strict = dataset_compare(df, "success_rate_strict", n_boot=args.n_boot)
    compare_lenient = dataset_compare(df, "success_rate_lenient", n_boot=args.n_boot)
    compare_strict.to_csv(out_dir / "dataset_compare_strict.csv", index=False)
    compare_lenient.to_csv(out_dir / "dataset_compare_lenient.csv", index=False)
    if not compare_strict.empty:
        print("\nMAPS vs OpenSubs (strict) — directions present in both:")
        print(_df_table(compare_strict))

    buckets_strict = distribution_buckets(
        df, "success_rate_strict", ["dataset_source", "direction", "model"])
    buckets_lenient = distribution_buckets(
        df, "success_rate_lenient", ["dataset_source", "direction", "model"])
    buckets_strict.to_csv(out_dir / "distribution_buckets_strict.csv", index=False)
    buckets_lenient.to_csv(out_dir / "distribution_buckets_lenient.csv", index=False)

    if not df_crit.empty:
        sub = df_crit.dropna(subset=["criteria_text"]).copy()
        sub["met_int"] = sub["met"].astype(int)
        crit_agg = (
            sub.groupby(["direction", "layer", "criteria_text"], dropna=False)["met_int"]
            .agg(n="count", pass_rate="mean").reset_index()
        )
        crit_agg = crit_agg[crit_agg["n"] >= 3].sort_values("pass_rate")
        crit_agg.to_csv(out_dir / "criteria_pass_rate.csv", index=False)
    else:
        crit_agg = pd.DataFrame()

    failure_cases, diff_cases = find_failure_and_diff_cases(
        df, "success_rate_strict", min_models=args.min_models_per_record,
    )
    failure_cases.to_csv(out_dir / "failure_cases.csv", index=False)
    diff_cases.to_csv(out_dir / "differentiating_cases.csv", index=False)
    print(f"\nFailure cases (strict<0.5 for all models): {len(failure_cases)}")
    print(f"Differentiating cases (best≥0.7 ∧ worst<0.4): {len(diff_cases)}")

    # Consolidated significant-effects digest (p < 0.05)
    sig_digest = collect_significant_effects(
        pairwise_models=pairwise,
        pairwise_models_overall=pairwise_model_overall,
        pairwise_target_lang=pairwise_target,
        pairwise_source_lang=pairwise_source,
        layer_pairs=pairwise_layer,
        dataset_compare=compare_strict,
        alpha=0.05,
    )
    sig_digest.to_csv(out_dir / "significant_effects.csv", index=False)
    print(f"\nSignificant effects (p < 0.05): {len(sig_digest)} rows "
          f"saved to significant_effects.csv")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generating plots…")
    print("=" * 70)
    _setup_plot_theme()

    plot_means_per_source(df, "success_rate_strict", out_dir,
                           "fig01_mean_strict", "Strict success rate")
    plot_means_per_source(df, "success_rate_lenient", out_dir,
                           "fig02_mean_lenient", "Lenient success rate")
    plot_lid_penalty_facet(df, out_dir / "fig03_lid_penalty.png")

    plot_distribution_grid(df, "success_rate_strict",
                            out_dir / "fig04_distribution_strict.png",
                            "Strict success rate distribution")
    plot_distribution_grid(df, "success_rate_lenient",
                            out_dir / "fig05_distribution_lenient.png",
                            "Lenient success rate distribution")

    plot_distribution_buckets_per_source(
        buckets_strict, out_dir, "fig06_buckets_strict",
        "Strict success rate — distribution buckets")
    plot_distribution_buckets_per_source(
        buckets_lenient, out_dir, "fig07_buckets_lenient",
        "Lenient success rate — distribution buckets")

    plot_layer_heatmap_per_source(layer_df, out_dir, "fig08_layer_heatmap")
    plot_lid_failure_facet(lid_df, out_dir / "fig09_lid_failure.png")

    plot_dataset_compare(compare_strict,
                          out_dir / "fig10_dataset_compare_strict.png",
                          "MAPS vs OpenSubs — strict success rate")
    plot_dataset_compare(compare_lenient,
                          out_dir / "fig11_dataset_compare_lenient.png",
                          "MAPS vs OpenSubs — lenient success rate")

    plot_lang_summary(df, "source_lang", "success_rate_strict",
                       out_dir / "fig12_source_lang_strict.png",
                       "Strict success by source language × model")
    plot_lang_summary(df, "target_lang", "success_rate_strict",
                       out_dir / "fig13_target_lang_strict.png",
                       "Strict success by target language × model")
    plot_lang_summary(df, "source_lang", "success_rate_lenient",
                       out_dir / "fig14_source_lang_lenient.png",
                       "Lenient success by source language × model")
    plot_lang_summary(df, "target_lang", "success_rate_lenient",
                       out_dir / "fig15_target_lang_lenient.png",
                       "Lenient success by target language × model")

    plot_category(df, "success_rate_strict",
                   out_dir / "fig16_category_strict.png",
                   "Strict success rate by category × model")
    plot_category(df, "success_rate_lenient",
                   out_dir / "fig17_category_lenient.png",
                   "Lenient success rate by category × model")
    print(f"  Plots saved to {out_dir}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    print("\nWriting report.md…")
    write_report(
        out_dir, df, df_crit,
        summary_strict, summary_lenient,
        by_dir_strict, by_dir_lenient,
        by_dir_src_strict, by_dir_src_lenient,
        src_lang_stats, tgt_lang_stats,
        layer_df, lid_df, pairwise,
        compare_strict, compare_lenient,
        buckets_strict, buckets_lenient,
        failure_cases, diff_cases, crit_agg,
        model_overall=model_overall_strict,
        pairwise_model_overall=pairwise_model_overall,
        pairwise_target=pairwise_target,
        pairwise_source=pairwise_source,
        pairwise_layer=pairwise_layer,
        sig_digest=sig_digest,
    )
    print(f"  {out_dir / 'report.md'}")

    print("\nDone. All outputs under:", out_dir)


if __name__ == "__main__":
    main()
