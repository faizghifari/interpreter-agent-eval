"""
OpenSubtitles id-ko scoring pipeline.

Usage
-----
  python score_opensubs_segments.py preview  --input <file.tsv>   [opts]
  python score_opensubs_segments.py bulk     --input-dir <dir>    [opts]
  python score_opensubs_segments.py refilter --input <scored.tsv> [opts]

preview   — score a single segment file, write TSV + markdown report
bulk      — score a directory of segment files, write per-file TSVs + merged output
refilter  — apply LaBSE similarity filter to an already-scored TSV
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from _opensubs_scoring import (
    Candidate,
    compute_alignment_risk,
    compute_pair_similarities,
    has_digit,
    step1_filter,
    step2_score,
)


# ── Shared: embedding calibration ────────────────────────────────────────────


@dataclass
class EmbeddingCalibration:
    aligned_count: int
    misaligned_count: int
    aligned_median: float
    misaligned_median: float
    aligned_mean: float
    misaligned_mean: float
    threshold: float


def _compute_calibration(
    model: Any,
    aligned_rows: List[Dict[str, str]],
    misaligned_rows: List[Dict[str, str]],
    batch_size: int,
    max_chars: int,
) -> Optional[EmbeddingCalibration]:
    if len(aligned_rows) < 4 or len(misaligned_rows) < 4:
        return None
    aligned_sims = compute_pair_similarities(
        model, aligned_rows, batch_size=batch_size, max_chars=max_chars,
        log_prefix="calibration/aligned",
    )
    misaligned_sims = compute_pair_similarities(
        model, misaligned_rows, batch_size=batch_size, max_chars=max_chars,
        log_prefix="calibration/misaligned",
    )
    if not aligned_sims or not misaligned_sims:
        return None
    a_sorted = sorted(aligned_sims)
    m_sorted = sorted(misaligned_sims)
    a_p25 = a_sorted[int(0.25 * (len(a_sorted) - 1))]
    m_p75 = m_sorted[int(0.75 * (len(m_sorted) - 1))]
    return EmbeddingCalibration(
        aligned_count=len(aligned_sims),
        misaligned_count=len(misaligned_sims),
        aligned_median=float(median(aligned_sims)),
        misaligned_median=float(median(misaligned_sims)),
        aligned_mean=float(mean(aligned_sims)),
        misaligned_mean=float(mean(misaligned_sims)),
        threshold=round((a_p25 + m_p75) / 2.0, 4),
    )


# ── preview: helpers ─────────────────────────────────────────────────────────


def _calibration_sets_from_raw(
    rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Select aligned/misaligned calibration pairs from raw segment rows."""
    aligned: List[Dict[str, str]] = []
    for x in rows:
        ok, _, metrics = step1_filter(x)
        if not ok:
            continue
        risk, _ = compute_alignment_risk(x, metrics)
        if risk <= 0.4:
            aligned.append(x)
        if len(aligned) >= 30:
            break

    misaligned: List[Dict[str, str]] = []
    for x in rows:
        src = x.get("source_text") or ""
        tgt = x.get("target_text") or ""
        src_tok = len(src.split())
        tgt_tok = len(tgt.split())
        digit_mismatch = has_digit(src) != has_digit(tgt)
        asym = (src_tok >= 10 and tgt_tok <= 4) or (tgt_tok >= 10 and src_tok <= 4)
        q_mismatch = ("?" in src) != ("?" in tgt)
        if (digit_mismatch and asym) or (q_mismatch and asym):
            misaligned.append(x)
        if len(misaligned) >= 40:
            break
    return aligned, misaligned


def _apply_embedding_filter(
    selected: List[Candidate],
    model: Any,
    threshold: float,
    batch_size: int,
    eval_topn: int,
    max_chars: int,
) -> Tuple[List[Candidate], int]:
    if not selected:
        return selected, 0
    eval_slice = selected[: min(eval_topn, len(selected))]
    pair_rows = [
        {"source_text": c.source_text, "target_text": c.target_text} for c in eval_slice
    ]
    sims = compute_pair_similarities(
        model, pair_rows, batch_size=batch_size, max_chars=max_chars,
        log_prefix="candidates",
    )
    kept: List[Candidate] = []
    dropped = 0
    for c, sim in zip(eval_slice, sims):
        c.embedding_similarity = sim
        if sim < threshold:
            dropped += 1
        else:
            kept.append(c)
    kept.sort(
        key=lambda x: (-x.worthiness_score, -x.embedding_similarity, x.alignment_risk, x.segment_id)
    )
    return kept, dropped


def _build_preview_report(
    out_md: Path,
    input_file: Path,
    total_rows: int,
    step1_pass: int,
    step2_pass: int,
    dropped: Counter,
    selected: List[Candidate],
    preview_limit: int,
    emb_enabled: bool,
    emb_threshold: Optional[float],
    emb_calibration: Optional[EmbeddingCalibration],
) -> None:
    lines: List[str] = []
    lines.append("# Step 1-2 Filter Preview (OpenSubtitles id-ko)")
    lines.append(f"\nInput chunk: {input_file}\n")
    lines.append("## Summary\n")
    lines.append(f"- Total rows read: {total_rows}")
    lines.append(f"- Step 1 passed (hard quality filter): {step1_pass}")
    lines.append(f"- Step 2 passed (worthiness score): {step2_pass}")
    if emb_enabled:
        lines.append(f"- Embedding filter enabled: yes")
        lines.append(f"- Embedding similarity threshold: {emb_threshold}")
        if emb_calibration:
            cal = emb_calibration
            lines.append(
                f"- Embedding calibration: aligned={cal.aligned_count}, "
                f"misaligned={cal.misaligned_count}, "
                f"aligned_p50={cal.aligned_median:.4f}, "
                f"misaligned_p50={cal.misaligned_median:.4f}, "
                f"auto_threshold={cal.threshold:.4f}"
            )
    else:
        lines.append("- Embedding filter enabled: no")
    lines.append("\n## Step 1 Drop Reasons\n")
    lines.append("| reason | count |")
    lines.append("|---|---:|")
    for reason, count in dropped.most_common():
        lines.append(f"| {reason} | {count} |")
    lines.append(f"\n## Top {min(preview_limit, len(selected))} Selected Candidates\n")
    lines.append(
        "| segment_id | score | quality | complexity | align_risk | emb_sim | source_text | target_text | reasons |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---|---|---|")
    for c in selected[:preview_limit]:
        src = c.source_text.replace("|", "\\|")
        tgt = c.target_text.replace("|", "\\|")
        emb_text = f"{c.embedding_similarity:.4f}" if c.embedding_similarity >= 0 else "n/a"
        lines.append(
            f"| {c.segment_id} | {c.worthiness_score:.3f} | {c.quality_score:.3f} "
            f"| {c.complexity_score:.3f} | {c.alignment_risk:.3f} | {emb_text} "
            f"| {src} | {tgt} | {', '.join(c.reasons).replace('|', chr(92) + '|')} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def run_preview(args: argparse.Namespace) -> None:
    input_file = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected: List[Candidate] = []
    dropped: Counter = Counter()
    total_rows = step1_pass = step2_pass = 0
    all_rows: List[Dict[str, str]] = []
    collect_all = args.use_embedding_filter and args.auto_calibrate_embedding_threshold

    with input_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if collect_all:
                all_rows.append(row)
            total_rows += 1
            ok, reason, metrics = step1_filter(row)
            if not ok:
                dropped[reason] += 1
                continue
            step1_pass += 1
            quality, complexity, alignment_risk, worthiness, reasons = step2_score(row, metrics)
            if alignment_risk > args.max_alignment_risk:
                dropped["alignment_risk_gt_threshold"] += 1
                continue
            if worthiness < args.min_worthiness or complexity < args.min_complexity:
                dropped["below_step2_threshold"] += 1
                continue
            step2_pass += 1
            selected.append(Candidate(
                segment_id=int(row.get("segment_id") or 0),
                source_text=row["source_text"],
                target_text=row["target_text"],
                film_key=row.get("film_key") or "",
                quality_score=quality,
                complexity_score=complexity,
                alignment_risk=alignment_risk,
                worthiness_score=worthiness,
                embedding_similarity=-1.0,
                reasons=reasons,
            ))

    selected.sort(
        key=lambda c: (-c.worthiness_score, c.alignment_risk, -c.complexity_score, c.segment_id)
    )

    emb_calibration: Optional[EmbeddingCalibration] = None
    emb_threshold: Optional[float] = None

    if args.use_embedding_filter:
        if len(selected) > args.embedding_eval_topn:
            selected = selected[: args.embedding_eval_topn]
        import gc; gc.collect()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed.") from exc

        print(f"Embedding filter: loading {args.embedding_model}", flush=True)
        emb_model = SentenceTransformer(args.embedding_model)
        emb_model.max_seq_length = args.embedding_max_seq_length
        print("Embedding filter: model loaded", flush=True)

        if args.auto_calibrate_embedding_threshold:
            aligned, misaligned = _calibration_sets_from_raw(all_rows)
            emb_calibration = _compute_calibration(
                emb_model, aligned, misaligned,
                args.embedding_batch_size, args.embedding_max_chars,
            )

        threshold_to_use: float = (
            emb_calibration.threshold if emb_calibration else args.embedding_threshold
        )
        emb_threshold = threshold_to_use
        print(f"Embedding filter: threshold={threshold_to_use}", flush=True)

        selected, n_dropped = _apply_embedding_filter(
            selected, emb_model, threshold=threshold_to_use,
            batch_size=args.embedding_batch_size,
            eval_topn=args.embedding_eval_topn,
            max_chars=args.embedding_max_chars,
        )
        dropped["embedding_similarity_lt_threshold"] += n_dropped
        step2_pass = len(selected)
        print(f"Embedding filter: kept={len(selected)} dropped={n_dropped}", flush=True)

    out_base = input_file.stem + f"_step12_{args.output_tag}"
    out_tsv = output_dir / f"{out_base}_selected_top{args.top_tsv}.tsv"
    out_md = output_dir / f"{out_base}_report.md"

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "segment_id", "worthiness_score", "complexity_score", "quality_score",
            "alignment_risk", "embedding_similarity", "film_key", "reasons",
            "source_text", "target_text",
        ])
        for c in selected[: args.top_tsv]:
            writer.writerow([
                c.segment_id, f"{c.worthiness_score:.3f}", f"{c.complexity_score:.3f}",
                f"{c.quality_score:.3f}", f"{c.alignment_risk:.3f}",
                f"{c.embedding_similarity:.4f}" if c.embedding_similarity >= 0 else "",
                c.film_key, ",".join(c.reasons), c.source_text, c.target_text,
            ])

    _build_preview_report(
        out_md=out_md, input_file=input_file, total_rows=total_rows,
        step1_pass=step1_pass, step2_pass=step2_pass, dropped=dropped,
        selected=selected, preview_limit=args.preview_limit,
        emb_enabled=args.use_embedding_filter,
        emb_threshold=emb_threshold, emb_calibration=emb_calibration,
    )
    print(f"Input: {input_file}\nTotal rows: {total_rows}\nStep1 pass: {step1_pass}")
    print(f"Step2 pass: {step2_pass}\nReport: {out_md}\nTop TSV: {out_tsv}")


# ── bulk: helpers ────────────────────────────────────────────────────────────

_BULK_HEADER = [
    "segment_id", "worthiness_score", "worthiness_raw_score", "complexity_score",
    "quality_score", "alignment_risk", "embedding_similarity", "metadata_match_type",
    "film_key", "reasons", "source_text", "target_text",
]


def run_bulk(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_files = sorted(input_dir.glob(args.segment_glob))
    if args.limit > 0:
        segment_files = segment_files[: args.limit]
    if not segment_files:
        raise RuntimeError(f"No segment files found in {input_dir} matching {args.segment_glob}")

    print(f"Found {len(segment_files)} segment files to process.", flush=True)
    print("\n--- LOADING LABSE MODEL ONCE ---", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model)
    model.max_seq_length = args.embedding_max_seq_length
    print("LaBSE loaded.\n", flush=True)

    merged_rows: List[List] = []
    cumulative_kept = 0

    for i, input_file in enumerate(segment_files, start=1):
        segment_name = input_file.stem
        out_tsv = output_dir / f"{segment_name}_{args.output_tag}_selected_top{args.top_tsv}.tsv"
        if args.skip_existing and out_tsv.exists():
            print(f"[{i}/{len(segment_files)}] {segment_name} | skipped (already exists)", flush=True)
            continue

        total_rows = step1_pass = step2_pass = 0
        candidates: List[Candidate] = []

        with input_file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                total_rows += 1
                ok, _, metrics = step1_filter(row)
                if not ok:
                    continue
                step1_pass += 1
                quality, complexity, alignment_risk, worthiness_raw, reasons = step2_score(row, metrics)
                worthiness = worthiness_raw * args.worthiness_scale
                if alignment_risk > args.max_alignment_risk:
                    continue
                if complexity < args.min_complexity or worthiness < args.min_worthiness:
                    continue
                step2_pass += 1
                candidates.append(Candidate(
                    segment_id=int(row.get("segment_id") or 0),
                    source_text=row["source_text"],
                    target_text=row["target_text"],
                    film_key=row.get("film_key") or "",
                    quality_score=quality,
                    complexity_score=complexity,
                    alignment_risk=alignment_risk,
                    worthiness_score=worthiness,
                    embedding_similarity=-1.0,
                    reasons=reasons,
                    metadata_match_type=row.get("metadata_match_type") or "",
                    worthiness_raw_score=worthiness_raw,
                ))

        kept: List[Candidate] = []
        if candidates:
            pair_rows = [
                {"source_text": c.source_text, "target_text": c.target_text}
                for c in candidates
            ]
            sims = compute_pair_similarities(
                model, pair_rows,
                batch_size=args.embedding_batch_size,
                max_chars=args.embedding_max_chars,
            )
            for c, sim in zip(candidates, sims):
                if args.embedding_min_sim <= sim < args.embedding_max_sim:
                    c.embedding_similarity = sim
                    kept.append(c)

        kept.sort(
            key=lambda x: (
                -x.worthiness_score, -x.embedding_similarity,
                x.alignment_risk, -x.complexity_score, x.segment_id,
            )
        )
        selected = kept[: args.top_tsv]
        cumulative_kept += len(selected)

        with out_tsv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(_BULK_HEADER)
            for c in selected:
                row_out = [
                    c.segment_id, f"{c.worthiness_score:.3f}", f"{c.worthiness_raw_score:.3f}",
                    f"{c.complexity_score:.3f}", f"{c.quality_score:.3f}",
                    f"{c.alignment_risk:.3f}", f"{c.embedding_similarity:.4f}",
                    c.metadata_match_type, c.film_key, ",".join(c.reasons),
                    c.source_text, c.target_text,
                ]
                writer.writerow(row_out)
                merged_rows.append(row_out)

        print(
            f"[{i}/{len(segment_files)}] {segment_name} | "
            f"Rows: {total_rows} | Step1: {step1_pass} | Step2: {step2_pass} | "
            f"Kept(sim): {len(kept)} | Saved: {len(selected)} | Cumulative: {cumulative_kept}",
            flush=True,
        )

    final_output = output_dir / f"final_filtered_{args.output_tag}.tsv"
    with final_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(_BULK_HEADER)
        writer.writerows(merged_rows)

    print(f"\n--- RUN COMPLETE ---\nFinal: {final_output}\nTotal retained: {len(merged_rows)}", flush=True)


# ── refilter: helpers ────────────────────────────────────────────────────────


def _calibration_sets_from_scored(
    rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Select calibration pairs from already-scored TSVs using metadata_match_type."""
    aligned = [
        x for x in rows
        if (x.get("metadata_match_type") or "") == "exact_1to1_text_match"
    ][:30]

    misaligned: List[Dict[str, str]] = []
    for x in rows:
        if (x.get("metadata_match_type") or "") != "unmatched":
            continue
        s = x.get("source_text") or ""
        t = x.get("target_text") or ""
        s_tok = len(s.split())
        t_tok = len(t.split())
        digit_mismatch = has_digit(s) != has_digit(t)
        asym = (s_tok >= 10 and t_tok <= 4) or (t_tok >= 10 and s_tok <= 4)
        strong_turn_gap = (s.count("-") >= 2 and t.count("-") == 0) or (t.count("-") >= 2 and s.count("-") == 0)
        q_mismatch = ("?" in s) != ("?" in t)
        if (strong_turn_gap and (asym or digit_mismatch or q_mismatch)) or (digit_mismatch and asym):
            misaligned.append(x)
        if len(misaligned) >= 40:
            break
    return aligned, misaligned


def _build_refilter_report(
    report_path: Path,
    input_path: Path,
    threshold: float,
    total: int,
    n_kept: int,
    n_dropped: int,
    calibration: Optional[EmbeddingCalibration],
    kept_rows: List[Dict[str, str]],
    dropped_rows: List[Dict[str, str]],
) -> None:
    lines: List[str] = [
        "# LaBSE Embedding Filter Report", "",
        f"Input: {input_path}", "",
        "## Summary", "",
        f"- Similarity threshold: {threshold:.4f}",
        f"- Total candidates: {total}",
        f"- Kept: {n_kept}",
        f"- Dropped: {n_dropped}",
        "",
    ]
    if calibration:
        cal = calibration
        lines += [
            "## Calibration", "",
            f"- aligned_count: {cal.aligned_count}",
            f"- misaligned_count: {cal.misaligned_count}",
            f"- aligned_median: {cal.aligned_median:.4f}",
            f"- misaligned_median: {cal.misaligned_median:.4f}",
            f"- aligned_mean: {cal.aligned_mean:.4f}",
            f"- misaligned_mean: {cal.misaligned_mean:.4f}",
            f"- auto_threshold: {cal.threshold:.4f}",
            "",
        ]

    def sample_table(title: str, rows: List[Dict[str, str]], reverse: bool) -> None:
        lines += [
            f"## {title}", "",
            "| segment_id | emb_sim | source_text | target_text |",
            "|---:|---:|---|---|",
        ]
        for row in sorted(
            rows, key=lambda x: float(x.get("embedding_similarity") or "0"), reverse=reverse
        )[:20]:
            sim = float(row.get("embedding_similarity") or "0")
            src = (row.get("source_text") or "").replace("|", "\\|")
            tgt = (row.get("target_text") or "").replace("|", "\\|")
            lines.append(f"| {row.get('segment_id', '')} | {sim:.4f} | {src} | {tgt} |")

    sample_table("Lowest Similarity Dropped Samples", dropped_rows, reverse=False)
    lines.append("")
    sample_table("Highest Similarity Kept Samples", kept_rows, reverse=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_refilter(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = output_dir / f"{input_path.stem}_{args.output_tag}.tsv"
    out_report = output_dir / f"{input_path.stem}_{args.output_tag}_report.md"

    print(f"Loading candidates: {input_path}", flush=True)
    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"Candidate rows: {len(rows)}", flush=True)

    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {args.embedding_model}", flush=True)
    model = SentenceTransformer(args.embedding_model)
    model.max_seq_length = args.embedding_max_seq_length
    print("Model loaded", flush=True)

    calibration: Optional[EmbeddingCalibration] = None
    threshold = args.threshold
    if args.auto_calibrate:
        if args.calibration_chunk is None:
            raise ValueError("--calibration-chunk is required when --auto-calibrate is set")
        with args.calibration_chunk.resolve().open("r", encoding="utf-8", newline="") as f:
            cal_rows = list(csv.DictReader(f, delimiter="\t"))
        aligned, misaligned = _calibration_sets_from_scored(cal_rows)
        calibration = _compute_calibration(
            model, aligned, misaligned,
            args.embedding_batch_size, args.embedding_max_chars,
        )
        if calibration:
            threshold = calibration.threshold
        print(f"Calibration threshold: {threshold}", flush=True)

    sims = compute_pair_similarities(
        model, rows,
        batch_size=args.embedding_batch_size,
        max_chars=args.embedding_max_chars,
        log_prefix="candidates",
    )

    kept_rows: List[Dict[str, str]] = []
    dropped_rows: List[Dict[str, str]] = []
    for row, sim in zip(rows, sims):
        row = dict(row)
        row["embedding_similarity"] = f"{sim:.4f}"
        (kept_rows if sim >= threshold else dropped_rows).append(row)

    kept_rows.sort(key=lambda x: (
        -float(x.get("worthiness_score") or "0"),
        -float(x.get("embedding_similarity") or "0"),
        float(x.get("alignment_risk") or "0"),
        int(x.get("segment_id") or "0"),
    ))

    fieldnames = list(rows[0].keys()) if rows else []
    if "embedding_similarity" not in fieldnames:
        fieldnames.append("embedding_similarity")
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept_rows)

    _build_refilter_report(
        report_path=out_report, input_path=input_path, threshold=threshold,
        total=len(rows), n_kept=len(kept_rows), n_dropped=len(dropped_rows),
        calibration=calibration, kept_rows=kept_rows, dropped_rows=dropped_rows,
    )
    print(
        f"Threshold: {threshold:.4f} | Kept: {len(kept_rows)} | Dropped: {len(dropped_rows)}\n"
        f"Output TSV: {out_tsv}\nReport: {out_report}",
        flush=True,
    )


# ── windows: helpers ────────────────────────────────────────────────────────


def _read_final_rows(
    path: Path,
) -> Tuple[List[Dict[str, str]], Dict[int, List[int]]]:
    """Load a final_filtered TSV and return rows + index mapping segment_id → row positions."""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    index: Dict[int, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        try:
            index[int(row.get("segment_id") or "")].append(i)
        except ValueError:
            pass
    return rows, index


def _context_slice(rows: List[Dict[str, str]], start: int, end: int) -> List[Dict[str, str]]:
    return [
        {
            "segment_id": int(r.get("segment_id") or 0),
            "source_text": (r.get("source_text") or "").strip(),
            "target_text": (r.get("target_text") or "").strip(),
        }
        for r in rows[start:end]
    ]


def run_windows(args: argparse.Namespace) -> None:
    """Build context windows: match scored rows back to original segments and write JSONL."""
    final_tsv = args.final_filtered_tsv.resolve()
    segments_dir = args.original_segments_dir.resolve()
    output_jsonl = args.output.resolve()

    final_rows, wanted_by_id = _read_final_rows(final_tsv)
    if not final_rows:
        raise RuntimeError(f"No rows found in {final_tsv}")

    segment_files = sorted(segments_dir.glob("segments_*.tsv"))
    if not segment_files:
        raise RuntimeError(f"No segment TSV files found in {segments_dir}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    matched: set = set()

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for seg_file in segment_files:
            with seg_file.open("r", encoding="utf-8", newline="") as f:
                seg_rows = list(csv.DictReader(f, delimiter="\t"))
            if not seg_rows:
                continue

            local_idx: Dict[int, List[int]] = defaultdict(list)
            for i, r in enumerate(seg_rows):
                try:
                    local_idx[int(r.get("segment_id") or "")].append(i)
                except ValueError:
                    pass

            for sid in set(local_idx) & set(wanted_by_id):
                for fi in wanted_by_id[sid]:
                    if fi in matched:
                        continue
                    fr = final_rows[fi]
                    f_src = (fr.get("source_text") or "").strip()
                    f_tgt = (fr.get("target_text") or "").strip()

                    # Match by text first, fall back to first occurrence by segment_id.
                    match_pos = next(
                        (
                            pos for pos in local_idx[sid]
                            if (seg_rows[pos].get("source_text") or "").strip() == f_src
                            and (seg_rows[pos].get("target_text") or "").strip() == f_tgt
                        ),
                        local_idx[sid][0] if local_idx[sid] else None,
                    )
                    if match_pos is None:
                        continue

                    prev_rows = _context_slice(seg_rows, max(0, match_pos - args.n_prev), match_pos)
                    after_rows = _context_slice(seg_rows, match_pos + 1, min(len(seg_rows), match_pos + args.n_after + 1))

                    out_f.write(json.dumps({
                        "segment_file": seg_file.name,
                        "segment_id": sid,
                        "worthiness_score": float(fr.get("worthiness_score") or 0.0),
                        "worthiness_raw_score": float(fr.get("worthiness_raw_score") or 0.0),
                        "complexity_score": float(fr.get("complexity_score") or 0.0),
                        "quality_score": float(fr.get("quality_score") or 0.0),
                        "alignment_risk": float(fr.get("alignment_risk") or 0.0),
                        "embedding_similarity": float(fr.get("embedding_similarity") or 0.0),
                        "metadata_match_type": fr.get("metadata_match_type") or "",
                        "film_key": fr.get("film_key") or "",
                        "reasons": fr.get("reasons") or "",
                        "source_text": f_src,
                        "target_text": f_tgt,
                        "n_prev": args.n_prev,
                        "n_after": args.n_after,
                        "prev_context": prev_rows,
                        "after_context": after_rows,
                    }, ensure_ascii=False) + "\n")
                    matched.add(fi)

    print(f"input_rows={len(final_rows)}", flush=True)
    print(f"matched_rows={len(matched)}", flush=True)
    print(f"unmatched_rows={len(final_rows) - len(matched)}", flush=True)
    print(f"output={output_jsonl}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _add_embedding_args(p: argparse.ArgumentParser, batch_size: int = 256, max_chars: int = 256) -> None:
    p.add_argument("--embedding-model", default="sentence-transformers/LaBSE")
    p.add_argument("--embedding-batch-size", type=int, default=batch_size)
    p.add_argument("--embedding-max-chars", type=int, default=max_chars)
    p.add_argument("--embedding-max-seq-length", type=int, default=256)


def main() -> None:
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    repo_root = this_file.parents[1]

    parser = argparse.ArgumentParser(
        description="OpenSubtitles id-ko scoring pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── preview ──────────────────────────────────────────────────────────────
    p_prev = sub.add_parser("preview", help="Score a single segment file + markdown report.")
    p_prev.add_argument(
        "--input", type=Path,
        default=workspace_root / "opensubtitles_report" / "extracted_readable" / "segments_0001.tsv",
    )
    p_prev.add_argument(
        "--output-dir", type=Path,
        default=repo_root / "outputs" / "opensubtitles_step12_preview",
    )
    p_prev.add_argument("--min-worthiness", type=float, default=2.8)
    p_prev.add_argument("--min-complexity", type=float, default=2.0)
    p_prev.add_argument("--max-alignment-risk", type=float, default=1.6)
    p_prev.add_argument("--top-tsv", type=int, default=300)
    p_prev.add_argument("--preview-limit", type=int, default=60)
    p_prev.add_argument("--output-tag", type=str, default="strict")
    p_prev.add_argument("--use-embedding-filter", action="store_true")
    p_prev.add_argument("--embedding-threshold", type=float, default=0.60)
    p_prev.add_argument("--embedding-eval-topn", type=int, default=3000)
    p_prev.add_argument("--auto-calibrate-embedding-threshold", action="store_true")
    _add_embedding_args(p_prev, batch_size=1, max_chars=600)

    # ── bulk ─────────────────────────────────────────────────────────────────
    p_bulk = sub.add_parser("bulk", help="Score a directory of segment TSVs.")
    p_bulk.add_argument("--input-dir", type=Path, required=True)
    p_bulk.add_argument(
        "--output-dir", type=Path,
        default=repo_root / "outputs" / "opensubtitles_final_eval",
    )
    p_bulk.add_argument("--segment-glob", type=str, default="segments_*.tsv")
    p_bulk.add_argument("--limit", type=int, default=0, help="Max segment files to process. 0=all.")
    p_bulk.add_argument("--min-worthiness", type=float, default=6.0)
    p_bulk.add_argument("--worthiness-scale", type=float, default=2.0)
    p_bulk.add_argument("--min-complexity", type=float, default=2.0)
    p_bulk.add_argument("--max-alignment-risk", type=float, default=1.6)
    p_bulk.add_argument("--embedding-min-sim", type=float, default=0.60)
    p_bulk.add_argument("--embedding-max-sim", type=float, default=0.80)
    p_bulk.add_argument("--top-tsv", type=int, default=500)
    p_bulk.add_argument("--output-tag", type=str, default="worthiness6_labse060_080")
    p_bulk.add_argument("--skip-existing", action="store_true")
    _add_embedding_args(p_bulk)

    # ── refilter ──────────────────────────────────────────────────────────────
    p_ref = sub.add_parser("refilter", help="Apply embedding filter to an already-scored TSV.")
    p_ref.add_argument("--input", type=Path, required=True)
    p_ref.add_argument(
        "--output-dir", type=Path,
        default=repo_root / "outputs" / "opensubtitles_step12_preview",
    )
    p_ref.add_argument("--output-tag", type=str, default="labse_emb")
    p_ref.add_argument("--threshold", type=float, default=0.60)
    p_ref.add_argument("--auto-calibrate", action="store_true")
    p_ref.add_argument("--calibration-chunk", type=Path, default=None)
    _add_embedding_args(p_ref, batch_size=1, max_chars=600)

    # ── windows ───────────────────────────────────────────────────────────────
    p_win = sub.add_parser(
        "windows",
        help="Build context windows from scored rows + original segment files → JSONL.",
    )
    p_win.add_argument("--final-filtered-tsv", type=Path, required=True,
                       help="Merged TSV from 'bulk' (final_filtered_*.tsv).")
    p_win.add_argument("--original-segments-dir", type=Path, required=True,
                       help="Directory of original segments_*.tsv files used during bulk scoring.")
    p_win.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    p_win.add_argument("--n-prev", type=int, default=15, help="Context lines before the target segment.")
    p_win.add_argument("--n-after", type=int, default=2, help="Context lines after the target segment.")

    args = parser.parse_args()
    {
        "preview": run_preview,
        "bulk": run_bulk,
        "refilter": run_refilter,
        "windows": run_windows,
    }[args.command](args)


if __name__ == "__main__":
    main()
