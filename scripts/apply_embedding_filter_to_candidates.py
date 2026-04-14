import argparse
import csv
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple


def normalize_for_embedding(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    return compact[:max_chars] if max_chars > 0 else compact


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_tsv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in (text or ""))


def select_calibration_sets(
    chunk_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    aligned = [
        x
        for x in chunk_rows
        if (x.get("metadata_match_type") or "") == "exact_1to1_text_match"
    ][:30]

    misaligned: List[Dict[str, str]] = []
    for x in chunk_rows:
        if (x.get("metadata_match_type") or "") != "unmatched":
            continue
        s = x.get("source_text") or ""
        t = x.get("target_text") or ""
        s_tok = len(s.split())
        t_tok = len(t.split())
        turn_s = s.count("-")
        turn_t = t.count("-")
        digit_mismatch = has_digit(s) != has_digit(t)
        asym = (s_tok >= 10 and t_tok <= 4) or (t_tok >= 10 and s_tok <= 4)
        strong_turn_gap = (turn_s >= 2 and turn_t == 0) or (turn_t >= 2 and turn_s == 0)
        q_mismatch = ("?" in s) != ("?" in t)
        if (strong_turn_gap and (asym or digit_mismatch or q_mismatch)) or (
            digit_mismatch and asym
        ):
            misaligned.append(x)
        if len(misaligned) >= 40:
            break

    return aligned, misaligned


def pair_sims(
    model,
    rows: List[Dict[str, str]],
    batch_size: int,
    max_chars: int,
    log_prefix: str,
) -> List[float]:
    if not rows:
        return []

    sims: List[float] = []
    total = len(rows)
    chunk_size = max(1, batch_size * 16)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = rows[start:end]
        src = [
            normalize_for_embedding(x["source_text"], max_chars=max_chars)
            for x in chunk
        ]
        tgt = [
            normalize_for_embedding(x["target_text"], max_chars=max_chars)
            for x in chunk
        ]
        emb_src = model.encode(
            src,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb_tgt = model.encode(
            tgt,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims.extend(float((emb_src[i] * emb_tgt[i]).sum()) for i in range(len(chunk)))
        print(f"{log_prefix}: encoded {end}/{total}", flush=True)

    return sims


def calibrate_threshold(
    model: object,
    chunk_rows: List[Dict[str, str]],
    batch_size: int,
    max_chars: int,
) -> Optional[Dict[str, float]]:
    aligned_rows, misaligned_rows = select_calibration_sets(chunk_rows)
    if len(aligned_rows) < 4 or len(misaligned_rows) < 4:
        return None

    aligned_sims = pair_sims(
        model,
        aligned_rows,
        batch_size=batch_size,
        max_chars=max_chars,
        log_prefix="calibration/aligned",
    )
    misaligned_sims = pair_sims(
        model,
        misaligned_rows,
        batch_size=batch_size,
        max_chars=max_chars,
        log_prefix="calibration/misaligned",
    )
    if not aligned_sims or not misaligned_sims:
        return None

    a_sorted = sorted(aligned_sims)
    m_sorted = sorted(misaligned_sims)
    a_p25 = a_sorted[int(0.25 * (len(a_sorted) - 1))]
    m_p75 = m_sorted[int(0.75 * (len(m_sorted) - 1))]
    threshold = (a_p25 + m_p75) / 2.0

    return {
        "aligned_count": float(len(aligned_sims)),
        "misaligned_count": float(len(misaligned_sims)),
        "aligned_median": float(median(aligned_sims)),
        "misaligned_median": float(median(misaligned_sims)),
        "aligned_mean": float(mean(aligned_sims)),
        "misaligned_mean": float(mean(misaligned_sims)),
        "threshold": float(round(threshold, 4)),
    }


def build_report(
    report_path: Path,
    input_path: Path,
    threshold: float,
    total: int,
    kept: int,
    dropped: int,
    calibration: Optional[Dict[str, float]],
    kept_rows: List[Dict[str, str]],
    dropped_rows: List[Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append("# LaBSE Embedding Filter Report")
    lines.append("")
    lines.append(f"Input: {input_path}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Similarity threshold: {threshold:.4f}")
    lines.append(f"- Total candidates: {total}")
    lines.append(f"- Kept: {kept}")
    lines.append(f"- Dropped: {dropped}")
    lines.append("")

    if calibration:
        lines.append("## Calibration")
        lines.append("")
        lines.append(f"- aligned_count: {int(calibration['aligned_count'])}")
        lines.append(f"- misaligned_count: {int(calibration['misaligned_count'])}")
        lines.append(f"- aligned_median: {calibration['aligned_median']:.4f}")
        lines.append(f"- misaligned_median: {calibration['misaligned_median']:.4f}")
        lines.append(f"- aligned_mean: {calibration['aligned_mean']:.4f}")
        lines.append(f"- misaligned_mean: {calibration['misaligned_mean']:.4f}")
        lines.append(f"- auto_threshold: {calibration['threshold']:.4f}")
        lines.append("")

    lines.append("## Lowest Similarity Dropped Samples")
    lines.append("")
    lines.append("| segment_id | emb_sim | source_text | target_text |")
    lines.append("|---:|---:|---|---|")
    for row in sorted(
        dropped_rows, key=lambda x: float(x.get("embedding_similarity") or "0")
    )[:20]:
        seg = row.get("segment_id", "")
        sim = float(row.get("embedding_similarity") or "0")
        src = (row.get("source_text") or "").replace("|", "\\|")
        tgt = (row.get("target_text") or "").replace("|", "\\|")
        lines.append(f"| {seg} | {sim:.4f} | {src} | {tgt} |")

    lines.append("")
    lines.append("## Highest Similarity Kept Samples")
    lines.append("")
    lines.append("| segment_id | emb_sim | source_text | target_text |")
    lines.append("|---:|---:|---|---|")
    for row in sorted(
        kept_rows,
        key=lambda x: float(x.get("embedding_similarity") or "0"),
        reverse=True,
    )[:20]:
        seg = row.get("segment_id", "")
        sim = float(row.get("embedding_similarity") or "0")
        src = (row.get("source_text") or "").replace("|", "\\|")
        tgt = (row.get("target_text") or "").replace("|", "\\|")
        lines.append(f"| {seg} | {sim:.4f} | {src} | {tgt} |")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply LaBSE embedding filter to candidate TSV."
    )
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    workspace_root = script_path.parents[2]

    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root
        / "outputs"
        / "opensubtitles_step12_preview"
        / "segments_0001_step12_strict_v3_selected_top300.tsv",
    )
    parser.add_argument("--output-tag", type=str, default="labse_emb")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--model", type=str, default="sentence-transformers/LaBSE")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--auto-calibrate", action="store_true")
    parser.add_argument(
        "--calibration-chunk",
        type=Path,
        default=None,
        help=(
            "Path to a raw segments TSV used to calibrate the embedding threshold "
            "(required when --auto-calibrate is set). Typically a file from "
            "opensubtitles_report/extracted_readable/<lang-pair>_with_metadata/."
        ),
    )
    args = parser.parse_args()
    if args.auto_calibrate and args.calibration_chunk is None:
        parser.error("--calibration-chunk is required when --auto-calibrate is set")

    input_path = args.input.resolve()
    out_dir = (repo_root / "outputs" / "opensubtitles_step12_preview").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / f"{input_path.stem}_{args.output_tag}.tsv"
    out_report = out_dir / f"{input_path.stem}_{args.output_tag}_report.md"

    print(f"Loading candidates: {input_path}", flush=True)
    rows = read_tsv(input_path)
    print(f"Candidate rows: {len(rows)}", flush=True)

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {args.model}", flush=True)
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length
    print("Model loaded", flush=True)

    calibration = None
    threshold = args.threshold
    if args.auto_calibrate:
        cal_rows = read_tsv(args.calibration_chunk.resolve())
        calibration = calibrate_threshold(
            model,
            cal_rows,
            batch_size=args.batch_size,
            max_chars=args.max_chars,
        )
        if calibration:
            threshold = float(calibration["threshold"])
        print(f"Calibration threshold: {threshold}", flush=True)

    sims = pair_sims(
        model,
        rows,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        log_prefix="candidates",
    )
    kept_rows: List[Dict[str, str]] = []
    dropped_rows: List[Dict[str, str]] = []

    for row, sim in zip(rows, sims):
        row = dict(row)
        row["embedding_similarity"] = f"{sim:.4f}"
        if sim >= threshold:
            kept_rows.append(row)
        else:
            dropped_rows.append(row)

    kept_rows.sort(
        key=lambda x: (
            -float(x.get("worthiness_score") or "0"),
            -float(x.get("embedding_similarity") or "0"),
            float(x.get("alignment_risk") or "0"),
            int(x.get("segment_id") or "0"),
        )
    )

    fieldnames = list(rows[0].keys()) if rows else []
    if "embedding_similarity" not in fieldnames:
        fieldnames.append("embedding_similarity")
    write_tsv(out_tsv, kept_rows, fieldnames=fieldnames)

    build_report(
        report_path=out_report,
        input_path=input_path,
        threshold=threshold,
        total=len(rows),
        kept=len(kept_rows),
        dropped=len(dropped_rows),
        calibration=calibration,
        kept_rows=kept_rows,
        dropped_rows=dropped_rows,
    )

    print(f"Threshold used: {threshold:.4f}", flush=True)
    print(f"Kept rows: {len(kept_rows)}", flush=True)
    print(f"Dropped rows: {len(dropped_rows)}", flush=True)
    print(f"Output TSV: {out_tsv}", flush=True)
    print(f"Output report: {out_report}", flush=True)


if __name__ == "__main__":
    main()
