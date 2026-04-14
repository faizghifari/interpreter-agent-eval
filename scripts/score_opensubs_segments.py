import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Candidate:
    segment_id: int
    source_text: str
    target_text: str
    metadata_match_type: str
    film_key: str
    quality_score: float
    complexity_score: float
    alignment_risk: float
    worthiness_raw_score: float
    worthiness_score: float
    embedding_similarity: float
    reasons: List[str]


def _load_step12_module(this_file: Path):
    script_path = this_file.parent / "run_step12_opensubtitles_preview.py"
    spec = importlib.util.spec_from_file_location("step12", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load step12 module from {script_path}")
    step12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step12)
    return step12


def _pair_sims(
    model: SentenceTransformer,
    step12,
    rows: List[Dict[str, str]],
    batch_size: int,
    max_chars: int,
) -> List[float]:
    if not rows:
        return []

    src = [
        step12.normalize_for_embedding(x["source_text"], max_chars=max_chars)
        for x in rows
    ]
    tgt = [
        step12.normalize_for_embedding(x["target_text"], max_chars=max_chars)
        for x in rows
    ]
    emb_src = model.encode(
        src, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    emb_tgt = model.encode(
        tgt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return np.sum(emb_src * emb_tgt, axis=-1).astype(float).tolist()


def main() -> None:
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    repo_root = this_file.parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Score and filter OpenSubtitles segment TSV files using Step1/Step2 heuristics "
            "and LaBSE embedding similarity. Reusable for any language pair supported by "
            "run_step12_opensubtitles_preview.py — adapt the scoring dictionaries there for "
            "a new pair, then point --input-dir at the corresponding extracted_readable/ subdirectory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing segments_*.tsv files for the target language pair "
            "(e.g. opensubtitles_report/extracted_readable/id-ko)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "opensubtitles_final_eval",
    )
    parser.add_argument("--segment-glob", type=str, default="segments_*.tsv")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N segments. 0 means all.",
    )

    parser.add_argument("--min-worthiness", type=float, default=6.0)
    parser.add_argument(
        "--worthiness-scale",
        type=float,
        default=2.0,
        help="Scale factor applied to raw step2 worthiness before thresholding and output.",
    )
    parser.add_argument("--min-complexity", type=float, default=2.0)
    parser.add_argument("--max-alignment-risk", type=float, default=1.6)

    parser.add_argument(
        "--embedding-model", type=str, default="sentence-transformers/LaBSE"
    )
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-max-chars", type=int, default=256)
    parser.add_argument("--embedding-max-seq-length", type=int, default=256)
    parser.add_argument("--embedding-min-sim", type=float, default=0.60)
    parser.add_argument("--embedding-max-sim", type=float, default=0.80)

    parser.add_argument("--top-tsv", type=int, default=500)
    parser.add_argument("--output-tag", type=str, default="worthiness6_labse060_080")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    step12 = _load_step12_module(this_file)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_files = sorted(input_dir.glob(args.segment_glob))
    if args.limit > 0:
        segment_files = segment_files[: args.limit]
    if not segment_files:
        raise RuntimeError(
            f"No segment files found in {input_dir} matching {args.segment_glob}"
        )

    print(f"Found {len(segment_files)} segments to process.", flush=True)
    print("\n--- LOADING LABSE MODEL ONCE ---", flush=True)
    model = SentenceTransformer(args.embedding_model)
    model.max_seq_length = args.embedding_max_seq_length
    print("LaBSE loaded once and ready.\n", flush=True)

    merged_rows: List[List[str]] = []
    cumulative_kept = 0

    for i, input_file in enumerate(segment_files, start=1):
        segment_name = input_file.stem
        out_tsv = (
            output_dir
            / f"{segment_name}_{args.output_tag}_selected_top{args.top_tsv}.tsv"
        )
        if args.skip_existing and out_tsv.exists():
            print(
                f"[{i}/{len(segment_files)}] {segment_name} | skipped (already exists)",
                flush=True,
            )
            continue

        total_rows = 0
        step1_pass = 0
        step2_pass = 0
        candidates: List[Candidate] = []

        with input_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total_rows += 1
                ok, _, metrics = step12.step1_filter(row)
                if not ok:
                    continue
                step1_pass += 1

                quality, complexity, alignment_risk, worthiness_raw, reasons = (
                    step12.step2_score(row, metrics)
                )
                worthiness = worthiness_raw * args.worthiness_scale
                if alignment_risk > args.max_alignment_risk:
                    continue
                if complexity < args.min_complexity or worthiness < args.min_worthiness:
                    continue

                step2_pass += 1
                candidates.append(
                    Candidate(
                        segment_id=int(row.get("segment_id") or 0),
                        source_text=row["source_text"],
                        target_text=row["target_text"],
                        metadata_match_type=row.get("metadata_match_type") or "",
                        film_key=row.get("film_key") or "",
                        quality_score=quality,
                        complexity_score=complexity,
                        alignment_risk=alignment_risk,
                        worthiness_raw_score=worthiness_raw,
                        worthiness_score=worthiness,
                        embedding_similarity=-1.0,
                        reasons=reasons,
                    )
                )

        kept: List[Candidate] = []
        if candidates:
            pair_rows = [
                {"source_text": c.source_text, "target_text": c.target_text}
                for c in candidates
            ]
            sims = _pair_sims(
                model,
                step12,
                pair_rows,
                batch_size=args.embedding_batch_size,
                max_chars=args.embedding_max_chars,
            )
            for idx, c in enumerate(candidates):
                sim = float(sims[idx])
                if args.embedding_min_sim <= sim < args.embedding_max_sim:
                    c.embedding_similarity = sim
                    kept.append(c)

        kept.sort(
            key=lambda x: (
                -x.worthiness_score,
                -x.embedding_similarity,
                x.alignment_risk,
                -x.complexity_score,
                x.segment_id,
            )
        )
        selected = kept[: args.top_tsv]
        cumulative_kept += len(selected)

        with out_tsv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "segment_id",
                    "worthiness_score",
                    "worthiness_raw_score",
                    "complexity_score",
                    "quality_score",
                    "alignment_risk",
                    "embedding_similarity",
                    "metadata_match_type",
                    "film_key",
                    "reasons",
                    "source_text",
                    "target_text",
                ]
            )
            for c in selected:
                row = [
                    c.segment_id,
                    f"{c.worthiness_score:.3f}",
                    f"{c.worthiness_raw_score:.3f}",
                    f"{c.complexity_score:.3f}",
                    f"{c.quality_score:.3f}",
                    f"{c.alignment_risk:.3f}",
                    f"{c.embedding_similarity:.4f}",
                    c.metadata_match_type,
                    c.film_key,
                    ",".join(c.reasons),
                    c.source_text,
                    c.target_text,
                ]
                writer.writerow(row)
                merged_rows.append(row)

        print(
            f"[{i}/{len(segment_files)}] {segment_name} | Rows: {total_rows} | Step1: {step1_pass} | Step2: {step2_pass} | Kept(sim): {len(kept)} | Saved: {len(selected)} | Cumulative Saved: {cumulative_kept}",
            flush=True,
        )

    final_output = output_dir / f"final_filtered_{args.output_tag}.tsv"
    with final_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "segment_id",
                "worthiness_score",
                "worthiness_raw_score",
                "complexity_score",
                "quality_score",
                "alignment_risk",
                "embedding_similarity",
                "metadata_match_type",
                "film_key",
                "reasons",
                "source_text",
                "target_text",
            ]
        )
        writer.writerows(merged_rows)

    print("\n--- RUN COMPLETE ---", flush=True)
    print(f"Final merged dataset: {final_output}", flush=True)
    print(f"Total retained rows: {len(merged_rows)}", flush=True)


if __name__ == "__main__":
    main()
