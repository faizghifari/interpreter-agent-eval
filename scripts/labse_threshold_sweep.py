import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer

import run_step12_opensubtitles_preview as r


def encode_in_chunks(model: SentenceTransformer, texts: list[str], batch_size: int, chunk_size: int, label: str):
    vectors = []
    total = len(texts)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = texts[start:end]
        emb = model.encode(
            chunk,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors.extend(emb)
        print(f"{label}_encoded={end}/{total}", flush=True)
    return vectors


def main() -> None:
    input_path = Path(
        "d:/dev/mt-eval/opensubtitles_report/extracted_readable/id-ko_with_metadata/segments_0001.tsv"
    )
    output_dir = Path("d:/dev/mt-eval/interpreter-agent-eval/outputs/opensubtitles_step12_preview")
    summary_path = output_dir / "labse_threshold_sweep_summary.tsv"
    t050_full_path = output_dir / "segments_0001_step12_labse_t050_full.tsv"
    t050_low25_path = output_dir / "segments_0001_step12_labse_t050_lowest25.tsv"

    min_worthiness = 2.8
    min_complexity = 2.0
    max_alignment_risk = 1.6

    selected = []
    step1_pass = 0
    total_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            ok, _, metrics = r.step1_filter(row)
            if not ok:
                continue
            step1_pass += 1

            quality, complexity, alignment_risk, worthiness, reasons = r.step2_score(row, metrics)
            if alignment_risk > max_alignment_risk:
                continue
            if worthiness < min_worthiness or complexity < min_complexity:
                continue

            selected.append(
                r.Candidate(
                    segment_id=int(row.get("segment_id") or 0),
                    source_text=row["source_text"],
                    target_text=row["target_text"],
                    metadata_match_type=row.get("metadata_match_type") or "",
                    film_key=row.get("film_key") or "",
                    quality_score=quality,
                    complexity_score=complexity,
                    alignment_risk=alignment_risk,
                    worthiness_score=worthiness,
                    embedding_similarity=-1.0,
                    reasons=reasons,
                )
            )

    selected.sort(
        key=lambda c: (-c.worthiness_score, c.alignment_risk, -c.complexity_score, c.segment_id)
    )

    print(f"total_rows={total_rows}", flush=True)
    print(f"step1_pass={step1_pass}", flush=True)
    print(f"pre_embedding_selected={len(selected)}", flush=True)

    model = SentenceTransformer("sentence-transformers/LaBSE")
    model.max_seq_length = 256

    batch_size = 4
    chunk_size = 64
    max_chars = 600
    src_all = [r.normalize_for_embedding(c.source_text, max_chars=max_chars) for c in selected]
    tgt_all = [r.normalize_for_embedding(c.target_text, max_chars=max_chars) for c in selected]

    print(f"encoding_source_count={len(src_all)}", flush=True)
    emb_src = encode_in_chunks(
        model,
        src_all,
        batch_size=batch_size,
        chunk_size=chunk_size,
        label="source",
    )

    print(f"encoding_target_count={len(tgt_all)}", flush=True)
    emb_tgt = encode_in_chunks(
        model,
        tgt_all,
        batch_size=batch_size,
        chunk_size=chunk_size,
        label="target",
    )

    sims = [float((emb_src[i] * emb_tgt[i]).sum()) for i in range(len(selected))]

    thresholds = [0.50, 0.55, 0.60]
    rows = []
    for t in thresholds:
        kept = sum(1 for s in sims if s >= t)
        dropped = len(selected) - kept
        rows.append((t, kept, dropped, total_rows, step1_pass, len(selected)))
        print(f"threshold={t:.2f} kept={kept} dropped={dropped}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "threshold",
                "kept_after_embedding",
                "dropped_by_embedding",
                "total_rows",
                "step1_pass",
                "pre_embedding_selected",
            ]
        )
        for row in rows:
            w.writerow(row)

    # Export full kept rows at threshold 0.50 and a 25-row borderline review set.
    kept_idx_t050 = [i for i, s in enumerate(sims) if s >= 0.50]

    for i in kept_idx_t050:
        selected[i].embedding_similarity = sims[i]

    kept_candidates = [selected[i] for i in kept_idx_t050]
    kept_candidates.sort(
        key=lambda c: (-c.worthiness_score, -c.embedding_similarity, c.alignment_risk, c.segment_id)
    )
    low25_candidates = sorted(
        kept_candidates, key=lambda c: (c.embedding_similarity, c.segment_id)
    )[:25]

    fieldnames = [
        "segment_id",
        "embedding_similarity",
        "worthiness_score",
        "complexity_score",
        "quality_score",
        "alignment_risk",
        "metadata_match_type",
        "film_key",
        "reasons",
        "source_text",
        "target_text",
    ]

    def to_row(c: r.Candidate) -> dict:
        return {
            "segment_id": c.segment_id,
            "embedding_similarity": f"{c.embedding_similarity:.4f}",
            "worthiness_score": f"{c.worthiness_score:.3f}",
            "complexity_score": f"{c.complexity_score:.3f}",
            "quality_score": f"{c.quality_score:.3f}",
            "alignment_risk": f"{c.alignment_risk:.3f}",
            "metadata_match_type": c.metadata_match_type,
            "film_key": c.film_key,
            "reasons": ",".join(c.reasons),
            "source_text": c.source_text,
            "target_text": c.target_text,
        }

    with t050_full_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for c in kept_candidates:
            w.writerow(to_row(c))

    with t050_low25_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for c in low25_candidates:
            w.writerow(to_row(c))

    print(f"summary_tsv={summary_path}", flush=True)
    print(f"t050_full_tsv={t050_full_path}", flush=True)
    print(f"t050_low25_tsv={t050_low25_path}", flush=True)


if __name__ == "__main__":
    main()
