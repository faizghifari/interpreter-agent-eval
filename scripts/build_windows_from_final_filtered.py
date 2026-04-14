import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _read_final_rows(path: Path) -> Tuple[List[Dict[str, str]], Dict[int, List[int]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    wanted: Dict[int, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        sid_raw = row.get("segment_id") or ""
        try:
            sid = int(sid_raw)
        except ValueError:
            continue
        wanted[sid].append(i)
    return rows, wanted


def _context_slice(rows: List[Dict[str, str]], start: int, end: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows[start:end]:
        out.append(
            {
                "segment_id": int(r.get("segment_id") or 0),
                "source_text": (r.get("source_text") or "").strip(),
                "target_text": (r.get("target_text") or "").strip(),
            }
        )
    return out


def build_windows(
    final_filtered_tsv: Path,
    original_segments_dir: Path,
    output_jsonl: Path,
    n_prev: int,
    n_after: int,
) -> None:
    final_rows, wanted_by_id = _read_final_rows(final_filtered_tsv)

    if not final_rows:
        raise RuntimeError(f"No rows found in {final_filtered_tsv}")

    matched_final_idx = set()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        segment_files = sorted(original_segments_dir.glob("segments_*.tsv"))
        if not segment_files:
            raise RuntimeError(f"No segment TSV files found in {original_segments_dir}")

        for seg_file in segment_files:
            with seg_file.open("r", encoding="utf-8", newline="") as f:
                seg_rows = list(csv.DictReader(f, delimiter="\t"))

            if not seg_rows:
                continue

            # Build index by segment_id for this segment file.
            local_idx: Dict[int, List[int]] = defaultdict(list)
            for i, r in enumerate(seg_rows):
                sid_raw = r.get("segment_id") or ""
                try:
                    sid = int(sid_raw)
                except ValueError:
                    continue
                local_idx[sid].append(i)

            candidate_ids = set(local_idx.keys()).intersection(set(wanted_by_id.keys()))
            if not candidate_ids:
                continue

            for sid in candidate_ids:
                final_indices = wanted_by_id[sid]
                row_positions = local_idx[sid]

                for fi in final_indices:
                    if fi in matched_final_idx:
                        continue

                    fr = final_rows[fi]
                    f_src = (fr.get("source_text") or "").strip()
                    f_tgt = (fr.get("target_text") or "").strip()

                    match_pos = None
                    for pos in row_positions:
                        rr = seg_rows[pos]
                        r_src = (rr.get("source_text") or "").strip()
                        r_tgt = (rr.get("target_text") or "").strip()
                        if r_src == f_src and r_tgt == f_tgt:
                            match_pos = pos
                            break

                    # Fallback by segment_id if text comparison fails.
                    if match_pos is None and row_positions:
                        match_pos = row_positions[0]

                    if match_pos is None:
                        continue

                    start_prev = max(0, match_pos - n_prev)
                    end_after = min(len(seg_rows), match_pos + n_after + 1)

                    prev_rows = _context_slice(seg_rows, start_prev, match_pos)
                    after_rows = _context_slice(seg_rows, match_pos + 1, end_after)

                    out_record = {
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
                        "n_prev": n_prev,
                        "n_after": n_after,
                        "prev_context": prev_rows,
                        "after_context": after_rows,
                    }
                    out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    matched_final_idx.add(fi)

    unmatched = len(final_rows) - len(matched_final_idx)
    print(f"input_rows={len(final_rows)}")
    print(f"matched_rows={len(matched_final_idx)}")
    print(f"unmatched_rows={unmatched}")
    print(f"output={output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sliding windows for rows in final_filtered TSV using original segment streams."
    )
    parser.add_argument(
        "--final-filtered-tsv",
        type=Path,
        required=True,
        help="Path to final filtered TSV (merged).",
    )
    parser.add_argument(
        "--original-segments-dir",
        type=Path,
        required=True,
        help="Directory containing original segments_*.tsv files.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--n-prev", type=int, default=15)
    parser.add_argument("--n-after", type=int, default=2)
    args = parser.parse_args()

    build_windows(
        final_filtered_tsv=args.final_filtered_tsv.resolve(),
        original_segments_dir=args.original_segments_dir.resolve(),
        output_jsonl=args.output.resolve(),
        n_prev=args.n_prev,
        n_after=args.n_after,
    )


if __name__ == "__main__":
    main()
