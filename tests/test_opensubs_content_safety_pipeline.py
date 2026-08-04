import argparse
import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from opensubs_pipeline import run_filter, run_scene_filter, run_windows  # noqa: E402


FINAL_FIELDS = [
    "segment_id",
    "direction",
    "worthiness_score",
    "worthiness_raw_score",
    "src_complexity_score",
    "tgt_complexity_score",
    "quality_score",
    "alignment_risk",
    "embedding_similarity",
    "metadata_match_type",
    "film_key",
    "reasons",
    "source_text",
    "target_text",
]


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _final_row(segment_id: str, source: str, target: str) -> dict[str, str]:
    return {
        "segment_id": segment_id,
        "direction": "both",
        "worthiness_score": "3.0",
        "worthiness_raw_score": "3.0",
        "src_complexity_score": "2.0",
        "tgt_complexity_score": "2.0",
        "quality_score": "2.0",
        "alignment_risk": "0.0",
        "embedding_similarity": "0.7",
        "metadata_match_type": "exact_1to1_text_match",
        "film_key": "film-a",
        "reasons": "question",
        "source_text": source,
        "target_text": target,
    }


def test_windows_drop_blocked_context_but_allow_review_terms(tmp_path: Path) -> None:
    selected = tmp_path / "selected.tsv"
    segments = tmp_path / "segments"
    segments.mkdir()
    output = tmp_path / "windows.jsonl"

    _write_tsv(
        selected,
        FINAL_FIELDS,
        [
            _final_row("2", "Where are you going?", "어디에 가세요?"),
            _final_row("4", "Are you ready now?", "지금 준비됐어요?"),
        ],
    )
    _write_tsv(
        segments / "segments_0001.tsv",
        ["segment_id", "source_text", "target_text", "film_key"],
        [
            {
                "segment_id": "1",
                "source_text": "You idiot",
                "target_text": "너",
                "film_key": "film-a",
            },
            {
                "segment_id": "2",
                "source_text": "Where are you going?",
                "target_text": "어디에 가세요?",
                "film_key": "film-a",
            },
            {
                "segment_id": "3",
                "source_text": "anjing",
                "target_text": "개",
                "film_key": "film-a",
            },
            {
                "segment_id": "4",
                "source_text": "Are you ready now?",
                "target_text": "지금 준비됐어요?",
                "film_key": "film-a",
            },
        ],
    )

    run_windows(
        argparse.Namespace(
            final_filtered_tsv=selected,
            original_segments_dir=segments,
            output=output,
            n_prev=1,
            n_after=0,
        )
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["segment_id"] for record in records] == [4]


def test_filter_rechecks_legacy_scored_rows(tmp_path: Path) -> None:
    scored_dir = tmp_path / "scored"
    output_dir = tmp_path / "filtered"
    scored_dir.mkdir()
    _write_tsv(
        scored_dir / "segments_0001_scored.tsv",
        [
            "segment_id",
            "src_complexity_score",
            "tgt_complexity_score",
            "quality_score",
            "alignment_risk",
            "worthiness_score",
            "tgt_worthiness_score",
            "film_key",
            "reasons",
            "metadata_match_type",
            "source_text",
            "target_text",
        ],
        [
            {
                "segment_id": "1",
                "src_complexity_score": "2",
                "tgt_complexity_score": "2",
                "quality_score": "2",
                "alignment_risk": "0",
                "worthiness_score": "2",
                "tgt_worthiness_score": "2",
                "film_key": "film-a",
                "reasons": "question",
                "metadata_match_type": "exact_1to1_text_match",
                "source_text": "Email jane.doe@example.org",
                "target_text": "연락해 주세요",
            },
            {
                "segment_id": "2",
                "src_complexity_score": "2",
                "tgt_complexity_score": "2",
                "quality_score": "2",
                "alignment_risk": "0",
                "worthiness_score": "2",
                "tgt_worthiness_score": "2",
                "film_key": "film-a",
                "reasons": "question",
                "metadata_match_type": "exact_1to1_text_match",
                "source_text": "Are you ready now?",
                "target_text": "지금 준비됐어요?",
            },
        ],
    )

    run_filter(
        argparse.Namespace(
            input_dir=scored_dir,
            output_dir=output_dir,
            scored_glob="segments_*_scored.tsv",
            limit=0,
            min_src_complexity=None,
            min_tgt_complexity=None,
            min_complexity=0.0,
            top_k_pct=None,
            min_worthiness=0.0,
            worthiness_scale=1.0,
            max_alignment_risk=1.6,
            skip_embedding=True,
            pre_embed_top=0,
            top_fwd=0,
            top_rev=0,
            top_tsv=500,
            top_per_direction=0,
            output_tag="test",
            skip_existing=False,
            embedding_model="sentence-transformers/LaBSE",
            embedding_batch_size=1,
            embedding_max_chars=256,
            embedding_max_seq_length=256,
            embedding_min_sim=0.6,
            embedding_max_sim=0.8,
        )
    )

    with (output_dir / "final_filtered_test.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["segment_id"] for row in rows] == ["2"]


def test_scene_filter_rejects_legacy_unsafe_window(tmp_path: Path) -> None:
    input_path = tmp_path / "windows.jsonl"
    output_path = tmp_path / "safe.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "film_key": "film-a",
                "source_text": "Contact jane.doe@example.org",
                "target_text": "연락해 주세요",
                "prev_context": [
                    {
                        "film_key": "film-a",
                        "source_text": "Hello there",
                        "target_text": "안녕하세요",
                    }
                ],
                "after_context": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_scene_filter(argparse.Namespace(input=input_path, output=output_path))

    assert output_path.read_text(encoding="utf-8") == ""
