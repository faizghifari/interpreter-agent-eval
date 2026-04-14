import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


ID_DISCOURSE_MARKERS = {
    "dong",
    "deh",
    "sih",
    "kok",
    "nih",
    "lah",
    "kan",
    "ya",
}

ID_REQUEST_MARKERS = {
    "tolong",
    "mohon",
    "harap",
    "bisa",
    "bisakah",
    "jangan",
    "ayo",
    "mari",
}

ID_NEGATION_MARKERS = {"tidak", "tak", "jangan", "ga", "nggak", "enggak"}

ID_SLANG_MARKERS = {
    "banget",
    "gitu",
    "begitu",
    "nih",
    "dong",
    "deh",
    "sih",
    "kok",
    "aja",
    "udah",
    "nggak",
    "ga",
    "kayak",
    "emang",
    "beneran",
}

ID_PRONOUN_MARKERS = {
    "aku",
    "saya",
    "gue",
    "gua",
    "kami",
    "kita",
    "kamu",
    "kau",
    "anda",
    "dia",
    "mereka",
}

KO_COLLOQUIAL_MARKERS = {
    "진짜",
    "정말",
    "대박",
    "헐",
    "완전",
    "제발",
    "설마",
    "어쩌라고",
    "그러니까",
    "아무튼",
}

KO_PRONOUN_MARKERS = {
    "나",
    "저",
    "우리",
    "너",
    "당신",
    "그",
    "그녀",
    "얘",
    "걔",
    "쟤",
}

KO_SUBJECT_TOPIC_PARTICLES = {"은", "는", "이", "가"}

KO_SENTENCE_ENDINGS = {
    "잖아",
    "거든",
    "거지",
    "구나",
    "군요",
    "더라",
    "더라고",
    "지요",
    "죠",
    "네요",
    "나요",
}

KO_COMPLEX_VERB_ENDINGS = {
    "겠습니다",
    "겠어",
    "겠네",
    "겠군",
    "겠지",
    "더라고요",
    "더라고",
    "던데",
    "다니까",
    "잖아요",
    "거든요",
}

KO_HONORIFIC_MARKERS = {
    "요",
    "습니다",
    "님",
    "씨",
    "선생",
    "형",
    "누나",
    "언니",
    "오빠",
    "아저씨",
    "아줌마",
}


# Tunable weights for alignment risk and translation-difficulty scoring.
ALIGNMENT_RISK_WEIGHTS = {
    "qmark_mismatch": 0.8,
    "exclaim_mismatch": 0.5,
    "digit_mismatch": 1.1,
    "long_short_asymmetry": 1.0,
}

STEP2_WEIGHTS = {
    "question": 0.25,
    "exclamation": 0.25,
    "ellipsis": 0.25,
    "id_discourse_marker": 1.0,
    "request_or_imperative": 1.0,
    "negation": 0.5,
    "id_slang_or_colloquial": 1.0,
    "id_reduplication": 0.5,
    "id_affix_complexity": 0.5,
    "ko_honorific_or_social_marker": 0.5,
    "ko_colloquial_expression": 1.0,
    "ko_sentence_ending_particle": 0.5,
    "ko_complex_verb_ending": 0.5,
    "pronoun_asymmetry": 0.5,
    "ko_subject_topic_omission_likely": 0.5,
    "multi_clause_or_turn": 0.5,
}

QUALITY_ALIGNMENT_PENALTY_RATE = 0.35
QUALITY_ALIGNMENT_PENALTY_CAP = 1.2
WORTHINESS_COMPLEXITY_WEIGHT = 0.65
WORTHINESS_QUALITY_WEIGHT = 0.35


@dataclass
class Candidate:
    segment_id: int
    source_text: str
    target_text: str
    film_key: str
    quality_score: float
    complexity_score: float
    alignment_risk: float
    worthiness_score: float
    embedding_similarity: float
    reasons: List[str]


@dataclass
class EmbeddingCalibration:
    aligned_count: int
    misaligned_count: int
    aligned_median: float
    misaligned_median: float
    aligned_mean: float
    misaligned_mean: float
    threshold: float


def normalize_for_embedding(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    return compact[:max_chars] if max_chars > 0 else compact


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def count_alpha(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def latin_ratio(text: str) -> float:
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    latin = 0
    for ch in alpha_chars:
        code = ord(ch)
        if (0x0041 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):
            latin += 1
    return latin / len(alpha_chars)


def korean_script_ratio(text: str) -> float:
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    korean = 0
    for ch in alpha_chars:
        code = ord(ch)
        if (
            (0xAC00 <= code <= 0xD7AF)
            or (0x1100 <= code <= 0x11FF)
            or (0x3130 <= code <= 0x318F)
            or (0x4E00 <= code <= 0x9FFF)
        ):
            korean += 1
    return korean / len(alpha_chars)


def is_sfx_only(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    if re.fullmatch(r"\[[^\]]+\]", compact):
        return True
    if re.fullmatch(r"\([^\)]+\)", compact):
        return True
    if re.fullmatch(r"<[^>]+>", compact):
        return True
    return False


def count_turn_markers(text: str) -> int:
    # Subtitle speaker turns are typically marked as "- ".
    return len(re.findall(r"(?:(?<=\s)|^)-\s", text or ""))


def has_middle_turn_marker(text: str) -> bool:
    # Reject cases like "... - ..." that indicate multiple turns in one subtitle line.
    for match in re.finditer(r"(?:(?<=\s)|^)-\s", text or ""):
        if match.start() != 0:
            return True
    return False


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def has_id_reduplication(text: str) -> bool:
    # Indonesian reduplication often needs non-literal rendering (e.g., jalan-jalan, tiba-tiba).
    lowered = (text or "").lower()
    if re.search(r"\b([a-z]{2,})-\1\b", lowered):
        return True
    return bool(re.search(r"\b[a-z]{2,}-[a-z]{2,}\b", lowered))


def has_id_affix_complexity(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(
        re.search(r"\bdi[a-z]{3,}\b", lowered)
        or re.search(r"\bter[a-z]{3,}\b", lowered)
        or re.search(r"\bke[a-z]{2,}an\b", lowered)
    )


def has_ko_subject_topic_marking(text: str) -> bool:
    tokens = re.findall(r"[가-힣]+", text or "")
    for token in tokens:
        if len(token) < 2:
            continue
        if token[-1] in KO_SUBJECT_TOPIC_PARTICLES:
            return True
    return False


def has_ko_sentence_ending_marker(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return False
    eojeols = re.findall(r"[가-힣]+", compact)
    for eojeol in eojeols:
        if any(eojeol.endswith(ending) for ending in KO_SENTENCE_ENDINGS):
            return True
    return False


def has_ko_complex_verb_ending(text: str) -> bool:
    eojeols = re.findall(r"[가-힣]+", text or "")
    for eojeol in eojeols:
        if any(eojeol.endswith(ending) for ending in KO_COMPLEX_VERB_ENDINGS):
            return True
    return False


def step1_filter(row: Dict[str, str]) -> Tuple[bool, str, Dict[str, float]]:
    source_text = (row.get("source_text") or "").strip()
    target_text = (row.get("target_text") or "").strip()

    if not source_text or not target_text:
        return False, "empty_text", {}
    if row.get("source_lang") != "id" or row.get("target_lang") != "ko":
        return False, "lang_code_mismatch", {}
    if len(source_text) < 4 or len(target_text) < 4:
        return False, "too_short_chars", {}
    if "�" in source_text or "�" in target_text:
        return False, "replacement_char", {}
    if is_sfx_only(source_text) or is_sfx_only(target_text):
        return False, "sfx_only", {}
    if count_alpha(source_text) < 2 or count_alpha(target_text) < 2:
        return False, "not_enough_alpha", {}

    # Keep only single-speaker utterances. Reject lines where turn marker appears mid-utterance.
    if has_middle_turn_marker(source_text) or has_middle_turn_marker(target_text):
        return False, "middle_turn_marker", {}

    # Also reject lines with multiple turn markers.
    src_turns = count_turn_markers(source_text)
    tgt_turns = count_turn_markers(target_text)
    if src_turns > 1 or tgt_turns > 1:
        return False, "multi_turn_utterance", {}

    src_tokens = len(tokenize(source_text))
    tgt_tokens = len(tokenize(target_text))
    if src_tokens < 2 or tgt_tokens < 2:
        return False, "too_short_tokens", {}

    length_ratio = max(src_tokens, tgt_tokens) / max(1, min(src_tokens, tgt_tokens))
    if length_ratio > 4.0:
        return False, "length_ratio_gt_4", {}

    src_latin = latin_ratio(source_text)
    tgt_ko = korean_script_ratio(target_text)
    if src_latin < 0.55:
        return False, "low_source_latin_ratio", {}
    if tgt_ko < 0.35:
        return False, "low_target_korean_ratio", {}

    metrics = {
        "src_tokens": float(src_tokens),
        "tgt_tokens": float(tgt_tokens),
        "length_ratio": length_ratio,
        "src_latin": src_latin,
        "tgt_ko": tgt_ko,
    }
    return True, "pass", metrics


def compute_alignment_risk(
    row: Dict[str, str], metrics: Dict[str, float]
) -> Tuple[float, List[str]]:
    source_text = row["source_text"]
    target_text = row["target_text"]

    flags: List[str] = []
    risk = 0.0

    src_q = ("?" in source_text) or ("؟" in source_text)
    tgt_q = ("?" in target_text) or ("？" in target_text)
    if src_q != tgt_q:
        risk += ALIGNMENT_RISK_WEIGHTS["qmark_mismatch"]
        flags.append("qmark_mismatch")

    src_exc = "!" in source_text
    tgt_exc = "!" in target_text
    if src_exc != tgt_exc:
        risk += ALIGNMENT_RISK_WEIGHTS["exclaim_mismatch"]
        flags.append("exclaim_mismatch")

    if has_digit(source_text) != has_digit(target_text):
        risk += ALIGNMENT_RISK_WEIGHTS["digit_mismatch"]
        flags.append("digit_mismatch")

    if metrics["src_tokens"] >= 9 and metrics["tgt_tokens"] <= 4:
        risk += ALIGNMENT_RISK_WEIGHTS["long_short_asymmetry"]
        flags.append("long_src_short_tgt")
    if metrics["tgt_tokens"] >= 9 and metrics["src_tokens"] <= 4:
        risk += ALIGNMENT_RISK_WEIGHTS["long_short_asymmetry"]
        flags.append("long_tgt_short_src")

    return round(risk, 3), flags


def step2_score(
    row: Dict[str, str], metrics: Dict[str, float]
) -> Tuple[float, float, float, float, List[str]]:
    source_text = row["source_text"]
    target_text = row["target_text"]
    src_lower = source_text.lower()
    tgt_lower = target_text.lower()

    reasons: List[str] = []
    complexity = 0.0

    if "?" in source_text or "؟" in source_text:
        complexity += STEP2_WEIGHTS["question"]
        reasons.append("question")
    if "!" in source_text:
        complexity += STEP2_WEIGHTS["exclamation"]
        reasons.append("exclamation")
    if "..." in source_text or "…" in source_text:
        complexity += STEP2_WEIGHTS["ellipsis"]
        reasons.append("ellipsis")

    src_tokens = set(tokenize(source_text))
    tgt_tokens = set(tokenize(target_text))
    if src_tokens.intersection(ID_DISCOURSE_MARKERS):
        complexity += STEP2_WEIGHTS["id_discourse_marker"]
        reasons.append("id_discourse_marker")
    if src_tokens.intersection(ID_REQUEST_MARKERS):
        complexity += STEP2_WEIGHTS["request_or_imperative"]
        reasons.append("request_or_imperative")
    if src_tokens.intersection(ID_NEGATION_MARKERS):
        complexity += STEP2_WEIGHTS["negation"]
        reasons.append("negation")

    slang_tokens = src_tokens.intersection(ID_SLANG_MARKERS - ID_DISCOURSE_MARKERS)
    if slang_tokens:
        complexity += STEP2_WEIGHTS["id_slang_or_colloquial"]
        reasons.append("id_slang_or_colloquial")

    if has_id_reduplication(source_text):
        complexity += STEP2_WEIGHTS["id_reduplication"]
        reasons.append("id_reduplication")

    if has_id_affix_complexity(source_text):
        complexity += STEP2_WEIGHTS["id_affix_complexity"]
        reasons.append("id_affix_complexity")

    if any(marker in tgt_lower for marker in KO_HONORIFIC_MARKERS):
        complexity += STEP2_WEIGHTS["ko_honorific_or_social_marker"]
        reasons.append("ko_honorific_or_social_marker")

    if tgt_tokens.intersection(KO_COLLOQUIAL_MARKERS):
        complexity += STEP2_WEIGHTS["ko_colloquial_expression"]
        reasons.append("ko_colloquial_expression")

    if has_ko_sentence_ending_marker(target_text):
        complexity += STEP2_WEIGHTS["ko_sentence_ending_particle"]
        reasons.append("ko_sentence_ending_particle")

    if has_ko_complex_verb_ending(target_text):
        complexity += STEP2_WEIGHTS["ko_complex_verb_ending"]
        reasons.append("ko_complex_verb_ending")

    has_id_pronoun = bool(src_tokens.intersection(ID_PRONOUN_MARKERS))
    has_ko_pronoun = bool(tgt_tokens.intersection(KO_PRONOUN_MARKERS))
    if has_id_pronoun != has_ko_pronoun:
        complexity += STEP2_WEIGHTS["pronoun_asymmetry"]
        reasons.append("pronoun_asymmetry")

    if not has_ko_subject_topic_marking(target_text) and metrics["src_tokens"] >= 5:
        complexity += STEP2_WEIGHTS["ko_subject_topic_omission_likely"]
        reasons.append("ko_subject_topic_omission_likely")

    if "," in source_text or ";" in source_text or " -" in source_text:
        complexity += STEP2_WEIGHTS["multi_clause_or_turn"]
        reasons.append("multi_clause_or_turn")

    quality = 2.0
    length_ratio = metrics["length_ratio"]
    quality -= min(1.0, abs(length_ratio - 1.0) / 3.0)
    quality -= min(0.5, max(0.0, 0.65 - metrics["src_latin"]))
    quality -= min(0.5, max(0.0, 0.55 - metrics["tgt_ko"]))
    quality = max(0.0, round(quality, 3))

    alignment_risk, risk_flags = compute_alignment_risk(row, metrics)
    reasons.extend(risk_flags)

    # Penalize alignment-risky rows so they are unlikely to surface in top picks.
    quality = max(
        0.0,
        round(
            quality
            - min(
                QUALITY_ALIGNMENT_PENALTY_CAP,
                alignment_risk * QUALITY_ALIGNMENT_PENALTY_RATE,
            ),
            3,
        ),
    )

    # Worthiness blends translation difficulty (complexity) and baseline quality.
    worthiness = round(
        complexity * WORTHINESS_COMPLEXITY_WEIGHT + quality * WORTHINESS_QUALITY_WEIGHT,
        3,
    )
    return round(quality, 3), round(complexity, 3), alignment_risk, worthiness, reasons


def build_report(
    out_md: Path,
    input_file: Path,
    total_rows: int,
    step1_pass: int,
    step2_pass: int,
    dropped: Counter,
    selected: List[Candidate],
    preview_limit: int,
    embedding_filter_enabled: bool,
    embedding_threshold: Optional[float],
    embedding_calibration: Optional[EmbeddingCalibration],
) -> None:
    lines: List[str] = []
    lines.append("# Step 1-2 Filter Preview (OpenSubtitles id-ko)")
    lines.append("")
    lines.append(f"Input chunk: {input_file}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows read: {total_rows}")
    lines.append(f"- Step 1 passed (hard quality filter): {step1_pass}")
    lines.append(f"- Step 2 passed (worthiness score): {step2_pass}")
    if embedding_filter_enabled:
        lines.append(f"- Embedding filter enabled: yes")
        lines.append(f"- Embedding similarity threshold: {embedding_threshold}")
        if embedding_calibration:
            lines.append(
                f"- Embedding calibration: aligned={embedding_calibration.aligned_count}, misaligned={embedding_calibration.misaligned_count}, aligned_p50={embedding_calibration.aligned_median:.4f}, misaligned_p50={embedding_calibration.misaligned_median:.4f}, auto_threshold={embedding_calibration.threshold:.4f}"
            )
    else:
        lines.append("- Embedding filter enabled: no")
    lines.append("")
    lines.append("## Step 1 Drop Reasons")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---:|")
    for reason, count in dropped.most_common():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append(f"## Top {min(preview_limit, len(selected))} Selected Candidates")
    lines.append("")
    lines.append(
        "| segment_id | score | quality | complexity | align_risk | emb_sim | source_text | target_text | reasons |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---|---|---|")

    for c in selected[:preview_limit]:
        src = c.source_text.replace("|", "\\|")
        tgt = c.target_text.replace("|", "\\|")
        reason_text = ", ".join(c.reasons).replace("|", "\\|")
        emb_text = (
            f"{c.embedding_similarity:.4f}" if c.embedding_similarity >= 0 else "n/a"
        )
        lines.append(
            f"| {c.segment_id} | {c.worthiness_score:.3f} | {c.quality_score:.3f} | {c.complexity_score:.3f} | {c.alignment_risk:.3f} | {emb_text} | {src} | {tgt} | {reason_text} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def _select_embedding_calibration_sets(
    rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
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


def _pair_sims(
    model: Any,
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


def calibrate_embedding_threshold(
    model: Any,
    rows: List[Dict[str, str]],
    batch_size: int,
    max_chars: int,
) -> Optional[EmbeddingCalibration]:
    aligned_rows, misaligned_rows = _select_embedding_calibration_sets(rows)
    if len(aligned_rows) < 4 or len(misaligned_rows) < 4:
        return None

    aligned_sims = _pair_sims(
        model,
        aligned_rows,
        batch_size=batch_size,
        max_chars=max_chars,
        log_prefix="calibration/aligned",
    )
    misaligned_sims = _pair_sims(
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
    threshold = round((a_p25 + m_p75) / 2.0, 4)

    return EmbeddingCalibration(
        aligned_count=len(aligned_sims),
        misaligned_count=len(misaligned_sims),
        aligned_median=float(median(aligned_sims)),
        misaligned_median=float(median(misaligned_sims)),
        aligned_mean=float(mean(aligned_sims)),
        misaligned_mean=float(mean(misaligned_sims)),
        threshold=threshold,
    )


def apply_embedding_filter(
    selected: List[Candidate],
    model: Any,
    threshold: float,
    batch_size: int,
    eval_topn: int,
    max_chars: int,
) -> Tuple[List[Candidate], int]:
    if not selected:
        return selected, 0

    eval_count = min(eval_topn, len(selected))
    eval_slice = selected[:eval_count]
    pair_rows = [
        {"source_text": c.source_text, "target_text": c.target_text} for c in eval_slice
    ]
    sims = _pair_sims(
        model,
        pair_rows,
        batch_size=batch_size,
        max_chars=max_chars,
        log_prefix="candidates",
    )

    kept: List[Candidate] = []
    dropped = 0
    for idx, c in enumerate(eval_slice):
        sim = sims[idx]
        c.embedding_similarity = sim
        if sim < threshold:
            dropped += 1
            continue
        kept.append(c)

    # Keep ordering stable with embedding as a tie-breaker after worthiness.
    kept.sort(
        key=lambda x: (
            -x.worthiness_score,
            -x.embedding_similarity,
            x.alignment_risk,
            x.segment_id,
        )
    )
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run step 1-2 OpenSubtitles filter preview on one chunk."
    )
    this_file = Path(__file__).resolve()
    workspace_root = this_file.parents[2]
    repo_root = this_file.parents[1]
    default_input = (
        workspace_root
        / "opensubtitles_report"
        / "extracted_readable"
        / "segments_0001.tsv"
    )
    default_output_dir = repo_root / "outputs" / "opensubtitles_step12_preview"

    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--min-worthiness", type=float, default=2.8)
    parser.add_argument("--min-complexity", type=float, default=2.0)
    parser.add_argument("--preview-limit", type=int, default=60)
    parser.add_argument("--top-tsv", type=int, default=300)
    parser.add_argument("--max-alignment-risk", type=float, default=1.6)
    parser.add_argument("--output-tag", type=str, default="strict")
    parser.add_argument("--use-embedding-filter", action="store_true")
    parser.add_argument(
        "--embedding-model", type=str, default="sentence-transformers/LaBSE"
    )
    parser.add_argument("--embedding-threshold", type=float, default=0.60)
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--embedding-max-chars", type=int, default=600)
    parser.add_argument("--embedding-max-seq-length", type=int, default=256)
    parser.add_argument("--embedding-eval-topn", type=int, default=3000)
    parser.add_argument("--auto-calibrate-embedding-threshold", action="store_true")
    args = parser.parse_args()

    input_file = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected: List[Candidate] = []
    dropped = Counter()
    total_rows = 0
    step1_pass = 0
    step2_pass = 0
    all_rows: List[Dict[str, str]] = []
    collect_all_rows = (
        args.use_embedding_filter and args.auto_calibrate_embedding_threshold
    )

    with input_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if collect_all_rows:
                all_rows.append(row)
            total_rows += 1

            ok, reason, metrics = step1_filter(row)
            if not ok:
                dropped[reason] += 1
                continue
            step1_pass += 1

            quality, complexity, alignment_risk, worthiness, reasons = step2_score(
                row, metrics
            )

            if alignment_risk > args.max_alignment_risk:
                dropped["alignment_risk_gt_threshold"] += 1
                continue

            if worthiness < args.min_worthiness or complexity < args.min_complexity:
                dropped["below_step2_threshold"] += 1
                continue

            step2_pass += 1
            selected.append(
                Candidate(
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
                )
            )

    selected.sort(
        key=lambda c: (
            -c.worthiness_score,
            c.alignment_risk,
            -c.complexity_score,
            c.segment_id,
        )
    )

    embedding_calibration: Optional[EmbeddingCalibration] = None
    effective_embedding_threshold: Optional[float] = None
    if args.use_embedding_filter:
        if len(selected) > args.embedding_eval_topn:
            selected = selected[: args.embedding_eval_topn]

        # Free memory pressure before loading a large embedding model on CPU.
        import gc

        gc.collect()

        print(
            f"Embedding filter: loading model {args.embedding_model} (max_seq_length={args.embedding_max_seq_length})",
            flush=True,
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding filter requested but sentence-transformers is not installed."
            ) from exc

        emb_model = SentenceTransformer(args.embedding_model)
        emb_model.max_seq_length = args.embedding_max_seq_length
        print("Embedding filter: model loaded", flush=True)

        if args.auto_calibrate_embedding_threshold:
            print("Embedding filter: calibrating threshold", flush=True)
            embedding_calibration = calibrate_embedding_threshold(
                emb_model,
                all_rows,
                batch_size=args.embedding_batch_size,
                max_chars=args.embedding_max_chars,
            )

        if embedding_calibration:
            effective_embedding_threshold = embedding_calibration.threshold
        else:
            effective_embedding_threshold = args.embedding_threshold

        print(
            f"Embedding filter: applying threshold={effective_embedding_threshold} on top {min(args.embedding_eval_topn, len(selected))} candidates",
            flush=True,
        )

        threshold_for_filter = float(
            effective_embedding_threshold
            if effective_embedding_threshold is not None
            else args.embedding_threshold
        )

        selected, dropped_by_embedding = apply_embedding_filter(
            selected,
            emb_model,
            threshold=threshold_for_filter,
            batch_size=args.embedding_batch_size,
            eval_topn=args.embedding_eval_topn,
            max_chars=args.embedding_max_chars,
        )
        dropped["embedding_similarity_lt_threshold"] += dropped_by_embedding
        step2_pass = len(selected)
        print(
            f"Embedding filter: kept={len(selected)} dropped_by_embedding={dropped_by_embedding}",
            flush=True,
        )

    out_base = input_file.stem + f"_step12_{args.output_tag}"
    out_tsv = output_dir / f"{out_base}_selected_top{args.top_tsv}.tsv"
    out_md = output_dir / f"{out_base}_report.md"

    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "segment_id",
                "worthiness_score",
                "complexity_score",
                "quality_score",
                "alignment_risk",
                "embedding_similarity",
                "film_key",
                "reasons",
                "source_text",
                "target_text",
            ]
        )
        for c in selected[: args.top_tsv]:
            writer.writerow(
                [
                    c.segment_id,
                    f"{c.worthiness_score:.3f}",
                    f"{c.complexity_score:.3f}",
                    f"{c.quality_score:.3f}",
                    f"{c.alignment_risk:.3f}",
                    (
                        f"{c.embedding_similarity:.4f}"
                        if c.embedding_similarity >= 0
                        else ""
                    ),
                    c.film_key,
                    ",".join(c.reasons),
                    c.source_text,
                    c.target_text,
                ]
            )

    build_report(
        out_md=out_md,
        input_file=input_file,
        total_rows=total_rows,
        step1_pass=step1_pass,
        step2_pass=step2_pass,
        dropped=dropped,
        selected=selected,
        preview_limit=args.preview_limit,
        embedding_filter_enabled=args.use_embedding_filter,
        embedding_threshold=effective_embedding_threshold,
        embedding_calibration=embedding_calibration,
    )

    print(f"Input: {input_file}")
    print(f"Total rows: {total_rows}")
    print(f"Step1 pass: {step1_pass}")
    print(f"Step2 pass: {step2_pass}")
    print(f"Report: {out_md}")
    print(f"Top TSV: {out_tsv}")


if __name__ == "__main__":
    main()
