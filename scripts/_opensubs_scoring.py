"""
Shared scoring primitives for OpenSubtitles subtitle-pair filtering (id→ko).

Public API
----------
step1_filter(row)           → (pass: bool, reason: str, metrics: dict)
step2_score(row, metrics)   → (quality, complexity, alignment_risk, worthiness, reasons)
compute_pair_similarities(model, rows, batch_size, max_chars, log_prefix) → List[float]
Candidate                   — dataclass for a scored subtitle pair
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ── Indonesian marker sets ───────────────────────────────────────────────────

ID_DISCOURSE_MARKERS = {"dong", "deh", "sih", "kok", "nih", "lah", "kan", "ya"}
ID_REQUEST_MARKERS = {"tolong", "mohon", "harap", "bisa", "bisakah", "jangan", "ayo", "mari"}
ID_NEGATION_MARKERS = {"tidak", "tak", "jangan", "ga", "nggak", "enggak"}
ID_SLANG_MARKERS = {
    "banget", "gitu", "begitu", "nih", "dong", "deh", "sih", "kok", "aja", "udah",
    "nggak", "ga", "kayak", "emang", "beneran",
}
ID_PRONOUN_MARKERS = {
    "aku", "saya", "gue", "gua", "kami", "kita", "kamu", "kau", "anda", "dia", "mereka",
}

# ── Korean marker sets ───────────────────────────────────────────────────────

KO_COLLOQUIAL_MARKERS = {
    "진짜", "정말", "대박", "헐", "완전", "제발", "설마", "어쩌라고", "그러니까", "아무튼",
}
KO_PRONOUN_MARKERS = {"나", "저", "우리", "너", "당신", "그", "그녀", "얘", "걔", "쟤"}
KO_SUBJECT_TOPIC_PARTICLES = {"은", "는", "이", "가"}
KO_SENTENCE_ENDINGS = {
    "잖아", "거든", "거지", "구나", "군요", "더라", "더라고", "지요", "죠", "네요", "나요",
}
KO_COMPLEX_VERB_ENDINGS = {
    "겠습니다", "겠어", "겠네", "겠군", "겠지", "더라고요", "더라고", "던데", "다니까", "잖아요", "거든요",
}
KO_HONORIFIC_MARKERS = {
    "요", "습니다", "님", "씨", "선생", "형", "누나", "언니", "오빠", "아저씨", "아줌마",
}

# ── Tunable weights ──────────────────────────────────────────────────────────

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

_QUALITY_ALIGNMENT_PENALTY_RATE = 0.35
_QUALITY_ALIGNMENT_PENALTY_CAP = 1.2
_WORTHINESS_COMPLEXITY_WEIGHT = 0.65
_WORTHINESS_QUALITY_WEIGHT = 0.35

# ── Data model ───────────────────────────────────────────────────────────────


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
    # Bulk-scoring pipeline only
    metadata_match_type: str = ""
    worthiness_raw_score: float = 0.0


# ── Text utilities ───────────────────────────────────────────────────────────


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
    latin = sum(
        1 for ch in alpha_chars
        if (0x0041 <= ord(ch) <= 0x007A) or (0x00C0 <= ord(ch) <= 0x024F)
    )
    return latin / len(alpha_chars)


def korean_script_ratio(text: str) -> float:
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    korean = sum(
        1 for ch in alpha_chars
        if (0xAC00 <= ord(ch) <= 0xD7AF)
        or (0x1100 <= ord(ch) <= 0x11FF)
        or (0x3130 <= ord(ch) <= 0x318F)
        or (0x4E00 <= ord(ch) <= 0x9FFF)
    )
    return korean / len(alpha_chars)


def is_sfx_only(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    return bool(
        re.fullmatch(r"\[[^\]]+\]", compact)
        or re.fullmatch(r"\([^\)]+\)", compact)
        or re.fullmatch(r"<[^>]+>", compact)
    )


def count_turn_markers(text: str) -> int:
    # Subtitle speaker turns are typically marked as "- ".
    return len(re.findall(r"(?:(?<=\s)|^)-\s", text or ""))


def has_middle_turn_marker(text: str) -> bool:
    # Reject "... - ..." patterns that indicate multiple speakers in one line.
    for match in re.finditer(r"(?:(?<=\s)|^)-\s", text or ""):
        if match.start() != 0:
            return True
    return False


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


def has_id_reduplication(text: str) -> bool:
    # Indonesian reduplication often needs non-literal rendering (e.g. jalan-jalan).
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
    for token in re.findall(r"[가-힣]+", text or ""):
        if len(token) >= 2 and token[-1] in KO_SUBJECT_TOPIC_PARTICLES:
            return True
    return False


def has_ko_sentence_ending_marker(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return any(
        eojeol.endswith(ending)
        for eojeol in re.findall(r"[가-힣]+", compact)
        for ending in KO_SENTENCE_ENDINGS
    )


def has_ko_complex_verb_ending(text: str) -> bool:
    return any(
        eojeol.endswith(ending)
        for eojeol in re.findall(r"[가-힣]+", text or "")
        for ending in KO_COMPLEX_VERB_ENDINGS
    )


# ── Step 1: hard quality filter ──────────────────────────────────────────────


def step1_filter(row: Dict[str, str]) -> Tuple[bool, str, Dict[str, float]]:
    source_text = (row.get("source_text") or "").strip()
    target_text = (row.get("target_text") or "").strip()

    if not source_text or not target_text:
        return False, "empty_text", {}
    if row.get("source_lang") != "id" or row.get("target_lang") != "ko":
        return False, "lang_code_mismatch", {}
    if len(source_text) < 4 or len(target_text) < 4:
        return False, "too_short_chars", {}
    if "\ufffd" in source_text or "\ufffd" in target_text:
        return False, "replacement_char", {}
    if is_sfx_only(source_text) or is_sfx_only(target_text):
        return False, "sfx_only", {}
    if count_alpha(source_text) < 2 or count_alpha(target_text) < 2:
        return False, "not_enough_alpha", {}
    if has_middle_turn_marker(source_text) or has_middle_turn_marker(target_text):
        return False, "middle_turn_marker", {}
    if count_turn_markers(source_text) > 1 or count_turn_markers(target_text) > 1:
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

    return True, "pass", {
        "src_tokens": float(src_tokens),
        "tgt_tokens": float(tgt_tokens),
        "length_ratio": length_ratio,
        "src_latin": src_latin,
        "tgt_ko": tgt_ko,
    }


# ── Alignment risk ───────────────────────────────────────────────────────────


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

    if ("!" in source_text) != ("!" in target_text):
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


# ── Step 2: translation-difficulty scoring ───────────────────────────────────


def step2_score(
    row: Dict[str, str], metrics: Dict[str, float]
) -> Tuple[float, float, float, float, List[str]]:
    """Return (quality, complexity, alignment_risk, worthiness, reasons)."""
    source_text = row["source_text"]
    target_text = row["target_text"]
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

    if src_tokens & ID_DISCOURSE_MARKERS:
        complexity += STEP2_WEIGHTS["id_discourse_marker"]
        reasons.append("id_discourse_marker")
    if src_tokens & ID_REQUEST_MARKERS:
        complexity += STEP2_WEIGHTS["request_or_imperative"]
        reasons.append("request_or_imperative")
    if src_tokens & ID_NEGATION_MARKERS:
        complexity += STEP2_WEIGHTS["negation"]
        reasons.append("negation")
    if src_tokens & (ID_SLANG_MARKERS - ID_DISCOURSE_MARKERS):
        complexity += STEP2_WEIGHTS["id_slang_or_colloquial"]
        reasons.append("id_slang_or_colloquial")
    if has_id_reduplication(source_text):
        complexity += STEP2_WEIGHTS["id_reduplication"]
        reasons.append("id_reduplication")
    if has_id_affix_complexity(source_text):
        complexity += STEP2_WEIGHTS["id_affix_complexity"]
        reasons.append("id_affix_complexity")

    tgt_lower = target_text.lower()
    if any(marker in tgt_lower for marker in KO_HONORIFIC_MARKERS):
        complexity += STEP2_WEIGHTS["ko_honorific_or_social_marker"]
        reasons.append("ko_honorific_or_social_marker")
    if tgt_tokens & KO_COLLOQUIAL_MARKERS:
        complexity += STEP2_WEIGHTS["ko_colloquial_expression"]
        reasons.append("ko_colloquial_expression")
    if has_ko_sentence_ending_marker(target_text):
        complexity += STEP2_WEIGHTS["ko_sentence_ending_particle"]
        reasons.append("ko_sentence_ending_particle")
    if has_ko_complex_verb_ending(target_text):
        complexity += STEP2_WEIGHTS["ko_complex_verb_ending"]
        reasons.append("ko_complex_verb_ending")

    has_id_pronoun = bool(src_tokens & ID_PRONOUN_MARKERS)
    has_ko_pronoun = bool(tgt_tokens & KO_PRONOUN_MARKERS)
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
    quality -= min(1.0, abs(metrics["length_ratio"] - 1.0) / 3.0)
    quality -= min(0.5, max(0.0, 0.65 - metrics["src_latin"]))
    quality -= min(0.5, max(0.0, 0.55 - metrics["tgt_ko"]))
    quality = max(0.0, round(quality, 3))

    alignment_risk, risk_flags = compute_alignment_risk(row, metrics)
    reasons.extend(risk_flags)

    quality = max(
        0.0,
        round(
            quality - min(_QUALITY_ALIGNMENT_PENALTY_CAP, alignment_risk * _QUALITY_ALIGNMENT_PENALTY_RATE),
            3,
        ),
    )

    worthiness = round(
        complexity * _WORTHINESS_COMPLEXITY_WEIGHT + quality * _WORTHINESS_QUALITY_WEIGHT,
        3,
    )
    return round(quality, 3), round(complexity, 3), alignment_risk, worthiness, reasons


# ── Embedding similarity ─────────────────────────────────────────────────────


def compute_pair_similarities(
    model,
    rows: List[Dict[str, str]],
    batch_size: int,
    max_chars: int,
    log_prefix: str = "",
) -> List[float]:
    """Return cosine similarity between source/target embedding pairs for each row."""
    if not rows:
        return []

    sims: List[float] = []
    chunk_size = max(1, batch_size * 16)

    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        src = [normalize_for_embedding(x["source_text"], max_chars) for x in chunk]
        tgt = [normalize_for_embedding(x["target_text"], max_chars) for x in chunk]
        emb_src = model.encode(src, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        emb_tgt = model.encode(tgt, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        sims.extend(float((emb_src[i] * emb_tgt[i]).sum()) for i in range(len(chunk)))
        if log_prefix:
            print(f"{log_prefix}: encoded {start + len(chunk)}/{len(rows)}", flush=True)

    return sims
