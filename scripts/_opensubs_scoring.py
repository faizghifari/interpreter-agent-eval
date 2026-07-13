"""
Shared scoring primitives for OpenSubtitles subtitle-pair filtering.

Language-specific content (marker sets, script ratio functions, LanguageFeatures
instances, LANG_REGISTRY, CROSS_FEATURES) lives in _lang_features.py.  To add a
new language, edit that file only — no changes are needed here.

Public API
----------
step1_filter(row)           → (pass: bool, reason: str, metrics: dict)
step2_score(row, metrics)   → (quality, src_complexity, tgt_complexity,
                               alignment_risk, worthiness, reasons)
compute_pair_similarities(model, rows, batch_size, max_chars, log_prefix) → List[float]
Candidate                   — dataclass for a scored subtitle pair
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from _lang_features import get_cross_features, get_lang_features

# ── Tunable weights ──────────────────────────────────────────────────────────
# Language-specific feature weights are stored inside each FeatureDef in
# _lang_features.py.  Only universal (language-independent) features live here.

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
    "multi_clause_or_turn": 0.5,
    "pronoun_asymmetry": 0.5,
}

_QUALITY_ALIGNMENT_PENALTY_RATE = 0.35
_QUALITY_ALIGNMENT_PENALTY_CAP = 1.2
WORTHINESS_COMPLEXITY_WEIGHT = 0.65
WORTHINESS_QUALITY_WEIGHT = 0.35

# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    segment_id: int
    source_text: str
    target_text: str
    film_key: str
    quality_score: float
    complexity_score: float      # source-side complexity (drives worthiness ranking)
    tgt_complexity_score: float  # target-side complexity (informational)
    alignment_risk: float
    worthiness_score: float      # src-driven: src_complexity * w + quality * w
    tgt_worthiness_score: float  # tgt-driven: tgt_complexity * w + quality * w
    embedding_similarity: float
    reasons: List[str]
    # "fwd" = src→tgt direction qualifies, "rev" = tgt→src qualifies, "both" = both
    direction: str = "fwd"
    # Bulk-scoring pipeline only
    metadata_match_type: str = ""
    worthiness_raw_score: float = 0.0


# ── Generic text utilities ────────────────────────────────────────────────────


def normalize_for_embedding(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    return compact[:max_chars] if max_chars > 0 else compact


def tokenize(text: str) -> List[str]:
    # \w matches Unicode letters/digits but NOT combining marks (Unicode category M).
    # Indic scripts (Bengali, Devanagari, …) use Mc/Mn vowel signs that attach to
    # consonants — without them, "মানে" becomes {"ম", "ন"} instead of {"মানে"}.
    # We extend the pattern to include the relevant combining-mark ranges.
    return re.findall(
        r"[\w"
        r"̀-ͯ"        # Combining diacritical marks (general)
        r"ؐ-ؚ"        # Arabic extended combining
        r"ً-ٰٟ"  # Arabic tashkeel + superscript alef
        r"ۖ-ۭ"        # Arabic additional marks
        r"ऀ-ॣ"        # Devanagari block (incl. combining marks)
        r"ঀ-৿"        # Bengali block (incl. vowel signs + virama)
        r"]+",
        text.lower(), flags=re.UNICODE,
    )


def count_alpha(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


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


# ── Step 1: hard quality filter ──────────────────────────────────────────────


def step1_filter(row: Dict[str, str]) -> Tuple[bool, str, Dict[str, float]]:
    source_text = (row.get("source_text") or "").strip()
    target_text = (row.get("target_text") or "").strip()

    if not source_text or not target_text:
        return False, "empty_text", {}

    src_lang = (row.get("source_lang") or "").lower()
    tgt_lang = (row.get("target_lang") or "").lower()
    src_feats = get_lang_features(src_lang)
    tgt_feats = get_lang_features(tgt_lang)
    if src_feats is None:
        return False, "unsupported_source_language", {}
    if tgt_feats is None:
        return False, "unsupported_target_language", {}

    if len(source_text) < 4 or len(target_text) < 4:
        return False, "too_short_chars", {}
    if "�" in source_text or "�" in target_text:
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

    src_script = src_feats.script_ratio_fn(source_text)
    tgt_script = tgt_feats.script_ratio_fn(target_text)
    if src_script < src_feats.min_script_ratio:
        return False, "low_source_script_ratio", {}
    if tgt_script < tgt_feats.min_script_ratio:
        return False, "low_target_script_ratio", {}

    return True, "pass", {
        "src_tokens": float(src_tokens),
        "tgt_tokens": float(tgt_tokens),
        "length_ratio": length_ratio,
        "src_script_ratio": src_script,
        "tgt_script_ratio": tgt_script,
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
) -> Tuple[float, float, float, float, float, List[str]]:
    """Return (quality, src_complexity, tgt_complexity, alignment_risk, worthiness, reasons).

    src_complexity  — difficulty of the source-language text; drives worthiness ranking
                      so that segments selected for a src→tgt task reflect genuine
                      source-language challenge.
    tgt_complexity  — difficulty of the target-language text; informational, useful
                      for selecting segments where the translation itself is non-trivial.
    worthiness      — ranking score for src→tgt direction: src_complexity-weighted + quality.
    """
    source_text = row["source_text"]
    target_text = row["target_text"]

    src_lang = (row.get("source_lang") or "").lower()
    tgt_lang = (row.get("target_lang") or "").lower()
    src_feats = get_lang_features(src_lang)
    tgt_feats = get_lang_features(tgt_lang)
    cross = get_cross_features(src_lang, tgt_lang)

    reasons: List[str] = []
    src_complexity = 0.0
    tgt_complexity = 0.0

    # Universal punctuation/structure signals → source complexity
    if "?" in source_text or "؟" in source_text:
        src_complexity += STEP2_WEIGHTS["question"]
        reasons.append("question")
    if "!" in source_text:
        src_complexity += STEP2_WEIGHTS["exclamation"]
        reasons.append("exclamation")
    if "..." in source_text or "…" in source_text:
        src_complexity += STEP2_WEIGHTS["ellipsis"]
        reasons.append("ellipsis")
    if "," in source_text or ";" in source_text or " -" in source_text:
        src_complexity += STEP2_WEIGHTS["multi_clause_or_turn"]
        reasons.append("multi_clause_or_turn")

    src_token_set = set(tokenize(source_text))
    tgt_token_set = set(tokenize(target_text))

    # Source-language intrinsic features → source complexity
    if src_feats is not None:
        for feat in src_feats.features:
            if feat.fn(source_text, src_token_set):
                src_complexity += feat.weight
                reasons.append(feat.name)

    # Target-language intrinsic features → target complexity
    if tgt_feats is not None:
        for feat in tgt_feats.features:
            if feat.fn(target_text, tgt_token_set):
                tgt_complexity += feat.weight
                reasons.append(feat.name)

    # Pronoun asymmetry: cross-lingual signal → target complexity
    # (one side uses pronouns the other drops, creating translation choices)
    if src_feats is not None and tgt_feats is not None:
        has_src_pronoun = bool(src_token_set & src_feats.pronoun_tokens)
        has_tgt_pronoun = bool(tgt_token_set & tgt_feats.pronoun_tokens)
        if has_src_pronoun != has_tgt_pronoun:
            tgt_complexity += STEP2_WEIGHTS["pronoun_asymmetry"]
            reasons.append("pronoun_asymmetry")

    # Cross features: direction-specific, require both sides → target complexity
    for feat in cross:
        if feat.fn(source_text, target_text, src_token_set, tgt_token_set, metrics):
            tgt_complexity += feat.weight
            reasons.append(feat.name)

    # Quality: alignment quality, direction-agnostic
    src_ideal = src_feats.ideal_script_ratio if src_feats else 0.65
    tgt_ideal = tgt_feats.ideal_script_ratio if tgt_feats else 0.55
    quality = 2.0
    quality -= min(1.0, abs(metrics["length_ratio"] - 1.0) / 3.0)
    quality -= min(0.5, max(0.0, src_ideal - metrics.get("src_script_ratio", 0.0)))
    quality -= min(0.5, max(0.0, tgt_ideal - metrics.get("tgt_script_ratio", 0.0)))
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

    # Worthiness ranks segments for the src→tgt direction: source complexity drives it.
    worthiness = round(
        src_complexity * WORTHINESS_COMPLEXITY_WEIGHT + quality * WORTHINESS_QUALITY_WEIGHT,
        3,
    )
    return round(quality, 3), round(src_complexity, 3), round(tgt_complexity, 3), alignment_risk, worthiness, reasons


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
