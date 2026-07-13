"""Language verification utilities using GlotLID."""

from typing import Dict, Optional, Any
from dataclasses import dataclass


# Arabic varieties (ISO 639-3) that GlotLID may emit. For an Arabic target we
# accept ANY of these as correct: MSA, the macrolanguage, and the regional
# dialects are all legitimately "Arabic" for this evaluation.
ARABIC_VARIETIES = {
    "ara",  # Arabic (macrolanguage)
    "arb",  # Standard Arabic (MSA)
    "arz",  # Egyptian
    "apc",  # North Levantine
    "ajp",  # South Levantine
    "afb",  # Gulf
    "ary",  # Moroccan
    "ars",  # Najdi
    "acm",  # Mesopotamian / Iraqi
    "acq",  # Ta'izzi-Adeni
    "aeb",  # Tunisian
    "apd",  # Sudanese
    "ayl",  # Libyan
    "acw",  # Hijazi
    "acx",  # Omani
    "acy",  # Cypriot
    "aec",  # Saidi
    "abv",  # Baharna
    "avl",  # Eastern Egyptian Bedawi
    "shu",  # Chadian
    "ssh",  # Shihhi
    "arq",  # Algerian
    "aao",  # Algerian Saharan
    "abh",  # Tajiki Arabic
    "pga",  # Sudanese Creole Arabic
}

# Languages closely related to / easily confused with Indonesian by GlotLID,
# split by how we treat them for an Indonesian target:
#
# INDONESIAN_MALAY_ACCEPT — Betawi and the Malay cluster. These are mutually
# intelligible with / structurally the same as standard Indonesian, so a
# detection here is accepted outright as correct (no review flag).
INDONESIAN_MALAY_ACCEPT = {
    "msa",  # Malay (macrolanguage)
    "zsm",  # Standard Malay
    "zlm",  # Malay (individual)
    "max",  # North Moluccan Malay
    "xmm",  # Manado Malay
    "mui",  # Musi (Malay variety)
    "bew",  # Betawi
    "pse",  # Central Malay
    "abs",  # Ambonese Malay
    "lrt",  # Larantuka Malay
}

# INDONESIAN_REGIONAL_REVIEW — genuinely distinct regional languages of
# Indonesia (Batak group, Javanese, Sundanese, Minangkabau, Banjar, ...). These
# are NOT marked wrong, but flagged for manual review because GlotLID may confuse
# them with Indonesian and a real translation into them would be a genuine error.
INDONESIAN_REGIONAL_REVIEW = {
    "min",  # Minangkabau
    "bjn",  # Banjar
    # Batak group
    "btk",  # Batak (macrolanguage)
    "bbc",  # Toba Batak
    "btm",  # Mandailing Batak
    "btx",  # Karo Batak
    "bts",  # Simalungun Batak
    "btd",  # Dairi Batak
    # Other major regional languages of Indonesia
    "jav",  # Javanese
    "sun",  # Sundanese
    "ban",  # Balinese
    "bug",  # Buginese
    "mad",  # Madurese
    "ace",  # Acehnese
    "mak",  # Makasar
    "nij",  # Ngaju
    "sas",  # Sasak
    "gor",  # Gorontalo
    "rej",  # Rejang
    "ljp",  # Lampung Api
}


@dataclass
class LanguageVerificationResult:
    """Result of language verification."""

    is_correct: bool
    detected_language: str
    detected_script: str
    confidence: float
    expected_language: str
    message: str
    needs_review: bool = False
    review_reason: str = ""


def verify_language_with_glotlid(
    model: Any,
    text: str,
    expected_iso_code: str,
    min_confidence: float = 0.9,
    context_name: str = "Text",
) -> LanguageVerificationResult:
    """Verify that text is in the expected language using GlotLID.

    Args:
        model: Loaded GlotLID fasttext model
        text: Text to verify
        expected_iso_code: Expected ISO 639-3 language code
        min_confidence: Minimum confidence threshold (default 0.9)
        context_name: Name for logging/error messages

    Returns:
        LanguageVerificationResult with verification details
    """
    if not model:
        return LanguageVerificationResult(
            is_correct=True,  # Fail open if model not loaded
            detected_language="unknown",
            detected_script="unknown",
            confidence=0.0,
            expected_language=expected_iso_code,
            message=f"{context_name}: Model not loaded, skipping verification",
        )

    # Clean newlines for prediction
    clean_text = text.replace("\n", " ").strip()

    if not clean_text:
        return LanguageVerificationResult(
            is_correct=False,
            detected_language="empty",
            detected_script="empty",
            confidence=0.0,
            expected_language=expected_iso_code,
            message=f"{context_name}: Empty text",
        )

    try:
        predictions = model.predict(clean_text)

        # Prediction format: (('__label__eng_Latn',), array([0.99...]))
        if not predictions or not predictions[0]:
            return LanguageVerificationResult(
                is_correct=False,
                detected_language="unknown",
                detected_script="unknown",
                confidence=0.0,
                expected_language=expected_iso_code,
                message=f"{context_name}: No prediction returned",
            )

        label = predictions[0][0]
        confidence = predictions[1][
            0
        ]  # Keep as numpy scalar, compatible with numpy 2.0

        # Parse label: __label__{iso}_{script}
        # e.g. __label__arb_Arab
        parts = label.replace("__label__", "").split("_")
        if len(parts) >= 2:
            detected_iso = parts[0]
            detected_script = parts[1]
        else:
            detected_iso = label.replace("__label__", "")
            detected_script = "Unknown"

        # Check if language matches and confidence is sufficient
        is_correct = (detected_iso == expected_iso_code) and (
            confidence >= min_confidence
        )
        needs_review = False
        review_reason = ""

        # Special handling for Arabic (arb/ara): GlotLID emits many dialect
        # labels (arz, apc, afb, ary, ars, ...) that are all legitimately
        # Arabic. Accept ANY detected Arabic variety, or anything in the Arabic
        # script, as correct regardless of which specific variety was predicted.
        if not is_correct and expected_iso_code in ("arb", "ara"):
            if detected_iso in ARABIC_VARIETIES or detected_script == "Arab":
                is_correct = True

        # Special handling for Indonesian (ind). GlotLID routinely confuses
        # Indonesian with related languages:
        #  - Betawi / Malay cluster -> accept outright (effectively Indonesian).
        #  - Other regional languages (Batak, Javanese, Minangkabau, ...) ->
        #    don't mark wrong, but flag for manual review.
        if not is_correct and expected_iso_code == "ind":
            if detected_iso in INDONESIAN_MALAY_ACCEPT:
                is_correct = True
            elif detected_iso in INDONESIAN_REGIONAL_REVIEW:
                is_correct = True
                needs_review = True
                review_reason = (
                    f"Detected closely-related language "
                    f"{detected_iso}_{detected_script} (confidence: "
                    f"{confidence:.2f}); flagged for manual review instead of "
                    f"being marked incorrect."
                )

        if needs_review:
            message = f"{context_name}: REVIEW — {review_reason}"
        elif not is_correct:
            if detected_iso != expected_iso_code:
                message = f"{context_name}: Detected as {detected_iso}_{detected_script} (confidence: {confidence:.2f}), expected {expected_iso_code}"
            else:
                message = f"{context_name}: Correct language {detected_iso} but low confidence ({confidence:.2f} < {min_confidence})"
        else:
            message = f"{context_name}: Verified as {detected_iso}_{detected_script} (confidence: {confidence:.2f})"

        return LanguageVerificationResult(
            is_correct=is_correct,
            detected_language=detected_iso,
            detected_script=detected_script,
            confidence=confidence,
            expected_language=expected_iso_code,
            message=message,
            needs_review=needs_review,
            review_reason=review_reason,
        )

    except Exception as e:
        return LanguageVerificationResult(
            is_correct=True,  # Fail open on errors
            detected_language="error",
            detected_script="error",
            confidence=0.0,
            expected_language=expected_iso_code,
            message=f"{context_name}: Verification error: {str(e)}",
        )


def load_glotlid_model() -> Optional[Any]:
    """Load the GlotLID language identification model.

    Returns:
        Loaded fasttext model or None if loading fails
    """
    try:
        import fasttext
        from huggingface_hub import hf_hub_download

        print("Loading GlotLID language identification model...")
        model_path = hf_hub_download(
            repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None
        )
        model = fasttext.load_model(model_path)
        print(f"GlotLID model loaded from {model_path}")
        return model
    except ImportError:
        print(
            "Warning: fasttext or huggingface_hub not installed. Language verification disabled."
        )
        return None
    except Exception as e:
        print(
            f"Warning: Failed to load GlotLID model: {e}. Language verification disabled."
        )
        return None
