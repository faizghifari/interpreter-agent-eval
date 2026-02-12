"""Language verification utilities using GlotLID."""

from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class LanguageVerificationResult:
    """Result of language verification."""

    is_correct: bool
    detected_language: str
    detected_script: str
    confidence: float
    expected_language: str
    message: str


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

        # Special handling for Arabic (arb/ara)
        # GlotLID may detect various Arabic dialects (ars, arz, apc, etc.)
        # We accept any "Arab" script result with a slightly lower confidence threshold
        if expected_iso_code in ["arb", "ara"] and not is_correct:
            if detected_script == "Arab" and confidence >= 0.7:
                is_correct = True

        if not is_correct:
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
