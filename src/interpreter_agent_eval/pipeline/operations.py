"""Per-record operations for each pipeline stage.

Every operation takes one work-unit dict and returns a NEW dict with added
fields. They never mutate the input and never raise: failures are captured in
an ``error`` field so the stage can keep going and the failure is inspectable
in the materialized output. Each function is pure w.r.t. its declared inputs,
which is what lets the stages be reordered, parallelized, or batched.
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

from interpreter_agent_eval.interpreter import InterpreterAgent
from interpreter_agent_eval.user import User
from interpreter_agent_eval.models import JudgeEvaluation, LanguageCheckResult
from interpreter_agent_eval.prompts.templates import JUDGE_EVALUATION_PROMPT
from interpreter_agent_eval.utils.language_verification import (
    verify_language_with_glotlid,
)

from .io import record_id


# ---------------------------------------------------------------------------
# Stage 0: prepare — flatten enriched data files into work units
# ---------------------------------------------------------------------------
def to_work_unit(
    sample: Dict[str, Any], sample_index: int, source_file: str
) -> Dict[str, Any]:
    """Normalize one enriched-data sample into a pipeline work unit."""
    raw_b_ctx = sample.get("user_b_context", "") or ""
    clean_b_ctx = raw_b_ctx.removeprefix("User B (target-side speaker). ")
    return {
        "record_id": record_id(sample),
        "sample_index": sample_index,
        "source_file": source_file,
        "segment_id": sample.get("segment_id"),
        "direction": sample.get("direction"),
        "category": sample.get("Category"),
        "conversation_context": sample.get(
            "conversation_context", "A general conversation between two users."
        ),
        "source_lang": sample.get("source_language_code"),
        "target_lang": sample.get("target_language_code"),
        "source_text": sample.get("source_text"),
        "user_a_context": sample.get("user_a_context"),
        "user_b_context": clean_b_ctx,
        "verification_prompt": sample.get("verification_prompt"),
    }


# ---------------------------------------------------------------------------
# Stage 1: translate
#
# Split into build_request / apply_response so the same prompt feeds both the
# synchronous backend (translate_record) and the async batch backend, which
# can only build requests up front and apply responses after collection.
# ---------------------------------------------------------------------------
def build_translate_request(
    record: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(system_prompt, user_prompt)`` for translating this record.

    Returns ``(None, None)`` when language codes are missing (caller skips).
    Builds the identical prompt the synchronous interpreter uses.
    """
    source_lang = record.get("source_lang")
    target_lang = record.get("target_lang")
    if not source_lang or not target_lang:
        return None, None
    # Provider is unused for prompt construction; pass None.
    interpreter = InterpreterAgent(
        llm_provider=None,
        source_language=source_lang,
        target_language=target_lang,
        conversation_context=record.get("conversation_context"),
        name="AI Interpreter",
    )
    user_prompt = interpreter._build_translation_prompt(
        record["source_text"], source_lang, target_lang, None
    )
    return interpreter.translation_brief, user_prompt


# Targets whose correct translations must contain non-Latin-1 code points. Output
# that is fully Latin-1-encodable for these targets is mojibake (double-encoded
# UTF-8, as some OpenRouter upstreams return) or wrong-language output. We treat
# it as a failure so the record retries — OpenRouter re-routes to a different
# upstream on the next run, the same way a transient error retries.
_NONLATIN_TARGETS = {"kor", "arb", "ben", "zho", "jpn", "rus", "tha", "hin"}


def looks_mojibake(text: Optional[str], target_lang: Optional[str]) -> bool:
    """True if ``text`` is suspiciously Latin-1-only for a non-Latin target."""
    if not text or target_lang not in _NONLATIN_TARGETS:
        return False
    try:
        text.encode("latin-1")
        return True  # correct kor/arb/ben text would raise here
    except UnicodeEncodeError:
        return False


def _assign_translation(
    out: Dict[str, Any], text: Optional[str], target_lang: Optional[str]
) -> None:
    """Set translated_text, or mark a mojibake failure (kept for inspection)."""
    if text is None:
        out["translated_text"] = None
        out.setdefault("translate_error", "no response")
    elif looks_mojibake(text, target_lang):
        out["translated_text"] = None
        out["translate_error"] = "mojibake/non-target-script output (latin-1 encodable)"
        out["raw_translation"] = text
    else:
        out["translated_text"] = text


def apply_translate_response(
    record: Dict[str, Any], text: Optional[str], interpreter_label: str
) -> Dict[str, Any]:
    out = dict(record)
    out["interpreter"] = interpreter_label
    _assign_translation(out, text, record.get("target_lang"))
    return out


def translate_record_local(
    record: Dict[str, Any],
    translator: Any,
    label: str,
) -> Dict[str, Any]:
    """Translate via a direct NMT model (NLLB/Seamless): raw text + lang codes.

    No prompt/brief — the model takes source_text and ISO-639-3 src/tgt codes.
    Shares the mojibake guard and failure semantics with the LLM path.
    """
    out = dict(record)
    out["interpreter"] = label
    src = record.get("source_lang")
    tgt = record.get("target_lang")
    if not src or not tgt:
        out["translated_text"] = None
        out["translate_error"] = "missing source/target language code"
        return out
    try:
        t0 = time.time()
        text = translator.translate(record["source_text"], src, tgt)
        out["translation_time"] = time.time() - t0
        _assign_translation(out, text, tgt)
    except Exception as e:  # noqa: BLE001 — operations must not raise
        out["translated_text"] = None
        out["translate_error"] = str(e)
    return out


def translate_record(
    record: Dict[str, Any],
    interpreter_provider: Any,
    interpreter_label: str,
) -> Dict[str, Any]:
    out = dict(record)
    out["interpreter"] = interpreter_label
    system_prompt, user_prompt = build_translate_request(record)
    if user_prompt is None:
        out["translated_text"] = None
        out["translate_error"] = "missing source/target language code"
        return out
    try:
        t0 = time.time()
        translation = interpreter_provider.generate(
            user_prompt, system_prompt=system_prompt
        )
        out["translation_time"] = time.time() - t0
        _assign_translation(out, translation, record.get("target_lang"))
    except Exception as e:  # noqa: BLE001 — operations must not raise
        out["translated_text"] = None
        out["translate_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Stage 2: respond (User B simulation)
# ---------------------------------------------------------------------------
def _is_context_size_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return (
        "context size" in s
        or "context_length" in s
        or "context window" in s
        or ("400" in s and ("context" in s or "token" in s))
    )


def respond_record(
    record: Dict[str, Any],
    user_sim_factory: Callable[[str], Tuple[Any, str, str]],
) -> Dict[str, Any]:
    out = dict(record)
    translated = record.get("translated_text")
    if not translated:
        out["user_b_response"] = None
        out["respond_skipped"] = "no translation"
        return out

    target_lang = record["target_lang"]
    try:
        provider, model_label, lang_full = user_sim_factory(target_lang)
    except Exception as e:  # noqa: BLE001
        out["user_b_response"] = None
        out["respond_error"] = f"provider init: {e}"
        return out

    clean_ctx = record.get("user_b_context", "") or ""
    user_b = User(
        name=f"User B ({lang_full})",
        language=target_lang,
        language_name=lang_full,
        is_llm=True,
        llm_provider=provider,
        context=clean_ctx,
    )
    actual_ctx = clean_ctx
    out["user_b_model"] = model_label

    try:
        try:
            resp = user_b.send_message(translated)
        except Exception as send_err:  # noqa: BLE001
            if _is_context_size_error(send_err):
                # Retry with truncated context to fit the model's window while
                # preserving persona (matches legacy run_eval behavior).
                truncated = (
                    clean_ctx[:200].rsplit(" ", 1)[0]
                    if len(clean_ctx) > 200
                    else clean_ctx
                )
                user_b.context = truncated
                actual_ctx = truncated
                resp = user_b.send_message(translated)
            else:
                raise
        out["user_b_response"] = resp
    except Exception as e:  # noqa: BLE001
        out["user_b_response"] = None
        out["respond_error"] = str(e)

    out["actual_user_b_context"] = actual_ctx
    return out


# ---------------------------------------------------------------------------
# Stage 3: verify (GlotLID) — split out of the judge step
# ---------------------------------------------------------------------------
def _lang_check_dict(verification) -> Dict[str, Any]:
    return LanguageCheckResult(
        is_correct=verification.is_correct,
        detected_language=verification.detected_language,
        detected_script=verification.detected_script,
        confidence=verification.confidence,
        expected_language=verification.expected_language,
        message=verification.message,
        needs_review=getattr(verification, "needs_review", False),
        review_reason=getattr(verification, "review_reason", ""),
    ).model_dump()


def verify_record(
    record: Dict[str, Any],
    glotlid_model: Any,
    min_confidence: float = 0.8,
) -> Dict[str, Any]:
    out = dict(record)
    target_lang = record.get("target_lang")
    translated = record.get("translated_text")
    response = record.get("user_b_response")

    trans_check = None
    resp_check = None

    if glotlid_model and translated:
        tv = verify_language_with_glotlid(
            model=glotlid_model,
            text=translated,
            expected_iso_code=target_lang,
            min_confidence=min_confidence,
            context_name="Translation",
        )
        trans_check = _lang_check_dict(tv)

    if glotlid_model and response:
        rv = verify_language_with_glotlid(
            model=glotlid_model,
            text=response,
            expected_iso_code=target_lang,
            min_confidence=min_confidence,
            context_name="Target Response",
        )
        resp_check = _lang_check_dict(rv)

    # Pass/fail semantics preserved from evaluator.evaluate_with_judge:
    # the response check sets the verdict; a failed translation check forces fail.
    passed = True
    if resp_check is not None:
        passed = resp_check["is_correct"]
    if trans_check is not None and not trans_check["is_correct"]:
        passed = False

    needs_review = bool(
        (trans_check is not None and trans_check.get("needs_review"))
        or (resp_check is not None and resp_check.get("needs_review"))
    )

    out["translation_language_check"] = trans_check
    out["response_language_check"] = resp_check
    out["language_check_passed"] = passed
    out["language_check_needs_review"] = needs_review
    return out


# ---------------------------------------------------------------------------
# Stage 4: judge
# ---------------------------------------------------------------------------
def _build_lang_verification_info(
    trans_check: Optional[Dict[str, Any]], resp_check: Optional[Dict[str, Any]]
) -> str:
    if not trans_check and not resp_check:
        return "Language verification not performed (model not provided)."
    parts = []
    if trans_check:
        status = "✓ PASSED" if trans_check["is_correct"] else "✗ FAILED"
        parts.append(
            f"- Translation Language Check: {status}\n  {trans_check['message']}"
        )
    if resp_check:
        status = "✓ PASSED" if resp_check["is_correct"] else "✗ FAILED"
        parts.append(
            f"- Target Response Language Check: {status}\n  {resp_check['message']}"
        )
    return "\n".join(parts)


def _clean_json_text(resp: str) -> str:
    resp = resp.strip()
    if resp.startswith("```json"):
        resp = resp[7:]
    if resp.endswith("```"):
        resp = resp[:-3]
    return resp


def _parse_judge_text(text: str) -> JudgeEvaluation:
    return JudgeEvaluation.model_validate_json(_clean_json_text(text))


def parse_judge_evaluation(judge_provider: Any, prompt: str) -> JudgeEvaluation:
    """Call the judge with structured output, falling back to plain generation.

    Used by the synchronous backend. Shared parse logic with
    ``EvaluationFramework.evaluate_with_judge`` and the batch path.
    """
    try:
        response = judge_provider.generate(
            prompt,
            response_mime_type="application/json",
            response_schema=JudgeEvaluation,
        )
        return _parse_judge_text(response)
    except Exception as e:  # noqa: BLE001 — try plain generation before giving up
        response = judge_provider.generate(prompt)
        try:
            return _parse_judge_text(response)
        except Exception as inner_e:  # noqa: BLE001
            raise RuntimeError(
                f"Judge evaluation failed: {e}. Fallback also failed: {inner_e}"
            )


def build_judge_prompt(record: Dict[str, Any]) -> Optional[str]:
    """Build the judge prompt for a record, or None if there's no translation.

    Identical to the prompt the synchronous judge uses, so sync and batch judge
    the same way (batch relies on the prompt's JSON instructions + text parse).
    """
    translated = record.get("translated_text")
    if not translated:
        return None
    target_response = record.get("user_b_response") or "No response recorded"
    lang_info = _build_lang_verification_info(
        record.get("translation_language_check"), record.get("response_language_check")
    )
    return JUDGE_EVALUATION_PROMPT.format(
        conversation_context=record.get("conversation_context"),
        source_text=record.get("source_text"),
        translated_text=translated,
        target_response=target_response,
        verification_prompt=record.get("verification_prompt"),
        language_verification_info=lang_info,
    )


def _attach_evaluation(
    out: Dict[str, Any], record: Dict[str, Any], evaluation: JudgeEvaluation
) -> None:
    # Re-attach the stage-3 language checks so the persisted evaluation dump
    # matches the legacy schema (which embedded them in the judge result).
    trans_check = record.get("translation_language_check")
    resp_check = record.get("response_language_check")
    evaluation.translation_language_check = (
        LanguageCheckResult(**trans_check) if trans_check else None
    )
    evaluation.response_language_check = (
        LanguageCheckResult(**resp_check) if resp_check else None
    )
    evaluation.language_check_passed = record.get("language_check_passed", True)
    out["evaluation"] = evaluation.model_dump()
    out["completion_rate"] = evaluation.get_completion_rate()
    out["success_rate"] = evaluation.get_success_rate()


def apply_judge_response(
    record: Dict[str, Any], text: Optional[str], judge_label: str
) -> Dict[str, Any]:
    out = dict(record)
    out["judge"] = judge_label
    out["completion_rate"] = "0/0"
    out["success_rate"] = 0.0
    if not record.get("translated_text"):
        out["evaluation"] = None
        out["judge_skipped"] = "no translation"
        return out
    if text is None:
        out["evaluation"] = None
        out["judge_error"] = "no batch response"
        return out
    try:
        _attach_evaluation(out, record, _parse_judge_text(text))
    except Exception as e:  # noqa: BLE001
        out["evaluation"] = None
        out["judge_error"] = str(e)
    return out


def judge_record(
    record: Dict[str, Any],
    judge_provider: Any,
    judge_label: str,
) -> Dict[str, Any]:
    out = dict(record)
    out["judge"] = judge_label
    out["completion_rate"] = "0/0"
    out["success_rate"] = 0.0

    prompt = build_judge_prompt(record)
    if prompt is None:
        out["evaluation"] = None
        out["judge_skipped"] = "no translation"
        return out

    try:
        _attach_evaluation(out, record, parse_judge_evaluation(judge_provider, prompt))
    except Exception as e:  # noqa: BLE001
        out["evaluation"] = None
        out["judge_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Stage 5: consolidate — emit the exact legacy run_eval result schema
# ---------------------------------------------------------------------------
def consolidate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    from datetime import datetime

    return {
        "record_id": record.get("record_id"),
        "sample_index": record.get("sample_index"),
        "timestamp": datetime.now().isoformat(),
        "interpreter": record.get("interpreter"),
        "judge": record.get("judge"),
        "category": record.get("category"),
        "conversation_context": record.get("conversation_context"),
        "source_lang": record.get("source_lang"),
        "target_lang": record.get("target_lang"),
        "source_text": record.get("source_text"),
        "user_a_context": record.get("user_a_context"),
        "user_b_context": record.get("actual_user_b_context")
        or record.get("user_b_context"),
        "translated_text": record.get("translated_text"),
        "user_b_response": record.get("user_b_response"),
        "verification_prompt": record.get("verification_prompt"),
        "evaluation": record.get("evaluation"),
        "completion_rate": record.get("completion_rate", "0/0"),
        "success_rate": record.get("success_rate", 0.0),
    }
