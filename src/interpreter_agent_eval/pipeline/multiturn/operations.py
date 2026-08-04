"""Per-turn-record operations for the multi-turn pipeline stages.

Every operation takes one turn-record (or conversation) dict and returns a NEW
dict with added fields — pure, never mutate the input, never raise (capture
failures in an ``error`` field). Mirrors the convention in
``pipeline/operations.py`` (read-only reference, never modified).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from interpreter_agent_eval.interpreter import InterpreterAgent
from interpreter_agent_eval.models import JudgeEvaluation, LanguageCheckResult
from interpreter_agent_eval.prompts.templates import (
    MULTITURN_JUDGE_CONVERSATION_PROMPT,
    MULTITURN_JUDGE_TURN_PROMPT,
    MULTITURN_TRANSLATION_TASK,
    MULTITURN_USER_SIM_PROMPT,
)
from interpreter_agent_eval.user import User
from interpreter_agent_eval.utils.language_verification import verify_language_with_glotlid

from ..io import record_id
from ..operations import _assign_translation, _lang_name
from . import checklist_gen as cg

# ---------------------------------------------------------------------------
# Stage: mt-prepare — flatten a scripted scenario into per-turn work units
# ---------------------------------------------------------------------------


def conversation_to_turn_units(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one scripted ``MultiTurnScenario`` dict into N turn units.

    Sets ``segment_id``/``direction``/``record_id`` per the D3 resume rule
    (``segment_id = f"{conversation_id}_t{turn_index:02d}"``,
    ``direction = f"{source_lang}-{target_lang}"``, then
    ``record_id = io.record_id(unit)`` — this is correctness-critical: without
    both fields set, ``io.record_id`` falls back to a content hash and every
    turn of a direction collapses onto one resume key), ``listener_context``
    by side, and ``authored_history`` (the static pre-authored source-language
    scaffold — prior turns' speaker + source_text only; mt-translate merges
    this with its own accumulated prior translations to build the
    transcript-so-far block).
    """
    conversation_id = conversation["conversation_id"]
    mode = conversation.get("mode", "scripted")
    lang_a = conversation["lang_a"]
    lang_b = conversation["lang_b"]
    turns = sorted(conversation.get("turns", []), key=lambda t: t["turn_index"])
    num_turns = len(turns)

    units: List[Dict[str, Any]] = []
    authored_history: List[Dict[str, Any]] = []
    for turn in turns:
        turn_index = turn["turn_index"]
        speaker = turn["speaker"]
        source_lang = lang_a if speaker == "A" else lang_b
        target_lang = lang_b if speaker == "A" else lang_a

        unit: Dict[str, Any] = {
            "segment_id": f"{conversation_id}_t{turn_index:02d}",
            "direction": f"{source_lang}-{target_lang}",
            "conversation_id": conversation_id,
            "mode": mode,
            "guidance": conversation.get("guidance"),
            "turn_index": turn_index,
            "num_turns": num_turns,
            "speaker": speaker,
            "lang_a": lang_a,
            "lang_b": lang_b,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_text": turn.get("text"),
            "checklist_items": turn.get("checklist_items", []),
            "verification_prompt": turn.get("verification_prompt"),
            "conversation_context": conversation.get("conversation_context"),
            "user_a_context": conversation.get("user_a_context"),
            "user_b_context": conversation.get("user_b_context"),
            "listener_context": (
                conversation.get("user_b_context")
                if speaker == "A"
                else conversation.get("user_a_context")
            ),
            "authored_history": list(authored_history),
            "category": conversation.get("Category"),
            "seed_file": conversation.get("seed_file"),
            "seed_row_id": conversation.get("seed_row_id"),
        }
        unit["record_id"] = record_id(unit)
        units.append(unit)

        authored_history.append(
            {"turn_index": turn_index, "speaker": speaker, "source_text": turn.get("text")}
        )

    return units


# ---------------------------------------------------------------------------
# Stage: mt-translate — scripted, wave-loop driven (plan D3)
#
# Split into build_request / apply_response, same reason as the single-turn
# pipeline: the same prompt feeds both the synchronous backend
# (translate_turn_record) and, later, the async batch backend (Step 8).
# ---------------------------------------------------------------------------
def build_turn_translate_request(
    unit: Dict[str, Any],
    prior_translations: Dict[Tuple[str, int], str],
    context_mode: str = "transcript",
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(system_prompt, user_prompt)`` for translating this turn.

    Returns ``(None, None)`` when language codes or source_text are missing
    (caller skips). ``prior_translations`` maps
    ``(conversation_id, turn_index) -> translated_text`` for turns already
    completed in this run or resumed from a prior run — combined with the
    unit's ``authored_history`` (source text only) to build the
    transcript-so-far block when ``context_mode == "transcript"``.

    System prompt is always the production ``cultural_context`` brief
    (``InterpreterAgent``'s default) — multi-turn has no brief-ablation
    condition, unlike the single-turn prompt-ablation study.
    """
    source_lang = unit.get("source_lang")
    target_lang = unit.get("target_lang")
    source_text = unit.get("source_text")
    if not source_lang or not target_lang or not source_text:
        return None, None

    # Provider is unused for prompt construction; pass None (mirrors
    # pipeline.operations.build_translate_request).
    interpreter = InterpreterAgent(
        llm_provider=None,
        source_language=source_lang,
        target_language=target_lang,
        conversation_context=unit.get("conversation_context"),
        name="AI Interpreter",
    )

    transcript_block = ""
    if context_mode == "transcript":
        history = unit.get("authored_history") or []
        if history:
            lines = []
            for prior in history:
                translated = prior_translations.get((unit["conversation_id"], prior["turn_index"]))
                line = f"Turn {prior['turn_index']} ({prior['speaker']}): {prior['source_text']}"
                if translated:
                    line += f" | this run's translation: {translated}"
                lines.append(line)
            transcript_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    context_str = f"Context: {unit.get('conversation_context')}" if unit.get("conversation_context") else ""
    user_prompt = MULTITURN_TRANSLATION_TASK.format(
        from_language=_lang_name(source_lang),
        to_language=_lang_name(target_lang),
        context=context_str,
        transcript_block=transcript_block,
        speaker=unit.get("speaker"),
        turn_index=unit.get("turn_index"),
        message=source_text,
    )
    return interpreter.translation_brief, user_prompt


def _build_turn_history(
    unit: Dict[str, Any], prior_translations: Dict[Tuple[str, int], str]
) -> List[Dict[str, Any]]:
    """Final ``history`` field: source_text + this-run's translations.

    Built regardless of ``context_mode`` — it's stored provenance of what
    happened, not itself a prompt input.
    """
    history = []
    for prior in unit.get("authored_history") or []:
        history.append(
            {
                "turn_index": prior["turn_index"],
                "speaker": prior["speaker"],
                "source_text": prior["source_text"],
                "translated_text": prior_translations.get((unit["conversation_id"], prior["turn_index"])),
            }
        )
    return history


def apply_turn_translate_response(
    unit: Dict[str, Any],
    text: Optional[str],
    interpreter_label: str,
    model_slug: str,
    prior_translations: Dict[Tuple[str, int], str],
    context_mode: str = "transcript",
) -> Dict[str, Any]:
    out = dict(unit)
    out["interpreter"] = interpreter_label
    out["model"] = model_slug
    out["context_mode"] = context_mode
    out["history"] = _build_turn_history(unit, prior_translations)
    out.pop("authored_history", None)
    _assign_translation(out, text, unit.get("target_lang"))
    return out


def translate_turn_record(
    unit: Dict[str, Any],
    interpreter_provider: Any,
    interpreter_label: str,
    model_slug: str,
    prior_translations: Dict[Tuple[str, int], str],
    context_mode: str = "transcript",
) -> Dict[str, Any]:
    system_prompt, user_prompt = build_turn_translate_request(unit, prior_translations, context_mode)
    if user_prompt is None:
        out = apply_turn_translate_response(
            unit, None, interpreter_label, model_slug, prior_translations, context_mode
        )
        out["translate_error"] = "missing source/target language code or source_text"
        return out
    try:
        text = interpreter_provider.generate(user_prompt, system_prompt=system_prompt)
    except Exception as e:  # noqa: BLE001 — operations must not raise
        out = apply_turn_translate_response(
            unit, None, interpreter_label, model_slug, prior_translations, context_mode
        )
        out["translate_error"] = str(e)
        return out
    return apply_turn_translate_response(
        unit, text, interpreter_label, model_slug, prior_translations, context_mode
    )


def translate_turn_record_local(
    unit: Dict[str, Any],
    translator: Any,
    label: str,
    prior_translations: Optional[Dict[Tuple[str, int], str]] = None,
) -> Dict[str, Any]:
    """Translate a turn via a direct NMT model (NLLB/Seamless): context-free.

    Mirrors ``pipeline.operations.translate_record_local`` — raw text + ISO-
    639-3 codes, no prompt/brief. Forces ``context_mode="none"`` because this
    is the NMT-floor arm (plan D8 Q1) and it has no notion of interpreter
    context.  The prior-turn ``history`` is still retained as evaluation
    provenance; it is never passed to the translator.
    """
    out = dict(unit)
    out["interpreter"] = label
    out["model"] = label
    out["context_mode"] = "none"
    out["history"] = _build_turn_history(unit, prior_translations or {})
    out.pop("authored_history", None)
    src = unit.get("source_lang")
    tgt = unit.get("target_lang")
    if not src or not tgt:
        out["translated_text"] = None
        out["translate_error"] = "missing source/target language code"
        return out
    try:
        text = translator.translate(unit["source_text"], src, tgt)
        _assign_translation(out, text, tgt)
    except Exception as e:  # noqa: BLE001 — operations must not raise
        out["translated_text"] = None
        out["translate_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Shared: render prior turns entirely in one side's own language. Used by both
# mt-respond's listener probe (scripted) and mt-converse's user-sim (dynamic).
# ---------------------------------------------------------------------------
def render_history_for_side(history: Optional[List[Dict[str, Any]]], side: str) -> str:
    """Render prior turns in ``side``'s own language ("A" or "B").

    For each prior turn: if ``side`` themselves spoke that turn, use their own
    ``source_text`` (already in their language); otherwise use that turn's
    ``translated_text`` (the other side's words, translated into ``side``'s
    language). Mirrors how a real participant would recall the conversation
    so far, entirely in their own language.
    """
    lines = []
    for h in history or []:
        text = h.get("source_text") if h.get("speaker") == side else h.get("translated_text")
        if text:
            lines.append(f"Turn {h['turn_index']} ({h['speaker']}): {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage: mt-respond — scripted only, optional comprehension probe (plan D2)
# ---------------------------------------------------------------------------
def build_turn_respond_history_text(unit: Dict[str, Any]) -> str:
    """Render prior turns in the LISTENER's own language (see ``render_history_for_side``)."""
    listener = "B" if unit.get("speaker") == "A" else "A"
    return render_history_for_side(unit.get("history"), listener)


def respond_turn_record(
    unit: Dict[str, Any],
    user_sim_factory: Callable[[str], Tuple[Any, str, str]],
) -> Dict[str, Any]:
    """Listener comprehension probe (scripted mode, optional auxiliary judge evidence).

    Mirrors ``pipeline.operations.respond_record``'s shape and error/skip
    convention. Unlike single-turn (fixed User B target), the listener here is
    whichever side did NOT speak this turn — ``listener_context`` and
    ``target_lang`` already encode that (set by ``conversation_to_turn_units``).
    """
    out = dict(unit)
    translated = unit.get("translated_text")
    if not translated:
        out["listener_response"] = None
        out["respond_skipped"] = "no translation"
        return out

    target_lang = unit["target_lang"]
    try:
        provider, model_label, lang_full = user_sim_factory(target_lang)
    except Exception as e:  # noqa: BLE001
        out["listener_response"] = None
        out["respond_skipped"] = f"no user-sim model configured for '{target_lang}': {e}"
        return out

    listener_ctx = unit.get("listener_context") or ""
    history_text = build_turn_respond_history_text(unit)
    full_context = listener_ctx
    if history_text:
        full_context = f"{listener_ctx}\n\nConversation so far:\n{history_text}"

    listener_user = User(
        name=f"Listener ({lang_full})",
        language=target_lang,
        language_name=lang_full,
        is_llm=True,
        llm_provider=provider,
        context=full_context,
    )
    out["listener_model"] = model_label
    try:
        out["listener_response"] = listener_user.send_message(translated)
    except Exception as e:  # noqa: BLE001
        out["listener_response"] = None
        out["respond_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Stage: mt-verify — GlotLID on translated_text (+ listener_response)
# ---------------------------------------------------------------------------
def _lang_check_dict(verification: Any) -> Dict[str, Any]:
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


def verify_turn_record(
    unit: Dict[str, Any],
    glotlid_model: Any,
    min_confidence: float = 0.8,
) -> Dict[str, Any]:
    """Mirrors ``pipeline.operations.verify_record``'s pass/fail semantics.

    A failed translation check forces failure; the listener-response check
    (when present) sets the verdict otherwise — same convention as
    single-turn's translation/user_b_response pair.
    """
    out = dict(unit)
    target_lang = unit.get("target_lang") or ""
    translated = unit.get("translated_text")
    response = unit.get("listener_response")

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
            context_name="Listener Response",
        )
        resp_check = _lang_check_dict(rv)

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
# Shared judge-parsing helpers (mirrors pipeline.operations's structured-then-
# plain-fallback parse convention — mirror, don't import; that module is
# protected)
# ---------------------------------------------------------------------------
def _clean_json_text(resp: str) -> str:
    resp = resp.strip()
    if resp.startswith("```json"):
        resp = resp[7:]
    if resp.endswith("```"):
        resp = resp[:-3]
    return resp


def _parse_judge_text(text: str) -> JudgeEvaluation:
    return JudgeEvaluation.model_validate_json(_clean_json_text(text))


def _call_judge_provider(judge_provider: Any, prompt: str) -> JudgeEvaluation:
    try:
        response = judge_provider.generate(
            prompt, response_mime_type="application/json", response_schema=JudgeEvaluation
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


# ---------------------------------------------------------------------------
# Stage: mt-judge-turns — always transcript-conditioned (plan D5)
# ---------------------------------------------------------------------------
def _build_lang_verification_info(
    trans_check: Optional[Dict[str, Any]], resp_check: Optional[Dict[str, Any]]
) -> str:
    if not trans_check and not resp_check:
        return "Language verification not performed (model not provided)."
    parts = []
    if trans_check:
        status = "✓ PASSED" if trans_check["is_correct"] else "✗ FAILED"
        parts.append(f"- Translation Language Check: {status}\n  {trans_check['message']}")
    if resp_check:
        status = "✓ PASSED" if resp_check["is_correct"] else "✗ FAILED"
        parts.append(f"- Listener Response Language Check: {status}\n  {resp_check['message']}")
    return "\n".join(parts)


def _render_turn_transcript_block(history: Optional[List[Dict[str, Any]]]) -> str:
    if not history:
        return "(this is the first turn; no prior turns)"
    lines = [
        f"Turn {h['turn_index']} ({h['speaker']}): {h.get('source_text')} -> {h.get('translated_text')}"
        for h in history
    ]
    return "\n".join(lines)


def _render_judge_history_block(prior_judgments: Optional[List[Dict[str, Any]]]) -> str:
    """Optional --judge-history slot (plan D5): prior turns' criterion verdicts.

    Default off; when omitted (``None``), the slot is blank — no prior judge
    output is injected, keeping per-turn judging independent (D5's decision).
    """
    if not prior_judgments:
        return ""
    lines = ["Prior turns' judged criteria (context only — do not re-litigate):"]
    for pj in sorted(prior_judgments, key=lambda p: p.get("turn_index", 0)):
        ti = pj.get("turn_index")
        evaluation = pj.get("evaluation") or {}
        for r in evaluation.get("results", []):
            status = "Yes" if r.get("met") else "No"
            lines.append(f"  Turn {ti} - {r.get('criteria')}: {status}")
    return "\n".join(lines) + "\n\n"


def build_turn_judge_prompt(
    record: Dict[str, Any], prior_judgments: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """Build the per-turn judge prompt: always includes the transcript-so-far.

    ``prior_judgments`` (the experimental ``--judge-history`` slot) is None by
    default — every funded tier and the default CLI leave it off.
    """
    translated = record.get("translated_text")
    if not translated:
        return None
    listener_response = record.get("listener_response") or "No response recorded"
    lang_info = _build_lang_verification_info(
        record.get("translation_language_check"), record.get("response_language_check")
    )
    return MULTITURN_JUDGE_TURN_PROMPT.format(
        conversation_context=record.get("conversation_context"),
        transcript_block=_render_turn_transcript_block(record.get("history")),
        turn_index=record.get("turn_index"),
        speaker=record.get("speaker"),
        source_text=record.get("source_text"),
        translated_text=translated,
        listener_response=listener_response,
        language_verification_info=lang_info,
        judge_history_block=_render_judge_history_block(prior_judgments),
        verification_prompt=record.get("verification_prompt"),
    )


def _attach_turn_evaluation(
    out: Dict[str, Any], record: Dict[str, Any], evaluation: JudgeEvaluation
) -> None:
    trans_check = record.get("translation_language_check")
    resp_check = record.get("response_language_check")
    evaluation.translation_language_check = LanguageCheckResult(**trans_check) if trans_check else None
    evaluation.response_language_check = LanguageCheckResult(**resp_check) if resp_check else None
    evaluation.language_check_passed = record.get("language_check_passed", True)
    out["evaluation"] = evaluation.model_dump()
    out["completion_rate"] = evaluation.get_completion_rate()
    out["success_rate"] = evaluation.get_success_rate()


def apply_turn_judge_response(
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
        out["judge_error"] = "no response"
        return out
    try:
        _attach_turn_evaluation(out, record, _parse_judge_text(text))
    except Exception as e:  # noqa: BLE001
        out["evaluation"] = None
        out["judge_error"] = str(e)
    return out


def judge_turn_record(
    record: Dict[str, Any],
    judge_provider: Any,
    judge_label: str,
    prior_judgments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out = dict(record)
    out["judge"] = judge_label
    out["completion_rate"] = "0/0"
    out["success_rate"] = 0.0

    prompt = build_turn_judge_prompt(record, prior_judgments)
    if prompt is None:
        out["evaluation"] = None
        out["judge_skipped"] = "no translation"
        return out

    try:
        _attach_turn_evaluation(out, record, _call_judge_provider(judge_provider, prompt))
    except Exception as e:  # noqa: BLE001
        out["evaluation"] = None
        out["judge_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Stage: mt-judge-conv — one call per conversation, over the full transcript
# ---------------------------------------------------------------------------
def conversation_level_unit(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """Build the conversation-level record consumed by mt-judge-conv.

    ``record_id`` is ``"{conversation_id}_conv"`` (plan's Stage-flow table) —
    never collides with a per-turn ``record_id`` (those always end in
    ``_{direction}``, never the literal suffix ``_conv``).
    """
    conversation_id = conversation["conversation_id"]
    return {
        "record_id": f"{conversation_id}_conv",
        "conversation_id": conversation_id,
        "mode": conversation.get("mode", "scripted"),
        "guidance": conversation.get("guidance"),
        "lang_a": conversation.get("lang_a"),
        "lang_b": conversation.get("lang_b"),
        "conversation_context": conversation.get("conversation_context"),
        "conversation_checklist_items": conversation.get("conversation_checklist_items"),
        "conversation_verification_prompt": conversation.get("conversation_verification_prompt"),
    }


def _render_full_transcript(turn_records: List[Dict[str, Any]]) -> str:
    ordered = sorted(turn_records, key=lambda r: r["turn_index"])
    return "\n".join(
        f"Turn {r['turn_index']} ({r['speaker']}): {r.get('source_text')} -> {r.get('translated_text')}"
        for r in ordered
    )


def _render_failed_turns_note(turn_records: List[Dict[str, Any]]) -> str:
    failed = [
        r["turn_index"]
        for r in sorted(turn_records, key=lambda r: r["turn_index"])
        if r.get("language_check_passed") is False
    ]
    if not failed:
        return ""
    return f"Note: turn(s) {failed} failed language verification and should be weighed accordingly.\n\n"


def ensure_conversation_checklist(
    conversation_unit: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
    checklist_provider: Any,
    target_lang: Optional[str],
    use_grounding: bool = True,
    consistency_runs: int = 1,
    filter_by_annotation: bool = False,
    meaningful_threshold: float = cg.DEFAULT_MEANINGFUL_THRESHOLD,
) -> Dict[str, Any]:
    """Fill in the conversation-level checklist if missing (free-flow, D4).

    Scripted / guided-dynamic conversations already carry
    ``conversation_checklist_items`` (authored at generation/seed time) —
    tagged ``checklist_provenance: "authored"``. Free-flow dynamic seeds have
    neither field; this generates them here, post-hoc, from the completed
    transcript — tagged ``checklist_provenance: "posthoc"``.
    """
    out = dict(conversation_unit)
    if out.get("conversation_checklist_items"):
        out["checklist_provenance"] = "authored"
        return out

    taxonomy = None
    if use_grounding and target_lang and cg.taxonomy_available(target_lang):
        taxonomy = cg.load_function_taxonomy(target_lang)

    transcript_basis = _render_full_transcript(turn_records)
    cultural_context = cg.get_cultural_context(out.get("lang_a") or "", target_lang or "")
    items = cg.generate_conversation_checklist(
        checklist_provider,
        target_lang or "",
        out.get("conversation_context") or "",
        transcript_basis,
        taxonomy=taxonomy,
        cultural_context=cultural_context,
        consistency_runs=consistency_runs,
        filter_by_annotation=filter_by_annotation,
        meaningful_threshold=meaningful_threshold,
    )
    out["conversation_checklist_items"] = [item.model_dump() for item in items]
    out["conversation_verification_prompt"] = cg.compose_verification_prompt(items)
    out["checklist_provenance"] = "posthoc"
    return out


def build_conversation_judge_prompt(
    conversation_unit: Dict[str, Any], turn_records: List[Dict[str, Any]]
) -> Optional[str]:
    verification_prompt = conversation_unit.get("conversation_verification_prompt")
    if not verification_prompt or not turn_records:
        return None
    return MULTITURN_JUDGE_CONVERSATION_PROMPT.format(
        conversation_context=conversation_unit.get("conversation_context"),
        transcript_block=_render_full_transcript(turn_records),
        failed_turns_note=_render_failed_turns_note(turn_records),
        verification_prompt=verification_prompt,
    )


def apply_conversation_judge_response(
    conversation_unit: Dict[str, Any], text: Optional[str], judge_label: str
) -> Dict[str, Any]:
    """Batch-reusable counterpart of ``judge_conversation_record`` — applies an
    already-fetched response instead of calling the provider synchronously.
    """
    out = dict(conversation_unit)
    out["judge"] = judge_label
    out["completion_rate"] = "0/0"
    out["success_rate"] = 0.0
    if text is None:
        out["evaluation"] = None
        out["judge_error"] = "no batch response"
        return out
    try:
        evaluation = _parse_judge_text(text)
        out["evaluation"] = evaluation.model_dump()
        out["completion_rate"] = evaluation.get_completion_rate()
        out["success_rate"] = evaluation.get_success_rate()
    except Exception as e:  # noqa: BLE001
        out["evaluation"] = None
        out["judge_error"] = str(e)
    return out


def judge_conversation_record(
    conversation_unit: Dict[str, Any],
    turn_records: List[Dict[str, Any]],
    judge_provider: Any,
    judge_label: str,
) -> Dict[str, Any]:
    out = dict(conversation_unit)
    out["judge"] = judge_label
    out["completion_rate"] = "0/0"
    out["success_rate"] = 0.0

    prompt = build_conversation_judge_prompt(conversation_unit, turn_records)
    if prompt is None:
        out["evaluation"] = None
        out["judge_skipped"] = "no conversation checklist or no judged turns"
        return out

    try:
        evaluation = _call_judge_provider(judge_provider, prompt)
        out["evaluation"] = evaluation.model_dump()
        out["completion_rate"] = evaluation.get_completion_rate()
        out["success_rate"] = evaluation.get_success_rate()
    except Exception as e:  # noqa: BLE001
        out["evaluation"] = None
        out["judge_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Stage: mt-converse — dynamic mode, both variants (plan D1/D3/D4)
#
# Per conversation: speaker utterance (local, live) -> checklist generation
# (cloud, independent of translation) -> translation (cloud, reuses the
# scripted translate operations unchanged — a dynamic turn's ``authored_history``
# scaffold is built progressively here instead of pre-authored).
# ---------------------------------------------------------------------------
def build_user_turn_prompt(
    seed: Dict[str, Any],
    speaker: str,
    history_text: str,
    intent: Optional[str],
) -> str:
    """Persona + running history (own language) + intent beat or free instruction.

    This is the chat-history-aware user simulation the legacy
    ``User.send_message()`` never had — history lives in the prompt; the
    provider itself stays stateless.
    """
    lang = seed["lang_a"] if speaker == "A" else seed["lang_b"]
    persona = seed.get("user_a_context") if speaker == "A" else seed.get("user_b_context")

    if intent:
        instruction = (
            f"For this turn, your goal is: {intent}. Express this goal naturally in your own words "
            "and in your own voice — do not recite it verbatim."
        )
    else:
        instruction = (
            "Continue the conversation naturally based on what has been said so far. Do not try to "
            "wrap up or end the conversation yet — the exchange continues for more turns."
        )

    return MULTITURN_USER_SIM_PROMPT.format(
        language_name=_lang_name(lang),
        persona=persona,
        conversation_context=seed.get("conversation_context"),
        history_block=history_text or "(this is the first turn; nothing has been said yet)",
        instruction=instruction,
    )


def _intent_for_turn(seed: Dict[str, Any], turn_index: int) -> Optional[str]:
    if seed.get("guidance") != "guided":
        return None
    beat = next(
        (b for b in (seed.get("intent_outline") or []) if b.get("turn_index") == turn_index), None
    )
    return beat.get("intent") if beat else None


def render_bilingual_history(history: Optional[List[Dict[str, Any]]]) -> str:
    """Render prior turns as ``Turn N (speaker): source -> translation`` lines.

    Shared by ``converse_next_turn``'s inline checklist-gen and
    ``run_mt_checklist_batch``'s post-hoc pass (plan Step 8 optimization) —
    the checklist a turn gets is identical either way; only *when* it's
    generated differs. Empty (not a placeholder) when there's no history,
    matching ``checklist_gen.generate_turn_checklist``'s own "no history"
    convention (an empty ``history_text`` omits the "Prior turns:" block).
    """
    if not history:
        return ""
    return "\n".join(
        f"Turn {h['turn_index']} ({h['speaker']}): {h.get('source_text')} -> {h.get('translated_text')}"
        for h in history
    )


def converse_next_turn(
    seed: Dict[str, Any],
    turn_index: int,
    history_entries: List[Dict[str, Any]],
    user_sim_factory: Callable[[str], Tuple[Any, str, str]],
    checklist_provider: Any,
    interpreter_provider: Any,
    interpreter_label: str,
    model_slug: str,
    prior_translations: Dict[Tuple[str, int], str],
    context_mode: str = "transcript",
    use_grounding: bool = True,
    skip_checklist: bool = False,
) -> Dict[str, Any]:
    """Produce ONE dynamic turn record: user-sim utterance -> checklist -> translate.

    Never raises: on any failure (user-sim, checklist-gen, or translation)
    returns a record with an ``error`` field and no ``translated_text``. The
    driver treats that as "abandon this conversation for this run" (plan D3)
    — ``history_entries`` (turn_index/speaker/source_text/translated_text) is
    the caller's running state, not mutated here.

    ``skip_checklist`` (plan Step 8 optimization): checklist generation is
    independent of translation either way (D4) — deferring it to a later,
    batched pass (``run_mt_checklist_batch``) instead of generating it inline
    here doesn't change what gets generated, only when/how cheaply. When
    True, ``checklist_provider`` is unused and may be ``None``.
    """
    conversation_id = seed["conversation_id"]
    speaker = "A" if turn_index % 2 == 0 else "B"
    lang_a = seed["lang_a"]
    lang_b = seed["lang_b"]
    source_lang = lang_a if speaker == "A" else lang_b
    target_lang = lang_b if speaker == "A" else lang_a

    base: Dict[str, Any] = {
        "segment_id": f"{conversation_id}_t{turn_index:02d}",
        "direction": f"{source_lang}-{target_lang}",
        "conversation_id": conversation_id,
        "mode": "dynamic",
        "guidance": seed.get("guidance"),
        "turn_index": turn_index,
        "num_turns": seed.get("num_turns"),
        "speaker": speaker,
        "lang_a": lang_a,
        "lang_b": lang_b,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "conversation_context": seed.get("conversation_context"),
        "user_a_context": seed.get("user_a_context"),
        "user_b_context": seed.get("user_b_context"),
        "listener_context": seed.get("user_b_context") if speaker == "A" else seed.get("user_a_context"),
        "category": seed.get("Category"),
        "seed_file": seed.get("seed_file"),
        "seed_row_id": seed.get("seed_row_id"),
        "intent": _intent_for_turn(seed, turn_index),
    }
    base["record_id"] = record_id(base)

    # 1. user-sim utterance (local, live — independent of translation/checklist)
    try:
        provider, _model_label, _lang_full = user_sim_factory(source_lang)
        history_own_lang = render_history_for_side(history_entries, speaker)
        prompt = build_user_turn_prompt(seed, speaker, history_own_lang, base["intent"])
        utterance = provider.generate(prompt, system_prompt=None)
    except Exception as e:  # noqa: BLE001
        out = dict(base)
        out["source_text"] = None
        out["translated_text"] = None
        out["error"] = f"user-sim: {e}"
        return out

    if not utterance or not utterance.strip():
        out = dict(base)
        out["source_text"] = None
        out["translated_text"] = None
        out["error"] = "user-sim: empty utterance"
        return out

    base["source_text"] = utterance

    # 2. checklist generation (cloud) — independent of translation; grounds on
    # the same bilingual transcript-so-far shape as scripted's turn checklists.
    # Skippable (plan Step 8): deferred to a later batched pass instead.
    if skip_checklist:
        base["checklist_items"] = []
        base["verification_prompt"] = None
    else:
        try:
            taxonomy = (
                cg.load_function_taxonomy(target_lang)
                if use_grounding and cg.taxonomy_available(target_lang)
                else None
            )
            history_bilingual = render_bilingual_history(history_entries)
            cultural_context = cg.get_cultural_context(lang_a, lang_b)
            items = cg.generate_turn_checklist(
                checklist_provider,
                target_lang,
                seed.get("conversation_context") or "",
                speaker,
                utterance,
                history_text=history_bilingual,
                taxonomy=taxonomy,
                cultural_context=cultural_context,
            )
            base["checklist_items"] = [item.model_dump() for item in items]
            base["verification_prompt"] = cg.compose_verification_prompt(items)
        except Exception as e:  # noqa: BLE001
            out = dict(base)
            out["translated_text"] = None
            out["error"] = f"checklist-gen: {e}"
            return out

    # 3. translation (cloud) — reuse the scripted translate operations
    # unchanged; `history_entries` is exactly the `authored_history` scaffold
    # those functions expect (turn_index/speaker/source_text).
    base["authored_history"] = [
        {"turn_index": h["turn_index"], "speaker": h["speaker"], "source_text": h.get("source_text")}
        for h in history_entries
    ]
    return translate_turn_record(
        base, interpreter_provider, interpreter_label, model_slug, prior_translations, context_mode
    )


# ---------------------------------------------------------------------------
# Stage: mt-consolidate — one line per conversation (+ flat results_turns.jsonl)
# ---------------------------------------------------------------------------
def _sorted_checklist_items(checklist_items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Re-sort a checklist L3->L2->L1, exactly how ``compose_verification_prompt``
    numbered it — judge result ``id`` fields are 1-indexed in that same order,
    so this is what lets a judge criterion be joined back to its function_id.
    """
    return sorted(checklist_items or [], key=lambda it: cg._LAYER_PRIORITY.get(it.get("layer", ""), 3))


def _join_criteria(
    checklist_items: Optional[List[Dict[str, Any]]], evaluation: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    sorted_items = _sorted_checklist_items(checklist_items)
    results = (evaluation or {}).get("results", [])
    criteria = []
    for i, res in enumerate(results):
        item = sorted_items[i] if i < len(sorted_items) else {}
        criteria.append(
            {
                "function_id": item.get("function_id"),
                "layer": item.get("layer"),
                "criteria": res.get("criteria"),
                "met": res.get("met"),
                "reasoning": res.get("reasoning"),
            }
        )
    return criteria


def consolidate_conversation(
    turn_records: List[Dict[str, Any]],
    conv_judge_record: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the consolidated conversation line + its flat per-turn lines.

    ``model`` is the canonical join key (results-store convention) — the
    interpreter model slug, consistent across every turn of one conversation.
    Returns ``(conversation_line, turn_lines)``; the caller writes
    conversation_line to ``results.jsonl`` and extends ``results_turns.jsonl``
    with turn_lines.
    """
    ordered_turns = sorted(turn_records, key=lambda r: r["turn_index"])
    first = ordered_turns[0] if ordered_turns else {}
    conversation_id = conv_judge_record.get("conversation_id")

    turns_nested = []
    turn_lines = []
    turn_success_rates = []
    all_lang_checks_passed = True
    for t in ordered_turns:
        criteria = _join_criteria(t.get("checklist_items"), t.get("evaluation"))
        shared_turn_fields = {
            "turn_index": t.get("turn_index"),
            "speaker": t.get("speaker"),
            "source_lang": t.get("source_lang"),
            "target_lang": t.get("target_lang"),
            "source_text": t.get("source_text"),
            "translated_text": t.get("translated_text"),
            "intent": t.get("intent"),
            "success_rate": t.get("success_rate", 0.0),
            "completion_rate": t.get("completion_rate", "0/0"),
            "criteria": criteria,
            "language_check_passed": t.get("language_check_passed", True),
        }
        turns_nested.append(dict(shared_turn_fields))
        turn_lines.append(
            {
                "record_id": t.get("record_id"),
                "conversation_id": conversation_id,
                "model": t.get("model"),
                "mode": t.get("mode"),
                "guidance": t.get("guidance"),
                "context_mode": t.get("context_mode"),
                **shared_turn_fields,
            }
        )
        turn_success_rates.append(t.get("success_rate", 0.0))
        if not t.get("language_check_passed", True):
            all_lang_checks_passed = False

    conversation_criteria = _join_criteria(
        conv_judge_record.get("conversation_checklist_items"), conv_judge_record.get("evaluation")
    )

    conversation_line = {
        "record_id": conv_judge_record.get("record_id"),
        "conversation_id": conversation_id,
        "model": first.get("model"),
        "interpreter": first.get("interpreter"),
        "mode": conv_judge_record.get("mode"),
        "guidance": conv_judge_record.get("guidance"),
        "context_mode": first.get("context_mode"),
        "lang_a": conv_judge_record.get("lang_a"),
        "lang_b": conv_judge_record.get("lang_b"),
        "conversation_context": conv_judge_record.get("conversation_context"),
        "num_turns": len(ordered_turns),
        "turns": turns_nested,
        "conversation_criteria": conversation_criteria,
        "checklist_provenance": conv_judge_record.get("checklist_provenance"),
        "conversation_completion_rate": conv_judge_record.get("completion_rate", "0/0"),
        "conversation_success_rate": conv_judge_record.get("success_rate", 0.0),
        "mean_turn_success_rate": (
            sum(turn_success_rates) / len(turn_success_rates) if turn_success_rates else 0.0
        ),
        "all_language_checks_passed": all_lang_checks_passed,
    }
    return conversation_line, turn_lines
