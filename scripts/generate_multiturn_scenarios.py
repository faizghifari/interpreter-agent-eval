"""Generate multi-turn scripted transcripts and dynamic seeds (plan D7).

Sibling of ``generate_maps_scenarios.py``, reusing its seed-xlsx loading
(``parse_seed_rows``) and ``augment_opensubs_maps.py``'s cultural-context-pair
paragraphs. Checklist generation is delegated once to
``pipeline/multiturn/checklist_gen.py`` (D4) so scripted and dynamic scenarios
are never confounded by different checklist logic.

Two independent outputs, selected by ``--mode``:
- scripted (``*_mts_<tag>.jsonl``): a fully pre-authored N-turn transcript
  (one LLM call) plus a checklist per turn and one conversation-level
  checklist (each an independent ``checklist_gen`` call — batchable later).
- dynamic (``*_mtd_<tag>.jsonl``, ``--guidance guided|free``): a seed (personas
  + premise + num_turns); ``guided`` additionally gets a per-turn intent
  outline and a seed-time conversation-level checklist authored from it;
  ``free`` gets neither (its conversation checklist is generated post-hoc,
  inside mt-judge-conv, once a real transcript exists).
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)
sys.path.append(CURRENT_DIR)

from interpreter_agent_eval.pipeline import registry  # noqa: E402
from interpreter_agent_eval.pipeline.batch import (  # noqa: E402
    FAILED,
    TERMINAL,
    BatchRequest,
    build_batch_client,
)
from interpreter_agent_eval.pipeline.multiturn import checklist_gen as cg  # noqa: E402
from interpreter_agent_eval.pipeline.multiturn.checklist_gen import ChecklistItem  # noqa: E402
from interpreter_agent_eval.utils.language_verification import (  # noqa: E402
    load_glotlid_model,
    verify_language_with_glotlid,
)

from generate_maps_scenarios import parse_seed_rows  # noqa: E402 — reuse seed-xlsx loading
from augment_opensubs_maps import CULTURAL_CONTEXT  # noqa: E402 — reuse cultural-context paragraphs

load_dotenv()

# 2-letter (CLI/file-naming convention) <-> ISO 639-3 (record convention).
LANG_ISO3 = {"id": "ind", "ko": "kor", "ar": "arb", "bn": "ben"}
ISO3_TO_2 = {v: k for k, v in LANG_ISO3.items()}
LANG_NAME = cg.LANG_NAMES

MIN_TURNS, MAX_TURNS, DEFAULT_TURNS = 4, 8, 6  # plan D6


# ---------------------------------------------------------------------------
# LLM structured-output schemas (generation-time only; not stored as-is)
# ---------------------------------------------------------------------------
class GeneratedTurn(BaseModel):
    turn_index: int = Field(description="0-based turn index")
    speaker: str = Field(description="Exactly the single letter 'A' or 'B' — never a name or 'User A'")
    text: str = Field(description="Utterance text, entirely in this speaker's own language")


class GeneratedTranscript(BaseModel):
    conversation_context: str = Field(description="One short English sentence, surface-level only")
    user_a_context: str = Field(description="Persona/backstory for User A, in User A's language")
    user_b_context: str = Field(description="Persona/backstory for User B, in User B's language")
    turns: List[GeneratedTurn]


class GeneratedIntentBeat(BaseModel):
    turn_index: int
    speaker: str = Field(description="Exactly the single letter 'A' or 'B' — never a name or 'User A'")
    intent: str = Field(description="One-line English beat describing that turn's goal")


class GeneratedSeed(BaseModel):
    conversation_context: str
    user_a_context: str
    user_b_context: str
    intent_outline: Optional[List[GeneratedIntentBeat]] = None


# ---------------------------------------------------------------------------
# Stored record schemas (script-local per the plan's architecture table;
# only ChecklistItem lives in checklist_gen.py — the pipeline works on dicts)
# ---------------------------------------------------------------------------
class TurnRecord(BaseModel):
    turn_index: int
    speaker: str
    text: str
    checklist_items: List[ChecklistItem]
    verification_prompt: str


class MultiTurnScenario(BaseModel):
    conversation_id: str
    mode: str = "scripted"
    lang_a: str
    lang_b: str
    Category: str
    conversation_context: str
    user_a_context: str
    user_b_context: str
    turns: List[TurnRecord]
    conversation_checklist_items: List[ChecklistItem]
    conversation_verification_prompt: str
    seed_file: Optional[str] = None
    seed_row_id: Optional[Any] = None
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


class IntentBeatRecord(BaseModel):
    turn_index: int
    speaker: str
    intent: str


class DynamicSeed(BaseModel):
    conversation_id: str
    mode: str = "dynamic"
    guidance: str
    lang_a: str
    lang_b: str
    Category: str
    conversation_context: str
    user_a_context: str
    user_b_context: str
    num_turns: int
    intent_outline: Optional[List[IntentBeatRecord]] = None
    conversation_checklist_items: Optional[List[ChecklistItem]] = None
    conversation_verification_prompt: Optional[str] = None
    seed_file: Optional[str] = None
    seed_row_id: Optional[Any] = None
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt templates (script-local — NOT appended to prompts/templates.py,
# mirroring how generate_maps_scenarios.py keeps its own PROMPT_TEMPLATE local)
# ---------------------------------------------------------------------------
SCRIPTED_TRANSCRIPT_PROMPT = """You are an expert cross-cultural linguist designing a multi-turn dialogue simulation for evaluating an AI interpreter.

{seed_block}Required conversation:
- User A speaks {lang_a_name} ({lang_a_code}) and speaks first (turn 0, 2, 4, ...).
- User B speaks {lang_b_name} ({lang_b_code}) (turn 1, 3, 5, ...).
- Exactly {num_turns} turns total, strictly alternating starting with A.

{cultural_context_block}Task:
1) Build a realistic, natural back-and-forth conversation of exactly {num_turns} turns between User A and User B{seed_grounding_clause}.
2) Each turn's text must be written entirely in that turn's speaker's own language — User A's turns in {lang_a_name}, User B's turns in {lang_b_name}.
3) Build a realistic conversation_context (one short English sentence, surface-level only — no leaked constraints, cultural traps, or expected outcomes) and user_a_context / user_b_context (persona/backstory, in the respective user's own language, with no hints about what the other side will say or how they should react).
4) The dialogue must read as natural for two ordinary people — do not have either side over-explain, define terms, or narrate evaluation criteria.

Output constraints:
- turns must alternate strictly A,B,A,B,... starting with A; turn_index is 0-based and sequential.
- conversation_context: exactly one short English sentence.
- user_a_context / user_b_context: persona/backstory only, no hints about the other side's expected reaction.

Output ONLY a JSON object with exactly these fields, no markdown fencing, no commentary:
{{"conversation_context": "<one English sentence>", "user_a_context": "<persona in {lang_a_name}>", "user_b_context": "<persona in {lang_b_name}>", "turns": [{{"turn_index": 0, "speaker": "A", "text": "<utterance>"}}, ...]}}
"""

DYNAMIC_SEED_PROMPT = """You are designing a seed for a live-simulated multi-turn cross-cultural conversation, to be role-played by two separate AI personas without a pre-written script.

{seed_block}Required setup:
- User A speaks {lang_a_name} ({lang_a_code}) and speaks first.
- User B speaks {lang_b_name} ({lang_b_code}).
- The conversation will run exactly {num_turns} turns, strictly alternating starting with A.

{cultural_context_block}Task:
1) Build a realistic conversation_context (one short English sentence, surface-level, no leaked outcomes) and user_a_context / user_b_context (persona/backstory in the respective user's own language, no hints about the other side's expected behavior) that will let two independent AI personas plausibly improvise a conversation{seed_grounding_clause}, without ever being told the seed idea or the checklist.
{intent_instruction}
"""


def _render_opensubs_context_lines(context: List[Dict[str, Any]]) -> str:
    return "\n".join(f"  {c.get('source_text')} / {c.get('target_text')}" for c in context)


def _opensubs_seed_block(row: Dict[str, Any]) -> str:
    prev_lines = _render_opensubs_context_lines(row.get("prev_context") or [])
    after_lines = _render_opensubs_context_lines(row.get("after_context") or [])
    reasons = row.get("reasons", "") or "(none recorded)"
    return (
        "Seed Data (real OpenSubtitles bilingual excerpt — for cultural/pragmatic INSPIRATION\n"
        "only; do NOT reuse any line verbatim and do NOT reference a film, character, or plot):\n"
        f"{prev_lines}\n"
        f"-> anchor: {row.get('source_text')} / {row.get('target_text')}\n"
        f"{after_lines}\n"
        f"What makes this excerpt hard to translate well: {reasons}\n\n"
    )


def _seed_block(row: Optional[Dict[str, Any]], seed_split: str) -> str:
    if row is None:
        return ""
    if row.get("_kind") == "opensubs":
        return _opensubs_seed_block(row)
    return (
        f"Seed Data ({seed_split}):\n"
        f"- Proverb: {row['proverb']}\n"
        f"- Conversation Seed: {row['conversation']}\n"
        f"- Explanation Seed (noisy/weak quality possible): {row['explanation']}\n"
        f"- Candidate Meaning A: {row['answer1']}\n"
        f"- Candidate Meaning B: {row['answer2']}\n"
        f"- Figurative Flag: {row['is_figurative']}\n"
        f"- Annotated Key: {row['answer_key']}\n\n"
    )


def _cultural_block(cultural_context: Optional[str]) -> str:
    return f"{cultural_context}\n\n" if cultural_context else ""


def _grounding_clause(row: Optional[Dict[str, Any]], topic_hint: Optional[str]) -> str:
    if row is not None:
        if row.get("_kind") == "opensubs":
            return (
                " built around a similar kind of interpretation challenge to the excerpt above "
                "(matching its cultural/pragmatic difficulty), adapted into a natural, ordinary "
                "two-party premise of your own invention — not the excerpt's topic, plot, or characters"
            )
        return (
            " that grows out of the proverb/seed idea above (its pragmatic point should surface "
            "naturally in how the conversation unfolds, not be quoted or defined verbatim)"
        )
    if topic_hint:
        return f" on the following everyday topic: {topic_hint}"
    return " on a natural everyday topic between two ordinary people"


def build_scripted_prompt(
    row: Optional[Dict[str, Any]],
    seed_split: str,
    lang_a: str,
    lang_b: str,
    num_turns: int,
    cultural_context: Optional[str],
    topic_hint: Optional[str] = None,
) -> str:
    return SCRIPTED_TRANSCRIPT_PROMPT.format(
        seed_block=_seed_block(row, seed_split),
        lang_a_name=LANG_NAME[lang_a],
        lang_a_code=lang_a,
        lang_b_name=LANG_NAME[lang_b],
        lang_b_code=lang_b,
        num_turns=num_turns,
        cultural_context_block=_cultural_block(cultural_context),
        seed_grounding_clause=_grounding_clause(row, topic_hint),
    )


def build_dynamic_prompt(
    row: Optional[Dict[str, Any]],
    seed_split: str,
    lang_a: str,
    lang_b: str,
    num_turns: int,
    guidance: str,
    cultural_context: Optional[str],
    topic_hint: Optional[str] = None,
) -> str:
    if guidance == "guided":
        intent_instruction = (
            f"2) Also produce an intent_outline: exactly {num_turns} beats, one per turn, alternating "
            "A/B starting with A (turn_index 0-based), each a short ENGLISH one-line description of "
            "what that speaker should try to accomplish/say in that turn (e.g. 'politely decline, "
            "citing family') — concrete enough to keep a small local model on-trajectory toward the "
            "seed idea's pragmatic point, but not a scripted line."
        )
    else:
        intent_instruction = (
            "2) Do NOT produce an intent_outline (omit it) — the two personas will improvise turn by "
            "turn from conversation_context and their own persona alone."
        )
    return DYNAMIC_SEED_PROMPT.format(
        seed_block=_seed_block(row, seed_split),
        lang_a_name=LANG_NAME[lang_a],
        lang_a_code=lang_a,
        lang_b_name=LANG_NAME[lang_b],
        lang_b_code=lang_b,
        num_turns=num_turns,
        cultural_context_block=_cultural_block(cultural_context),
        seed_grounding_clause=_grounding_clause(row, topic_hint),
        intent_instruction=intent_instruction,
    )


# ---------------------------------------------------------------------------
# Validators (D7: strict A/B alternation starting with A; 4<=N<=8; GlotLID
# sanity for scripted; checklist caps + non-empty enforced via checklist_gen)
# ---------------------------------------------------------------------------
def _strip_json_fences(text: str) -> str:
    """Defensively strip a ```json ... ``` wrapper some batch responses add.

    The batch backend's ``config={"json": True}`` doesn't guarantee raw JSON
    the way the sync path's ``response_mime_type``/``response_schema`` does —
    observed on dynamic seed-gen batch responses (scripted's transcript-gen
    batch responses haven't needed this, but stripping defensively costs
    nothing when there's no fence to strip).
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _coerce_intent_outline(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Repair a flat list-of-strings ``intent_outline`` into the expected shape.

    The batch backend's ``config={"json": True}`` doesn't enforce
    ``response_schema`` the way the sync path does, and the guided-mode
    prompt's "produce an intent_outline: exactly N beats..." wording doesn't
    show an explicit object shape — observed result: the model sometimes
    returns ``["beat 1 text", "beat 2 text", ...]`` instead of
    ``[{"turn_index": 0, "speaker": "A", "intent": "beat 1 text"}, ...]``.
    Turn index and alternating A/B speaker are fully recoverable from list
    position, so this is a safe, lossless repair, not a guess.
    """
    outline = raw.get("intent_outline")
    if not isinstance(outline, list) or not outline:
        return raw

    def _alt_speaker(i: int) -> str:
        return "A" if i % 2 == 0 else "B"

    _INTENT_ALIASES = ("intent", "text", "content", "beat", "message", "utterance", "user")

    def _find_intent(b: Dict[str, Any]) -> Any:
        for key in _INTENT_ALIASES:
            if key in b and isinstance(b[key], str):
                return b[key]
        for key, val in b.items():
            if key not in ("turn_index", "speaker") and isinstance(val, str):
                return val
        return None

    if isinstance(outline[0], str):
        raw = dict(raw)
        raw["intent_outline"] = [
            {"turn_index": i, "speaker": _alt_speaker(i), "intent": text}
            for i, text in enumerate(outline)
        ]
    elif isinstance(outline[0], dict) and any(
        "speaker" not in b or "intent" not in b for b in outline
    ):
        raw = dict(raw)
        raw["intent_outline"] = [
            {
                "turn_index": b.get("turn_index", i),
                "speaker": b.get("speaker") or _alt_speaker(b.get("turn_index", i)),
                "intent": b.get("intent") if isinstance(b.get("intent"), str) else _find_intent(b),
            }
            for i, b in enumerate(outline)
        ]
    return raw


def validate_num_turns(n: int) -> List[str]:
    if not (MIN_TURNS <= n <= MAX_TURNS):
        return [f"num_turns={n} outside allowed range [{MIN_TURNS},{MAX_TURNS}]"]
    return []


def validate_alternation(items: List[Any]) -> List[str]:
    """Check strict 0-based, A/B-alternating-starting-with-A ordering.

    Works on anything with ``.turn_index`` / ``.speaker`` — both
    ``GeneratedTurn`` and ``GeneratedIntentBeat`` satisfy this.
    """
    errors: List[str] = []
    if not items:
        errors.append("no items generated")
        return errors
    ordered = sorted(items, key=lambda it: it.turn_index)
    for i, it in enumerate(ordered):
        if it.turn_index != i:
            errors.append(f"turn_index not sequential/0-based at position {i} (got {it.turn_index})")
        expected_speaker = "A" if i % 2 == 0 else "B"
        if it.speaker != expected_speaker:
            errors.append(
                f"turn {i} speaker '{it.speaker}' != expected '{expected_speaker}' "
                "(must alternate A,B,... starting with A)"
            )
        text = getattr(it, "text", None) or getattr(it, "intent", None)
        if not text or not str(text).strip():
            errors.append(f"turn {i} has empty text/intent")
    return errors


def glotlid_sanity_check(
    turns: List[GeneratedTurn], lang_a: str, lang_b: str, glotlid_model: Any
) -> List[str]:
    """Per-turn language sanity check (scripted only); no-op without a model."""
    if glotlid_model is None:
        return []
    errors = []
    for t in turns:
        expected = lang_a if t.speaker == "A" else lang_b
        v = verify_language_with_glotlid(
            model=glotlid_model,
            text=t.text,
            expected_iso_code=expected,
            min_confidence=0.5,
            context_name=f"turn {t.turn_index}",
        )
        if not v.is_correct and not getattr(v, "needs_review", False):
            errors.append(f"turn {t.turn_index} ({t.speaker}) failed GlotLID sanity check: {v.message}")
    return errors


# ---------------------------------------------------------------------------
# Retry loop (mirrors generate_maps_scenarios.augment_maps_data's structure)
# ---------------------------------------------------------------------------
def _retry(fn, max_retries: int = 3, sleep_s: float = 2.0, label: str = ""):
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    {label} attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Generation: one scripted scenario / one dynamic seed
# ---------------------------------------------------------------------------
def parse_and_validate_transcript(
    text: str, lang_a: str, lang_b: str, num_turns: int, glotlid_model: Any
) -> GeneratedTranscript:
    """Shared by the sync retry loop and the batch path — parsing/validation
    is identical either way; only how a failure is handled differs (sync
    retries immediately, batch just skips the conversation this run).
    """
    transcript = GeneratedTranscript.model_validate_json(text)
    errors = validate_num_turns(len(transcript.turns)) + validate_alternation(transcript.turns)
    errors += glotlid_sanity_check(transcript.turns, lang_a, lang_b, glotlid_model)
    if len(transcript.turns) != num_turns:
        errors.append(f"generated {len(transcript.turns)} turns, expected exactly {num_turns}")
    if errors:
        raise ValueError("; ".join(errors))
    return transcript


def generate_transcript_only(
    transcript_provider: Any,
    prompt: str,
    lang_a: str,
    lang_b: str,
    num_turns: int,
    glotlid_model: Any,
    conversation_id: str,
) -> GeneratedTranscript:
    """Phase 1 of scripted generation, isolated for reuse by the batch checklist
    path (``run_scripted_batch``) — transcripts are generated synchronously
    either way; only the checklist calls that follow get batched.
    """

    def _gen() -> GeneratedTranscript:
        text = transcript_provider.generate(
            prompt, response_mime_type="application/json", response_schema=GeneratedTranscript
        )
        return parse_and_validate_transcript(text, lang_a, lang_b, num_turns, glotlid_model)

    return _retry(_gen, label=f"{conversation_id} transcript")


def generate_one_scripted_scenario(
    transcript_provider: Any,
    checklist_provider: Any,
    prompt: str,
    lang_a: str,
    lang_b: str,
    num_turns: int,
    glotlid_model: Any,
    use_grounding: bool,
    seed_file: Optional[str],
    seed_row_id: Optional[Any],
    category: str,
    conversation_id: str,
    cultural_context: Optional[str] = None,
) -> MultiTurnScenario:
    transcript = generate_transcript_only(
        transcript_provider, prompt, lang_a, lang_b, num_turns, glotlid_model, conversation_id
    )

    turns_sorted = sorted(transcript.turns, key=lambda t: t.turn_index)
    turn_records: List[TurnRecord] = []
    history_lines: List[str] = []
    for t in turns_sorted:
        target_lang = lang_b if t.speaker == "A" else lang_a
        taxonomy = (
            cg.load_function_taxonomy(target_lang)
            if use_grounding and cg.taxonomy_available(target_lang)
            else None
        )
        history_text = "\n".join(history_lines)

        def _gen_turn_checklist(t=t, target_lang=target_lang, taxonomy=taxonomy, history_text=history_text):
            items = cg.generate_turn_checklist(
                checklist_provider,
                target_lang,
                transcript.conversation_context,
                t.speaker,
                t.text,
                history_text=history_text,
                taxonomy=taxonomy,
                cultural_context=cultural_context,
            )
            errs = cg.validate_checklist_items(items, cg.TURN_HARD_CEILING)
            if errs:
                raise ValueError("; ".join(errs))
            return items

        items = _retry(_gen_turn_checklist, label=f"{conversation_id} turn {t.turn_index} checklist")
        note = cg.checklist_count_note(items, cg.TURN_ITEM_CAP, cg.TURN_HARD_CEILING)
        if note:
            print(f"    {conversation_id} turn {t.turn_index}: {note}")
        turn_records.append(
            TurnRecord(
                turn_index=t.turn_index,
                speaker=t.speaker,
                text=t.text,
                checklist_items=items,
                verification_prompt=cg.compose_verification_prompt(items),
            )
        )
        history_lines.append(f"Turn {t.turn_index} ({t.speaker}): {t.text}")

    # Conversation-level: the 53-function taxonomy is identical across target
    # languages (only phrasing/examples differ per-language) and a conversation
    # exercises both translation directions, so lang_b's taxonomy is used by
    # convention for cross-turn items regardless of which turn they reference.
    conv_taxonomy = (
        cg.load_function_taxonomy(lang_b) if use_grounding and cg.taxonomy_available(lang_b) else None
    )
    transcript_basis = "\n".join(history_lines)

    def _gen_conv_checklist():
        items = cg.generate_conversation_checklist(
            checklist_provider,
            lang_b,
            transcript.conversation_context,
            transcript_basis,
            taxonomy=conv_taxonomy,
            cultural_context=cultural_context,
        )
        errs = cg.validate_checklist_items(items, cg.CONVERSATION_HARD_CEILING)
        if errs:
            raise ValueError("; ".join(errs))
        return items

    conv_items = _retry(_gen_conv_checklist, label=f"{conversation_id} conversation checklist")
    conv_note = cg.checklist_count_note(conv_items, cg.CONVERSATION_ITEM_CAP, cg.CONVERSATION_HARD_CEILING)
    if conv_note:
        print(f"    {conversation_id} conversation: {conv_note}")

    return MultiTurnScenario(
        conversation_id=conversation_id,
        mode="scripted",
        lang_a=lang_a,
        lang_b=lang_b,
        Category=category,
        conversation_context=transcript.conversation_context,
        user_a_context=transcript.user_a_context,
        user_b_context=transcript.user_b_context,
        turns=turn_records,
        conversation_checklist_items=conv_items,
        conversation_verification_prompt=cg.compose_verification_prompt(conv_items),
        seed_file=seed_file,
        seed_row_id=seed_row_id,
        generation_metadata={
            "transcript_model": getattr(transcript_provider, "model_name", None),
            "checklist_model": getattr(checklist_provider, "model_name", None),
            "function_grounding": use_grounding,
            "cultural_context_used": cultural_context is not None,
        },
    )


def generate_one_dynamic_seed(
    seed_provider: Any,
    checklist_provider: Any,
    prompt: str,
    lang_a: str,
    lang_b: str,
    num_turns: int,
    guidance: str,
    use_grounding: bool,
    seed_file: Optional[str],
    seed_row_id: Optional[Any],
    category: str,
    conversation_id: str,
    cultural_context: Optional[str] = None,
) -> DynamicSeed:
    def _gen_seed() -> GeneratedSeed:
        text = seed_provider.generate(
            prompt, response_mime_type="application/json", response_schema=GeneratedSeed
        )
        seed = GeneratedSeed.model_validate_json(text)
        errors: List[str] = []
        if guidance == "guided":
            if not seed.intent_outline:
                errors.append("guided seed missing intent_outline")
            else:
                errors += validate_num_turns(len(seed.intent_outline))
                errors += validate_alternation(seed.intent_outline)
        if errors:
            raise ValueError("; ".join(errors))
        return seed

    seed = _retry(_gen_seed, label=f"{conversation_id} dynamic seed")

    intent_outline_records: Optional[List[IntentBeatRecord]] = None
    conv_items: Optional[List[ChecklistItem]] = None
    conv_vprompt: Optional[str] = None

    if guidance == "guided":
        ordered = sorted(seed.intent_outline, key=lambda b: b.turn_index)
        intent_outline_records = [
            IntentBeatRecord(turn_index=b.turn_index, speaker=b.speaker, intent=b.intent) for b in ordered
        ]
        basis = "\n".join(f"Turn {b.turn_index} ({b.speaker}) intent: {b.intent}" for b in intent_outline_records)
        conv_taxonomy = (
            cg.load_function_taxonomy(lang_b) if use_grounding and cg.taxonomy_available(lang_b) else None
        )

        def _gen_conv_checklist():
            items = cg.generate_conversation_checklist(
                checklist_provider,
                lang_b,
                seed.conversation_context,
                basis,
                taxonomy=conv_taxonomy,
                cultural_context=cultural_context,
            )
            errs = cg.validate_checklist_items(items, cg.CONVERSATION_HARD_CEILING)
            if errs:
                raise ValueError("; ".join(errs))
            return items

        conv_items = _retry(_gen_conv_checklist, label=f"{conversation_id} conversation checklist")
        conv_vprompt = cg.compose_verification_prompt(conv_items)

    return DynamicSeed(
        conversation_id=conversation_id,
        mode="dynamic",
        guidance=guidance,
        lang_a=lang_a,
        lang_b=lang_b,
        Category=category,
        conversation_context=seed.conversation_context,
        user_a_context=seed.user_a_context,
        user_b_context=seed.user_b_context,
        num_turns=num_turns,
        intent_outline=intent_outline_records,
        conversation_checklist_items=conv_items,
        conversation_verification_prompt=conv_vprompt,
        seed_file=seed_file,
        seed_row_id=seed_row_id,
        generation_metadata={
            "seed_model": getattr(seed_provider, "model_name", None),
            "checklist_model": getattr(checklist_provider, "model_name", None) if guidance == "guided" else None,
            "function_grounding": use_grounding,
            "cultural_context_used": cultural_context is not None if guidance == "guided" else None,
        },
    )


# ---------------------------------------------------------------------------
# I/O helpers (mirror generate_maps_scenarios.py's append/dedup pattern)
# ---------------------------------------------------------------------------
def load_existing_conversation_ids(output_path: str) -> set:
    ids: set = set()
    if not os.path.exists(output_path):
        return ids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cid = json.loads(line).get("conversation_id")
            except json.JSONDecodeError:
                continue
            if cid:
                ids.add(cid)
    return ids


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_pair(pair_arg: str):
    two_a, two_b = pair_arg.split("-")
    if two_a not in LANG_ISO3 or two_b not in LANG_ISO3:
        raise ValueError(f"Unknown pair '{pair_arg}'. Known 2-letter codes: {sorted(LANG_ISO3)}")
    return two_a, two_b, LANG_ISO3[two_a], LANG_ISO3[two_b]


def _cultural_context_for(two_a: str, two_b: str) -> Optional[str]:
    return CULTURAL_CONTEXT.get("-".join(sorted([two_a, two_b])))


def _opensubs_top500_path(two_a: str, two_b: str) -> str:
    pair_key = "-".join(sorted([two_a, two_b]))
    return os.path.join(ROOT_DIR, "outputs", "opensubs_pipeline", "top500", pair_key, "top500.jsonl")


def load_opensubs_windows(two_a: str, two_b: str, start_row: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Real, already-quality-scored OpenSubtitles bilingual dialogue windows —
    the same source data ``augment_opensubs_maps.py`` used for the completed
    single-turn grid on pairs with no MAPS_Final proverb seed (e.g. any pair
    involving Arabic). Each window carries ``prev_context``/``after_context``
    (real prior/following subtitle lines) around an anchor line, used here as
    seed *inspiration* for a new multi-turn transcript — not replayed verbatim.
    """
    path = _opensubs_top500_path(two_a, two_b)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No OpenSubtitles top500 window data at {path}. This pair may not have been through "
            "scripts/opensubs_pipeline.py yet."
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["_kind"] = "opensubs"
            d["seed_row_id"] = i
            rows.append(d)
    rows = [r for r in rows if r["seed_row_id"] >= start_row]
    return rows[:limit] if limit is not None else rows


def _load_rows(args, two_a: str, two_b: str) -> Optional[List[Dict[str, Any]]]:
    if args.no_seed:
        return None
    if args.seed_source == "opensubs":
        return load_opensubs_windows(two_a, two_b, args.start_row, args.num_scenarios)
    seed_xlsx = args.seed_xlsx or f"data/MAPS_Final/{two_a}/test_proverbs.xlsx"
    if not os.path.exists(seed_xlsx):
        raise FileNotFoundError(
            f"No seed spreadsheet at {seed_xlsx}. Pass --seed-xlsx explicitly, --seed-source opensubs "
            "(real OpenSubtitles window data — check it exists for this pair first), or --no-seed for "
            "a hand-authored/topic-hint premise."
        )
    rows = parse_seed_rows(seed_xlsx, args.seed_split)
    rows = [r for r in rows if r["seed_row_id"] >= args.start_row]
    return rows[: args.num_scenarios]


def _seed_provenance_name(args, two_a: str, two_b: str) -> Optional[str]:
    """Human-readable seed-file name for the ``seed_file`` provenance field."""
    if args.no_seed:
        return None
    if args.seed_source == "opensubs":
        return os.path.basename(_opensubs_top500_path(two_a, two_b))
    return os.path.basename(args.seed_xlsx or f"data/MAPS_Final/{two_a}/test_proverbs.xlsx")


def _category_for(args, rows: Optional[List[Dict[str, Any]]]) -> str:
    if rows is None:
        return "Hand-Authored-MT"
    if args.seed_source == "opensubs":
        return "OpenSubs-Inspired-MT"
    return "MAPS-Proverb-Pragmatics-MT"


def run_scripted(args) -> Optional[str]:
    two_a, two_b, lang_a, lang_b = _resolve_pair(args.pair)
    cultural_context = _cultural_context_for(two_a, two_b)

    use_grounding = not args.no_function_grounding
    if use_grounding:
        cg.assert_taxonomies_available((lang_a, lang_b))
    else:
        print("[warn] --no-function-grounding set: checklists will be ungrounded (function_id=null).")

    rows = _load_rows(args, two_a, two_b)
    seed_xlsx_name = None
    if rows is not None:
        if not rows:
            print("No seed rows available after filters.")
            return None
        seed_xlsx_name = _seed_provenance_name(args, two_a, two_b)

    topic_hints = [h.strip() for h in args.topic_hints.split(",")] if args.topic_hints else [None]
    n = len(rows) if rows is not None else args.num_scenarios
    category = _category_for(args, rows)

    output_path = os.path.join(args.output_dir, f"{two_a}_{two_b}_mts_{args.tag}.jsonl")
    existing = load_existing_conversation_ids(output_path)

    transcript_provider = checklist_provider = glotlid_model = None
    if not args.dry_run:
        transcript_provider = registry.build_interpreter_provider(
            args.transcript_provider, args.transcript_model, args.transcript_thinking
        )
        checklist_provider = registry.build_judge_provider(
            args.checklist_provider, args.checklist_model, args.checklist_thinking
        )
        if args.verify_language:
            glotlid_model = load_glotlid_model()

    for i in range(n):
        row = rows[i] if rows is not None else None
        topic_hint = topic_hints[i % len(topic_hints)]
        conversation_id = f"{lang_a}{lang_b}_mts_{i + 1:04d}"
        prompt = build_scripted_prompt(
            row, args.seed_split, lang_a, lang_b, args.num_turns, cultural_context, topic_hint
        )

        if args.dry_run:
            print(f"\n=== [dry-run] scripted transcript prompt ({conversation_id}) ===\n{prompt}")
            continue

        if conversation_id in existing:
            print(f"  [{i + 1}/{n}] {conversation_id} skipped (already exists)")
            continue

        try:
            scenario = generate_one_scripted_scenario(
                transcript_provider,
                checklist_provider,
                prompt,
                lang_a,
                lang_b,
                args.num_turns,
                glotlid_model,
                use_grounding,
                seed_file=seed_xlsx_name or "hand-authored",
                seed_row_id=row.get("seed_row_id") if row else None,
                category=category,
                conversation_id=conversation_id,
                cultural_context=cultural_context,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{i + 1}/{n}] {conversation_id} FAILED: {e}")
            continue

        append_jsonl(output_path, scenario.model_dump())
        existing.add(conversation_id)
        print(f"  [{i + 1}/{n}] {conversation_id} generated ({len(scenario.turns)} turns)")

    if args.dry_run:
        return None
    print(f"\nOutput: {output_path}")
    return output_path


def run_scripted_batch(args) -> Optional[str]:
    """Scripted generation, batch checklist backend (plan Step 8).

    Two-phase: all transcripts are generated synchronously first (cheap
    relative to the checklist calls, and each needs its own retry loop), then
    every turn checklist + conversation checklist across ALL pending
    conversations is submitted as ONE batch job. Unlike the pipeline stages,
    there's no per-conversation resume within the checklist phase — this is a
    one-shot setup script, not a long-running experiment; a failed job means
    re-running the whole command (already-written conversations are skipped
    via the usual ``existing`` dedup).
    """
    two_a, two_b, lang_a, lang_b = _resolve_pair(args.pair)
    cultural_context = _cultural_context_for(two_a, two_b)

    use_grounding = not args.no_function_grounding
    if use_grounding:
        cg.assert_taxonomies_available((lang_a, lang_b))
    else:
        print("[warn] --no-function-grounding set: checklists will be ungrounded (function_id=null).")

    rows = _load_rows(args, two_a, two_b)
    seed_xlsx_name = None
    if rows is not None:
        if not rows:
            print("No seed rows available after filters.")
            return None
        seed_xlsx_name = _seed_provenance_name(args, two_a, two_b)

    topic_hints = [h.strip() for h in args.topic_hints.split(",")] if args.topic_hints else [None]
    n = len(rows) if rows is not None else args.num_scenarios
    category = _category_for(args, rows)

    output_path = os.path.join(args.output_dir, f"{two_a}_{two_b}_mts_{args.tag}.jsonl")
    existing = load_existing_conversation_ids(output_path)

    glotlid_model = load_glotlid_model() if args.verify_language else None

    # Phase 1: build every candidate's prompt, then submit ALL transcripts as
    # ONE batch job too (as batchable as the checklists — each transcript is
    # independent of every other conversation's). A conversation whose
    # response is missing/invalid is simply skipped this run (retryable on
    # the next invocation, same convention as a failed pipeline batch item).
    candidates: List[Dict[str, Any]] = []
    for i in range(n):
        row = rows[i] if rows is not None else None
        topic_hint = topic_hints[i % len(topic_hints)]
        conversation_id = f"{lang_a}{lang_b}_mts_{i + 1:04d}"
        if conversation_id in existing:
            print(f"  [{i + 1}/{n}] {conversation_id} skipped (already exists)")
            continue
        prompt = build_scripted_prompt(
            row, args.seed_split, lang_a, lang_b, args.num_turns, cultural_context, topic_hint
        )
        candidates.append({"conversation_id": conversation_id, "row": row, "prompt": prompt})

    if not candidates:
        print("\nNothing new to generate.")
        return output_path if os.path.exists(output_path) else None

    transcript_batch_client = build_batch_client(args.transcript_provider)
    transcript_requests = [
        BatchRequest(
            custom_id=c["conversation_id"],
            prompt=c["prompt"],
            config={"json": True, "thinking_level": args.transcript_thinking},
        )
        for c in candidates
    ]
    print(f"\nSubmitting {len(transcript_requests)} transcript request(s) as ONE batch job...")
    transcript_job_id = transcript_batch_client.submit(transcript_requests, args.transcript_model)
    print(f"Batch job: {transcript_job_id}")

    state = transcript_batch_client.poll(transcript_job_id)
    while state not in TERMINAL:
        detail = transcript_batch_client.progress(transcript_job_id)
        detail_str = f" ({detail})" if detail else ""
        print(f"  job {transcript_job_id} state={state}{detail_str}; waiting {args.poll_interval:.0f}s")
        time.sleep(args.poll_interval)
        state = transcript_batch_client.poll(transcript_job_id)
    if state == FAILED:
        print(f"Transcript batch job {transcript_job_id} FAILED — no scenarios written this run.")
        return output_path if os.path.exists(output_path) else None

    transcript_req_stubs = [BatchRequest(custom_id=r.custom_id, prompt="") for r in transcript_requests]
    transcript_results = transcript_batch_client.collect(transcript_job_id, transcript_req_stubs)

    pending: List[Dict[str, Any]] = []
    for c in candidates:
        conv_id = c["conversation_id"]
        text = transcript_results.get(conv_id)
        if text is None:
            print(f"  {conv_id} transcript FAILED: no batch response")
            continue
        try:
            transcript = parse_and_validate_transcript(text, lang_a, lang_b, args.num_turns, glotlid_model)
        except Exception as e:  # noqa: BLE001
            print(f"  {conv_id} transcript INVALID: {e}")
            continue
        pending.append({"conversation_id": conv_id, "transcript": transcript, "row": c["row"]})
        print(f"  {conv_id} transcript generated ({len(transcript.turns)} turns) via batch")

    if not pending:
        print("\nNo valid transcripts to batch-generate checklists for.")
        return output_path if os.path.exists(output_path) else None

    # Phase 2: ONE batch job for every turn + conversation checklist, across
    # every pending conversation (D7: "independent -> batchable").
    batch_client = build_batch_client(args.checklist_provider)
    requests: List[BatchRequest] = []
    for pc in pending:
        conv_id = pc["conversation_id"]
        transcript = pc["transcript"]
        history_lines: List[str] = []
        for t in sorted(transcript.turns, key=lambda t: t.turn_index):
            target_lang = lang_b if t.speaker == "A" else lang_a
            taxonomy = (
                cg.load_function_taxonomy(target_lang)
                if use_grounding and cg.taxonomy_available(target_lang)
                else None
            )
            requests.append(
                cg.build_turn_checklist_batch_request(
                    f"{conv_id}::turn::{t.turn_index}",
                    target_lang,
                    transcript.conversation_context,
                    t.speaker,
                    t.text,
                    history_text="\n".join(history_lines),
                    taxonomy=taxonomy,
                    thinking_level=args.checklist_thinking,
                    cultural_context=cultural_context,
                )
            )
            history_lines.append(f"Turn {t.turn_index} ({t.speaker}): {t.text}")

        conv_taxonomy = (
            cg.load_function_taxonomy(lang_b) if use_grounding and cg.taxonomy_available(lang_b) else None
        )
        requests.append(
            cg.build_conversation_checklist_batch_request(
                f"{conv_id}::conv",
                lang_b,
                transcript.conversation_context,
                "\n".join(history_lines),
                taxonomy=conv_taxonomy,
                thinking_level=args.checklist_thinking,
                cultural_context=cultural_context,
            )
        )

    print(f"\nSubmitting {len(requests)} checklist request(s) as ONE batch job for {len(pending)} conversation(s)...")
    job_id = batch_client.submit(requests, args.checklist_model)
    print(f"Batch job: {job_id}")

    state = batch_client.poll(job_id)
    while state not in TERMINAL:
        detail = batch_client.progress(job_id)
        detail_str = f" ({detail})" if detail else ""
        print(f"  job {job_id} state={state}{detail_str}; waiting {args.poll_interval:.0f}s")
        time.sleep(args.poll_interval)
        state = batch_client.poll(job_id)
    if state == FAILED:
        print(f"Batch job {job_id} FAILED — no scenarios written this run.")
        return output_path if os.path.exists(output_path) else None

    req_stubs = [BatchRequest(custom_id=r.custom_id, prompt="") for r in requests]
    results = batch_client.collect(job_id, req_stubs)

    # Phase 3: assemble, validate, write. A conversation with any invalid
    # checklist is skipped entirely (retryable on the next invocation — its
    # conversation_id stays absent from `existing`).
    written = 0
    for pc in pending:
        conv_id = pc["conversation_id"]
        transcript = pc["transcript"]
        turns_sorted = sorted(transcript.turns, key=lambda t: t.turn_index)

        turn_records = []
        ok = True
        for t in turns_sorted:
            items = cg.parse_checklist_batch_response(
                results.get(f"{conv_id}::turn::{t.turn_index}"), cg.TURN_HARD_CEILING
            )
            errs = cg.validate_checklist_items(items, cg.TURN_HARD_CEILING)
            if errs:
                print(f"  {conv_id} turn {t.turn_index} checklist INVALID: {'; '.join(errs)}")
                ok = False
                break
            note = cg.checklist_count_note(items, cg.TURN_ITEM_CAP, cg.TURN_HARD_CEILING)
            if note:
                print(f"    {conv_id} turn {t.turn_index}: {note}")
            turn_records.append(
                TurnRecord(
                    turn_index=t.turn_index,
                    speaker=t.speaker,
                    text=t.text,
                    checklist_items=items,
                    verification_prompt=cg.compose_verification_prompt(items),
                )
            )
        if not ok:
            continue

        conv_items = cg.parse_checklist_batch_response(results.get(f"{conv_id}::conv"), cg.CONVERSATION_HARD_CEILING)
        conv_errs = cg.validate_checklist_items(conv_items, cg.CONVERSATION_HARD_CEILING)
        if conv_errs:
            print(f"  {conv_id} conversation checklist INVALID: {'; '.join(conv_errs)}")
            continue
        conv_note = cg.checklist_count_note(conv_items, cg.CONVERSATION_ITEM_CAP, cg.CONVERSATION_HARD_CEILING)
        if conv_note:
            print(f"    {conv_id} conversation: {conv_note}")

        scenario = MultiTurnScenario(
            conversation_id=conv_id,
            mode="scripted",
            lang_a=lang_a,
            lang_b=lang_b,
            Category=category,
            conversation_context=transcript.conversation_context,
            user_a_context=transcript.user_a_context,
            user_b_context=transcript.user_b_context,
            turns=turn_records,
            conversation_checklist_items=conv_items,
            conversation_verification_prompt=cg.compose_verification_prompt(conv_items),
            seed_file=seed_xlsx_name or "hand-authored",
            seed_row_id=(pc["row"].get("seed_row_id") if pc["row"] else None),
            generation_metadata={
                "transcript_model": args.transcript_model,
                "checklist_model": args.checklist_model,
                "function_grounding": use_grounding,
                "checklist_backend": "batch",
                "cultural_context_used": cultural_context is not None,
            },
        )
        append_jsonl(output_path, scenario.model_dump())
        existing.add(conv_id)
        written += 1
        print(f"  {conv_id} generated ({len(scenario.turns)} turns) via batch checklist collection")

    print(f"\n{written}/{len(pending)} conversation(s) written. Output: {output_path}")
    if written:
        with open(output_path, "r", encoding="utf-8") as f:
            written_scenarios = [json.loads(line) for line in f if line.strip()]
        turn_hist = Counter(
            len(t["checklist_items"]) for s in written_scenarios for t in s["turns"]
        )
        conv_hist = Counter(len(s["conversation_checklist_items"]) for s in written_scenarios)
        print(f"Turn-level item-count distribution (all conversations in {output_path}): {dict(sorted(turn_hist.items()))}")
        print(f"Conversation-level item-count distribution: {dict(sorted(conv_hist.items()))}")
    return output_path


def run_dynamic(args) -> Optional[str]:
    two_a, two_b, lang_a, lang_b = _resolve_pair(args.pair)
    cultural_context = _cultural_context_for(two_a, two_b)
    guidance = args.guidance

    use_grounding = not args.no_function_grounding
    if use_grounding and guidance == "guided":
        cg.assert_taxonomies_available((lang_a, lang_b))
    elif not use_grounding:
        print("[warn] --no-function-grounding set: checklists will be ungrounded (function_id=null).")

    rows = _load_rows(args, two_a, two_b)
    seed_xlsx_name = None
    if rows is not None:
        if not rows:
            print("No seed rows available after filters.")
            return None
        seed_xlsx_name = _seed_provenance_name(args, two_a, two_b)

    topic_hints = [h.strip() for h in args.topic_hints.split(",")] if args.topic_hints else [None]
    n = len(rows) if rows is not None else args.num_scenarios
    category = _category_for(args, rows)

    output_path = os.path.join(args.output_dir, f"{two_a}_{two_b}_mtd_{args.tag}.jsonl")
    existing = load_existing_conversation_ids(output_path)

    seed_provider = checklist_provider = None
    if not args.dry_run:
        seed_provider = registry.build_interpreter_provider(
            args.transcript_provider, args.transcript_model, args.transcript_thinking
        )
        if guidance == "guided":
            checklist_provider = registry.build_judge_provider(
                args.checklist_provider, args.checklist_model, args.checklist_thinking
            )

    for i in range(n):
        row = rows[i] if rows is not None else None
        topic_hint = topic_hints[i % len(topic_hints)]
        # guidance folded into the id so guided/free runs sharing one --tag
        # (and thus one output file) never collide on conversation_id.
        conversation_id = f"{lang_a}{lang_b}_mtd_{guidance[0]}{i + 1:04d}"
        prompt = build_dynamic_prompt(
            row, args.seed_split, lang_a, lang_b, args.num_turns, guidance, cultural_context, topic_hint
        )

        if args.dry_run:
            print(f"\n=== [dry-run] dynamic seed prompt ({conversation_id}, guidance={guidance}) ===\n{prompt}")
            continue

        if conversation_id in existing:
            print(f"  [{i + 1}/{n}] {conversation_id} skipped (already exists)")
            continue

        try:
            seed = generate_one_dynamic_seed(
                seed_provider,
                checklist_provider,
                prompt,
                lang_a,
                lang_b,
                args.num_turns,
                guidance,
                use_grounding,
                seed_file=seed_xlsx_name or "hand-authored",
                seed_row_id=row.get("seed_row_id") if row else None,
                category=category,
                conversation_id=conversation_id,
                cultural_context=cultural_context,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{i + 1}/{n}] {conversation_id} FAILED: {e}")
            continue

        append_jsonl(output_path, seed.model_dump())
        existing.add(conversation_id)
        print(f"  [{i + 1}/{n}] {conversation_id} generated")

    if args.dry_run:
        return None
    print(f"\nOutput: {output_path}")
    return output_path


def run_dynamic_batch(args) -> Optional[str]:
    """Dynamic seed generation, batch backend (mirrors ``run_scripted_batch``'s two-phase pattern).

    Phase 1: every candidate's seed (+ intent outline, for guided) is independent of every other
    conversation's, exactly like scripted mode's transcript-gen — batched as ONE job regardless of
    guidance. Phase 2 (guided only): the conversation-level checklist, built from the validated
    intent outline, batched as ONE job across every pending conversation. Free-mode seeds skip
    phase 2 entirely — their checklist is generated later, post-hoc, from the completed live
    transcript (``ops.ensure_conversation_checklist``), not at seed-gen time.
    """
    two_a, two_b, lang_a, lang_b = _resolve_pair(args.pair)
    cultural_context = _cultural_context_for(two_a, two_b)
    guidance = args.guidance

    use_grounding = not args.no_function_grounding
    if use_grounding and guidance == "guided":
        cg.assert_taxonomies_available((lang_a, lang_b))
    elif not use_grounding:
        print("[warn] --no-function-grounding set: checklists will be ungrounded (function_id=null).")

    rows = _load_rows(args, two_a, two_b)
    seed_xlsx_name = None
    if rows is not None:
        if not rows:
            print("No seed rows available after filters.")
            return None
        seed_xlsx_name = _seed_provenance_name(args, two_a, two_b)

    topic_hints = [h.strip() for h in args.topic_hints.split(",")] if args.topic_hints else [None]
    n = len(rows) if rows is not None else args.num_scenarios
    category = _category_for(args, rows)

    output_path = os.path.join(args.output_dir, f"{two_a}_{two_b}_mtd_{args.tag}.jsonl")
    existing = load_existing_conversation_ids(output_path)

    candidates: List[Dict[str, Any]] = []
    for i in range(n):
        row = rows[i] if rows is not None else None
        topic_hint = topic_hints[i % len(topic_hints)]
        conversation_id = f"{lang_a}{lang_b}_mtd_{guidance[0]}{i + 1:04d}"
        if conversation_id in existing:
            print(f"  [{i + 1}/{n}] {conversation_id} skipped (already exists)")
            continue
        prompt = build_dynamic_prompt(
            row, args.seed_split, lang_a, lang_b, args.num_turns, guidance, cultural_context, topic_hint
        )
        candidates.append({"conversation_id": conversation_id, "row": row, "prompt": prompt})

    if not candidates:
        print("\nNothing new to generate.")
        return output_path if os.path.exists(output_path) else None

    # Phase 1: batch every candidate's seed (+ intent outline) as ONE job.
    seed_batch_client = build_batch_client(args.transcript_provider)
    seed_requests = [
        BatchRequest(
            custom_id=c["conversation_id"],
            prompt=c["prompt"],
            config={"json": True, "thinking_level": args.transcript_thinking},
        )
        for c in candidates
    ]
    print(f"\nSubmitting {len(seed_requests)} dynamic-seed request(s) as ONE batch job...")
    seed_job_id = seed_batch_client.submit(seed_requests, args.transcript_model)
    print(f"Batch job: {seed_job_id}")

    state = seed_batch_client.poll(seed_job_id)
    while state not in TERMINAL:
        detail = seed_batch_client.progress(seed_job_id)
        detail_str = f" ({detail})" if detail else ""
        print(f"  job {seed_job_id} state={state}{detail_str}; waiting {args.poll_interval:.0f}s")
        time.sleep(args.poll_interval)
        state = seed_batch_client.poll(seed_job_id)
    if state == FAILED:
        print(f"Seed batch job {seed_job_id} FAILED — no scenarios written this run.")
        return output_path if os.path.exists(output_path) else None

    seed_req_stubs = [BatchRequest(custom_id=r.custom_id, prompt="") for r in seed_requests]
    seed_results = seed_batch_client.collect(seed_job_id, seed_req_stubs)

    pending: List[Dict[str, Any]] = []
    for c in candidates:
        conv_id = c["conversation_id"]
        text = seed_results.get(conv_id)
        if text is None:
            print(f"  {conv_id} seed FAILED: no batch response")
            continue
        try:
            raw = _coerce_intent_outline(json.loads(_strip_json_fences(text)))
            seed = GeneratedSeed.model_validate(raw)
            errors: List[str] = []
            if guidance == "guided":
                if not seed.intent_outline:
                    errors.append("guided seed missing intent_outline")
                else:
                    errors += validate_num_turns(len(seed.intent_outline))
                    errors += validate_alternation(seed.intent_outline)
            if errors:
                raise ValueError("; ".join(errors))
        except Exception as e:  # noqa: BLE001
            print(f"  {conv_id} seed INVALID: {e}")
            continue
        pending.append({"conversation_id": conv_id, "seed": seed, "row": c["row"]})
        print(f"  {conv_id} seed generated via batch")

    if not pending:
        print("\nNo valid seeds to batch-generate checklists for.")
        return output_path if os.path.exists(output_path) else None

    # Phase 2 (guided only): batch every conversation-level checklist as ONE job.
    conv_checklist_results: Dict[str, Any] = {}
    if guidance == "guided":
        checklist_batch_client = build_batch_client(args.checklist_provider)
        conv_taxonomy = (
            cg.load_function_taxonomy(lang_b) if use_grounding and cg.taxonomy_available(lang_b) else None
        )
        requests = []
        for pc in pending:
            seed = pc["seed"]
            ordered = sorted(seed.intent_outline, key=lambda b: b.turn_index)
            basis = "\n".join(f"Turn {b.turn_index} ({b.speaker}) intent: {b.intent}" for b in ordered)
            requests.append(
                cg.build_conversation_checklist_batch_request(
                    pc["conversation_id"],
                    lang_b,
                    seed.conversation_context,
                    basis,
                    taxonomy=conv_taxonomy,
                    thinking_level=args.checklist_thinking,
                    cultural_context=cultural_context,
                )
            )
        print(f"\nSubmitting {len(requests)} conversation-checklist request(s) as ONE batch job...")
        job_id = checklist_batch_client.submit(requests, args.checklist_model)
        print(f"Batch job: {job_id}")

        state = checklist_batch_client.poll(job_id)
        while state not in TERMINAL:
            detail = checklist_batch_client.progress(job_id)
            detail_str = f" ({detail})" if detail else ""
            print(f"  job {job_id} state={state}{detail_str}; waiting {args.poll_interval:.0f}s")
            time.sleep(args.poll_interval)
            state = checklist_batch_client.poll(job_id)
        if state == FAILED:
            print(f"Checklist batch job {job_id} FAILED — no scenarios written this run.")
            return output_path if os.path.exists(output_path) else None

        req_stubs = [BatchRequest(custom_id=r.custom_id, prompt="") for r in requests]
        conv_checklist_results = checklist_batch_client.collect(job_id, req_stubs)

    # Phase 3: assemble, validate, write.
    written = 0
    for pc in pending:
        conv_id = pc["conversation_id"]
        seed = pc["seed"]
        intent_outline_records = None
        conv_items = None
        conv_vprompt = None
        if guidance == "guided":
            ordered = sorted(seed.intent_outline, key=lambda b: b.turn_index)
            intent_outline_records = [
                IntentBeatRecord(turn_index=b.turn_index, speaker=b.speaker, intent=b.intent)
                for b in ordered
            ]
            conv_items = cg.parse_checklist_batch_response(
                conv_checklist_results.get(conv_id), cg.CONVERSATION_HARD_CEILING
            )
            conv_errs = cg.validate_checklist_items(conv_items, cg.CONVERSATION_HARD_CEILING)
            if conv_errs:
                print(f"  {conv_id} conversation checklist INVALID: {'; '.join(conv_errs)}")
                continue
            conv_vprompt = cg.compose_verification_prompt(conv_items)

        dyn_seed = DynamicSeed(
            conversation_id=conv_id,
            mode="dynamic",
            guidance=guidance,
            lang_a=lang_a,
            lang_b=lang_b,
            Category=category,
            conversation_context=seed.conversation_context,
            user_a_context=seed.user_a_context,
            user_b_context=seed.user_b_context,
            num_turns=args.num_turns,
            intent_outline=intent_outline_records,
            conversation_checklist_items=conv_items,
            conversation_verification_prompt=conv_vprompt,
            seed_file=seed_xlsx_name or "hand-authored",
            seed_row_id=(pc["row"].get("seed_row_id") if pc["row"] else None),
            generation_metadata={
                "seed_model": args.transcript_model,
                "checklist_model": args.checklist_model if guidance == "guided" else None,
                "function_grounding": use_grounding,
                "cultural_context_used": cultural_context is not None if guidance == "guided" else None,
                "seed_backend": "batch",
            },
        )
        append_jsonl(output_path, dyn_seed.model_dump())
        existing.add(conv_id)
        written += 1
        print(f"  {conv_id} generated via batch")

    print(f"\n{written}/{len(pending)} conversation(s) written. Output: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate multi-turn scripted transcripts or dynamic seeds (docs/multiturn_expansion_plan.md)."
    )
    p.add_argument("--mode", choices=["scripted", "dynamic"], required=True)
    p.add_argument("--guidance", choices=["guided", "free"], default=None, help="Required for --mode dynamic")
    p.add_argument(
        "--pair",
        required=True,
        help="2-letter pair, order = A-then-B, e.g. id-ko, ar-ko, ar-id, bn-id",
    )
    p.add_argument("--num-scenarios", type=int, default=5)
    p.add_argument("--num-turns", type=int, default=DEFAULT_TURNS)
    p.add_argument(
        "--seed-source",
        choices=["maps", "opensubs"],
        default="maps",
        help="'maps' = MAPS_Final proverb spreadsheets (default); 'opensubs' = real quality-scored "
        "OpenSubtitles bilingual windows from outputs/opensubs_pipeline/top500/<pair>/top500.jsonl "
        "(needed for pairs with no MAPS_Final seed folder, e.g. any pair involving Arabic).",
    )
    p.add_argument(
        "--seed-xlsx",
        default=None,
        help="Defaults to data/MAPS_Final/{2-letter of lang_a}/test_proverbs.xlsx. Ignored when "
        "--seed-source opensubs.",
    )
    p.add_argument("--seed-split", default="test_proverbs")
    p.add_argument("--start-row", type=int, default=1)
    p.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip proverb seed data; invent premise directly (needed for languages with no "
        "MAPS_Final seed folder, e.g. Arabic).",
    )
    p.add_argument(
        "--topic-hints",
        default=None,
        help="Comma-separated topic hints, cycled through when --no-seed is set.",
    )
    p.add_argument("--output-dir", default="data/enriched/multiturn")
    p.add_argument("--tag", default="gen")
    p.add_argument(
        "--no-function-grounding",
        action="store_true",
        help="Escape hatch for languages without a function taxonomy: checklists are still "
        "generated but with function_id=null.",
    )
    p.add_argument("--transcript-provider", default=registry.DEFAULT_JUDGE_PROVIDER)
    p.add_argument("--transcript-model", default=registry.DEFAULT_JUDGE_MODEL)
    p.add_argument("--transcript-thinking", default=registry.DEFAULT_JUDGE_THINKING_LEVEL)
    p.add_argument("--checklist-provider", default=registry.DEFAULT_JUDGE_PROVIDER)
    p.add_argument("--checklist-model", default=registry.DEFAULT_JUDGE_MODEL)
    p.add_argument("--checklist-thinking", default=registry.DEFAULT_JUDGE_THINKING_LEVEL)
    p.add_argument(
        "--checklist-backend",
        default="sync",
        choices=["sync", "batch"],
        help="'batch' submits every turn/conversation checklist (scripted) or every seed + "
        "conversation checklist (dynamic) across all pending conversations as ONE batch job each "
        "(plan Step 8), instead of one call each.",
    )
    p.add_argument("--poll-interval", type=float, default=30.0, help="--checklist-backend batch only")
    p.add_argument(
        "--dry-run", action="store_true", help="Print generation prompts only; no LLM calls, no spend."
    )
    p.add_argument("--verify-language", dest="verify_language", action="store_true", default=True)
    p.add_argument("--no-verify-language", dest="verify_language", action="store_false")
    return p.parse_args()


def main() -> None:
    from interpreter_agent_eval.providers.google_ai import get_usage_totals, reset_usage_totals

    args = parse_args()
    if args.mode == "dynamic" and not args.guidance:
        raise SystemExit("--guidance is required when --mode dynamic")
    if not args.dry_run and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not found in environment.")

    reset_usage_totals()

    if args.mode == "scripted":
        if args.checklist_backend == "batch" and not args.dry_run:
            run_scripted_batch(args)
        else:
            run_scripted(args)
        scope = "mts"
    else:
        if args.checklist_backend == "batch" and not args.dry_run:
            run_dynamic_batch(args)
        else:
            run_dynamic(args)
        scope = "mtd"

    totals = get_usage_totals()
    if totals:
        import json

        two_a, two_b = args.pair.split("-")
        output_path = os.path.join(args.output_dir, f"{two_a}_{two_b}_{scope}_{args.tag}.jsonl")
        sidecar_path = f"{output_path}.usage_totals.{os.getpid()}.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump({f"{k[0]}|{k[1]}": v for k, v in totals.items()}, f, indent=2)
        print(f"[usage] token totals written to {sidecar_path}")


if __name__ == "__main__":
    main()
