"""Single-pass, function-grounded checklist generation (plan D4).

Replaces the single-turn grid's 3-lists-then-dedup approach: every multi-turn
checklist — scripted per-turn, scripted/guided conversation-level, dynamic
on-the-fly per-turn, and free-flow's post-hoc conversation-level — is produced
by ONE LLM call returning one compact set, implemented once here and used by
both the scenario generator (``scripts/generate_multiturn_scenarios.py``,
offline) and mt-converse (dynamic, on the fly) so mode comparisons aren't
confounded by checklist provenance.

Grounding: the prompt embeds the target language's evaluation-function
taxonomy (``outputs/evalet/by_target/{lang}_functions.json``, ~53 clusters).
The generator selects only the functions genuinely applicable to the text at
hand and writes one specific yes/no item per selected function. Never raises
on cap/count violations — callers validate and retry/repair as needed
(mirrors the pipeline's "never raise, capture in a field" convention via
plain return values here since this module has no per-record ``error`` dict).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from interpreter_agent_eval.prompts.templates import (
    MULTITURN_CHECKLIST_GEN_PROMPT,
    MULTITURN_CHECKLIST_GEN_PROMPT_DEEP,
)
from interpreter_agent_eval.utils.semantic_dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    semantic_dedup_indices,
)
from interpreter_agent_eval.utils.checklist_validity import (
    DEFAULT_MEANINGFUL_THRESHOLD,
    filter_items_by_annotation,
)

from ..batch import BatchRequest

# src/interpreter_agent_eval/pipeline/multiturn/checklist_gen.py -> repo root
# is 4 levels up (checklist_gen.py -> multiturn -> pipeline -> interpreter_agent_eval -> src -> root).
_REPO_ROOT = Path(__file__).resolve().parents[4]
TAXONOMY_DIR = _REPO_ROOT / "outputs" / "evalet" / "by_target"

LANG_NAMES = {"arb": "Arabic", "ben": "Bengali", "ind": "Indonesian", "kor": "Korean"}

# (advisory floor, informational "typical" ceiling) — used only by
# checklist_count_note() for run-log monitoring. NOT sent to the model (the
# prompt states no upper limit) and NOT enforced; see *_HARD_CEILING below for
# the real, code-only cap.
TURN_ITEM_CAP: Tuple[int, int] = (3, 7)
CONVERSATION_ITEM_CAP: Tuple[int, int] = (3, 6)

# Runaway-response guards only (2x the old typical ceiling) — never mentioned
# in the prompt. Honest output has never come close to these (verified: 0 of
# 1,750 real responses exceeded the old stated ceiling of 7/6 even when it WAS
# in the prompt); they exist purely to bound judge cost against a malformed
# response, not to shape genuine checklist length.
TURN_HARD_CEILING = 14
CONVERSATION_HARD_CEILING = 12

_VALID_LAYERS = {"layer_1", "layer_2", "layer_3", "layer_unknown"}
_LAYER_PRIORITY = {"layer_3": 0, "layer_2": 1, "layer_1": 2}

# Finalized checklist-gen setup (validated in the floor-self-consistency
# investigation): retry generation until the layer_1/2/3 >=1 floor + priority
# rule is met (up to this many attempts, then accept the last draw as-is),
# optionally pool multiple independent generations (consistency_runs > 1) for
# self-consistency, and ALWAYS semantic-dedup the result -- single run or
# pooled -- to strip within- or across-run near-duplicates.
MAX_FLOOR_RETRIES = 3


class ChecklistItem(BaseModel):
    """One grounded, concrete yes/no checklist item (plan D4)."""

    function_id: Optional[str] = Field(
        default=None,
        description="Taxonomy cluster id (e.g. 'L1_f6'); None when ungrounded.",
    )
    layer: str = Field(description="layer_1 | layer_2 | layer_3 | layer_unknown")
    text: str = Field(description="Concrete yes/no item; Yes means the interpreter succeeded.")


class ChecklistGenResponse(BaseModel):
    """Structured-output schema for one checklist-generation LLM call."""

    items: List[ChecklistItem]
    # D1 screening pilot (docs/multiturn_gap_diagnosis.md): populated only by
    # MULTITURN_CHECKLIST_GEN_PROMPT_DEEP's forced STEP 1 pragmatic-analysis
    # stage. None for the unmodified MULTITURN_CHECKLIST_GEN_PROMPT response,
    # which has no such field.
    pragmatic_analysis: Optional[str] = None


# ---------------------------------------------------------------------------
# Taxonomy loading (read-only data dependency; outputs/ is manually synced)
# ---------------------------------------------------------------------------
_taxonomy_cache: Dict[str, List[Dict[str, str]]] = {}


def taxonomy_path(target_lang: str) -> Path:
    return TAXONOMY_DIR / f"{target_lang}_functions.json"


def taxonomy_available(target_lang: str) -> bool:
    return taxonomy_path(target_lang).exists()


def load_function_taxonomy(target_lang: str) -> List[Dict[str, str]]:
    """Load + cache the compact ``function_id | layer | label`` taxonomy.

    Raises ``FileNotFoundError`` if the taxonomy file isn't present. Callers
    that want the ``--no-function-grounding`` escape hatch should check
    ``taxonomy_available()`` first and pass ``taxonomy=None`` instead of
    calling this directly.
    """
    if target_lang in _taxonomy_cache:
        return _taxonomy_cache[target_lang]
    path = taxonomy_path(target_lang)
    if not path.exists():
        raise FileNotFoundError(
            f"No function taxonomy for '{target_lang}' at {path}. "
            "outputs/ is manually synced onto this machine — verify it's present, "
            "or pass --no-function-grounding to bypass grounding for this language."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    compact = [
        {"function_id": e["function_id"], "layer": e["layer"], "label": e["label"]}
        for e in raw
    ]
    _taxonomy_cache[target_lang] = compact
    return compact


def assert_taxonomies_available(langs: Tuple[str, ...] = ("arb", "ben", "ind", "kor")) -> None:
    """Fail loudly and early if any required taxonomy file is missing."""
    missing = [lang for lang in langs if not taxonomy_available(lang)]
    if missing:
        raise FileNotFoundError(
            f"Missing function taxonomy file(s) for: {missing}. Expected under {TAXONOMY_DIR}. "
            "outputs/ is manually synced onto this machine — verify it's present, or pass "
            "--no-function-grounding to bypass grounding for those languages."
        )


def format_taxonomy_listing(entries: List[Dict[str, str]]) -> str:
    return "\n".join(f"{e['function_id']} | {e['layer']} | {e['label']}" for e in entries)


# ---------------------------------------------------------------------------
# Cultural-context grounding: read-only data dependency, same pattern as the
# function taxonomy above. The canonical copy of these pair-specific
# cultural-asymmetry paragraphs lives in the frozen single-turn grid generator
# (``scripts/augment_opensubs_maps.py``) — imported lazily and never modified
# from here.
# ---------------------------------------------------------------------------
_ISO3_TO_2 = {"arb": "ar", "ben": "bn", "ind": "id", "kor": "ko"}
_cultural_context_table: Optional[Dict[str, str]] = None


def _load_cultural_context_table() -> Dict[str, str]:
    global _cultural_context_table
    if _cultural_context_table is None:
        scripts_dir = str(_REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from augment_opensubs_maps import CULTURAL_CONTEXT  # type: ignore  # noqa: E402

        _cultural_context_table = CULTURAL_CONTEXT
    return _cultural_context_table


def get_cultural_context(lang_a: str, lang_b: str) -> Optional[str]:
    """Pair-specific cultural-asymmetry paragraph for two ISO-639-3 codes.

    Returns ``None`` if either code has no 2-letter mapping or the pair has no
    authored entry — callers pass that through as "no cultural block" rather
    than treating it as an error (a future 5th language, e.g., legitimately
    has none yet).
    """
    two_a, two_b = _ISO3_TO_2.get(lang_a), _ISO3_TO_2.get(lang_b)
    if not two_a or not two_b:
        return None
    return _load_cultural_context_table().get("-".join(sorted([two_a, two_b])))


def _cultural_context_block(cultural_context: Optional[str]) -> str:
    return f"{cultural_context}\n\n" if cultural_context else ""


# ---------------------------------------------------------------------------
# Prompt building + LLM call
# ---------------------------------------------------------------------------
def _difficulty_tags_block(difficulty_tags: Optional[str]) -> str:
    return difficulty_tags if difficulty_tags else "(none detected)"


def _build_prompt(
    target_lang: str,
    scope: str,
    conversation_context: str,
    scope_content: str,
    taxonomy: Optional[List[Dict[str, str]]],
    cultural_context: Optional[str] = None,
    prompt_variant: str = "current",
    difficulty_tags: Optional[str] = None,
) -> str:
    """``prompt_variant``: "current" (unmodified, shipped-grid prompt) or "deep"
    (D1 screening pilot's forced-pragmatic-analysis prompt,
    ``MULTITURN_CHECKLIST_GEN_PROMPT_DEEP`` — see docs/multiturn_gap_diagnosis.md
    §4b). Defaults to today's behaviour so no existing caller changes."""
    target_language = LANG_NAMES.get(target_lang, target_lang)
    if taxonomy:
        taxonomy_listing = format_taxonomy_listing(taxonomy)
        grounding_note = ""
    else:
        taxonomy_listing = (
            "(no taxonomy available for this language — select functions freely "
            "and set function_id to null for every item)"
        )
        grounding_note = (
            " Since no taxonomy is available, set function_id to null on every item, "
            "but still write specific, concrete yes/no items grounded in the text above."
        )
    scope_noun = "utterance" if scope == "turn" else "conversation"
    if prompt_variant == "deep":
        return MULTITURN_CHECKLIST_GEN_PROMPT_DEEP.format(
            target_language=target_language,
            target_lang_code=target_lang,
            taxonomy_listing=taxonomy_listing,
            scope_noun=scope_noun,
            conversation_context=conversation_context,
            cultural_context_block=_cultural_context_block(cultural_context),
            scope_content=scope_content,
            grounding_note=grounding_note,
            difficulty_tags_block=_difficulty_tags_block(difficulty_tags),
        )
    if prompt_variant != "current":
        raise ValueError(f"Unknown prompt_variant '{prompt_variant}'; use 'current' or 'deep'")
    return MULTITURN_CHECKLIST_GEN_PROMPT.format(
        target_language=target_language,
        target_lang_code=target_lang,
        taxonomy_listing=taxonomy_listing,
        scope_noun=scope_noun,
        conversation_context=conversation_context,
        cultural_context_block=_cultural_context_block(cultural_context),
        scope_content=scope_content,
        grounding_note=grounding_note,
    )


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _call_checklist_provider(provider: Any, prompt: str) -> List[ChecklistItem]:
    """Call the checklist-gen provider with structured output, plain fallback.

    Mirrors ``pipeline.operations.parse_judge_evaluation``'s structured-then-
    plain-generation fallback (mirror, don't modify the original).
    """
    try:
        text = provider.generate(
            prompt,
            response_mime_type="application/json",
            response_schema=ChecklistGenResponse,
        )
        return ChecklistGenResponse.model_validate_json(_clean_json_text(text)).items
    except Exception:  # noqa: BLE001 — try plain generation before giving up
        text = provider.generate(prompt)
        return ChecklistGenResponse.model_validate_json(_clean_json_text(text)).items


def _enforce_cap(items: List[ChecklistItem], hard_ceiling: int) -> List[ChecklistItem]:
    """No longer truncates (instrumentation only, pending pilot data).

    Previously dropped lowest-priority-layer items past ``hard_ceiling``. The
    real cap policy (if any) is being decided from real pilot data instead of
    truncating blindly, so this now just logs when a response would have
    exceeded the ceiling and returns ``items`` unchanged. See
    ``checklist_count_note()`` for the analysis-facing signal.
    """
    if len(items) > hard_ceiling:
        print(
            f"[checklist-gen] NOTE: {len(items)} item(s) exceeds hard ceiling "
            f"{hard_ceiling} — not truncated (instrumentation only)"
        )
    return items


def _generate_once_with_floor_retry(
    provider: Any,
    target_lang: str,
    scope: str,
    conversation_context: str,
    scope_content: str,
    taxonomy: Optional[List[Dict[str, str]]],
    cultural_context: Optional[str],
    hard_ceiling: int,
) -> List[ChecklistItem]:
    """One generation, retried up to MAX_FLOOR_RETRIES times if the layer
    floor / priority rule isn't met. Accepts the last draw even if still
    unmet (never blocks forever on a genuinely thin utterance)."""
    prompt = _build_prompt(target_lang, scope, conversation_context, scope_content, taxonomy, cultural_context)
    items: List[ChecklistItem] = []
    for _ in range(MAX_FLOOR_RETRIES):
        items = _call_checklist_provider(provider, prompt)
        if not validate_checklist_items(items, hard_ceiling):
            return items
    return items


def _dedup_items(items: List[ChecklistItem], threshold: float) -> List[ChecklistItem]:
    """Always applied (single run or pooled) -- strips near-duplicate
    criteria, greedy first-occurrence priority."""
    text_items = [it for it in items if it.text and it.text.strip()]
    if not text_items:
        return []
    texts = [it.text for it in text_items]
    kept_idx = semantic_dedup_indices(texts, threshold)
    return [text_items[i] for i in kept_idx]


def generate_turn_checklist(
    provider: Any,
    target_lang: str,
    conversation_context: str,
    speaker: str,
    source_text: str,
    history_text: str = "",
    taxonomy: Optional[List[Dict[str, str]]] = None,
    cultural_context: Optional[str] = None,
    consistency_runs: int = 1,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    filter_by_annotation: bool = False,
    meaningful_threshold: float = DEFAULT_MEANINGFUL_THRESHOLD,
) -> List[ChecklistItem]:
    """Checklist for a single turn's utterance (scripted or dynamic).

    Finalized setup: generate ``consistency_runs`` independent draws (1 =
    single-pass, 3 = self-consistency pooling), each retried internally until
    the layer floor is met, pool them, and ALWAYS semantic-dedup the result
    (even for a single run, to strip within-run near-duplicates). Optionally
    (``filter_by_annotation=True``) drop criteria whose taxonomy function
    fails the Task A human-validity check (Situation-specific / below
    ``meaningful_threshold``) -- off by default since it needs the
    ``outputs/annotation_sheets/results/`` data present.

    ``history_text`` is a caller-rendered block of prior turns (speaker,
    source text, translation) — this module stays agnostic of turn-dict
    shape, matching how ``pipeline/operations.py`` builders take pre-rendered
    prompt pieces rather than raw record structures. ``cultural_context`` is
    the caller-precomputed pair-specific paragraph (see
    ``get_cultural_context``) — passed in rather than looked up here, mirroring
    how ``taxonomy`` is already caller-loaded and passed in.
    """
    if taxonomy is None and taxonomy_available(target_lang):
        taxonomy = load_function_taxonomy(target_lang)
    scope_content = _turn_scope_content(speaker, source_text, history_text)
    pooled: List[ChecklistItem] = []
    for _ in range(max(1, consistency_runs)):
        pooled.extend(
            _generate_once_with_floor_retry(
                provider, target_lang, "turn", conversation_context, scope_content,
                taxonomy, cultural_context, TURN_HARD_CEILING,
            )
        )
    items = _dedup_items(pooled, dedup_threshold)
    if filter_by_annotation:
        items = filter_items_by_annotation(items, target_lang, meaningful_threshold)
    return _enforce_cap(items, TURN_HARD_CEILING)


def generate_conversation_checklist(
    provider: Any,
    target_lang: str,
    conversation_context: str,
    basis_text: str,
    taxonomy: Optional[List[Dict[str, str]]] = None,
    cultural_context: Optional[str] = None,
    consistency_runs: int = 1,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    filter_by_annotation: bool = False,
    meaningful_threshold: float = DEFAULT_MEANINGFUL_THRESHOLD,
) -> List[ChecklistItem]:
    """Conversation-level checklist, cross-turn functions only.

    Same finalized setup as ``generate_turn_checklist`` — see its docstring
    for the consistency_runs / dedup / filter_by_annotation semantics.

    ``basis_text`` is the caller-rendered basis: the full authored transcript
    (scripted), the intent outline + goals (guided dynamic, at seed time), or
    the completed transcript (free-flow, generated post-hoc inside
    mt-judge-conv).
    """
    if taxonomy is None and taxonomy_available(target_lang):
        taxonomy = load_function_taxonomy(target_lang)
    scope_content = _conversation_scope_content(basis_text)
    pooled: List[ChecklistItem] = []
    for _ in range(max(1, consistency_runs)):
        pooled.extend(
            _generate_once_with_floor_retry(
                provider, target_lang, "conversation", conversation_context, scope_content,
                taxonomy, cultural_context, CONVERSATION_HARD_CEILING,
            )
        )
    items = _dedup_items(pooled, dedup_threshold)
    if filter_by_annotation:
        items = filter_items_by_annotation(items, target_lang, meaningful_threshold)
    return _enforce_cap(items, CONVERSATION_HARD_CEILING)


def _turn_scope_content(speaker: str, source_text: str, history_text: str = "") -> str:
    return (
        (f"Prior turns:\n{history_text}\n\n" if history_text else "")
        + f"Current utterance (speaker {speaker}): {source_text}"
    )


def _conversation_scope_content(basis_text: str) -> str:
    return f"Basis for cross-turn evaluation:\n{basis_text}"


# ---------------------------------------------------------------------------
# Batch request/response helpers (plan Step 8): "single-job batch for ... the
# generator's scripted checklist pre-generation." Pure — building a
# BatchRequest or parsing one response never calls a provider.
# ---------------------------------------------------------------------------
def build_turn_checklist_batch_request(
    custom_id: str,
    target_lang: str,
    conversation_context: str,
    speaker: str,
    source_text: str,
    history_text: str = "",
    taxonomy: Optional[List[Dict[str, str]]] = None,
    thinking_level: Optional[str] = None,
    cultural_context: Optional[str] = None,
    prompt_variant: str = "current",
    difficulty_tags: Optional[str] = None,
) -> BatchRequest:
    if taxonomy is None and taxonomy_available(target_lang):
        taxonomy = load_function_taxonomy(target_lang)
    scope_content = _turn_scope_content(speaker, source_text, history_text)
    prompt = _build_prompt(
        target_lang, "turn", conversation_context, scope_content, taxonomy, cultural_context,
        prompt_variant=prompt_variant, difficulty_tags=difficulty_tags,
    )
    config: Dict[str, Any] = {"json": True, "response_schema": ChecklistGenResponse}
    if thinking_level:
        config["thinking_level"] = thinking_level
    return BatchRequest(custom_id=custom_id, prompt=prompt, config=config)


def build_conversation_checklist_batch_request(
    custom_id: str,
    target_lang: str,
    conversation_context: str,
    basis_text: str,
    taxonomy: Optional[List[Dict[str, str]]] = None,
    thinking_level: Optional[str] = None,
    cultural_context: Optional[str] = None,
    prompt_variant: str = "current",
    difficulty_tags: Optional[str] = None,
) -> BatchRequest:
    if taxonomy is None and taxonomy_available(target_lang):
        taxonomy = load_function_taxonomy(target_lang)
    scope_content = _conversation_scope_content(basis_text)
    prompt = _build_prompt(
        target_lang, "conversation", conversation_context, scope_content, taxonomy, cultural_context,
        prompt_variant=prompt_variant, difficulty_tags=difficulty_tags,
    )
    config: Dict[str, Any] = {"json": True, "response_schema": ChecklistGenResponse}
    if thinking_level:
        config["thinking_level"] = thinking_level
    return BatchRequest(custom_id=custom_id, prompt=prompt, config=config)


def parse_checklist_batch_response(
    text: Optional[str],
    hard_ceiling: int,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    filter_by_annotation: bool = False,
    target_lang: Optional[str] = None,
    meaningful_threshold: float = DEFAULT_MEANINGFUL_THRESHOLD,
) -> List[ChecklistItem]:
    """Parse one batch-collected response into capped ChecklistItems.

    Returns ``[]`` (never raises) on a missing/malformed response — the
    caller's own validation (``validate_checklist_items``) surfaces that as a
    retryable failure, same as the sync path's structured-parse fallback.
    Note: this intentionally conflates "malformed/missing response" with a
    genuine (invalid) empty ``{"items": []}`` — both produce ``[]`` here, and
    both correctly map to "checklist is empty" -> discard/retry downstream.

    Always semantic-dedups (same as the sync path — batch mode can't retry
    mid-batch on a floor violation, but dedup applies regardless). Passing
    ``filter_by_annotation=True`` also needs ``target_lang`` set.
    """
    if text is None:
        return []
    try:
        items = ChecklistGenResponse.model_validate_json(_clean_json_text(text)).items
    except Exception:  # noqa: BLE001
        return []
    items = _dedup_items(items, dedup_threshold)
    if filter_by_annotation and target_lang:
        items = filter_items_by_annotation(items, target_lang, meaningful_threshold)
    return _enforce_cap(items, hard_ceiling)


# ---------------------------------------------------------------------------
# Composition + validation
# ---------------------------------------------------------------------------
def compose_verification_prompt(items: List[ChecklistItem]) -> str:
    """Numbered checklist string, L3 -> L2 -> L1 priority (stable within layer).

    Same ordering convention as
    ``scripts.generate_maps_scenarios.compose_verification_prompt`` — judge
    prompts don't need to know which pipeline produced the checklist.
    """
    ordered = sorted(items, key=lambda it: _LAYER_PRIORITY.get(it.layer, 3))
    cleaned = [it.text.strip() for it in ordered if it.text and it.text.strip()]
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(cleaned, 1))


def validate_checklist_items(items: List[ChecklistItem], hard_ceiling: int) -> List[str]:
    """Return a list of hard violations (empty list = valid).

    An empty checklist is a hard error: it is useless to the judge and, on
    the batch path, indistinguishable from a malformed/failed response
    (``parse_checklist_batch_response`` returns ``[]`` for both). A count
    below the advisory floor is NOT an error — the prompt asks for as many
    items as genuinely apply (at least 1), so an honest 1-2 item checklist
    for simple content must survive validation. Use ``checklist_count_note()``
    for a non-fatal floor/typical-ceiling/hard-ceiling signal — exceeding
    ``hard_ceiling`` is no longer a hard failure (pending pilot data on
    whether any cap is warranted at all; see ``_enforce_cap``).

    Standardized with the single-turn generator
    (``scripts.augment_opensubs_maps._validate_generated_sample``): each of
    layer_1/layer_2/layer_3 must have >=1 item, and
    layer_3_count >= layer_2_count >= layer_1_count. ``layer_unknown`` items
    are excluded from these counts (ungrounded, e.g. --no-function-grounding).
    """
    errors: List[str] = []
    if not items:
        errors.append("checklist is empty")
    for it in items:
        if not it.text or not it.text.strip():
            errors.append(f"empty checklist text (function_id={it.function_id})")
        if it.layer not in _VALID_LAYERS:
            errors.append(f"unexpected layer '{it.layer}' (function_id={it.function_id})")

    if items:
        counts = {"layer_1": 0, "layer_2": 0, "layer_3": 0}
        for it in items:
            if it.layer in counts:
                counts[it.layer] += 1
        if counts["layer_1"] < 1:
            errors.append("layer_1 must have at least 1 item")
        if counts["layer_2"] < 1:
            errors.append("layer_2 must have at least 1 item")
        if counts["layer_3"] < 1:
            errors.append("layer_3 must have at least 1 item")
        if not (counts["layer_3"] >= counts["layer_2"] >= counts["layer_1"]):
            errors.append("checklist count priority must satisfy layer_3 >= layer_2 >= layer_1")

    return errors


def checklist_count_note(
    items: List[ChecklistItem], cap: Tuple[int, int], hard_ceiling: Optional[int] = None
) -> Optional[str]:
    """Non-fatal note when the count is below the advisory floor, above the
    old typical ceiling (``cap = (floor, typical_max)``), or above
    ``hard_ceiling`` (the runaway-guard threshold, no longer enforced by
    truncation — see ``_enforce_cap``).

    Purely informational (run logs / distribution monitoring) — never blocks
    acceptance and never triggers a retry or discard. The prompt no longer
    states an upper limit, so an above-typical count is expected on genuinely
    dense content; this just makes it visible in run logs rather than letting
    the distribution shift silently.
    """
    min_n, typical_max = cap
    n = len(items)
    if hard_ceiling is not None and n > hard_ceiling:
        return (
            f"checklist has {n} item(s), EXCEEDS hard ceiling {hard_ceiling} "
            "(not truncated) — flag for pilot cost/ceiling analysis"
        )
    if items and n < min_n:
        return f"checklist has {n} item(s), below advisory floor {min_n}"
    if n > typical_max:
        return (
            f"checklist has {n} item(s), above the old typical ceiling {typical_max} "
            "(prompt is uncapped) — verify this reflects genuine content complexity"
        )
    return None
