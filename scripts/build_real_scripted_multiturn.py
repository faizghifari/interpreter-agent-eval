"""Build scripted multi-turn scenarios by EXPANDING real single-turn records,
instead of inventing a transcript from scratch (contrast with
``generate_multiturn_scenarios.py --mode scripted``, whose transcript is
100% LLM-authored even under ``--seed-source opensubs`` — the real window is
explicitly "inspiration only, do not reuse verbatim" there).

Root cause this addresses: single-turn's ``source_text`` is real OpenSubtitles
dialogue; multi-turn's whole transcript used to be synthetic. That gap, not
checklist methodology, is why multi-turn scores ran higher (see TODO.md).

Turn construction: turn 0 reuses an already-published single-turn record's
real anchor line (verbatim, in lang_a). Turns 1..N-1 come from that anchor's
real ``after_context`` (the following subtitle lines from the same film),
alternating speaker A/B and picking each line's ``source_text`` (lang_a) for
A-turns or ``target_text`` (lang_b) for B-turns — consecutive real lines
standing in for alternating conversational turns; who actually spoke which
line in the film is not tracked by the corpus, so this is a simplification,
not a literal transcript. No transcript-generation LLM call is made at all;
only per-turn + conversation checklist generation (the same finalized setup
used by ``generate_multiturn_scenarios.py``) costs anything.

Output schema matches ``generate_multiturn_scenarios.py``'s ``MultiTurnScenario``
(mode="scripted") so it drops into the existing multiturn pipeline unchanged.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)
sys.path.append(CURRENT_DIR)

from interpreter_agent_eval.pipeline import registry  # noqa: E402
from interpreter_agent_eval.pipeline.multiturn import checklist_gen as cg  # noqa: E402
from interpreter_agent_eval.utils.language_verification import (  # noqa: E402
    load_glotlid_model,
    verify_language_with_glotlid,
)

from generate_multiturn_scenarios import (  # noqa: E402 — reuse schemas/helpers
    LANG_ISO3,
    MultiTurnScenario,
    TurnRecord,
    _cultural_context_for,
    _resolve_pair,
    append_jsonl,
    load_existing_conversation_ids,
)

load_dotenv()

CATEGORY = "OpenSubs-RealContinuation-MT"


def _load_single_turn_index(two_a: str, lang_a: str, lang_b: str) -> Dict[str, Dict[str, Any]]:
    """{source_text: record} for the published lang_a -> lang_b single-turn file."""
    path = os.path.join(ROOT_DIR, "data", "enriched", f"{lang_a}_{lang_b}_full500.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No published single-turn file at {path}")
    index: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            index.setdefault(rec["source_text"], rec)
    return index


def _load_windows(pair_key: str, min_after: int) -> List[Dict[str, Any]]:
    path = os.path.join(ROOT_DIR, "outputs", "opensubs_pipeline", "top500", pair_key, "top500_n6.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No extended-window file at {path}. Regenerate via "
            "`scripts/opensubs_pipeline.py windows --n-after <N>` and filter to the top500 "
            "segment_ids first (see build_real_scripted_multiturn.py's design notes)."
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if len(row.get("after_context") or []) >= min_after:
                rows.append(row)
    return rows


def build_real_turns(
    anchor_source_text: str,
    after_context: List[Dict[str, Any]],
    num_turns: int,
) -> List[Dict[str, str]]:
    """Turn 0 = anchor (speaker A, lang_a). Turns 1..num_turns-1 alternate,
    each pulled from one real after_context line: A-turns use that line's
    ``source_text`` (lang_a), B-turns use its ``target_text`` (lang_b)."""
    turns = [{"turn_index": 0, "speaker": "A", "text": anchor_source_text}]
    for i in range(1, num_turns):
        line = after_context[i - 1]
        speaker = "A" if i % 2 == 0 else "B"
        text = line["source_text"] if speaker == "A" else line["target_text"]
        turns.append({"turn_index": i, "speaker": speaker, "text": text})
    return turns


def glotlid_sanity_check(turns: List[Dict[str, str]], lang_a: str, lang_b: str, glotlid_model: Any) -> List[str]:
    if glotlid_model is None:
        return []
    errors = []
    for t in turns:
        expected = lang_a if t["speaker"] == "A" else lang_b
        v = verify_language_with_glotlid(
            model=glotlid_model,
            text=t["text"],
            expected_iso_code=expected,
            min_confidence=0.5,
            context_name=f"turn {t['turn_index']}",
        )
        if not v.is_correct and not getattr(v, "needs_review", False):
            errors.append(f"turn {t['turn_index']} ({t['speaker']}) failed GlotLID sanity check: {v.message}")
    return errors


def build_one_scenario(
    checklist_provider: Any,
    row: Dict[str, Any],
    single_turn_rec: Dict[str, Any],
    lang_a: str,
    lang_b: str,
    num_turns: int,
    use_grounding: bool,
    cultural_context: Optional[str],
    conversation_id: str,
    glotlid_model: Any,
    consistency_runs: int,
    filter_by_annotation: bool,
    meaningful_threshold: float,
) -> MultiTurnScenario:
    turns_raw = build_real_turns(row["source_text"], row["after_context"], num_turns)
    lid_errors = glotlid_sanity_check(turns_raw, lang_a, lang_b, glotlid_model)
    if lid_errors:
        raise ValueError("; ".join(lid_errors))

    conversation_context = single_turn_rec.get("conversation_context", "")
    turn_records: List[TurnRecord] = []
    history_lines: List[str] = []
    for t in turns_raw:
        target_lang = lang_b if t["speaker"] == "A" else lang_a
        taxonomy = (
            cg.load_function_taxonomy(target_lang) if use_grounding and cg.taxonomy_available(target_lang) else None
        )
        items = cg.generate_turn_checklist(
            checklist_provider,
            target_lang,
            conversation_context,
            t["speaker"],
            t["text"],
            history_text="\n".join(history_lines),
            taxonomy=taxonomy,
            cultural_context=cultural_context,
            consistency_runs=consistency_runs,
            filter_by_annotation=filter_by_annotation,
            meaningful_threshold=meaningful_threshold,
        )
        errs = cg.validate_checklist_items(items, cg.TURN_HARD_CEILING)
        if errs:
            raise ValueError(f"turn {t['turn_index']} checklist: " + "; ".join(errs))
        turn_records.append(
            TurnRecord(
                turn_index=t["turn_index"],
                speaker=t["speaker"],
                text=t["text"],
                checklist_items=items,
                verification_prompt=cg.compose_verification_prompt(items),
            )
        )
        history_lines.append(f"Turn {t['turn_index']} ({t['speaker']}): {t['text']}")

    conv_taxonomy = cg.load_function_taxonomy(lang_b) if use_grounding and cg.taxonomy_available(lang_b) else None
    conv_items = cg.generate_conversation_checklist(
        checklist_provider,
        lang_b,
        conversation_context,
        "\n".join(history_lines),
        taxonomy=conv_taxonomy,
        cultural_context=cultural_context,
        consistency_runs=consistency_runs,
        filter_by_annotation=filter_by_annotation,
        meaningful_threshold=meaningful_threshold,
    )
    errs = cg.validate_checklist_items(conv_items, cg.CONVERSATION_HARD_CEILING)
    if errs:
        raise ValueError("conversation checklist: " + "; ".join(errs))

    return MultiTurnScenario(
        conversation_id=conversation_id,
        mode="scripted",
        lang_a=lang_a,
        lang_b=lang_b,
        Category=CATEGORY,
        conversation_context=conversation_context,
        user_a_context=single_turn_rec.get("user_a_context", ""),
        user_b_context=single_turn_rec.get("user_b_context", ""),
        turns=turn_records,
        conversation_checklist_items=conv_items,
        conversation_verification_prompt=cg.compose_verification_prompt(conv_items),
        seed_file="top500_n6.jsonl (real after_context expansion)",
        seed_row_id=row.get("segment_id"),
        generation_metadata={
            "source_single_turn_record_id": single_turn_rec.get("record_id"),
            "checklist_model": getattr(checklist_provider, "model_name", None),
            "function_grounding": use_grounding,
            "cultural_context_used": cultural_context is not None,
            "consistency_runs": consistency_runs,
            "filter_by_annotation": filter_by_annotation,
            "real_data_expansion": True,
            "transcript_provenance": "real_opensubtitles_after_context (verbatim, not LLM-authored)",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", required=True, help="2-letter pair, order = A-then-B, e.g. id-ko")
    p.add_argument("--num-scenarios", type=int, default=10)
    p.add_argument("--num-turns", type=int, default=6)
    p.add_argument("--output-dir", default="data/enriched/multiturn")
    p.add_argument("--tag", default="real")
    p.add_argument("--no-function-grounding", action="store_true")
    p.add_argument("--checklist-provider", default=registry.DEFAULT_JUDGE_PROVIDER)
    p.add_argument("--checklist-model", default=registry.DEFAULT_JUDGE_MODEL)
    p.add_argument("--checklist-thinking", default=registry.DEFAULT_JUDGE_THINKING_LEVEL)
    p.add_argument("--consistency-runs", type=int, default=1)
    p.add_argument("--filter-by-annotation", action="store_true")
    p.add_argument("--meaningful-threshold", type=float, default=cg.DEFAULT_MEANINGFUL_THRESHOLD)
    p.add_argument("--dry-run", action="store_true", help="Print planned turns only; no LLM calls.")
    p.add_argument("--no-verify-language", dest="verify_language", action="store_false", default=True)
    args = p.parse_args()

    two_a, two_b, lang_a, lang_b = _resolve_pair(args.pair)
    pair_key = "-".join(sorted([two_a, two_b]))
    cultural_context = _cultural_context_for(two_a, two_b)

    use_grounding = not args.no_function_grounding
    if use_grounding:
        cg.assert_taxonomies_available((lang_a, lang_b))

    single_turn_index = _load_single_turn_index(two_a, lang_a, lang_b)
    windows = _load_windows(pair_key, min_after=args.num_turns - 1)

    matched = []
    for row in windows:
        rec = single_turn_index.get(row["source_text"])
        if rec is not None:
            matched.append((row, rec))
    print(f"{len(matched)}/{len(windows)} windows matched a published single-turn record "
          f"(of {len(single_turn_index)} unique {lang_a}->{lang_b} source texts)")
    matched = matched[: args.num_scenarios]

    if args.dry_run:
        for row, rec in matched:
            turns = build_real_turns(row["source_text"], row["after_context"], args.num_turns)
            print(f"\n=== record_id={rec.get('record_id')} segment_id={row.get('segment_id')} ===")
            for t in turns:
                print(f"  [{t['turn_index']}] {t['speaker']}: {t['text']}")
        return

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not found in environment.")

    checklist_provider = registry.build_judge_provider(
        args.checklist_provider, args.checklist_model, args.checklist_thinking
    )
    glotlid_model = load_glotlid_model() if args.verify_language else None

    output_path = os.path.join(args.output_dir, f"{two_a}_{two_b}_mts_{args.tag}.jsonl")
    existing = load_existing_conversation_ids(output_path)

    n_written = 0
    for i, (row, rec) in enumerate(matched, 1):
        conversation_id = f"{lang_a}{lang_b}_mts_{args.tag}_{i:04d}"
        if conversation_id in existing:
            print(f"  [{i}/{len(matched)}] {conversation_id} skipped (already exists)")
            continue
        try:
            scenario = build_one_scenario(
                checklist_provider,
                row,
                rec,
                lang_a,
                lang_b,
                args.num_turns,
                use_grounding,
                cultural_context,
                conversation_id,
                glotlid_model,
                args.consistency_runs,
                args.filter_by_annotation,
                args.meaningful_threshold,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(matched)}] {conversation_id} FAILED: {e}")
            continue
        append_jsonl(output_path, scenario.model_dump())
        existing.add(conversation_id)
        n_written += 1
        print(f"  [{i}/{len(matched)}] {conversation_id} written ({len(scenario.turns)} turns)")

    print(f"\n{n_written}/{len(matched)} scenario(s) written. Output: {output_path}")


if __name__ == "__main__":
    main()
