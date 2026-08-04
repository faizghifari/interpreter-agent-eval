"""Regenerate turn + conversation checklists for an existing scripted
multi-turn scenario file, reusing already-generated transcripts (no re-spend
on transcript-gen). Submits ONE batch job for every turn + conversation
checklist request across all conversations in the file, then overwrites
``checklist_items``/``verification_prompt`` fields in place.

Use this after a checklist-gen prompt/logic fix, to re-run checklist
generation over data whose transcripts are already good.

    uv run python scripts/regenerate_multiturn_checklists.py \
        --input data/enriched/multiturn/id_ko_mts_tier300.jsonl --pair id-ko
"""

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_ROOT, "src"))

from dotenv import load_dotenv  # noqa: E402

from interpreter_agent_eval.pipeline import registry  # noqa: E402
from interpreter_agent_eval.pipeline.batch import (  # noqa: E402
    FAILED,
    TERMINAL,
    BatchRequest,
    build_batch_client,
)
from interpreter_agent_eval.pipeline.multiturn import checklist_gen as cg  # noqa: E402

load_dotenv()

_ISO3_TO_2 = {"ind": "id", "kor": "ko", "arb": "ar", "ben": "bn"}
_2_TO_ISO3 = {v: k for k, v in _ISO3_TO_2.items()}


def regenerate(
    input_path: str,
    lang_a: str,
    lang_b: str,
    checklist_provider_type: str,
    checklist_model: str,
    checklist_thinking: str,
    use_grounding: bool,
    poll_interval: float,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        scenarios: List[Dict[str, Any]] = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(scenarios)} scenario(s) from {input_path}")

    cultural_context = cg.get_cultural_context(lang_a, lang_b)
    requests: List[BatchRequest] = []
    for s in scenarios:
        conv_id = s["conversation_id"]
        turns_sorted = sorted(s["turns"], key=lambda t: t["turn_index"])
        history_lines: List[str] = []
        for t in turns_sorted:
            target_lang = lang_b if t["speaker"] == "A" else lang_a
            taxonomy = (
                cg.load_function_taxonomy(target_lang)
                if use_grounding and cg.taxonomy_available(target_lang)
                else None
            )
            requests.append(
                cg.build_turn_checklist_batch_request(
                    f"{conv_id}::turn::{t['turn_index']}",
                    target_lang,
                    s["conversation_context"],
                    t["speaker"],
                    t["text"],
                    history_text="\n".join(history_lines),
                    taxonomy=taxonomy,
                    thinking_level=checklist_thinking,
                    cultural_context=cultural_context,
                )
            )
            history_lines.append(f"Turn {t['turn_index']} ({t['speaker']}): {t['text']}")

        conv_taxonomy = (
            cg.load_function_taxonomy(lang_b) if use_grounding and cg.taxonomy_available(lang_b) else None
        )
        requests.append(
            cg.build_conversation_checklist_batch_request(
                f"{conv_id}::conv",
                lang_b,
                s["conversation_context"],
                "\n".join(history_lines),
                taxonomy=conv_taxonomy,
                thinking_level=checklist_thinking,
                cultural_context=cultural_context,
            )
        )

    print(f"Submitting {len(requests)} checklist request(s) as ONE batch job for {len(scenarios)} conversation(s)...")
    batch_client = build_batch_client(checklist_provider_type)
    job_id = batch_client.submit(requests, checklist_model)
    print(f"Batch job: {job_id}")

    state = batch_client.poll(job_id)
    while state not in TERMINAL:
        detail = batch_client.progress(job_id)
        detail_str = f" ({detail})" if detail else ""
        print(f"  job {job_id} state={state}{detail_str}; waiting {poll_interval:.0f}s")
        time.sleep(poll_interval)
        state = batch_client.poll(job_id)
    if state == FAILED:
        print(f"Batch job {job_id} FAILED — no changes written.")
        return

    req_stubs = [BatchRequest(custom_id=r.custom_id, prompt="") for r in requests]
    results = batch_client.collect(job_id, req_stubs)

    updated = 0
    for s in scenarios:
        conv_id = s["conversation_id"]
        turns_sorted = sorted(s["turns"], key=lambda t: t["turn_index"])
        ok = True
        new_turns = []
        for t in turns_sorted:
            items = cg.parse_checklist_batch_response(
                results.get(f"{conv_id}::turn::{t['turn_index']}"), cg.TURN_HARD_CEILING
            )
            errs = cg.validate_checklist_items(items, cg.TURN_HARD_CEILING)
            if errs:
                print(f"  {conv_id} turn {t['turn_index']} checklist INVALID: {'; '.join(errs)} — keeping old checklist")
                ok = False
                break
            note = cg.checklist_count_note(items, cg.TURN_ITEM_CAP, cg.TURN_HARD_CEILING)
            if note:
                print(f"    {conv_id} turn {t['turn_index']}: {note}")
            t2 = dict(t)
            t2["checklist_items"] = [item.model_dump() for item in items]
            t2["verification_prompt"] = cg.compose_verification_prompt(items)
            new_turns.append(t2)
        if not ok:
            continue

        conv_items = cg.parse_checklist_batch_response(results.get(f"{conv_id}::conv"), cg.CONVERSATION_HARD_CEILING)
        conv_errs = cg.validate_checklist_items(conv_items, cg.CONVERSATION_HARD_CEILING)
        if conv_errs:
            print(f"  {conv_id} conversation checklist INVALID: {'; '.join(conv_errs)} — keeping old checklist")
            continue
        conv_note = cg.checklist_count_note(conv_items, cg.CONVERSATION_ITEM_CAP, cg.CONVERSATION_HARD_CEILING)
        if conv_note:
            print(f"    {conv_id} conversation: {conv_note}")

        s["turns"] = new_turns
        s["conversation_checklist_items"] = [item.model_dump() for item in conv_items]
        s["conversation_verification_prompt"] = cg.compose_verification_prompt(conv_items)
        s.setdefault("generation_metadata", {})["checklist_regenerated_with"] = checklist_model
        updated += 1

    with open(input_path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    turn_hist = Counter(len(t["checklist_items"]) for s in scenarios for t in s["turns"])
    conv_hist = Counter(len(s["conversation_checklist_items"]) for s in scenarios)
    print(f"\n{updated}/{len(scenarios)} conversation(s) regenerated. Rewrote {input_path}")
    print(f"Turn-level item-count distribution: {dict(sorted(turn_hist.items()))}")
    print(f"Conversation-level item-count distribution: {dict(sorted(conv_hist.items()))}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Scripted multi-turn scenario JSONL to update in place")
    p.add_argument("--pair", required=True, help="2-letter pair, order = A-then-B, e.g. id-ko, ar-ko")
    p.add_argument("--checklist-provider", default=registry.DEFAULT_JUDGE_PROVIDER)
    p.add_argument("--checklist-model", default=registry.DEFAULT_JUDGE_MODEL)
    p.add_argument("--checklist-thinking", default=registry.DEFAULT_JUDGE_THINKING_LEVEL)
    p.add_argument("--no-function-grounding", action="store_true")
    p.add_argument("--poll-interval", type=float, default=30.0)
    args = p.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not found in environment.")

    two_a, two_b = args.pair.split("-")
    lang_a, lang_b = _2_TO_ISO3[two_a], _2_TO_ISO3[two_b]
    regenerate(
        args.input,
        lang_a,
        lang_b,
        args.checklist_provider,
        args.checklist_model,
        args.checklist_thinking,
        use_grounding=not args.no_function_grounding,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
