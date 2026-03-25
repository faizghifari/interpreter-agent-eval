import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure src is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

from interpreter_agent_eval.evaluator import EvaluationFramework
from interpreter_agent_eval.interpreter import InterpreterAgent
from interpreter_agent_eval.providers.google_ai import GoogleAIProvider
from interpreter_agent_eval.user import User
from interpreter_agent_eval.utils.language_verification import load_glotlid_model

load_dotenv()


TARGETS = {
    "kor": "Korean",
    "arb": "Arabic",
}


class LayeredChecklist(BaseModel):
    layer_1_semantic_core: List[str] = Field(
        description="Yes/No criteria for literal semantic transfer. Every criterion must be phrased so Yes means success."
    )
    layer_2_pragmatic_function: List[str] = Field(
        description="Yes/No criteria for underlying speech act success. Every criterion must be phrased so Yes means success."
    )
    layer_3_cultural_social_constraints: List[str] = Field(
        description="Yes/No criteria for culture/social adaptation success. Every criterion must be phrased so Yes means success."
    )


class MAPSAugmentedData(BaseModel):
    source_language: str = Field(description="Full source language name")
    target_language: str = Field(description="Full target language name")
    source_language_code: str = Field(description="ISO 639-3 source code")
    target_language_code: str = Field(description="ISO 639-3 target code")
    speech_act_intent: str = Field(
        description="Primary pragmatic intent of source message, e.g., request, complaint, warning"
    )
    semantic_core: str = Field(
        description="Short literal meaning summary that must be preserved"
    )
    mandatory_cultural_constraints: List[str] = Field(
        description="Concrete cultural/social constraints needed for faithful translation"
    )
    conversation_context: str = Field(
        description="One-sentence neutral conversation setup in English"
    )
    user_a_context: str = Field(
        description="Detailed context for source speaker in Indonesian"
    )
    user_b_context: str = Field(
        description="Detailed context for target listener in target language"
    )
    source_text: str = Field(
        description="Single-turn source message in Indonesian that naturally reflects the proverb use"
    )
    checklist: LayeredChecklist = Field(
        description="Three-layer checklist where every question is Yes=success"
    )
    verification_prompt: str = Field(
        description="Flattened numbered yes/no checklist for judge LLM; can be assessed via translated text and/or target response"
    )


PROMPT_TEMPLATE = """You are an expert cross-cultural linguist and pragmatic evaluation designer.

You must convert Indonesian MAPS proverb seed data into a simulation-ready interpretation task.

Seed Data:
- Split: {seed_split}
- Proverb: {proverb}
- Conversation Seed: {conversation}
- Explanation Seed (noisy/weak quality possible): {explanation}
- Candidate Meaning A: {answer1}
- Candidate Meaning B: {answer2}
- Figurative Flag: {is_figurative}
- Annotated Key: {answer_key}

Required language direction:
- source language: Indonesian (ind)
- target language: {target_language} ({target_language_code})

Task requirements:
1) Infer the most likely underlying speech act / pragmatic intent from the seed context.
2) Identify explicit cultural/social constraints that the interpreter must preserve or explain in the target culture if unknown.
3) Build realistic conversation and user contexts for a one-turn simulation.
4) Produce a strict 3-layer checklist:
   - Layer 1 Semantic Core (Know): literal content fidelity.
   - Layer 2 Pragmatic Function (Do): speech act and communicative function fidelity.
   - Layer 3 Cultural/Social Constraints (Feel): target-culture adaptation (e.g., honorifics, hierarchy, politeness norms, social scripts, implied norms from CultureAtlas-style reasoning).
5) Every checklist item must be binary yes/no where YES means translation success.
6) The checklist must be evaluable from either translated text and/or target user response.

Output constraints:
- user_a_context must be in Indonesian.
- user_b_context must be in the target language and script.
- source_text must be a natural Indonesian single-turn utterance, not a glossary explanation.
- verification_prompt must be numbered (1., 2., 3., ...), concise, and only yes/no-oriented success criteria.
- Keep cultural constraints actionable and concrete (avoid vague statements).
"""


def create_generation_provider() -> GoogleAIProvider:
    return GoogleAIProvider(
        model_name="gemini-3.1-pro-preview",
        max_tokens=8192,
        temperature=0.9,
        thinking_config={"thinking_level": "high"},
    )


def create_flash_lite_provider() -> GoogleAIProvider:
    return GoogleAIProvider(
        model_name="gemini-3.1-flash-lite-preview",
        max_tokens=4096,
        temperature=0.7,
        thinking_config={"thinking_level": "minimal"},
    )


def parse_seed_rows(seed_xlsx_path: str, seed_split: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(seed_xlsx_path, sheet_name=seed_split)
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "seed_row_id": int(idx) + 1,
                "proverb": str(row.get("proverb", "")).strip(),
                "conversation": str(row.get("conversation", "")).strip(),
                "explanation": str(row.get("explanation", "")).strip(),
                "answer1": str(row.get("answer1", "")).strip(),
                "answer2": str(row.get("answer2", "")).strip(),
                "is_figurative": (
                    int(row.get("is_figurative", 0))
                    if str(row.get("is_figurative", "")).strip() != ""
                    else 0
                ),
                "answer_key": str(row.get("answer_key", "")).strip(),
            }
        )
    return rows


def load_existing_ids(output_path: str) -> set:
    existing = set()
    if not os.path.exists(output_path):
        return existing

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                existing.add(
                    (item.get("seed_row_id"), item.get("target_language_code"))
                )
            except json.JSONDecodeError:
                continue

    return existing


def generate_one_sample(
    provider: GoogleAIProvider,
    row: Dict[str, Any],
    seed_split: str,
    target_code: str,
    target_language: str,
) -> MAPSAugmentedData:
    prompt = PROMPT_TEMPLATE.format(
        seed_split=seed_split,
        proverb=row["proverb"],
        conversation=row["conversation"],
        explanation=row["explanation"],
        answer1=row["answer1"],
        answer2=row["answer2"],
        is_figurative=row["is_figurative"],
        answer_key=row["answer_key"],
        target_language=target_language,
        target_language_code=target_code,
    )

    text = provider.generate(
        prompt,
        response_mime_type="application/json",
        response_schema=MAPSAugmentedData,
    )
    generated = MAPSAugmentedData.model_validate_json(text)
    generated.verification_prompt = normalize_verification_prompt(
        generated.verification_prompt
    )
    return generated


def normalize_verification_prompt(raw_prompt: str) -> str:
    """Ensure verification criteria are newline-separated as `1.`, `2.`, ... items."""
    if not raw_prompt:
        return raw_prompt

    # Collapse excessive whitespace first.
    normalized = " ".join(raw_prompt.split())

    # Split on numbered item boundaries, preserving the numbering token.
    parts = re.split(r"(?=\b\d+\.)", normalized)
    items = [p.strip() for p in parts if p.strip()]

    # If model returned plain bullets with no numbering, keep original with line breaks.
    if not any(re.match(r"^\d+\.", item) for item in items):
        return raw_prompt.strip()

    return "\n".join(items)


def append_jsonl(output_path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_output_record(
    seed_file: str,
    seed_split: str,
    row: Dict[str, Any],
    generated: MAPSAugmentedData,
) -> Dict[str, Any]:
    base = {
        "seed_file": seed_file,
        "seed_split": seed_split,
        "seed_row_id": row["seed_row_id"],
        "Category": "MAPS-Proverb-Pragmatics",
        "Source Concept (Indonesian)": row["proverb"],
        "Verification Goal (Target Receiver)": generated.semantic_core,
        'Linguistic/Cultural "Trap"': " | ".join(
            generated.mandatory_cultural_constraints
        ),
        "seed_conversation": row["conversation"],
        "seed_explanation": row["explanation"],
        "seed_answer1": row["answer1"],
        "seed_answer2": row["answer2"],
        "seed_is_figurative": row["is_figurative"],
        "seed_answer_key": row["answer_key"],
    }

    enriched = generated.model_dump()
    enriched["checklist_layer_1_semantic_core"] = (
        generated.checklist.layer_1_semantic_core
    )
    enriched["checklist_layer_2_pragmatic_function"] = (
        generated.checklist.layer_2_pragmatic_function
    )
    enriched["checklist_layer_3_cultural_social_constraints"] = (
        generated.checklist.layer_3_cultural_social_constraints
    )
    del enriched["checklist"]

    return {**base, **enriched}


def augment_maps_data(
    seed_xlsx_path: str,
    seed_split: str,
    output_dir: str,
    limit: Optional[int],
    start_row: int,
    targets: List[str],
) -> List[str]:
    print("Initializing generation provider: gemini-3.1-pro-preview")
    provider = create_generation_provider()

    rows = parse_seed_rows(seed_xlsx_path, seed_split)
    rows = [r for r in rows if r["seed_row_id"] >= start_row]
    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print("No rows available after applying filters.")
        return []

    generated_paths = []

    for target_code in targets:
        if target_code not in TARGETS:
            raise ValueError(f"Unsupported target code: {target_code}")

        target_language = TARGETS[target_code]
        output_path = os.path.join(output_dir, f"id_{target_code}_maps.jsonl")
        existing = load_existing_ids(output_path)
        generated_paths.append(output_path)

        print(
            f"\nGenerating {len(rows)} rows for ind -> {target_code} ({target_language}) -> {output_path}"
        )

        for i, row in enumerate(rows, 1):
            key = (row["seed_row_id"], target_code)
            if key in existing:
                if i % 25 == 0 or i == len(rows):
                    print(
                        f"  [{i}/{len(rows)}] seed_row_id={row['seed_row_id']} skipped (already exists)"
                    )
                continue

            max_retries = 3
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    generated = generate_one_sample(
                        provider=provider,
                        row=row,
                        seed_split=seed_split,
                        target_code=target_code,
                        target_language=target_language,
                    )
                    output_record = build_output_record(
                        seed_file=os.path.basename(seed_xlsx_path),
                        seed_split=seed_split,
                        row=row,
                        generated=generated,
                    )
                    append_jsonl(output_path, output_record)
                    if i % 10 == 0 or i == len(rows):
                        print(
                            f"  [{i}/{len(rows)}] seed_row_id={row['seed_row_id']} generated"
                        )
                    success = True
                    break
                except Exception as e:
                    print(
                        f"  [{i}/{len(rows)}] seed_row_id={row['seed_row_id']} attempt {attempt}/{max_retries} failed: {e}"
                    )
                    time.sleep(2)

            if not success:
                print(
                    f"  [{i}/{len(rows)}] seed_row_id={row['seed_row_id']} failed after retries"
                )

            time.sleep(0.2)

    return generated_paths


def simulate_and_evaluate(
    data_file: str,
    num_samples: int,
    output_dir: str,
    verify_language: bool,
) -> str:
    """Run one-turn simulation and judge evaluation on generated JSONL.

    Uses gemini-3.1-flash-lite-preview for:
    - Interpreter
    - Target user simulation (Arabic/Korean users)
    - Judge evaluation
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(data_file).replace(".jsonl", "")
    eval_path = os.path.join(output_dir, f"eval_{base_name}_{timestamp}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    glotlid_model = load_glotlid_model() if verify_language else None

    interpreter_provider = create_flash_lite_provider()
    user_provider = create_flash_lite_provider()
    judge_provider = create_flash_lite_provider()

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break

    if not samples:
        print(f"No samples found in {data_file}")
        return eval_path

    for idx, sample in enumerate(samples, 1):
        source_code = sample["source_language_code"]
        target_code = sample["target_language_code"]

        interpreter = InterpreterAgent(
            llm_provider=interpreter_provider,
            source_language=source_code,
            target_language=target_code,
            conversation_context=sample.get("conversation_context"),
            name="AI Interpreter",
        )

        user_a = User(
            name="User A",
            language=source_code,
            is_llm=False,
            llm_provider=user_provider,
            context=sample.get("user_a_context", ""),
        )
        user_b = User(
            name="User B",
            language=target_code,
            is_llm=True,
            llm_provider=user_provider,
            context=sample.get("user_b_context", ""),
        )

        evaluator = EvaluationFramework(
            user1=user_a,
            user2=user_b,
            interpreter=interpreter,
            conversation_context=sample.get("conversation_context"),
            name=f"maps_eval_{base_name}_{idx}",
        )

        source_text = sample["source_text"]
        evaluator.run_conversation([source_text], from_user=1)

        # Trigger target response generation based on received translated message
        response_text = user_b.send_message("")

        judge_eval = evaluator.evaluate_with_judge(
            judge_llm_provider=judge_provider,
            verification_prompt=sample["verification_prompt"],
            turn_index=0,
            glotlid_model=glotlid_model,
            min_confidence=0.8,
            response_text=response_text,
            response_language=target_code,
        )

        turn = evaluator.conversation_log[0]
        out = {
            "sample_index": idx,
            "seed_row_id": sample.get("seed_row_id"),
            "timestamp": datetime.now().isoformat(),
            "category": sample.get("Category"),
            "source_lang": source_code,
            "target_lang": target_code,
            "source_text": source_text,
            "translated_text": turn.get("translated_message"),
            "target_response": response_text,
            "verification_prompt": sample.get("verification_prompt"),
            "evaluation": judge_eval.model_dump(),
            "completion_rate": judge_eval.get_completion_rate(),
            "success_rate": judge_eval.get_success_rate(),
        }

        append_jsonl(eval_path, out)
        print(
            f"  Eval [{idx}/{len(samples)}] seed_row_id={sample.get('seed_row_id')} completion={out['completion_rate']}"
        )

    return eval_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MAPS-based Indonesian->(Korean, Arabic) augmentation data and optionally run simulation/evaluation."
    )
    parser.add_argument(
        "--seed-xlsx",
        default="data/MAPS_Final/id/test_proverbs.xlsx",
        help="Path to MAPS Indonesian Excel file.",
    )
    parser.add_argument(
        "--seed-split",
        default="test_proverbs",
        help="Sheet name in the seed Excel file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/enriched",
        help="Directory for generated JSONL outputs.",
    )
    parser.add_argument(
        "--targets",
        default="kor,arb",
        help="Comma-separated target language codes. Supported: kor,arb",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of seed rows to process.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based seed row index to start from.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run simulation + judge evaluation after generation.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=5,
        help="How many samples to evaluate per generated file when --run-eval is set.",
    )
    parser.add_argument(
        "--eval-output-dir",
        default="outputs",
        help="Directory for evaluation JSONL outputs.",
    )
    parser.add_argument(
        "--verify-language",
        action="store_true",
        help="Enable GlotLID language checks in judge evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not found in environment.")

    generated_files = augment_maps_data(
        seed_xlsx_path=args.seed_xlsx,
        seed_split=args.seed_split,
        output_dir=args.output_dir,
        limit=args.limit,
        start_row=args.start_row,
        targets=targets,
    )

    print("\nGenerated files:")
    for path in generated_files:
        print(f"- {path}")

    if args.run_eval:
        print("\nRunning simulation + evaluation (gemini-3.1-flash-lite-preview)...")
        for data_file in generated_files:
            eval_file = simulate_and_evaluate(
                data_file=data_file,
                num_samples=args.eval_samples,
                output_dir=args.eval_output_dir,
                verify_language=args.verify_language,
            )
            print(f"- Evaluation output: {eval_file}")


if __name__ == "__main__":
    main()
