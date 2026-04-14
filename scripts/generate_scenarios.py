import csv
import os
import json
import sys
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from huggingface_hub import hf_hub_download
import fasttext

# Adjust path to find src
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from interpreter_agent_eval.providers.google_ai import GoogleAIProvider
from google.genai import types

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# --- Pydantic Models for Structured Output ---


class SimulationData(BaseModel):
    source_language: str = Field(
        description="Full English name of source language. Choose the direction (Source->Target) that maximizes the difficulty/relevance of the trap."
    )
    target_language: str = Field(description="Full English name of target language.")
    source_language_code: str = Field(
        description="ISO 639-3 three-letter language code for the source language (e.g., 'arb', 'ind', 'kor', 'jpn')"
    )
    target_language_code: str = Field(
        description="ISO 639-3 three-letter language code for the target language"
    )
    conversation_context: str = Field(
        description="A very brief, surface-level one-sentence description in English of the conversation context (e.g., 'An Indonesian student wants to introduce himself to his new Korean colleague' or 'An Indonesian tourist came to a Korean restaurant'). Must NOT reveal any hints about the source concept or linguistic/cultural trap. This will be given to both the interpreter and all users."
    )
    user_a_context: str = Field(
        description="Detailed system prompt for User A written in [Source Language]. MUST include specific personal details (e.g., exact age '25', specific dietary requirements 'no alcohol', occupation, background) that determine their intent. Focus on User A's personal information and situation ONLY. Do NOT compare cultures, do NOT mention User B's culture or language, and do NOT hint at potential misunderstandings or traps. User A should act naturally."
    )
    user_b_context: str = Field(
        description="Detailed system prompt for User B written in [Target Language]. Define User B's personal details, occupation, background, and relevant information about themselves. Focus on User B's personal situation ONLY. Do NOT compare cultures, do NOT mention User A's culture or language, and do NOT hint at potential misunderstandings or what User A might say. They know they are using an AI interpreter."
    )
    source_text: str = Field(
        description="Natural, IMPLICIT utterance in [Source Language]. Do NOT explain the trap. Do NOT be helpful. Example: Say 'Umur saya 25' (ambiguous), NOT 'Umur saya 25 sistem internasional'. Force the interpreter to handle the ambiguity."
    )
    verification_prompt: str = Field(
        description="A simplified checklist for a Judge LLM to score success. Generate a single list of around 3-6 (can be less or more depending on cases) Yes/No questions. PHRASING CRITICAL: 'Yes' must ALWAYS mean SUCCESS (e.g., 'Did the interpreter correctly...', 'Did User B understand...'). Failure to handle the trap must yield a 'No'. Criteria should cover both the translation accuracy and the pragmatic success of User B's response. The list should be written in English."
    )


PROMPT_TEMPLATE = """You are an expert linguist and cultural consultant. I have a dataset of "translation traps".
I need to generate simulation data to test an AI Interpreter Agent.

Input Data:
Context/File: {filename}
Category: {category}
Source Concept: {concept}
Verification Goal: {goal}
Linguistic/Cultural Trap: {trap}

Task:
1. Analyze the trap. Determine the most effective Direction (Language A -> Language B).
2. Generate the simulation data.
   - **Conversation Context**: Write a very brief, surface-level one-sentence description in English about what this conversation is about. Do NOT reveal any hints about the linguistic/cultural trap or the source concept. Keep it generic (e.g., "An Indonesian student wants to introduce himself to a Korean colleague" or "A tourist asks about menu options at a restaurant").
   - **User Contexts**: Be deep. Write User A's context in [Source Language] and User B's context in [Target Language].
     - CRITICAL: Focus ONLY on each user's PERSONAL information (age, occupation, background, dietary restrictions, beliefs, situation, etc.).
     - Do NOT compare cultures or languages in either context.
     - Do NOT mention the other user's culture, language, or potential misunderstandings.
     - Do NOT provide hints about the trap or what might go wrong.
     - Just describe each person's individual situation and relevant personal details.
   - **Source Text**: Be HARD. Be IMPLICIT. Use [Source Language].
   - **Verification**: Generate a single list of 3-5 "Success Criteria".
     - Each item must be a binary Question where "Yes" means PASS/SUCCESS.
     - Questions must evaluate:
       1. Translation Accuracy: Did it handle the trap correctly?
       2. Pragmatic Outcome: Did the communication succeed based on User B's response?
     - Do NOT split into named sections. Just provide the list of numbered questions.

Ensure that the text for User A and User B is written in the correct language AND the correct script (e.g., Arabic in Arabic script, NOT Latin).
"""

# --- GlotLID Setup ---


def load_glotlid_model():
    print("Loading GlotLID language identification model...")
    try:
        model_path = hf_hub_download(
            repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None
        )
        model = fasttext.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load GlotLID model: {e}")
        return None


def verify_language(model, text, expected_iso_code, context_name="Text"):
    if not model:
        return True, "Model not loaded"

    # Clean newlines for prediction
    clean_text = text.replace("\n", " ")
    try:
        predictions = model.predict(clean_text)
        # Prediction format: (('__label__eng_Latn',), array([0.99...]))
        if not predictions or not predictions[0]:
            return False, "No prediction returned"

        label = predictions[0][0]
        confidence = predictions[1][0]

        # Parse label: __label__{iso}_{script}
        # e.g. __label__arb_Arab
        parts = label.replace("__label__", "").split("_")
        if len(parts) >= 2:
            pred_iso = parts[0]
            pred_script = parts[1]
        else:
            pred_iso = label
            pred_script = "Unknown"

        # Check ISO code
        if pred_iso != expected_iso_code:
            # Allow some flexibility if confidence is low or specific known mismatches?
            # For now, strict check
            msg = f"{context_name} detected as {pred_iso} ({pred_script}), expected {expected_iso_code}."
            return False, msg

        # Specific Script Checks
        script_map = {
            "arb": "Arab",
            "kor": ["Hang", "Kore"],
            "jpn": ["Jpan", "Hira", "Kana"],
            "mal": "Latn",  # Malay
            "ind": "Latn",  # Indonesian
            "zho": ["Hani", "Hans", "Hant"],
            "eng": "Latn",
        }

        if expected_iso_code in script_map:
            allowed_scripts = script_map[expected_iso_code]
            if isinstance(allowed_scripts, str):
                allowed_scripts = [allowed_scripts]

            if pred_script not in allowed_scripts:
                return (
                    False,
                    f"{context_name} in {expected_iso_code} but script is {pred_script}, expected {allowed_scripts}.",
                )

        return True, f"Verified as {pred_iso}_{pred_script} ({confidence:.2f})"

    except Exception as e:
        # print("Verification Error:", e)
        return True, f"Verification error: {e}"  # Fail open if prediction fails


# --- Main Logic ---


def process_file(provider, model, filename, data_dir, output_dir, limit=None):
    input_path = os.path.join(data_dir, filename)
    output_path = os.path.join(output_dir, filename.replace(".csv", ".jsonl"))

    print(f"Processing {filename} -> {output_path}")

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Load existing processed concepts to avoid duplicates
    processed_concepts = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f_out_read:
                for line in f_out_read:
                    try:
                        data = json.loads(line)
                        # Normalize keys for checking
                        concept_check = (
                            data.get("Source Concept")
                            or data.get("Source Concept (Arabic)")
                            or data.get("Source Concept (Korean)")
                            or data.get("Source Concept (Indonesian)")
                        )
                        if concept_check:
                            processed_concepts.add(concept_check)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    generated_count = 0

    with open(output_path, "a", encoding="utf-8") as f_out:  # Append mode
        for i, row in enumerate(rows):
            if limit and generated_count >= limit:
                print(f"Reached limit of {limit} samples. Stopping for this file.")
                break

            # Normalize keys
            concept = (
                row.get("Source Concept")
                or row.get("Source Concept (Arabic)")
                or row.get("Source Concept (Korean)")
                or row.get("Source Concept (Indonesian)")
            )
            goal = row.get("Verification Goal") or row.get(
                "Verification Goal (Target Receiver)"
            )
            trap = (
                row.get('Linguistic/Cultural "Trap"')
                or row.get('Pragmatic/Linguistic "Trap"')
                or row.get('Linguistic/Cultural ""Trap""')
            )

            if not concept:
                continue

            if concept in processed_concepts:
                print(
                    f"  Row {i+1}/{len(rows)}: Skipping (Already processed): {concept[:30]}..."
                )
                continue

            print(f"  Row {i+1}/{len(rows)}: Generating: {concept[:30]}...")

            prompt = PROMPT_TEMPLATE.format(
                filename=filename,
                category=row.get("Category", ""),
                concept=concept,
                goal=goal,
                trap=trap,
            )

            # Retry logic for API + Verification
            max_retries = 3
            success = False

            for attempt in range(max_retries):
                try:
                    # Generate with Structured Output
                    response_text = provider.generate(
                        prompt,
                        response_mime_type="application/json",
                        response_json_schema=SimulationData.model_json_schema(),
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                    )

                    # Parse
                    sim_data = SimulationData.model_validate_json(response_text)

                    # Verify Language
                    checks = []
                    if model:
                        c1, m1 = verify_language(
                            model,
                            sim_data.user_a_context,
                            sim_data.source_language_code,
                            "User A Context",
                        )
                        checks.append((c1, m1))

                        c2, m2 = verify_language(
                            model,
                            sim_data.source_text,
                            sim_data.source_language_code,
                            "Source Text",
                        )
                        checks.append((c2, m2))

                        c3, m3 = verify_language(
                            model,
                            sim_data.user_b_context,
                            sim_data.target_language_code,
                            "User B Context",
                        )
                        checks.append((c3, m3))

                        failures = [m for c, m in checks if not c]
                        if failures:
                            print(
                                f"    Validation Failed (Attempt {attempt+1}): {failures}"
                            )
                            # Add feedback to prompt for next retry?
                            # For simplicity, just retry the generation.
                            continue
                        else:
                            # print(f"    Validation Passed: {[m for c, m in checks]}")
                            pass

                    # Save
                    output_data = {**row, **sim_data.model_dump()}
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    f_out.flush()
                    print(f"    Success.")
                    success = True
                    generated_count += 1
                    break

                except Exception as e:
                    print(f"    Error (Attempt {attempt+1}): {e}")
                    time.sleep(2)

            if not success:
                print(f"    Skipping row after {max_retries} failures.")

            time.sleep(1)


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found.")
        return

    print("Initializing GoogleAIProvider...")
    provider = GoogleAIProvider(
        api_key=api_key,
        model_name="gemini-3-pro-preview",
        max_tokens=8192,  # Increased as requested
        temperature=1.0,
        thinking_config={"thinking_level": "high"},
    )

    # Load GlotLID
    glotlid_model = load_glotlid_model()

    data_dir = os.path.join(ROOT_DIR, "data")
    output_dir = os.path.join(ROOT_DIR, "data", "enriched")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    # Set limit here (e.g., 5 samples per file for testing, or None for all)
    SAMPLES_PER_FILE_LIMIT = None
    print(f"Limit set to: {SAMPLES_PER_FILE_LIMIT} new samples per file.")

    for filename in files:
        process_file(
            provider,
            glotlid_model,
            filename,
            data_dir,
            output_dir,
            limit=SAMPLES_PER_FILE_LIMIT,
        )


if __name__ == "__main__":
    main()
