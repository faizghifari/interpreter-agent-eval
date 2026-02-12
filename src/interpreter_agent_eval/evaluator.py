"""Evaluation framework for interpreter agents."""

from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime
from pathlib import Path

from .user import User
from .interpreter import InterpreterAgent
from .models import JudgeEvaluation, LanguageCheckResult, JudgeCriterionResult
from .prompts.templates import JUDGE_EVALUATION_PROMPT
from .utils.language_verification import verify_language_with_glotlid


class EvaluationFramework:
    """Framework for evaluating interpreter/translator agents.

    Orchestrates conversations between two users (potentially LLM-powered)
    speaking different languages, with an interpreter agent facilitating communication.
    """

    def __init__(
        self,
        user1: User,
        user2: User,
        interpreter: InterpreterAgent,
        conversation_context: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Initialize the evaluation framework.

        Args:
            user1: First user
            user2: Second user
            interpreter: Interpreter agent
            conversation_context: Optional brief description of the conversation scenario
            name: Optional name for this evaluation session
        """
        self.user1 = user1
        self.user2 = user2
        self.interpreter = interpreter
        self.conversation_context = (
            conversation_context or "A general conversation between two users."
        )
        self.name = name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_log = []
        self.metrics = {}
        self.judge_evaluation = None

    def run_conversation(
        self, messages: List[str], from_user: int = 1
    ) -> List[Dict[str, Any]]:
        """Run a conversation between the two users via the interpreter.

        Args:
            messages: List of messages to exchange. Each message is sent by alternating users.
            from_user: Which user starts (1 or 2)

        Returns:
            List of conversation exchanges

        Note:
            Users only communicate through the interpreter agent. Each user's conversation
            history contains only their own messages and the interpreter's translated responses.
            For LLM-powered users, they generate responses based on the translated messages
            they receive from the interpreter.
        """
        current_user = self.user1 if from_user == 1 else self.user2
        other_user = self.user2 if from_user == 1 else self.user1

        for turn, message in enumerate(messages):
            # Record the turn
            turn_data = {
                "turn": turn + 1,
                "timestamp": datetime.now().isoformat(),
                "from_user": current_user.name,
                "to_user": other_user.name,
            }

            # Current user sends message in their language
            sent_message = current_user.send_message(message)
            turn_data["original_message"] = sent_message
            turn_data["original_language"] = current_user.language

            # Interpreter translates
            start_time = time.time()
            translation_result = self.interpreter.facilitate_conversation(
                sent_message,
                current_user.language,
                other_user.language,
                context=f"Turn {turn + 1} of conversation",
            )
            translation_time = time.time() - start_time

            turn_data["translated_message"] = translation_result["translation"]
            turn_data["translated_language"] = translation_result[
                "translation_language"
            ]
            turn_data["translation_time"] = translation_time

            # Other user receives the translation from interpreter (not directly from the user)
            other_user.receive_message(
                translation_result["translation"], metadata={"from": "interpreter"}
            )

            # Log the turn
            self.conversation_log.append(turn_data)

            # Swap users for next turn
            current_user, other_user = other_user, current_user

        return self.conversation_log

    def evaluate_translation_quality(self) -> Dict[str, Any]:
        """Evaluate the quality of translations.

        This is a placeholder for various evaluation metrics.
        In practice, you might use BLEU, COMET, or other MT metrics.

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.conversation_log:
            return {"error": "No conversation data to evaluate"}

        metrics = {
            "total_turns": len(self.conversation_log),
            "average_translation_time": sum(
                turn.get("translation_time", 0) for turn in self.conversation_log
            )
            / len(self.conversation_log),
            "languages": {
                self.user1.language: self.user1.name,
                self.user2.language: self.user2.name,
            },
        }

        self.metrics = metrics
        return metrics

    def evaluate_with_judge(
        self,
        judge_llm_provider: Any,
        verification_prompt: str,
        turn_index: int = -1,
        glotlid_model: Optional[Any] = None,
        min_confidence: float = 0.9,
        response_text: Optional[str] = None,
        response_language: Optional[str] = None,
    ) -> JudgeEvaluation:
        """Evaluate a translation turn using a judge LLM with optional language verification.

        Args:
            judge_llm_provider: LLM provider to use as judge
            verification_prompt: Verification checklist with criteria questions
            turn_index: Which turn to evaluate (-1 for last turn)
            glotlid_model: Optional GlotLID model for language verification
            min_confidence: Minimum confidence for language verification (default 0.9)
            response_text: Optional explicit response text from target user
            response_language: Optional language code for response verification

        Returns:
            JudgeEvaluation object with structured results
        """
        if not self.conversation_log:
            raise ValueError("No conversation data to evaluate")

        turn = self.conversation_log[turn_index]

        # Get the target user (who received the translation)
        target_user = self.user1 if turn["to_user"] == self.user1.name else self.user2
        source_user = self.user2 if turn["to_user"] == self.user1.name else self.user1

        # Use provided response text or try to get from conversation history
        if response_text:
            target_response = response_text
        else:
            target_response = "No response recorded"
            if target_user.conversation_history:
                # Get the most recent message sent by target user
                for msg in reversed(target_user.conversation_history):
                    if msg.get("role") == "assistant":
                        target_response = msg.get("content", "No response recorded")
                        break

        # Determine response language for verification
        response_lang_code = response_language or target_user.language

        # Language verification (if model provided)
        translation_lang_check = None
        response_lang_check = None
        language_check_passed = True

        if glotlid_model:
            # Verify translation language
            translation_verification = verify_language_with_glotlid(
                model=glotlid_model,
                text=turn["translated_message"],
                expected_iso_code=target_user.language,
                min_confidence=min_confidence,
                context_name="Translation",
            )
            translation_lang_check = LanguageCheckResult(
                is_correct=translation_verification.is_correct,
                detected_language=translation_verification.detected_language,
                detected_script=translation_verification.detected_script,
                confidence=translation_verification.confidence,
                expected_language=translation_verification.expected_language,
                message=translation_verification.message,
            )

            # Verify target response language (if response exists)
            if target_response != "No response recorded":
                response_verification = verify_language_with_glotlid(
                    model=glotlid_model,
                    text=target_response,
                    expected_iso_code=response_lang_code,
                    min_confidence=min_confidence,
                    context_name="Target Response",
                )
                response_lang_check = LanguageCheckResult(
                    is_correct=response_verification.is_correct,
                    detected_language=response_verification.detected_language,
                    detected_script=response_verification.detected_script,
                    confidence=response_verification.confidence,
                    expected_language=response_verification.expected_language,
                    message=response_verification.message,
                )
                language_check_passed = response_verification.is_correct

            # If translation language check failed, immediately fail all criteria
            if not translation_lang_check.is_correct:
                language_check_passed = False
                print(
                    f"Translation language check failed: {translation_lang_check.message}"
                )

                # Parse verification prompt to get criteria count
                criteria_lines = [
                    line.strip()
                    for line in verification_prompt.split("\n")
                    if line.strip() and line.strip()[0].isdigit()
                ]

                # Create failed evaluation without calling judge
                failed_results = []
                for i, line in enumerate(criteria_lines, 1):
                    failed_results.append(
                        JudgeCriterionResult(
                            id=i,
                            criteria=line.lstrip("0123456789. "),
                            met=False,
                            reasoning="Translation was in the wrong language - automatic failure",
                        )
                    )

                evaluation = JudgeEvaluation(
                    results=failed_results,
                    translation_language_check=translation_lang_check,
                    response_language_check=response_lang_check,
                    language_check_passed=False,
                )
                self.judge_evaluation = evaluation
                return evaluation

        # Build language verification info for prompt
        lang_verification_info = (
            "Language verification not performed (model not provided)."
        )
        if translation_lang_check or response_lang_check:
            lang_info_parts = []
            if translation_lang_check:
                status = "✓ PASSED" if translation_lang_check.is_correct else "✗ FAILED"
                lang_info_parts.append(
                    f"- Translation Language Check: {status}\n  {translation_lang_check.message}"
                )
            if response_lang_check:
                status = "✓ PASSED" if response_lang_check.is_correct else "✗ FAILED"
                lang_info_parts.append(
                    f"- Target Response Language Check: {status}\n  {response_lang_check.message}"
                )
            lang_verification_info = "\n".join(lang_info_parts)

        # Build the judge prompt
        prompt = JUDGE_EVALUATION_PROMPT.format(
            conversation_context=self.conversation_context,
            source_text=turn["original_message"],
            translated_text=turn["translated_message"],
            target_response=target_response,
            verification_prompt=verification_prompt,
            language_verification_info=lang_verification_info,
        )

        # Get judge's evaluation using structured output
        try:
            # Check if provider supports response_mime_type (Google AI)
            response = judge_llm_provider.generate(
                prompt,
                response_mime_type="application/json",
                response_schema=JudgeEvaluation,
            )

            # Parse the response
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]

            # Parse with Pydantic
            evaluation = JudgeEvaluation.model_validate_json(clean_response)

            # Add language check results
            evaluation.translation_language_check = translation_lang_check
            evaluation.response_language_check = response_lang_check
            evaluation.language_check_passed = language_check_passed

            self.judge_evaluation = evaluation
            return evaluation

        except Exception as e:
            # Fallback: try without structured output
            try:
                response = judge_llm_provider.generate(prompt)
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]

                evaluation = JudgeEvaluation.model_validate_json(clean_response)

                # Add language check results
                evaluation.translation_language_check = translation_lang_check
                evaluation.response_language_check = response_lang_check
                evaluation.language_check_passed = language_check_passed

                self.judge_evaluation = evaluation
                return evaluation
            except Exception as inner_e:
                raise RuntimeError(
                    f"Judge evaluation failed: {str(e)}. Fallback also failed: {str(inner_e)}"
                )

    def export_results(self, filepath: str, format: str = "json") -> None:
        """Export evaluation results to a file.

        Args:
            filepath: Path to save the results
            format: Export format ('json' or 'txt')
        """
        results = {
            "session_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "conversation_context": self.conversation_context,
            "users": {
                "user1": {
                    "name": self.user1.name,
                    "language": self.user1.language,
                    "is_llm": self.user1.is_llm,
                },
                "user2": {
                    "name": self.user2.name,
                    "language": self.user2.language,
                    "is_llm": self.user2.is_llm,
                },
            },
            "interpreter": {
                "name": self.interpreter.name,
                "translation_brief": self.interpreter.translation_brief,
            },
            "conversation": self.conversation_log,
            "metrics": self.metrics,
            "judge_evaluation": (
                self.judge_evaluation.model_dump() if self.judge_evaluation else None
            ),
        }

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Evaluation Session: {self.name}\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")
                f.write(f"Users:\n")
                f.write(f"  User 1: {self.user1.name} ({self.user1.language})\n")
                f.write(f"  User 2: {self.user2.name} ({self.user2.language})\n\n")
                f.write(f"Interpreter: {self.interpreter.name}\n\n")
                f.write("Conversation:\n")
                f.write("=" * 80 + "\n")
                for turn in self.conversation_log:
                    f.write(f"\nTurn {turn['turn']}:\n")
                    f.write(
                        f"  From: {turn['from_user']} ({turn['original_language']})\n"
                    )
                    f.write(f"  Message: {turn['original_message']}\n")
                    f.write(f"  Translation: {turn['translated_message']}\n")
                    f.write(f"  Time: {turn['translation_time']:.3f}s\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("\nMetrics:\n")
                for key, value in self.metrics.items():
                    f.write(f"  {key}: {value}\n")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation.

        Returns:
            Summary dictionary
        """
        return {
            "session_name": self.name,
            "total_turns": len(self.conversation_log),
            "user1": f"{self.user1.name} ({self.user1.language})",
            "user2": f"{self.user2.name} ({self.user2.language})",
            "interpreter": self.interpreter.name,
            "metrics": self.metrics,
        }
