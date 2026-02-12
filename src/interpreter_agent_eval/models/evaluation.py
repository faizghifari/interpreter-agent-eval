"""Pydantic models for evaluation results."""

from pydantic import BaseModel, Field
from typing import List, Optional


class LanguageCheckResult(BaseModel):
    """Result of language verification check."""

    is_correct: bool = Field(description="Whether the language check passed")
    detected_language: str = Field(description="Detected ISO 639-3 language code")
    detected_script: str = Field(description="Detected script (e.g., Arab, Latn, Hang)")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    expected_language: str = Field(description="Expected ISO 639-3 language code")
    message: str = Field(description="Verification message")


class JudgeCriterionResult(BaseModel):
    """Result for a single evaluation criterion."""

    id: int = Field(description="Criterion ID (1-indexed)")
    criteria: str = Field(description="The criterion being evaluated")
    met: bool = Field(description="Whether the criterion was met (True/False)")
    reasoning: str = Field(description="Brief explanation of the judgment")


class JudgeEvaluation(BaseModel):
    """Complete judge evaluation result."""

    results: List[JudgeCriterionResult] = Field(
        description="List of evaluation results for each criterion"
    )
    translation_language_check: Optional[LanguageCheckResult] = Field(
        default=None, description="Language verification result for the translation"
    )
    response_language_check: Optional[LanguageCheckResult] = Field(
        default=None,
        description="Language verification result for the target user's response",
    )
    language_check_passed: bool = Field(
        default=True, description="Whether all language checks passed"
    )

    def get_completion_rate(self) -> str:
        """Calculate completion rate as 'X/Y' format.

        Returns:
            String in format "{met}/{total}"
        """
        met_count = sum(1 for r in self.results if r.met)
        total_count = len(self.results)
        return f"{met_count}/{total_count}"

    def get_success_rate(self) -> float:
        """Calculate success rate as a percentage.

        Returns:
            Float between 0.0 and 1.0
        """
        if not self.results:
            return 0.0
        met_count = sum(1 for r in self.results if r.met)
        return met_count / len(self.results)
