"""LLM-based evaluation handler for Tier 2 evaluation.

Uses an LLM judge to evaluate Nyaya reasoning quality according to a rubric.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample

from pramana.application.evaluation.handlers import EvaluationHandler
from pramana.application.evaluation.results import TierResult


class LLMClient(Protocol):
    """Protocol for LLM client interface.

    Implementations should provide a generate method that takes a prompt
    and optional temperature, returning a string response.
    """

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (default 0.0 for deterministic)

        Returns:
            Generated text response
        """
        ...


@dataclass
class NyayaRubric:
    """Rubric weights for Nyaya evaluation.

    Each phase and overall quality are weighted equally (1/7 each).
    Scores are normalized to 0-1 range after weighted aggregation.
    """

    samshaya: float = field(default=1 / 7)
    pramana: float = field(default=1 / 7)
    pancha_avayava: float = field(default=1 / 7)
    tarka: float = field(default=1 / 7)
    hetvabhasa: float = field(default=1 / 7)
    nirnaya: float = field(default=1 / 7)
    overall: float = field(default=1 / 7)

    def __post_init__(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        total = (
            self.samshaya
            + self.pramana
            + self.pancha_avayava
            + self.tarka
            + self.hetvabhasa
            + self.nirnaya
            + self.overall
        )
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Rubric weights must sum to 1.0, got {total}")


class Tier2LLMJudgeHandler(EvaluationHandler):
    """Tier 2 handler: LLM-based evaluation using Nyaya rubric.

    Uses an LLM judge (GPT-4/Claude) to evaluate the quality of Nyaya reasoning
    according to a structured rubric. Scores each of the 6 phases (0-10) and
    calculates a weighted aggregate score.
    """

    # Prompt template for LLM judge
    JUDGE_PROMPT_TEMPLATE = """You are evaluating a Nyaya reasoning solution according to the 6-phase Nyaya methodology.

Problem:
{problem}

Model Output:
{output}

Evaluate the solution on each of the 6 Nyaya phases, scoring each from 0-10:
- samshaya: Quality of doubt analysis (appropriate doubt type, clear justification)
- pramana: Quality of knowledge sources (appropriate use of Pratyaksha, Anumana, Upamana, Shabda)
- pancha_avayava: Quality of 5-member syllogisms (complete, logically sound, proper structure)
- tarka: Quality of counterfactual reasoning (meaningful hypothesis, clear analysis)
- hetvabhasa: Quality of fallacy detection (thorough checking, correct identification)
- nirnaya: Quality of final conclusion (clear answer, appropriate confidence, justification)
- overall: Overall coherence and adherence to Nyaya methodology

Return your evaluation as a JSON object with integer scores (0-10) for each phase:
{{
  "samshaya": <score>,
  "pramana": <score>,
  "pancha_avayava": <score>,
  "tarka": <score>,
  "hetvabhasa": <score>,
  "nirnaya": <score>,
  "overall": <score>
}}

Respond with ONLY the JSON object, no additional text.
"""

    def __init__(
        self,
        llm_client: LLMClient,
        rubric: NyayaRubric | None = None,
        next_handler: EvaluationHandler | None = None,
    ) -> None:
        """Initialize Tier2LLMJudgeHandler.

        Args:
            llm_client: LLM client implementation (must follow LLMClient protocol)
            rubric: Evaluation rubric with weights (defaults to equal weights)
            next_handler: Optional next handler in the evaluation chain
        """
        super().__init__(next_handler)
        self._llm_client = llm_client
        self._rubric = rubric or NyayaRubric()

    def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
        """Evaluate the output using LLM judge.

        Args:
            example: The NyayaExample to evaluate against
            output: The model output string to evaluate

        Returns:
            TierResult with tier=2, weighted score (0-1), and phase scores in details
        """
        try:
            # Build prompt
            prompt = self._build_prompt(example.problem, output)

            # Call LLM
            llm_response = self._llm_client.generate(prompt, temperature=0.0)

            # Parse JSON response
            phase_scores = self._parse_response(llm_response)

            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(phase_scores)

            # Normalize to 0-1 range (scores are 0-10)
            normalized_score = weighted_score / 10.0

            # Determine pass/fail (threshold: 0.7 = 70%)
            passed = normalized_score >= 0.7

            return TierResult(
                tier=2,
                passed=passed,
                score=normalized_score,
                details={
                    "phase_scores": phase_scores,
                    "weighted_score": weighted_score,
                    "rubric_weights": {
                        "samshaya": self._rubric.samshaya,
                        "pramana": self._rubric.pramana,
                        "pancha_avayava": self._rubric.pancha_avayava,
                        "tarka": self._rubric.tarka,
                        "hetvabhasa": self._rubric.hetvabhasa,
                        "nirnaya": self._rubric.nirnaya,
                        "overall": self._rubric.overall,
                    },
                },
                errors=[],
            )

        except Exception as e:
            # Handle LLM failures, JSON parsing errors, etc.
            return TierResult(
                tier=2,
                passed=False,
                score=0.0,
                details={"error_type": type(e).__name__},
                errors=[f"LLM evaluation failed: {e!s}"],
            )

    def _build_prompt(self, problem: str, output: str) -> str:
        """Build the prompt for the LLM judge.

        Args:
            problem: The problem statement
            output: The model output to evaluate

        Returns:
            Formatted prompt string
        """
        return self.JUDGE_PROMPT_TEMPLATE.format(problem=problem, output=output)

    def _parse_response(self, response: str) -> dict[str, int]:
        """Parse JSON response from LLM.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary mapping phase names to scores (0-10)

        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        # Try to extract JSON from response (in case LLM adds extra text)
        response = response.strip()
        if response.startswith("```json"):
            # Extract JSON from code block
            start = response.find("{")
            end = response.rfind("}") + 1
            response = response[start:end]
        elif response.startswith("```"):
            # Extract from generic code block
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        try:
            scores = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e

        # Validate required fields and provide defaults for missing ones
        required_phases = [
            "samshaya",
            "pramana",
            "pancha_avayava",
            "tarka",
            "hetvabhasa",
            "nirnaya",
            "overall",
        ]

        parsed_scores: dict[str, int] = {}
        for phase in required_phases:
            if phase in scores:
                score = scores[phase]
                # Ensure score is integer and in valid range
                if isinstance(score, (int, float)):
                    parsed_scores[phase] = max(0, min(10, int(score)))
                else:
                    parsed_scores[phase] = 0  # Default to 0 for invalid scores
            else:
                parsed_scores[phase] = 0  # Default to 0 for missing phases

        return parsed_scores

    def _calculate_weighted_score(self, phase_scores: dict[str, int]) -> float:
        """Calculate weighted aggregate score.

        Args:
            phase_scores: Dictionary mapping phase names to scores (0-10)

        Returns:
            Weighted score (0-10 range, not yet normalized)
        """
        weighted_sum = (
            phase_scores.get("samshaya", 0) * self._rubric.samshaya
            + phase_scores.get("pramana", 0) * self._rubric.pramana
            + phase_scores.get("pancha_avayava", 0) * self._rubric.pancha_avayava
            + phase_scores.get("tarka", 0) * self._rubric.tarka
            + phase_scores.get("hetvabhasa", 0) * self._rubric.hetvabhasa
            + phase_scores.get("nirnaya", 0) * self._rubric.nirnaya
            + phase_scores.get("overall", 0) * self._rubric.overall
        )

        return weighted_sum
