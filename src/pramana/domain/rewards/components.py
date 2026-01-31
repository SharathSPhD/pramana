"""Individual reward components for GRPO training.

Each component calculates a specific aspect of reward (0-1 scale):
- FormatRewardComponent: Structural adherence to 6-phase Nyaya format
- ValidityRewardComponent: Logical validity (placeholder for Z3 verification)
- ConsistencyRewardComponent: Internal consistency checks
- CorrectnessRewardComponent: Ground truth matching
- StyleRewardComponent: Appropriate verbosity/clarity
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample


class RewardComponent:
    """Base class for reward components."""

    def calculate(self, example: "NyayaExample", output: str) -> float:
        """Calculate reward component score (0-1).

        Args:
            example: The NyayaExample with ground truth
            output: Generated output string to evaluate

        Returns:
            Score between 0.0 and 1.0
        """
        raise NotImplementedError


class FormatRewardComponent(RewardComponent):
    """Reward component for structural adherence to 6-phase Nyaya format.

    Checks for presence of all 6 required phases:
    - Samshaya (Doubt Analysis)
    - Pramana (Sources of Knowledge)
    - Pancha Avayava (5-Member Syllogism)
    - Tarka (Counterfactual Reasoning)
    - Hetvabhasa (Fallacy Detection)
    - Nirnaya (Conclusion)
    """

    REQUIRED_PHASES = [
        "samshaya",
        "pramana",
        "pancha avayava",
        "tarka",
        "hetvabhasa",
        "nirnaya",
    ]

    def calculate(self, example: "NyayaExample", output: str) -> float:
        """Calculate format reward based on phase presence.

        Returns:
            Score between 0.0 and 1.0 based on how many phases are present
        """
        if not output or not output.strip():
            return 0.0

        output_lower = output.lower()
        phases_found = 0

        for phase in self.REQUIRED_PHASES:
            # Check for markdown headers (## Phase Name or # Phase Name)
            pattern = rf"^#+\s+{re.escape(phase)}"
            if re.search(pattern, output_lower, re.MULTILINE | re.IGNORECASE):
                phases_found += 1

        # Return fraction of phases found
        return phases_found / len(self.REQUIRED_PHASES)


class ValidityRewardComponent(RewardComponent):
    """Reward component for logical validity.

    Placeholder for Z3 verification. Currently returns a simple heuristic.
    Future: Integrate with Z3Verifier for formal logic problems.
    """

    def calculate(self, example: "NyayaExample", output: str) -> float:
        """Calculate validity reward.

        Currently uses heuristics. Future: Z3 verification for formalizable problems.

        Returns:
            Score between 0.0 and 1.0
        """
        if not output or not output.strip():
            return 0.0

        # Heuristic: Check for logical connectives and reasoning structure
        logical_indicators = [
            r"\btherefore\b",
            r"\bthus\b",
            r"\bhence\b",
            r"\bif\s+.*\s+then\b",
            r"\bcontradiction\b",
            r"\bimplies\b",
            r"\b→\b",
            r"\b→\b",
        ]

        found_indicators = sum(1 for pattern in logical_indicators if re.search(pattern, output, re.IGNORECASE))

        # Normalize to [0, 1] based on number of indicators found
        # Cap at 1.0 if 3+ indicators found
        return min(found_indicators / 3.0, 1.0)


class ConsistencyRewardComponent(RewardComponent):
    """Reward component for internal consistency.

    Checks that:
    - Answer in Nirnaya matches conclusion in Pancha Avayava
    - Tarka hypothesis contradicts the conclusion
    - No contradictory statements
    """

    def calculate(self, example: "NyayaExample", output: str) -> float:
        """Calculate consistency reward.

        Returns:
            Score between 0.0 and 1.0
        """
        if not output or not output.strip():
            return 0.0

        score = 0.0
        checks = 0

        # Check 1: Extract answer from Nirnaya section
        nirnaya_match = re.search(r"##\s+nirnaya.*?answer[:\s]+([^\n]+)", output, re.IGNORECASE | re.DOTALL)
        if nirnaya_match:
            checks += 1
            answer = nirnaya_match.group(1).strip().lower()

            # Check 2: Answer appears in Pancha Avayava conclusion
            pancha_match = re.search(
                r"##\s+pancha\s+avayava.*?nigamana[:\s]+([^\n]+)", output, re.IGNORECASE | re.DOTALL
            )
            if pancha_match:
                checks += 1
                conclusion = pancha_match.group(1).strip().lower()
                # Simple substring match (can be improved with semantic similarity)
                if answer in conclusion or conclusion in answer:
                    score += 1.0

            # Check 3: Tarka hypothesis contradicts conclusion
            tarka_match = re.search(
                r"##\s+tarka.*?hypothesis[:\s]+([^\n]+)", output, re.IGNORECASE | re.DOTALL
            )
            if tarka_match:
                checks += 1
                hypothesis = tarka_match.group(1).strip().lower()
                # Check for negation words
                negation_words = ["not", "doesn't", "does not", "cannot", "cannot be"]
                if any(neg in hypothesis for neg in negation_words):
                    score += 1.0

        # Normalize by number of checks performed
        if checks > 0:
            return score / checks
        return 0.0


class CorrectnessRewardComponent(RewardComponent):
    """Reward component for ground truth matching.

    Extracts answer from Nirnaya section and compares with ground_truth.
    """

    def calculate(self, example: "NyayaExample", output: str) -> float:
        """Calculate correctness reward by matching ground truth.

        Returns:
            1.0 if answer matches ground truth, 0.0 otherwise
        """
        if not output or not output.strip():
            return 0.0

        # Extract answer from Nirnaya section
        nirnaya_match = re.search(r"##\s+nirnaya.*?answer[:\s]+([^\n]+)", output, re.IGNORECASE | re.DOTALL)
        if not nirnaya_match:
            return 0.0

        extracted_answer = nirnaya_match.group(1).strip()
        ground_truth = example.ground_truth.strip()

        # Normalize for comparison (lowercase, remove extra whitespace)
        extracted_normalized = " ".join(extracted_answer.lower().split())
        ground_truth_normalized = " ".join(ground_truth.lower().split())

        # Exact match
        if extracted_normalized == ground_truth_normalized:
            return 1.0

        # Substring match (for cases where answer includes extra context)
        if ground_truth_normalized in extracted_normalized or extracted_normalized in ground_truth_normalized:
            return 0.8  # Partial credit for substring match

        return 0.0


class StyleRewardComponent(RewardComponent):
    """Reward component for appropriate verbosity and clarity.

    Checks:
    - Appropriate length (not too short, not too verbose)
    - Clear structure with markdown headers
    - Readable formatting
    """

    MIN_WORDS = 50  # Minimum words for a complete reasoning trace
    MAX_WORDS = 2000  # Maximum words to avoid excessive verbosity

    def calculate(self, example: "NyayaExample", output: str) -> float:
        """Calculate style reward based on verbosity and clarity.

        Returns:
            Score between 0.0 and 1.0
        """
        if not output or not output.strip():
            return 0.0

        score = 0.0

        # Check 1: Word count in reasonable range
        word_count = len(output.split())
        if self.MIN_WORDS <= word_count <= self.MAX_WORDS:
            score += 0.4
        elif word_count < self.MIN_WORDS:
            # Penalize too short (but less harshly)
            score += 0.2 * (word_count / self.MIN_WORDS)
        else:
            # Penalize too verbose
            excess_ratio = (word_count - self.MAX_WORDS) / self.MAX_WORDS
            score += max(0.0, 0.4 * (1.0 - min(excess_ratio, 1.0)))

        # Check 2: Has markdown structure
        header_count = len(re.findall(r"^#+\s+", output, re.MULTILINE))
        if header_count >= 3:  # At least 3 sections
            score += 0.3
        elif header_count >= 1:
            score += 0.15

        # Check 3: Has some structure (lists, paragraphs)
        has_lists = bool(re.search(r"^[-*+]\s+", output, re.MULTILINE))
        has_paragraphs = len(re.split(r"\n\s*\n", output)) >= 3

        if has_lists or has_paragraphs:
            score += 0.3

        return min(score, 1.0)


__all__ = [
    "ConsistencyRewardComponent",
    "CorrectnessRewardComponent",
    "FormatRewardComponent",
    "RewardComponent",
    "StyleRewardComponent",
    "ValidityRewardComponent",
]
