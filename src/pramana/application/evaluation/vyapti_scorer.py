"""Tier 3: Vyapti explicitness scoring for the vyapti benchmark.

Extracts universal rules stated by a model and compares them against ground truth.
"""

import re
from dataclasses import dataclass


@dataclass
class VyaptiScore:
    """Result of vyapti explicitness analysis."""
    stated: bool           # Did the model state a universal rule?
    correct: bool          # Is the stated rule correct wrt ground truth?
    vyapti_text: str       # The extracted universal rule text
    negation_detected: bool  # Did the model identify a vyapti violation?
    details: str           # Explanation of scoring


class VyaptiScorer:
    """Score vyapti explicitness in model responses."""

    # Patterns that indicate a universal rule is being stated
    UNIVERSAL_PATTERNS = [
        r"[Ww]herever\s+.+?,\s+there\s+is\s+.+",
        r"[Aa]ll\s+\w+.*?(?:are|have|is|must|work|report)",
        r"[Ee]very\s+\w+.*?(?:is|has|must|should)",
        r"[Ff]or\s+all\s+\w+",
        r"[Uu]niversal(?:ly)?\s+.*?(?:rule|relation|claim)",
        r"[Ii]f\s+.+?,\s+then\s+.+",  # conditional form
        r"[Ww]henever\s+.+?,\s+.+",
        r"\bvyāpti\b.*?:?\s*[\"'].+?[\"']",  # explicit vyapti mention
    ]

    # Patterns that indicate violation/negation of a universal rule
    NEGATION_PATTERNS = [
        r"[Nn]ot\s+all\s+\w+",
        r"[Tt]here\s+exists?\s+.*?(?:counter|exception|violat)",
        r"[Cc]ounterexample",
        r"[Ff]alsif(?:y|ied|ies)",
        r"[Vv]iolat(?:e[ds]?|ion)",
        r"[Ee]xception",
        r"[Dd]oes\s+not\s+(?:universally|always|necessarily)",
        r"[Cc]annot\s+(?:validly\s+)?conclude",
        r"[Nn]ot\s+(?:universally|invariably)",
        r"[Ss]avyabhich[āa]ra",
        r"[Ee]rratic",
    ]

    def score(self, response: str, problem: dict, solution: dict) -> VyaptiScore:
        """Score a model response for vyapti explicitness.

        Args:
            response: The model's full text response
            problem: The problem dict from problems.json
            solution: The solution dict from solutions.json

        Returns:
            VyaptiScore with analysis results
        """
        # Extract any universal rule stated by the model
        vyapti_text = self._extract_universal(response)
        stated = bool(vyapti_text)

        # Check if model identified a violation
        negation_detected = self._check_negation(response)

        # Determine correctness
        gt_vyapti_holds = solution.get("vyapti_holds")

        if problem["type"] == "probe":
            # For probes: correct if model identifies the vyapti violation
            correct = negation_detected and (gt_vyapti_holds is False or gt_vyapti_holds is None)
        else:
            # For controls: correct if model states a valid universal
            correct = stated and (gt_vyapti_holds is True or gt_vyapti_holds is None)

        details = self._build_details(stated, negation_detected, gt_vyapti_holds, problem["type"])

        return VyaptiScore(
            stated=stated,
            correct=correct,
            vyapti_text=vyapti_text,
            negation_detected=negation_detected,
            details=details,
        )

    def _extract_universal(self, text: str) -> str:
        """Extract the first universal rule statement from text."""
        for pattern in self.UNIVERSAL_PATTERNS:
            match = re.search(pattern, text)
            if match:
                # Get the full sentence containing the match
                start = max(0, text.rfind(".", 0, match.start()) + 1)
                end = text.find(".", match.end())
                if end == -1:
                    end = min(len(text), match.end() + 100)
                return text[start:end].strip()
        return ""

    def _check_negation(self, text: str) -> bool:
        """Check if the text identifies a vyapti violation."""
        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _build_details(self, stated: bool, negation: bool, gt_holds: bool | None, prob_type: str) -> str:
        parts = []
        parts.append(f"Universal rule {'stated' if stated else 'not stated'}")
        parts.append(f"Violation {'detected' if negation else 'not detected'}")
        parts.append(f"Ground truth vyapti {'holds' if gt_holds else 'fails' if gt_holds is False else 'N/A'}")
        parts.append(f"Problem type: {prob_type}")
        return "; ".join(parts)
