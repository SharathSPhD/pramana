"""Tier 5: Hetvabhasa (logical fallacy) classification for model failures.

When a model produces an incorrect answer, classify the failure into
one of the five Nyaya Hetvabhasa categories.
"""

import re
from dataclasses import dataclass


@dataclass
class HetvabhasaResult:
    """Classification result for a model failure."""

    classified_type: str  # One of the 5 types or "unclassified"
    confidence: float  # 0.0 to 1.0
    evidence: list[str]  # What led to this classification
    ground_truth_type: str  # What the benchmark says it should be
    matches_ground_truth: bool  # Does classification match?
    used_fallback: bool = False  # True when category is assigned via default prior


class HetvabhasaClassifier:
    """Classify model failures into Hetvabhasa categories."""

    CATEGORIES = ["savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalatita"]

    def classify(
        self,
        problem: dict,
        response: str,
        solution: dict,
        answer_correct: bool,
    ) -> HetvabhasaResult:
        """Classify a model's incorrect reasoning.

        Args:
            problem: Problem dict from problems.json
            response: Model's full text response
            solution: Solution dict from solutions.json
            answer_correct: Whether the model's final answer was correct

        Returns:
            HetvabhasaResult with classification
        """
        gt_type = problem.get("hetvabhasa_type", "unknown")

        # If the model got the answer correct, no failure to classify
        if answer_correct:
            return HetvabhasaResult(
                classified_type="none",
                confidence=1.0,
                evidence=["Answer correct; no failure to classify"],
                ground_truth_type=gt_type,
                matches_ground_truth=True,
                used_fallback=False,
            )

        # Strategy 1: Check if model followed the trap answer
        trap_match = self._check_trap_answer(response, problem)

        # Strategy 2: Pattern-based classification from response text
        pattern_type, pattern_evidence = self._pattern_classify(response, problem)

        # Strategy 3: Use problem's ground truth category as strong prior
        # (since each problem is designed to probe a specific fallacy type)

        # Combine signals
        evidence = []
        classified = "unclassified"
        confidence = 0.0
        used_fallback = False

        if trap_match:
            # Model followed the trap answer -> likely the designed fallacy
            classified = gt_type
            confidence = 0.8
            evidence.append("Model response matches trap answer pattern")

        if pattern_type:
            if classified == "unclassified":
                classified = pattern_type
                confidence = 0.6
            elif pattern_type == classified:
                confidence = min(1.0, confidence + 0.2)
            evidence.extend(pattern_evidence)

        if classified == "unclassified":
            # Default: assign the ground truth category with low confidence
            # (since the problem was designed to probe this specific fallacy)
            classified = gt_type
            confidence = 0.4
            used_fallback = True
            evidence.append(f"Defaulting to problem's designed category: {gt_type}")

        return HetvabhasaResult(
            classified_type=classified,
            confidence=confidence,
            evidence=evidence,
            ground_truth_type=gt_type,
            matches_ground_truth=(classified == gt_type),
            used_fallback=used_fallback,
        )

    def _check_trap_answer(self, response: str, problem: dict) -> bool:
        """Check if the model's response aligns with the trap answer."""
        trap = problem.get("trap_answer", "")
        if not trap:
            return False

        response_lower = response.lower()

        # Extract key phrases from trap answer
        # Simple heuristic: if the trap answer says "Yes" and model says "Yes"
        trap_lower = trap.lower()

        if trap_lower.startswith("yes") and re.search(r"\byes\b", response_lower[:200]):
            return True
        if trap_lower.startswith("no") and re.search(r"\bno\b", response_lower[:200]):
            return True

        # Check for key trap phrases in the response
        # Extract quoted or key terms from trap
        trap_keywords = re.findall(r'"([^"]+)"', trap)
        for kw in trap_keywords:
            if kw.lower() in response_lower:
                return True

        return False

    def _pattern_classify(self, response: str, problem: dict) -> tuple[str | None, list[str]]:
        """Pattern-based classification from response text."""
        resp_lower = response.lower()
        evidence = []
        scores = {cat: 0.0 for cat in self.CATEGORIES}

        # Savyabhichara indicators: model follows majority pattern
        if any(
            p in resp_lower
            for p in ["most", "majority", "pattern", "typically", "usually", "80%", "83%", "75%"]
        ):
            scores["savyabhichara"] += 1.0
            evidence.append("Response references majority/pattern/typically")
        if not any(p in resp_lower for p in ["counterexample", "exception", "however", "but"]):
            scores["savyabhichara"] += 0.5
            evidence.append("Response does not mention counterexamples")

        # Viruddha indicators: incorrect logical direction
        if any(p in resp_lower for p in ["therefore", "implies", "since", "because"]):
            # Check if the logic direction might be wrong
            if problem.get("category") == "viruddha":
                scores["viruddha"] += 0.8
                evidence.append("Response uses deductive language on viruddha-type problem")
        if any(p in resp_lower for p in ["converse", "reverse", "contrapositive"]):
            scores["viruddha"] += 0.5

        # Prakaranasama indicators: model uses irrelevant info
        if problem.get("category") == "prakaranasama":
            # Check if model references the known red-herring elements
            why_fails = problem.get("why_it_fails", "").lower()
            irrelevant_keywords = re.findall(r"\b\w+\b", why_fails)
            for kw in irrelevant_keywords:
                if len(kw) > 4 and kw in resp_lower:
                    scores["prakaranasama"] += 0.3
                    evidence.append(f"Response references potentially irrelevant term: {kw}")

        # Sadhyasama indicators: circular reasoning
        if any(p in resp_lower for p in ["circular", "assumes", "presupposes", "begging"]):
            scores["sadhyasama"] += 1.0
            evidence.append("Response mentions circular/assumption")

        # Kalatita indicators: temporal/contextual confusion
        if any(p in resp_lower for p in ["originally", "was", "before", "previously", "used to"]):
            scores["kalatita"] += 0.5
            evidence.append("Response references past state")
        if not any(p in resp_lower for p in ["updated", "revoked", "amended", "current", "now"]):
            if problem.get("category") == "kalatita":
                scores["kalatita"] += 0.8
                evidence.append("Response does not acknowledge temporal update")

        # Return highest scoring category if above threshold
        best_cat = max(scores, key=scores.get)
        if scores[best_cat] > 0.0:
            return best_cat, evidence
        return None, evidence
