"""Content quality validation for Nyaya phase semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pramana.domain.models import HetvabhasaType, NyayaExample


@dataclass
class ContentQualityResult:
    """Structured content quality scores for a Nyaya example."""

    pratyaksha_score: float
    udaharana_valid: bool
    tarka_meaningful: bool
    hetvabhasa_completeness: float
    overall_score: float


class ContentQualityValidator:
    """Heuristic validators for Nyaya phase content quality."""

    _udaharana_patterns = [
        r"\bwherever\b.+,\s*.+",
        r"\bwhenever\b.+",
        r"\bin all cases where\b.+",
        r"\bfor any\b.+\bif\b.+\bthen\b.+",
    ]

    _negation_markers = [
        "not",
        "no",
        "never",
        "suppose",
        "assume",
        "contrary",
        "opposite",
    ]

    _contradiction_markers = [
        "contradiction",
        "contradicts",
        "impossible",
        "cannot",
        "absurd",
        "violates",
    ]

    def validate(self, example: NyayaExample) -> ContentQualityResult:
        """Validate content quality for each Nyaya phase."""
        pratyaksha_score = self._validate_pratyaksha(example)
        udaharana_valid = self._validate_udaharana(example)
        tarka_meaningful = self._validate_tarka(example)
        hetvabhasa_completeness = self._validate_hetvabhasa(example)

        overall_score = (
            pratyaksha_score
            + (1.0 if udaharana_valid else 0.0)
            + (1.0 if tarka_meaningful else 0.0)
            + hetvabhasa_completeness
        ) / 4.0

        return ContentQualityResult(
            pratyaksha_score=pratyaksha_score,
            udaharana_valid=udaharana_valid,
            tarka_meaningful=tarka_meaningful,
            hetvabhasa_completeness=hetvabhasa_completeness,
            overall_score=overall_score,
        )

    def _validate_pratyaksha(self, example: NyayaExample) -> float:
        """Score how many Pratyaksha claims appear in the problem statement."""
        claims = [c for c in example.pramana.pratyaksha if c.strip()]
        if not claims:
            return 0.0

        problem_norm = self._normalize_text(example.problem)
        grounded = 0
        for claim in claims:
            claim_norm = self._normalize_text(claim)
            if not claim_norm:
                continue
            claim_tokens = claim_norm.split()
            if claim_norm in problem_norm:
                grounded += 1
                continue
            if len(claim_tokens) > 5 and self._token_overlap_ratio(claim_norm, problem_norm) >= 0.6:
                grounded += 1

        return grounded / len(claims)

    def _validate_udaharana(self, example: NyayaExample) -> bool:
        """Check for universal-rule phrasing in any Udaharana."""
        for syllogism in example.pancha_avayava:
            text = syllogism.udaharana.strip()
            if self._matches_any_pattern(text, self._udaharana_patterns):
                return True
        return False

    def _validate_tarka(self, example: NyayaExample) -> bool:
        """Check for counterfactual negation and contradiction cues."""
        tarka_text = " ".join(
            [
                example.tarka.hypothesis,
                example.tarka.consequence,
                example.tarka.analysis,
                example.tarka.resolution or "",
            ]
        )
        hypothesis_text = example.tarka.hypothesis.lower()
        tarka_text_lower = tarka_text.lower()

        has_negation = any(marker in hypothesis_text for marker in self._negation_markers)
        has_contradiction = any(
            marker in tarka_text_lower for marker in self._contradiction_markers
        )

        return has_negation and has_contradiction

    def _validate_hetvabhasa(self, example: NyayaExample) -> float:
        """Score how many fallacy types are explicitly checked."""
        analysis = example.hetvabhasa.analysis.lower()
        fallacies_found = 0
        for fallacy_type in HetvabhasaType:
            if fallacy_type.value in analysis:
                fallacies_found += 1
        return fallacies_found / len(HetvabhasaType)

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _token_overlap_ratio(claim: str, problem: str) -> float:
        claim_tokens = set(claim.split())
        if not claim_tokens:
            return 0.0
        problem_tokens = set(problem.split())
        return len(claim_tokens & problem_tokens) / len(claim_tokens)

    @staticmethod
    def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in patterns)


__all__ = ["ContentQualityResult", "ContentQualityValidator"]
