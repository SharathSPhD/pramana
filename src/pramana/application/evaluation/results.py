"""Evaluation result types for tier-based evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TierResult:
    """Result of a tier evaluation.

    Attributes:
        tier: The tier number (1, 2, 3, etc.)
        passed: Whether the evaluation passed
        score: Score from 0.0 to 1.0
        details: Additional details dictionary
        errors: List of error messages
    """

    tier: int
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate score is in valid range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
