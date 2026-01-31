"""Tests for TierResult dataclass."""

from dataclasses import dataclass
from typing import Any

import pytest

from pramana.application.evaluation.results import TierResult


class TestTierResult:
    """Tests for TierResult dataclass."""

    def test_tier_result_creation_with_all_fields(self) -> None:
        """Test that TierResult can be created with all required fields."""
        result = TierResult(
            tier=1,
            passed=True,
            score=0.95,
            details={"validation_errors": 0, "validation_warnings": 0},
            errors=[],
        )

        assert result.tier == 1
        assert result.passed is True
        assert result.score == 0.95
        assert result.details == {"validation_errors": 0, "validation_warnings": 0}
        assert result.errors == []

    def test_tier_result_with_errors(self) -> None:
        """Test TierResult creation with errors."""
        result = TierResult(
            tier=1,
            passed=False,
            score=0.0,
            details={"validation_errors": 2},
            errors=["Missing phase: tarka", "Pramana has no sources"],
        )

        assert result.tier == 1
        assert result.passed is False
        assert result.score == 0.0
        assert len(result.errors) == 2
        assert "Missing phase: tarka" in result.errors

    def test_tier_result_score_range(self) -> None:
        """Test that TierResult score is between 0.0 and 1.0."""
        # Valid scores
        result1 = TierResult(tier=1, passed=True, score=0.0, details={}, errors=[])
        result2 = TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])
        result3 = TierResult(tier=1, passed=True, score=0.5, details={}, errors=[])

        assert 0.0 <= result1.score <= 1.0
        assert 0.0 <= result2.score <= 1.0
        assert 0.0 <= result3.score <= 1.0

    def test_tier_result_invalid_score_raises_error(self) -> None:
        """Test that invalid score raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between"):
            TierResult(tier=1, passed=True, score=1.5, details={}, errors=[])

        with pytest.raises(ValueError, match="Score must be between"):
            TierResult(tier=1, passed=True, score=-0.1, details={}, errors=[])
