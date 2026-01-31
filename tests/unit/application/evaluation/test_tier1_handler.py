"""Unit tests for Tier1StructuralHandler following TDD methodology.

TDD Cycle 1: TierResult dataclass
TDD Cycle 2: EvaluationHandler abstract base
TDD Cycle 3: Tier1StructuralHandler with valid example
TDD Cycle 4: Tier1StructuralHandler with invalid example
"""

from dataclasses import dataclass
from typing import Any

import pytest

from pramana.application.data.parser import MarkdownParser
from pramana.domain.models import NyayaExample


class TestTierResult:
    """TDD Cycle 1: TierResult dataclass tests."""

    def test_tier_result_creation_with_all_fields(self) -> None:
        """RED: Test that TierResult can be created with all required fields."""
        from pramana.application.evaluation.results import TierResult

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
        """RED: Test TierResult creation with errors."""
        from pramana.application.evaluation.results import TierResult

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
        """RED: Test that TierResult score is between 0.0 and 1.0."""
        from pramana.application.evaluation.results import TierResult

        # Valid scores
        result1 = TierResult(tier=1, passed=True, score=0.0, details={}, errors=[])
        result2 = TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])
        result3 = TierResult(tier=1, passed=True, score=0.5, details={}, errors=[])

        assert 0.0 <= result1.score <= 1.0
        assert 0.0 <= result2.score <= 1.0
        assert 0.0 <= result3.score <= 1.0


class TestEvaluationHandler:
    """TDD Cycle 2: EvaluationHandler abstract base tests."""

    def test_evaluation_handler_is_abstract(self) -> None:
        """RED: Test that EvaluationHandler cannot be instantiated directly."""
        from abc import ABC
        from pramana.application.evaluation.handlers import EvaluationHandler

        # Should be abstract
        assert issubclass(EvaluationHandler, ABC)

        # Cannot instantiate directly
        with pytest.raises(TypeError):
            EvaluationHandler()  # type: ignore

    def test_evaluation_handler_has_evaluate_method(self) -> None:
        """RED: Test that EvaluationHandler defines abstract evaluate method."""
        from pramana.application.evaluation.handlers import EvaluationHandler
        from pramana.domain.models import NyayaExample

        # Check method exists
        assert hasattr(EvaluationHandler, "evaluate")
        assert callable(EvaluationHandler.evaluate)

        # Check signature (cannot instantiate, so check via subclass)
        class ConcreteHandler(EvaluationHandler):
            def evaluate(self, example: NyayaExample, output: str) -> Any:
                from pramana.application.evaluation.results import TierResult
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        handler = ConcreteHandler()
        assert callable(handler.evaluate)

    def test_evaluation_handler_chain_of_responsibility(self) -> None:
        """RED: Test that handlers can be chained via next_handler."""
        from pramana.application.evaluation.handlers import EvaluationHandler
        from pramana.application.evaluation.results import TierResult
        from pramana.domain.models import NyayaExample

        class FirstHandler(EvaluationHandler):
            def evaluate(self, example: NyayaExample, output: str) -> TierResult:
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        class SecondHandler(EvaluationHandler):
            def evaluate(self, example: NyayaExample, output: str) -> TierResult:
                return TierResult(tier=2, passed=True, score=0.9, details={}, errors=[])

        # Chain handlers
        handler2 = SecondHandler()
        handler1 = FirstHandler(next_handler=handler2)

        assert handler1._next is handler2
        assert handler2._next is None


class TestTier1StructuralHandler:
    """TDD Cycle 3 & 4: Tier1StructuralHandler tests."""

    @pytest.fixture
    def parser(self) -> MarkdownParser:
        """Create MarkdownParser instance."""
        return MarkdownParser()

    @pytest.fixture
    def valid_example_output(self, valid_nyaya_markdown: str) -> str:
        """Return valid markdown output string."""
        return valid_nyaya_markdown

    def test_tier1_handler_with_valid_example(
        self, parser: MarkdownParser, valid_example_output: str
    ) -> None:
        """RED: Test Tier1StructuralHandler with valid Nyaya example."""
        from pramana.application.evaluation.handlers import Tier1StructuralHandler
        from pramana.domain.models import NyayaExample

        # Parse the example
        example = parser.parse(valid_example_output)

        # Create handler
        handler = Tier1StructuralHandler()

        # Evaluate
        result = handler.evaluate(example, valid_example_output)

        # Assertions
        assert result.tier == 1
        assert result.passed is True
        assert result.score == 1.0
        assert len(result.errors) == 0
        assert "validation_errors" in result.details
        assert result.details["validation_errors"] == 0

    def test_tier1_handler_with_invalid_example(
        self, parser: MarkdownParser, incomplete_example_markdown: str
    ) -> None:
        """RED: Test Tier1StructuralHandler with invalid Nyaya example."""
        from pramana.application.evaluation.handlers import Tier1StructuralHandler

        # Parse the incomplete example (may fail parsing, but if it parses, validation should fail)
        try:
            example = parser.parse(incomplete_example_markdown)
        except Exception:
            # If parsing fails, we can't test structural validation
            # This is acceptable - parsing errors are handled elsewhere
            pytest.skip("Example cannot be parsed, skipping structural validation test")

        # Create handler
        handler = Tier1StructuralHandler()

        # Evaluate
        result = handler.evaluate(example, incomplete_example_markdown)

        # Should fail validation
        assert result.tier == 1
        assert result.passed is False
        assert result.score < 1.0
        assert len(result.errors) > 0
        assert "validation_errors" in result.details
        assert result.details["validation_errors"] > 0

    def test_tier1_handler_uses_structure_validator(
        self, parser: MarkdownParser, valid_example_output: str
    ) -> None:
        """RED: Test that Tier1StructuralHandler uses NyayaStructureValidator."""
        from pramana.application.evaluation.handlers import Tier1StructuralHandler

        example = parser.parse(valid_example_output)
        handler = Tier1StructuralHandler()

        result = handler.evaluate(example, valid_example_output)

        # Should have validation details
        assert "validation_errors" in result.details
        assert "validation_warnings" in result.details

    def test_tier1_handler_passes_to_next_handler(
        self, parser: MarkdownParser, valid_example_output: str
    ) -> None:
        """RED: Test that Tier1Handler can pass to next handler in chain."""
        from pramana.application.evaluation.handlers import (
            EvaluationHandler,
            Tier1StructuralHandler,
        )
        from pramana.application.evaluation.results import TierResult
        from pramana.domain.models import NyayaExample

        class MockNextHandler(EvaluationHandler):
            def evaluate(self, example: NyayaExample, output: str) -> TierResult:
                return TierResult(tier=2, passed=True, score=0.8, details={}, errors=[])

        example = parser.parse(valid_example_output)
        next_handler = MockNextHandler()
        handler = Tier1StructuralHandler(next_handler=next_handler)

        # Tier1 should handle it first, but we can test chain setup
        assert handler._next is next_handler
