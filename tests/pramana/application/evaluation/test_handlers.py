"""Tests for evaluation handlers."""

import pytest

from pramana.application.data.parser import MarkdownParser
from pramana.application.evaluation.handlers import (
    EvaluationHandler,
    Tier1StructuralHandler,
)
from pramana.application.evaluation.results import TierResult
from pramana.domain.models import NyayaExample


class TestEvaluationHandler:
    """Tests for EvaluationHandler abstract base."""

    def test_evaluation_handler_is_abstract(self) -> None:
        """Test that EvaluationHandler cannot be instantiated directly."""
        from abc import ABC

        # Should be abstract
        assert issubclass(EvaluationHandler, ABC)

        # Cannot instantiate directly
        with pytest.raises(TypeError):
            EvaluationHandler()  # type: ignore

    def test_evaluation_handler_has_evaluate_method(self) -> None:
        """Test that EvaluationHandler defines abstract evaluate method."""
        # Check method exists
        assert hasattr(EvaluationHandler, "evaluate")
        assert callable(EvaluationHandler.evaluate)

        # Check signature (cannot instantiate, so check via subclass)
        class ConcreteHandler(EvaluationHandler):
            def evaluate(self, example: NyayaExample, output: str) -> TierResult:
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        handler = ConcreteHandler()
        assert callable(handler.evaluate)

    def test_evaluation_handler_chain_of_responsibility(self) -> None:
        """Test that handlers can be chained via next_handler."""
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
    """Tests for Tier1StructuralHandler."""

    @pytest.fixture
    def parser(self) -> MarkdownParser:
        """Create MarkdownParser instance."""
        return MarkdownParser()

    def test_tier1_handler_with_valid_example(
        self, parser: MarkdownParser, valid_nyaya_markdown: str
    ) -> None:
        """Test Tier1StructuralHandler with valid Nyaya example."""
        # Parse the example
        example = parser.parse(valid_nyaya_markdown)

        # Create handler
        handler = Tier1StructuralHandler()

        # Evaluate
        result = handler.evaluate(example, valid_nyaya_markdown)

        # Assertions
        assert result.tier == 1
        assert result.passed is True
        assert result.score == 1.0
        assert len(result.errors) == 0
        assert "validation_errors" in result.details
        assert result.details["validation_errors"] == 0

    def test_tier1_handler_uses_structure_validator(
        self, parser: MarkdownParser, valid_nyaya_markdown: str
    ) -> None:
        """Test that Tier1StructuralHandler uses NyayaStructureValidator."""
        example = parser.parse(valid_nyaya_markdown)
        handler = Tier1StructuralHandler()

        result = handler.evaluate(example, valid_nyaya_markdown)

        # Should have validation details
        assert "validation_errors" in result.details
        assert "validation_warnings" in result.details

    def test_tier1_handler_passes_to_next_handler(
        self, parser: MarkdownParser, valid_nyaya_markdown: str
    ) -> None:
        """Test that Tier1Handler can pass to next handler in chain."""
        class MockNextHandler(EvaluationHandler):
            def evaluate(self, example: NyayaExample, output: str) -> TierResult:
                return TierResult(tier=2, passed=True, score=0.8, details={}, errors=[])

        example = parser.parse(valid_nyaya_markdown)
        next_handler = MockNextHandler()
        handler = Tier1StructuralHandler(next_handler=next_handler)

        # Tier1 should handle it first, but we can test chain setup
        assert handler._next is next_handler
