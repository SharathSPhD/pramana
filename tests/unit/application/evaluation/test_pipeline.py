"""Unit tests for EvaluationPipeline orchestrator following TDD methodology.

TDD Cycle 1: PipelineResult dataclass
TDD Cycle 2: EvaluationPipeline basic initialization
TDD Cycle 3: EvaluationPipeline single handler evaluation
TDD Cycle 4: EvaluationPipeline multiple handlers chaining
TDD Cycle 5: EvaluationPipeline stops on failure
TDD Cycle 6: EvaluationPipeline timing tracking
"""

import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample

from pramana.application.evaluation.handlers import EvaluationHandler
from pramana.application.evaluation.results import TierResult


class TestPipelineResult:
    """TDD Cycle 1: PipelineResult dataclass tests."""

    def test_pipeline_result_creation_with_all_fields(self) -> None:
        """RED: Test that PipelineResult can be created with all required fields."""
        from pramana.application.evaluation.pipeline import PipelineResult
        from pramana.application.evaluation.results import TierResult

        tier_result = TierResult(tier=1, passed=True, score=0.95, details={}, errors=[])
        result = PipelineResult(
            overall_passed=True,
            tier_results=[tier_result],
            final_tier=1,
            total_duration_ms=150,
        )

        assert result.overall_passed is True
        assert len(result.tier_results) == 1
        assert result.tier_results[0].tier == 1
        assert result.final_tier == 1
        assert result.total_duration_ms == 150

    def test_pipeline_result_with_multiple_tiers(self) -> None:
        """RED: Test PipelineResult with multiple tier results."""
        from pramana.application.evaluation.pipeline import PipelineResult
        from pramana.application.evaluation.results import TierResult

        tier1 = TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])
        tier2 = TierResult(tier=2, passed=True, score=0.9, details={}, errors=[])
        result = PipelineResult(
            overall_passed=True,
            tier_results=[tier1, tier2],
            final_tier=2,
            total_duration_ms=500,
        )

        assert len(result.tier_results) == 2
        assert result.tier_results[0].tier == 1
        assert result.tier_results[1].tier == 2
        assert result.final_tier == 2

    def test_pipeline_result_failed_evaluation(self) -> None:
        """RED: Test PipelineResult with failed evaluation."""
        from pramana.application.evaluation.pipeline import PipelineResult
        from pramana.application.evaluation.results import TierResult

        tier_result = TierResult(
            tier=1, passed=False, score=0.3, details={}, errors=["Validation failed"]
        )
        result = PipelineResult(
            overall_passed=False,
            tier_results=[tier_result],
            final_tier=1,
            total_duration_ms=50,
        )

        assert result.overall_passed is False
        assert result.tier_results[0].passed is False


class TestEvaluationPipeline:
    """TDD Cycle 2-6: EvaluationPipeline tests."""

    def test_pipeline_initialization_with_handlers(self) -> None:
        """RED: Test that EvaluationPipeline can be initialized with handlers."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler
        from pramana.domain.models import NyayaExample

        # Create a mock handler
        class MockHandler(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        handler = MockHandler()
        pipeline = EvaluationPipeline(handlers=[handler])

        assert pipeline is not None
        assert len(pipeline._handlers) == 1

    def test_pipeline_initialization_with_multiple_handlers(self) -> None:
        """RED: Test EvaluationPipeline with multiple handlers."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler
        from pramana.domain.models import NyayaExample

        class MockHandler1(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        class MockHandler2(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                return TierResult(tier=2, passed=True, score=0.9, details={}, errors=[])

        handlers = [MockHandler1(), MockHandler2()]
        pipeline = EvaluationPipeline(handlers=handlers)

        assert len(pipeline._handlers) == 2

    def test_pipeline_evaluate_single_handler(self, complete_nyaya_example: "NyayaExample") -> None:
        """RED: Test EvaluationPipeline with single handler."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler

        class MockHandler(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        handler = MockHandler()
        pipeline = EvaluationPipeline(handlers=[handler])
        result = pipeline.evaluate(complete_nyaya_example, "test output")

        assert result.overall_passed is True
        assert len(result.tier_results) == 1
        assert result.tier_results[0].tier == 1
        assert result.final_tier == 1
        assert result.total_duration_ms >= 0

    def test_pipeline_evaluate_chains_handlers(
        self, complete_nyaya_example: "NyayaExample"
    ) -> None:
        """RED: Test that EvaluationPipeline chains handlers in order."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler

        call_order = []

        class MockHandler1(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                call_order.append(1)
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        class MockHandler2(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                call_order.append(2)
                return TierResult(tier=2, passed=True, score=0.9, details={}, errors=[])

        handlers = [MockHandler1(), MockHandler2()]
        pipeline = EvaluationPipeline(handlers=handlers)
        result = pipeline.evaluate(complete_nyaya_example, "test output")

        assert call_order == [1, 2]
        assert len(result.tier_results) == 2
        assert result.tier_results[0].tier == 1
        assert result.tier_results[1].tier == 2
        assert result.final_tier == 2
        assert result.overall_passed is True

    def test_pipeline_stops_on_failure(self, complete_nyaya_example: "NyayaExample") -> None:
        """RED: Test that EvaluationPipeline stops when a handler fails."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler

        call_order = []

        class MockHandler1(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                call_order.append(1)
                return TierResult(
                    tier=1, passed=False, score=0.3, details={}, errors=["Failed"]
                )

        class MockHandler2(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                call_order.append(2)  # Should not be called
                return TierResult(tier=2, passed=True, score=1.0, details={}, errors=[])

        handlers = [MockHandler1(), MockHandler2()]
        pipeline = EvaluationPipeline(handlers=handlers)
        result = pipeline.evaluate(complete_nyaya_example, "test output")

        assert call_order == [1]  # Handler2 should not be called
        assert len(result.tier_results) == 1
        assert result.tier_results[0].passed is False
        assert result.final_tier == 1
        assert result.overall_passed is False

    def test_pipeline_tracks_timing(self, complete_nyaya_example: "NyayaExample") -> None:
        """RED: Test that EvaluationPipeline tracks total duration."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler

        class SlowHandler(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                time.sleep(0.1)  # 100ms delay
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        handler = SlowHandler()
        pipeline = EvaluationPipeline(handlers=[handler])
        result = pipeline.evaluate(complete_nyaya_example, "test output")

        assert result.total_duration_ms >= 100  # Should be at least 100ms
        assert result.total_duration_ms < 1000  # But not too long

    def test_pipeline_empty_handlers_list(self, complete_nyaya_example: "NyayaExample") -> None:
        """RED: Test EvaluationPipeline with empty handlers list."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(handlers=[])
        result = pipeline.evaluate(complete_nyaya_example, "test output")

        assert result.overall_passed is True  # No handlers = no failures
        assert len(result.tier_results) == 0
        assert result.final_tier == 0
        assert result.total_duration_ms >= 0

    def test_pipeline_all_tiers_pass(self, complete_nyaya_example: "NyayaExample") -> None:
        """RED: Test EvaluationPipeline when all tiers pass."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler

        class MockHandler1(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        class MockHandler2(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                return TierResult(tier=2, passed=True, score=0.9, details={}, errors=[])

        class MockHandler3(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                return TierResult(tier=3, passed=True, score=0.8, details={}, errors=[])

        handlers = [MockHandler1(), MockHandler2(), MockHandler3()]
        pipeline = EvaluationPipeline(handlers=handlers)
        result = pipeline.evaluate(complete_nyaya_example, "test output")

        assert result.overall_passed is True
        assert len(result.tier_results) == 3
        assert result.final_tier == 3
        assert all(tier.passed for tier in result.tier_results)

    def test_pipeline_middle_tier_fails(self, complete_nyaya_example: "NyayaExample") -> None:
        """RED: Test EvaluationPipeline when middle tier fails."""
        from pramana.application.evaluation.pipeline import EvaluationPipeline
        from pramana.application.evaluation.handlers import EvaluationHandler

        call_order = []

        class MockHandler1(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                call_order.append(1)
                return TierResult(tier=1, passed=True, score=1.0, details={}, errors=[])

        class MockHandler2(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                call_order.append(2)
                return TierResult(
                    tier=2, passed=False, score=0.3, details={}, errors=["Failed"]
                )

        class MockHandler3(EvaluationHandler):
            def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
                call_order.append(3)  # Should not be called
                return TierResult(tier=3, passed=True, score=1.0, details={}, errors=[])

        handlers = [MockHandler1(), MockHandler2(), MockHandler3()]
        pipeline = EvaluationPipeline(handlers=handlers)
        result = pipeline.evaluate(complete_nyaya_example, "test output")

        assert call_order == [1, 2]  # Handler3 should not be called
        assert len(result.tier_results) == 2
        assert result.tier_results[0].passed is True
        assert result.tier_results[1].passed is False
        assert result.final_tier == 2
        assert result.overall_passed is False
