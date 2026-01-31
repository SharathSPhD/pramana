"""Unit tests for Tier2LLMJudgeHandler following TDD methodology.

Tests LLM-based evaluation using Nyaya rubric with mock LLM client.
"""

from typing import Protocol
from unittest.mock import Mock

import pytest

from pramana.application.data.parser import MarkdownParser
from pramana.domain.models import NyayaExample


class TestLLMClientProtocol:
    """Test LLMClient protocol definition."""

    def test_llm_client_protocol_has_generate_method(self) -> None:
        """RED: Test that LLMClient protocol defines generate method."""
        from pramana.application.evaluation.llm_judge import LLMClient
        from typing import Protocol

        # Protocol should exist and be a Protocol subclass
        assert issubclass(LLMClient, Protocol)
        # Check that generate method exists in the protocol
        assert hasattr(LLMClient, "generate")
        assert callable(getattr(LLMClient, "generate", None))

    def test_llm_client_can_be_implemented(self) -> None:
        """RED: Test that a class can implement LLMClient protocol."""
        from pramana.application.evaluation.llm_judge import LLMClient

        class MockLLMClient:
            def generate(self, prompt: str, temperature: float = 0.0) -> str:
                return '{"samshaya": 8, "pramana": 7, "pancha_avayava": 9, "tarka": 8, "hetvabhasa": 7, "nirnaya": 9, "overall": 8}'

        # Should be compatible with protocol
        client: LLMClient = MockLLMClient()  # type: ignore
        assert callable(client.generate)


class TestNyayaRubric:
    """Test NyayaRubric dataclass."""

    def test_nyaya_rubric_creation(self) -> None:
        """RED: Test that NyayaRubric can be created with weights."""
        from pramana.application.evaluation.llm_judge import NyayaRubric

        rubric = NyayaRubric(
            samshaya=1 / 7,
            pramana=1 / 7,
            pancha_avayava=1 / 7,
            tarka=1 / 7,
            hetvabhasa=1 / 7,
            nirnaya=1 / 7,
            overall=1 / 7,
        )

        assert rubric.samshaya == pytest.approx(1 / 7)
        assert rubric.pramana == pytest.approx(1 / 7)
        assert rubric.pancha_avayava == pytest.approx(1 / 7)
        assert rubric.tarka == pytest.approx(1 / 7)
        assert rubric.hetvabhasa == pytest.approx(1 / 7)
        assert rubric.nirnaya == pytest.approx(1 / 7)
        assert rubric.overall == pytest.approx(1 / 7)

    def test_nyaya_rubric_default_weights(self) -> None:
        """RED: Test that NyayaRubric has default equal weights."""
        from pramana.application.evaluation.llm_judge import NyayaRubric

        rubric = NyayaRubric()
        expected_weight = 1 / 7

        assert rubric.samshaya == pytest.approx(expected_weight)
        assert rubric.pramana == pytest.approx(expected_weight)
        assert rubric.pancha_avayava == pytest.approx(expected_weight)
        assert rubric.tarka == pytest.approx(expected_weight)
        assert rubric.hetvabhasa == pytest.approx(expected_weight)
        assert rubric.nirnaya == pytest.approx(expected_weight)
        assert rubric.overall == pytest.approx(expected_weight)

    def test_nyaya_rubric_weights_sum_to_one(self) -> None:
        """RED: Test that NyayaRubric weights sum to approximately 1.0."""
        from pramana.application.evaluation.llm_judge import NyayaRubric

        rubric = NyayaRubric()
        total = (
            rubric.samshaya
            + rubric.pramana
            + rubric.pancha_avayava
            + rubric.tarka
            + rubric.hetvabhasa
            + rubric.nirnaya
            + rubric.overall
        )

        assert total == pytest.approx(1.0)


class TestTier2LLMJudgeHandler:
    """Test Tier2LLMJudgeHandler implementation."""

    @pytest.fixture
    def parser(self) -> MarkdownParser:
        """Create MarkdownParser instance."""
        return MarkdownParser()

    @pytest.fixture
    def mock_llm_client(self) -> Mock:
        """Create mock LLM client."""
        client = Mock()
        client.generate = Mock(return_value='{"samshaya": 8, "pramana": 7, "pancha_avayava": 9, "tarka": 8, "hetvabhasa": 7, "nirnaya": 9, "overall": 8}')
        return client

    @pytest.fixture
    def valid_example(self, parser: MarkdownParser, valid_nyaya_markdown: str) -> NyayaExample:
        """Parse valid Nyaya example."""
        return parser.parse(valid_nyaya_markdown)

    def test_tier2_handler_extends_evaluation_handler(
        self, mock_llm_client: Mock
    ) -> None:
        """RED: Test that Tier2LLMJudgeHandler extends EvaluationHandler."""
        from pramana.application.evaluation.handlers import EvaluationHandler
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        assert isinstance(handler, EvaluationHandler)

    def test_tier2_handler_evaluate_returns_tier_result(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that evaluate returns TierResult."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler
        from pramana.application.evaluation.results import TierResult

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        result = handler.evaluate(valid_example, "test output")

        assert isinstance(result, TierResult)
        assert result.tier == 2

    def test_tier2_handler_calls_llm_client(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that handler calls LLM client with prompt."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        handler.evaluate(valid_example, "test output")

        # Verify LLM client was called
        assert mock_llm_client.generate.called
        call_args = mock_llm_client.generate.call_args
        assert call_args is not None
        prompt = call_args[0][0] if call_args[0] else ""
        assert "samshaya" in prompt.lower() or "nyaya" in prompt.lower()

    def test_tier2_handler_parses_json_response(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that handler parses JSON response from LLM."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        # Mock response with specific scores
        mock_llm_client.generate.return_value = (
            '{"samshaya": 8, "pramana": 7, "pancha_avayava": 9, '
            '"tarka": 8, "hetvabhasa": 7, "nirnaya": 9, "overall": 8}'
        )

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        result = handler.evaluate(valid_example, "test output")

        # Check that scores are in details
        assert "phase_scores" in result.details
        phase_scores = result.details["phase_scores"]
        assert phase_scores["samshaya"] == 8
        assert phase_scores["pramana"] == 7
        assert phase_scores["pancha_avayava"] == 9

    def test_tier2_handler_calculates_weighted_score(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that handler calculates weighted score correctly."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        # All scores are 8, so weighted average should be 8/10 = 0.8
        mock_llm_client.generate.return_value = (
            '{"samshaya": 8, "pramana": 8, "pancha_avayava": 8, '
            '"tarka": 8, "hetvabhasa": 8, "nirnaya": 8, "overall": 8}'
        )

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        result = handler.evaluate(valid_example, "test output")

        # Score should be normalized to 0-1 range (8/10 = 0.8)
        assert result.score == pytest.approx(0.8, abs=0.01)

    def test_tier2_handler_handles_llm_failure(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that handler handles LLM client failures gracefully."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        # Simulate LLM failure
        mock_llm_client.generate.side_effect = Exception("LLM API error")

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        result = handler.evaluate(valid_example, "test output")

        # Should return failed result with error
        assert result.tier == 2
        assert result.passed is False
        assert result.score == 0.0
        assert len(result.errors) > 0
        assert "LLM" in result.errors[0] or "error" in result.errors[0].lower()

    def test_tier2_handler_handles_invalid_json(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that handler handles invalid JSON responses."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        # Invalid JSON response
        mock_llm_client.generate.return_value = "This is not JSON"

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        result = handler.evaluate(valid_example, "test output")

        # Should return failed result with error
        assert result.tier == 2
        assert result.passed is False
        assert result.score == 0.0
        assert len(result.errors) > 0
        assert "JSON" in result.errors[0] or "parse" in result.errors[0].lower()

    def test_tier2_handler_handles_missing_phase_scores(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that handler handles missing phase scores in JSON."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        # JSON missing some phase scores
        mock_llm_client.generate.return_value = '{"samshaya": 8, "pramana": 7}'

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        result = handler.evaluate(valid_example, "test output")

        # Should handle gracefully (either fail or use defaults)
        assert result.tier == 2
        # May fail or use default scores for missing phases
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_tier2_handler_passes_to_next_handler(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that handler can pass to next handler in chain."""
        from pramana.application.evaluation.handlers import EvaluationHandler
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler
        from pramana.application.evaluation.results import TierResult

        class MockNextHandler(EvaluationHandler):
            def evaluate(self, example: NyayaExample, output: str) -> TierResult:
                return TierResult(tier=3, passed=True, score=0.9, details={}, errors=[])

        next_handler = MockNextHandler()
        handler = Tier2LLMJudgeHandler(mock_llm_client, next_handler=next_handler)

        # Verify chain setup
        assert handler._next is next_handler

    def test_tier2_handler_score_normalization(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that scores are normalized from 0-10 to 0-1 range."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        # Test with perfect scores (10)
        mock_llm_client.generate.return_value = (
            '{"samshaya": 10, "pramana": 10, "pancha_avayava": 10, '
            '"tarka": 10, "hetvabhasa": 10, "nirnaya": 10, "overall": 10}'
        )

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        result = handler.evaluate(valid_example, "test output")

        assert result.score == pytest.approx(1.0, abs=0.01)

        # Test with zero scores
        mock_llm_client.generate.return_value = (
            '{"samshaya": 0, "pramana": 0, "pancha_avayava": 0, '
            '"tarka": 0, "hetvabhasa": 0, "nirnaya": 0, "overall": 0}'
        )

        result = handler.evaluate(valid_example, "test output")
        assert result.score == pytest.approx(0.0, abs=0.01)

    def test_tier2_handler_prompt_includes_problem_and_output(
        self, mock_llm_client: Mock, valid_example: NyayaExample
    ) -> None:
        """RED: Test that prompt includes problem statement and model output."""
        from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler

        handler = Tier2LLMJudgeHandler(mock_llm_client)
        model_output = "This is the model's reasoning output"
        handler.evaluate(valid_example, model_output)

        # Check prompt contains problem and output
        call_args = mock_llm_client.generate.call_args
        assert call_args is not None
        prompt = call_args[0][0] if call_args[0] else ""
        assert valid_example.problem in prompt or "Alice" in prompt
        assert model_output in prompt
