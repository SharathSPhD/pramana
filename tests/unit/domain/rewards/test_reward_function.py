"""Unit tests for CompositeRewardFunction following TDD methodology.

Tests the composite reward function with 5 components:
- format_reward: Structural adherence (0-1)
- validity_reward: Logical validity (0-1)
- consistency_reward: Internal consistency (0-1)
- correctness_reward: Ground truth matching (0-1)
- style_reward: Appropriate verbosity/clarity (0-1)
"""

import pytest

from pramana.domain.models import NyayaExample
from pramana.domain.rewards import (
    CompositeRewardFunction,
    ConsistencyRewardComponent,
    CorrectnessRewardComponent,
    FormatRewardComponent,
    RewardComponent,
    RewardResult,
    RewardWeights,
    StyleRewardComponent,
    ValidityRewardComponent,
)
from tests.conftest import complete_nyaya_example


class TestRewardComponentProtocol:
    """Test RewardComponent protocol interface."""

    def test_reward_component_interface(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that reward components follow the protocol."""
        # Mock component that implements the protocol
        class MockComponent:
            def calculate(self, example: NyayaExample, output: str) -> float:
                return 0.5

        component: RewardComponent = MockComponent()
        assert component.calculate(complete_nyaya_example, "test output") == 0.5


class TestRewardWeights:
    """Test RewardWeights dataclass."""

    def test_default_weights_sum_to_one(self) -> None:
        """Test that default weights sum to 1.0."""
        weights = RewardWeights()
        assert abs(weights.format + weights.validity + weights.consistency + weights.correctness + weights.style - 1.0) < 1e-6

    def test_custom_weights_sum_to_one(self) -> None:
        """Test that custom weights can be provided."""
        weights = RewardWeights(format=0.3, validity=0.3, consistency=0.2, correctness=0.15, style=0.05)
        assert abs(weights.format + weights.validity + weights.consistency + weights.correctness + weights.style - 1.0) < 1e-6

    def test_invalid_weights_raise_error(self) -> None:
        """Test that weights not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            RewardWeights(format=0.5, validity=0.3, consistency=0.2, correctness=0.2, style=0.1)


class TestCompositeRewardFunction:
    """Test CompositeRewardFunction implementation."""

    def test_composite_reward_calculates_weighted_sum(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that composite reward calculates weighted sum correctly."""
        # Create components
        weights = RewardWeights(format=0.2, validity=0.3, consistency=0.2, correctness=0.2, style=0.1)
        components = {
            "format": FormatRewardComponent(),
            "validity": ValidityRewardComponent(),
            "consistency": ConsistencyRewardComponent(),
            "correctness": CorrectnessRewardComponent(),
            "style": StyleRewardComponent(),
        }

        reward_function = CompositeRewardFunction(weights=weights, components=components)

        # Use a well-formed output that should score well
        output = """
## Samshaya
Doubt type: samana_dharma_upapatti
Justification: Multiple entities could satisfy the condition

## Pramana
- Pratyaksha: There are exactly 3 people
- Anumana: From constraint A → inference B

## Pancha Avayava
Pratijna: Bob has the dog
Hetu: By elimination
Udaharana: Wherever all alternatives except one are eliminated, the remaining must be true
Upanaya: Applied here
Nigamana: Therefore, Bob has the dog

## Tarka
Hypothesis: Suppose Bob does NOT have the dog
Consequence: Then either Alice or Carol has the dog
Analysis: This leads to contradiction
Resolution: Therefore Bob must have the dog

## Hetvabhasa
Fallacies detected: None
Analysis: No fallacies detected

## Nirnaya
Answer: Bob has the dog
Confidence: high
Justification: Through systematic elimination
"""

        result = reward_function.calculate(complete_nyaya_example, output)

        # Verify result structure
        assert isinstance(result, RewardResult)
        assert "format" in result.components
        assert "validity" in result.components
        assert "consistency" in result.components
        assert "correctness" in result.components
        assert "style" in result.components

        # Verify all components return values in [0, 1]
        for component_name, score in result.components.items():
            assert 0.0 <= score <= 1.0, f"Component {component_name} score {score} not in [0, 1]"

        # Verify total reward is normalized to [-1, 1]
        assert -1.0 <= result.total_reward <= 1.0, f"Total reward {result.total_reward} not in [-1, 1]"

        # Verify weighted sum calculation
        expected_weighted_sum = (
            weights.format * result.components["format"]
            + weights.validity * result.components["validity"]
            + weights.consistency * result.components["consistency"]
            + weights.correctness * result.components["correctness"]
            + weights.style * result.components["style"]
        )
        # Total reward should be normalized: (weighted_sum - 0.5) * 2 to map [0, 1] -> [-1, 1]
        expected_total = (expected_weighted_sum - 0.5) * 2
        assert abs(result.total_reward - expected_total) < 1e-6, f"Total reward {result.total_reward} != expected {expected_total}"

    def test_composite_reward_with_perfect_scores(self, complete_nyaya_example: NyayaExample) -> None:
        """Test composite reward with perfect component scores."""
        weights = RewardWeights()
        components = {
            "format": FormatRewardComponent(),
            "validity": ValidityRewardComponent(),
            "consistency": ConsistencyRewardComponent(),
            "correctness": CorrectnessRewardComponent(),
            "style": StyleRewardComponent(),
        }

        reward_function = CompositeRewardFunction(weights=weights, components=components)

        # Perfect output matching ground truth
        perfect_output = """
## Samshaya
Doubt type: samana_dharma_upapatti
Justification: Multiple entities could satisfy the condition

## Pramana
- Pratyaksha: There are exactly 3 people
- Anumana: From constraint A → inference B

## Pancha Avayava
Pratijna: Bob has the dog
Hetu: By elimination
Udaharana: Wherever all alternatives except one are eliminated, the remaining must be true
Upanaya: Applied here
Nigamana: Therefore, Bob has the dog

## Tarka
Hypothesis: Suppose Bob does NOT have the dog
Consequence: Then either Alice or Carol has the dog
Analysis: This leads to contradiction
Resolution: Therefore Bob must have the dog

## Hetvabhasa
Fallacies detected: None
Analysis: No fallacies detected

## Nirnaya
Answer: Bob has the dog
Confidence: high
Justification: Through systematic elimination
"""

        result = reward_function.calculate(complete_nyaya_example, perfect_output)

        # With perfect scores, total reward should be close to 1.0 (after normalization)
        # If all components = 1.0, weighted_sum = 1.0, normalized = (1.0 - 0.5) * 2 = 1.0
        assert result.total_reward > 0.5, "Perfect scores should yield high total reward"

    def test_composite_reward_with_zero_scores(self, complete_nyaya_example: NyayaExample) -> None:
        """Test composite reward with zero component scores."""
        weights = RewardWeights()
        components = {
            "format": FormatRewardComponent(),
            "validity": ValidityRewardComponent(),
            "consistency": ConsistencyRewardComponent(),
            "correctness": CorrectnessRewardComponent(),
            "style": StyleRewardComponent(),
        }

        reward_function = CompositeRewardFunction(weights=weights, components=components)

        # Empty output should score poorly
        empty_output = ""
        result = reward_function.calculate(complete_nyaya_example, empty_output)

        # With zero scores, total reward should be close to -1.0 (after normalization)
        # If all components = 0.0, weighted_sum = 0.0, normalized = (0.0 - 0.5) * 2 = -1.0
        assert result.total_reward < 0.0, "Zero scores should yield negative total reward"

    def test_composite_reward_handles_missing_components(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that composite reward handles missing components gracefully."""
        weights = RewardWeights(format=1.0, validity=0.0, consistency=0.0, correctness=0.0, style=0.0)
        components = {
            "format": FormatRewardComponent(),
        }

        reward_function = CompositeRewardFunction(weights=weights, components=components)

        output = "## Samshaya\nTest content"
        result = reward_function.calculate(complete_nyaya_example, output)

        # Should only calculate format reward
        assert "format" in result.components
        assert result.components["format"] >= 0.0


class TestFormatRewardComponent:
    """Test FormatRewardComponent implementation."""

    def test_format_reward_detects_all_phases(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that format reward detects all 6 phases."""
        component = FormatRewardComponent()

        # Output with all phases
        complete_output = """
## Samshaya
Test

## Pramana
Test

## Pancha Avayava
Test

## Tarka
Test

## Hetvabhasa
Test

## Nirnaya
Test
"""

        score = component.calculate(complete_nyaya_example, complete_output)
        assert 0.0 <= score <= 1.0
        assert score > 0.5, "Complete output should score well"

    def test_format_reward_penalizes_missing_phases(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that format reward penalizes missing phases."""
        component = FormatRewardComponent()

        # Output with only some phases
        incomplete_output = """
## Samshaya
Test

## Pramana
Test
"""

        score = component.calculate(complete_nyaya_example, incomplete_output)
        assert 0.0 <= score <= 1.0
        assert score < 0.5, "Incomplete output should score poorly"


class TestCorrectnessRewardComponent:
    """Test CorrectnessRewardComponent implementation."""

    def test_correctness_reward_matches_ground_truth(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that correctness reward matches ground truth."""
        component = CorrectnessRewardComponent()

        # Output with correct answer
        correct_output = "## Nirnaya\nAnswer: Bob has the dog"
        score = component.calculate(complete_nyaya_example, correct_output)
        assert score == 1.0, "Correct answer should score 1.0"

    def test_correctness_reward_penalizes_wrong_answer(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that correctness reward penalizes wrong answer."""
        component = CorrectnessRewardComponent()

        # Output with wrong answer
        wrong_output = "## Nirnaya\nAnswer: Alice has the cat"
        score = component.calculate(complete_nyaya_example, wrong_output)
        assert score == 0.0, "Wrong answer should score 0.0"

    def test_correctness_reward_handles_missing_answer(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that correctness reward handles missing answer."""
        component = CorrectnessRewardComponent()

        # Output without answer
        no_answer_output = "Some text without answer"
        score = component.calculate(complete_nyaya_example, no_answer_output)
        assert score == 0.0, "Missing answer should score 0.0"
