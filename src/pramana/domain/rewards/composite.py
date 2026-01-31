"""Composite reward function for GRPO training.

Combines multiple reward components with configurable weights.
Total reward is normalized to [-1, 1] range for GRPO.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample
    from pramana.domain.rewards.components import RewardComponent


@dataclass
class RewardWeights:
    """Weights for composite reward function components.

    All weights must sum to 1.0.
    """

    format: float = 0.2
    validity: float = 0.3
    consistency: float = 0.2
    correctness: float = 0.2
    style: float = 0.1

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = self.format + self.validity + self.consistency + self.correctness + self.style
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class RewardResult:
    """Result of composite reward calculation."""

    total_reward: float  # Normalized to [-1, 1]
    components: dict[str, float]  # Individual component scores (0-1)
    weights: RewardWeights


class CompositeRewardFunction:
    """Composite reward function combining multiple reward components.

    Calculates weighted sum of component rewards and normalizes to [-1, 1].
    """

    def __init__(
        self,
        weights: RewardWeights,
        components: dict[str, "RewardComponent"],
    ) -> None:
        """Initialize composite reward function.

        Args:
            weights: RewardWeights configuration
            components: Dictionary mapping component names to RewardComponent instances
                        Expected keys: "format", "validity", "consistency", "correctness", "style"

        Raises:
            ValueError: If component names don't match weight names
        """
        self.weights = weights
        self.components = components

        # Validate component names match weight names
        expected_keys = {"format", "validity", "consistency", "correctness", "style"}
        component_keys = set(components.keys())
        if not component_keys.issubset(expected_keys):
            raise ValueError(f"Component keys {component_keys} must be subset of {expected_keys}")

    def calculate(self, example: "NyayaExample", output: str) -> RewardResult:
        """Calculate composite reward.

        Args:
            example: The NyayaExample with ground truth
            output: Generated output string to evaluate

        Returns:
            RewardResult with total reward (normalized to [-1, 1]) and component scores
        """
        component_scores: dict[str, float] = {}

        # Calculate each component score
        for component_name, component in self.components.items():
            score = component.calculate(example, output)
            # Ensure score is in [0, 1]
            score = max(0.0, min(1.0, score))
            component_scores[component_name] = score

        # Fill in missing components with 0.0
        for key in ["format", "validity", "consistency", "correctness", "style"]:
            if key not in component_scores:
                component_scores[key] = 0.0

        # Calculate weighted sum
        weighted_sum = (
            self.weights.format * component_scores["format"]
            + self.weights.validity * component_scores["validity"]
            + self.weights.consistency * component_scores["consistency"]
            + self.weights.correctness * component_scores["correctness"]
            + self.weights.style * component_scores["style"]
        )

        # Normalize to [-1, 1]: map [0, 1] -> [-1, 1]
        # Formula: (weighted_sum - 0.5) * 2
        total_reward = (weighted_sum - 0.5) * 2

        return RewardResult(
            total_reward=total_reward,
            components=component_scores,
            weights=self.weights,
        )


__all__ = [
    "CompositeRewardFunction",
    "RewardResult",
    "RewardWeights",
]
