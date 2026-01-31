"""Evaluation pipeline orchestrator for multi-tier evaluation.

Implements a chain-of-responsibility pattern where handlers are executed
in sequence, stopping on the first failure.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample

from pramana.application.evaluation.handlers import EvaluationHandler
from pramana.application.evaluation.results import TierResult


@dataclass
class PipelineResult:
    """Result of running the evaluation pipeline.

    Attributes:
        overall_passed: Whether all tiers passed (True) or pipeline stopped on failure (False)
        tier_results: List of TierResult objects from each handler that was executed
        final_tier: The tier number of the last handler that was executed (0 if no handlers)
        total_duration_ms: Total time taken to run the pipeline in milliseconds
    """

    overall_passed: bool
    tier_results: list[TierResult] = field(default_factory=list)
    final_tier: int = 0
    total_duration_ms: int = 0


class EvaluationPipeline:
    """Orchestrates a chain of evaluation handlers.

    The pipeline executes handlers in order, stopping on the first failure.
    All tier results are collected, and timing is tracked for the entire pipeline.

    Example:
        ```python
        handler1 = Tier1StructuralHandler()
        handler2 = Tier2LLMJudgeHandler()
        pipeline = EvaluationPipeline(handlers=[handler1, handler2])

        result = pipeline.evaluate(example, model_output)
        if result.overall_passed:
            print(f"All {result.final_tier} tiers passed")
        else:
            print(f"Failed at tier {result.final_tier}")
        ```
    """

    def __init__(self, handlers: list[EvaluationHandler]) -> None:
        """Initialize the evaluation pipeline with a list of handlers.

        Args:
            handlers: List of EvaluationHandler instances to execute in order.
                     Handlers will be chained together automatically.
        """
        self._handlers = handlers
        self._chain_handlers()

    def _chain_handlers(self) -> None:
        """Chain handlers together in order.

        Each handler's next_handler is set to the following handler,
        creating a chain-of-responsibility pattern.
        """
        for i in range(len(self._handlers) - 1):
            self._handlers[i]._next = self._handlers[i + 1]

    def evaluate(self, example: "NyayaExample", output: str) -> PipelineResult:
        """Run the evaluation pipeline through all handlers.

        Executes handlers in order, stopping on the first failure.
        Collects all tier results and tracks total duration.

        Args:
            example: The NyayaExample to evaluate against
            output: The model output string to evaluate

        Returns:
            PipelineResult with overall status, tier results, final tier, and timing
        """
        start_time = time.perf_counter()
        tier_results: list[TierResult] = []

        # If no handlers, return empty result
        if not self._handlers:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return PipelineResult(
                overall_passed=True,
                tier_results=[],
                final_tier=0,
                total_duration_ms=duration_ms,
            )

        # Execute handlers in order, stopping on first failure
        for handler in self._handlers:
            tier_result = handler.evaluate(example, output)
            tier_results.append(tier_result)

            # Stop on first failure
            if not tier_result.passed:
                break

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Determine overall status: all tiers must pass
        overall_passed = all(tier.passed for tier in tier_results)

        # Final tier is the last executed tier
        final_tier = tier_results[-1].tier if tier_results else 0

        return PipelineResult(
            overall_passed=overall_passed,
            tier_results=tier_results,
            final_tier=final_tier,
            total_duration_ms=duration_ms,
        )
