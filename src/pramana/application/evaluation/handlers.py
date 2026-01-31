"""Evaluation handlers for tier-based evaluation pipeline.

Implements chain-of-responsibility pattern for multi-tier evaluation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample

from pramana.application.evaluation.results import TierResult


class EvaluationHandler(ABC):
    """Abstract base class for evaluation handlers.

    Implements chain-of-responsibility pattern where each handler
    can process an evaluation and optionally pass to the next handler.
    """

    def __init__(self, next_handler: "EvaluationHandler | None" = None) -> None:
        """Initialize handler with optional next handler in chain.

        Args:
            next_handler: Next handler in the chain, or None if this is the last handler
        """
        self._next = next_handler

    @abstractmethod
    def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
        """Evaluate the output against the example.

        Args:
            example: The NyayaExample to evaluate against
            output: The model output string to evaluate

        Returns:
            TierResult with evaluation results
        """
        ...

    def _pass_to_next(
        self, example: "NyayaExample", output: str
    ) -> TierResult | None:
        """Pass evaluation to next handler in chain.

        Args:
            example: The NyayaExample to evaluate against
            output: The model output string to evaluate

        Returns:
            TierResult from next handler, or None if no next handler
        """
        if self._next:
            return self._next.evaluate(example, output)
        return None


class Tier1StructuralHandler(EvaluationHandler):
    """Tier 1 handler: Structural validation using NyayaStructureValidator.

    Validates that the output follows the proper Nyaya structure with all
    6 phases and required components.
    """

    def __init__(self, next_handler: EvaluationHandler | None = None) -> None:
        """Initialize Tier1StructuralHandler.

        Args:
            next_handler: Optional next handler in the evaluation chain
        """
        super().__init__(next_handler)
        from pramana.domain.validators.structure import NyayaStructureValidator

        self._validator = NyayaStructureValidator()

    def evaluate(self, example: "NyayaExample", output: str) -> TierResult:  # noqa: ARG002
        """Evaluate structural validity of the example.

        Uses NyayaStructureValidator to check:
        - All 6 phases are present
        - Pramana has at least one knowledge source
        - Pancha Avayava syllogisms have all 5 members

        Args:
            example: The NyayaExample to validate
            output: The model output string (not used in this tier, but required by interface)

        Returns:
            TierResult with tier=1, pass/fail status, score, and validation details
        """
        # Validate structure
        validation_result = self._validator.validate(example)

        # Calculate score: 1.0 if valid, 0.0 if invalid
        # Could be more nuanced based on number of errors
        score = 1.0 if validation_result.is_valid else 0.0

        # Extract error messages
        errors = [
            f"{error.phase}: {error.message}" for error in validation_result.errors
        ]

        # Build details dictionary
        details = {
            "validation_errors": len(validation_result.errors),
            "validation_warnings": len(validation_result.warnings),
            "is_valid": validation_result.is_valid,
        }

        return TierResult(
            tier=1,
            passed=validation_result.is_valid,
            score=score,
            details=details,
            errors=errors,
        )
