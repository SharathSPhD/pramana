"""Tier 3 Z3 verification handler for formal logic constraints."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import TYPE_CHECKING

from pramana.application.evaluation.handlers import EvaluationHandler
from pramana.application.evaluation.results import TierResult
from pramana.infrastructure.verification.z3_verifier import Z3Verifier

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample


class Tier3Z3VerifierHandler(EvaluationHandler):
    """Tier 3 handler: Z3-based verification for formal logic problems."""

    def __init__(
        self,
        z3_verifier: Z3Verifier | None = None,
        next_handler: EvaluationHandler | None = None,
    ) -> None:
        super().__init__(next_handler)
        self._verifier = z3_verifier or Z3Verifier()

    def evaluate(self, example: "NyayaExample", output: str) -> TierResult:
        if not example.metadata.z3_verifiable:
            return TierResult(
                tier=3,
                passed=True,
                score=1.0,
                details={"skipped": True, "reason": "example_not_marked_z3_verifiable"},
                errors=[],
            )

        constraints = self._extract_smtlib(output)
        if not constraints:
            return TierResult(
                tier=3,
                passed=False,
                score=0.0,
                details={"error_type": "missing_constraints"},
                errors=["No SMT-LIB constraints found in model output."],
            )

        result = self._verifier.verify(constraints, expected=example.ground_truth)
        passed = result.is_valid and result.is_satisfiable
        score = 1.0 if passed else 0.0

        return TierResult(
            tier=3,
            passed=passed,
            score=score,
            details={"z3_result": asdict(result)},
            errors=[] if passed else [result.error or "Z3 verification failed"],
        )

    @staticmethod
    def _extract_smtlib(output: str) -> str | None:
        """Extract SMT-LIB constraints from model output."""

        fenced = re.search(
            r"```(?:smt2|smtlib|smt)?\s*\n(.*?)```",
            output,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            return fenced.group(1).strip()

        marker = "(set-logic"
        if marker in output:
            return output[output.index(marker) :].strip()

        return None
