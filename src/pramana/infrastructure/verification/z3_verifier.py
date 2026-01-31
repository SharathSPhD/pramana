"""Z3 verification adapter for SMT-LIB constraint verification."""

import time
from dataclasses import dataclass
from typing import Any

try:
    import z3
except ImportError:
    z3 = None  # type: ignore[assignment, misc]


@dataclass
class VerificationResult:
    """Result of Z3 verification.

    Attributes:
        is_valid: Whether verification completed without errors
        is_satisfiable: Whether constraints are satisfiable
        model: Satisfying assignment if satisfiable, None otherwise
        execution_time_ms: Execution time in milliseconds
        error: Error message if verification failed, None otherwise
    """

    is_valid: bool
    is_satisfiable: bool
    model: dict[str, Any] | None
    execution_time_ms: int
    error: str | None


class Z3Verifier:
    """Z3 solver adapter for verifying SMT-LIB constraints."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        """Initialize Z3 verifier.

        Args:
            timeout_seconds: Maximum time to wait for solver (default: 30)
        """
        self.timeout_seconds = timeout_seconds

    def verify(self, constraints: str, expected: str | dict[str, Any] | None = None) -> VerificationResult:
        """Verify SMT-LIB constraints against expected answer.

        Args:
            constraints: SMT-LIB format constraint string
            expected: Expected model (dict) or expected result (str), optional

        Returns:
            VerificationResult with verification outcome
        """
        start_time = time.time()

        # Check if Z3 is available
        if z3 is None:
            return VerificationResult(
                is_valid=False,
                is_satisfiable=False,
                model=None,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error="Z3 solver not installed. Install with: pip install z3-solver",
            )

        try:
            # Parse SMT-LIB string using Z3's parser
            solver = z3.Solver()
            solver.set("timeout", self.timeout_seconds * 1000)  # Z3 timeout in milliseconds

            # Parse constraints - Z3 can parse SMT-LIB format
            try:
                # Use Z3's SMT-LIB parser
                # parse_smt2_string returns an AstVector containing parsed expressions
                parsed = z3.parse_smt2_string(constraints)
                # Add all parsed assertions to solver
                for assertion in parsed:
                    solver.add(assertion)
            except Exception as parse_error:
                return VerificationResult(
                    is_valid=False,
                    is_satisfiable=False,
                    model=None,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    error=f"Failed to parse constraints: {parse_error!s}",
                )

            # Check satisfiability
            check_result = solver.check()

            execution_time_ms = int((time.time() - start_time) * 1000)

            if check_result == z3.sat:
                # Extract model
                model = solver.model()
                model_dict = self._extract_model(model)

                return VerificationResult(
                    is_valid=True,
                    is_satisfiable=True,
                    model=model_dict,
                    execution_time_ms=execution_time_ms,
                    error=None,
                )
            elif check_result == z3.unsat:
                return VerificationResult(
                    is_valid=True,
                    is_satisfiable=False,
                    model=None,
                    execution_time_ms=execution_time_ms,
                    error=None,
                )
            else:  # z3.unknown
                return VerificationResult(
                    is_valid=False,
                    is_satisfiable=False,
                    model=None,
                    execution_time_ms=execution_time_ms,
                    error="Solver returned unknown (possibly due to timeout)",
                )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            if "timeout" in error_msg.lower() or execution_time_ms >= self.timeout_seconds * 1000:
                error_msg = f"Verification timeout after {self.timeout_seconds}s"

            return VerificationResult(
                is_valid=False,
                is_satisfiable=False,
                model=None,
                execution_time_ms=execution_time_ms,
                error=error_msg,
            )

    def _extract_model(self, model: Any) -> dict[str, Any]:
        """Extract model values from Z3 model.

        Args:
            model: Z3 model object

        Returns:
            Dictionary mapping variable names to their values
        """
        model_dict: dict[str, Any] = {}

        if model is None:
            return model_dict

        # Iterate through model declarations
        for decl in model:
            var_name = decl.name()
            value = model[decl]

            # Convert Z3 values to Python types
            if value is None:
                continue

            # Handle different Z3 types
            if z3.is_true(value):
                model_dict[var_name] = True
            elif z3.is_false(value):
                model_dict[var_name] = False
            elif z3.is_int_value(value):
                model_dict[var_name] = value.as_long()
            elif z3.is_rational_value(value):
                model_dict[var_name] = float(value.as_fraction())
            elif z3.is_algebraic_value(value):
                # For algebraic numbers, use approximation
                model_dict[var_name] = float(value.approx(10).as_fraction())
            elif z3.is_bv_value(value):
                model_dict[var_name] = value.as_long()
            else:
                # Fallback: convert to string representation
                model_dict[var_name] = str(value)

        return model_dict
