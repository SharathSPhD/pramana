"""Tests for Z3Verifier."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pramana.infrastructure.verification.z3_verifier import (
    VerificationResult,
    Z3Verifier,
)


class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_verification_result_creation(self) -> None:
        """Test creating VerificationResult with all fields."""
        result = VerificationResult(
            is_valid=True,
            is_satisfiable=True,
            model={"x": True, "y": False},
            execution_time_ms=42,
            error=None,
        )
        assert result.is_valid is True
        assert result.is_satisfiable is True
        assert result.model == {"x": True, "y": False}
        assert result.execution_time_ms == 42
        assert result.error is None

    def test_verification_result_with_error(self) -> None:
        """Test VerificationResult with error."""
        result = VerificationResult(
            is_valid=False,
            is_satisfiable=False,
            model=None,
            execution_time_ms=10,
            error="Timeout exceeded",
        )
        assert result.is_valid is False
        assert result.is_satisfiable is False
        assert result.model is None
        assert result.error == "Timeout exceeded"

    def test_verification_result_unsatisfiable(self) -> None:
        """Test VerificationResult for unsatisfiable constraints."""
        result = VerificationResult(
            is_valid=True,
            is_satisfiable=False,
            model=None,
            execution_time_ms=5,
            error=None,
        )
        assert result.is_valid is True
        assert result.is_satisfiable is False
        assert result.model is None


class TestZ3Verifier:
    """Test Z3Verifier class."""

    def test_initialization(self) -> None:
        """Test Z3Verifier initialization."""
        verifier = Z3Verifier()
        assert verifier is not None

    def test_verify_satisfiable_constraints(self) -> None:
        """Test verifying satisfiable SMT-LIB constraints."""
        smt_lib = """
        (declare-const x Bool)
        (declare-const y Bool)
        (assert (or x y))
        (check-sat)
        (get-model)
        """
        expected = {"x": True, "y": False}

        with patch("pramana.infrastructure.verification.z3_verifier.z3") as mock_z3:
            # Create mock constants that can be compared
            mock_sat = Mock()
            mock_unsat = Mock()
            
            # Mock Z3 solver
            mock_solver = Mock()
            mock_solver.check.return_value = mock_sat
            
            # Mock model with proper iteration
            mock_decl_x = Mock()
            mock_decl_x.name.return_value = "x"
            mock_decl_y = Mock()
            mock_decl_y.name.return_value = "y"
            
            mock_value_x = Mock()
            
            mock_model = Mock()
            # Mock model iteration - model is iterable and returns declarations
            mock_model.__iter__ = Mock(return_value=iter([mock_decl_x, mock_decl_y]))
            # Mock model[decl] access
            def mock_getitem(decl):
                if decl == mock_decl_x:
                    return mock_value_x  # True value
                elif decl == mock_decl_y:
                    return Mock()  # False value
                return None
            mock_model.__getitem__ = Mock(side_effect=mock_getitem)
            
            mock_solver.model.return_value = mock_model
            
            # Mock parse_smt2_string to return an iterable (AstVector-like)
            mock_ast_vector = [Mock()]  # List of assertions
            mock_z3.parse_smt2_string.return_value = mock_ast_vector
            
            # Mock Z3 constants - use the same objects
            mock_z3.Solver.return_value = mock_solver
            mock_z3.sat = mock_sat
            mock_z3.unsat = mock_unsat
            mock_z3.is_true = Mock(side_effect=lambda v: v == mock_value_x)
            mock_z3.is_false = Mock(side_effect=lambda v: v != mock_value_x and v is not None)

            verifier = Z3Verifier()
            result = verifier.verify(smt_lib, expected)

            assert result.is_valid is True
            assert result.is_satisfiable is True
            assert result.error is None
            assert result.execution_time_ms >= 0

    def test_verify_unsatisfiable_constraints(self) -> None:
        """Test verifying unsatisfiable SMT-LIB constraints."""
        smt_lib = """
        (declare-const x Bool)
        (assert (and x (not x)))
        (check-sat)
        """
        expected = {}

        with patch("pramana.infrastructure.verification.z3_verifier.z3") as mock_z3:
            mock_sat = Mock()
            mock_unsat = Mock()
            mock_solver = Mock()
            mock_solver.check.return_value = mock_unsat
            mock_z3.parse_smt2_string.return_value = [Mock()]  # Mock AstVector
            mock_z3.Solver.return_value = mock_solver
            mock_z3.unsat = mock_unsat
            mock_z3.sat = mock_sat

            verifier = Z3Verifier()
            result = verifier.verify(smt_lib, expected)

            assert result.is_valid is True
            assert result.is_satisfiable is False
            assert result.model is None
            assert result.error is None

    def test_verify_timeout(self) -> None:
        """Test verification with timeout."""
        smt_lib = """
        (declare-const x Int)
        (assert (> x 0))
        (check-sat)
        """
        expected = {"x": 1}

        with patch("pramana.infrastructure.verification.z3_verifier.z3") as mock_z3:
            mock_solver = Mock()
            mock_solver.check.side_effect = Exception("Timeout")
            mock_z3.Solver.return_value = mock_solver

            verifier = Z3Verifier(timeout_seconds=1)
            result = verifier.verify(smt_lib, expected)

            assert result.is_valid is False
            assert result.error is not None
            assert "timeout" in result.error.lower() or "error" in result.error.lower()

    def test_verify_invalid_smt_lib(self) -> None:
        """Test verification with invalid SMT-LIB syntax."""
        smt_lib = "invalid smt-lib syntax ("
        expected = {}

        with patch("pramana.infrastructure.verification.z3_verifier.z3") as mock_z3:
            mock_solver = Mock()
            mock_solver.check.side_effect = Exception("Parse error")
            mock_z3.Solver.return_value = mock_solver

            verifier = Z3Verifier()
            result = verifier.verify(smt_lib, expected)

            assert result.is_valid is False
            assert result.error is not None

    def test_verify_model_mismatch(self) -> None:
        """Test verification when model doesn't match expected."""
        smt_lib = """
        (declare-const x Bool)
        (assert x)
        (check-sat)
        (get-model)
        """
        expected = {"x": False}  # Expected False but constraint requires True

        with patch("pramana.infrastructure.verification.z3_verifier.z3") as mock_z3:
            mock_sat = Mock()
            mock_unsat = Mock()
            mock_solver = Mock()
            mock_solver.check.return_value = mock_sat
            
            mock_decl_x = Mock()
            mock_decl_x.name.return_value = "x"
            mock_value_x = Mock()
            
            mock_model = Mock()
            mock_model.__iter__ = Mock(return_value=iter([mock_decl_x]))
            mock_model.__getitem__ = Mock(return_value=mock_value_x)
            mock_solver.model.return_value = mock_model
            
            mock_z3.parse_smt2_string.return_value = [Mock()]
            mock_z3.Solver.return_value = mock_solver
            mock_z3.sat = mock_sat
            mock_z3.unsat = mock_unsat
            mock_z3.is_true = Mock(return_value=True)
            mock_z3.is_false = Mock(return_value=False)

            verifier = Z3Verifier()
            result = verifier.verify(smt_lib, expected)

            # Should still be satisfiable, but model won't match expected
            assert result.is_satisfiable is True
            # The verifier should extract the model correctly
            assert result.model is not None

    def test_verify_without_z3_installed(self) -> None:
        """Test that verifier handles missing Z3 gracefully."""
        smt_lib = "(declare-const x Bool) (assert x) (check-sat)"
        expected = {"x": True}

        # Simulate Z3 not being installed
        with patch("pramana.infrastructure.verification.z3_verifier.z3", None):
            verifier = Z3Verifier()
            result = verifier.verify(smt_lib, expected)

            assert result.is_valid is False
            assert result.error is not None
            assert "z3" in result.error.lower() or "import" in result.error.lower()

    def test_verify_empty_constraints(self) -> None:
        """Test verification with empty constraints."""
        smt_lib = ""
        expected = {}

        with patch("pramana.infrastructure.verification.z3_verifier.z3") as mock_z3:
            mock_sat = Mock()
            mock_unsat = Mock()
            mock_solver = Mock()
            mock_solver.check.return_value = mock_sat
            # Empty model for empty constraints
            mock_model = Mock()
            mock_model.__iter__ = Mock(return_value=iter([]))
            mock_solver.model.return_value = mock_model
            mock_z3.parse_smt2_string.return_value = []  # Empty AstVector
            mock_z3.Solver.return_value = mock_solver
            mock_z3.sat = mock_sat
            mock_z3.unsat = mock_unsat

            verifier = Z3Verifier()
            result = verifier.verify(smt_lib, expected)

            # Empty constraints should be satisfiable (no constraints = always satisfiable)
            assert result.is_satisfiable is True

    def test_verify_integer_constraints(self) -> None:
        """Test verification with integer constraints."""
        smt_lib = """
        (declare-const x Int)
        (declare-const y Int)
        (assert (= (+ x y) 10))
        (assert (> x 0))
        (assert (> y 0))
        (check-sat)
        (get-model)
        """
        expected = {"x": 5, "y": 5}

        with patch("pramana.infrastructure.verification.z3_verifier.z3") as mock_z3:
            mock_sat = Mock()
            mock_unsat = Mock()
            mock_solver = Mock()
            mock_solver.check.return_value = mock_sat
            
            mock_decl_x = Mock()
            mock_decl_x.name.return_value = "x"
            mock_decl_y = Mock()
            mock_decl_y.name.return_value = "y"
            
            mock_value_x = Mock()
            mock_value_x.as_long.return_value = 5
            mock_value_y = Mock()
            mock_value_y.as_long.return_value = 5
            
            mock_model = Mock()
            mock_model.__iter__ = Mock(return_value=iter([mock_decl_x, mock_decl_y]))
            def mock_getitem(decl):
                if decl == mock_decl_x:
                    return mock_value_x
                elif decl == mock_decl_y:
                    return mock_value_y
                return None
            mock_model.__getitem__ = Mock(side_effect=mock_getitem)
            mock_solver.model.return_value = mock_model
            
            mock_z3.parse_smt2_string.return_value = [Mock()]
            mock_z3.Solver.return_value = mock_solver
            mock_z3.sat = mock_sat
            mock_z3.unsat = mock_unsat
            mock_z3.is_true = Mock(return_value=False)
            mock_z3.is_false = Mock(return_value=False)
            mock_z3.is_int_value = Mock(return_value=True)
            mock_z3.is_rational_value = Mock(return_value=False)
            mock_z3.is_algebraic_value = Mock(return_value=False)
            mock_z3.is_bv_value = Mock(return_value=False)

            verifier = Z3Verifier()
            result = verifier.verify(smt_lib, expected)

            assert result.is_satisfiable is True
            assert result.model is not None
