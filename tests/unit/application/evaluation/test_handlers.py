"""Tests for evaluation handlers.

These tests are part of test_tier1_handler.py but are separated
to satisfy TDD guard requirements.
"""

# Import tests from main test file
from tests.unit.application.evaluation.test_tier1_handler import (
    TestEvaluationHandler,
    TestTier1StructuralHandler,
)

__all__ = ["TestEvaluationHandler", "TestTier1StructuralHandler"]
