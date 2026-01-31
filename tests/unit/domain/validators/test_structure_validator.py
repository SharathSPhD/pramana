"""Unit tests for NyayaStructureValidator following TDD methodology.

TDD Cycle 1: Phase Completeness
TDD Cycle 2: Pramana Validation  
TDD Cycle 3: Syllogism Validation
"""

from datetime import date

import pytest

from pramana.domain.models import (
    DoubtType,
    ExampleMetadata,
    Hetvabhasa,
    Nirnaya,
    NyayaExample,
    PanchaAvayava,
    Pramana,
    Samshaya,
    Tarka,
)


@pytest.fixture
def complete_nyaya_example() -> NyayaExample:
    """Create a complete NyayaExample with all 6 phases."""
    return NyayaExample(
        id="test-001",
        problem="Alice, Bob, and Carol each have one pet: a cat, a dog, or a fish.",
        problem_type="constraint_satisfaction",
        difficulty="simple",
        variables=3,
        ground_truth="Bob has the dog",
        samshaya=Samshaya(
            doubt_type=DoubtType.SAMANA_DHARMA_UPAPATTI,
            justification="Multiple entities could satisfy the condition",
        ),
        pramana=Pramana(
            pratyaksha=["There are exactly 3 people"],
            anumana=["From constraint A → inference B"],
            upamana=[],
            shabda=[],
        ),
        pancha_avayava=[
            PanchaAvayava(
                pratijna="Bob has the dog",
                hetu="By elimination",
                udaharana="Wherever all alternatives except one are eliminated, the remaining must be true",
                upanaya="Applied here",
                nigamana="Therefore, Bob has the dog",
            )
        ],
        tarka=Tarka(
            hypothesis="Suppose Bob does NOT have the dog",
            consequence="Then either Alice or Carol has the dog",
            analysis="This leads to contradiction",
            resolution="Therefore Bob must have the dog",
        ),
        hetvabhasa=Hetvabhasa(
            fallacies_detected=[],
            analysis="No fallacies detected",
        ),
        nirnaya=Nirnaya(
            answer="Bob has the dog",
            confidence="high",
            justification="Through systematic elimination",
        ),
        metadata=ExampleMetadata(
            created_date=date(2025, 1, 30),
            author="test",
            validated=True,
            z3_verifiable=True,
            stage=0,
        ),
    )


class TestPhaseCompleteness:
    """TDD Cycle 1: Phase Completeness Tests."""

    def test_validates_complete_example_passes(self, complete_nyaya_example: NyayaExample) -> None:
        """RED: Test that a complete example with all 6 phases passes validation."""
        from pramana.domain.validators.structure import NyayaStructureValidator

        validator = NyayaStructureValidator()
        result = validator.validate(complete_nyaya_example)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_phase_fails(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that missing a phase fails validation.
        
        Since Pydantic ensures all fields are present, we test that
        the validator checks all 6 phases exist and pancha_avayava is not empty.
        """
        from pramana.domain.validators.structure import NyayaStructureValidator

        validator = NyayaStructureValidator()
        
        # Complete example should pass
        result = validator.validate(complete_nyaya_example)
        assert result.is_valid is True
        
        # Test empty pancha_avayava fails
        # Use model_construct to bypass Pydantic validation for testing
        example_empty_syllogisms = NyayaExample.model_construct(
            id="test-empty-syllogisms",
            problem="Test problem",
            problem_type="constraint_satisfaction",
            difficulty="simple",
            variables=1,
            ground_truth="Test answer",
            samshaya=complete_nyaya_example.samshaya,
            pramana=complete_nyaya_example.pramana,
            pancha_avayava=[],  # Empty list should fail
            tarka=complete_nyaya_example.tarka,
            hetvabhasa=complete_nyaya_example.hetvabhasa,
            nirnaya=complete_nyaya_example.nirnaya,
            metadata=complete_nyaya_example.metadata,
        )
        result = validator.validate(example_empty_syllogisms)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(
            error.phase == "pancha_avayava" and "at least one syllogism" in error.message
            for error in result.errors
        )


class TestPramanaValidation:
    """TDD Cycle 2: Pramana Validation Tests."""

    def test_pramana_requires_at_least_one_source(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that Pramana must have at least one knowledge source."""
        from pramana.domain.validators.structure import NyayaStructureValidator

        validator = NyayaStructureValidator()
        
        # Valid example has pratyaksha and anumana
        result = validator.validate(complete_nyaya_example)
        assert result.is_valid is True
        
        # Test empty pramana fails validation
        # Use model_construct to bypass Pydantic validation for testing
        empty_pramana = Pramana.model_construct(
            pratyaksha=[],  # All empty - should fail
            anumana=[],
            upamana=[],
            shabda=[],
        )
        example_empty_pramana = NyayaExample.model_construct(
            id="test-empty-pramana",
            problem="Test problem",
            problem_type="constraint_satisfaction",
            difficulty="simple",
            variables=1,
            ground_truth="Test answer",
            samshaya=complete_nyaya_example.samshaya,
            pramana=empty_pramana,
            pancha_avayava=complete_nyaya_example.pancha_avayava,
            tarka=complete_nyaya_example.tarka,
            hetvabhasa=complete_nyaya_example.hetvabhasa,
            nirnaya=complete_nyaya_example.nirnaya,
            metadata=complete_nyaya_example.metadata,
        )
        result = validator.validate(example_empty_pramana)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(
            error.phase == "pramana" and "at least one knowledge source" in error.message
            for error in result.errors
        )


class TestSyllogismValidation:
    """TDD Cycle 3: Syllogism Validation Tests."""

    def test_syllogism_requires_all_five_members(self, complete_nyaya_example: NyayaExample) -> None:
        """Test that each syllogism must have all 5 members."""
        from pramana.domain.validators.structure import NyayaStructureValidator

        validator = NyayaStructureValidator()
        
        # Valid example has complete syllogism with all 5 members
        result = validator.validate(complete_nyaya_example)
        assert result.is_valid is True
        
        # Test incomplete syllogism fails validation
        # Use model_construct to bypass Pydantic validation for testing
        incomplete_syllogism = PanchaAvayava.model_construct(
            pratijna="Bob has the dog",
            hetu="By elimination",
            udaharana="",  # Missing udaharana
            upanaya="Applied here",
            nigamana="",  # Missing nigamana
        )
        example_incomplete_syllogism = NyayaExample.model_construct(
            id="test-incomplete-syllogism",
            problem="Test problem",
            problem_type="constraint_satisfaction",
            difficulty="simple",
            variables=1,
            ground_truth="Test answer",
            samshaya=complete_nyaya_example.samshaya,
            pramana=complete_nyaya_example.pramana,
            pancha_avayava=[incomplete_syllogism],
            tarka=complete_nyaya_example.tarka,
            hetvabhasa=complete_nyaya_example.hetvabhasa,
            nirnaya=complete_nyaya_example.nirnaya,
            metadata=complete_nyaya_example.metadata,
        )
        result = validator.validate(example_incomplete_syllogism)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any(
            error.phase == "pancha_avayava" and "missing required members" in error.message
            for error in result.errors
        )
        # Verify specific missing members are reported
        error_messages = [error.message for error in result.errors]
        assert any("udaharana" in msg for msg in error_messages)
        assert any("nigamana" in msg for msg in error_messages)
