"""Unit tests for ContentQualityValidator following TDD methodology."""

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
def base_example() -> NyayaExample:
    """Create a NyayaExample with predictable content for quality checks."""
    return NyayaExample(
        id="test-quality-001",
        problem=(
            "Alice, Bob, and Carol each have one pet: a cat, a dog, or a fish.\n"
            "Alice does not have the dog.\n"
            "Bob does not have the cat.\n"
            "Carol does not have the fish."
        ),
        problem_type="constraint_satisfaction",
        difficulty="simple",
        variables=3,
        ground_truth="Bob has the dog",
        samshaya=Samshaya(
            doubt_type=DoubtType.SAMANA_DHARMA_UPAPATTI,
            justification="Multiple assignments could satisfy the constraints.",
        ),
        pramana=Pramana(
            pratyaksha=[
                "Alice does not have the dog",
                "Bob does not have the cat",
            ],
            anumana=["By elimination, one pet remains for Bob."],
            upamana=[],
            shabda=[],
        ),
        pancha_avayava=[
            PanchaAvayava(
                pratijna="Bob has the dog",
                hetu="Alice cannot have the dog, and Bob cannot have the cat.",
                udaharana="Wherever all but one option are eliminated, the remaining option must hold.",
                upanaya="Here, only the dog remains for Bob.",
                nigamana="Therefore, Bob has the dog.",
            )
        ],
        tarka=Tarka(
            hypothesis="Suppose Bob does not have the dog.",
            consequence="Then Alice or Carol must have the dog.",
            analysis="That contradicts the constraints, so the assumption fails.",
            resolution="Bob must have the dog.",
        ),
        hetvabhasa=Hetvabhasa(
            fallacies_detected=[],
            analysis=(
                "Check for savyabhichara: No. "
                "Check for viruddha: No. "
                "Check for asiddha: No. "
                "Check for satpratipaksha: No. "
                "Check for badhita: No."
            ),
        ),
        nirnaya=Nirnaya(
            answer="Bob has the dog",
            confidence="high",
            justification="Only assignment that satisfies all constraints.",
        ),
        metadata=ExampleMetadata(
            created_date=date(2025, 1, 30),
            author="test",
            validated=True,
            z3_verifiable=True,
            stage=0,
        ),
    )


class TestContentQualityValidator:
    """TDD Cycle: Content quality checks across Nyaya phases."""

    def test_pratyaksha_score_is_full_when_grounded(self, base_example: NyayaExample) -> None:
        """RED: Pratyaksha score should be 1.0 when claims match problem text."""
        from pramana.application.evaluation.content_quality import ContentQualityValidator

        validator = ContentQualityValidator()
        result = validator.validate(base_example)

        assert result.pratyaksha_score == pytest.approx(1.0)

    def test_pratyaksha_score_penalizes_unobserved_claims(
        self, base_example: NyayaExample
    ) -> None:
        """RED: Pratyaksha score should drop when claims are not in problem text."""
        from pramana.application.evaluation.content_quality import ContentQualityValidator

        base_example.pramana.pratyaksha.append("Carol has the cat")

        validator = ContentQualityValidator()
        result = validator.validate(base_example)

        assert result.pratyaksha_score == pytest.approx(2 / 3)

    def test_udaharana_requires_universal_rule(self, base_example: NyayaExample) -> None:
        """RED: Udaharana should require a universal rule pattern."""
        from pramana.application.evaluation.content_quality import ContentQualityValidator

        validator = ContentQualityValidator()
        valid_result = validator.validate(base_example)
        assert valid_result.udaharana_valid is True

        base_example.pancha_avayava[0].udaharana = "Example: smoke with fire."
        invalid_result = validator.validate(base_example)
        assert invalid_result.udaharana_valid is False

    def test_tarka_requires_counterfactual_and_contradiction(
        self, base_example: NyayaExample
    ) -> None:
        """RED: Tarka should be meaningful only with negation and contradiction cues."""
        from pramana.application.evaluation.content_quality import ContentQualityValidator

        validator = ContentQualityValidator()
        meaningful = validator.validate(base_example)
        assert meaningful.tarka_meaningful is True

        base_example.tarka = Tarka(
            hypothesis="Bob has the dog.",
            consequence="Bob has the dog.",
            analysis="This is consistent.",
            resolution=None,
        )
        meaningless = validator.validate(base_example)
        assert meaningless.tarka_meaningful is False

    def test_hetvabhasa_completeness_scores_all_checks(
        self, base_example: NyayaExample
    ) -> None:
        """RED: Hetvabhasa completeness should reflect how many fallacies are checked."""
        from pramana.application.evaluation.content_quality import ContentQualityValidator

        validator = ContentQualityValidator()
        result = validator.validate(base_example)

        assert result.hetvabhasa_completeness == pytest.approx(1.0)
