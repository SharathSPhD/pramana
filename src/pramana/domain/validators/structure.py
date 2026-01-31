"""NyayaStructureValidator for validating Nyaya example structure.

Validates:
1. 6-phase completeness (all phases present)
2. Pramana type validity (at least one knowledge source)
3. Pancha Avayava completeness (all 5 members in each syllogism)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pramana.domain.models import NyayaExample


@dataclass
class ValidationError:
    """Represents a validation error."""

    phase: str
    message: str


@dataclass
class ValidationWarning:
    """Represents a validation warning."""

    phase: str
    message: str


@dataclass
class ValidationResult:
    """Result of Nyaya structure validation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure is_valid reflects error presence."""
        if len(self.errors) > 0:
            self.is_valid = False


class NyayaStructureValidator:
    """Validates Nyaya example structure and content."""

    def validate(self, example: "NyayaExample") -> ValidationResult:
        """Validate Nyaya example structure.

        Args:
            example: The NyayaExample to validate

        Returns:
            ValidationResult with is_valid, errors, and warnings
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationWarning] = []

        # TDD Cycle 1: Phase Completeness
        self._validate_phase_completeness(example, errors)

        # TDD Cycle 2: Pramana Validation
        self._validate_pramana(example, errors)

        # TDD Cycle 3: Syllogism Validation
        self._validate_syllogisms(example, errors)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_phase_completeness(
        self, example: "NyayaExample", errors: list[ValidationError]
    ) -> None:
        """Validate that all 6 phases are present.

        TDD Cycle 1: Phase Completeness
        """
        # Check all 6 phases exist (Pydantic ensures non-null, but we verify they have content)
        phases_to_check = [
            ("samshaya", example.samshaya),
            ("pramana", example.pramana),
            ("pancha_avayava", example.pancha_avayava),
            ("tarka", example.tarka),
            ("hetvabhasa", example.hetvabhasa),
            ("nirnaya", example.nirnaya),
        ]

        for phase_name, phase_value in phases_to_check:
            # Defensive check: Even though Pydantic enforces non-null fields,
            # we verify phase values are not None as a safety measure
            if phase_value is None:
                errors.append(
                    ValidationError(
                        phase=phase_name,
                        message=f"Phase {phase_name} is missing",
                    )
                )

        # Defensive check: Even though Pydantic enforces non-empty lists,
        # we verify pancha_avayava has at least one syllogism as a safety measure
        if len(example.pancha_avayava) == 0:
            errors.append(
                ValidationError(
                    phase="pancha_avayava",
                    message="Pancha Avayava must contain at least one syllogism",
                )
            )

    def _validate_pramana(
        self, example: "NyayaExample", errors: list[ValidationError]
    ) -> None:
        """Validate Pramana has at least one knowledge source.

        TDD Cycle 2: Pramana Validation
        """
        pramana = example.pramana
        total_sources = (
            len(pramana.pratyaksha)
            + len(pramana.anumana)
            + len(pramana.upamana)
            + len(pramana.shabda)
        )

        if total_sources == 0:
            errors.append(
                ValidationError(
                    phase="pramana",
                    message="Pramana must have at least one knowledge source (pratyaksha, anumana, upamana, or shabda)",
                )
            )

    def _validate_syllogisms(
        self, example: "NyayaExample", errors: list[ValidationError]
    ) -> None:
        """Validate each syllogism has all 5 members.

        TDD Cycle 3: Syllogism Validation

        Checks that each PanchaAvayava has all 5 required members:
        - Pratijna (Thesis)
        - Hetu (Reason)
        - Udaharana (Universal Example)
        - Upanaya (Application)
        - Nigamana (Conclusion)
        """
        required_members = ["pratijna", "hetu", "udaharana", "upanaya", "nigamana"]

        for idx, syllogism in enumerate(example.pancha_avayava):
            missing_members = []
            for member in required_members:
                value = getattr(syllogism, member, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_members.append(member)

            if missing_members:
                errors.append(
                    ValidationError(
                        phase="pancha_avayava",
                        message=f"Syllogism {idx + 1} is missing required members: {', '.join(missing_members)}",
                    )
                )
