"""MarkdownParser for Nyaya examples.

Parses structured markdown files with YAML frontmatter into NyayaExample domain models.
"""

import re
from datetime import date
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from pramana.domain.models.nyaya_example import (
    DoubtType,
    ExampleMetadata,
    Hetvabhasa,
    HetvabhasaType,
    Nirnaya,
    NyayaExample,
    PanchaAvayava,
    Pramana,
    Samshaya,
    Tarka,
)


class ParseError(Exception):
    """Raised when markdown parsing fails (e.g., invalid YAML)."""

    pass


class ValidationError(Exception):
    """Raised when parsed content fails domain validation."""

    pass


class MarkdownParser:
    """Parses markdown files with YAML frontmatter into NyayaExample domain models."""

    def parse(self, markdown_content: str) -> NyayaExample:
        """Parse markdown content into a NyayaExample.

        Args:
            markdown_content: Complete markdown string with YAML frontmatter

        Returns:
            Parsed NyayaExample domain model

        Raises:
            ParseError: If YAML frontmatter is invalid or missing
            ValidationError: If required sections are missing or content is invalid
        """
        # Extract YAML frontmatter
        yaml_data, content = self._extract_frontmatter(markdown_content)

        # Validate required fields early
        if not yaml_data.get("id"):
            raise ValidationError("Required field 'id' is missing from YAML frontmatter")
        if not yaml_data.get("problem_type"):
            raise ValidationError("Required field 'problem_type' is missing from YAML frontmatter")

        # Extract and parse sections
        problem = self._extract_problem(content)
        samshaya = self._extract_samshaya(content)
        pramana = self._extract_pramana(content)
        pancha_avayava = self._extract_pancha_avayava(content)
        tarka = self._extract_tarka(content)
        hetvabhasa = self._extract_hetvabhasa(content)
        nirnaya = self._extract_nirnaya(content)

        # Build metadata
        metadata_dict = yaml_data.get("metadata", {})
        metadata = ExampleMetadata(
            created_date=self._parse_date(metadata_dict.get("created_date", "2025-01-30")),
            author=metadata_dict.get("author", "unknown"),
            validated=metadata_dict.get("validated", False),
            z3_verifiable=metadata_dict.get("z3_verifiable", False),
            stage=metadata_dict.get("stage", 0),
        )

        # Construct and return NyayaExample
        try:
            return NyayaExample(
                id=yaml_data.get("id", ""),
                problem=problem,
                problem_type=yaml_data.get("problem_type", ""),
                difficulty=yaml_data.get("difficulty", "unknown"),
                variables=yaml_data.get("variables", 1),
                ground_truth=yaml_data.get("ground_truth", ""),
                samshaya=samshaya,
                pramana=pramana,
                pancha_avayava=pancha_avayava,
                tarka=tarka,
                hetvabhasa=hetvabhasa,
                nirnaya=nirnaya,
                metadata=metadata,
            )
        except PydanticValidationError as e:
            raise ValidationError(f"Domain validation failed: {e}") from e

    def _extract_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Extract YAML frontmatter from markdown.

        Args:
            content: Markdown content with YAML frontmatter

        Returns:
            Tuple of (yaml_data dict, markdown content without frontmatter)

        Raises:
            ParseError: If YAML is invalid or frontmatter is missing
        """
        # Match YAML frontmatter (between --- markers)
        # Handle both empty and non-empty YAML
        # Use ^--- for closing marker to handle empty content case
        pattern = r"^---\s*\n(.*?)^---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)

        if not match:
            raise ParseError("Missing or invalid YAML frontmatter")

        yaml_str = match.group(1).strip()
        markdown_content = match.group(2)

        try:
            # Empty YAML string results in None
            if not yaml_str:
                raise ParseError("YAML frontmatter is empty - required fields missing")
            yaml_data = yaml.safe_load(yaml_str)
            if yaml_data is None:
                raise ParseError("YAML frontmatter is empty - required fields missing")
            return yaml_data, markdown_content
        except yaml.YAMLError as e:
            raise ParseError(f"Invalid YAML in frontmatter: {e}") from e

    def _extract_problem(self, content: str) -> str:
        """Extract the Problem section."""
        section = self._extract_section(content, r"^#\s+Problem", next_section_pattern=r"^##\s+")
        return section.strip()

    def _extract_samshaya(self, content: str) -> Samshaya:
        """Extract and parse Samshaya section."""
        section_text = self._extract_section(
            content, r"^##\s+Samshaya\s*\(.*?\)", next_section_pattern=r"^##\s+"
        )

        # Extract doubt type
        doubt_type_match = re.search(r"\*\*Doubt Type\*\*:\s*(.+)", section_text, re.IGNORECASE)
        if not doubt_type_match:
            raise ValidationError("Samshaya section missing 'Doubt Type'")

        doubt_type_str = doubt_type_match.group(1).strip()
        doubt_type = self._parse_doubt_type(doubt_type_str)

        # Extract justification
        justification_match = re.search(
            r"\*\*Justification\*\*:\s*(.+?)(?=\*\*|\Z)", section_text, re.DOTALL | re.IGNORECASE
        )
        if not justification_match:
            raise ValidationError("Samshaya section missing 'Justification'")

        justification = justification_match.group(1).strip()

        return Samshaya(doubt_type=doubt_type, justification=justification)

    def _extract_pramana(self, content: str) -> Pramana:
        """Extract and parse Pramana section with subsections."""
        section_text = self._extract_section(
            content, r"^##\s+Pramana\s*\(.*?\)", next_section_pattern=r"^##\s+"
        )

        # Extract each Pramana type subsection
        pratyaksha = self._extract_pramana_subsection(section_text, "Pratyaksha")
        anumana = self._extract_pramana_subsection(section_text, "Anumana")
        upamana = self._extract_pramana_subsection(section_text, "Upamana")
        shabda = self._extract_pramana_subsection(section_text, "Shabda")

        return Pramana(
            pratyaksha=pratyaksha, anumana=anumana, upamana=upamana, shabda=shabda
        )

    def _extract_pramana_subsection(self, content: str, subsection_name: str) -> list[str]:
        """Extract items from a Pramana subsection (e.g., Pratyaksha)."""
        # Match subsection header and content until next ### or ##
        pattern = rf"###\s+{re.escape(subsection_name)}.*?\n(.*?)(?=###\s+|##\s+|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if not match:
            return []

        subsection_text = match.group(1).strip()
        # Extract list items (lines starting with -)
        items = []
        for line in subsection_text.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                items.append(line[1:].strip())
            elif line and not line.startswith("#"):
                # Handle non-list content
                items.append(line)

        return items

    def _extract_pancha_avayava(self, content: str) -> list[PanchaAvayava]:
        """Extract and parse all syllogisms from Pancha Avayava section."""
        section_text = self._extract_section(
            content, r"^##\s+Pancha Avayava\s*\(.*?\)", next_section_pattern=r"^##\s+"
        )

        # Find all syllogism subsections
        syllogism_pattern = r"###\s+Syllogism\s+\d+.*?\n(.*?)(?=###\s+Syllogism\s+\d+|##\s+|\Z)"
        syllogism_matches = re.finditer(syllogism_pattern, section_text, re.DOTALL)

        syllogisms = []
        for match in syllogism_matches:
            syllogism_text = match.group(1)

            # Extract each member
            pratijna = self._extract_syllogism_member(syllogism_text, "Pratijna")
            hetu = self._extract_syllogism_member(syllogism_text, "Hetu")
            udaharana = self._extract_syllogism_member(syllogism_text, "Udaharana")
            upanaya = self._extract_syllogism_member(syllogism_text, "Upanaya")
            nigamana = self._extract_syllogism_member(syllogism_text, "Nigamana")

            syllogisms.append(
                PanchaAvayava(
                    pratijna=pratijna,
                    hetu=hetu,
                    udaharana=udaharana,
                    upanaya=upanaya,
                    nigamana=nigamana,
                )
            )

        if not syllogisms:
            raise ValidationError("Pancha Avayava section must contain at least one syllogism")

        return syllogisms

    def _extract_syllogism_member(self, content: str, member_name: str) -> str:
        """Extract a syllogism member (Pratijna, Hetu, etc.)."""
        pattern = rf"\*\*{re.escape(member_name)}\s*\(.*?\)\*\*:\s*(.+?)(?=\*\*|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValidationError(f"Syllogism missing required member: {member_name}")
        return match.group(1).strip()

    def _extract_tarka(self, content: str) -> Tarka:
        """Extract and parse Tarka section."""
        section_text = self._extract_section(
            content, r"^##\s+Tarka\s*\(.*?\)", next_section_pattern=r"^##\s+"
        )

        hypothesis = self._extract_field(section_text, "Hypothesis")
        consequence = self._extract_field(section_text, "Consequence")
        analysis = self._extract_field(section_text, "Analysis")
        resolution = self._extract_field(section_text, "Resolution", required=False)

        return Tarka(
            hypothesis=hypothesis,
            consequence=consequence,
            analysis=analysis,
            resolution=resolution,
        )

    def _extract_hetvabhasa(self, content: str) -> Hetvabhasa:
        """Extract and parse Hetvabhasa section."""
        section_text = self._extract_section(
            content, r"^##\s+Hetvabhasa\s*\(.*?\)", next_section_pattern=r"^##\s+"
        )

        # Detect fallacies mentioned in the analysis
        fallacies_detected: list[HetvabhasaType] = []
        for fallacy_type in HetvabhasaType:
            # Check if fallacy is mentioned as detected
            pattern = rf"Check for\s+{re.escape(fallacy_type.value.title())}.*?:\s*(Yes|Detected|Found)"
            if re.search(pattern, section_text, re.IGNORECASE | re.DOTALL):
                fallacies_detected.append(fallacy_type)

        # Extract analysis text (everything in the section)
        analysis = section_text.strip()

        return Hetvabhasa(fallacies_detected=fallacies_detected, analysis=analysis)

    def _extract_nirnaya(self, content: str) -> Nirnaya:
        """Extract and parse Nirnaya section."""
        section_text = self._extract_section(
            content, r"^##\s+Nirnaya\s*\(.*?\)", next_section_pattern=r"^##\s+|\Z"
        )

        answer = self._extract_field(section_text, "Final Answer")
        confidence_str = self._extract_field(section_text, "Confidence", required=False)
        justification = self._extract_field(section_text, "Justification")

        # Parse confidence level
        confidence = self._parse_confidence(confidence_str or "moderate")

        return Nirnaya(answer=answer, confidence=confidence, justification=justification)

    def _extract_section(
        self, content: str, section_pattern: str, next_section_pattern: str | None = None
    ) -> str:
        """Extract a markdown section by header pattern.

        Args:
            content: Markdown content
            section_pattern: Regex pattern for section header
            next_section_pattern: Optional pattern for next section (stops extraction)

        Returns:
            Section content as string

        Raises:
            ValidationError: If section is not found
        """
        if next_section_pattern:
            pattern = rf"{section_pattern}\s*\n(.*?)(?={next_section_pattern}|\Z)"
        else:
            pattern = rf"{section_pattern}\s*\n(.*?)(?=^##\s+|\Z)"

        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if not match:
            section_name = section_pattern.replace(r"^##\s+", "").replace(r"\(", "").replace(
                r"\)", ""
            )
            raise ValidationError(f"Missing required section: {section_name}")

        return match.group(1).strip()

    def _extract_field(
        self, content: str, field_name: str, required: bool = True
    ) -> str | None:
        """Extract a field value from markdown content.

        Args:
            content: Markdown content
            field_name: Field name to extract (e.g., "Hypothesis")
            required: Whether field is required

        Returns:
            Field value or None if not required and not found

        Raises:
            ValidationError: If required field is missing
        """
        pattern = rf"\*\*{re.escape(field_name)}\*\*:\s*(.+?)(?=\*\*|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            if required:
                raise ValidationError(f"Missing required field: {field_name}")
            return None
        return match.group(1).strip()

    def _parse_doubt_type(self, doubt_type_str: str) -> DoubtType:
        """Parse doubt type string to DoubtType enum."""
        doubt_type_str = doubt_type_str.lower().replace(" ", "_")
        # Map common variations
        mapping = {
            "samana_dharma_upapatti": DoubtType.SAMANA_DHARMA_UPAPATTI,
            "vipratipatti": DoubtType.VIPRATIPATTI,
            "anadhyavasaya": DoubtType.ANADHYAVASAYA,
        }
        if doubt_type_str not in mapping:
            raise ValidationError(f"Invalid doubt type: {doubt_type_str}")
        return mapping[doubt_type_str]

    def _parse_confidence(self, confidence_str: str) -> str:
        """Parse confidence string to valid confidence level."""
        confidence_str = confidence_str.lower().strip()
        # Extract confidence level from string like "Moderate - explanation"
        if "high" in confidence_str:
            return "high"
        elif "moderate" in confidence_str or "medium" in confidence_str:
            return "moderate"
        elif "low" in confidence_str:
            return "low"
        elif "insufficient" in confidence_str:
            return "insufficient"
        return "moderate"  # Default

    def _parse_date(self, date_str: str | date) -> date:
        """Parse date string to date object."""
        if isinstance(date_str, date):
            return date_str
        # Handle YAML date format (YYYY-MM-DD)
        if isinstance(date_str, str):
            try:
                return date.fromisoformat(date_str)
            except ValueError as e:
                raise ParseError(f"Invalid date format: {date_str}") from e
        raise ParseError(f"Invalid date format: {date_str}")
