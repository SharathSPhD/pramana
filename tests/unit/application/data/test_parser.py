"""Tests for MarkdownParser - TDD approach.

Tests define expected behavior before implementation.
"""

from datetime import date

import pytest
import yaml

from pramana.application.data.parser import MarkdownParser, ParseError, ValidationError
from pramana.domain.models.nyaya_example import (
    DoubtType,
    HetvabhasaType,
    NyayaExample,
)


class TestYAMLParsing:
    """Tests for YAML frontmatter extraction."""

    def test_parse_valid_yaml_frontmatter(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract YAML frontmatter correctly."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert result.id == "pramana-test-001"
        assert result.problem_type == "constraint_satisfaction"
        assert result.difficulty == "simple"
        assert result.variables == 3
        assert result.ground_truth == "Bob has the dog"
        assert result.metadata.stage == 0
        assert result.metadata.author == "test"
        assert result.metadata.validated is True

    def test_parse_yaml_with_nested_metadata(self, valid_nyaya_markdown: str) -> None:
        """Parser should handle nested YAML structures."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert result.metadata.created_date == date(2025, 1, 30)
        assert result.metadata.z3_verifiable is True

    def test_parse_invalid_yaml_raises_parse_error(
        self, malformed_yaml_markdown: str
    ) -> None:
        """Invalid YAML should raise ParseError."""
        parser = MarkdownParser()
        with pytest.raises(ParseError, match="YAML"):
            parser.parse(malformed_yaml_markdown)

    def test_parse_missing_yaml_frontmatter(self) -> None:
        """Markdown without YAML frontmatter should raise ParseError."""
        parser = MarkdownParser()
        markdown = "# Problem\n\nSome problem text."
        with pytest.raises(ParseError, match="frontmatter"):
            parser.parse(markdown)

    def test_parse_empty_yaml_frontmatter(self) -> None:
        """Empty YAML frontmatter should raise ParseError."""
        parser = MarkdownParser()
        markdown = "---\n---\n\n# Problem\n\nText"
        with pytest.raises(ParseError, match="required"):
            parser.parse(markdown)


class TestSectionExtraction:
    """Tests for markdown section extraction."""

    def test_extract_problem_section(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract the Problem section."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert "Alice, Bob, and Carol" in result.problem
        assert "Who has the dog?" in result.problem

    def test_extract_samshaya_section(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract Samshaya section and parse doubt type."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert result.samshaya is not None
        assert result.samshaya.doubt_type == DoubtType.SAMANA_DHARMA_UPAPATTI
        assert "Multiple entities" in result.samshaya.justification

    def test_extract_pramana_section(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract Pramana section with all subsections."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert result.pramana is not None
        assert len(result.pramana.pratyaksha) > 0
        assert len(result.pramana.anumana) > 0
        assert "3 people" in result.pramana.pratyaksha[0]
        assert "Alice does not have the cat" in result.pramana.anumana[0]

    def test_extract_pancha_avayava_section(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract multiple syllogisms from Pancha Avayava."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert len(result.pancha_avayava) == 2
        assert "Carol does not have the dog" in result.pancha_avayava[0].pratijna
        assert "Bob has the dog" in result.pancha_avayava[1].pratijna
        assert "Wherever" in result.pancha_avayava[0].udaharana
        assert "Wherever" in result.pancha_avayava[1].udaharana

    def test_extract_tarka_section(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract Tarka section."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert result.tarka is not None
        assert "Suppose Bob does NOT have the dog" in result.tarka.hypothesis
        assert result.tarka.resolution is not None
        assert "ambiguity" in result.tarka.resolution

    def test_extract_hetvabhasa_section(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract Hetvabhasa section and detect fallacies."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert result.hetvabhasa is not None
        assert len(result.hetvabhasa.fallacies_detected) == 0
        assert "Savyabhichara" in result.hetvabhasa.analysis
        assert "Viruddha" in result.hetvabhasa.analysis

    def test_extract_nirnaya_section(self, valid_nyaya_markdown: str) -> None:
        """Parser should extract Nirnaya section with confidence level."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert result.nirnaya is not None
        assert "Bob has the dog" in result.nirnaya.answer
        assert result.nirnaya.confidence == "moderate"
        assert "systematic elimination" in result.nirnaya.justification


class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_missing_samshaya_raises_validation_error(self) -> None:
        """Missing Samshaya section should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.

## Pramana (Sources of Knowledge)

### Pratyaksha
- Observation
"""
        with pytest.raises(ValidationError, match="Samshaya"):
            parser.parse(markdown)

    def test_missing_pramana_raises_validation_error(self) -> None:
        """Missing Pramana section should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test
"""
        with pytest.raises(ValidationError, match="Pramana"):
            parser.parse(markdown)

    def test_missing_pancha_avayava_raises_validation_error(self) -> None:
        """Missing Pancha Avayava section should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test

## Pramana (Sources of Knowledge)

### Pratyaksha
- Observation
"""
        with pytest.raises(ValidationError, match="Pancha Avayava"):
            parser.parse(markdown)

    def test_missing_tarka_raises_validation_error(
        self, incomplete_example_markdown: str
    ) -> None:
        """Missing Tarka section should raise ValidationError."""
        parser = MarkdownParser()
        with pytest.raises(ValidationError, match="Tarka"):
            parser.parse(incomplete_example_markdown)

    def test_missing_hetvabhasa_raises_validation_error(self) -> None:
        """Missing Hetvabhasa section should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test

## Pramana (Sources of Knowledge)

### Pratyaksha
- Observation

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1

**Pratijna (Thesis)**: Test
**Hetu (Reason)**: Test
**Udaharana (Example)**: Wherever X, Y
**Upanaya (Application)**: Test
**Nigamana (Conclusion)**: Test

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Test
**Consequence**: Test
**Analysis**: Test
"""
        with pytest.raises(ValidationError, match="Hetvabhasa"):
            parser.parse(markdown)

    def test_missing_nirnaya_raises_validation_error(self) -> None:
        """Missing Nirnaya section should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test

## Pramana (Sources of Knowledge)

### Pratyaksha
- Observation

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1

**Pratijna (Thesis)**: Test
**Hetu (Reason)**: Test
**Udaharana (Example)**: Wherever X, Y
**Upanaya (Application)**: Test
**Nigamana (Conclusion)**: Test

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Test
**Consequence**: Test
**Analysis**: Test

## Hetvabhasa (Fallacy Check)

No fallacies detected.
"""
        with pytest.raises(ValidationError, match="Nirnaya"):
            parser.parse(markdown)

    def test_malformed_syllogism_raises_validation_error(self) -> None:
        """Syllogism missing required members should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test

## Pramana (Sources of Knowledge)

### Pratyaksha
- Observation

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1

**Pratijna (Thesis)**: Test
**Hetu (Reason)**: Test
**Udaharana (Example)**: 
**Upanaya (Application)**: Test
**Nigamana (Conclusion)**: Test
"""
        with pytest.raises(ValidationError):
            parser.parse(markdown)

    def test_missing_id_raises_validation_error(self) -> None:
        """Missing id field should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.
"""
        with pytest.raises(ValidationError, match="id"):
            parser.parse(markdown)

    def test_missing_problem_type_raises_validation_error(self) -> None:
        """Missing problem_type field should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.
"""
        with pytest.raises(ValidationError, match="problem_type"):
            parser.parse(markdown)

    def test_invalid_doubt_type_raises_validation_error(self) -> None:
        """Invalid doubt type should raise ValidationError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Some problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: InvalidDoubtType
**Justification**: Test
"""
        with pytest.raises(ValidationError, match="Invalid doubt type"):
            parser.parse(markdown)

    def test_invalid_date_format_raises_parse_error(self) -> None:
        """Invalid date format should raise ParseError."""
        parser = MarkdownParser()
        markdown = """---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 3
ground_truth: "Answer"
metadata:
  created_date: "not-a-date"
  author: test
  stage: 0
---

# Problem

Some problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test

## Pramana (Sources of Knowledge)

### Pratyaksha
- Observation

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1

**Pratijna (Thesis)**: Test
**Hetu (Reason)**: Test
**Udaharana (Example)**: Wherever X, Y
**Upanaya (Application)**: Test
**Nigamana (Conclusion)**: Test

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Test
**Consequence**: Test
**Analysis**: Test

## Hetvabhasa (Fallacy Check)

No fallacies detected.

## Nirnaya (Ascertainment)

**Final Answer**: Test
**Confidence**: Moderate
**Justification**: Test
"""
        with pytest.raises(ParseError, match="Invalid date format"):
            parser.parse(markdown)


class TestParserIntegration:
    """Integration tests for complete parsing workflow."""

    def test_parse_complete_valid_example(
        self, valid_nyaya_markdown: str
    ) -> None:
        """Parser should successfully parse a complete valid example."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        assert isinstance(result, NyayaExample)
        assert result.is_complete()
        assert len(result.pancha_avayava) >= 1
        assert result.tarka is not None
        assert result.nirnaya is not None

    def test_parse_returns_nyaya_example_instance(
        self, valid_nyaya_markdown: str
    ) -> None:
        """Parser should return a properly constructed NyayaExample."""
        parser = MarkdownParser()
        result = parser.parse(valid_nyaya_markdown)

        # Verify all required fields are populated
        assert result.id
        assert result.problem
        assert result.problem_type
        assert result.ground_truth
        assert result.samshaya
        assert result.pramana
        assert result.pancha_avayava
        assert result.tarka
        assert result.hetvabhasa
        assert result.nirnaya
        assert result.metadata
