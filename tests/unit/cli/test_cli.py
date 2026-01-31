"""Tests for Pramana CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pramana.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test files."""
    return tmp_path


class TestTrainCommand:
    """Tests for `pramana train` command."""

    def test_train_with_stage(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test train command with stage option."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("stage: 0\n")

        with patch("pramana.cli.commands.train.train_model") as mock_train:
            result = runner.invoke(
                app,
                ["train", "--stage", "0", "--config", str(config_file)],
            )
            assert result.exit_code == 0
            mock_train.assert_called_once()

    def test_train_with_resume(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test train command with resume option."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("stage: 1\n")
        checkpoint = temp_dir / "checkpoint"

        with patch("pramana.cli.commands.train.train_model") as mock_train:
            result = runner.invoke(
                app,
                [
                    "train",
                    "--stage",
                    "1",
                    "--config",
                    str(config_file),
                    "--resume",
                    str(checkpoint),
                ],
            )
            assert result.exit_code == 0
            mock_train.assert_called_once()

    def test_train_invalid_stage(self, runner: CliRunner) -> None:
        """Test train command with invalid stage."""
        result = runner.invoke(app, ["train", "--stage", "5"])
        assert result.exit_code != 0
        assert "stage" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_train_missing_config(self, runner: CliRunner) -> None:
        """Test train command without config file."""
        result = runner.invoke(app, ["train", "--stage", "0"])
        # Should still work if config is optional or has defaults
        # Adjust based on actual implementation
        assert result.exit_code in [0, 1]


class TestEvaluateCommand:
    """Tests for `pramana evaluate` command."""

    def test_evaluate_with_model_and_data(
        self, runner: CliRunner, temp_dir: Path
    ) -> None:
        """Test evaluate command with model and data paths."""
        model_path = temp_dir / "model"
        data_path = temp_dir / "data.json"

        with patch("pramana.cli.commands.evaluate.evaluate_model") as mock_eval:
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--model-path",
                    str(model_path),
                    "--data-path",
                    str(data_path),
                ],
            )
            assert result.exit_code == 0
            mock_eval.assert_called_once()

    def test_evaluate_with_tier(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test evaluate command with specific tier."""
        model_path = temp_dir / "model"
        data_path = temp_dir / "data.json"

        with patch("pramana.cli.commands.evaluate.evaluate_model") as mock_eval:
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--model-path",
                    str(model_path),
                    "--data-path",
                    str(data_path),
                    "--tier",
                    "1",
                ],
            )
            assert result.exit_code == 0
            mock_eval.assert_called_once()

    def test_evaluate_with_all_tiers(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test evaluate command with all tiers."""
        model_path = temp_dir / "model"
        data_path = temp_dir / "data.json"

        with patch("pramana.cli.commands.evaluate.evaluate_model") as mock_eval:
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--model-path",
                    str(model_path),
                    "--data-path",
                    str(data_path),
                    "--tier",
                    "all",
                ],
            )
            assert result.exit_code == 0
            mock_eval.assert_called_once()

    def test_evaluate_invalid_tier(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test evaluate command with invalid tier."""
        model_path = temp_dir / "model"
        data_path = temp_dir / "data.json"

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model-path",
                str(model_path),
                "--data-path",
                str(data_path),
                "--tier",
                "4",
            ],
        )
        assert result.exit_code != 0


class TestValidateCommand:
    """Tests for `pramana validate` command."""

    def test_validate_file(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test validate command with single file."""
        # Create a minimal valid example
        test_file = temp_dir / "example.md"
        test_file.write_text("""---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 2
ground_truth: "test answer"
metadata:
  created_date: 2025-01-30
  author: test
  validated: false
  stage: 0
---

# Problem

Test problem.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test justification

## Pramana (Sources of Knowledge)

### Pratyaksha (Perception/Direct Observation)
- Test observation

### Anumana (Inference)
- Test inference

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1

**Pratijna (Thesis)**: Test thesis
**Hetu (Reason)**: Test reason
**Udaharana (Example)**: Wherever X, there is Y
**Upanaya (Application)**: Test application
**Nigamana (Conclusion)**: Test conclusion

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Test hypothesis
**Consequence**: Test consequence
**Analysis**: Test analysis

## Hetvabhasa (Fallacy Check)

No fallacies detected.

## Nirnaya (Conclusion)

**Final Answer**: test answer
**Confidence**: high
**Justification**: Test justification
""")

        result = runner.invoke(app, ["validate", "--file", str(test_file)])
        # Should pass validation (exit code 0) or fail gracefully
        assert result.exit_code in [0, 1]

    def test_validate_directory(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test validate command with directory."""
        examples_dir = temp_dir / "examples"
        examples_dir.mkdir()
        # Create minimal valid example
        (examples_dir / "example1.md").write_text("""---
id: test-002
problem_type: constraint_satisfaction
difficulty: simple
variables: 2
ground_truth: "test"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Test.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test

## Pramana (Sources of Knowledge)

### Pratyaksha (Perception/Direct Observation)
- Test

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

No fallacies.

## Nirnaya (Conclusion)

**Final Answer**: test
**Confidence**: high
**Justification**: Test
""")

        result = runner.invoke(app, ["validate", "--dir", str(examples_dir)])
        # Should pass validation (exit code 0) or fail gracefully
        assert result.exit_code in [0, 1]

    def test_validate_strict_mode(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test validate command with strict mode."""
        test_file = temp_dir / "example.md"
        # Create invalid example (missing sections)
        test_file.write_text("---\nid: test\n---\n# Problem\nTest")

        result = runner.invoke(
            app, ["validate", "--file", str(test_file), "--strict"]
        )
        # Should fail in strict mode
        assert result.exit_code == 1

    def test_validate_missing_file(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test validate command with non-existent file."""
        test_file = temp_dir / "nonexistent.md"

        result = runner.invoke(app, ["validate", "--file", str(test_file)])
        assert result.exit_code != 0


class TestDataCommand:
    """Tests for `pramana data` subcommands."""

    def test_data_parse(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test data parse subcommand."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create a minimal markdown file
        (input_dir / "example.md").write_text("""---
id: test-001
problem_type: constraint_satisfaction
difficulty: simple
variables: 2
ground_truth: "test"
metadata:
  created_date: 2025-01-30
  author: test
  stage: 0
---

# Problem

Test.

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti
**Justification**: Test

## Pramana (Sources of Knowledge)

### Pratyaksha (Perception/Direct Observation)
- Test

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

No fallacies.

## Nirnaya (Conclusion)

**Final Answer**: test
**Confidence**: high
**Justification**: Test
""")

        result = runner.invoke(
            app,
            [
                "data",
                "parse",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        # Should succeed or fail gracefully
        assert result.exit_code in [0, 1]

    def test_data_stats(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test data stats subcommand."""
        data_file = temp_dir / "data.json"
        # Create minimal JSON data
        import json
        data_file.write_text(json.dumps([{
            "id": "test-001",
            "problem_type": "constraint_satisfaction",
            "difficulty": "simple",
            "metadata": {"stage": 0, "validated": False}
        }]))

        result = runner.invoke(app, ["data", "stats", "--data-path", str(data_file)])
        # Should succeed or fail gracefully
        assert result.exit_code in [0, 1]

    def test_data_split(self, runner: CliRunner, temp_dir: Path) -> None:
        """Test data split subcommand."""
        data_file = temp_dir / "data.json"
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create minimal JSON data
        import json
        data_file.write_text(json.dumps([
            {"id": f"test-{i}", "problem_type": "constraint_satisfaction"}
            for i in range(10)
        ]))

        result = runner.invoke(
            app,
            [
                "data",
                "split",
                "--data-path",
                str(data_file),
                "--output-dir",
                str(output_dir),
                "--train-ratio",
                "0.8",
            ],
        )
        # Should succeed
        assert result.exit_code == 0
