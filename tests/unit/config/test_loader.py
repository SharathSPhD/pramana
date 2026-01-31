"""Tests for StageConfigLoader."""

from pathlib import Path

import pytest
import yaml

from pramana.config.loader import StageConfigLoader


class TestStageConfigLoader:
    """Test StageConfigLoader."""

    def test_load_base_config_only(self, tmp_path: Path) -> None:
        """Test loading configuration from base.yaml only."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        base_config = {
            "model": {"name": "meta-llama/Llama-3.2-3B", "revision": "main"},
            "lora": {"rank": 64, "alpha": 128, "target_modules": ["q_proj"]},
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "epochs": 10,
            },
            "data": {
                "train_path": "/data/train.jsonl",
                "max_length": 4096,
            },
            "evaluation": {"tier1_threshold": 0.9, "tier2_threshold": 0.8},
        }

        base_file = config_dir / "base.yaml"
        base_file.write_text(yaml.dump(base_config))

        # Load stage 0 (should use base.yaml)
        loader = StageConfigLoader()
        config = loader.load(0, config_dir)

        assert config.stage == 0
        assert config.model.name == "meta-llama/Llama-3.2-3B"
        assert config.lora.rank == 64
        assert config.training.learning_rate == 2e-5
        assert config.data.train_path == Path("/data/train.jsonl")

    def test_load_with_stage_override(self, tmp_path: Path) -> None:
        """Test loading with stage-specific overrides."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        base_config = {
            "model": {"name": "meta-llama/Llama-3.2-3B", "revision": "main"},
            "lora": {"rank": 64, "alpha": 128},
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "epochs": 10,
            },
            "data": {
                "train_path": "/data/train.jsonl",
                "max_length": 4096,
            },
        }

        stage_config = {
            "stage": 1,
            "lora": {"rank": 128},  # Override rank
            "training": {"learning_rate": 1e-5},  # Override learning rate
        }

        base_file = config_dir / "base.yaml"
        base_file.write_text(yaml.dump(base_config))

        stage_file = config_dir / "stage_1.yaml"
        stage_file.write_text(yaml.dump(stage_config))

        loader = StageConfigLoader()
        config = loader.load(1, config_dir)

        assert config.stage == 1
        assert config.model.name == "meta-llama/Llama-3.2-3B"  # From base
        assert config.lora.rank == 128  # Overridden
        assert config.lora.alpha == 128  # From base
        assert config.training.learning_rate == 1e-5  # Overridden
        assert config.training.batch_size == 4  # From base

    def test_load_nested_override(self, tmp_path: Path) -> None:
        """Test that nested configs merge correctly."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        base_config = {
            "model": {"name": "meta-llama/Llama-3.2-3B", "revision": "main"},
            "lora": {
                "rank": 64,
                "alpha": 128,
                "target_modules": ["q_proj", "v_proj"],
            },
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "epochs": 10,
                "warmup_steps": 100,
            },
            "data": {
                "train_path": "/data/train.jsonl",
                "max_length": 4096,
            },
        }

        stage_config = {
            "stage": 2,
            "lora": {"target_modules": ["k_proj", "o_proj"]},  # Override list
            "training": {"warmup_steps": 200},  # Override single field
        }

        base_file = config_dir / "base.yaml"
        base_file.write_text(yaml.dump(base_config))

        stage_file = config_dir / "stage_2.yaml"
        stage_file.write_text(yaml.dump(stage_config))

        loader = StageConfigLoader()
        config = loader.load(2, config_dir)

        assert config.stage == 2
        assert config.lora.rank == 64  # From base
        assert config.lora.alpha == 128  # From base
        assert config.lora.target_modules == ["k_proj", "o_proj"]  # Overridden
        assert config.training.warmup_steps == 200  # Overridden
        assert config.training.learning_rate == 2e-5  # From base

    def test_load_missing_base_file(self, tmp_path: Path) -> None:
        """Test that missing base.yaml raises FileNotFoundError."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        loader = StageConfigLoader()
        with pytest.raises(FileNotFoundError, match="base.yaml"):
            loader.load(0, config_dir)

    def test_load_missing_stage_file_uses_base(self, tmp_path: Path) -> None:
        """Test that missing stage file falls back to base.yaml."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        base_config = {
            "model": {"name": "test-model"},
            "lora": {"rank": 32},
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "epochs": 10,
            },
            "data": {"train_path": "/data/train.jsonl"},
        }

        base_file = config_dir / "base.yaml"
        base_file.write_text(yaml.dump(base_config))

        loader = StageConfigLoader()
        config = loader.load(3, config_dir)

        assert config.stage == 3
        assert config.model.name == "test-model"

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises appropriate error."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        base_file = config_dir / "base.yaml"
        base_file.write_text("invalid: yaml: [unclosed")

        loader = StageConfigLoader()
        with pytest.raises((ValueError, yaml.YAMLError)):
            loader.load(0, config_dir)

    def test_load_invalid_config_structure(self, tmp_path: Path) -> None:
        """Test that invalid config structure raises ValueError."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        base_config = {
            "model": {"name": "test"},  # Missing required fields
        }

        base_file = config_dir / "base.yaml"
        base_file.write_text(yaml.dump(base_config))

        loader = StageConfigLoader()
        with pytest.raises(ValueError):
            loader.load(0, config_dir)

    def test_load_all_stages(self, tmp_path: Path) -> None:
        """Test loading configurations for all stages."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        base_config = {
            "model": {"name": "test-model"},
            "lora": {"rank": 32},
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "epochs": 10,
            },
            "data": {"train_path": "/data/train.jsonl"},
        }

        base_file = config_dir / "base.yaml"
        base_file.write_text(yaml.dump(base_config))

        loader = StageConfigLoader()
        for stage in range(5):
            config = loader.load(stage, config_dir)
            assert config.stage == stage
            assert config.model.name == "test-model"
