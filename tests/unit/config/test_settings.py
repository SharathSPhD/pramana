"""Tests for PramanaSettings configuration."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

from pramana.config.settings import PramanaSettings


class TestPramanaSettings:
    """Test PramanaSettings configuration."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            settings = PramanaSettings()
            assert settings.data_dir == Path("/workspace/pramana/data")
            assert settings.models_dir == Path("/workspace/pramana/models")
            assert settings.wandb_project == "pramana"
            assert settings.hf_token is None
            assert settings.openai_api_key is None

    def test_environment_variable_override(self) -> None:
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "PRAMANA_DATA_DIR": "/custom/data",
                "PRAMANA_MODELS_DIR": "/custom/models",
                "WANDB_PROJECT": "custom-project",
                "HF_TOKEN": "hf_test_token",
                "OPENAI_API_KEY": "sk_test_key",
            },
            clear=True,
        ):
            settings = PramanaSettings()
            assert settings.data_dir == Path("/custom/data")
            assert settings.models_dir == Path("/custom/models")
            assert settings.wandb_project == "custom-project"
            assert settings.hf_token == "hf_test_token"
            assert settings.openai_api_key == "sk_test_key"

    def test_env_file_loading(self, tmp_path: Path) -> None:
        """Test loading configuration from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """PRAMANA_DATA_DIR=/env/data
PRAMANA_MODELS_DIR=/env/models
WANDB_PROJECT=env-project
HF_TOKEN=env_hf_token
OPENAI_API_KEY=env_openai_key
"""
        )

        with patch.dict(os.environ, {}, clear=True):
            # Pydantic Settings loads from .env automatically if _env_file is set
            # We'll test this by creating settings with explicit env_file
            settings = PramanaSettings(_env_file=str(env_file))
            assert settings.data_dir == Path("/env/data")
            assert settings.models_dir == Path("/env/models")
            assert settings.wandb_project == "env-project"
            assert settings.hf_token == "env_hf_token"
            assert settings.openai_api_key == "env_openai_key"

    def test_env_file_with_prefix(self, tmp_path: Path) -> None:
        """Test that PRAMANA_ prefix is respected."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """PRAMANA_DATA_DIR=/prefixed/data
PRAMANA_MODELS_DIR=/prefixed/models
# Non-prefixed should be ignored for PRAMANA_ prefixed fields
DATA_DIR=/should/be/ignored
"""
        )

        with patch.dict(os.environ, {}, clear=True):
            settings = PramanaSettings(_env_file=str(env_file))
            assert settings.data_dir == Path("/prefixed/data")
            assert settings.models_dir == Path("/prefixed/models")

    def test_optional_fields_none(self) -> None:
        """Test that optional fields can be None."""
        with patch.dict(os.environ, {}, clear=True):
            settings = PramanaSettings()
            assert settings.hf_token is None
            assert settings.openai_api_key is None

    def test_path_expansion(self) -> None:
        """Test that Path fields handle tilde expansion."""
        with patch.dict(
            os.environ,
            {
                "PRAMANA_DATA_DIR": "~/pramana/data",
                "PRAMANA_MODELS_DIR": "~/pramana/models",
            },
            clear=True,
        ):
            settings = PramanaSettings()
            # Path should expand ~ to home directory
            assert str(settings.data_dir).startswith(str(Path.home()))
            assert str(settings.models_dir).startswith(str(Path.home()))

    def test_environment_takes_precedence_over_env_file(
        self, tmp_path: Path
    ) -> None:
        """Test that environment variables take precedence over .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("PRAMANA_DATA_DIR=/env/data\nWANDB_PROJECT=env-project\n")

        with patch.dict(
            os.environ,
            {
                "PRAMANA_DATA_DIR": "/env/override",
                "WANDB_PROJECT": "env-override",
            },
            clear=True,
        ):
            settings = PramanaSettings(_env_file=str(env_file))
            # Environment should override .env file
            assert settings.data_dir == Path("/env/override")
            assert settings.wandb_project == "env-override"

    def test_singleton_pattern(self) -> None:
        """Test that settings can be instantiated multiple times."""
        with patch.dict(os.environ, {}, clear=True):
            settings1 = PramanaSettings()
            settings2 = PramanaSettings()
            # Each instantiation should work independently
            assert settings1.data_dir == settings2.data_dir
            assert settings1.models_dir == settings2.models_dir
