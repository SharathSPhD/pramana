"""Tests for HuggingFaceUploader."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest

from pramana.infrastructure.storage.hf_uploader import HuggingFaceUploader


class TestHuggingFaceUploader:
    """Test HuggingFaceUploader."""

    def test_initialization_with_token(self) -> None:
        """Test initialization with explicit token."""
        uploader = HuggingFaceUploader(token="test-token-123")
        assert uploader.token == "test-token-123"

    def test_initialization_from_env_var(self) -> None:
        """Test initialization from HF_TOKEN environment variable."""
        with patch.dict(os.environ, {"HF_TOKEN": "env-token-456"}):
            uploader = HuggingFaceUploader()
            assert uploader.token == "env-token-456"

    def test_initialization_no_token(self) -> None:
        """Test initialization without token raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HuggingFace token is required"):
                HuggingFaceUploader()

    def test_initialization_explicit_token_overrides_env(self) -> None:
        """Test explicit token overrides environment variable."""
        with patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
            uploader = HuggingFaceUploader(token="explicit-token")
            assert uploader.token == "explicit-token"

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_model_creates_repo_if_not_exists(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_model creates repo if it doesn't exist."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = False
        mock_create_repo.return_value = None
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{"test": "config"}')

            repo_url = uploader.upload_model(
                model_path=model_path,
                repo_id="test-org/test-model",
                private=True,
            )

            mock_api_instance.repo_exists.assert_called_once_with(
                repo_id="test-org/test-model",
                repo_type="model",
                token="test-token",
            )
            mock_create_repo.assert_called_once_with(
                repo_id="test-org/test-model",
                repo_type="model",
                private=True,
                token="test-token",
                exist_ok=False,
            )
            assert repo_url == "https://huggingface.co/test-org/test-model"

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_model_skips_repo_creation_if_exists(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_model skips repo creation if it already exists."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = True
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{"test": "config"}')

            repo_url = uploader.upload_model(
                model_path=model_path,
                repo_id="test-org/test-model",
                private=False,
            )

            mock_api_instance.repo_exists.assert_called_once()
            mock_create_repo.assert_not_called()
            assert repo_url == "https://huggingface.co/test-org/test-model"

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_model_calls_upload_folder(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_model calls upload_folder with correct parameters."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = True
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{"test": "config"}')

            uploader.upload_model(
                model_path=model_path,
                repo_id="test-org/test-model",
                private=True,
            )

            mock_upload_folder.assert_called_once()
            call_kwargs = mock_upload_folder.call_args[1]
            assert call_kwargs["repo_id"] == "test-org/test-model"
            assert call_kwargs["repo_type"] == "model"
            assert call_kwargs["token"] == "test-token"
            assert Path(call_kwargs["folder_path"]) == model_path

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_dataset_creates_repo_if_not_exists(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_dataset creates repo if it doesn't exist."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = False
        mock_create_repo.return_value = None
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "dataset"
            data_path.mkdir()
            (data_path / "data.json").write_text('{"test": "data"}')

            repo_url = uploader.upload_dataset(
                data_path=data_path,
                repo_id="test-org/test-dataset",
                private=True,
            )

            mock_api_instance.repo_exists.assert_called_once_with(
                repo_id="test-org/test-dataset",
                repo_type="dataset",
                token="test-token",
            )
            mock_create_repo.assert_called_once_with(
                repo_id="test-org/test-dataset",
                repo_type="dataset",
                private=True,
                token="test-token",
                exist_ok=False,
            )
            assert repo_url == "https://huggingface.co/datasets/test-org/test-dataset"

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_dataset_skips_repo_creation_if_exists(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_dataset skips repo creation if it already exists."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = True
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "dataset"
            data_path.mkdir()
            (data_path / "data.json").write_text('{"test": "data"}')

            repo_url = uploader.upload_dataset(
                data_path=data_path,
                repo_id="test-org/test-dataset",
                private=False,
            )

            mock_api_instance.repo_exists.assert_called_once()
            mock_create_repo.assert_not_called()
            assert repo_url == "https://huggingface.co/datasets/test-org/test-dataset"

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_dataset_calls_upload_folder(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_dataset calls upload_folder with correct parameters."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = True
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "dataset"
            data_path.mkdir()
            (data_path / "data.json").write_text('{"test": "data"}')

            uploader.upload_dataset(
                data_path=data_path,
                repo_id="test-org/test-dataset",
                private=True,
            )

            mock_upload_folder.assert_called_once()
            call_kwargs = mock_upload_folder.call_args[1]
            assert call_kwargs["repo_id"] == "test-org/test-dataset"
            assert call_kwargs["repo_type"] == "dataset"
            assert call_kwargs["token"] == "test-token"
            assert Path(call_kwargs["folder_path"]) == data_path

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_model_with_readme_generation(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_model generates README.md if not present."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = True
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{"test": "config"}')
            # No README.md present

            uploader.upload_model(
                model_path=model_path,
                repo_id="test-org/test-model",
                private=True,
            )

            # Check that README.md was created
            readme_path = model_path / "README.md"
            assert readme_path.exists()
            readme_content = readme_path.read_text()
            assert "test-org/test-model" in readme_content
            assert "model" in readme_content.lower()

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_model_preserves_existing_readme(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_model preserves existing README.md."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = True
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text('{"test": "config"}')
            existing_readme = "# Custom Model\n\nThis is a custom README."
            (model_path / "README.md").write_text(existing_readme)

            uploader.upload_model(
                model_path=model_path,
                repo_id="test-org/test-model",
                private=True,
            )

            # Check that existing README.md was preserved
            readme_path = model_path / "README.md"
            assert readme_path.exists()
            assert readme_path.read_text() == existing_readme

    @patch("pramana.infrastructure.storage.hf_uploader.HfApi")
    @patch("pramana.infrastructure.storage.hf_uploader.create_repo")
    @patch("pramana.infrastructure.storage.hf_uploader.upload_folder")
    def test_upload_dataset_with_readme_generation(
        self,
        mock_upload_folder: Mock,
        mock_create_repo: Mock,
        mock_hf_api: Mock,
    ) -> None:
        """Test upload_dataset generates README.md if not present."""
        mock_api_instance = Mock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.repo_exists.return_value = True
        mock_upload_folder.return_value = None

        uploader = HuggingFaceUploader(token="test-token")
        with TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "dataset"
            data_path.mkdir()
            (data_path / "data.json").write_text('{"test": "data"}')
            # No README.md present

            uploader.upload_dataset(
                data_path=data_path,
                repo_id="test-org/test-dataset",
                private=True,
            )

            # Check that README.md was created
            readme_path = data_path / "README.md"
            assert readme_path.exists()
            readme_content = readme_path.read_text()
            assert "test-org/test-dataset" in readme_content
            assert "dataset" in readme_content.lower()

    def test_upload_model_path_not_exists(self) -> None:
        """Test upload_model raises error if path doesn't exist."""
        uploader = HuggingFaceUploader(token="test-token")
        non_existent_path = Path("/non/existent/path")

        with pytest.raises(ValueError, match="does not exist"):
            uploader.upload_model(
                model_path=non_existent_path,
                repo_id="test-org/test-model",
            )

    def test_upload_dataset_path_not_exists(self) -> None:
        """Test upload_dataset raises error if path doesn't exist."""
        uploader = HuggingFaceUploader(token="test-token")
        non_existent_path = Path("/non/existent/path")

        with pytest.raises(ValueError, match="does not exist"):
            uploader.upload_dataset(
                data_path=non_existent_path,
                repo_id="test-org/test-dataset",
            )
