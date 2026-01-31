"""Tests for UnslothAdapter."""

import sys
from unittest.mock import Mock, MagicMock, patch

import pytest

from pramana.config.loader import LoRAConfig
from pramana.infrastructure.ml.unsloth_adapter import UnslothAdapter


class TestUnslothAdapter:
    """Test UnslothAdapter."""

    def test_initialization(self) -> None:
        """Test adapter initialization."""
        adapter = UnslothAdapter()
        # UnslothAdapter doesn't have model/tokenizer attributes initially
        assert not hasattr(adapter, "model")
        assert not hasattr(adapter, "tokenizer")

    def test_load_model_requires_unsloth(self) -> None:
        """Test load_model raises ImportError if unsloth not installed."""
        adapter = UnslothAdapter()

        # Mock the import to raise ImportError only for 'unsloth'
        def mock_import(name, *args, **kwargs):
            if name == "unsloth":
                raise ImportError("No module named 'unsloth'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="Unsloth is not installed"):
                adapter.load_model("test-model", quantization="4bit")

    def test_apply_lora_requires_unsloth(self) -> None:
        """Test apply_lora raises ImportError if unsloth not installed."""
        adapter = UnslothAdapter()
        mock_model = Mock()
        lora_config = LoRAConfig(rank=32)

        # Mock the import to raise ImportError only for 'unsloth'
        def mock_import(name, *args, **kwargs):
            if name == "unsloth":
                raise ImportError("No module named 'unsloth'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="Unsloth is not installed"):
                adapter.apply_lora(mock_model, lora_config)
