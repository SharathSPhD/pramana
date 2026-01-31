"""Tests for UnslothAdapter."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pramana.config.loader import LoRAConfig
from pramana.infrastructure.ml.unsloth_adapter import UnslothAdapter


class TestUnslothAdapter:
    """Test UnslothAdapter."""

    def test_load_model_success(self) -> None:
        """Test successful model loading."""
        adapter = UnslothAdapter()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock the unsloth module before import
        mock_unsloth_module = MagicMock()
        mock_fast_model = MagicMock()
        mock_fast_model.from_pretrained.return_value = (
            mock_model,
            mock_tokenizer,
        )
        mock_unsloth_module.FastLanguageModel = mock_fast_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth_module}):
            model, tokenizer = adapter.load_model("meta-llama/Llama-3.2-3B", "4bit")

            assert model == mock_model
            assert tokenizer == mock_tokenizer
            mock_fast_model.from_pretrained.assert_called_once_with(
                model_name="meta-llama/Llama-3.2-3B",
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=True,
                load_in_8bit=False,
            )

    def test_load_model_8bit(self) -> None:
        """Test model loading with 8bit quantization."""
        adapter = UnslothAdapter()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock the unsloth module before import
        mock_unsloth_module = MagicMock()
        mock_fast_model = MagicMock()
        mock_fast_model.from_pretrained.return_value = (
            mock_model,
            mock_tokenizer,
        )
        mock_unsloth_module.FastLanguageModel = mock_fast_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth_module}):
            adapter.load_model("test-model", "8bit")

            mock_fast_model.from_pretrained.assert_called_once_with(
                model_name="test-model",
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=False,
                load_in_8bit=True,
            )

    def test_load_model_import_error(self) -> None:
        """Test ImportError when Unsloth is not installed."""
        adapter = UnslothAdapter()

        # Mock the import to raise ImportError only for 'unsloth'
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "unsloth":
                raise ImportError("No module named 'unsloth'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="Unsloth is not installed"):
                adapter.load_model("test-model")

    def test_load_model_runtime_error(self) -> None:
        """Test RuntimeError when model loading fails."""
        adapter = UnslothAdapter()

        # Mock the unsloth module before import
        mock_unsloth_module = MagicMock()
        mock_fast_model = MagicMock()
        mock_fast_model.from_pretrained.side_effect = RuntimeError(
            "CUDA out of memory"
        )
        mock_unsloth_module.FastLanguageModel = mock_fast_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth_module}):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                adapter.load_model("test-model")

    def test_apply_lora_success(self) -> None:
        """Test successful LoRA application."""
        adapter = UnslothAdapter()
        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        config = LoRAConfig(rank=64, alpha=128, target_modules=["q_proj", "v_proj"])

        # Mock the unsloth module before import
        mock_unsloth_module = MagicMock()
        mock_fast_model = MagicMock()
        mock_fast_model.get_peft_model.return_value = mock_peft_model
        mock_unsloth_module.FastLanguageModel = mock_fast_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth_module}):
            result = adapter.apply_lora(mock_model, config)

            assert result == mock_peft_model
            mock_fast_model.get_peft_model.assert_called_once_with(
                mock_model,
                r=64,
                target_modules=["q_proj", "v_proj"],
                lora_alpha=128,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )

    def test_apply_lora_all_linear(self) -> None:
        """Test LoRA application with 'all-linear' target modules."""
        adapter = UnslothAdapter()
        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        config = LoRAConfig(rank=32, target_modules=["all-linear"])

        # Mock the unsloth module before import
        mock_unsloth_module = MagicMock()
        mock_fast_model = MagicMock()
        mock_fast_model.get_peft_model.return_value = mock_peft_model
        mock_unsloth_module.FastLanguageModel = mock_fast_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth_module}):
            adapter.apply_lora(mock_model, config)

            # Should pass "all-linear" as string, not list
            mock_fast_model.get_peft_model.assert_called_once_with(
                mock_model,
                r=32,
                target_modules="all-linear",
                lora_alpha=32,  # Defaults to rank
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )

    def test_apply_lora_default_alpha(self) -> None:
        """Test LoRA application with default alpha (equals rank)."""
        adapter = UnslothAdapter()
        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        config = LoRAConfig(rank=64)  # alpha defaults to rank

        # Mock the unsloth module before import
        mock_unsloth_module = MagicMock()
        mock_fast_model = MagicMock()
        mock_fast_model.get_peft_model.return_value = mock_peft_model
        mock_unsloth_module.FastLanguageModel = mock_fast_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth_module}):
            adapter.apply_lora(mock_model, config)

            mock_fast_model.get_peft_model.assert_called_once()
            call_kwargs = mock_fast_model.get_peft_model.call_args[1]
            assert call_kwargs["lora_alpha"] == 64

    def test_apply_lora_import_error(self) -> None:
        """Test ImportError when Unsloth is not installed."""
        adapter = UnslothAdapter()
        mock_model = MagicMock()
        config = LoRAConfig(rank=32)

        # Mock the import to raise ImportError only for 'unsloth'
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "unsloth":
                raise ImportError("No module named 'unsloth'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="Unsloth is not installed"):
                adapter.apply_lora(mock_model, config)

    def test_apply_lora_runtime_error(self) -> None:
        """Test RuntimeError when LoRA application fails."""
        adapter = UnslothAdapter()
        mock_model = MagicMock()
        config = LoRAConfig(rank=32)

        # Mock the unsloth module before import
        mock_unsloth_module = MagicMock()
        mock_fast_model = MagicMock()
        mock_fast_model.get_peft_model.side_effect = RuntimeError(
            "Invalid target modules"
        )
        mock_unsloth_module.FastLanguageModel = mock_fast_model

        with patch.dict("sys.modules", {"unsloth": mock_unsloth_module}):
            with pytest.raises(RuntimeError, match="Failed to apply LoRA"):
                adapter.apply_lora(mock_model, config)
