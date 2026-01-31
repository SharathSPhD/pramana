"""Adapter for Unsloth's FastLanguageModel API.

This adapter provides a mockable interface to Unsloth, allowing tests to run
without GPU dependencies. The adapter wraps Unsloth's model loading and
LoRA configuration APIs.
"""

from typing import Any

from pramana.config.loader import LoRAConfig


class UnslothAdapter:
    """Adapter for Unsloth's FastLanguageModel.

    This adapter provides a clean interface to Unsloth operations, making it
    easy to mock for testing. All Unsloth-specific imports are contained here.

    Example:
        ```python
        adapter = UnslothAdapter()
        model, tokenizer = adapter.load_model("meta-llama/Llama-3.2-3B", "4bit")
        model = adapter.apply_lora(model, LoRAConfig(rank=64))
        ```
    """

    def load_model(
        self, model_name: str, quantization: str = "4bit"
    ) -> tuple[Any, Any]:
        """Load model and tokenizer via Unsloth.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-3B")
            quantization: Quantization method ("4bit", "8bit", or "16bit")

        Returns:
            Tuple of (model, tokenizer) from Unsloth

        Raises:
            ImportError: If unsloth is not installed
            RuntimeError: If model loading fails (e.g., GPU not available)

        Note:
            This method requires GPU access. For testing, mock this method.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth is not installed. Install with: pip install unsloth"
            ) from e

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=4096,
                dtype=None,  # Auto-detect
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
            )
            return model, tokenizer
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {model_name}: {e}"
            ) from e

    def apply_lora(self, model: Any, config: LoRAConfig) -> Any:
        """Apply LoRA adapters to the model.

        Args:
            model: Unsloth model instance
            config: LoRA configuration (rank, alpha, target_modules)

        Returns:
            Model with LoRA adapters applied

        Raises:
            RuntimeError: If LoRA application fails

        Note:
            This method modifies the model in-place and returns it for chaining.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth is not installed. Install with: pip install unsloth"
            ) from e

        try:
            # Determine target modules
            target_modules: list[str] | str = config.target_modules
            if target_modules == ["all-linear"]:
                # Unsloth's default: all linear layers
                target_modules = "all-linear"

            # Apply LoRA
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.rank,
                target_modules=target_modules,
                lora_alpha=config.alpha or config.rank,
                lora_dropout=0.0,  # Default from CLAUDE.md
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,  # Default seed from CLAUDE.md
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to apply LoRA: {e}") from e
