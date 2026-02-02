"""Backend selection utilities for model loading."""


def should_use_unsloth(*, prefer_unsloth: bool, torch_available: bool, has_gpu: bool) -> bool:
    """Decide whether to use Unsloth based on availability and preference."""
    return bool(prefer_unsloth and torch_available and has_gpu)


__all__ = ["should_use_unsloth"]
