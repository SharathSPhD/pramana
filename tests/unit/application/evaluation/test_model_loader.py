"""Unit tests for model loader backend selection."""


def test_should_use_unsloth_requires_preference_and_gpu() -> None:
    from pramana.application.evaluation.model_loader import should_use_unsloth

    assert should_use_unsloth(prefer_unsloth=True, torch_available=True, has_gpu=True)
    assert not should_use_unsloth(prefer_unsloth=True, torch_available=True, has_gpu=False)
    assert not should_use_unsloth(prefer_unsloth=False, torch_available=True, has_gpu=True)
    assert not should_use_unsloth(prefer_unsloth=True, torch_available=False, has_gpu=True)
