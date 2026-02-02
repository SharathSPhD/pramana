"""Unit tests for semantic answer scoring."""

import pytest


def test_normalize_text_strips_punctuation() -> None:
    from pramana.application.evaluation.scoring import normalize_text

    assert normalize_text("Bob has the dog.") == "bob has the dog"


def test_semantic_match_uses_token_overlap_when_embeddings_disabled() -> None:
    from pramana.application.evaluation.scoring import semantic_match

    predicted = "Bob has a dog"
    ground_truth = "Bob has the dog"

    assert semantic_match(predicted, ground_truth, threshold=0.7, use_embeddings=False) is True


def test_semantic_match_fails_for_low_overlap() -> None:
    from pramana.application.evaluation.scoring import semantic_match

    predicted = "Alice owns a cat"
    ground_truth = "Bob has the dog"

    assert semantic_match(predicted, ground_truth, threshold=0.7, use_embeddings=False) is False


def test_score_answers_reports_exact_and_semantic_matches() -> None:
    from pramana.application.evaluation.scoring import score_answers

    predicted = "Therefore, Bob has the dog."
    ground_truth = "Bob has the dog"

    scores = score_answers(predicted, ground_truth, use_embeddings=False, threshold=0.7)

    assert scores["exact_match"] is False
    assert scores["normalized_match"] is True
    assert scores["semantic_match"] is True


def test_wilson_interval_returns_valid_bounds() -> None:
    from pramana.application.evaluation.scoring import wilson_interval

    low, high = wilson_interval(successes=8, total=10)

    assert 0.0 <= low <= high <= 1.0
