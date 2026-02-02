"""Semantic answer scoring utilities."""

from __future__ import annotations

import math
import os
import re
from functools import lru_cache


def normalize_text(text: str) -> str:
    """Normalize text for comparisons."""
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def token_overlap_ratio(predicted: str, ground_truth: str) -> float:
    """Token overlap ratio relative to ground truth tokens."""
    pred_tokens = set(normalize_text(predicted).split())
    gt_tokens = set(normalize_text(ground_truth).split())
    if not gt_tokens:
        return 0.0
    return len(pred_tokens & gt_tokens) / len(gt_tokens)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=2)
def _get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def semantic_similarity(
    predicted: str,
    ground_truth: str,
    *,
    use_embeddings: bool = True,
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """Compute semantic similarity using embeddings when available."""
    disable_embeddings = os.getenv("PRAMANA_DISABLE_EMBEDDINGS", "0") == "1"
    if use_embeddings and not disable_embeddings:
        try:
            model = _get_embedding_model(model_name)
            embeddings = model.encode([predicted, ground_truth])
            return _cosine_similarity(embeddings[0], embeddings[1])
        except Exception:
            pass
    return token_overlap_ratio(predicted, ground_truth)


def semantic_match(
    predicted: str,
    ground_truth: str,
    *,
    threshold: float = 0.7,
    use_embeddings: bool = True,
) -> bool:
    """Return True if semantic similarity exceeds threshold."""
    return semantic_similarity(
        predicted, ground_truth, use_embeddings=use_embeddings
    ) >= threshold


def score_answers(
    predicted: str,
    ground_truth: str,
    *,
    threshold: float = 0.7,
    use_embeddings: bool = True,
) -> dict[str, bool | float]:
    """Calculate answer matching metrics for evaluation reporting."""
    exact_match = predicted.strip().lower() == ground_truth.strip().lower()
    normalized_pred = normalize_text(predicted)
    normalized_gt = normalize_text(ground_truth)
    normalized_match = normalized_gt in normalized_pred if normalized_gt else False
    similarity = semantic_similarity(
        predicted, ground_truth, use_embeddings=use_embeddings
    )
    return {
        "exact_match": exact_match,
        "normalized_match": normalized_match,
        "semantic_match": similarity >= threshold,
        "semantic_similarity": similarity,
        "token_overlap": token_overlap_ratio(predicted, ground_truth),
    }


def wilson_interval(*, successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1 + (z * z) / total
    center = (p_hat + (z * z) / (2 * total)) / denom
    margin = (
        z
        * math.sqrt((p_hat * (1 - p_hat) / total) + (z * z) / (4 * total * total))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


__all__ = [
    "normalize_text",
    "semantic_match",
    "semantic_similarity",
    "score_answers",
    "token_overlap_ratio",
    "wilson_interval",
]
