"""Hetvabhasa taxonomy coverage analysis.

Analyzes completeness, ambiguity, and predictive power of the
Hetvabhasa classification on evaluation results.
"""

from dataclasses import dataclass


CATEGORIES = ["savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalatita"]


@dataclass
class TaxonomyCoverageReport:
    total_failures: int
    classified_count: int
    unclassified_count: int
    coverage_pct: float
    distribution: dict[str, int]
    by_ground_truth: dict[str, dict]  # gt_type -> {classified_as: {type: count}}
    predictive_accuracy: float  # Assisted: includes fallback-assigned classifications
    strict_predictive_accuracy: float  # Strict: excludes fallback-assigned classifications
    fallback_count: int  # Number of failures classified via fallback prior
    ambiguous_count: int  # failures that could map to multiple categories


def analyze_taxonomy_coverage(results: dict) -> TaxonomyCoverageReport:
    """Analyze Hetvabhasa taxonomy coverage from evaluation results."""
    failures = []

    for model_name, model_results in results.items():
        for r in model_results:
            if not r.get("final_answer_correct", False):
                failures.append({
                    "model": model_name,
                    "problem_id": r.get("problem_id", ""),
                    "classified_type": r.get("hetvabhasa_classification", "unclassified"),
                    "ground_truth_type": r.get("category", "unknown"),
                    "used_fallback": r.get("hetvabhasa_used_fallback", False),
                })

    total = len(failures)
    if total == 0:
        return TaxonomyCoverageReport(0, 0, 0, 0.0, {}, {}, 0.0, 0.0, 0, 0)

    # Distribution
    dist = {}
    for f in failures:
        ct = f["classified_type"]
        dist[ct] = dist.get(ct, 0) + 1

    classified = sum(v for k, v in dist.items() if k in CATEGORIES)
    unclassified = total - classified

    # By ground truth
    by_gt = {}
    correct_classifications = 0
    strict_total = 0
    strict_correct = 0
    fallback_count = 0
    for f in failures:
        gt = f["ground_truth_type"]
        ct = f["classified_type"]
        used_fallback = bool(f.get("used_fallback", False))
        if gt not in by_gt:
            by_gt[gt] = {}
        by_gt[gt][ct] = by_gt[gt].get(ct, 0) + 1
        if gt == ct:
            correct_classifications += 1
        if used_fallback:
            fallback_count += 1
        else:
            strict_total += 1
            if gt == ct:
                strict_correct += 1

    return TaxonomyCoverageReport(
        total_failures=total,
        classified_count=classified,
        unclassified_count=unclassified,
        coverage_pct=(classified / total * 100) if total > 0 else 0.0,
        distribution=dist,
        by_ground_truth=by_gt,
        predictive_accuracy=(correct_classifications / total * 100) if total > 0 else 0.0,
        strict_predictive_accuracy=(strict_correct / strict_total * 100) if strict_total > 0 else 0.0,
        fallback_count=fallback_count,
        ambiguous_count=0,  # Would need more detailed per-failure analysis
    )
