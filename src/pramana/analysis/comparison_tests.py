"""Statistical comparisons for vyapti benchmark evaluation.

Implements the four key hypothesis tests from the vyapti implementation plan:
C1: Base models probe vs control accuracy (paired)
C2: Pramana Stage 1 vs base DeepSeek on probes
C3: Base + Nyaya template vs Stage 1 (methodology vs fine-tuning)
C4: Hetvabhasa taxonomy coverage
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComparisonResult:
    name: str
    description: str
    group_a_accuracy: float
    group_b_accuracy: float
    difference: float
    ci_lower: float
    ci_upper: float
    significant: bool
    p_value_approx: float
    n_samples: int
    details: dict


def bootstrap_ci(data_a: list[bool], data_b: list[bool], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval and two-sided p-value."""
    rng = np.random.default_rng(42)
    n = len(data_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mean_a = np.mean([data_a[i] for i in idx])
        mean_b = np.mean([data_b[i] for i in idx])
        diffs.append(mean_a - mean_b)
    diffs = sorted(diffs)
    ci_lo = diffs[int(n_boot * alpha / 2)]
    ci_hi = diffs[int(n_boot * (1 - alpha / 2))]
    p_pos = np.mean([d >= 0 for d in diffs])
    p_neg = np.mean([d <= 0 for d in diffs])
    p_value = min(1.0, 2 * min(p_pos, p_neg))
    return ci_lo, ci_hi, p_value


def _build_paired_lists(results: dict, model_names: list[str]) -> tuple[list[bool], list[bool]]:
    """Build properly paired probe/control lists by problem ID convention.

    For each base model, match each probe (e.g. SAV-01) to its control
    (SAV-01-C) using the benchmark naming convention, ensuring correct
    pairing for bootstrap inference.
    """
    probe_list: list[bool] = []
    control_list: list[bool] = []

    for model in model_names:
        if model not in results:
            continue
        # Index all results by problem_id for this model
        by_pid: dict[str, dict] = {}
        for r in results[model]:
            by_pid[r.get("problem_id", "")] = r

        # For each probe, find its matched control (probe_id + "-C")
        for r in sorted(results[model], key=lambda x: x.get("problem_id", "")):
            if r.get("problem_type") != "probe":
                continue
            pid = r.get("problem_id", "")
            control_id = f"{pid}-C"
            if control_id in by_pid:
                probe_list.append(r.get("final_answer_correct", False))
                control_list.append(by_pid[control_id].get("final_answer_correct", False))

    return probe_list, control_list


def comparison_1(results: dict) -> ComparisonResult:
    """C1: Base models on probes vs controls (paired by matched_pair).

    H0: Base models perform equally on probes and controls.
    H1: Base models show a non-zero probe--control gap.
    """
    base_models = ["llama_3b_base", "deepseek_8b_base"]
    probe_correct, control_correct = _build_paired_lists(results, base_models)

    if not probe_correct or not control_correct:
        return ComparisonResult("C1", "Probe vs Control (base models)", 0, 0, 0, 0, 0, False, 1.0, 0, {})

    n_pairs = len(probe_correct)
    probe_acc = np.mean(probe_correct)
    control_acc = np.mean(control_correct)
    ci_lo, ci_hi, p_val = bootstrap_ci(control_correct, probe_correct)

    return ComparisonResult(
        name="C1: Probe vs Control (Base Models)",
        description="Do entropy-minimizing models fail more on vyapti-requiring problems?",
        group_a_accuracy=float(control_acc),
        group_b_accuracy=float(probe_acc),
        difference=float(control_acc - probe_acc),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        significant=(ci_lo > 0 or ci_hi < 0),
        p_value_approx=float(p_val),
        n_samples=n_pairs,
        details={"base_models": base_models, "probe_n": n_pairs, "control_n": n_pairs},
    )


def _build_probe_paired_by_problem(results: dict, model_a: str, model_b: str) -> tuple[list[bool], list[bool]]:
    """Build paired probe lists across two models, aligned by problem_id."""
    a_by_pid: dict[str, bool] = {}
    b_by_pid: dict[str, bool] = {}

    for r in results.get(model_a, []):
        if r.get("problem_type") == "probe":
            a_by_pid[r["problem_id"]] = r.get("final_answer_correct", False)
    for r in results.get(model_b, []):
        if r.get("problem_type") == "probe":
            b_by_pid[r["problem_id"]] = r.get("final_answer_correct", False)

    # Only include problems present in both models
    common = sorted(set(a_by_pid) & set(b_by_pid))
    return [a_by_pid[p] for p in common], [b_by_pid[p] for p in common]


def comparison_2(results: dict) -> ComparisonResult:
    """C2: Pramana Stage 1 vs base DeepSeek on probes (paired by problem_id)."""
    pramana_probes, base_probes = _build_probe_paired_by_problem(
        results, "stage1_pramana", "deepseek_8b_base"
    )

    if not pramana_probes or not base_probes:
        return ComparisonResult("C2", "Pramana vs Base (probes)", 0, 0, 0, 0, 0, False, 1.0, 0, {})

    n_pairs = len(pramana_probes)
    pramana_acc = np.mean(pramana_probes)
    base_acc = np.mean(base_probes)
    ci_lo, ci_hi, p_val = bootstrap_ci(pramana_probes, base_probes)

    return ComparisonResult(
        name="C2: Pramana vs Base DeepSeek (Probes)",
        description="Does Nyaya training improve performance on vyapti-requiring problems?",
        group_a_accuracy=float(pramana_acc),
        group_b_accuracy=float(base_acc),
        difference=float(pramana_acc - base_acc),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        significant=(ci_lo > 0 or ci_hi < 0),
        p_value_approx=float(p_val),
        n_samples=n_pairs,
        details={},
    )


def comparison_3(results: dict) -> ComparisonResult:
    """C3: Base + Nyaya template vs Stage 1 (paired by problem_id)."""
    stage1_probes, template_probes = _build_probe_paired_by_problem(
        results, "stage1_pramana", "base_with_nyaya_template"
    )

    if not stage1_probes or not template_probes:
        return ComparisonResult("C3", "Template vs Fine-tuned (probes)", 0, 0, 0, 0, 0, False, 1.0, 0, {})

    n_pairs = len(stage1_probes)
    stage1_acc = np.mean(stage1_probes)
    template_acc = np.mean(template_probes)
    ci_lo, ci_hi, p_val = bootstrap_ci(stage1_probes, template_probes)

    return ComparisonResult(
        name="C3: Fine-tuned Stage 1 vs Prompted Template (Probes)",
        description="Does fine-tuning provide advantage over just prompting with Nyaya template?",
        group_a_accuracy=float(stage1_acc),
        group_b_accuracy=float(template_acc),
        difference=float(stage1_acc - template_acc),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        significant=(ci_lo > 0 or ci_hi < 0),
        p_value_approx=float(p_val),
        n_samples=n_pairs,
        details={},
    )


def comparison_4(results: dict) -> ComparisonResult:
    """C4: Hetvabhasa taxonomy coverage."""
    all_failures = []
    for model_name, model_results in results.items():
        for r in model_results:
            if not r.get("final_answer_correct", False):
                htype = r.get("hetvabhasa_classification", "unclassified")
                all_failures.append(htype)

    if not all_failures:
        return ComparisonResult("C4", "Taxonomy Coverage", 0, 0, 0, 0, 0, False, 1.0, 0, {})

    classified = sum(1 for h in all_failures if h not in ("unclassified", "none", ""))
    total = len(all_failures)
    coverage = classified / total if total > 0 else 0

    # Distribution
    dist = {}
    for h in all_failures:
        dist[h] = dist.get(h, 0) + 1

    return ComparisonResult(
        name="C4: Hetvabhasa Taxonomy Coverage (Descriptive)",
        description="What percentage of failures map to exactly one Hetvabhasa category?",
        group_a_accuracy=float(coverage),
        group_b_accuracy=0.0,
        difference=float(coverage),
        ci_lower=float(coverage),
        ci_upper=float(coverage),
        significant=False,
        p_value_approx=-1.0,
        n_samples=total,
        details={
            "distribution": dist,
            "classified": classified,
            "total_failures": total,
            "descriptive_only": True,
            "note": "Coverage metric is descriptive, not an inferential test.",
        },
    )


def run_all_comparisons(results: dict) -> list[ComparisonResult]:
    """Run all 4 comparisons and return results."""
    return [
        comparison_1(results),
        comparison_2(results),
        comparison_3(results),
        comparison_4(results),
    ]
