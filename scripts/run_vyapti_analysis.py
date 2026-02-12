#!/usr/bin/env python3
"""Phase 4a-b: Run statistical analysis and generate visualizations.

Usage:
    python scripts/run_vyapti_analysis.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pramana.analysis.comparison_tests import run_all_comparisons
from pramana.analysis.taxonomy_coverage import analyze_taxonomy_coverage
from pramana.analysis.visualization import generate_all_plots


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load evaluation results from per-model directories."""
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found at {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    # Also load individual results for detailed analysis
    all_results = {}
    for model_name in summary:
        model_dir = results_dir / model_name.replace("/", "_").replace(" ", "_")
        if not model_dir.exists():
            continue
        model_results = []
        for result_file in sorted(model_dir.glob("*.json")):
            if result_file.name == "summary.json":
                continue
            with open(result_file) as f:
                model_results.append(json.load(f))
        all_results[model_name] = model_results

    return all_results, summary


def main():
    results_dir = PROJECT_ROOT / "data" / "vyapti_probe" / "results_real"
    figures_dir = PROJECT_ROOT / "docs" / "paper_vyapti" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    all_results, summary = load_results(results_dir)
    print(f"Loaded results for {len(all_results)} models")

    # ── Run 4 key comparisons ──
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS")
    print("=" * 60)

    comparisons = run_all_comparisons(all_results)

    for c in comparisons:
        print(f"\n{c.name}")
        print(f"  {c.description}")
        print(f"  Group A accuracy: {c.group_a_accuracy:.3f}")
        print(f"  Group B accuracy: {c.group_b_accuracy:.3f}")
        print(f"  Difference: {c.difference:+.3f}")
        if c.p_value_approx >= 0:
            print(f"  95% CI: [{c.ci_lower:+.3f}, {c.ci_upper:+.3f}]")
            print(f"  Significant: {c.significant}")
            print(f"  Approx p-value: {c.p_value_approx:.4f}")
        else:
            print("  95% CI: N/A (descriptive metric)")
            print("  Significant: N/A (descriptive metric)")
            print("  Approx p-value: N/A")
        print(f"  N samples: {c.n_samples}")

    # Save comparisons
    comp_data = [asdict(c) for c in comparisons]
    with open(results_dir / "comparisons.json", "w") as f:
        json.dump(comp_data, f, indent=2, default=_json_default)
    print(f"\nComparisons saved to {results_dir / 'comparisons.json'}")

    # ── Taxonomy coverage ──
    print("\n" + "=" * 60)
    print("HETVABHASA TAXONOMY COVERAGE")
    print("=" * 60)

    taxonomy = analyze_taxonomy_coverage(all_results)
    print(f"  Total failures: {taxonomy.total_failures}")
    print(f"  Classified: {taxonomy.classified_count} ({taxonomy.coverage_pct:.1f}%)")
    print(f"  Unclassified: {taxonomy.unclassified_count}")
    print(f"  Assisted predictive accuracy: {taxonomy.predictive_accuracy:.1f}%")
    print(f"  Strict predictive accuracy (no fallback): {taxonomy.strict_predictive_accuracy:.1f}%")
    print(f"  Fallback classifications: {taxonomy.fallback_count}")
    print(f"  Distribution: {taxonomy.distribution}")

    # By ground truth
    print("\n  By ground truth type:")
    for gt, classified_as in taxonomy.by_ground_truth.items():
        print(f"    {gt}: {classified_as}")

    taxonomy_data = asdict(taxonomy)
    with open(results_dir / "taxonomy_coverage.json", "w") as f:
        json.dump(taxonomy_data, f, indent=2)
    print(f"\nTaxonomy analysis saved to {results_dir / 'taxonomy_coverage.json'}")

    # ── Visualizations ──
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    generate_all_plots(summary, comparisons, figures_dir)

    # ── Auto-generated report ──
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    report = generate_report(summary, comparisons, taxonomy)
    report_path = results_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    print("\n=== Analysis complete ===")


def generate_report(summary: dict, comparisons: list, taxonomy) -> str:
    """Generate a markdown report from analysis results."""
    lines = [
        "# Vyapti Probe Benchmark — Evaluation Report",
        "",
        "## Overview",
        "",
        "| Model | Accuracy | Probe Acc | Control Acc |",
        "|-------|----------|-----------|-------------|",
    ]

    for model, data in summary.items():
        lines.append(
            f"| {model} | {data['accuracy']:.1%} | "
            f"{data['probe_accuracy']:.1%} | {data['control_accuracy']:.1%} |"
        )

    lines += [
        "",
        "## Key Statistical Comparisons",
        "",
    ]

    for c in comparisons:
        if c.p_value_approx >= 0:
            sig = "**Significant**" if c.significant else "Not significant"
            diff_line = (
                f"- Difference: {c.difference:+.3f} "
                f"(95% CI: [{c.ci_lower:+.3f}, {c.ci_upper:+.3f}])"
            )
            stat_line = f"- {sig} (p ≈ {c.p_value_approx:.4f})"
        else:
            sig = "Descriptive metric (no inferential test)"
            diff_line = f"- Coverage: {c.difference:.3f}"
            stat_line = f"- {sig}"
        lines += [
            f"### {c.name}",
            f"",
            f"{c.description}",
            f"",
            diff_line,
            stat_line,
            f"- N = {c.n_samples}",
            "",
        ]

    lines += [
        "## Hetvabhasa Taxonomy Coverage",
        "",
        f"- Total failures across all models: {taxonomy.total_failures}",
        f"- Classified into Hetvabhasa categories: {taxonomy.classified_count} ({taxonomy.coverage_pct:.1f}%)",
        f"- Unclassified: {taxonomy.unclassified_count}",
        f"- Assisted predictive accuracy (includes fallback): {taxonomy.predictive_accuracy:.1f}%",
        f"- Strict predictive accuracy (excludes fallback): {taxonomy.strict_predictive_accuracy:.1f}%",
        f"- Fallback classifications: {taxonomy.fallback_count}",
        "",
        "### Distribution",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]

    for cat, count in sorted(taxonomy.distribution.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")

    lines += [
        "",
        "## Category-wise Performance",
        "",
    ]

    for model, data in summary.items():
        lines += [f"### {model}", ""]
        lines += [
            "| Category | Probe | Control |",
            "|----------|-------|---------|",
        ]
        for cat, cat_data in data.get("by_category", {}).items():
            pt = cat_data.get("probe_total", 0)
            ct = cat_data.get("control_total", 0)
            pc = cat_data.get("probe_correct", 0)
            cc = cat_data.get("control_correct", 0)
            pa = f"{pc}/{pt}" if pt else "N/A"
            ca = f"{cc}/{ct}" if ct else "N/A"
            lines.append(f"| {cat} | {pa} | {ca} |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
