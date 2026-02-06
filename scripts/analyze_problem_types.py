#!/usr/bin/env python3
"""
Analyze evaluation results by problem type.

Categorizes test examples and calculates per-type metrics:
- Format adherence rate
- Semantic correctness rate
- Average output length
- Parse success rate
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Problem type mapping from example IDs
PROBLEM_TYPE_MAP = {
    "pramana-003": "Transitive Reasoning",
    "pramana-005": "Multi-step Deduction",
    "test-001": "Constraint Satisfaction",
    "test-002": "Constraint Satisfaction",
    "test-003": "Constraint Satisfaction",
    "test-004": "Boolean SAT",
    "test-005": "Boolean SAT",
    "test-006": "Set Operations",
    "test-007": "Set Operations",
    "test-008": "Multi-step Deduction",
}

# Standardized problem type names for LaTeX
PROBLEM_TYPE_LATEX = {
    "Constraint Satisfaction": "Constraint Satisfaction",
    "Boolean SAT": "Boolean SAT",
    "Transitive Reasoning": "Transitive Reasoning",
    "Set Operations": "Set Operations",
    "Multi-step Deduction": "Multi-step Deduction",
}


def load_evaluation_results(filepath: Path) -> Dict:
    """Load evaluation results JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def categorize_by_problem_type(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by problem type."""
    categorized = defaultdict(list)
    
    for result in results:
        example_id = result["example_id"]
        problem_type = PROBLEM_TYPE_MAP.get(example_id, "Unknown")
        categorized[problem_type].append(result)
    
    return categorized


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate metrics for a group of results."""
    if not results:
        return {
            "count": 0,
            "format_rate": 0.0,
            "semantic_rate": 0.0,
            "parse_rate": 0.0,
            "avg_length": 0.0,
        }
    
    format_passed = sum(
        1 for r in results 
        if r.get("parse_success", False) and 
        r.get("format_metrics", {}).get("num_phases_present", 0) == 6
    )
    
    semantic_passed = sum(
        1 for r in results 
        if r.get("ground_truth_match", {}).get("semantic_match", False)
    )
    
    parse_passed = sum(1 for r in results if r.get("parse_success", False))
    
    total_length = sum(r.get("output_length", 0) for r in results)
    
    return {
        "count": len(results),
        "format_rate": format_passed / len(results) * 100,
        "semantic_rate": semantic_passed / len(results) * 100,
        "parse_rate": parse_passed / len(results) * 100,
        "avg_length": total_length / len(results) if results else 0.0,
    }


def generate_latex_table(metrics_by_type: Dict[str, Dict]) -> str:
    """Generate LaTeX table for problem type performance."""
    # Order problem types consistently
    type_order = [
        "Constraint Satisfaction",
        "Boolean SAT",
        "Transitive Reasoning",
        "Set Operations",
        "Multi-step Deduction",
    ]
    
    lines = [
        "\\begin{table}[t]",
        "\\caption{Performance breakdown by problem type.}",
        "\\label{tab:problem_type_performance}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Problem Type & Examples & Format Rate & Semantic Rate \\\\",
        "\\midrule",
    ]
    
    for ptype in type_order:
        if ptype in metrics_by_type:
            m = metrics_by_type[ptype]
            lines.append(
                f"{ptype} & {m['count']} & {m['format_rate']:.1f}\\% & {m['semantic_rate']:.1f}\\% \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)


def generate_csv_data(metrics_by_type: Dict[str, Dict], stage: str) -> str:
    """Generate CSV data for visualization."""
    type_order = [
        "Constraint Satisfaction",
        "Boolean SAT",
        "Transitive Reasoning",
        "Set Operations",
        "Multi-step Deduction",
    ]
    
    lines = ["problem_type,stage,count,format_rate,semantic_rate,parse_rate,avg_length"]
    
    for ptype in type_order:
        if ptype in metrics_by_type:
            m = metrics_by_type[ptype]
            lines.append(
                f"{ptype},{stage},{m['count']},{m['format_rate']:.2f},"
                f"{m['semantic_rate']:.2f},{m['parse_rate']:.2f},{m['avg_length']:.1f}"
            )
    
    return "\n".join(lines)


def analyze_stage(stage_num: int) -> Tuple[Dict[str, Dict], str]:
    """Analyze a single stage's evaluation results."""
    if stage_num == 0:
        results_file = Path("results/stage_0_final_validation.json")
    elif stage_num == 1:
        results_file = Path("results/stage_1_evaluation.json")
    else:
        raise ValueError(f"Unknown stage: {stage_num}")
    
    data = load_evaluation_results(results_file)
    categorized = categorize_by_problem_type(data["results"])
    
    metrics_by_type = {}
    for ptype, results in categorized.items():
        metrics_by_type[ptype] = calculate_metrics(results)
    
    return metrics_by_type, f"stage_{stage_num}"


def main():
    """Main analysis function."""
    # Analyze both stages
    metrics_0, stage_0 = analyze_stage(0)
    metrics_1, stage_1 = analyze_stage(1)
    
    # Combine results (for overall analysis, use stage 1 as primary)
    # Stage 0 and Stage 1 use same test set, so we'll analyze Stage 1
    combined_metrics = metrics_1
    
    # Generate outputs
    latex_table = generate_latex_table(combined_metrics)
    csv_data = generate_csv_data(combined_metrics, stage_1)
    
    # Print results
    print("=" * 80)
    print("PROBLEM TYPE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print("\nMetrics by Problem Type:\n")
    
    type_order = [
        "Constraint Satisfaction",
        "Boolean SAT",
        "Transitive Reasoning",
        "Set Operations",
        "Multi-step Deduction",
    ]
    
    for ptype in type_order:
        if ptype in combined_metrics:
            m = combined_metrics[ptype]
            print(f"{ptype}:")
            print(f"  Examples: {m['count']}")
            print(f"  Format Adherence: {m['format_rate']:.1f}%")
            print(f"  Semantic Correctness: {m['semantic_rate']:.1f}%")
            print(f"  Parse Success: {m['parse_rate']:.1f}%")
            print(f"  Avg Output Length: {m['avg_length']:.1f} tokens")
            print()
    
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(latex_table)
    
    print("\n" + "=" * 80)
    print("CSV DATA")
    print("=" * 80)
    print(csv_data)
    
    # Write outputs to files
    output_dir = Path("docs/figures_stage1_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "problem_type_performance_table.tex", "w") as f:
        f.write(latex_table)
    
    with open(output_dir / "problem_type_performance.csv", "w") as f:
        f.write(csv_data)
    
    print(f"\nOutputs written to:")
    print(f"  - {output_dir / 'problem_type_performance_table.tex'}")
    print(f"  - {output_dir / 'problem_type_performance.csv'}")


if __name__ == "__main__":
    main()
