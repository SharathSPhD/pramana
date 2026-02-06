#!/usr/bin/env python3
"""
Analyze ablation study results for Stage 0 and Stage 1.
Generates LaTeX tables and comprehensive analysis summary.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def load_ablation_data(stage: int) -> Tuple[List[Dict], List[Dict]]:
    """Load ablation summary and details for a stage."""
    base_path = Path("docs/figures_ablation_v1")
    
    # Load summary
    summary_path = base_path / f"stage{stage}_ablation_summary.json"
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    
    # Load details
    details_path = base_path / f"stage{stage}_ablation_details.json"
    with open(details_path, 'r') as f:
        details_data = json.load(f)
    
    return summary_data, details_data

def verify_data_quality(stage: int, summary_data: List[Dict], details: List[Dict]) -> Dict:
    """Verify data quality and completeness."""
    issues = []
    warnings = []
    
    # Check if all conditions are present
    expected_conditions = ["format_temp0", "format_temp07", "noformat_temp0", "noformat_temp07"]
    actual_conditions = set(item['condition'] for item in summary_data)
    if set(expected_conditions) != actual_conditions:
        issues.append(f"Missing conditions: {set(expected_conditions) - actual_conditions}")
    
    # Check format_rate consistency
    format_rates = [item['format_rate'] for item in summary_data]
    if all(rate == 0.0 for rate in format_rates):
        warnings.append(
            "CRITICAL: Format rate is 0.0% for all conditions. "
            "This suggests either: (1) parsing is failing, (2) model outputs don't match expected format, "
            "or (3) format_rate calculation is incorrect."
        )
    
    # Check sample sizes
    for cond_data in details:
        n_samples = len(cond_data['details'])
        if n_samples < 5:
            warnings.append(f"Low sample size ({n_samples}) for condition {cond_data['condition']}")
    
    # Check parse success rates
    parse_rates = {}
    for cond_data in details:
        parse_successes = sum(1 for d in cond_data['details'] if d['parse_success'])
        parse_rate = parse_successes / len(cond_data['details'])
        parse_rates[cond_data['condition']] = parse_rate
        if parse_rate == 0.0:
            warnings.append(
                f"Zero parse success rate for {cond_data['condition']}. "
                "All outputs failed format parsing."
            )
    
    return {
        'issues': issues,
        'warnings': warnings,
        'parse_rates': parse_rates,
        'n_samples_per_condition': len(details[0]['details']) if details else 0
    }

def calculate_statistics(stage: int, summary_data: List[Dict], details: List[Dict]) -> List[Dict]:
    """Calculate additional statistics from details."""
    stats = []
    
    # Create lookup dict for summary data
    summary_lookup = {item['condition']: item for item in summary_data}
    
    for cond_data in details:
        cond_name = cond_data['condition']
        cond_details = cond_data['details']
        
        # Get summary row
        summary_row = summary_lookup[cond_name]
        
        # Calculate additional stats from details
        parse_successes = sum(1 for d in cond_details if d['parse_success'])
        semantic_matches = sum(1 for d in cond_details if d['semantic_match'])
        n_samples = len(cond_details)
        
        parse_rate = parse_successes / n_samples if n_samples > 0 else 0.0
        
        stats.append({
            'stage': stage,
            'condition': cond_name,
            'format_rate': summary_row['format_rate'],
            'semantic_rate': summary_row['semantic_rate'],
            'avg_output_tokens': summary_row['avg_output_tokens'],
            'parse_rate': parse_rate,
            'n_samples': n_samples,
            'n_parse_success': parse_successes,
            'n_semantic_match': semantic_matches
        })
    
    return stats

def generate_latex_table(stage0_stats: List[Dict], stage1_stats: List[Dict]) -> str:
    """Generate LaTeX table for ablation results."""
    
    def format_condition(cond: str) -> str:
        """Format condition name for table."""
        if cond == "format_temp0":
            return "Format + Temp 0.0"
        elif cond == "format_temp07":
            return "Format + Temp 0.7"
        elif cond == "noformat_temp0":
            return "NoFormat + Temp 0.0"
        elif cond == "noformat_temp07":
            return "NoFormat + Temp 0.7"
        return cond
    
    def format_percentage(val: float) -> str:
        """Format as percentage."""
        return f"{val*100:.1f}\\%"
    
    def format_tokens(val: float) -> str:
        """Format token count."""
        return f"{val:.1f}"
    
    # Sort stats by condition for consistent ordering
    def sort_key(x):
        order = {"format_temp0": 0, "format_temp07": 1, "noformat_temp0": 2, "noformat_temp07": 3}
        return order.get(x['condition'], 99)
    
    stage0_sorted = sorted(stage0_stats, key=sort_key)
    stage1_sorted = sorted(stage1_stats, key=sort_key)
    
    latex_lines = [
        "\\begin{table}[t]",
        "\\caption{Ablation study: Format prompting × Temperature effects on format adherence and semantic correctness.}",
        "\\label{tab:ablation_summary}",
        "\\centering",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Stage & Condition & Format Rate & Semantic Rate & Avg Tokens \\\\",
        "\\midrule"
    ]
    
    # Stage 0 rows
    for row in stage0_sorted:
        cond = format_condition(row['condition'])
        latex_lines.append(
            f"Stage 0 & {cond} & {format_percentage(row['format_rate'])} & "
            f"{format_percentage(row['semantic_rate'])} & {format_tokens(row['avg_output_tokens'])} \\\\"
        )
    
    latex_lines.append("\\midrule")
    
    # Stage 1 rows
    for row in stage1_sorted:
        cond = format_condition(row['condition'])
        latex_lines.append(
            f"Stage 1 & {cond} & {format_percentage(row['format_rate'])} & "
            f"{format_percentage(row['semantic_rate'])} & {format_tokens(row['avg_output_tokens'])} \\\\"
        )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)

def analyze_effects(stage0_stats: List[Dict], stage1_stats: List[Dict]) -> Dict:
    """Analyze main effects and interactions."""
    
    def get_effect(stats_list: List[Dict], metric: str, format_enabled: bool, temp: float) -> float:
        """Get metric value for specific condition."""
        if format_enabled:
            cond = f"format_temp{int(temp)}" if temp == 0.0 else "format_temp07"
        else:
            cond = f"noformat_temp{int(temp)}" if temp == 0.0 else "noformat_temp07"
        
        for row in stats_list:
            if row['condition'] == cond:
                return row[metric]
        return 0.0
    
    def analyze_stage(stats_list: List[Dict], stage_name: str) -> Dict:
        """Analyze effects for a single stage."""
        results = {}
        
        # Format effect at temp 0.0
        format_effect_temp0 = (
            get_effect(stats_list, 'semantic_rate', True, 0.0) - 
            get_effect(stats_list, 'semantic_rate', False, 0.0)
        )
        
        # Format effect at temp 0.7
        format_effect_temp07 = (
            get_effect(stats_list, 'semantic_rate', True, 0.7) - 
            get_effect(stats_list, 'semantic_rate', False, 0.7)
        )
        
        # Temperature effect with format
        temp_effect_with_format = (
            get_effect(stats_list, 'semantic_rate', True, 0.7) - 
            get_effect(stats_list, 'semantic_rate', True, 0.0)
        )
        
        # Temperature effect without format
        temp_effect_no_format = (
            get_effect(stats_list, 'semantic_rate', False, 0.7) - 
            get_effect(stats_list, 'semantic_rate', False, 0.0)
        )
        
        # Interaction effect
        interaction = (
            (get_effect(stats_list, 'semantic_rate', True, 0.7) - get_effect(stats_list, 'semantic_rate', True, 0.0)) -
            (get_effect(stats_list, 'semantic_rate', False, 0.7) - get_effect(stats_list, 'semantic_rate', False, 0.0))
        )
        
        # Find optimal configuration
        best_row = max(stats_list, key=lambda x: x['semantic_rate'])
        best_cond = best_row['condition']
        best_semantic_rate = best_row['semantic_rate']
        
        results[f'{stage_name}_format_effect_temp0'] = format_effect_temp0
        results[f'{stage_name}_format_effect_temp07'] = format_effect_temp07
        results[f'{stage_name}_temp_effect_with_format'] = temp_effect_with_format
        results[f'{stage_name}_temp_effect_no_format'] = temp_effect_no_format
        results[f'{stage_name}_interaction'] = interaction
        results[f'{stage_name}_optimal_condition'] = best_cond
        results[f'{stage_name}_optimal_semantic_rate'] = best_semantic_rate
        
        return results
    
    stage0_effects = analyze_stage(stage0_stats, 'stage0')
    stage1_effects = analyze_stage(stage1_stats, 'stage1')
    
    # Cross-stage comparison
    comparison = {
        'stage0_vs_stage1_format_effect_temp0': (
            stage0_effects['stage0_format_effect_temp0'] - 
            stage1_effects['stage1_format_effect_temp0']
        ),
        'stage0_vs_stage1_format_effect_temp07': (
            stage0_effects['stage0_format_effect_temp07'] - 
            stage1_effects['stage1_format_effect_temp07']
        ),
        'stage0_vs_stage1_optimal_semantic_rate': (
            stage0_effects['stage0_optimal_semantic_rate'] - 
            stage1_effects['stage1_optimal_semantic_rate']
        )
    }
    
    return {
        'stage0': stage0_effects,
        'stage1': stage1_effects,
        'comparison': comparison
    }

def main():
    """Main analysis function."""
    print("=" * 80)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading ablation data...")
    stage0_summary, stage0_details = load_ablation_data(0)
    stage1_summary, stage1_details = load_ablation_data(1)
    
    # Verify data quality
    print("\n[2] Verifying data quality...")
    stage0_quality = verify_data_quality(0, stage0_summary, stage0_details)
    stage1_quality = verify_data_quality(1, stage1_summary, stage1_details)
    
    print("\n--- Stage 0 Data Quality ---")
    if stage0_quality['issues']:
        print("ISSUES:")
        for issue in stage0_quality['issues']:
            print(f"  ⚠️  {issue}")
    if stage0_quality['warnings']:
        print("WARNINGS:")
        for warning in stage0_quality['warnings']:
            print(f"  ⚠️  {warning}")
    print(f"Samples per condition: {stage0_quality['n_samples_per_condition']}")
    print(f"Parse rates: {stage0_quality['parse_rates']}")
    
    print("\n--- Stage 1 Data Quality ---")
    if stage1_quality['issues']:
        print("ISSUES:")
        for issue in stage1_quality['issues']:
            print(f"  ⚠️  {issue}")
    if stage1_quality['warnings']:
        print("WARNINGS:")
        for warning in stage1_quality['warnings']:
            print(f"  ⚠️  {warning}")
    print(f"Samples per condition: {stage1_quality['n_samples_per_condition']}")
    print(f"Parse rates: {stage1_quality['parse_rates']}")
    
    # Calculate statistics
    print("\n[3] Calculating statistics...")
    stage0_stats = calculate_statistics(0, stage0_summary, stage0_details)
    stage1_stats = calculate_statistics(1, stage1_summary, stage1_details)
    
    print("\n--- Stage 0 Statistics ---")
    print(f"{'Condition':<20} {'Format Rate':<15} {'Semantic Rate':<15} {'Avg Tokens':<15} {'Parse Rate':<15}")
    print("-" * 80)
    for row in stage0_stats:
        print(f"{row['condition']:<20} {row['format_rate']:<15.3f} {row['semantic_rate']:<15.3f} "
              f"{row['avg_output_tokens']:<15.1f} {row['parse_rate']:<15.3f}")
    
    print("\n--- Stage 1 Statistics ---")
    print(f"{'Condition':<20} {'Format Rate':<15} {'Semantic Rate':<15} {'Avg Tokens':<15} {'Parse Rate':<15}")
    print("-" * 80)
    for row in stage1_stats:
        print(f"{row['condition']:<20} {row['format_rate']:<15.3f} {row['semantic_rate']:<15.3f} "
              f"{row['avg_output_tokens']:<15.1f} {row['parse_rate']:<15.3f}")
    
    # Analyze effects
    print("\n[4] Analyzing main effects and interactions...")
    effects = analyze_effects(stage0_stats, stage1_stats)
    
    print("\n--- Stage 0 Effects ---")
    print(f"Format effect (temp 0.0): {effects['stage0']['stage0_format_effect_temp0']:.3f}")
    print(f"Format effect (temp 0.7): {effects['stage0']['stage0_format_effect_temp07']:.3f}")
    print(f"Temp effect (with format): {effects['stage0']['stage0_temp_effect_with_format']:.3f}")
    print(f"Temp effect (no format): {effects['stage0']['stage0_temp_effect_no_format']:.3f}")
    print(f"Interaction effect: {effects['stage0']['stage0_interaction']:.3f}")
    print(f"Optimal condition: {effects['stage0']['stage0_optimal_condition']}")
    print(f"Optimal semantic rate: {effects['stage0']['stage0_optimal_semantic_rate']:.3f}")
    
    print("\n--- Stage 1 Effects ---")
    print(f"Format effect (temp 0.0): {effects['stage1']['stage1_format_effect_temp0']:.3f}")
    print(f"Format effect (temp 0.7): {effects['stage1']['stage1_format_effect_temp07']:.3f}")
    print(f"Temp effect (with format): {effects['stage1']['stage1_temp_effect_with_format']:.3f}")
    print(f"Temp effect (no format): {effects['stage1']['stage1_temp_effect_no_format']:.3f}")
    print(f"Interaction effect: {effects['stage1']['stage1_interaction']:.3f}")
    print(f"Optimal condition: {effects['stage1']['stage1_optimal_condition']}")
    print(f"Optimal semantic rate: {effects['stage1']['stage1_optimal_semantic_rate']:.3f}")
    
    # Generate LaTeX table
    print("\n[5] Generating LaTeX table...")
    latex_table = generate_latex_table(stage0_stats, stage1_stats)
    
    # Save outputs
    output_dir = Path("docs/figures_ablation_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save LaTeX table
    latex_path = output_dir / "ablation_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\n✓ LaTeX table saved to: {latex_path}")
    
    # Save analysis summary
    summary_path = output_dir / "ablation_analysis_summary.md"
    with open(summary_path, 'w') as f:
        f.write(generate_analysis_summary(
            stage0_stats, stage1_stats, stage0_quality, stage1_quality, effects
        ))
    print(f"✓ Analysis summary saved to: {summary_path}")
    
    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(latex_table)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

def generate_analysis_summary(
    stage0_stats: List[Dict],
    stage1_stats: List[Dict],
    stage0_quality: Dict,
    stage1_quality: Dict,
    effects: Dict
) -> str:
    """Generate markdown analysis summary."""
    
    lines = [
        "# Ablation Study Analysis Summary",
        "",
        "## Data Quality Assessment",
        "",
        "### Stage 0",
        f"- **Samples per condition:** {stage0_quality['n_samples_per_condition']}",
        f"- **Parse success rates:** {stage0_quality['parse_rates']}",
        ""
    ]
    
    if stage0_quality['warnings']:
        lines.append("**Warnings:**")
        for warning in stage0_quality['warnings']:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")
    
    lines.extend([
        "### Stage 1",
        f"- **Samples per condition:** {stage1_quality['n_samples_per_condition']}",
        f"- **Parse success rates:** {stage1_quality['parse_rates']}",
        ""
    ])
    
    if stage1_quality['warnings']:
        lines.append("**Warnings:**")
        for warning in stage1_quality['warnings']:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")
    
    lines.extend([
        "## Key Findings",
        "",
        "### Stage 0",
        "",
        f"**Format Prompting Effect:**",
        f"- At temperature 0.0: {effects['stage0']['stage0_format_effect_temp0']:.1%} improvement",
        f"- At temperature 0.7: {effects['stage0']['stage0_format_effect_temp07']:.1%} improvement",
        "",
        f"**Temperature Effect:**",
        f"- With format prompting: {effects['stage0']['stage0_temp_effect_with_format']:.1%} change",
        f"- Without format prompting: {effects['stage0']['stage0_temp_effect_no_format']:.1%} change",
        "",
        f"**Interaction Effect:** {effects['stage0']['stage0_interaction']:.3f}",
        "",
        f"**Optimal Configuration:** {effects['stage0']['stage0_optimal_condition']} "
        f"(Semantic Rate: {effects['stage0']['stage0_optimal_semantic_rate']:.1%})",
        "",
        "### Stage 1",
        "",
        f"**Format Prompting Effect:**",
        f"- At temperature 0.0: {effects['stage1']['stage1_format_effect_temp0']:.1%} improvement",
        f"- At temperature 0.7: {effects['stage1']['stage1_format_effect_temp07']:.1%} improvement",
        "",
        f"**Temperature Effect:**",
        f"- With format prompting: {effects['stage1']['stage1_temp_effect_with_format']:.1%} change",
        f"- Without format prompting: {effects['stage1']['stage1_temp_effect_no_format']:.1%} change",
        "",
        f"**Interaction Effect:** {effects['stage1']['stage1_interaction']:.3f}",
        "",
        f"**Optimal Configuration:** {effects['stage1']['stage1_optimal_condition']} "
        f"(Semantic Rate: {effects['stage1']['stage1_optimal_semantic_rate']:.1%})",
        "",
        "### Stage 0 vs Stage 1 Comparison",
        "",
        f"- Format effect difference (temp 0.0): {effects['comparison']['stage0_vs_stage1_format_effect_temp0']:.3f}",
        f"- Format effect difference (temp 0.7): {effects['comparison']['stage0_vs_stage1_format_effect_temp07']:.3f}",
        f"- Optimal semantic rate difference: {effects['comparison']['stage0_vs_stage1_optimal_semantic_rate']:.3f}",
        "",
        "## Recommendations",
        "",
        "### Optimal Configuration",
        "",
        f"- **Stage 0:** Use `{effects['stage0']['stage0_optimal_condition']}` for best semantic performance",
        f"- **Stage 1:** Use `{effects['stage1']['stage1_optimal_condition']}` for best semantic performance",
        "",
        "### Paper Recommendations",
        "",
        "1. **Note the format_rate issue:** All conditions show 0% format rate, which requires investigation.",
        "2. **Focus on semantic_rate:** Given format_rate issues, semantic_rate is the primary metric for comparison.",
        "3. **Temperature sensitivity:** Stage 1 shows different temperature sensitivity than Stage 0.",
        "4. **Format prompting value:** Format prompting consistently improves semantic performance in both stages.",
        ""
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    main()
