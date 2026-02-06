#!/usr/bin/env python3
"""
Generate publication-quality PDF figures for NeurIPS paper submission.

Converts existing plot data to high-quality PDF format with LaTeX-friendly settings.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np

# Set matplotlib to use PDF backend and configure for NeurIPS publication quality
matplotlib.use('PDF')
plt.rcParams.update({
    'font.size': 9,  # Base font size (8-10pt range)
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'axes.labelsize': 10,  # Axis labels: 9-10pt
    'axes.titlesize': 10,  # Titles: 10pt
    'xtick.labelsize': 8,  # Tick labels: 8pt
    'ytick.labelsize': 8,  # Tick labels: 8pt
    'legend.fontsize': 9,  # Legend: 9pt
    'figure.titlesize': 10,
    'text.usetex': False,  # Set to True if LaTeX is installed
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,  # Thicker axes
    'grid.linewidth': 0.3,  # Lighter gridlines
    'lines.linewidth': 2.0,  # Thicker lines (was 1.5)
    'patch.linewidth': 0.8,  # Thicker bar edges
})

# Color-blind friendly palette (from ColorBrewer)
COLORS = {
    'blue': '#377eb8',
    'orange': '#ff7f00',
    'green': '#4daf4a',
    'red': '#e41a1c',
    'purple': '#984ea3',
    'brown': '#a65628',
    'pink': '#f781bf',
    'gray': '#999999',
    'yellow': '#dede00',
    'cyan': '#00ffff',
}

# Base directories
BASE_DIR = Path(__file__).parent.parent
FIGURES_DIR = BASE_DIR / 'docs' / 'paper_v2' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Source directories
STAGE0_DIR = BASE_DIR / 'docs' / 'figures_stage0_v2'
STAGE1_DIR = BASE_DIR / 'docs' / 'figures_stage1_v2'
ABLATION_DIR = BASE_DIR / 'docs' / 'figures_ablation_v1'
COMBINED_DIR = BASE_DIR / 'docs' / 'figures_combined_v1'


def generate_loss_curves_stage0():
    """Generate Stage 0 training and evaluation loss curves."""
    print("Generating Stage 0 loss curves...")
    
    # Load data
    train_df = pd.read_csv(STAGE0_DIR / 'stage0_train_loss.csv')
    eval_df = pd.read_csv(STAGE0_DIR / 'stage0_eval_loss.csv')
    
    # Create figure - Single column: 3.3" width, 2.5-3" height
    fig, ax = plt.subplots(figsize=(3.3, 2.5), dpi=300)
    
    # Plot training loss - thicker lines, larger markers
    ax.plot(train_df['epoch'], train_df['loss'], 
            color=COLORS['blue'], label='Training Loss', linewidth=2.0, marker='o', markersize=5)
    
    # Plot evaluation loss
    ax.plot(eval_df['epoch'], eval_df['eval_loss'], 
            color=COLORS['orange'], label='Validation Loss', linewidth=2.0, 
            marker='s', markersize=5)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Stage 0 Training Progress', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)  # Lighter gridlines
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'stage0_loss_curves.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path} (Size: 3.3\" × 2.5\")")
    return output_path


def generate_loss_curves_stage1():
    """Generate Stage 1 training and evaluation loss curves."""
    print("Generating Stage 1 loss curves...")
    
    # Load data
    train_df = pd.read_csv(STAGE1_DIR / 'stage1_train_loss.csv')
    eval_df = pd.read_csv(STAGE1_DIR / 'stage1_eval_loss.csv')
    
    # Create figure - Single column: 3.3" width, 2.5-3" height
    fig, ax = plt.subplots(figsize=(3.3, 2.5), dpi=300)
    
    # Plot training loss - thicker lines, larger markers
    ax.plot(train_df['epoch'], train_df['loss'], 
            color=COLORS['blue'], label='Training Loss', linewidth=2.0, marker='o', markersize=5)
    
    # Plot evaluation loss
    ax.plot(eval_df['epoch'], eval_df['eval_loss'], 
            color=COLORS['orange'], label='Validation Loss', linewidth=2.0,
            marker='s', markersize=5)
    
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Stage 1 Training Progress', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)  # Lighter gridlines
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'stage1_loss_curves.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path} (Size: 3.3\" × 2.5\")")
    return output_path


def generate_ablation_stage0():
    """Generate Stage 0 ablation study plot."""
    print("Generating Stage 0 ablation study...")
    
    # Load data
    with open(ABLATION_DIR / 'stage0_ablation_summary.json', 'r') as f:
        data = json.load(f)
    
    # Parse conditions
    conditions = []
    format_rates = []
    semantic_rates = []
    
    for item in data:
        conditions.append(item['condition'])
        format_rates.append(item['format_rate'])
        semantic_rates.append(item['semantic_rate'])
    
    # Create figure with subplots - Double column: 6.8" width, 3-4" height
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0), dpi=300)
    
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    # Format adherence rates
    bars1 = ax1.bar(x_pos - width/2, format_rates, width, 
                    label='Format Adherence', color=COLORS['blue'], alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Condition', fontsize=10)
    ax1.set_ylabel('Rate', fontsize=10)
    ax1.set_title('Format Adherence Rate', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.3)
    ax1.legend(fontsize=9, framealpha=0.9)
    
    # Semantic correctness rates
    bars2 = ax2.bar(x_pos - width/2, semantic_rates, width,
                    label='Semantic Correctness', color=COLORS['green'], alpha=0.8, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Condition', fontsize=10)
    ax2.set_ylabel('Rate', fontsize=10)
    ax2.set_title('Semantic Correctness Rate', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.3)
    ax2.legend(fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'stage0_ablation.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path} (Size: 6.8\" × 3.0\")")
    return output_path


def generate_ablation_stage1():
    """Generate Stage 1 ablation study plot."""
    print("Generating Stage 1 ablation study...")
    
    # Load data
    with open(ABLATION_DIR / 'stage1_ablation_summary.json', 'r') as f:
        data = json.load(f)
    
    # Parse conditions
    conditions = []
    format_rates = []
    semantic_rates = []
    
    for item in data:
        conditions.append(item['condition'])
        format_rates.append(item['format_rate'])
        semantic_rates.append(item['semantic_rate'])
    
    # Create figure with subplots - Double column: 6.8" width, 3-4" height
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0), dpi=300)
    
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    # Format adherence rates
    bars1 = ax1.bar(x_pos - width/2, format_rates, width,
                    label='Format Adherence', color=COLORS['blue'], alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Condition', fontsize=10)
    ax1.set_ylabel('Rate', fontsize=10)
    ax1.set_title('Format Adherence Rate', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.3)
    ax1.legend(fontsize=9, framealpha=0.9)
    
    # Semantic correctness rates
    bars2 = ax2.bar(x_pos - width/2, semantic_rates, width,
                    label='Semantic Correctness', color=COLORS['green'], alpha=0.8, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Condition', fontsize=10)
    ax2.set_ylabel('Rate', fontsize=10)
    ax2.set_title('Semantic Correctness Rate', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.3)
    ax2.legend(fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'stage1_ablation.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path} (Size: 6.8\" × 3.0\")")
    return output_path


def generate_cross_stage_comparison():
    """Generate cross-stage metrics comparison plot."""
    print("Generating cross-stage comparison...")
    
    # Load data
    df = pd.read_csv(COMBINED_DIR / 'stage_combined_metrics.csv')
    
    # Create figure - Double column: 6.8" width, 3-4" height
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0), dpi=300)
    
    stages = df['stage'].values
    format_rates = df['format_rate'].values
    semantic_rates = df['semantic_rate'].values
    
    x_pos = np.arange(len(stages))
    width = 0.6
    
    # Format adherence comparison
    bars1 = ax1.bar(x_pos, format_rates, width, 
                    color=[COLORS['blue'], COLORS['green']], alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Stage', fontsize=10)
    ax1.set_ylabel('Format Adherence Rate', fontsize=10)
    ax1.set_title('Format Adherence Across Stages', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stages, fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, format_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Semantic correctness comparison
    bars2 = ax2.bar(x_pos, semantic_rates, width,
                    color=[COLORS['blue'], COLORS['green']], alpha=0.8, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Stage', fontsize=10)
    ax2.set_ylabel('Semantic Correctness Rate', fontsize=10)
    ax2.set_title('Semantic Correctness Across Stages', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stages, fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, semantic_rates)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'cross_stage_metrics.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path} (Size: 6.8\" × 3.0\")")
    return output_path


def generate_parse_errors():
    """Generate parse error breakdown plot combining Stage 0 and Stage 1."""
    print("Generating parse error breakdown...")
    
    # Load data from both stages
    stage0_df = pd.read_csv(STAGE0_DIR / 'stage0_parse_error_breakdown.csv')
    stage1_df = pd.read_csv(STAGE1_DIR / 'stage1_parse_error_breakdown.csv')
    
    # Combine and aggregate errors
    stage0_df['stage'] = 'Stage 0'
    stage1_df['stage'] = 'Stage 1'
    combined_df = pd.concat([stage0_df, stage1_df], ignore_index=True)
    
    # Clean error names (remove regex patterns for readability)
    combined_df['error_clean'] = combined_df['error'].str.replace(r'\\s.*', '', regex=True)
    combined_df['error_clean'] = combined_df['error_clean'].str.replace('Missing required section: ', '')
    combined_df['error_clean'] = combined_df['error_clean'].str.replace('Missing required field: ', '')
    
    # Aggregate by error type and stage
    error_summary = combined_df.groupby(['error_clean', 'stage'])['count'].sum().reset_index()
    
    # Get unique error types
    error_types = error_summary['error_clean'].unique()
    
    # Create figure - Double column: 6.8" width, 3-4" height
    fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=300)
    
    x_pos = np.arange(len(error_types))
    width = 0.35
    
    # Get counts for each stage
    stage0_counts = []
    stage1_counts = []
    for err_type in error_types:
        stage0_val = error_summary[(error_summary['error_clean'] == err_type) & 
                                   (error_summary['stage'] == 'Stage 0')]['count'].sum()
        stage1_val = error_summary[(error_summary['error_clean'] == err_type) & 
                                   (error_summary['stage'] == 'Stage 1')]['count'].sum()
        stage0_counts.append(stage0_val)
        stage1_counts.append(stage1_val)
    
    bars1 = ax.bar(x_pos - width/2, stage0_counts, width,
                   label='Stage 0', color=COLORS['blue'], alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos + width/2, stage1_counts, width,
                   label='Stage 1', color=COLORS['orange'], alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Error Type', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Parse Error Distribution', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(error_types, rotation=45, ha='right', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y', linewidth=0.3)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'parse_errors.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path} (Size: 6.8\" × 3.5\")")
    return output_path


def main():
    """Generate all PDF figures."""
    print("=" * 60)
    print("Generating PDF figures for NeurIPS paper submission")
    print("=" * 60)
    
    generated_files = []
    
    # Generate all plots
    try:
        generated_files.append(('stage0_loss_curves.pdf', 
                              'Stage 0 training and validation loss curves over 30 epochs'))
        generate_loss_curves_stage0()
        
        generated_files.append(('stage1_loss_curves.pdf',
                              'Stage 1 training and validation loss curves over 10 epochs'))
        generate_loss_curves_stage1()
        
        generated_files.append(('stage0_ablation.pdf',
                              'Stage 0 ablation study: format prompting × temperature effects'))
        generate_ablation_stage0()
        
        generated_files.append(('stage1_ablation.pdf',
                              'Stage 1 ablation study: format prompting × temperature effects'))
        generate_ablation_stage1()
        
        generated_files.append(('cross_stage_metrics.pdf',
                              'Cross-stage comparison of format adherence and semantic correctness'))
        generate_cross_stage_comparison()
        
        generated_files.append(('parse_errors.pdf',
                              'Parse error breakdown showing distribution of missing sections/fields'))
        generate_parse_errors()
        
    except Exception as e:
        print(f"\nError generating figures: {e}")
        raise
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary of Generated PDF Figures")
    print("=" * 60)
    print(f"\nOutput directory: {FIGURES_DIR}\n")
    
    # Figure size mapping
    figure_sizes = {
        'stage0_loss_curves.pdf': '3.3" × 2.5" (Single column)',
        'stage1_loss_curves.pdf': '3.3" × 2.5" (Single column)',
        'stage0_ablation.pdf': '6.8" × 3.0" (Double column)',
        'stage1_ablation.pdf': '6.8" × 3.0" (Double column)',
        'cross_stage_metrics.pdf': '6.8" × 3.0" (Double column)',
        'parse_errors.pdf': '6.8" × 3.5" (Double column)',
    }
    
    for filename, description in generated_files:
        filepath = FIGURES_DIR / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            fig_size = figure_sizes.get(filename, 'Unknown')
            print(f"✓ {filename}")
            print(f"  Path: {filepath}")
            print(f"  Description: {description}")
            print(f"  Size: {size_kb:.1f} KB")
            print(f"  Dimensions: {fig_size}")
            print(f"  DPI: 300 (publication quality)\n")
        else:
            print(f"✗ {filename} - FILE NOT FOUND\n")
    
    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
    print("\nNeurIPS Formatting Applied:")
    print("  • Single column plots: 3.3\" width")
    print("  • Double column plots: 6.8\" width")
    print("  • Font sizes: 8-10pt (readable when printed)")
    print("  • Line width: 2.0 (thicker, more visible)")
    print("  • Marker size: 5 (larger data points)")
    print("  • Gridlines: lighter (alpha=0.2)")
    print("  • DPI: 300 (publication quality)")
    print("=" * 60)


if __name__ == '__main__':
    main()
