"""Visualization utilities for vyapti benchmark evaluation results.

Creates publication-quality plots for the diagnosis paper.
Outputs PDF figures suitable for LaTeX inclusion.
"""

import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CATEGORIES = ["savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalatita"]
CAT_SHORT = ["SAV", "VIR", "PRA", "SAD", "KAL"]
CAT_LABELS = [
    "Savyabhicāra\n(Erratic)",
    "Viruddha\n(Contradictory)",
    "Prakaraṇasama\n(Question-\nbegging)",
    "Sādhyasama\n(Unproven)",
    "Kālātīta\n(Temporal)",
]

# Model display names for publication
MODEL_DISPLAY = {
    "deepseek_8b_base": "DeepSeek-R1-8B",
    "base_with_cot": "DeepSeek-R1-8B + CoT",
    "base_with_nyaya_template": "DeepSeek-R1-8B + Nyāya",
    "stage1_pramana": "Pramāṇa Stage 1",
    "llama_3b_base": "Llama-3.2-3B",
    "stage0_pramana": "Pramāṇa Stage 0",
}

# Color palette
COLORS = {
    "probe": "#E74C3C",       # Red
    "control": "#27AE60",     # Green
    "significant": "#2C3E50", # Dark blue
    "nonsig": "#BDC3C7",      # Gray
    "pramana": "#8E44AD",     # Purple
    "base": "#3498DB",        # Blue
}


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z * z) / (4.0 * n * n)))
    return max(0.0, center - margin), min(1.0, center + margin)


def _setup_style():
    """Configure matplotlib for publication quality."""
    rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def plot_probe_vs_control_heatmap(summary: dict, output_path: Path) -> None:
    """Plot probe vs control accuracy heatmap by model and category."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    _setup_style()
    models = list(summary.keys())
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]
    n_models = len(models)
    n_cats = len(CATEGORIES)

    # Build gap matrix (control - probe) for each model/category
    gap_data = np.zeros((n_models, n_cats))
    probe_data = np.zeros((n_models, n_cats))
    control_data = np.zeros((n_models, n_cats))

    for i, model in enumerate(models):
        by_cat = summary[model].get("by_category", {})
        for j, cat in enumerate(CATEGORIES):
            cat_data = by_cat.get(cat, {})
            pt = cat_data.get("probe_total", 1)
            ct = cat_data.get("control_total", 1)
            p_acc = cat_data.get("probe_correct", 0) / max(pt, 1)
            c_acc = cat_data.get("control_correct", 0) / max(ct, 1)
            probe_data[i, j] = p_acc
            control_data[i, j] = c_acc
            gap_data[i, j] = c_acc - p_acc

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 1]})

    for ax, data, title, cmap in [
        (axes[0], probe_data, "Probe Accuracy", "YlOrRd_r"),
        (axes[1], control_data, "Control Accuracy", "YlGn"),
        (axes[2], gap_data, "Performance Gap\n(Control − Probe)", "RdBu_r"),
    ]:
        vmin, vmax = (0, 1) if "Gap" not in title else (-0.5, 0.5)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_cats))
        ax.set_xticklabels(CAT_SHORT, fontsize=9)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(display_names, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Annotate cells
        for ii in range(n_models):
            for jj in range(n_cats):
                val = data[ii, jj]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(jj, ii, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Vyāpti Probe Benchmark: Category-wise Performance", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "probe_vs_control_heatmap.pdf", bbox_inches="tight")
    plt.savefig(output_path / "probe_vs_control_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: probe_vs_control_heatmap.pdf")


def plot_hetvabhasa_distribution(summary: dict, output_path: Path) -> None:
    """Plot Hetvabhasa failure distribution by model."""
    if not HAS_MPL:
        return

    _setup_style()

    # Build per-model distribution
    models = list(summary.keys())
    htypes = CATEGORIES  # Use canonical order
    model_dists = {}

    for model in models:
        dist = summary[model].get("hetvabhasa_distribution", {})
        model_dists[model] = {h: dist.get(h, 0) for h in htypes}

    n_models = len(models)
    n_types = len(htypes)
    x = np.arange(n_types)
    width = 0.12

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        vals = [model_dists[model][h] for h in htypes]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_DISPLAY.get(model, model), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=9)
    ax.set_ylabel("Failure Count")
    ax.set_title("Hetvābhāsa Classification Distribution by Model", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "hetvabhasa_distribution.pdf", bbox_inches="tight")
    plt.savefig(output_path / "hetvabhasa_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: hetvabhasa_distribution.pdf")


def plot_bootstrap_ci(comparisons: list, output_path: Path) -> None:
    """Plot bootstrap confidence intervals forest plot."""
    if not HAS_MPL:
        return

    _setup_style()

    # Filter to comparisons with CIs
    valid = [c for c in comparisons if c.n_samples > 0 and c.ci_lower != c.ci_upper]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    names = [c.name for c in valid]
    diffs = [c.difference for c in valid]
    ci_los = [c.ci_lower for c in valid]
    ci_his = [c.ci_upper for c in valid]
    colors = [COLORS["significant"] if c.significant else COLORS["nonsig"] for c in valid]

    y_pos = np.arange(len(names))

    for i, (d, lo, hi, col, name) in enumerate(zip(diffs, ci_los, ci_his, colors, names)):
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.5, solid_capstyle="round")
        ax.plot(d, i, "o", color=col, markersize=8, zorder=5)
        ax.annotate(f"{d:+.1%}", (d, i), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Accuracy Difference")
    ax.set_title("Key Comparisons with 95% Bootstrap CI", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    if any(getattr(c, "p_value_approx", 0.0) < 0 for c in comparisons):
        ax.text(
            0.02,
            0.02,
            "Note: descriptive comparisons are omitted from CI plot.",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(output_path / "bootstrap_ci_comparisons.pdf", bbox_inches="tight")
    plt.savefig(output_path / "bootstrap_ci_comparisons.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: bootstrap_ci_comparisons.pdf")


def plot_overall_accuracy(summary: dict, output_path: Path) -> None:
    """Plot overall accuracy comparison: probe vs control for each model."""
    if not HAS_MPL:
        return

    _setup_style()

    models = list(summary.keys())
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]
    probe_accs = [summary[m].get("probe_accuracy", 0) for m in models]
    ctrl_accs = [summary[m].get("control_accuracy", 0) for m in models]
    probe_yerr_low = []
    probe_yerr_high = []
    ctrl_yerr_low = []
    ctrl_yerr_high = []

    for model in models:
        by_cat = summary[model].get("by_category", {})
        probe_total = sum(cat.get("probe_total", 0) for cat in by_cat.values())
        probe_correct = sum(cat.get("probe_correct", 0) for cat in by_cat.values())
        control_total = sum(cat.get("control_total", 0) for cat in by_cat.values())
        control_correct = sum(cat.get("control_correct", 0) for cat in by_cat.values())

        p_lo, p_hi = _wilson_ci(probe_correct, probe_total)
        c_lo, c_hi = _wilson_ci(control_correct, control_total)
        p_hat = summary[model].get("probe_accuracy", 0.0)
        c_hat = summary[model].get("control_accuracy", 0.0)

        probe_yerr_low.append(max(0.0, p_hat - p_lo))
        probe_yerr_high.append(max(0.0, p_hi - p_hat))
        ctrl_yerr_low.append(max(0.0, c_hat - c_lo))
        ctrl_yerr_high.append(max(0.0, c_hi - c_hat))

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, probe_accs, width, label="Probe (Vyāpti-requiring)",
                   color=COLORS["probe"], alpha=0.85,
                   yerr=np.array([probe_yerr_low, probe_yerr_high]), capsize=3,
                   error_kw={"elinewidth": 1, "alpha": 0.9})
    bars2 = ax.bar(x + width/2, ctrl_accs, width, label="Control",
                   color=COLORS["control"], alpha=0.85,
                   yerr=np.array([ctrl_yerr_low, ctrl_yerr_high]), capsize=3,
                   error_kw={"elinewidth": 1, "alpha": 0.9})

    # Annotate with values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe vs. Control Accuracy by Model", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "overall_accuracy.pdf", bbox_inches="tight")
    plt.savefig(output_path / "overall_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: overall_accuracy.pdf")


def plot_tier_comparison(summary: dict, output_path: Path) -> None:
    """Plot multi-tier score comparison across models."""
    if not HAS_MPL:
        return

    _setup_style()

    models = list(summary.keys())
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]
    tier_names = ["T1: Answer", "T2: Structure", "T3: Vyāpti", "T4: Z3", "T5: Hetvābhāsa"]

    data = np.zeros((len(models), 5))
    for i, model in enumerate(models):
        tavg = summary[model].get("tier_averages", {})
        for t in range(5):
            data[i, t] = tavg.get(f"tier_{t+1}", 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tier_names))
    width = 0.12

    for i, (model, dname) in enumerate(zip(models, display_names)):
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, data[i], width, label=dname, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tier_names, fontsize=9)
    ax.set_ylabel("Average Score")
    ax.set_title("5-Tier Evaluation Scores by Model", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "tier_comparison.pdf", bbox_inches="tight")
    plt.savefig(output_path / "tier_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: tier_comparison.pdf")


def generate_all_plots(summary: dict, comparisons: list, output_path: Path) -> None:
    """Generate all publication-quality visualization plots."""
    output_path.mkdir(parents=True, exist_ok=True)
    plot_probe_vs_control_heatmap(summary, output_path)
    plot_hetvabhasa_distribution(summary, output_path)
    plot_bootstrap_ci(comparisons, output_path)
    plot_overall_accuracy(summary, output_path)
    plot_tier_comparison(summary, output_path)
