#!/usr/bin/env python3
"""
Publication-Quality Result Visualization

Generates IEEE/ACM-compliant figures for WiFi CSI sensing papers:
1. ROC curves with confidence bands
2. Link selection comparison bar charts
3. Temporal validation heatmaps
4. Statistical significance annotations

Usage:
    python plot_results.py \
        --results /path/to/evaluation_results.json \
        --output-dir /path/to/figures

Requirements:
    - matplotlib, seaborn, numpy
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Publication style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.8),  # Single column IEEE
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
})


def plot_auc_comparison(results: List[Dict], output_path: Path):
    """
    Bar chart comparing AUC across link configurations.

    Args:
        results: List of evaluation results
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(4.5, 3))

    configs = [r['config'] for r in results]
    means = [r['auc_mean'] for r in results]
    stds = [r['auc_std'] for r in results]

    # Color coding
    colors = []
    for c in configs:
        if c.startswith('top_'):
            colors.append('#2ecc71')  # Green for top-k
        elif c == 'all_links':
            colors.append('#3498db')  # Blue for all
        else:
            colors.append('#95a5a6')  # Gray for random

    bars = ax.bar(range(len(configs)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5)

    # Annotations
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.2f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0)
    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0.4, 1.05)
    ax.set_title('Presence Detection: Link Selection')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_fold_comparison(results: List[Dict], output_path: Path):
    """
    Box plot of per-fold AUC scores.

    Args:
        results: List of evaluation results
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    data = []
    labels = []
    for r in results:
        if 'aucs' in r:
            data.append(r['aucs'])
            labels.append(r['config'].replace('_', '\n'))

    if not data:
        print("No per-fold data available for box plot")
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color the boxes
    for i, patch in enumerate(bp['boxes']):
        if 'top' in labels[i]:
            patch.set_facecolor('#2ecc71')
        elif 'all' in labels[i]:
            patch.set_facecolor('#3498db')
        else:
            patch.set_facecolor('#95a5a6')
        patch.set_alpha(0.7)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Per-Fold Performance Distribution')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_temporal_heatmap(results: List[Dict], output_path: Path):
    """
    Heatmap of AUC by test day and configuration.

    Args:
        results: List of evaluation results
        output_path: Path to save figure
    """
    # Build matrix
    configs = []
    folds = set()
    data = {}

    for r in results:
        config = r['config']
        configs.append(config)
        data[config] = {}

        if 'fold_results' in r:
            for fold in r['fold_results']:
                folds.add(fold['fold'])
                data[config][fold['fold']] = fold['auc']

    if not folds:
        print("No fold-level data available for heatmap")
        return

    folds = sorted(folds)
    matrix = np.zeros((len(configs), len(folds)))

    for i, config in enumerate(configs):
        for j, fold in enumerate(folds):
            matrix[i, j] = data[config].get(fold, np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[f[-5:] for f in folds],  # Just day portion
                yticklabels=[c.replace('_', ' ') for c in configs],
                vmin=0.5, vmax=1.0, ax=ax,
                cbar_kws={'label': 'AUC-ROC'})

    ax.set_xlabel('Test Day')
    ax.set_ylabel('Configuration')
    ax.set_title('Temporal Cross-Validation Performance')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_statistical_significance(stats: Dict, output_path: Path):
    """
    Forest plot showing effect sizes and confidence intervals.

    Args:
        stats: Statistical analysis results
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Extract data
    mean_diff = stats['bootstrap_ci']['mean_diff']
    ci_lower = stats['bootstrap_ci']['ci_lower']
    ci_upper = stats['bootstrap_ci']['ci_upper']
    p_value = stats['wilcoxon']['p_value']
    cohens_d = stats['effect_size']['cohens_d']

    # Plot
    ax.errorbar([0], [mean_diff], xerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]],
                fmt='o', color='#2ecc71', markersize=10, capsize=5, capthick=2, linewidth=2)

    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='No difference')

    # Annotations
    ax.text(0.05, 0.95, f"Mean: {mean_diff:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.text(0.95, 0.95, f"p = {p_value:.4f}\nd = {cohens_d:.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax.set_xlabel('AUC Improvement (Top-k vs All)')
    ax.set_yticks([])
    ax.set_title('Statistical Significance of Link Selection')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--results', type=Path, required=True,
                        help='JSON file with evaluation results')
    parser.add_argument('--output-dir', type=Path, default=Path('./figures'),
                        help='Output directory for figures')
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    results = data.get('results', [])
    stats = data.get('statistical_analysis', {})

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)

    # Generate figures
    print("\n[1/4] AUC comparison bar chart...")
    plot_auc_comparison(results, args.output_dir / 'auc_comparison.pdf')
    plot_auc_comparison(results, args.output_dir / 'auc_comparison.png')

    print("\n[2/4] Per-fold box plot...")
    plot_fold_comparison(results, args.output_dir / 'fold_boxplot.pdf')
    plot_fold_comparison(results, args.output_dir / 'fold_boxplot.png')

    print("\n[3/4] Temporal heatmap...")
    plot_temporal_heatmap(results, args.output_dir / 'temporal_heatmap.pdf')
    plot_temporal_heatmap(results, args.output_dir / 'temporal_heatmap.png')

    if stats:
        print("\n[4/4] Statistical significance plot...")
        plot_statistical_significance(stats, args.output_dir / 'significance.pdf')
        plot_statistical_significance(stats, args.output_dir / 'significance.png')

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
