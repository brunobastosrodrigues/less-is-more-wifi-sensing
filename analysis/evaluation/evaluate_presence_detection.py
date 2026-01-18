#!/usr/bin/env python3
"""
Presence Detection Evaluation

Evaluates WiFi CSI-based presence detection with:
1. Link selection comparison (Top-k vs All-links vs Random)
2. Temporal cross-validation (Leave-One-Day-Out)
3. Statistical significance testing
4. Publication-quality figures

Usage:
    python evaluate_presence_detection.py \
        --data-dir /path/to/csi_data \
        --labels /path/to/labels.csv \
        --output-dir /path/to/results

Requirements:
    - numpy, scipy, scikit-learn, pandas, matplotlib
"""

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
from scipy import stats as scipy_stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from statistical_tests import wilcoxon_test, cohens_d, bootstrap_ci, full_statistical_analysis
from reproducibility import set_all_seeds

warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(csi_data: list) -> Optional[np.ndarray]:
    """
    Extract statistical features from raw CSI data.

    Args:
        csi_data: Raw I/Q interleaved CSI values

    Returns:
        Feature vector or None if insufficient data
    """
    if not csi_data or len(csi_data) < 10:
        return None

    csi = np.array(csi_data, dtype=np.float32)

    # Parse I/Q to amplitude
    if len(csi) >= 128:
        i_vals = csi[0::2][:64]
        q_vals = csi[1::2][:64]
        amp = np.sqrt(i_vals**2 + q_vals**2)
    else:
        amp = np.abs(csi)

    if len(amp) < 5:
        return None

    # Statistical features
    return np.array([
        np.mean(amp),                                    # Mean amplitude
        np.std(amp),                                     # Standard deviation
        np.var(amp),                                     # Variance
        np.max(amp) - np.min(amp),                       # Range
        np.mean(np.abs(np.diff(amp))) if len(amp) > 1 else 0,  # Mean change
        np.percentile(amp, 75) - np.percentile(amp, 25),       # IQR
        np.median(amp),                                  # Median
    ])


def load_csi_data(data_dir: Path, dates: List[str], labels_df: pd.DataFrame,
                  samples_per_hour: int = 300) -> Dict:
    """
    Load CSI data organized by link and day.

    Args:
        data_dir: Directory containing CSI data
        dates: List of date strings (YYYY-MM-DD)
        labels_df: DataFrame with columns [date, hour, occupied/lr_occupied]
        samples_per_hour: Maximum samples per hour

    Returns:
        Dictionary: link_id -> day -> {X, y}
    """
    link_data = defaultdict(lambda: defaultdict(lambda: {'X': [], 'y': []}))

    for date_str in dates:
        day_dir = data_dir / date_str
        if not day_dir.exists():
            print(f"  Skipping {date_str}: directory not found")
            continue

        day_labels = labels_df[labels_df['date'] == date_str]
        if day_labels.empty:
            print(f"  Skipping {date_str}: no labels")
            continue

        print(f"  Loading {date_str}...")

        for _, row in day_labels.iterrows():
            hour = row['hour']
            # Support both 'occupied' and 'lr_occupied' column names
            label = row.get('occupied', row.get('lr_occupied', 0))

            # Find CSI files for this hour
            # Files are named csi_HHMMSS.jsonl where HH is the hour
            hour_files = list(day_dir.glob(f"csi_{hour:02d}*.jsonl*"))

            for csi_file in hour_files:
                try:
                    import gzip
                    opener = gzip.open if csi_file.suffix == '.gz' else open

                    with opener(csi_file, 'rt') as f:
                        for line_num, line in enumerate(f):
                            if line_num >= samples_per_hour:
                                break

                            try:
                                packet = json.loads(line)
                                tx = packet.get('tx_mac', '')
                                rx = packet.get('rx_mac', '')
                                csi = packet.get('csi_data', [])

                                if not tx or not rx:
                                    continue

                                features = extract_features(csi)
                                if features is not None:
                                    link_id = f"{tx}->{rx}"
                                    link_data[link_id][date_str]['X'].append(features)
                                    link_data[link_id][date_str]['y'].append(label)

                            except json.JSONDecodeError:
                                continue

                except Exception as e:
                    print(f"    Error reading {csi_file}: {e}")

    return link_data


# =============================================================================
# LINK SELECTION STRATEGIES
# =============================================================================

def select_links_by_variance(link_data: Dict, k: int) -> List[str]:
    """Select top-k links by label variance (most informative)."""
    link_scores = {}

    for link_id, day_data in link_data.items():
        all_y = []
        for day, data in day_data.items():
            all_y.extend(data['y'])

        if len(all_y) > 10:
            # Variance of labels (0.25 max for binary)
            link_scores[link_id] = np.var(all_y)

    sorted_links = sorted(link_scores.items(), key=lambda x: x[1], reverse=True)
    return [link for link, _ in sorted_links[:k]]


def select_random_links(link_data: Dict, k: int, seed: int = 42) -> List[str]:
    """Select k random links."""
    np.random.seed(seed)
    links = list(link_data.keys())
    return list(np.random.choice(links, min(k, len(links)), replace=False))


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_link_configuration(
    link_data: Dict,
    selected_links: List[str],
    dates: List[str],
    config_name: str = "config"
) -> Dict:
    """
    Evaluate a link configuration using Leave-One-Day-Out cross-validation.

    Args:
        link_data: Full dataset
        selected_links: Links to use
        dates: List of dates for CV folds
        config_name: Name for this configuration

    Returns:
        Dictionary with per-fold and aggregate metrics
    """
    fold_results = []

    for test_day in dates:
        train_days = [d for d in dates if d != test_day]

        # Aggregate training data from selected links
        X_train, y_train = [], []
        X_test, y_test = [], []

        for link_id in selected_links:
            if link_id not in link_data:
                continue

            for day in train_days:
                if day in link_data[link_id]:
                    X_train.extend(link_data[link_id][day]['X'])
                    y_train.extend(link_data[link_id][day]['y'])

            if test_day in link_data[link_id]:
                X_test.extend(link_data[link_id][test_day]['X'])
                y_test.extend(link_data[link_id][test_day]['y'])

        if len(X_train) < 10 or len(X_test) < 10:
            continue

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        # Evaluate
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5

        fold_results.append({
            'fold': test_day,
            'auc': auc,
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred),
            'n_train': len(y_train),
            'n_test': len(y_test)
        })

    # Aggregate
    aucs = [r['auc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    return {
        'config': config_name,
        'n_links': len(selected_links),
        'n_folds': len(fold_results),
        'fold_results': fold_results,
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'aucs': aucs
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate presence detection')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing CSI data')
    parser.add_argument('--labels', type=Path, required=True,
                        help='CSV file with ground truth labels')
    parser.add_argument('--output-dir', type=Path, default=Path('./results'),
                        help='Output directory for results')
    parser.add_argument('--dates', nargs='+', default=None,
                        help='Dates to evaluate (YYYY-MM-DD)')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 3, 5, 10],
                        help='Top-k values to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Setup
    set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Presence Detection Evaluation")
    print("=" * 60)

    # Load labels
    print(f"\nLoading labels from {args.labels}")
    labels_df = pd.read_csv(args.labels)

    # Determine dates
    if args.dates:
        dates = args.dates
    else:
        dates = sorted(labels_df['date'].unique())

    print(f"Evaluating {len(dates)} days: {dates[0]} to {dates[-1]}")

    # Load data
    print(f"\nLoading CSI data from {args.data_dir}")
    link_data = load_csi_data(args.data_dir, dates, labels_df)
    print(f"Loaded {len(link_data)} unique links")

    if not link_data:
        print("ERROR: No data loaded!")
        return

    # Evaluate configurations
    results = []

    # All links
    print("\n[1/4] Evaluating ALL links...")
    all_links = list(link_data.keys())
    result_all = evaluate_link_configuration(link_data, all_links, dates, "all_links")
    results.append(result_all)
    print(f"  AUC: {result_all['auc_mean']:.3f} ± {result_all['auc_std']:.3f}")

    # Top-k configurations
    for k in args.k_values:
        print(f"\n[2/4] Evaluating TOP-{k} links...")
        top_k_links = select_links_by_variance(link_data, k)
        result_topk = evaluate_link_configuration(
            link_data, top_k_links, dates, f"top_{k}"
        )
        results.append(result_topk)
        print(f"  AUC: {result_topk['auc_mean']:.3f} ± {result_topk['auc_std']:.3f}")
        print(f"  Selected links: {top_k_links[:3]}...")

    # Random-k (baseline)
    print(f"\n[3/4] Evaluating RANDOM-1 baseline...")
    random_links = select_random_links(link_data, 1, args.seed)
    result_random = evaluate_link_configuration(
        link_data, random_links, dates, "random_1"
    )
    results.append(result_random)
    print(f"  AUC: {result_random['auc_mean']:.3f} ± {result_random['auc_std']:.3f}")

    # Statistical comparison
    print("\n[4/4] Statistical Analysis...")
    best_topk = max([r for r in results if r['config'].startswith('top_')],
                    key=lambda x: x['auc_mean'])

    stats = full_statistical_analysis(
        np.array(best_topk['aucs']),
        np.array(result_all['aucs']),
        k=int(best_topk['config'].split('_')[1]),
        n_bootstrap=10000,
        random_state=args.seed
    )

    print(f"\n  {best_topk['config']} vs all_links:")
    print(f"    AUC improvement: {stats['effect_size']['absolute_improvement']:.3f}")
    print(f"    Wilcoxon p-value: {stats['wilcoxon']['p_value']:.4f}")
    print(f"    Cohen's d: {stats['effect_size']['cohens_d']:.2f} ({stats['effect_size']['interpretation']})")
    print(f"    95% CI: [{stats['bootstrap_ci']['ci_lower']:.3f}, {stats['bootstrap_ci']['ci_upper']:.3f}]")
    print(f"    Conclusion: {stats['conclusion']['recommendation']}")

    # Save results
    output_file = args.output_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_dir': str(args.data_dir),
                'labels': str(args.labels),
                'dates': dates,
                'k_values': args.k_values,
                'seed': args.seed
            },
            'results': results,
            'statistical_analysis': stats
        }, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_file}")

    # Generate summary figure
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = [r['config'] for r in results]
    means = [r['auc_mean'] for r in results]
    stds = [r['auc_std'] for r in results]

    bars = ax.bar(configs, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax.set_ylabel('AUC-ROC')
    ax.set_xlabel('Link Configuration')
    ax.set_title('Presence Detection: Link Selection Comparison')
    ax.set_ylim(0.4, 1.0)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output_dir / 'comparison.png', dpi=150)
    print(f"Figure saved to {args.output_dir / 'comparison.png'}")


if __name__ == '__main__':
    main()
