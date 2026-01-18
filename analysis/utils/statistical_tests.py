"""
Statistical Tests for Experiment Validation

Provides rigorous statistical analysis for comparing model configurations:
- Wilcoxon signed-rank test (non-parametric paired test)
- Bootstrap confidence intervals
- Cohen's d effect size
- Bonferroni multiple comparison correction

Author: Claude Code
Date: 2026-01-15
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = 'greater'
) -> Dict:
    """
    Paired Wilcoxon signed-rank test.

    Non-parametric test for comparing paired samples.
    Better than t-test when n < 30 or non-normal distributions.

    Args:
        x: First sample (e.g., Top-k AUC scores)
        y: Second sample (e.g., All-links AUC scores)
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        Dict with statistic, p_value, significant flags
    """
    x = np.array(x)
    y = np.array(y)

    # Compute differences
    diff = x - y

    # Remove zeros (ties)
    diff_nonzero = diff[diff != 0]

    if len(diff_nonzero) < 3:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant_05': False,
            'significant_01': False,
            'n_pairs': len(x),
            'n_nonzero': len(diff_nonzero),
            'warning': 'Too few non-zero differences for reliable test'
        }

    try:
        stat, p_value = stats.wilcoxon(diff_nonzero, alternative=alternative)
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant_05': False,
            'significant_01': False,
            'error': str(e)
        }

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'n_pairs': len(x),
        'n_nonzero': len(diff_nonzero),
        'mean_diff': float(np.mean(diff)),
        'median_diff': float(np.median(diff))
    }


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int = 42
) -> Dict:
    """
    Bootstrap confidence interval for the difference between paired samples.

    Args:
        x: First sample
        y: Second sample
        n_bootstrap: Number of bootstrap iterations
        ci_level: Confidence level (0.95 = 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dict with mean_diff, ci_lower, ci_upper, ci_excludes_zero
    """
    np.random.seed(random_state)

    x = np.array(x)
    y = np.array(y)
    n = len(x)

    # Bootstrap resampling
    diff_samples = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, n)
        boot_x = x[indices]
        boot_y = y[indices]
        diff_samples.append(np.mean(boot_x) - np.mean(boot_y))

    diff_samples = np.array(diff_samples)

    # Compute percentile CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(diff_samples, 100 * alpha / 2)
    ci_upper = np.percentile(diff_samples, 100 * (1 - alpha / 2))

    return {
        'mean_diff': float(np.mean(diff_samples)),
        'std_diff': float(np.std(diff_samples)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_level': ci_level,
        'ci_excludes_zero': ci_lower > 0 or ci_upper < 0,
        'n_bootstrap': n_bootstrap
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Cohen's d effect size for paired samples.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        x: First sample
        y: Second sample

    Returns:
        Dict with cohens_d, interpretation, absolute_improvement
    """
    x = np.array(x)
    y = np.array(y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Pooled standard deviation
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_std = np.sqrt((var_x + var_y) / 2)

    if pooled_std < 1e-10:
        d = 0.0 if abs(mean_x - mean_y) < 1e-10 else np.inf
    else:
        d = (mean_x - mean_y) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    return {
        'cohens_d': float(d),
        'interpretation': interpretation,
        'absolute_improvement': float(mean_x - mean_y),
        'relative_improvement': float((mean_x - mean_y) / mean_y * 100) if mean_y != 0 else np.inf,
        'mean_x': float(mean_x),
        'mean_y': float(mean_y),
        'pooled_std': float(pooled_std)
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Bonferroni correction for multiple comparisons.

    Adjusts the significance threshold to control family-wise error rate.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired family-wise error rate

    Returns:
        Dict with adjusted_alpha, significant_after_correction
    """
    n = len(p_values)
    adjusted_alpha = alpha / n

    return {
        'original_alpha': alpha,
        'adjusted_alpha': adjusted_alpha,
        'n_comparisons': n,
        'p_values': [float(p) for p in p_values],
        'significant_after_correction': [p < adjusted_alpha for p in p_values],
        'n_significant': sum(p < adjusted_alpha for p in p_values)
    }


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = 'greater',
    random_state: int = 42
) -> Dict:
    """
    Paired permutation test (exact, distribution-free).

    Args:
        x: First sample
        y: Second sample
        n_permutations: Number of permutations
        alternative: 'greater', 'less', or 'two-sided'
        random_state: Random seed

    Returns:
        Dict with observed_diff, p_value, significant
    """
    np.random.seed(random_state)

    x = np.array(x)
    y = np.array(y)

    observed_diff = np.mean(x) - np.mean(y)
    n = len(x)

    # Generate permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        # Randomly flip signs of differences
        signs = np.random.choice([-1, 1], size=n)
        perm_diff = np.mean(signs * (x - y))
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Compute p-value
    if alternative == 'greater':
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == 'less':
        p_value = np.mean(perm_diffs <= observed_diff)
    else:  # two-sided
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return {
        'observed_diff': float(observed_diff),
        'p_value': float(p_value),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'n_permutations': n_permutations,
        'alternative': alternative
    }


def full_statistical_analysis(
    topk_scores: np.ndarray,
    all_scores: np.ndarray,
    k: int,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Dict:
    """
    Complete statistical analysis comparing Top-k vs All links.

    Args:
        topk_scores: AUC scores for Top-k configuration
        all_scores: AUC scores for All-links configuration
        k: Number of links in Top-k
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Comprehensive statistical analysis results
    """
    topk_scores = np.array(topk_scores)
    all_scores = np.array(all_scores)

    results = {
        'comparison': f'top_{k}_vs_all',
        'n_folds': len(topk_scores),
        'topk_mean': float(np.mean(topk_scores)),
        'topk_std': float(np.std(topk_scores)),
        'all_mean': float(np.mean(all_scores)),
        'all_std': float(np.std(all_scores)),
    }

    # Wilcoxon test
    results['wilcoxon'] = wilcoxon_test(topk_scores, all_scores, alternative='greater')

    # Bootstrap CI
    results['bootstrap_ci'] = bootstrap_ci(
        topk_scores, all_scores,
        n_bootstrap=n_bootstrap,
        random_state=random_state
    )

    # Effect size
    results['effect_size'] = cohens_d(topk_scores, all_scores)

    # Permutation test
    results['permutation'] = permutation_test(
        topk_scores, all_scores,
        n_permutations=n_bootstrap,
        random_state=random_state
    )

    # Overall significance conclusion
    results['conclusion'] = {
        'significant': (
            results['wilcoxon']['significant_05'] and
            results['bootstrap_ci']['ci_excludes_zero']
        ),
        'effect_magnitude': results['effect_size']['interpretation'],
        'recommendation': (
            'SUPPORTED' if results['wilcoxon']['significant_05'] and
            results['effect_size']['cohens_d'] > 0.5
            else 'WEAK' if results['wilcoxon']['significant_05']
            else 'NOT SUPPORTED'
        )
    }

    return results
