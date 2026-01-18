# Reproducibility Guide

This document describes how to reproduce the experimental results from our WiFi CSI sensing paper.

## Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy scikit-learn pandas matplotlib seaborn

# 2. Build dataset from raw CSI data
python preprocessing/build_dataset.py \
    --input-dir /path/to/csi_data \
    --output /path/to/dataset.parquet \
    --window-ms 250

# 3. Run evaluation
python evaluation/evaluate_presence_detection.py \
    --data-dir /path/to/csi_data \
    --labels /path/to/labels.csv \
    --output-dir ./results

# 4. Generate figures
python visualization/plot_results.py \
    --results ./results/evaluation_results.json \
    --output-dir ./figures
```

## Directory Structure

```
analysis/
├── evaluation/
│   └── evaluate_presence_detection.py  # Main evaluation script
├── preprocessing/
│   └── build_dataset.py                # Data preparation
├── visualization/
│   └── plot_results.py                 # Figure generation
├── utils/
│   ├── statistical_tests.py            # Statistical analysis
│   ├── reproducibility.py              # Seed management
│   └── link_selection.py               # Link selection strategies
└── REPRODUCIBILITY.md                  # This file
```

## Data Format

### Raw CSI Data

The system expects CSI data in JSONL format (one JSON object per line):

```json
{
  "received_at": "2026-01-05T10:30:45.123456",
  "tx_mac": "E228",
  "rx_mac": "EC68",
  "rssi": -45,
  "csi_data": [12, -5, 8, 3, ...],
  "seq": 1234
}
```

Files should be organized as:
```
csi_data/
├── 2026-01-01/
│   ├── csi_000350.jsonl
│   ├── csi_010352.jsonl
│   └── ...
├── 2026-01-02/
│   └── ...
```

**Note:** Files use `csi_HHMMSS.jsonl` naming convention (hour-minute-second of collection start).

### Ground Truth Labels

CSV file with columns:
- `date`: YYYY-MM-DD
- `hour`: 0-23
- `lr_occupied`: 0 or 1 (living room occupancy)

Example:
```csv
date,hour,lr_occupied
2026-01-01,0,0
2026-01-01,1,0
2026-01-01,8,1
2026-01-01,9,1
```

## Evaluation Methodology

### Leave-One-Day-Out Cross-Validation

We use temporal cross-validation to ensure models generalize across days:

1. For each day D in the dataset:
   - Train on all other days
   - Test on day D
2. Report mean ± std of AUC-ROC across folds

This prevents temporal data leakage and tests real-world generalization.

### Link Selection Strategies

| Strategy | Description |
|----------|-------------|
| `all_links` | Use all available TX→RX pairs |
| `top_k` | Select k links with highest label variance |
| `random_k` | Randomly select k links (baseline) |

### Statistical Tests

We use multiple tests to ensure robustness:

1. **Wilcoxon signed-rank test**: Non-parametric paired test
   - Better than t-test for small samples (n < 30)
   - Reported p-value < 0.05 indicates significance

2. **Bootstrap confidence intervals**: 10,000 iterations
   - 95% CI that excludes zero indicates significant improvement

3. **Cohen's d effect size**:
   - < 0.2: negligible
   - 0.2-0.5: small
   - 0.5-0.8: medium
   - ≥ 0.8: large

## Reproducing Paper Results

### Main Result: Link Selection Benefit

```bash
python evaluation/evaluate_presence_detection.py \
    --data-dir /path/to/csi_data \
    --labels /path/to/labels.csv \
    --output-dir ./results \
    --k-values 1 3 5 10 \
    --seed 42
```

Expected output:
```
[1/4] Evaluating ALL links...
  AUC: 0.49 ± 0.05
[2/4] Evaluating TOP-1 links...
  AUC: 0.77 ± 0.11
...
Statistical Analysis:
  top_1 vs all_links:
    AUC improvement: 0.28
    Wilcoxon p-value: 0.0012
    Cohen's d: 2.45 (large)
    Conclusion: SUPPORTED
```

### Generating Figures

```bash
python visualization/plot_results.py \
    --results ./results/evaluation_results.json \
    --output-dir ./figures
```

Generates:
- `auc_comparison.pdf`: Bar chart of AUC by configuration
- `fold_boxplot.pdf`: Per-fold distribution
- `temporal_heatmap.pdf`: Day-by-day performance
- `significance.pdf`: Statistical test visualization

## Troubleshooting

### "No data loaded"
- Check that CSI files exist in the expected directory structure
- Verify JSONL format is correct (one JSON object per line)
- Check for `.gz` compression (supported automatically)

### Low AUC scores
- Verify ground truth labels are correct
- Check that occupied/empty periods have sufficient samples
- Ensure all nodes were operational during data collection

### Import errors
- Install all dependencies: `pip install -r requirements.txt`
- Python 3.8+ required

## Seed Management

All random operations use configurable seeds for reproducibility:

```python
from utils.reproducibility import set_all_seeds
set_all_seeds(42)  # Sets numpy, random, sklearn seeds
```

## Requirements

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Citation

If you use these scripts, please cite our paper:

```bibtex
@inproceedings{anonymous2026lessismore,
  title={Less is More: The Dilution Effect in Multi-Link Wireless Sensing},
  author={Anonymous},
  booktitle={Under Review},
  year={2026},
  note={Paper under double-blind review}
}
```

**Note:** Citation will be updated with full author details upon paper acceptance.
