# BenchmarkingEOEC

**Data Availability:** All data necessary to reproduce the results of this study (preprocessed timeseries, precomputed FC, and coverage masks) are available at: [https://disk.yandex.ru/d/kvxN8bP3xiw8nQ](https://disk.yandex.ru/d/kvxN8bP3xiw8nQ)

Code for the paper:
**Medvedeva T. et al. Benchmarking resting state fMRI connectivity pipelines for classification: Robust accuracy despite processing variability in cross-site eye state prediction // bioRxiv. – 2025. – С. 2025.10.20.683049.**
[https://doi.org/10.1101/2025.10.20.683049](https://doi.org/10.1101/2025.10.20.683049)

---

## Overview

This repository evaluates **259+ distinct functional connectivity (FC) pipelines** for the classification of **Eyes Open (EO)** vs. **Eyes Closed (EC)** states. We benchmark these pipelines across three critical dimensions:
1.  **Reliability**: Test-retest consistency using Intraclass Correlation (ICC).
2.  **Motion Control**: Residual motion artifacts using QC-FC correlations.
3.  **Predictive Validity**: Machine Learning (ML) classification accuracy across different scanners and sites.

---

## Key Features

- **Leakage-Free ML**: Strict separation of training and testing data for Tangent Space projection, PCA, and Scaling.
- **Subject-Grouped Splits**: Prevents identity leakage by ensuring both EO/EC scans from a subject stay in the same partition.
- **HCPex Alignment**: Unified 373 ROI preprocessing to allow fair benchmarking of high-resolution atlases across strategies.
- **Statistical Rigor**: 
    - N-way ANOVA with **Partial Eta-Squared ($\eta_p^2$)** for factor importance.
    - **Permutation Tests (configurable repeats)** for significance vs. chance (see `configs/ml_atlas.yaml`).
    - **Paired Sign-Flip Randomization (5,000 permutations)** for factor-level comparisons (see `results/summary/ml_statistical_analysis.md`).

---

## Installation & Environment

```bash
# Setup virtual environment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set PYTHONPATH to include the project root
export PYTHONPATH=.

# (Recommended) Configure data root once for all commands
export OPEN_CLOSE_BENCHMARK_DATA=~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data
```

---

## Core Workflows

### 1. Reliability Analysis (ICC)
Computes ICC(1,1), (2,1), and (3,1) for the China dataset.
```bash
python -m benchmarking.icc --config configs/icc_full.yaml
python analysis/analyze_icc.py
```

### 2. Motion Removal (QC-FC)
Computes the correlation between subject motion (RMS) and edge connectivity.
```bash
python -m benchmarking.qc_fc --config configs/qc_fc_full.yaml
python analysis/analyze_qcfc.py
```

### 3. Machine Learning (Direct Cross-Site & Few-Shot)
Runs the unified ML pipeline for generalization and domain adaptation.
```bash
# Full Benchmark (AAL, Schaefer, Brainnetome)
./scripts/run_classification.sh

# HCPex Specific Analysis
./scripts/run_hcpex_analysis.sh

# Statistical Aggregation and Factor Analysis
python analysis/analyze_ml.py
```

### 4. Few-Shot Performance Visualization
Generates facet grid boxplots for all 128+ pipelines.
```bash
python analysis/plot_few_shot_auc.py
```

---

## Results & Reports

All statistical summaries and master tables are gathered in:
**`results/summary/`**

- `icc_anova_results.md`: Reliability factor importance and top pipelines.
- `qcfc_anova_results.md`: Motion control factor importance.
- `ml_statistical_analysis.md`: Predictive validity, GSR effects, and ML rankings.
  
Raw (per-run) outputs are stored in `results/icc_results/`, `results/qcfc/`, and `results/ml/`.

---

## Detailed Documentation

- **[DataDescription.md](DataDescription.md)**: File formats and naming conventions.
- **[Methods.md](Methods.md)**: Technical details of ICC, QC-FC, and ML implementation.
- **[StatisticalAnalysis.md](StatisticalAnalysis.md)**: Description of ANOVA and Permutation testing protocols.
