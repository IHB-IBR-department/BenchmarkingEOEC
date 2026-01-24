# Benchmarking Methods

This document provides detailed descriptions of the benchmarking methodologies used in this study, covering Intraclass Correlation (ICC), Quality Control-Functional Connectivity (QC-FC), and Machine Learning (ML) classification.

---

## 1. Intraclass Correlation (ICC)

ICC measures the reliability of functional connectivity (FC) estimates across repeated sessions. We use the Beijing (China) EOEC dataset, which includes two Eyes Closed (EC) sessions for each subject.

### Approach
- **Data**: Beijing (China) EC session 1 and session 2.
- **Metrics**: We analyze three ICC variants to ensure robustness:
  - **ICC(3,1)**: Two-way mixed effects, consistency, single rater (Primary Metric).
  - **ICC(2,1)**: Two-way random effects, absolute agreement, single rater.
  - **ICC(1,1)**: One-way random effects, absolute agreement, single rater.
- **Computation**:
  - FC is computed for each session separately.
  - ICC is computed for every edge (ROI-ROI pair).
  - **Masking**: To avoid bias from low-signal edges in sparse methods (like Glasso), we compute "Masked ICC" by considering only the top 5% of edges with the highest group-average absolute FC weights.
- **Implementation**: `benchmarking/icc.py`

---

## 2. Quality Control - Functional Connectivity (QC-FC)

QC-FC evaluates the relationship between subject motion and functional connectivity estimates. High correlations indicate that the denoising strategy failed to remove motion artifacts.

### Approach
- **Data**: Combined subjects from IHB and China datasets.
- **Motion Metric**: Mean relative Root Mean Square (RMS) displacement.
- **Metric**: Pearson correlation between the motion metric and the strength of each FC edge across subjects.
- **Aggregation**: 
  - **Site Handling**: Computed separately for IHB and China.
  - **Condition Handling**: Computed separately for "Open" and "Close" states.
  - **China Session Averaging**: For the China site, the correlations from the two EC sessions are averaged to create a `unified_close_abs_r` metric.
- **Summary Statistics**:
  - **Mean |r|**: The average of the absolute values of the correlations across all edges.
  - **Fraction of Significant Edges**: The percentage of edges where the correlation with motion is significant ($p < 0.05$, uncorrected).
- **Implementation**: `benchmarking/qc_fc.py`

---

## 3. Machine Learning (ML) Classification

The ML pipeline evaluates the biological validity and cross-site generalizability of FC pipelines by classifying Eye State (EO vs. EC).

### Unified Pipeline Design
The pipeline in `benchmarking/ml/pipeline.py` ensures strict separation between training and testing data to prevent leakage.

#### A. Preprocessing & Feature Extraction
1. **HCPex Dimension Alignment**: All HCPex strategies (standard 1-6 and AROMA) are aligned to a unified **373 ROI mask** to ensure cross-site and cross-strategy consistency.
2. **Leakage-Safe FC**: For **Tangent Space** projection, the reference covariance matrix (Fréchet mean) is fitted **only on the training set** of the current fold/direction.
3. **ROI Masking**: For non-HCPex atlases, IHB coverage masks (threshold 0.1) are applied to ensure consistent feature sets.
4. **Standardization**: Features are Z-scored using the mean and SD of the **training set**.
5. **PCA**: Dimensionality reduction retaining **95% of the variance**, fitted on the training set only.

#### B. Validation Schemes
- **Direct Cross-Site (Zero-Shot)**: Train on Site A (Full), test on Site B (Full). This tests pure generalizability.
- **Few-Shot Domain Adaptation**: Train on Site A + $k$ subjects from Site B, test on the remaining subjects of Site B. Evaluated over `n_repeats` random repeats (configured in `configs/ml_atlas.yaml`).

#### C. Statistical Rigor
- **Permutation Testing**: Performed per pipeline (`n_permutations` configurable in `configs/ml_atlas.yaml`) by shuffling training labels.
- **Paired Randomization (Sign-Flip)**: Used for global factor-level comparisons (e.g., GSR vs noGSR) across all matched pipeline pairs (5,000 permutations).
- **Model Comparison**: McNemar test for Accuracy and DeLong test for ROC-AUC.

## 4. Benchmarking Scope

This study evaluates a total of **259+ distinct functional connectivity pipelines** (depending on the inclusion of exploratory models and atlases). The scope includes:
- **4 Atlases**: AAL, Schaefer200, Brainnetome, HCPex.
- **8 Denoising Strategies**: Standard (1-6) and ICA-AROMA (Aggressive/Non-Aggressive).
- **2 GSR Options**: GSR and noGSR.
- **4 FC Methods**: Pearson Correlation, Partial Correlation, Tangent Space Projection, and Graphical Lasso.

---

## 5. Data Access Statement

- **Beijing EOEC Dataset**: Publicly available via [NITRC](https://fcon_1000.projects.nitrc.org/indi/retro/BeijingEOEC.html).
- **IHB RAS Dataset**: Proprietary research data from the Institute of the Human Brain. Derivative time series and aggregated results are provided in the [repository's data releases], while raw NIfTI files are available upon reasonable request from the corresponding author, subject to ethics institutional board approval.

