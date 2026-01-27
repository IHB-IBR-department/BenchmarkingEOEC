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

## 5. Feature Interpretation via Subsampling Stability

To identify robust biomarkers distinguishing Eyes Open (EO) from Eyes Closed (EC) states, we employ a **subject-level subsampling stability analysis** that assesses which FC edges consistently contribute to classification across random data partitions.

### Motivation
Single-model coefficients can be noisy and sensitive to specific training splits, especially in high-dimensional neuroimaging where `n_features >> n_samples`. By evaluating feature stability across many subsamples, we isolate true biological signals from noise.

### Approach
- **Pooled Dataset**: IHB (84 subjects) + China (48 subjects) = 132 subjects total.
- **Atlas**: Schaefer200 (182 ROIs after coverage masking) with Yeo 7-network parcellation.
- **Strategy**: 3 (aCompCor(50%)+12P) — best performing denoising strategy.
- **FC Types**: Both Pearson Correlation and Tangent Space projection.

### Procedure

#### Step 1: Precompute FC on Full Dataset
Both correlation and tangent FC matrices are computed **once** for all 132 subjects:
- **Correlation**: Computed independently per subject.
- **Tangent**: Reference covariance (Fréchet mean) fitted on the full pooled dataset, then all subjects projected to tangent space.

This precomputation enables fast subsampling (~1.3 iterations/second vs ~0.07 without precomputation).

#### Step 2: Subject-Level Subsampling Loop
For each of S iterations (default: 1000):
1. **Subsample**: Randomly select 80% of subjects (without replacement), stratified by site.
2. **Paired Design**: If a subject is selected, both their EO and EC scans are included.
3. **Select FC**: Retrieve precomputed correlation and tangent FC for selected subjects.
4. **Fit Model**: `StandardScaler → PCA(95%) → Logistic Regression`.
5. **Backproject**: Extract edge weights from PCA-space to original feature space.
6. **Rank**: Compute importance ranks (1 = highest absolute weight).

### Stability Metrics
For each edge, we aggregate across S subsamples:
- **Sign Consistency**: Fraction of subsamples where the edge has the same sign direction. Values near 1.0 indicate stable directionality; ~0.5 indicates random.
- **Mean Rank**: Average importance rank across subsamples.
- **Mean Weight**: Average coefficient magnitude and direction.

### Selection Criteria for Stable Edges
Edges are considered "relevant" if they satisfy:
1. **High Stability**: Sign Consistency ≥ 80%
2. **Top Importance**: Mean Rank ≤ 500 (top ~3% of edges)

### Outputs
- **Volcano plots**: Sign consistency vs. mean weight for each FC type.
- **Network heatmaps**: 7×7 matrices showing count/importance of stable edges between Yeo networks.
- **FC comparison**: Overlap analysis between correlation and tangent stable edges.

### Implementation
`analysis/interpret_classification_coefficients.py`

---

## 6. Data Access Statement

- **Beijing EOEC Dataset**: Publicly available via [NITRC](https://fcon_1000.projects.nitrc.org/indi/retro/BeijingEOEC.html).
- **IHB RAS Dataset**: Proprietary research data from the Institute of the Human Brain. Derivative time series and aggregated results are provided in the [repository's data releases], while raw NIfTI files are available upon reasonable request from the corresponding author, subject to ethics institutional board approval.

