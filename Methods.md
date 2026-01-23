# Benchmarking Methods

This document provides detailed descriptions of the benchmarking methodologies used in this study, covering Intraclass Correlation (ICC), Quality Control-Functional Connectivity (QC-FC), and Machine Learning (ML) classification.

---

## 1. Intraclass Correlation (ICC)

ICC measures the reliability of functional connectivity (FC) estimates across repeated sessions. We use the Beijing (China) EOEC dataset, which includes two Eyes Closed (EC) sessions for each subject.

### Approach
- **Data**: Beijing (China) EC session 1 and session 2.
- **Metric**: We primarily use **ICC(3,1)** (two-way mixed effects, consistency, single rater/measurement), which is standard for test-retest reliability in fMRI.
- **Computation**:
  - FC is computed for each session separately.
  - ICC is computed for every edge (ROI-ROI pair).
  - **Masking**: To avoid bias from low-signal edges, we compute "Masked ICC" by considering only the top 2% of edges with the highest group-average absolute FC weights.
- **Implementation**: `benchmarking/icc.py`

### Formula
ICC is computed as:
$$ICC = \frac{MSR - MSE}{MSR + (k-1)MSE}$$
where $MSR$ is the mean square for rows (subjects), $MSE$ is the mean square for error, and $k$ is the number of sessions (2).

### Example Function Call
```python
from benchmarking.icc import compute_icc_edgewise

# data shape: (n_subjects, n_edges, n_sessions)
icc_vector = compute_icc_edgewise(fc_data, icc_kind="icc31")
mean_icc = np.nanmean(icc_vector)
```

### Batch Execution
```bash
python -m benchmarking.icc --config configs/icc_atlas.yaml
```

---

## 2. Quality Control - Functional Connectivity (QC-FC)

QC-FC evaluates the relationship between subject motion and functional connectivity estimates. High correlations indicate that the denoising strategy failed to remove motion artifacts.

### Approach
- **Data**: All subjects from IHB and China datasets.
- **Motion Metric**: Mean Root Mean Square (RMS) displacement.
- **Metric**: Pearson correlation between the motion metric and the strength of each FC edge across subjects.
- **Summary Statistics**:
  - **Mean |r|**: The average of the absolute values of the correlations across all edges.
  - **Fraction of Significant Edges**: The percentage of edges where the correlation with motion is significant ($p < 0.05$, uncorrected).
- **Implementation**: `benchmarking/qc_fc.py`

### Example Function Call
```python
from benchmarking.qc_fc import qc_fc_edges

# fc_vec: (n_subjects, n_edges)
# rms_vec: (n_subjects,)
results = qc_fc_edges(fc_vec, rms_vec)
print(f"Mean |r|: {results['mean_abs_r']}")
```

### Batch Execution
```bash
python -m benchmarking.qc_fc --config configs/qc_fc_atlas.yaml
```

---

## 3. Machine Learning (ML) Classification

The ML pipeline evaluates the biological validity and cross-site generalizability of FC pipelines by classifying Eye State (EO vs. EC).

### Unified Pipeline Design
The pipeline in `benchmarking/ml/pipeline.py` ensures strict separation between training and testing data to prevent leakage.

#### A. Preprocessing & Feature Extraction
1. **Heterogeneous TR Handling**: Supports lists of arrays for sites with different time series lengths (IHB: 120, China: 240).
2. **Leakage-Safe FC**: For **Tangent Space** projection, the reference covariance matrix is fitted **only on the training set**.
3. **ROI Masking**: Uses IHB coverage masks to ensure a consistent ROI set across different scanners.
4. **Standardization**: Features are scaled using `StandardScaler` fitted on the training set.
5. **PCA**: Dimensionality reduction (default: 95% variance) fitted on the training set to stabilize linear models.

#### B. Validation Schemes
- **Cross-Site (0-shot)**: Train on Site A (e.g., China), test on Site B (e.g., IHB).
- **Few-Shot Adaptation**: Train on Site A + $k$ subjects from Site B, test on the remaining subjects of Site B.

#### C. Statistical Rigor
- **Permutation Testing**: Performed per pipeline by shuffling training labels to establish a null distribution of accuracy.
- **Paired Comparisons**: McNemar test for accuracy and DeLong test for ROC-AUC are used to compare two pipelines on the same test set.

### Example Function Call
```python
from benchmarking.ml.pipeline import run_full_pipeline, PipelineConfig

config = PipelineConfig(
    atlas="AAL",
    strategy=1,
    gsr="GSR",
    models=["logreg"],
    pca_components=0.95,
    n_permutations=1000
)
results = run_full_pipeline(config)
```

### Batch Execution
```bash
# Run multiple pipelines defined in a YAML config
python -m benchmarking.run_ml_pipelines --config configs/ml_atlas.yaml
```

### Statistical Testing CLI
```bash
# Compare GSR vs noGSR for a specific run
python -m benchmarking.ml.pipeline_comparisons factor \
    --test-outputs results/pipelines/AAL_strategy-1_GSR/cross_site_ihb2china_test_outputs.csv \
    --factor gsr --level-a GSR --level-b noGSR
```
