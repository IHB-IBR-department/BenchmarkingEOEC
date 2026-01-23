# BenchmarkingEOEC

Code for the paper:
Medvedeva, T., Knyazeva, I., Masharipov, R., Korotkov, A., Cherednichenko, D., & Kireev, M. (2025).
Benchmarking resting state fMRI connectivity pipelines for classification: Robust accuracy despite processing variability in cross-site eye state prediction. Neuroscience.
https://doi.org/10.1101/2025.10.20.683049

## Data availability

Preprocessed time series data are available for download:
- **Beijing (China) EOEC dataset**: [Google Drive link](https://drive.google.com/file/d/1N1wdbF0tsrAzL9GSgUK_y4xBNDiPLH5X/view?usp=sharing)
- **IHB (St. Petersburg) dataset**: [link TBD]

**For detailed data format descriptions, see [DataDescription.md](DataDescription.md).
For detailed methodology descriptions, see [Methods.md](Methods.md).**

## Dataset description

This study uses two resting-state fMRI datasets with Eyes Open (EO) and Eyes Closed (EC) conditions:

### IHB dataset (St. Petersburg, Russia)
| Property | Value |
|----------|-------|
| Scanner | Philips 3T |
| Subjects | 84 |
| Conditions | 1 EC (run-1) + 1 EO (run-2) per subject |
| Timepoints | 120 TRs per run |
| TR | 2.5s |

### Beijing EOEC dataset (China)
| Property | Value |
|----------|-------|
| Scanner | Siemens 3T |
| Subjects | 48 |
| Conditions | 2 EC + 1 EO per subject (3 runs total) |
| Timepoints | 240 TRs per run |
| TR | 2.0s |
| Session mapping | Variable per subject (see BeijingEOEC.csv) |
| Original source | https://fcon_1000.projects.nitrc.org/indi/retro/BeijingEOEC.html |

### Preprocessing pipelines

Each dataset was processed through multiple denoising pipelines:

**Standard denoising strategies (1-6):**
| Strategy | Confounds |
|----------|-----------|
| 1 | 24 motion parameters (24P) |
| 2 | aCompCor (5 components) + 12P |
| 3 | aCompCor (50% variance) + 12P |
| 4 | aCompCor (5 components) + 24P |
| 5 | aCompCor (50% variance) + 24P |
| 6 | a/tCompCor (50% variance each) + 24P |

**ICA-AROMA variants:**
| Variant | Description |
|---------|-------------|
| AROMA_aggr | Aggressive denoising (full regression of noise ICs) |
| AROMA_nonaggr | Non-aggressive denoising (partial regression) |

**Global Signal Regression (GSR):**
- Each strategy available with and without GSR
- Total: (6 standard + 2 AROMA) × 2 GSR = 16 pipelines per atlas

### Atlases

| Atlas | Nominal ROIs | After coverage filtering |
|-------|--------------|-------------------------|
| Schaefer200 | 200 | 200 (China), 200 (IHB) |
| AAL | 116 | 116 |
| Brainnetome | 246 | 246 |
| HCPex | 426 | 421-426 (varies slightly) |

### AROMA data availability

**Complete AROMA coverage for cross-site validation:**

| Site | Subjects | Atlases with AROMA | Status |
|------|----------|-------------------|--------|
| IHB (St. Petersburg) | 84 | AAL, Brainnetome, HCPex, Schaefer200 | ✅ Complete |
| China (Beijing) | 48 | AAL, Brainnetome, HCPex, Schaefer200 | ✅ Complete |

Both datasets have complete ICA-AROMA preprocessing for all 4 atlases. Each atlas includes 8 AROMA files:
- 2 AROMA variants (aggressive, non-aggressive) × 2 GSR options × 2 conditions (open/close)

## Aggregated time series format

Time series are provided as numpy arrays (.npy) with consistent subject ordering.

### IHB data structure
```
timeseries_ihb/
├── Schaefer200/
│   ├── ihb_close_Schaefer200_strategy-1_noGSR.npy   # (84, 120, 200) float32
│   ├── ihb_open_Schaefer200_strategy-1_noGSR.npy    # (84, 120, 200) float32
│   ├── ihb_close_Schaefer200_strategy-1_GSR.npy
│   ├── ihb_open_Schaefer200_strategy-1_GSR.npy
│   ├── ... (strategies 2-6)
│   ├── ihb_close_Schaefer200_strategy-AROMA_aggr_noGSR.npy    # (84, 120, 200)
│   ├── ihb_open_Schaefer200_strategy-AROMA_aggr_noGSR.npy     # (84, 120, 200)
│   ├── ihb_close_Schaefer200_strategy-AROMA_aggr_GSR.npy
│   ├── ihb_open_Schaefer200_strategy-AROMA_aggr_GSR.npy
│   ├── ihb_close_Schaefer200_strategy-AROMA_nonaggr_noGSR.npy
│   ├── ihb_open_Schaefer200_strategy-AROMA_nonaggr_noGSR.npy
│   ├── ihb_close_Schaefer200_strategy-AROMA_nonaggr_GSR.npy
│   ├── ihb_open_Schaefer200_strategy-AROMA_nonaggr_GSR.npy
│   └── subject_order.txt                            # sub-001 to sub-084 (same for all strategies)
├── AAL/        # 84 subjects, 116 ROIs (complete AROMA)
├── Brainnetome/  # 84 subjects, 246 ROIs (complete AROMA)
└── HCPex/      # 84 subjects, 426 ROIs (complete AROMA)
```

**Array dimensions**: `(n_subjects=84, n_timepoints=120, n_rois)`
- Standard strategies: 6 strategies × 2 GSR options × 2 conditions = 24 files
- AROMA strategies: 2 variants × 2 GSR options × 2 conditions = 8 files
- Total: 32 files per atlas

### China data structure
```
timeseries_china/
├── Schaefer200/
│   ├── china_close_Schaefer200_strategy-1_noGSR.npy  # (48, 240, 200, 2) float32
│   ├── china_open_Schaefer200_strategy-1_noGSR.npy   # (48, 240, 200) float32
│   ├── ... (strategies 1-6)
│   ├── china_close_Schaefer200_strategy-AROMA_aggr_noGSR.npy    # (48, 240, 200, 2)
│   ├── china_open_Schaefer200_strategy-AROMA_aggr_noGSR.npy     # (48, 240, 200)
│   ├── ... (AROMA: aggr/nonaggr, GSR/noGSR)
│   └── subject_order_china.txt                       # Same for all strategies
├── AAL/        # 48 subjects, 116 ROIs (complete AROMA)
├── Brainnetome/  # 48 subjects, 246 ROIs (complete AROMA)
└── HCPex/      # 48 subjects, 426 ROIs (complete AROMA)
```

**Array dimensions**:
- **close**: `(n_subjects=48, n_timepoints=240, n_rois, 2)` — 4D array
  - `close[:,:,:,0]` = first closed session
  - `close[:,:,:,1]` = second closed session
- **open**: `(n_subjects=48, n_timepoints=240, n_rois)` — 3D array
- Standard strategies: 6 strategies × 2 GSR options × 2 conditions = 24 files per atlas
- AROMA strategies: 2 variants × 2 GSR options × 2 conditions = 8 files per atlas
- Total: 32 files per atlas

### Data quality notes

**Incomplete scans (zero-padded)**:

| Subject | Run | Condition | Actual TRs | Zeros | Impact |
|---------|-----|-----------|------------|-------|--------|
| sub-2021733 | run-2 | open | 239/240 | 0.4% | Minimal |
| **sub-3258811** | run-3 | close2 | 53/240 | **77.9%** | **Significant** |

**WARNING**: sub-3258811's second closed session (`close[:,:,:,1]`) contains 77.9% zeros.
- First closed session (`close[:,:,:,0]`) is complete
- Open session is complete
- Exclude this subject when using both closed sessions, or use only `close[:,:,:,0]`

### Aggregation scripts

To regenerate aggregated data from per-subject CSVs:

**IHB (St. Petersburg) data:**
```bash
# Standard denoising pipelines (strategies 1-6)
python -m data_utils.preprocessing.aggregate_ihb

# AROMA pipelines (all 4 atlases)
python -m data_utils.preprocessing.aggregate_ihb --aroma
```

**China (Beijing) data:**
```bash
# Standard denoising pipelines
python -m data_utils.preprocessing.aggregate_china

# AROMA pipelines (all 4 atlases)
python -m data_utils.preprocessing.aggregate_china --aroma
```

### Loading example

```python
import numpy as np

# Load IHB data
ihb_close = np.load('timeseries_ihb/Schaefer200/ihb_close_Schaefer200_strategy-1_noGSR.npy')
ihb_open = np.load('timeseries_ihb/Schaefer200/ihb_open_Schaefer200_strategy-1_noGSR.npy')
print(f"IHB: {ihb_close.shape}, {ihb_open.shape}")  # (84, 120, 200), (84, 120, 200)

# Load China data
china_close = np.load('timeseries_china/Schaefer200/china_close_Schaefer200_strategy-1_noGSR.npy')
china_open = np.load('timeseries_china/Schaefer200/china_open_Schaefer200_strategy-1_noGSR.npy')
print(f"China: {china_close.shape}, {china_open.shape}")  # (48, 240, 200, 2), (48, 240, 200)

# Load AROMA data
ihb_close_aroma = np.load('timeseries_ihb/Schaefer200/ihb_close_Schaefer200_strategy-AROMA_aggr_noGSR.npy')
china_close_aroma = np.load('timeseries_china/Schaefer200/china_close_Schaefer200_strategy-AROMA_aggr_noGSR.npy')

# Access individual closed sessions for China
china_close1 = china_close[:, :, :, 0]  # First EC session
china_close2 = china_close[:, :, :, 1]  # Second EC session (caution: sub-3258811 is 78% zeros)

# Load subject order (same for all strategies including AROMA)
with open('timeseries_ihb/Schaefer200/subject_order.txt') as f:
    ihb_subjects = [line.strip() for line in f]
with open('timeseries_china/Schaefer200/subject_order_china.txt') as f:
    china_subjects = [line.strip() for line in f]
```

### Data availability summary

**IHB (St. Petersburg) - `timeseries_ihb/`**

| Atlas | Files | Shape (close/open) | Size per file | ROIs |
|-------|-------|-------------------|---------------|------|
| AAL | 32 | (84, 120, 116) | 4.5 MB | 116 |
| Brainnetome | 32 | (84, 120, 246) | 9.5 MB | 246 |
| HCPex | 32 | (84, 120, 423) | 16 MB | 423 |
| Schaefer200 | 32 | (84, 120, 200) | 7.7 MB | 200 |

- **Total**: 128 numpy arrays across 4 atlases
- **Subjects**: 84 (same for all atlases)
- **Strategies per atlas**: 6 standard (1-6) + 2 AROMA (aggr, nonaggr) = 8 pipelines
- **GSR options**: Each strategy available with GSR and noGSR
- **Conditions**: Each pipeline has both eyes closed (close) and eyes open (open)
- **Files per atlas**: 8 strategies × 2 GSR × 2 conditions = 32 files

**China (Beijing) - `timeseries_china/`**

| Atlas | Files | Shape (close) | Shape (open) | Size per file | ROIs |
|-------|-------|---------------|--------------|---------------|------|
| AAL | 32 | (48, 240, 116, 2) | (48, 240, 116) | 10 MB (close), 5 MB (open) | 116 |
| Brainnetome | 32 | (48, 240, 246, 2) | (48, 240, 246) | 21 MB (close), 10.5 MB (open) | 246 |
| HCPex | 32 | (48, 240, 426, 2) | (48, 240, 426) | 37 MB (close), 18.5 MB (open) | 426 |
| Schaefer200 | 32 | (48, 240, 200, 2) | (48, 240, 200) | 17 MB (close), 8.5 MB (open) | 200 |

- **Total**: 128 numpy arrays across 4 atlases
- **Subjects**: 48 (same for all atlases)
- **Strategies per atlas**: 6 standard (1-6) + 2 AROMA (aggr, nonaggr) = 8 pipelines
- **GSR options**: Each strategy available with GSR and noGSR
- **Conditions**: Each pipeline has 2 closed sessions (4D array) and 1 open session (3D array)
- **Files per atlas**: 8 strategies × 2 GSR × 2 conditions = 32 files

**Combined dataset totals:**
- **256 numpy arrays** (128 IHB + 128 China)
- **192 unique pipelines** per dataset (4 atlases × 6 strategies × 2 GSR × 4 FC types)
- **Complete AROMA coverage** for both sites and all 4 atlases

### Computing functional connectivity

```python
from data_utils.fc import ConnectomeTransformer

# Leakage-safe FC computation (important for tangent space)
transformer = ConnectomeTransformer(kind='corr', vectorize=True)

# Fit on training data, transform both
X_train = transformer.fit_transform(train_timeseries)
X_test = transformer.transform(test_timeseries)
```

## Classification pipeline (ML)

The unified pipeline lives in `benchmarking/ml/pipeline.py` and runs:
cross-site (both directions) + few-shot adaptation for a single preprocessing
configuration.

1) Load FC matrices per site and pipeline
   - FC types: corr, partial, tangent, glasso
   - Atlases: AAL, Schaefer200, Brainnetome, HCPex (see configs)
   - Denoising strategies: 1..6
   - GSR: GSR or noGSR

2) Build labels and sample order
   - Each subject contributes two scans: EC (label 0) and EO (label 1).
   - Data are concatenated as [EC..., EO...] per site so the sample order is
     consistent across pipelines.

3) Vectorize FC matrices to edge features
   - Symmetric FC matrices are vectorized via `sym_matrix_to_vec` (diagonal removed).
   - This yields a high-dimensional edge feature vector per scan.

4) Optional scaling
   - StandardScaler is applied when `scale: true` (auto-enabled for SVM).
   - The scaler is fit on training data only, then applied to test data.

5) PCA dimensionality reduction
   - PCA is fit on training features only and then applied to test features.
   - `pca_components` can be:
     - a float in (0,1] for variance retained (default: 0.95)
     - an int for a fixed number of components
     - 0 or None to disable PCA

6) Classifier
   - Logistic regression (default) or SVM with RBF kernel.
   - Outputs include predicted labels and a continuous score
     (decision_function or predict_proba) when available.

7) Metrics and outputs
   - Metrics: accuracy, ROC-AUC, Brier score.
   - Optional permutation test vs chance when `--n-permutations > 0`.
   - Cross-site can also save per-sample outputs for paired comparisons.

## Preventing subject leakage and keeping splits identical

Cross-site (Scheme A):
- Train on one site and test on the other (no subject overlap by design).
- Each direction has a fixed test set, so all pipelines are evaluated on the
  same subjects in the same order.
- PCA and scaling are fit on training data only; test data are transformed only.

Few-shot (Scheme B):
- Subject-level splits are generated once and reused for all pipelines.
- Both EC and EO scans from the same subject stay in the same split.
- Splits are saved to `*_splits.yaml` for reproducibility and auditing.

Permutation test:
- Only `y_train` is permuted; `X_train`, `X_test`, and `y_test` remain fixed,
  avoiding leakage across train/test.

## PCA choice

The default `--pca-components 0.95` retains 95% variance to stabilize training on
high-dimensional FC edge features. This reduces overfitting and improves
numerical stability for linear models. The number of components is stored in
outputs for transparency. You can set a fixed number of components or disable
PCA entirely via CLI.

## Statistical testing and pipeline comparisons

There are two distinct questions:

1) Pipeline vs chance (per pipeline)
   - Permutation test with label shuffling on the training set.
   - Controlled by `n_permutations` in YAML configs.

2) Pipeline A vs B (paired comparisons on the same test set)
   - Exact McNemar test for accuracy.
   - DeLong test for correlated ROC-AUCs.
   - Requires per-sample outputs saved by cross-site runs.

Factor-level comparisons (GSR, FC type, atlas, strategy) use paired, subject-level
randomization (sign-flip) tests with matched pipelines. This aggregates per-sample
losses to subject-level means before testing.

Required inputs for paired comparisons:
- `results/pipelines/<atlas>_strategy-<strategy>_<gsr>/cross_site_<train>2<test>_test_outputs.csv` (from pipeline runs)
- `pipeline_abbreviations.csv` if you want abbrev-based comparisons

Helper scripts:
- `benchmarking/ml/pipeline_comparisons.py` (CLI for factor and A vs B tests)

## Tangent leakage experiment

Tangent FC uses a reference matrix; if computed on all subjects before splitting,
information can leak from test to train. The repo includes an explicit check:

- Script: `benchmarking/ml/tangent_leakage.py`
- Method: compare LEAK (reference on all subjects) vs NO_LEAK (reference fit on
  train subjects only) using GroupKFold with subject-level splits.
- EO and EC scans from the same subject are kept in the same fold.

Configs:
- `configs/tangent_leakage_quick.yaml` (quick sanity check)
- `configs/tangent_leakage_full.yaml` (full analysis)

Outputs:
- `results/tangent_leakage_*_folds.csv`
- `results/tangent_leakage_*_summary.csv`

## Outputs overview

Cross-site (per pipeline):
- `results/pipelines/<atlas>_strategy-<strategy>_<gsr>/cross_site_ihb2china_results.csv`
- `results/pipelines/<atlas>_strategy-<strategy>_<gsr>/cross_site_china2ihb_results.csv`
- `results/pipelines/<atlas>_strategy-<strategy>_<gsr>/cross_site_*_test_outputs.csv` per-sample outputs (if enabled)

Few-shot (per pipeline):
- `results/pipelines/<atlas>_strategy-<strategy>_<gsr>/few_shot_results.csv` per repeat
- `results/pipelines/<atlas>_strategy-<strategy>_<gsr>/summary.csv` combined summary

QC–FC:
- `results/qcfc/qc_fc_*.csv` per-pipeline QC–FC summaries. **Note:** Scripts skip computation if the pipeline exists in this file.
- `results/qcfc/edge_correlations/<site>/<atlas>/*.csv` edge-wise correlations (if enabled in config).

ICC (China close test–retest):
- `<data_root>/icc_precomputed_fc/<atlas>/*.npy` precomputed FC vectors with session axis
- `icc_results/icc_summary_*.csv` per-pipeline ICC summary table (mean/std, masked/unmasked). **Note:** Scripts skip computation if the pipeline exists in this file.
- `icc_results/<atlas>/*_edgewise_icc_all.pkl` and `icc_results/<atlas>/*_edgewise_icc_masked.pkl` edge-wise ICC vectors (if enabled)

## Coverage-based ROI masking (optional)

This option masks FC edges that touch ROIs with poor coverage in either site.
A ROI is marked \"bad\" if its parcel coverage is below the threshold in
**IHB or China**. The mask is applied by zeroing rows/cols for bad ROIs in
each FC matrix before vectorization.

### Option A: compute masks automatically (per run)
Set in YAML:
```
coverage_mask:
  enabled: true
  threshold: 0.1
  mask_dir: null
```
When `mask_dir` is null, masks are computed once per atlas from
`<data_root>/coverage/*_parcel_coverage.npy` and saved to:
`<output_dir>/coverage_masks/<atlas>_bad_parcels.npy`.

### Option B: use precomputed masks
Generate masks once:
```
PYTHONPATH=. python -m data_utils.coverage --threshold 0.1 --output-dir results/coverage_masks
```
Then point YAML to that folder:
```
coverage_mask:
  enabled: true
  threshold: 0.1
  mask_dir: "results/coverage_masks"
```
Coverage overlap summary (threshold 0.1, from `data_utils.coverage`):

| Atlas | IHB good | China good | Both good | Bad (either) |
|-------|----------|------------|-----------|--------------|
| AAL | 106 (91.38%) | 116 (100.00%) | 106 (91.38%) | 10 (8.62%) |
| Schaefer200 | 182 (91.00%) | 200 (100.00%) | 182 (91.00%) | 18 (9.00%) |
| Brainnetome | 220 (89.43%) | 246 (100.00%) | 220 (89.43%) | 26 (10.57%) |
| HCPex | 379 (88.97%) | 426 (100.00%) | 379 (88.97%) | 47 (11.03%) |

Default coverage for pipeline/ML comparisons: use **IHB** coverage masks unless explicitly overridden.

Notes:
- `threshold` must be in (0, 1).
- The same `coverage_mask` block is supported in both cross-site and few-shot configs.

## Quick usage (examples)

QC–FC:
- `source venv/bin/activate && PYTHONPATH=. python -m benchmarking.qc_fc --config configs/qc_fc_quick.yaml`

ICC (prepare + summary):
- `source venv/bin/activate && PYTHONPATH=. python -m benchmarking.icc --config configs/icc_atlas.yaml`

ML pipeline (cross-site + few-shot, one pipeline):
- `PYTHONPATH=. python -m benchmarking.ml.pipeline --atlas Schaefer200 --strategy 1 --gsr GSR --precomputed-glasso ~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/glasso_precomputed_fc`

Permutation demo (add permutations, small test):
- `PYTHONPATH=. python -m benchmarking.ml.pipeline --atlas Schaefer200 --strategy 1 --gsr GSR --skip-glasso --n-permutations 100`

Pipeline comparisons (after running ML pipeline with `save_test_outputs: true`):
- Factor-level test (GSR vs noGSR):
  `PYTHONPATH=. python -m benchmarking.ml.pipeline_comparisons factor --test-outputs results/pipelines/Schaefer200_strategy-1_GSR/cross_site_ihb2china_test_outputs.csv --factor gsr --level-a GSR --level-b noGSR`
- Factor-level test (FC type: tangent vs corr):
  `PYTHONPATH=. python -m benchmarking.ml.pipeline_comparisons factor --test-outputs results/pipelines/Schaefer200_strategy-1_GSR/cross_site_ihb2china_test_outputs.csv --factor fc_type --level-a tangent --level-b corr`
- Pipeline A vs B (by abbreviation, if you have an abbreviations CSV):
  `PYTHONPATH=. python -m benchmarking.ml.pipeline_comparisons compare --test-outputs results/pipelines/Schaefer200_strategy-1_GSR/cross_site_ihb2china_test_outputs.csv --abbrev results/pipelines/pipeline_abbreviations.csv --pipeline-a P0001 --pipeline-b P0003`
Use `cross_site_china2ihb_test_outputs.csv` for the opposite direction.
