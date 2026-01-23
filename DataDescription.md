# Data Description for EOEC Benchmarking Study

## Overview

This document describes the data structure and formats for benchmarking resting-state fMRI functional connectivity pipelines for Eyes Open vs Eyes Closed classification. The study evaluates 192 distinct preprocessing pipelines across two independent datasets.

**For analysis workflows and instructions, see `readme.md`.**

**Paper:** Medvedeva, T., Knyazeva, I., Masharipov, R., Korotkov, A., Cherednichenko, D., & Kireev, M. (2025). Benchmarking resting state fMRI connectivity pipelines for classification: Robust accuracy despite processing variability in cross-site eye state prediction. *Neuroscience*. https://doi.org/10.1101/2025.10.20.683049

## Data Organization

The local data directory (`~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data/`) contains:

```
OpenCloseBenchmark_data/
├── timeseries_ihb/          # IHB (St. Petersburg) preprocessed time series
├── timeseries_china/        # Beijing EOEC preprocessed time series
├── coverage/                # Atlas ROI coverage quality metrics
├── glasso_precomputed_fc/   # Precomputed graphical lasso FC matrices
└── icc_precomputed_fc/      # Precomputed FC for test-retest reliability
```

## Main Data Entry Points

### 1. Time Series Data (Primary Input)

The **time series folders** are the main entry points for all analyses. These contain preprocessed fMRI time series extracted from brain atlases after various denoising strategies.

#### IHB Dataset (St. Petersburg, Russia)

**Location:** `timeseries_ihb/`

**Structure:**
```
timeseries_ihb/
├── AAL/
│   ├── ihb_close_AAL_strategy-1_GSR.npy
│   ├── ihb_open_AAL_strategy-1_GSR.npy
│   ├── ihb_close_AAL_strategy-1_noGSR.npy
│   ├── ihb_open_AAL_strategy-1_noGSR.npy
│   ├── ... (strategies 2-6)
│   ├── ihb_close_AAL_strategy-AROMA_aggr_GSR.npy
│   ├── ihb_open_AAL_strategy-AROMA_aggr_GSR.npy
│   ├── ihb_close_AAL_strategy-AROMA_nonaggr_GSR.npy
│   ├── ihb_open_AAL_strategy-AROMA_nonaggr_GSR.npy
│   ├── subject_order.txt
│   └── rmse_AAL.npy
├── Schaefer200/
│   └── ... (same structure)
├── Brainnetome/
│   └── ... (same structure)
└── HCPex/
    └── ... (same structure)
```

**Array format:**
- **Shape:** `(n_subjects, n_timepoints, n_rois)`
- **dtype:** `float32`
- **Dimensions:**
  - `n_subjects = 84`
  - `n_timepoints = 120` (TR = 2.5s)
  - `n_rois`: varies by atlas (see Atlas Dimensions table)

**Naming convention:**
```
{site}_{condition}_{atlas}_strategy-{strategy}_{gsr}.npy
```
- `site`: "ihb" or "china"
- `condition`: "close" (eyes closed) or "open" (eyes open)
- `atlas`: "AAL", "Schaefer200", "Brainnetome", "HCPex"
- `strategy`: "1", "2", ..., "6", "AROMA_aggr", "AROMA_nonaggr"
- `gsr`: "GSR" (with global signal regression) or "noGSR"

**Files per atlas:**
- Standard strategies: 6 strategies × 2 GSR options × 2 conditions = 24 files
- AROMA strategies: 2 variants × 2 GSR options × 2 conditions = 8 files
- **Total: 32 .npy files per atlas**

#### Beijing EOEC Dataset (China)

**Location:** `timeseries_china/`

**Data source:** These data include 48 healthy controls from a community (student) sample from Beijing Normal University in China. The original dataset is available at https://fcon_1000.projects.nitrc.org/indi/retro/BeijingEOEC.html

Each participant has 3 resting state fMRI scans:
- First scan: Eyes Closed (EC)
- Second and third scans: Randomized between Eyes Open (EO) and Eyes Closed (EC)

**Session mapping:** The order of runs 2 and 3 varies per subject. The file `BeijingEOEC.csv` (located in the data root) maps run numbers to conditions for each subject.

**Funding:** Financial support for this dataset was provided by grants from the National Natural Science Foundation of China (30770594) and the National High Technology Program of China (863).

**Structure:** Same as IHB, with key differences:

**Array format:**
- **Eyes Closed (close):**
  - **Shape:** `(n_subjects, n_timepoints, n_rois, n_sessions)`
  - 4D array with 2 closed sessions per subject
  - `close[:, :, :, 0]` = first EC session
  - `close[:, :, :, 1]` = second EC session

- **Eyes Open (open):**
  - **Shape:** `(n_subjects, n_timepoints, n_rois)`
  - 3D array with 1 open session per subject

- **Dimensions:**
  - `n_subjects = 48`
  - `n_timepoints = 240` (TR = 2.0s)
  - `n_rois`: varies by atlas
  - `n_sessions = 2` (for close only)

**Data Quality Notes:**

⚠️ **Important:** Subject `sub-3258811` has incomplete data in the second closed session:
- **Run-3 (close2):** Only 53/240 TRs acquired (77.9% zeros)
- First closed session and open session are complete
- **Recommendation:** Exclude this subject when using both closed sessions, or use only `close[:, :, :, 0]`

### 2. Auxiliary Files in Time Series Folders

#### Subject Order Files

**Filename:** `subject_order.txt` (IHB) or `subject_order_china.txt` (China)

**Format:** Plain text, one subject ID per line

**Purpose:** Maps array row indices to subject identifiers

**Example (IHB):**
```
sub-001
sub-002
sub-003
...
sub-084
```

**Example (China):**
```
sub-2021733
sub-3258811
sub-3332820
...
```

**Usage:**
```python
with open('timeseries_ihb/AAL/subject_order.txt') as f:
    subjects = [line.strip() for line in f]
print(subjects[0])  # 'sub-001'
```

#### Beijing Session Mapping File

**Filename:** `BeijingEOEC.csv` (located in data root directory)

**Format:** CSV with columns for subject ID and run-to-condition mapping

**Purpose:** Maps the three resting-state fMRI runs to conditions (Eyes Open/Closed) for each Beijing EOEC subject. Since runs 2 and 3 were randomized, this file specifies which run corresponds to which condition.

**Columns:**
- Subject identifier
- Run 1: Always Eyes Closed
- Run 2: Eyes Open or Eyes Closed (varies by subject)
- Run 3: Eyes Open or Eyes Closed (varies by subject)

**Usage:** Consult this file to understand the session order for each Beijing EOEC subject when working with the original BIDS dataset or raw scans.

**Note:** The preprocessed time series files in `timeseries_china/` have already been organized by condition (`china_close_*` and `china_open_*`), so this file is primarily useful for reference or when working with original data.

#### RMSE Files

**Filename:** `rmse_{atlas}.npy`

**Format:** NumPy array

**Shape:**
- IHB: `(n_subjects, n_strategies, n_rois)`
- Example: `(84, 8, 116)` for AAL atlas

**Purpose:** Root Mean Square Error of time series reconstruction after denoising

**Dimensions:**
- Axis 0: Subjects
- Axis 1: Strategies (1-6, AROMA_aggr, AROMA_nonaggr)
- Axis 2: ROIs

**Usage:**
```python
rmse = np.load('timeseries_ihb/AAL/rmse_AAL.npy')
print(f"Subject 0, Strategy 1, ROI 0 RMSE: {rmse[0, 0, 0]:.4f}")
```

## Atlas Dimensions

### Standard Strategy ROI Counts

For denoising strategies 1-6:

| Atlas | Nominal ROIs | IHB ROIs | China ROIs | Notes |
|-------|-------------|----------|------------|-------|
| **AAL** | 116 | 116 | 116 | All ROIs retained |
| **Schaefer200** | 200 | 200 | 200 | All ROIs retained |
| **Brainnetome** | 246 | 246 | 246 | All ROIs retained |
| **HCPex** | 426 | **423** | **421** | ⚠️ Site-specific filtering (see below) |

### AROMA Strategy ROI Counts

For AROMA_aggr and AROMA_nonaggr strategies:

| Atlas | IHB ROIs | China ROIs |
|-------|----------|------------|
| **AAL** | 116 | 116 |
| **Schaefer200** | 200 | 200 |
| **Brainnetome** | 246 | 246 |
| **HCPex** | **426** | **426** |

**Key difference:** AROMA strategies preserve all 426 HCPex ROIs, while standard strategies have site-specific filtering.

### HCPex Atlas: Special Considerations

The HCPex atlas requires special handling due to quality filtering applied during ROI extraction.

#### The Problem

**Nilearn's NiftiLabelsMasker** removes ROIs that fail quality checks during extraction. The specific ROIs removed differ between sites:

- **Standard strategies (1-6):**
  - IHB: 423 ROIs (removed ROIs: 365, 398, 401)
  - China: 421 ROIs (removed ROIs: 365, 372, 396, 401, 405)
  - **Different dimensions prevent direct comparison**

- **AROMA strategies:**
  - Both sites: 426 ROIs (no filtering applied)

#### The Solution

A unified HCPex mask was created to ensure **373 ROIs** are consistently retained across:
- Both sites (IHB and China)
- All strategies (standard 1-6 and AROMA variants)
- All GSR conditions

**Mask location:** `coverage/hcp_mask.npy`

**Excluded ROIs (53 total):**
1. **47 ROIs with low coverage** (< 0.1 threshold) in IHB
2. **6 ROIs explicitly skipped** by nilearn: 365, 372, 396, 398, 401, 405

**Result:** All HCPex data is preprocessed to 373 ROIs before functional connectivity computation.

**Preprocessing is automatic** in:
- `data_utils/preprocessing/precompute_glasso.py`
- `data_utils/preprocessing/icc_data_preparation.py`

See `readme.md` for usage instructions and workflows.

## Coverage Data

**Location:** `coverage/`

**Purpose:** Atlas ROI quality metrics based on gray matter coverage in each site's mean anatomical image.

### Coverage Files

**Format:** NumPy array with coverage proportion (0.0 to 1.0) for each ROI

**Files:**
```
coverage/
├── ihb_AAL_parcel_coverage.npy           # (116,)
├── china_AAL_parcel_coverage.npy         # (116,)
├── ihb_Schaefer200_parcel_coverage.npy   # (200,)
├── china_Schaefer200_parcel_coverage.npy # (200,)
├── ihb_Brainnetome_parcel_coverage.npy   # (246,)
├── china_Brainnetome_parcel_coverage.npy # (246,)
├── ihb_HCPex_parcel_coverage.npy         # (426,)
├── china_HCPex_parcel_coverage.npy       # (426,)
├── ihb_skipped_rois_HCPex.txt            # 365, 398, 401
├── china_skipped_rois_HCPex.txt          # 365, 372, 396, 401, 405
└── hcp_mask.npy                          # (426,) boolean mask
```

**Coverage interpretation:**
- `1.0`: Perfect coverage (ROI fully in gray matter)
- `0.5`: Partial coverage (50% overlap)
- `0.0`: No coverage (ROI outside gray matter mask)

**Common threshold:** 0.1 (ROIs with < 10% coverage are masked in analyses)

### Coverage Overlap Summary

Threshold = 0.1:

| Atlas | IHB Good ROIs | China Good ROIs | Both Good | Excluded |
|-------|---------------|-----------------|-----------|----------|
| AAL | 106 (91.4%) | 116 (100%) | 106 (91.4%) | 10 (8.6%) |
| Schaefer200 | 182 (91.0%) | 200 (100%) | 182 (91.0%) | 18 (9.0%) |
| Brainnetome | 220 (89.4%) | 246 (100%) | 220 (89.4%) | 26 (10.6%) |
| HCPex | 379 (89.0%) | 426 (100%) | 379 (89.0%) | 47 (11.0%) |

**Note:** China site has consistently better coverage across all atlases.

**Usage in pipelines:**
```python
from data_utils.fc import compute_fc_from_strategy_file

# Automatically apply IHB coverage mask (default)
fc = compute_fc_from_strategy_file(
    strategy_path,
    coverage="ihb",
    coverage_threshold=0.1
)

# Use both-site intersection
fc = compute_fc_from_strategy_file(
    strategy_path,
    coverage="both",
    coverage_threshold=0.1
)
```

## Precomputed Functional Connectivity

To reduce computational burden, functional connectivity matrices are precomputed for computationally expensive methods.

### Graphical Lasso (Glasso)

**Location:** `glasso_precomputed_fc/`

**Structure:**
```
glasso_precomputed_fc/
├── ihb/
│   ├── AAL/
│   │   ├── ihb_close_AAL_strategy-1_GSR_glasso.npy
│   │   ├── ihb_open_AAL_strategy-1_GSR_glasso.npy
│   │   └── ... (32 files)
│   ├── Schaefer200/
│   ├── Brainnetome/
│   └── HCPex/
└── china/
    ├── AAL/
    ├── Schaefer200/
    ├── Brainnetome/
    └── HCPex/
```

**Array format:**
- **IHB (3D input):**
  - **Shape:** `(n_subjects, n_edges)`
  - Example (AAL, default ROI filtering): `(84, 5565)`
  - Vectorized upper triangle (diagonal excluded)

- **China (4D input for close):**
  - **Shape:** `(n_subjects, n_edges, n_sessions)`
  - Example (AAL, default ROI filtering): `(48, 5565, 2)` for close

- **China (3D input for open):**
  - **Shape:** `(n_subjects, n_edges)`
  - Example (AAL, default ROI filtering): `(48, 5565)` for open

**Edge count formula:**
```python
n_edges = n_rois * (n_rois - 1) // 2
```

**Examples (default ROI filtering):**
- AAL (106 ROIs): 5,565 edges
- Schaefer200 (182 ROIs): 16,471 edges
- Brainnetome (220 ROIs): 24,090 edges
- HCPex (373 ROIs after preprocessing): 69,378 edges

**Default ROI filtering used by precompute scripts:**
- Non-HCPex atlases: `coverage="ihb"` with `coverage_threshold=0.1` (to match edge counts across FC types and reuse glasso).
- HCPex: time series are preprocessed to 373 ROIs (masking is applied before FC; coverage-based filtering is not applied afterward).

**Parameters:**
- L1 regularization: λ = 0.03 (default)
- Solver: ADMM (Alternating Direction Method of Multipliers)
- Vectorization: Upper triangle without diagonal

**Computation time:**
- ~4-5 minutes per file (84 subjects)
- Total: ~5 hours for all 64 HCPex files

**Generation:**
```bash
python -m data_utils.preprocessing.precompute_glasso \
  --input timeseries_ihb/AAL \
  --output-dir glasso_precomputed_fc/ihb/AAL \
  --coverage ihb \
  --coverage-threshold 0.1
```

### ICC Precomputed FC

**Location:** `icc_precomputed_fc/`

**Purpose:** Functional connectivity matrices for test-retest reliability analysis (China dataset with 2 EC sessions)

**Structure:**
```
icc_precomputed_fc/
├── AAL/
│   ├── china_close_AAL_strategy-1_GSR_corr.npy
│   ├── china_close_AAL_strategy-1_GSR_pc.npy
│   ├── china_close_AAL_strategy-1_GSR_tang.npy
│   ├── china_close_AAL_strategy-1_GSR_glasso.npy
│   └── ... (all strategies, all FC types)
├── Schaefer200/
├── Brainnetome/
└── HCPex/
```

**Array format:**
- **Shape:** `(n_subjects, n_edges, n_sessions)`
- Example (AAL, default ROI filtering): `(47, 5565, 2)` (`sub-3258811` excluded)
- Vectorized upper triangle (diagonal excluded)
- Session axis enables ICC computation

**FC types:**
- `_corr.npy`: Pearson correlation
- `_pc.npy`: Partial correlation
- `_tang.npy`: Tangent (log-Euclidean) space
- `_glasso.npy`: Graphical lasso (sparse inverse covariance)

**Note:** `sub-3258811` is excluded by default due to incomplete second session.
**Note:** `_glasso.npy` is reused from `glasso_precomputed_fc/china/<atlas>/` (copied + subject drop). `data_utils.preprocessing.icc_data_preparation.py` does not fit glasso.

**Default ROI filtering (recommended):**
- Non-HCPex atlases: `--coverage ihb --coverage-threshold 0.1`
- HCPex: time series are pre-masked to 373 ROIs; coverage filtering is disabled automatically.

Expected `n_edges` with the recommended defaults:

| Atlas | Effective ROIs | `n_edges` |
|------|-----------------|----------|
| AAL | 106 | 5,565 |
| Schaefer200 | 182 | 16,471 |
| Brainnetome | 220 | 24,090 |
| HCPex | 373 | 69,378 |

**Generation:**
```bash
python -m data_utils.preprocessing.icc_data_preparation \
  --atlas AAL \
  --input-dir timeseries_china/AAL \
  --output-dir icc_precomputed_fc \
  --kinds corr pc tang glasso \
  --coverage ihb \
  --coverage-threshold 0.1 \
  --drop-subject sub-3258811
```

If `icc_precomputed_fc/` already contains older outputs computed with different settings (e.g. `--coverage none`), rerun with your desired options. The script validates existing shapes and recomputes stale outputs (use `--overwrite` to force full refresh).

## Denoising Strategies

The study evaluates 8 denoising approaches:

### Standard Strategies (1-6)

| ID | Confound Regressors | Description |
|----|---------------------|-------------|
| 1 | 24P | 24 motion parameters only |
| 2 | aCompCor(5) + 12P | 5 aCompCor components + 12 motion |
| 3 | aCompCor(50%) + 12P | aCompCor 50% variance + 12 motion |
| 4 | aCompCor(5) + 24P | 5 aCompCor components + 24 motion |
| 5 | aCompCor(50%) + 24P | aCompCor 50% variance + 24 motion |
| 6 | a/tCompCor(50%) + 24P | aCompCor + tCompCor 50% each + 24 motion |

**Motion parameters:**
- 12P: 6 realignment + 6 temporal derivatives
- 24P: 12P + 12 squared terms

**CompCor:**
- aCompCor: Anatomical (white matter + CSF)
- tCompCor: Temporal (high-variance voxels)

### AROMA Strategies

| ID | Description |
|----|-------------|
| AROMA_aggr | Aggressive denoising (full regression of noise ICs) |
| AROMA_nonaggr | Non-aggressive denoising (partial regression) |

**Note:** AROMA strategies independently identify and remove motion-related independent components.

### Global Signal Regression (GSR)

Each strategy is computed **with and without GSR**, yielding:
- 6 standard × 2 GSR = 12 variants
- 2 AROMA × 2 GSR = 4 variants
- **Total: 16 denoising pipelines per atlas**

## Functional Connectivity Methods

Four FC estimation methods are evaluated:

### 1. Pearson Correlation (corr)

**Formula:** Standard correlation between ROI time series

**Output:** Symmetric matrix, range [-1, 1]

**Vectorized:** Upper triangle (diagonal excluded)

### 2. Partial Correlation (pc/partial)

**Formula:** Correlation between two ROIs controlling for all others

**Computation:** Precision matrix → correlation

**Output:** Symmetric matrix, range [-1, 1]

### 3. Tangent Space (tang/tangent)

**Formula:** Log-Euclidean projection onto tangent space

**Reference:** Group mean covariance (fit on training data only)

**Output:** Symmetric matrix, range unrestricted

**Advantage:** Respects Riemannian geometry of covariance matrices

### 4. Graphical Lasso (glasso)

**Formula:** Sparse inverse covariance (precision matrix)

**Regularization:** L1 penalty (λ = 0.03)

**Output:** Sparse symmetric matrix

**Advantage:** Identifies direct (partial) connections, promotes sparsity

## Data Loading Examples

### Load Time Series

```python
import numpy as np
from pathlib import Path

data_root = Path("~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data").expanduser()

# IHB data
ihb_close = np.load(data_root / "timeseries_ihb/AAL/ihb_close_AAL_strategy-1_GSR.npy")
ihb_open = np.load(data_root / "timeseries_ihb/AAL/ihb_open_AAL_strategy-1_GSR.npy")

print(f"IHB close: {ihb_close.shape}")  # (84, 120, 116)
print(f"IHB open: {ihb_open.shape}")    # (84, 120, 116)

# China data
china_close = np.load(data_root / "timeseries_china/AAL/china_close_AAL_strategy-1_GSR.npy")
china_open = np.load(data_root / "timeseries_china/AAL/china_open_AAL_strategy-1_GSR.npy")

print(f"China close: {china_close.shape}")  # (48, 240, 116, 2)
print(f"China open: {china_open.shape}")    # (48, 240, 116)

# Access China sessions
session1 = china_close[:, :, :, 0]  # First EC session
session2 = china_close[:, :, :, 1]  # Second EC session
```

### Load Subject Order

```python
# IHB subjects
with open(data_root / "timeseries_ihb/AAL/subject_order.txt") as f:
    ihb_subjects = [line.strip() for line in f]

print(f"IHB subjects: {len(ihb_subjects)}")  # 84
print(f"First subject: {ihb_subjects[0]}")   # sub-001

# China subjects
with open(data_root / "timeseries_china/AAL/subject_order_china.txt") as f:
    china_subjects = [line.strip() for line in f]

print(f"China subjects: {len(china_subjects)}")  # 48
```

### Load Coverage Data

```python
# Load coverage for ROI filtering
ihb_coverage = np.load(data_root / "coverage/ihb_AAL_parcel_coverage.npy")
china_coverage = np.load(data_root / "coverage/china_AAL_parcel_coverage.npy")

# Apply coverage threshold
threshold = 0.1
ihb_good_rois = ihb_coverage >= threshold
china_good_rois = china_coverage >= threshold
both_good = ihb_good_rois & china_good_rois

print(f"IHB good ROIs: {ihb_good_rois.sum()} / {len(ihb_coverage)}")
print(f"Both sites good: {both_good.sum()} / {len(ihb_coverage)}")

# Filter time series
ihb_filtered = ihb_close[:, :, both_good]
print(f"Filtered shape: {ihb_filtered.shape}")  # (84, 120, 106)
```

### Load Precomputed Glasso

```python
# IHB glasso
ihb_glasso = np.load(data_root / "glasso_precomputed_fc/ihb/AAL/ihb_close_AAL_strategy-1_GSR_glasso.npy")
print(f"IHB glasso: {ihb_glasso.shape}")  # (84, 5565)

# China glasso (with sessions)
china_glasso = np.load(data_root / "glasso_precomputed_fc/china/AAL/china_close_AAL_strategy-1_GSR_glasso.npy")
print(f"China glasso: {china_glasso.shape}")  # (48, 5565, 2)

# Access sessions
glasso_session1 = china_glasso[:, :, 0]
glasso_session2 = china_glasso[:, :, 1]
```

### Compute FC from Time Series

```python
from data_utils.fc import ConnectomeTransformer

# Pearson correlation
transformer = ConnectomeTransformer(kind='corr', vectorize=True)
fc_corr = transformer.fit_transform(ihb_close)
print(f"FC correlation: {fc_corr.shape}")  # (84, 6670)  # full 116 ROIs (no coverage masking)

# Partial correlation
transformer = ConnectomeTransformer(kind='partial', vectorize=True)
fc_partial = transformer.fit_transform(ihb_close)

# Tangent space (requires fit on training data)
transformer = ConnectomeTransformer(kind='tangent', vectorize=True)
fc_train = transformer.fit_transform(ihb_close[:60])
fc_test = transformer.transform(ihb_close[60:])
```


## Data Dimensions Quick Reference

| Component | IHB | China | Notes |
|-----------|-----|-------|-------|
| **Subjects** | 84 | 48 | After QC |
| **Timepoints** | 120 | 240 | TR: 2.5s / 2.0s |
| **Conditions** | 1 EC + 1 EO | 2 EC + 1 EO | China has repeat EC |
| **Strategies** | 16 | 16 | 6 standard + 2 AROMA × 2 GSR |
| **Atlases** | 4 | 4 | AAL, Schaefer200, Brainnetome, HCPex |
| **Files per atlas** | 32 | 32 | Time series .npy files |
| **Total pipelines** | 192 | 192 | 4 atlases × 16 strategies × 3 FC types (corr/partial/tangent) + glasso |

### ROI Counts by Atlas

| Atlas | Standard Strategies | AROMA Strategies | After HCPex Preprocessing |
|-------|---------------------|------------------|---------------------------|
| AAL | IHB: 116, China: 116 | IHB: 116, China: 116 | N/A |
| Schaefer200 | IHB: 200, China: 200 | IHB: 200, China: 200 | N/A |
| Brainnetome | IHB: 246, China: 246 | IHB: 246, China: 246 | N/A |
| HCPex | IHB: 423, China: 421 | IHB: 426, China: 426 | **373 (both sites)** |

## Citation

If you use this data in your research, please cite:

```bibtex
@article{medvedeva2025benchmarking,
  title={Benchmarking resting state fMRI connectivity pipelines for classification: Robust accuracy despite processing variability in cross-site eye state prediction},
  author={Medvedeva, Tatiana and Knyazeva, Irina and Masharipov, Ruslan and Korotkov, Alexander and Cherednichenko, Denis and Kireev, Maxim},
  journal={Neuroscience},
  year={2025},
  doi={10.1101/2025.10.20.683049}
}
```
