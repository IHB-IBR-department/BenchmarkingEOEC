# HCPex Atlas Integration - Summary of Changes

## Problem Statement

HCPex atlas has varying ROI dimensions across sites due to nilearn's quality filtering:
- **IHB:** 423 ROIs (3 ROIs removed from 426)
- **China:** 421 ROIs (5 ROIs removed from 426)

This prevents direct functional connectivity computation and comparison across sites, as FC matrices must have matching dimensions.

## Solution Overview

Created a preprocessing pipeline that:
1. Identifies ROIs to exclude based on coverage and nilearn filtering
2. Applies consistent masking to produce 373 ROIs for both sites
3. Integrates automatically into existing FC computation workflows

## Files Created

### 1. **`benchmarking/create_hcpex_mask.py`**
Script to generate the unified HCPex mask.

**Functionality:**
- Loads coverage data from `ihb_HCPex_parcel_coverage.npy`
- Identifies ROIs with coverage < 0.1 threshold
- Loads skipped ROI lists from text files
- Combines all exclusions into `hcp_mask.npy`

**Output:**
- `coverage/hcp_mask.npy`: Boolean array (426,) where True = bad ROI
- **53 bad ROIs:** 47 from low coverage + 6 explicitly skipped
- **373 good ROIs** remain

**Usage:**
```bash
source venv/bin/activate && PYTHONPATH=. python -m benchmarking.create_hcpex_mask --threshold 0.1
```

### 2. **`benchmarking/hcpex_preprocess.py`**
Core preprocessing module with reusable functions.

**Functions:**
- `preprocess_hcpex_timeseries()`: Main preprocessing function
- `load_skipped_rois()`: Load skipped ROI indices from text files
- `adjust_mask_for_site()`: Adjust 426-dim mask to site-specific dimensions
- `apply_roi_mask()`: Apply mask to time series data
- `verify_consistent_dimensions()`: Verify consistency across sites

**Key Features:**
- Handles both 3D (IHB) and 4D (China with sessions) arrays
- Auto-detects AROMA vs standard strategies by ROI count
  - AROMA (426 ROIs): Apply mask directly
  - Standard (423/421 ROIs): Adjust mask for site-specific skipped ROIs
- Site-aware mask adjustment (removes skipped indices)
- Input validation and error handling

**Example:**
```python
from benchmarking.hcpex_preprocess import preprocess_hcpex_timeseries

ihb_data = np.load("ihb_close_HCPex_strategy-1_GSR.npy")  # (84, 120, 423)
preprocessed = preprocess_hcpex_timeseries(ihb_data, site="ihb", mask_path="coverage/hcp_mask.npy")
# Result: (84, 120, 373)
```

### 3. **`benchmarking/hcpex_example.py`**
Demonstration script showing preprocessing usage.

**Demonstrates:**
- Loading raw time series from both sites
- Applying preprocessing
- Verifying consistent dimensions
- Preparing data for FC computation

### 4. **`run_hcpex_glasso.sh`**
Convenience script for batch glasso computation.

**Runs:**
- IHB HCPex: All strategies with preprocessing
- China HCPex: All strategies with preprocessing
- Saves to `glasso_precomputed_fc/` with proper directory structure

### 5. **`HCPEX_GLASSO_GUIDE.md`**
Comprehensive documentation for HCPex processing.

**Covers:**
- Overview and file descriptions
- Usage examples (single file and batch)
- Output file structure
- Performance notes
- Verification procedures
- ICC data preparation

## Files Modified

### 1. **`benchmarking/precompute_glasso.py`**

**Added:**
- `import re` for pattern matching
- `from benchmarking.hcpex_preprocess import preprocess_hcpex_timeseries`
- `from benchmarking.project import resolve_data_root`

**New Functions:**
- `parse_atlas_and_site()`: Extract atlas/site from file path
- `preprocess_timeseries_if_needed()`: Automatically detect and preprocess HCPex

**Modified:**
- Line 286-288: Added HCPex preprocessing before glasso computation
- Line 291: Skip coverage masking if HCPex preprocessing was applied

**Behavior:**
- Automatically detects HCPex atlas from filename or parent directory
- Applies preprocessing transparently (423/421 → 373 ROIs)
- Returns tuple `(preprocessed_data, was_hcpex_preprocessed)`
- Coverage masking skipped for HCPex (already included in preprocessing)

### 2. **`benchmarking/icc_data_preparation.py`**

**Added:**
- `from benchmarking.hcpex_preprocess import preprocess_hcpex_timeseries`

**New Function:**
- `preprocess_hcpex_if_needed()`: Apply HCPex preprocessing for China data

**Modified:**
- Line 426: Added HCPex preprocessing in folder processing mode
- Line 645: Added HCPex preprocessing in single-file mode

**Integration Points:**
- Preprocessing applied after loading time series
- Preprocessing applied before subject filtering
- Works with existing glasso lookup in `glasso_precomputed_fc/china/HCPex/`

## Directory Structure

### Coverage Files
```
coverage/
├── hcp_mask.npy                          # 426-dim unified mask (53 bad, 373 good)
├── ihb_HCPex_parcel_coverage.npy         # IHB coverage values
├── china_HCPex_parcel_coverage.npy       # China coverage values
├── ihb_skipped_rois_HCPex.txt            # 365, 398, 401
└── china_skipped_rois_HCPex.txt          # 365, 372, 396, 401, 405
```

### Glasso Precomputed FC
```
glasso_precomputed_fc/
├── ihb/
│   └── HCPex/
│       ├── ihb_close_HCPex_strategy-1_GSR_glasso.npy     # (84, 69378)
│       ├── ihb_open_HCPex_strategy-1_GSR_glasso.npy      # (84, 69378)
│       └── ... (32 files total)
└── china/
    └── HCPex/
        ├── china_close_HCPex_strategy-1_GSR_glasso.npy   # (48, 69378, 2)
        ├── china_open_HCPex_strategy-1_GSR_glasso.npy    # (48, 69378)
        └── ... (32 files total)
```

**Edge count:** 69378 = 373 × 372 / 2 (upper triangle without diagonal)

### ICC Precomputed FC
```
icc_precomputed_fc/
└── HCPex/
    ├── china_close_HCPex_strategy-1_GSR_corr.npy
    ├── china_close_HCPex_strategy-1_GSR_pc.npy
    ├── china_close_HCPex_strategy-1_GSR_tang.npy
    ├── china_close_HCPex_strategy-1_GSR_glasso.npy
    └── ... (all strategies)
```

## ROI Breakdown

### Original Dimensions
- **HCPex atlas nominal:** 426 ROIs
- **Standard strategies:**
  - IHB after nilearn filtering: 423 ROIs (removed 3)
  - China after nilearn filtering: 421 ROIs (removed 5)
- **AROMA strategies:**
  - IHB: 426 ROIs (no ROIs removed)
  - China: 426 ROIs (no ROIs removed)

### Excluded ROIs
- **Low coverage (< 0.1):** 47 ROIs
- **IHB skipped by nilearn:** 365, 398, 401
- **China skipped by nilearn:** 365, 372, 396, 401, 405
- **Unique skipped:** 365, 372, 396, 398, 401, 405 (6 total)
- **Total excluded:** 53 ROIs (some overlap between coverage and skipped)

### Final Dimensions
- **Both IHB and China:** 373 ROIs
- **Edge count:** 69,378 (373 × 372 / 2)
- **Consistency:** ✓ Matching dimensions across sites

## Workflow Integration

### Glasso Computation
```bash
# Single file - automatic HCPex detection
python -m benchmarking.precompute_glasso \
  --input timeseries_ihb/HCPex/ihb_close_HCPex_strategy-1_GSR.npy \
  --output-dir glasso_precomputed_fc/ihb/HCPex

# Batch processing
bash run_hcpex_glasso.sh
```

### ICC Preparation
```bash
# Automatic HCPex preprocessing + glasso lookup
python -m benchmarking.icc_data_preparation \
  --atlas HCPex \
  --input-dir timeseries_china/HCPex \
  --output-dir icc_precomputed_fc \
  --kinds corr pc tang glasso
```

## Key Features

1. **Automatic Detection:** Scripts detect HCPex from file paths
2. **Transparent Processing:** No special flags needed for HCPex
3. **Consistent Interface:** Same commands work for all atlases
4. **Skip Logic:** Already-processed files are automatically skipped
5. **Error Handling:** Graceful fallback if mask is missing
6. **Validation:** Built-in dimension verification

## Verification

All preprocessing verified with test scripts:
- ✓ IHB standard strategies: 423 → 373 ROIs
- ✓ China standard strategies: 421 → 373 ROIs
- ✓ IHB AROMA strategies: 426 → 373 ROIs
- ✓ China AROMA strategies: 426 → 373 ROIs
- ✓ Edge count: 69,378 for all strategies and sites
- ✓ Glasso computation successful
- ✓ Skip logic working correctly
- ✓ ICC preparation compatible

## Performance

- **Glasso computation:** ~4-5 minutes per file (84 subjects)
- **Total HCPex files:** 64 files (32 IHB + 32 China)
- **Estimated total time:** ~5 hours for full batch processing
- **Resume support:** Already-computed files are skipped

## Usage Summary

### Create the mask (one-time):
```bash
python -m benchmarking.create_hcpex_mask --threshold 0.1
```

### Compute glasso (IHB and China):
```bash
bash run_hcpex_glasso.sh
```

### Prepare ICC data (China only):
```bash
python -m benchmarking.icc_data_preparation \
  --atlas HCPex \
  --kinds corr pc tang glasso
```

All scripts now work seamlessly with HCPex atlas! 🎉
