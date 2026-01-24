import numpy as np
from pathlib import Path

data_root = Path("~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data").expanduser()
ts_path = data_root / "timeseries_china/HCPex/china_close_HCPex_strategy-1_GSR.npy"
coverage_path = data_root / "coverage/ihb_HCPex_parcel_coverage.npy"

print(f"Loading {ts_path}...")
try:
    ts = np.load(ts_path)
    print(f"Timeseries shape: {ts.shape}")
except Exception as e:
    print(f"Failed to load TS: {e}")

print(f"Loading coverage {coverage_path}...")
try:
    cov = np.load(coverage_path)
    print(f"Coverage shape: {cov.shape}")
except Exception as e:
    print(f"Failed to load coverage: {e}")
