from __future__ import annotations

import os
from pathlib import Path
DATA_ROOT_ENV = "OPEN_CLOSE_BENCHMARK_DATA"
DEFAULT_DATA_ROOT = Path("~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data")


def resolve_data_root(data_path: str | Path | None = None) -> Path:
    """
    Resolve the OpenCloseBenchmark_data root directory.

    Resolution order:
    1) explicit `data_path` argument
    2) env var `OPEN_CLOSE_BENCHMARK_DATA`
    3) default `~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data`
    """
    if data_path is None or str(data_path).strip() == "":
        data_path = os.environ.get(DATA_ROOT_ENV, str(DEFAULT_DATA_ROOT))
    return Path(os.path.expanduser(str(data_path)))


__all__ = ["DATA_ROOT_ENV", "DEFAULT_DATA_ROOT", "resolve_data_root"]
