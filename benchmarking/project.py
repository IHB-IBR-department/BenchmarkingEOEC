from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

Site = Literal["china", "ihb"]
FCType = Literal["corr", "partial", "tangent", "glasso"]

DATA_ROOT_ENV = "OPEN_CLOSE_BENCHMARK_DATA"
DEFAULT_DATA_ROOT = Path("~/Yandex.Disk.localized/IHB/OpenCloseBenchmark_data")

# IMPORTANT: standard FC file naming differs between sites.
# See `.claude/skills/data_handling.md`.
FC_NAME_MAP: dict[Site, dict[Literal["corr", "partial", "tangent"], str]] = {
    "china": {"corr": "corr", "partial": "pc", "tangent": "tang"},
    "ihb": {"corr": "corr", "partial": "partial_corr", "tangent": "tangent"},
}


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


def standard_fc_filename(
    *,
    site: Site,
    session_or_condition: str,
    fc_type: Literal["corr", "partial", "tangent"],
    atlas: str,
    strategy: int,
    gsr: str,
) -> str:
    if site not in FC_NAME_MAP:
        raise ValueError(f"Unknown site: {site}. Expected one of: {sorted(FC_NAME_MAP)}")
    site_prefix = "china" if site == "china" else "ihb"
    fc_name = FC_NAME_MAP[site][fc_type]
    return f"{site_prefix}_{session_or_condition}_{fc_name}_{atlas}_strategy-{strategy}_{gsr}.npy"


def standard_fc_path(
    *,
    data_root: str | Path,
    site: Site,
    session_or_condition: str,
    fc_type: Literal["corr", "partial", "tangent"],
    atlas: str,
    strategy: int,
    gsr: str,
) -> Path:
    data_root = resolve_data_root(data_root)
    if site not in FC_NAME_MAP:
        raise ValueError(f"Unknown site: {site}. Expected one of: {sorted(FC_NAME_MAP)}")
    base_dir = "fc_data_china" if site == "china" else "fc_data_ihb"
    split_dir = "open" if session_or_condition.startswith("open") else "close"
    return (
        data_root
        / base_dir
        / split_dir
        / standard_fc_filename(
            site=site,
            session_or_condition=session_or_condition,
            fc_type=fc_type,
            atlas=atlas,
            strategy=strategy,
            gsr=gsr,
        )
    )


def glasso_path(
    *,
    data_root: str | Path,
    site: Site,
    atlas: str,
    strategy: int,
    gsr: str,
) -> Path:
    data_root = resolve_data_root(data_root)
    if site not in ("china", "ihb"):
        raise ValueError(f"Unknown site: {site}. Expected 'china' or 'ihb'")
    if site == "ihb":
        return data_root / "glasso_output" / "ihb" / f"ihb_{atlas}_{strategy}_{gsr}.npy"
    return data_root / "glasso_output" / "ch_corrected" / f"china_{atlas}_{strategy}_{gsr}.npy"
