"""
BenchmarkingEOEC: benchmarking utilities and experiment scripts.

Data layout and naming conventions live in `.claude/skills/data_handling.md`.
Use `benchmarking.project.resolve_data_root()` to standardize data-root handling.
"""

from .project import (
    DATA_ROOT_ENV,
    DEFAULT_DATA_ROOT,
    FC_NAME_MAP,
    glasso_path,
    resolve_data_root,
    standard_fc_filename,
    standard_fc_path,
)

__all__ = [
    "DATA_ROOT_ENV",
    "DEFAULT_DATA_ROOT",
    "FC_NAME_MAP",
    "resolve_data_root",
    "standard_fc_filename",
    "standard_fc_path",
    "glasso_path",
]
