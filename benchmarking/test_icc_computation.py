import unittest
from pathlib import Path

import numpy as np

from benchmarking.icc_computation import (
    compute_icc_edgewise,
    compute_icc_mask,
    compute_icc_summary,
)
from benchmarking.project import resolve_data_root


class TestICCComputation(unittest.TestCase):
    def test_icc_perfect_consistency(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(12, 5))
        data = np.stack([base, base], axis=-1)

        icc31 = compute_icc_edgewise(data, "icc31")
        icc21 = compute_icc_edgewise(data, "icc21")
        icc11 = compute_icc_edgewise(data, "icc11")

        np.testing.assert_allclose(icc31, 1.0, atol=1e-6)
        np.testing.assert_allclose(icc21, 1.0, atol=1e-6)
        np.testing.assert_allclose(icc11, 1.0, atol=1e-6)

    def test_icc_session_shift_consistency_vs_agreement(self) -> None:
        rng = np.random.default_rng(1)
        base = np.linspace(0.0, 1.0, 20)[:, None]
        base = base + rng.normal(scale=0.01, size=(20, 3))
        shift = 5.0
        data = np.stack([base, base + shift], axis=-1)

        icc31 = compute_icc_edgewise(data, "icc31")
        icc21 = compute_icc_edgewise(data, "icc21")

        self.assertTrue(np.all(icc31 > 0.999))
        self.assertTrue(np.all(icc21 < 0.2))

    def test_icc_inconsistent_data_low_reliability(self) -> None:
        rng = np.random.default_rng(2)
        session1 = rng.normal(size=(50, 10))
        session2 = rng.normal(size=(50, 10))
        data = np.stack([session1, session2], axis=-1)

        icc31 = compute_icc_edgewise(data, "icc31")
        icc21 = compute_icc_edgewise(data, "icc21")
        icc11 = compute_icc_edgewise(data, "icc11")

        self.assertLess(np.nanmean(np.abs(icc31)), 0.2)
        self.assertLess(np.nanmean(np.abs(icc21)), 0.2)
        self.assertLess(np.nanmean(np.abs(icc11)), 0.2)

    def test_icc_output_shape(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.normal(size=(10, 100, 2))

        icc31 = compute_icc_edgewise(data, "icc31")

        self.assertEqual(icc31.shape, (100,))

    def test_icc_mask_global_threshold(self) -> None:
        data = np.zeros((2, 3, 2), dtype=float)
        data[:, 1, :] = 1.0
        data[:, 2, :] = 10.0

        mask = compute_icc_mask(data, percentile=50)

        abs_data = np.abs(data)
        edge_means = abs_data.mean(axis=(0, 2))
        global_threshold = np.percentile(abs_data, 50)
        expected = edge_means >= global_threshold

        np.testing.assert_array_equal(mask, expected)

    def test_icc_summary_real_data_consistency(self) -> None:
        data_root = Path(resolve_data_root(None))
        path = (
            data_root
            / "icc_precomputed_fc"
            / "AAL"
            / "china_close_AAL_strategy-1_GSR_corr.npy"
        )
        if not path.exists():
            self.skipTest(f"Missing test data: {path}")

        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 3:
            first = np.asarray(arr[..., 0])
        elif arr.ndim == 2:
            first = np.asarray(arr)
        else:
            self.skipTest(f"Unexpected shape for test data: {arr.shape}")

        data = np.stack([first, first], axis=-1)
        summary = compute_icc_summary(data, ["icc31", "icc21", "icc11"], percentile=98)

        for icc_kind, stats in summary.items():
            self.assertTrue(np.isfinite(stats["mean"]))
            self.assertTrue(np.isfinite(stats["mean_masked"]))
            self.assertAlmostEqual(stats["mean"], 1.0, places=6)
            self.assertAlmostEqual(stats["mean_masked"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
