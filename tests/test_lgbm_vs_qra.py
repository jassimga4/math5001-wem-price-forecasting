"""Leakage checks for the LightGBM lag-feature baseline in scripts/lgbm_vs_qra.py."""
import unittest

import numpy as np
import pandas as pd

from scripts.lgbm_vs_qra import EXTRA_LAGS, lag_features_from_grid


class LagFeatureTests(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2025-01-01", periods=3000, freq="5min")
        rng = np.random.default_rng(0)
        self.grid = pd.Series(rng.normal(100, 20, len(idx)), index=idx)

    def test_features_do_not_change_when_target_or_future_changes(self):
        base = lag_features_from_grid(self.grid)
        t = self.grid.index[2500]
        changed = self.grid.copy()
        changed.loc[t:] = changed.loc[t:] * 10 + 999  # target interval and everything after
        moved = lag_features_from_grid(changed)
        pd.testing.assert_series_equal(base.loc[t], moved.loc[t], check_names=False)

    def test_lags_are_time_offsets(self):
        feats = lag_features_from_grid(self.grid)
        t = self.grid.index[2400]
        for name, steps in EXTRA_LAGS.items():
            self.assertEqual(feats.loc[t, name], self.grid.iloc[2400 - steps])
        self.assertAlmostEqual(feats.loc[t, "mcp_roll_max_30min"], self.grid.iloc[2394:2400].max())
        self.assertAlmostEqual(feats.loc[t, "mcp_roll_mean_60min"], self.grid.iloc[2388:2400].mean())


if __name__ == "__main__":
    unittest.main()
