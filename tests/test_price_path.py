"""Price-path features may read only prices realised by the origin (interval T - 5 min or earlier)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_design import CONTEMPORANEOUS_BANNED, HORIZON  # noqa: E402
from scripts.price_path import PATH_FEATURES, price_path_features  # noqa: E402


def prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n, freq="5min")
    return pd.Series(60 + np.cumsum(rng.normal(0, 15, n)), index=index)


class PricePathTests(unittest.TestCase):
    def setUp(self):
        self.p = prices()
        self.targets = self.p.index[30:380:7]
        self.base = price_path_features(self.p, self.targets, near_up=120.0, near_down=0.0)

    def test_prices_at_or_after_the_target_cannot_change_a_feature(self):
        for target in self.targets:
            changed = self.p.copy()
            changed.loc[changed.index >= target] = 1e6  # the target interval and everything later
            again = price_path_features(changed, pd.DatetimeIndex([target]), near_up=120.0, near_down=0.0)
            pd.testing.assert_series_equal(self.base.loc[target], again.loc[target], check_names=False)

    def test_dropping_prices_after_the_origin_changes_nothing(self):
        for target in self.targets:
            origin = target - HORIZON
            # the last realised interval at the origin is labelled T - 5 min = origin
            again = price_path_features(self.p.loc[: origin], pd.DatetimeIndex([target]), near_up=120.0, near_down=0.0)
            pd.testing.assert_series_equal(self.base.loc[target], again.loc[target], check_names=False)

    def test_the_last_realised_price_is_used(self):
        target = self.targets[10]
        changed = self.p.copy()
        changed.loc[target - HORIZON] += 500.0
        again = price_path_features(changed, pd.DatetimeIndex([target]), near_up=120.0, near_down=0.0)
        self.assertNotEqual(self.base.loc[target, "path_slope_15"], again.loc[target, "path_slope_15"])

    def test_known_path(self):
        index = pd.date_range("2025-01-01", periods=40, freq="5min")
        ramp = pd.Series(np.arange(40.0), index=index)
        got = price_path_features(ramp, index[[20]], near_up=17.0, near_down=-1.0).iloc[0]
        self.assertAlmostEqual(got["path_slope_15"], 1.0)
        self.assertEqual(got["path_climb_run"], 19.0)  # prices 0..19 rise every step before T
        self.assertEqual(got["path_fall_run"], 0.0)
        self.assertEqual(got["path_near_up_30"], 3.0)  # 17, 18, 19 among 14..19
        self.assertEqual(got["path_max60_minus_last"], 0.0)

    def test_names_are_not_mcp_lags_or_same_interval_columns(self):
        self.assertFalse(CONTEMPORANEOUS_BANNED.intersection(PATH_FEATURES))
        self.assertFalse(any(f.startswith("mcp_lag") for f in PATH_FEATURES))


if __name__ == "__main__":
    unittest.main()
