import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_panel import trading_interval_end
from scripts.conformal import empirical_quantile, lower_rank, upper_rank
from scripts.forecast_design import (
    CONTEMPORANEOUS_BANNED,
    HORIZON,
    RIDGE_FEATURES,
    TREE_FEATURES,
    build_exante_frame,
    reindex_5min,
    split_mask,
    time_lag,
)
from scripts.metrics import crps_from_residual_samples
from scripts.paths import as_posix_path


class ForecastDesignTests(unittest.TestCase):
    def test_reindex_inserts_gap_and_lag_is_timestamp_based(self):
        index = pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 00:05:00", "2024-01-01 00:15:00"]
        )
        raw = pd.DataFrame({"interval_end": index, "mcp": [10.0, 20.0, 40.0]})
        panel = reindex_5min(raw)
        self.assertEqual(len(panel), 4)
        self.assertTrue(panel.index.is_unique)
        self.assertTrue(panel.index.is_monotonic_increasing)
        self.assertEqual(panel.index.to_series().diff().dropna().unique().tolist(), [pd.Timedelta(minutes=5)])
        lagged = time_lag(panel["mcp"], 1)
        self.assertEqual(lagged.iloc[2], 20.0)
        self.assertTrue(np.isnan(lagged.iloc[3]))

    def test_trading_interval_mapping(self):
        stamps = pd.to_datetime(
            [
                "2024-01-01 08:00:00",
                "2024-01-01 08:05:00",
                "2024-01-01 08:30:00",
                "2024-01-01 08:35:00",
                "2024-01-01 08:55:00",
            ]
        )
        mapped = trading_interval_end(pd.Series(stamps))
        expected = pd.to_datetime(
            [
                "2024-01-01 08:00:00",
                "2024-01-01 08:30:00",
                "2024-01-01 08:30:00",
                "2024-01-01 09:00:00",
                "2024-01-01 09:00:00",
            ]
        )
        self.assertTrue(mapped.reset_index(drop=True).equals(pd.Series(expected)))

    def test_rtp_is_not_same_trading_interval(self):
        rows = []
        start = pd.Timestamp("2024-01-01 08:00:00")
        for step in range(12):
            interval = start + pd.Timedelta(minutes=5 * step)
            minute = interval.minute
            trading = interval.floor("h") + pd.Timedelta(minutes=30)
            if minute == 0:
                trading = interval
            elif minute > 30:
                trading = interval.floor("h") + pd.Timedelta(hours=1)
            rows.append(
                {
                    "interval_end": interval,
                    "trading_interval_end": trading,
                    "mcp": float(step),
                    "rtp": 100.0 if trading.minute == 0 and trading.hour == 8 else 200.0,
                    "operational_demand_mw": 1000.0,
                    "operational_withdrawal_mw": 1.0,
                    "dpv_mw": 10.0,
                    "cr_raise": 1.0,
                    "cr_lower": 1.0,
                    "reg_raise": 1.0,
                    "reg_lower": 1.0,
                    "stem_price": 50.0,
                    "stem_bid_mwh": 10.0,
                    "stem_offer_mwh": 12.0,
                }
            )
        frame = build_exante_frame(pd.DataFrame(rows))
        at_0810 = frame.loc[pd.Timestamp("2024-01-01 08:10:00"), "rtp_last_complete"]
        self.assertEqual(at_0810, 100.0)
        at_0835 = frame.loc[pd.Timestamp("2024-01-01 08:35:00"), "rtp_last_complete"]
        self.assertEqual(at_0835, 200.0)

    def test_split_dates_do_not_overlap(self):
        index = pd.date_range("2025-09-30 23:50:00", "2026-04-01 00:05:00", freq="5min")
        labels = split_mask(index)
        self.assertFalse(set(labels.unique()) - {"train", "calibration", "test"})
        self.assertLess(index[labels.eq("train")].max(), index[labels.eq("calibration")].min())
        self.assertLess(index[labels.eq("calibration")].max(), index[labels.eq("test")].min())

    def test_feature_set_has_no_contemporaneous_target_inputs(self):
        self.assertFalse(CONTEMPORANEOUS_BANNED.intersection(RIDGE_FEATURES + TREE_FEATURES))
        self.assertNotIn("year", RIDGE_FEATURES + TREE_FEATURES)

    def test_lag_timestamp_matches_horizon(self):
        index = pd.date_range("2024-06-01", periods=300, freq="5min")
        raw = pd.DataFrame(
            {
                "interval_end": index,
                "trading_interval_end": index,
                "mcp": np.arange(len(index), dtype=float),
                "rtp": np.arange(len(index), dtype=float),
                "operational_demand_mw": 1.0,
                "operational_withdrawal_mw": 0.0,
                "dpv_mw": 0.0,
                "cr_raise": 0.0,
                "cr_lower": 0.0,
                "reg_raise": 0.0,
                "reg_lower": 0.0,
                "stem_price": 1.0,
                "stem_bid_mwh": 1.0,
                "stem_offer_mwh": 1.0,
            }
        )
        frame = build_exante_frame(raw)
        ts = frame.index[100]
        self.assertEqual(frame.loc[ts, "mcp_lag_5min"], frame.loc[ts - HORIZON, "mcp"])
        self.assertEqual(frame.loc[ts, "forecast_origin"], ts - HORIZON)

    def test_windows_absolute_path_is_not_joined_to_the_repo(self):
        path = as_posix_path("C:/data/raw")
        self.assertTrue(path.is_absolute())
        self.assertNotIn("math5001", str(path).lower())

    def test_conformal_rank_widens_small_samples(self):
        scores = np.arange(10, dtype=float)
        self.assertEqual(upper_rank(10, 0.2), 9)
        self.assertLess(empirical_quantile(scores, 0.2, "lower"), empirical_quantile(scores, 0.2, "upper"))

    def test_crps_is_zero_for_a_point_mass_at_the_observation(self):
        samples = np.zeros((2, 4))
        scores = crps_from_residual_samples(samples, y=np.array([3.0, 3.0]), yhat=np.array([3.0, 3.0]))
        self.assertTrue(np.allclose(scores, 0.0))


if __name__ == "__main__":
    unittest.main()
