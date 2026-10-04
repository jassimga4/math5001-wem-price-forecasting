import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.metrics import crps_from_quantiles
from scripts.qra import (
    QRA_FORECASTS,
    comparison_levels,
    empirical_quantile_levels,
    fit_qra,
    level_name,
    predict_qra,
    run_qra_comparison,
)


class QraTests(unittest.TestCase):
    def test_quantile_integral_is_zero_for_a_point_mass_at_the_outcome(self):
        levels = np.array([0.25, 0.5, 0.75])
        grid = np.zeros((2, 3))
        score = crps_from_quantiles(np.array([0.0, 0.0]), grid, levels)
        self.assertTrue(np.allclose(score, 0.0))

    def test_quantile_integral_matches_a_two_level_trapezoid(self):
        levels = np.array([0.25, 0.75])
        quantiles = np.array([[0.0, 0.0]])
        # y - q = 1, pinball at 0.25 is 0.75, at 0.75 is 0.25.
        # Twice that is 1.5 and 0.5. Trapezoid over width 0.5 is 0.5.
        score = crps_from_quantiles(np.array([1.0]), quantiles, levels)
        self.assertTrue(np.allclose(score, 0.5))

    def test_empirical_quantile_uses_the_ceil_rank_of_a_sorted_sample(self):
        samples = np.array([[10.0, 20.0, 30.0, 40.0]])
        grid = empirical_quantile_levels(samples, np.array([0.25, 0.5]))
        self.assertEqual(grid[0, 0], 10.0)
        self.assertEqual(grid[0, 1], 20.0)

    def test_crossed_quantiles_are_sorted_without_using_the_outcome(self):
        index = pd.RangeIndex(4)
        forecasts = pd.DataFrame({name: np.zeros(4) for name in QRA_FORECASTS}, index=index)
        coefficients = pd.DataFrame(
            [
                {"level": 0.1, "converged": True, "intercept": 5.0, **{name: 0.0 for name in QRA_FORECASTS}},
                {"level": 0.9, "converged": True, "intercept": 1.0, **{name: 0.0 for name in QRA_FORECASTS}},
            ]
        )
        predicted, share = predict_qra(forecasts, coefficients)
        self.assertEqual(share, 1.0)
        self.assertTrue((predicted[level_name(0.1)] < predicted[level_name(0.9)]).all())
        self.assertTrue(predicted[level_name(0.1)].eq(1.0).all())
        self.assertTrue(predicted[level_name(0.9)].eq(5.0).all())

    def test_one_informative_forecast_tracks_a_gaussian_quantile(self):
        rng = np.random.default_rng(7)
        n = 4000
        signal = rng.normal(size=n)
        y = pd.Series(signal + rng.normal(scale=1.0, size=n))
        forecasts = pd.DataFrame({name: rng.normal(size=n) for name in QRA_FORECASTS})
        forecasts["lightgbm"] = signal
        coefficients = fit_qra(forecasts, y, levels=np.array([0.1, 0.9]))
        self.assertTrue(bool(coefficients["converged"].all()))
        predicted, _share = predict_qra(forecasts, coefficients)
        oracle = signal + 1.2815515655446004
        self.assertLess(float(np.mean(np.abs(predicted[level_name(0.9)] - oracle))), 0.15)

    def test_fit_uses_only_the_rows_it_is_given(self):
        rng = np.random.default_rng(2)
        n = 300
        forecasts = pd.DataFrame({name: rng.normal(size=n) for name in QRA_FORECASTS})
        y = pd.Series(forecasts["persistence_5min"] + rng.normal(scale=0.4, size=n))
        first = fit_qra(forecasts.iloc[:200], y.iloc[:200], levels=np.array([0.2, 0.8]))
        second = fit_qra(forecasts.iloc[:200], y.iloc[:200], levels=np.array([0.2, 0.8]))
        pd.testing.assert_frame_equal(first, second)
        altered = y.iloc[:200].copy()
        altered.iloc[-1] = 999.0
        third = fit_qra(forecasts.iloc[:200], altered, levels=np.array([0.2, 0.8]))
        self.assertFalse(np.allclose(first["intercept"], third["intercept"]))

    def test_comparison_scores_qra_and_frozen_conformal_on_the_same_origins(self):
        rng = np.random.default_rng(3)
        index = pd.date_range("2024-01-01", periods=80, freq="5min")
        signal = pd.Series(rng.normal(size=80), index=index)
        y = signal + rng.normal(scale=0.3, size=80)
        frame = pd.DataFrame({"mcp": y}, index=index)
        forecasts = {name: signal + rng.normal(scale=0.2, size=80) for name in QRA_FORECASTS}
        forecasts["lightgbm"] = signal
        forecasts["persistence_5min"] = signal.shift(1).fillna(signal.iloc[0])
        predictions = {name: pd.Series(values, index=index) for name, values in forecasts.items()}
        result = {
            "calibration": frame.iloc[:50],
            "test": frame.iloc[50:],
            "predictions": predictions,
            "spike_cut": float(y.iloc[:40].quantile(0.95)),
            "floor_cut": float(y.iloc[:40].quantile(0.05)),
        }
        compared = run_qra_comparison(
            result,
            primary_method="absolute",
            primary_window=12,
            frozen_source="test fixture",
            lightgbm_source="test fixture",
        )
        table = compared["test"]
        self.assertEqual(set(table["role"]), {"qra", "frozen_conformal"})
        self.assertEqual(int(table.loc[table["role"].eq("qra"), "n"].iloc[0]), 30)
        qra_90 = table.loc[table["role"].eq("qra") & np.isclose(table["alpha"], 0.10)].iloc[0]
        self.assertGreaterEqual(qra_90["mean_width"], 0.0)
        self.assertGreaterEqual(qra_90["coverage"], 0.0)
        self.assertLessEqual(qra_90["coverage"], 1.0)
        crps = compared["crps"].set_index(["method", "base"])
        self.assertTrue(np.isnan(crps.loc[("qra", "qra"), "crps_empirical"]))
        self.assertTrue(np.isfinite(crps.loc[("qra", "qra"), "crps_quantile_integral"]))
        self.assertTrue(np.isfinite(crps.loc[("absolute", "lightgbm"), "crps_empirical"]))
        self.assertTrue(np.isfinite(crps.loc[("absolute", "lightgbm"), "crps_quantile_integral"]))
        self.assertIn(0.05, comparison_levels())
        self.assertIn(0.95, comparison_levels())


if __name__ == "__main__":
    unittest.main()
