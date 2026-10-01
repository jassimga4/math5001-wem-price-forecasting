import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.conformal import (
    ALPHAS,
    WINDOW_CANDIDATES,
    _mark_selected,
    apply_sliding,
    choose_primary,
    lagged_scale,
    lower_rank,
    selection_table,
    select_window,
    upper_rank,
)
from scripts.metrics import crps_from_residual_samples, interval_group_scores


class ConformalTests(unittest.TestCase):
    def test_ranks_match_the_notebook_on_candidate_windows(self):
        for n in WINDOW_CANDIDATES.values():
            for alpha in ALPHAS:
                lower = int(np.floor((n + 1) * (alpha / 2.0)))
                upper = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
                self.assertGreaterEqual(lower, 1)
                self.assertLessEqual(upper, n)
                self.assertEqual(lower_rank(n, alpha), lower - 1)
                self.assertEqual(upper_rank(n, alpha), upper - 1)

    def test_window_excludes_the_residual_at_the_origin(self):
        index = pd.date_range("2024-01-01", periods=30, freq="5min")
        residuals = pd.Series(np.arange(30, dtype=float), index=index)
        yhat = pd.Series(0.0, index=index[20:21])
        interval = apply_sliding(residuals, yhat, 5, 0.50, "absolute")
        self.assertEqual(interval["lower"].iloc[0], 15.0)
        self.assertEqual(interval["upper"].iloc[0], 19.0)
        changed = residuals.copy()
        changed.iloc[20] = -999.0
        again = apply_sliding(changed, yhat, 5, 0.50, "absolute")
        self.assertEqual(again["lower"].iloc[0], interval["lower"].iloc[0])
        self.assertEqual(again["upper"].iloc[0], interval["upper"].iloc[0])

    def test_scale_uses_the_previous_day_and_ignores_the_current_residual(self):
        index = pd.date_range("2024-01-01", periods=40, freq="5min")
        residuals = pd.Series(np.arange(40, dtype=float), index=index)
        sigma = lagged_scale(residuals, window=10, floor=1.0)
        self.assertEqual(sigma.iloc[20], float(np.median(np.abs(residuals.iloc[10:20]))))
        changed = residuals.copy()
        changed.iloc[20] = 0.0
        shifted = lagged_scale(changed, window=10, floor=1.0)
        self.assertEqual(shifted.iloc[20], sigma.iloc[20])
        self.assertNotEqual(shifted.iloc[21], sigma.iloc[21])

    def test_normalized_interval_rescales_and_respects_the_floor(self):
        index = pd.date_range("2024-01-01", periods=400, freq="5min")
        yhat = pd.Series(10.0, index=index[-1:])
        flat = apply_sliding(pd.Series(0.0, index=index), yhat, 10, 0.10, "normalized")
        self.assertEqual(flat["lower"].iloc[0], 10.0)
        self.assertEqual(flat["upper"].iloc[0], 10.0)
        scaled = apply_sliding(pd.Series(4.0, index=index), yhat, 10, 0.10, "normalized")
        self.assertEqual(scaled["lower"].iloc[0], 14.0)
        self.assertEqual(scaled["upper"].iloc[0], 14.0)

    def test_methods_are_scored_on_the_same_origins(self):
        index = pd.date_range("2024-01-01", periods=80, freq="5min")
        residuals = pd.Series(np.sin(np.arange(80)) + 0.2, index=index)
        yhat = pd.Series(0.0, index=index)
        table = selection_table(
            residuals,
            yhat,
            yhat + residuals,
            candidates={"short": 8, "long": 16},
            scale_window=5,
        )
        self.assertEqual(table["n_selection"].nunique(), 1)
        self.assertEqual(int(table["n_selection"].iloc[0]), 80 - (16 + 5))
        self.assertEqual(table.groupby("method")["selected"].sum().tolist(), [1, 1])

    def test_eligible_window_is_the_narrowest_and_ineligible_keeps_coverage(self):
        index = pd.date_range("2024-01-01", periods=40, freq="5min")
        zeros = pd.Series(0.0, index=index)
        yhat = pd.Series(1.0, index=index)
        chosen = select_window(zeros, yhat, yhat, "absolute", candidates={"short": 4, "long": 8})
        self.assertEqual(chosen.loc[chosen["selected"], "window"].item(), "short")
        self.assertTrue(bool(chosen["eligible"].all()))
        ineligible = _mark_selected(
            pd.DataFrame(
                [
                    {"window": "a", "window_steps": 10, "coverage_90": 0.80, "mean_width_90": 1.0, "eligible": False},
                    {"window": "b", "window_steps": 20, "coverage_90": 0.88, "mean_width_90": 5.0, "eligible": False},
                    {"window": "c", "window_steps": 30, "coverage_90": 0.88, "mean_width_90": 9.0, "eligible": False},
                ]
            )
        )
        self.assertEqual(ineligible.loc[ineligible["selected"], "window"].item(), "c")

    def test_primary_choice_uses_the_narrowest_eligible_method(self):
        selection = pd.DataFrame(
            [
                {"method": "absolute", "window": "7d", "window_steps": 20, "mean_width_90": 5.0, "eligible": True, "selected": True},
                {"method": "absolute", "window": "2d", "window_steps": 10, "mean_width_90": 1.0, "eligible": False, "selected": False},
                {"method": "normalized", "window": "7d", "window_steps": 20, "mean_width_90": 4.0, "eligible": True, "selected": True},
            ]
        )
        primary = choose_primary(selection)
        self.assertEqual(primary["method"], "normalized")
        self.assertEqual(int(primary["window_steps"]), 20)

    def test_crps_of_a_two_point_empirical_distribution(self):
        samples = np.array([[0.0, 2.0]])
        score = crps_from_residual_samples(samples, y=np.array([1.0]), yhat=np.array([0.0]))
        self.assertTrue(np.allclose(score, 0.5))

    def test_group_scores_report_width_pinball_and_crps(self):
        y = np.array([0.0, 0.0, 10.0])
        lower = np.array([-1.0, -1.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        table = interval_group_scores(y, lower, upper, ["a", "a", "b"], 0.10, crps=np.array([1.0, 3.0, 5.0]))
        row_a = table.loc[table["group"].eq("a")].iloc[0]
        self.assertEqual(int(row_a["n"]), 2)
        self.assertEqual(row_a["coverage"], 1.0)
        self.assertEqual(row_a["mean_width"], 2.0)
        self.assertEqual(row_a["crps"], 2.0)
        self.assertGreater(row_a["pinball_upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
