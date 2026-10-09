"""Tests for the conformal predictive systems in scripts/spike_cps.py."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts import spike_cps as cps
from scripts.metrics import crps_from_residual_samples

TEST_ROWS = cps.OUT / "final_cps_test_rows.parquet"


def synthetic_base(n_cal=4000, n_test=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-10-01", periods=n_cal + n_test, freq="5min")
    point = rng.normal(100, 10, len(idx))
    y = point + rng.standard_t(3, len(idx)) * 5
    base = pd.DataFrame({"split": ["calibration"] * n_cal + ["test"] * n_test, "y": y, "regime": point,
                         "fc_pit": rng.uniform(size=len(idx))}, index=idx)
    return base


class SplitCPSTests(unittest.TestCase):
    def test_crps_matches_empirical_formula(self):
        rng = np.random.default_rng(1)
        r = rng.normal(size=500)
        cps_ = cps.SplitCPS(r)
        y, yhat = rng.normal(size=50) * 3, rng.normal(size=50)
        want = crps_from_residual_samples(np.tile(r, (50, 1)), y, yhat)
        np.testing.assert_allclose(cps_.crps(y, yhat), want, rtol=1e-10, atol=1e-10)

    def test_cdf_monotone_and_inside_unit_interval(self):
        cps_ = cps.SplitCPS(np.random.default_rng(2).normal(size=300))
        grid = np.linspace(-6, 6, 2001)
        q = cps_.cdf(grid, np.zeros_like(grid))
        self.assertTrue(np.all(np.diff(q) >= 0))
        self.assertTrue(np.all((q > 0) & (q < 1)))

    def test_coverage_on_exchangeable_data(self):
        base = synthetic_base()
        folds = cps.fold_labels(base)
        res, _ = cps.residual_cps(base, np.full(len(base), 2), folds)
        t = folds == "T"
        y = base["y"].to_numpy()
        for key in ("cps_split", "cps_mondrian"):
            cov = np.mean((y[t] >= res[key]["lo"][t]) & (y[t] <= res[key]["hi"][t]))
            self.assertGreater(cov, 0.87, key)
            self.assertLess(cov, 0.93, key)

    def test_mondrian_uses_own_category(self):
        r = np.concatenate([np.zeros(400), np.full(400, 100.0)])
        cats = np.array([2] * 400 + [5] * 400)
        m = cps.MondrianCPS(r, cats)
        lo, hi = m.interval(np.zeros(2), np.array([2, 5]))
        np.testing.assert_allclose(lo, [0.0, 100.0])
        np.testing.assert_allclose(m.crps(np.array([100.0]), np.zeros(1), np.array([5])), [0.0])
        lo_pool, _ = m.interval(np.zeros(1), np.array([-1]))  # unknown / merged category uses pooled residuals
        self.assertEqual(lo_pool[0], 0.0)


class LeakageTests(unittest.TestCase):
    def test_test_outcomes_never_change_other_test_forecasts(self):
        base = synthetic_base(seed=3)
        folds = cps.fold_labels(base)
        cat = np.where(np.arange(len(base)) % 3 == 0, 3, 2)
        res1, _ = cps.residual_cps(base, cat, folds)
        bad = base.copy()
        t = np.flatnonzero(folds == "T")
        bad.iloc[t[1:], bad.columns.get_loc("y")] += 1e4  # all test outcomes except the first
        res2, _ = cps.residual_cps(bad, cat, folds)
        for key in ("cps_split", "cps_mondrian"):
            for k in ("lo", "hi"):
                np.testing.assert_array_equal(res1[key][k][t], res2[key][k][t])
            self.assertEqual(res1[key]["crps"][t[0]], res2[key]["crps"][t[0]])
            self.assertEqual(res1[key]["pit"][t[0]], res2[key]["pit"][t[0]])

    def test_calibration_halves_are_cross_fitted(self):
        base = synthetic_base(seed=4)
        folds = cps.fold_labels(base)
        a = np.flatnonzero(folds == "A")
        res1, _ = cps.residual_cps(base, np.full(len(base), 2), folds)
        bad = base.copy()
        bad.iloc[a, bad.columns.get_loc("y")] += 1e4  # half A outcomes must not move half A intervals
        res2, _ = cps.residual_cps(bad, np.full(len(base), 2), folds)
        np.testing.assert_array_equal(res1["cps_split"]["lo"][a], res2["cps_split"]["lo"][a])

    def test_risk_category_uses_origin_inputs_only(self):
        cat = cps.risk_category([True, True, False, False], [0.001, 0.3, np.nan, np.nan], [50, 60, 200, -80], 170.3, -54.2)
        np.testing.assert_array_equal(cat, [2, 5, 0, 1])


class PITTests(unittest.TestCase):
    def test_recalibrated_weights_are_a_distribution_and_cdf_monotone(self):
        rng = np.random.default_rng(5)
        cal = cps.PITCalibrator(rng.beta(2, 5, 1000))
        c = np.sort(rng.uniform(size=(20, 50)), axis=1)
        c[:, -1] = 1.0
        w = cps.recalibrated_weights(c, cal)
        self.assertTrue(np.all(w >= 0))
        np.testing.assert_allclose(w.sum(axis=1), 1.0)
        u = np.linspace(0, 1, 101)
        self.assertTrue(np.all(np.diff(cal.cdf(u)) >= 0))

    def test_recalibration_makes_pits_uniform(self):
        rng = np.random.default_rng(6)
        cal = cps.PITCalibrator(rng.beta(2, 5, 5000))
        new = cal.cdf(rng.beta(2, 5, 5000))
        hist, _ = np.histogram(new, bins=10, range=(0, 1))
        self.assertLess(np.abs(hist / 5000 - 0.1).max(), 0.02)


@unittest.skipUnless(TEST_ROWS.exists(), "run scripts/spike_cps.py first")
class RealSliceTests(unittest.TestCase):
    def test_coverage_and_monotone_bands_on_test_slice(self):
        rows = pd.read_parquet(TEST_ROWS)
        month = rows.loc["2026-05"]
        for key in ("cps_split", "cps_mondrian", "cps_pit", "cps_pit_mondrian"):
            self.assertTrue(np.all(month[f"{key}_lo"] <= month[f"{key}_hi"]), key)
            pit = month[f"{key}_pit"]
            self.assertTrue(((pit > 0) & (pit < 1)).all(), key)
        cov = ((month["y"] >= month["cps_mondrian_lo"]) & (month["y"] <= month["cps_mondrian_hi"])).mean()
        self.assertGreater(cov, 0.80)


if __name__ == "__main__":
    unittest.main()
