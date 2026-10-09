"""Tests for the implemented spike forecaster (scripts/spike_forecaster.py).

The real-data tests need the committed panel, the fitted model
(models/spike_forecaster.joblib, from `python scripts/spike_forecaster.py fit`)
and the reference detail written by scripts/spike_onset.py; they are skipped
when those are missing.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts import spike_forecaster as sf
from scripts.spike_onset import OUT, summary

REFERENCE = OUT / "spike_forecaster_reference_test.parquet"
REALISED = ["mcp", "cr_raise", "cr_lower", "reg_raise", "reg_lower", "rocof", "operational_demand_mw",
            "unscheduled_demand_mw", "operational_withdrawal_mw", "dpv_mw", "rtp", "sent_out_mwh", "sent_out_mw",
            "dpv_mw_30min", "scada_mwh", "scada_mw", "scheduled_energy_mw", "residual_demand_mw", "mcp_lag1",
            "mcp_lag6", "mcp_lag12"]
DAY_AHEAD = ["stem_price", "stem_qty_mwh", "stem_bid_mwh", "stem_offer_mwh"]


class WeightedQuantileTests(unittest.TestCase):
    def test_equal_weights_match_inverted_cdf(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(5, 200))
        w = np.ones_like(x)
        levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        got = sf.weighted_quantiles(x, w, levels)
        want = np.quantile(x, levels, axis=1, method="inverted_cdf").T
        np.testing.assert_allclose(got, want)

    def test_weights_move_mass(self):
        x = np.array([[0.0, 1.0, 100.0]])
        w = np.array([[0.5, 0.3, 0.2]])
        np.testing.assert_allclose(sf.weighted_quantiles(x, w, [0.4, 0.6, 0.85]), [[0.0, 1.0, 100.0]])

    def test_monotone_in_level(self):
        rng = np.random.default_rng(2)
        x, w = rng.normal(size=(50, 30)), rng.uniform(size=(50, 30))
        q = sf.weighted_quantiles(x, w, sf.QUANTILES)
        self.assertTrue(np.all(np.diff(q, axis=1) >= 0))


@unittest.skipUnless(sf.MODEL_PATH.exists() and REFERENCE.exists(), "fitted spike forecaster or reference detail missing")
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = sf.SpikeForecaster.load()
        cls.raw = sf.read_panel()
        ref = pd.read_parquet(REFERENCE)
        onsets = ref.index[ref["onset"].to_numpy(bool)]
        cls.day = onsets[len(onsets) // 2].normalize()  # a test day with at least one spike onset
        cls.target = onsets[len(onsets) // 2]
        cls.ref = ref.loc[(ref.index >= cls.day) & (ref.index < cls.day + pd.Timedelta(days=1))]
        cls.raw_slice = cls.raw.loc[(cls.raw["interval_end"] >= cls.day - pd.Timedelta(days=10))
                                    & (cls.raw["interval_end"] < cls.day + pd.Timedelta(days=1))].copy()
        cls.frame = cls.forecast_frame(cls.raw_slice)
        cls.fc = cls.forecast(cls.frame, cls.ref.index)

    @classmethod
    def forecast_frame(cls, raw):
        frame = sf.build_frame(raw, cls.model.config)
        return frame

    @classmethod
    def forecast(cls, frame, targets):
        model = sf.SpikeForecaster.load()
        model.set_history(frame)
        return model.forecast(frame, targets)

    def test_no_leakage_at_origin(self):
        """Changing everything realised at or after the target interval leaves its forecast unchanged."""
        bad = self.raw_slice.copy()
        after = bad["interval_end"] >= self.target
        bad.loc[after, REALISED] = bad.loc[after, REALISED] * 7.0 + 1e4
        later = bad["interval_end"] > self.target  # day-ahead STEM for the target interval is published in advance
        bad.loc[later, DAY_AHEAD] = -999.0
        frame = self.forecast_frame(bad)
        got = self.forecast(frame, [self.target])
        want = self.fc.loc[[self.target]]
        pd.testing.assert_frame_equal(got, want)
        # the scored price itself did change
        self.assertNotEqual(frame.loc[self.target, "y"], self.frame.loc[self.target, "y"])

    def test_quantiles_monotone(self):
        q = self.fc[[c for c in self.fc.columns if c.startswith("q")]].to_numpy()
        self.assertTrue(np.all(np.diff(q, axis=1) >= 0))
        self.assertTrue(np.all(np.isfinite(q)))

    def test_probabilities_in_unit_interval(self):
        for col in ("p_up", "p_down", "mix_weight_up", "mix_weight_down", "prob_at_or_above_spike", "prob_at_or_below_floor"):
            v = self.fc[col].dropna().to_numpy()
            self.assertTrue(np.all((v >= 0) & (v <= 1)), col)
        self.assertTrue(self.fc.loc[~self.fc["eligible"], "p_up"].isna().all())
        self.assertTrue(np.all(self.fc["mix_weight_up"] + self.fc["mix_weight_down"] <= 0.95 + 1e-12))

    def test_save_load_round_trip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = sf.SpikeForecaster.load().save(Path(tmp) / "m.joblib")
            again = sf.SpikeForecaster.load(path)
        again.set_history(self.frame)
        pd.testing.assert_frame_equal(again.forecast(self.frame, self.ref.index), self.fc)
        self.assertEqual(again.cutoffs, self.model.cutoffs)

    def test_reproduces_reference_on_slice(self):
        """The module matches scripts/spike_onset.py's frozen row on a test day, row by row and in metrics."""
        np.testing.assert_allclose(self.fc["point"], self.ref["yhat"], rtol=0, atol=1e-9)
        elig = self.fc["eligible"].to_numpy()
        np.testing.assert_allclose(self.fc["p_up"].to_numpy()[elig], self.ref["p_up"].to_numpy()[elig], atol=1e-12)
        np.testing.assert_allclose(self.fc["p_down"].to_numpy()[elig], self.ref["p_down"].to_numpy()[elig], atol=1e-12)
        model = sf.SpikeForecaster.load()
        model.set_history(self.frame)
        block = self.frame.loc[self.ref.index]
        crps = model.crps(block)
        np.testing.assert_allclose(crps, self.ref["crps"], rtol=1e-9, atol=1e-9)
        mine = model.metrics(block, self.fc)
        called = np.maximum(self.ref["p_up"], self.ref["p_down"]).to_numpy() >= model.cutoffs["detection"]
        theirs = summary("ref", block, self.ref["yhat"].to_numpy(), self.ref["crps"].to_numpy(), called)
        for key in ("mae", "crps", "tail_mae", "tail_crps", "onset_mae", "onset_crps", "onset_precision", "onset_recall"):
            np.testing.assert_allclose(mine[key], theirs[key], rtol=1e-9, atol=1e-9, err_msg=key)


if __name__ == "__main__":
    unittest.main()
