import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qra import QRA_FORECASTS, fit_qra, level_name, predict_qra
from scripts.qra_variants import (
    LGBM_QUANTILE,
    Spec,
    default_specs,
    fit_spec,
    predict_spec,
    run_variant_study,
    score_quantiles,
    select_variant,
    window_rows,
)

LEVELS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def _synthetic(n=600, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq="5min")
    signal = pd.Series(rng.normal(size=n), index=index)
    y = signal + rng.normal(scale=0.5, size=n)
    forecasts = {name: signal + rng.normal(scale=0.3, size=n) for name in QRA_FORECASTS}
    forecasts["lightgbm"] = signal
    return index, y, pd.DataFrame(forecasts, index=index)


class QraVariantTests(unittest.TestCase):
    def test_subset_qra_matches_fit_qra_on_the_same_columns(self):
        _index, y, forecasts = _synthetic()
        spec = Spec("qra", "lgbm_p5", ("lightgbm", "persistence_5min"))
        fitted = fit_spec(spec, forecasts, y, LEVELS)
        direct = fit_qra(forecasts, y, LEVELS, columns=spec.members)
        pd.testing.assert_frame_equal(fitted["coefficients"]["qra"], direct)
        self.assertEqual(list(direct.columns[-2:]), ["lightgbm", "persistence_5min"])
        predicted, _ = predict_spec(fitted, forecasts, LEVELS)
        expected, _ = predict_qra(forecasts, direct)
        pd.testing.assert_frame_equal(predicted, expected)

    def test_default_qra_columns_are_unchanged(self):
        _index, y, forecasts = _synthetic()
        coefficients = fit_qra(forecasts, y, LEVELS)
        self.assertEqual(list(coefficients.columns[3:]), list(QRA_FORECASTS))

    def test_quantile_average_is_the_mean_of_sorted_member_quantiles(self):
        _index, y, forecasts = _synthetic()
        spec = Spec("qave", "lgbm_p5", ("lightgbm", "persistence_5min"))
        fitted = fit_spec(spec, forecasts, y, LEVELS)
        averaged, _ = predict_spec(fitted, forecasts, LEVELS)
        members = []
        for member in spec.members:
            single = fit_spec(Spec("qra", member, (member,)), forecasts, y, LEVELS)
            members.append(predict_spec(single, forecasts, LEVELS)[0].to_numpy())
        np.testing.assert_allclose(averaged.to_numpy(), np.mean(members, axis=0))
        self.assertTrue((np.diff(averaged.to_numpy(), axis=1) >= 0).all())

    def test_qrm_regresses_on_the_member_mean(self):
        _index, y, forecasts = _synthetic()
        spec = Spec("qrm", "lgbm_p5", ("lightgbm", "persistence_5min"))
        fitted = fit_spec(spec, forecasts, y, LEVELS)
        frame = pd.DataFrame({"member_mean": forecasts[["lightgbm", "persistence_5min"]].mean(axis=1)})
        direct = fit_qra(frame, y, LEVELS, columns=("member_mean",))
        pd.testing.assert_frame_equal(fitted["coefficients"]["member_mean"], direct)

    def test_native_quantiles_are_sorted_and_need_no_calibration_fit(self):
        index = pd.RangeIndex(3)
        forecasts = pd.DataFrame({"lightgbm": np.zeros(3)}, index=index)
        native = pd.DataFrame(
            [[3.0, 1.0, 2.0, 4.0, 5.0]] * 3, index=index, columns=[level_name(level) for level in LEVELS]
        )
        spec = Spec(LGBM_QUANTILE, "native", (LGBM_QUANTILE,))
        fitted = fit_spec(spec, forecasts, pd.Series(np.zeros(3)), LEVELS)
        self.assertEqual(fitted["coefficients"], {})
        predicted, crossed = predict_spec(fitted, forecasts, LEVELS, native)
        self.assertEqual(crossed, 1.0)
        np.testing.assert_allclose(predicted.iloc[0].to_numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_window_rows_keeps_only_the_last_days_before_the_end(self):
        index = pd.date_range("2024-01-01", periods=5 * 288, freq="5min")
        end = pd.Timestamp("2024-01-04 23:55")
        mask = window_rows(index, end, 2)
        self.assertEqual(int(mask.sum()), 2 * 288)
        self.assertEqual(index[mask].min(), pd.Timestamp("2024-01-03 00:00"))
        self.assertEqual(index[mask].max(), end)
        self.assertEqual(int(window_rows(index, end, None).sum()), 4 * 288)

    def test_selection_is_narrowest_width_inside_the_coverage_band(self):
        table = pd.DataFrame(
            {
                "variant": ["a", "b", "c", "d"],
                "coverage_90": [0.95, 0.90, 0.885, 0.80],
                "mean_width_90": [5.0, 20.0, 18.0, 1.0],
                "crps_quantile_integral": [1.0, 1.0, 2.0, 0.5],
                "candidate": [True, True, True, True],
            }
        )
        self.assertEqual(select_variant(table), "c")
        table["coverage_90"] = [0.95, 0.96, 0.70, 0.80]
        self.assertIsNone(select_variant(table))

    def test_reference_rows_are_never_selected(self):
        table = pd.DataFrame(
            {
                "variant": ["ref", "b"],
                "coverage_90": [0.90, 0.90],
                "mean_width_90": [1.0, 20.0],
                "crps_quantile_integral": [1.0, 1.0],
                "candidate": [False, True],
            }
        )
        self.assertEqual(select_variant(table), "b")

    def test_score_quantiles_reports_interval_and_crps_columns(self):
        _index, y, forecasts = _synthetic()
        spec = Spec("qra", "lgbm", ("lightgbm",))
        from scripts.qra import comparison_levels

        levels = comparison_levels()
        fitted = fit_spec(spec, forecasts, y, levels)
        predicted, _ = predict_spec(fitted, forecasts, levels)
        row = score_quantiles(y, predicted, levels)
        for key in ("coverage_90", "mean_width_90", "median_width_90", "coverage_50", "crps_quantile_integral"):
            self.assertIn(key, row)
        self.assertTrue(0.8 <= row["coverage_90"] <= 1.0)

    def test_study_never_uses_test_outcomes_for_selection(self):
        n = 40 * 288
        index, y, forecasts = _synthetic(n=n, seed=5, start="2026-01-01")
        frame = pd.DataFrame({"mcp": y}, index=index)
        cal_end = pd.Timestamp("2026-01-30 23:55")
        calibration = frame.loc[frame.index <= cal_end]
        test = frame.loc[frame.index > cal_end]
        predictions = {name: forecasts[name] for name in QRA_FORECASTS}
        point = {"calibration": calibration, "test": test, "predictions": predictions}
        specs = [s for s in default_specs() if LGBM_QUANTILE not in s.members][:4]
        kwargs = dict(
            specs=specs,
            windows={"5d": 5, "all": None},
            inner_fit_end=pd.Timestamp("2026-01-23 23:55"),
            inner_val_start=pd.Timestamp("2026-01-24 00:00"),
            cal_end=cal_end,
            primary_method="absolute",
            primary_window=288,
        )
        first = run_variant_study(point, None, **kwargs)
        corrupted = test.copy()
        corrupted["mcp"] = corrupted["mcp"] * 50.0 + 1000.0
        second = run_variant_study({**point, "test": corrupted}, None, **kwargs)
        pd.testing.assert_frame_equal(first["calib"], second["calib"])
        self.assertEqual(first["meta"]["selected_variant"], second["meta"]["selected_variant"])
        variants = first["test"].loc[first["test"]["role"].eq("variant")]
        self.assertEqual(len(variants), len(specs) * 2)
        self.assertLessEqual(int(variants["selected_on_calibration"].sum()), 1)
        self.assertTrue(variants["is_original_qra"].any())
        self.assertFalse(first["meta"]["test_outcomes_used_for_fit_or_selection"])


if __name__ == "__main__":
    unittest.main()
