"""No external feature may use information published after its forecast origin."""

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.external_features import (
    EXTERNAL_PREDISPATCH,
    EXTERNAL_PREDISPATCH_ALL,
    EXTERNAL_PREDISPATCH_HALF,
    EXTERNAL_REVISION_HALF,
    EXTERNAL_REVISION,
    EXTERNAL_WEATHER,
    PD_FIELDS,
    PREDISPATCH_CONSOLIDATED,
    PREDISPATCH_DIR,
    WEATHER_LAG,
    WEATHER_PATH,
    build_external,
    load_predispatch,
    on_the_hour_runs,
    check_availability,
    predispatch_available_at,
    predispatch_features,
    predispatch_features_all,
    predispatch_features_half,
    weather_at,
    weather_features,
)
from scripts.forecast_design import CONTEMPORANEOUS_BANNED, HORIZON

SITES = ("perth_metro", "geraldton", "badgingarra", "merredin", "albany")
VARS = ("temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_100m")


def synthetic_weather(start="2025-01-01", hours=96, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=hours, freq="h")
    cols = [f"{s}_{v}" for s in SITES for v in VARS]
    return pd.DataFrame(rng.uniform(1, 20, size=(hours, len(cols))), index=index, columns=cols)


def synthetic_runs(start="2025-01-01 00:00", n_runs=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_runs):
        label = pd.Timestamp(start) + pd.Timedelta(minutes=30 * k)
        issued = label + pd.Timedelta(minutes=15)
        for j in range(16):
            rows.append(
                {
                    "run_label": label,
                    "issued": issued,
                    "interval": label + pd.Timedelta(minutes=30 * j),
                    "price": 1000.0 * k + j,  # encodes the run, so the chosen run is visible
                    "reg_raise_price": rng.uniform(),
                    "cont_raise_price": rng.uniform(),
                    "energy_req": 1500 + rng.uniform(),
                    "cont_raise_req": 200.0,
                    "in_service_cap": 3800.0,
                    "available_cap": 1700.0,
                    "cont_raise_avail": 300.0,
                    "energy_deficit": 0.0,
                }
            )
    runs = pd.DataFrame(rows)
    for col in ("run_label", "issued", "interval"):
        runs[col] = runs[col].astype("datetime64[ns]")
    runs["available_at"] = predispatch_available_at(runs["run_label"], runs["issued"])
    return runs


class WeatherAvailabilityTests(unittest.TestCase):
    def test_interpolation_reads_only_the_bracketing_hours(self):
        hourly = synthetic_weather()
        when = pd.DatetimeIndex([pd.Timestamp("2025-01-02 10:20")])
        values, read = weather_at(hourly, when)
        col = hourly.columns[0]
        a, b = hourly.loc["2025-01-02 10:00", col], hourly.loc["2025-01-02 11:00", col]
        self.assertAlmostEqual(values.iloc[0][col], a + (b - a) / 3.0)
        self.assertEqual(read.iloc[0], pd.Timestamp("2025-01-02 11:00"))

    def test_every_row_is_available_at_its_origin(self):
        hourly = synthetic_weather()
        targets = pd.date_range("2025-01-02 00:05", "2025-01-03 12:00", freq="5min")
        table = weather_features(targets, hourly)
        self.assertTrue((table["wx_available_at"] <= table["forecast_origin"]).all())
        check_availability(table)

    def test_values_published_after_the_origin_cannot_change_a_feature(self):
        hourly = synthetic_weather()
        targets = pd.date_range("2025-01-02 00:05", "2025-01-03 12:00", freq="5min")
        base = weather_features(targets, hourly)
        for target in targets[::37]:
            origin = target - HORIZON
            changed = hourly.copy()
            # Every value whose availability time (valid - 12 h) is after this origin.
            late = changed.index - WEATHER_LAG > origin
            changed.loc[late] = 1e6
            again = weather_features(pd.DatetimeIndex([target]), changed)
            pd.testing.assert_series_equal(
                base.loc[target, EXTERNAL_WEATHER].astype(float),
                again.loc[target, EXTERNAL_WEATHER].astype(float),
                check_names=False,
            )


class PredispatchAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.runs = synthetic_runs()
        self.targets = pd.date_range("2025-01-01 01:30", "2025-01-01 12:00", freq="5min")
        self.realised = pd.DataFrame({"mcp_lag_5min": 50.0, "demand_lag_5min": 1500.0}, index=self.targets)

    def test_uses_the_latest_run_available_at_the_origin(self):
        table = predispatch_features(self.targets, self.runs, self.realised)
        meta = self.runs.groupby("run_label")["available_at"].max()
        for target, row in table.iterrows():
            origin = target - HORIZON
            usable = meta[meta <= origin]
            if usable.empty:
                self.assertTrue(np.isnan(row["pd_price_t"]))
                continue
            expected_run = list(meta.index).index(usable.index.max())
            self.assertEqual(int(row["pd_price_t"] // 1000), expected_run)
            self.assertLessEqual(row["pd_available_at"], origin)

    def test_a_run_published_after_the_origin_cannot_change_a_feature(self):
        base = predispatch_features(self.targets, self.runs, self.realised)
        for target in self.targets[::11]:
            origin = target - HORIZON
            changed = self.runs.copy()
            late = changed["available_at"] > origin
            changed.loc[late, list(PD_FIELDS.values())] = 1e6
            again = predispatch_features(pd.DatetimeIndex([target]), changed, self.realised.loc[[target]])
            pd.testing.assert_series_equal(
                base.loc[target, EXTERNAL_PREDISPATCH].astype(float),
                again.loc[target, EXTERNAL_PREDISPATCH].astype(float),
                check_names=False,
            )

    def test_a_late_issue_time_keeps_the_run_out_until_published(self):
        runs = self.runs.copy()
        late_label = runs["run_label"].unique()[6]
        mask = runs["run_label"].eq(late_label)
        runs.loc[mask, "issued"] = late_label + pd.Timedelta(minutes=50)
        runs.loc[mask, "available_at"] = predispatch_available_at(runs.loc[mask, "run_label"], runs.loc[mask, "issued"])
        published = pd.Timestamp(late_label) + pd.Timedelta(minutes=75)
        table = predispatch_features(self.targets, runs, self.realised)
        origins = table.index - HORIZON
        before = (origins >= pd.Timestamp(late_label) + pd.Timedelta(minutes=40)) & (origins < published)
        self.assertTrue(before.any())
        self.assertTrue((table.loc[before, "pd_price_t"] // 1000 != 6).all())
        self.assertTrue((table.loc[origins >= published, "pd_price_t"].iloc[:1] // 1000 == 6).all())

    def test_all_run_and_revision_features_ignore_runs_published_after_the_origin(self):
        base = predispatch_features_all(self.targets, self.runs, self.realised)
        self.assertTrue(base["pda_rev_price_t"].notna().any())
        for target in self.targets[::7]:
            origin = target - HORIZON
            changed = self.runs.copy()
            late = changed["available_at"] > origin
            changed.loc[late, list(PD_FIELDS.values())] = 1e6
            again = predispatch_features_all(pd.DatetimeIndex([target]), changed, self.realised.loc[[target]])
            dropped = predispatch_features_all(pd.DatetimeIndex([target]), self.runs.loc[~late], self.realised.loc[[target]])
            for other in (again, dropped):
                pd.testing.assert_series_equal(
                    base.loc[target, EXTERNAL_PREDISPATCH_ALL].astype(float),
                    other.loc[target, EXTERNAL_PREDISPATCH_ALL].astype(float),
                    check_names=False,
                )

    def test_revision_compares_the_latest_run_with_the_one_before(self):
        table = predispatch_features_all(self.targets, self.runs, self.realised)
        meta = self.runs.groupby("run_label")["available_at"].max()
        has = table["pda_rev_price_t"].notna()
        self.assertTrue(has.any())
        for target, row in table.loc[has].iterrows():
            origin = target - HORIZON
            latest = meta[meta <= origin].index.max()
            k = list(meta.index).index(latest)
            self.assertEqual(int(row["pda_price_t"] // 1000), k)
            # price = 1000 * run + interval index; the same half hour is one index later in the previous run
            self.assertAlmostEqual(row["pda_rev_price_t"], 999.0)
            self.assertLessEqual(row["pda_prev_available_at"], origin)
            self.assertLessEqual(row["pda_available_at"], origin)

    def test_a_previous_run_published_late_gives_no_revision(self):
        runs = self.runs.copy()
        labels = runs["run_label"].unique()
        mask = runs["run_label"].eq(labels[5])
        runs.loc[mask, "issued"] = labels[5] + pd.Timedelta(hours=3)
        runs.loc[mask, "available_at"] = predispatch_available_at(runs.loc[mask, "run_label"], runs.loc[mask, "issued"])
        table = predispatch_features_all(self.targets, runs, self.realised)
        origins = table.index - HORIZON
        latest_is_6 = (table["pda_price_t"] // 1000 == 6) & (origins < runs.loc[mask, "available_at"].iloc[0])
        self.assertTrue(latest_is_6.any())
        self.assertTrue(table.loc[latest_is_6, EXTERNAL_REVISION].isna().all().all())

    def test_half_hourly_features_ignore_runs_published_after_the_origin(self):
        base = predispatch_features_half(self.targets, self.runs, self.realised)
        self.assertTrue(base["pdh_rev30_price_t"].notna().any())
        self.assertTrue(base["pdh_rev60_price_t"].notna().any())
        for target in self.targets[::7]:
            origin = target - HORIZON
            late = self.runs["available_at"] > origin
            changed = self.runs.copy()
            changed.loc[late, list(PD_FIELDS.values())] = 1e6
            for runs in (changed, self.runs.loc[~late]):
                again = predispatch_features_half(pd.DatetimeIndex([target]), runs, self.realised.loc[[target]])
                pd.testing.assert_series_equal(
                    base.loc[target, EXTERNAL_PREDISPATCH_HALF].astype(float),
                    again.loc[target, EXTERNAL_PREDISPATCH_HALF].astype(float),
                    check_names=False,
                )

    def test_half_hourly_revisions_compare_runs_30_and_60_minutes_apart(self):
        table = predispatch_features_half(self.targets, self.runs, self.realised)
        meta = self.runs.groupby("run_label")["available_at"].max()
        has = table["pdh_rev60_price_t"].notna()
        self.assertTrue(has.any())
        for target, row in table.loc[has].iterrows():
            origin = target - HORIZON
            k = list(meta.index).index(meta[meta <= origin].index.max())
            self.assertEqual(int(row["pdh_price_t"] // 1000), k)
            # price = 1000 * run + interval index; runs are 30 min apart in the synthetic data
            self.assertAlmostEqual(row["pdh_rev30_price_t"], 999.0)
            self.assertAlmostEqual(row["pdh_rev60_price_t"], 1998.0)
            for col in ("pdh_available_at", "pdh_rev30_prev_available_at", "pdh_rev60_prev_available_at"):
                self.assertLessEqual(row[col], origin)

    def test_a_late_previous_half_hour_run_gives_no_30_minute_revision(self):
        runs = self.runs.copy()
        labels = runs["run_label"].unique()
        mask = runs["run_label"].eq(labels[5])
        runs.loc[mask, "issued"] = labels[5] + pd.Timedelta(hours=3)
        runs.loc[mask, "available_at"] = predispatch_available_at(runs.loc[mask, "run_label"], runs.loc[mask, "issued"])
        table = predispatch_features_half(self.targets, runs, self.realised)
        origins = table.index - HORIZON
        latest_is_6 = (table["pdh_price_t"] // 1000 == 6) & (origins < runs.loc[mask, "available_at"].iloc[0])
        self.assertTrue(latest_is_6.any())
        rev30 = [c for c in EXTERNAL_REVISION_HALF if "rev30" in c]
        self.assertTrue(table.loc[latest_is_6, rev30].isna().all().all())
        # the 60 minute revision falls back to run 4, which was published on time
        self.assertTrue((table.loc[latest_is_6, "pdh_rev60_price_t"] == 1998.0).all())

    def test_late_issue_time_delays_availability(self):
        label = pd.Series(pd.to_datetime(["2025-01-01 10:00"]))
        issued = pd.Series(pd.to_datetime(["2025-01-01 10:50"]))
        got = predispatch_available_at(label, issued).iloc[0]
        self.assertEqual(got, pd.Timestamp("2025-01-01 11:15"))
        got = predispatch_available_at(label, pd.Series(pd.to_datetime(["2025-01-01 10:10"]))).iloc[0]
        self.assertEqual(got, pd.Timestamp("2025-01-01 10:40"))

    def test_check_availability_rejects_a_late_row(self):
        table = predispatch_features(self.targets, self.runs, self.realised)
        table["forecast_origin"] = table.index - HORIZON
        bad = table.copy()
        row = bad["pd_available_at"].first_valid_index()
        bad.loc[row, "pd_available_at"] = row + pd.Timedelta(minutes=1)
        with self.assertRaises(RuntimeError):
            check_availability(bad)


class FeatureListTests(unittest.TestCase):
    def test_no_same_interval_realised_columns(self):
        self.assertFalse(CONTEMPORANEOUS_BANNED.intersection(EXTERNAL_WEATHER + EXTERNAL_PREDISPATCH))


@unittest.skipUnless(WEATHER_PATH.exists(), "external weather file not pulled")
class RealDataTests(unittest.TestCase):
    def test_built_table_respects_availability(self):
        from scripts.regime_switch import load

        ready, _ = load()
        table = build_external(ready)
        check_availability(table)
        self.assertTrue((table["wx_available_at"] <= table["forecast_origin"]).all())
        has_pd = table["pd_available_at"].notna()
        from scripts.forecast_design import split_mask

        split = split_mask(table.index)
        for name in ("train", "calibration", "test"):
            self.assertGreater(has_pd[split.eq(name).to_numpy()].mean(), 0.9, f"pre-dispatch missing in {name}")
        self.assertTrue((table.loc[has_pd, "pd_available_at"] <= table.loc[has_pd, "forecast_origin"]).all())


@unittest.skipUnless(PREDISPATCH_CONSOLIDATED.exists(), "consolidated pre-dispatch file not built")
class RealPredispatchTests(unittest.TestCase):
    """The committed pre-dispatch runs: nothing published after an origin reaches its features."""

    @classmethod
    def setUpClass(cls):
        cls.runs = load_predispatch()
        rng = np.random.default_rng(3)
        lo, hi = pd.Timestamp("2023-10-02 08:00"), pd.Timestamp("2026-08-18 07:55")
        grid = pd.date_range(lo, hi, freq="5min")
        cls.targets = pd.DatetimeIndex(np.sort(rng.choice(grid, size=300, replace=False)))
        cls.realised = pd.DataFrame({"mcp_lag_5min": 50.0, "demand_lag_5min": 1500.0}, index=cls.targets)
        cls.table = predispatch_features(cls.targets, cls.runs, cls.realised)

    def test_every_run_used_was_published_before_the_origin(self):
        origin = self.table.index - HORIZON
        used = self.table["pd_available_at"].notna()
        self.assertGreater(used.mean(), 0.9)
        self.assertTrue((self.table.loc[used, "pd_available_at"] <= origin[used]).all())
        # Independent of the available_at column: the public file is posted ~35 min
        # after the label and ~20 min after the issue time; the rule adds 5 min to each.
        label = pd.Series(origin - pd.to_timedelta(self.table["pd_run_age_min"].to_numpy(), unit="min"), index=self.table.index)
        issued = self.runs.groupby("run_label")["issued"].first()
        published = np.maximum(label + pd.Timedelta(minutes=40), label.map(issued).fillna(label) + pd.Timedelta(minutes=25))
        self.assertTrue((published[used] <= origin[used]).all())

    def test_runs_published_after_the_origin_cannot_change_a_feature(self):
        for target in self.targets[::10]:
            origin = target - HORIZON
            changed = self.runs.copy()
            late = changed["available_at"] > origin
            changed.loc[late, list(PD_FIELDS.values())] = 1e6
            again = predispatch_features(pd.DatetimeIndex([target]), changed, self.realised.loc[[target]])
            pd.testing.assert_series_equal(
                self.table.loc[target, EXTERNAL_PREDISPATCH].astype(float),
                again.loc[target, EXTERNAL_PREDISPATCH].astype(float),
                check_names=False,
            )
            dropped = predispatch_features(pd.DatetimeIndex([target]), self.runs.loc[~late], self.realised.loc[[target]])
            pd.testing.assert_series_equal(
                self.table.loc[target, EXTERNAL_PREDISPATCH].astype(float),
                dropped.loc[target, EXTERNAL_PREDISPATCH].astype(float),
                check_names=False,
            )

    def test_all_run_and_revision_features_use_only_published_runs(self):
        self.runs_hourly = on_the_hour_runs(self.runs)
        table = predispatch_features_all(self.targets, self.runs_hourly, self.realised)
        origin = table.index - HORIZON
        for col in ("pda_available_at", "pda_prev_available_at"):
            used = table[col].notna()
            self.assertTrue((table.loc[used, col] <= origin[used]).all(), col)
        used = table["pda_available_at"].notna()
        self.assertGreater(used.mean(), 0.9)
        label = pd.Series(origin - pd.to_timedelta(table["pda_run_age_min"].to_numpy(), unit="min"), index=table.index)
        issued = self.runs_hourly.groupby("run_label")["issued"].first()
        published = np.maximum(label + pd.Timedelta(minutes=40), label.map(issued).fillna(label) + pd.Timedelta(minutes=25))
        self.assertTrue((published[used] <= origin[used]).all())
        for target in self.targets[::10]:
            late = self.runs_hourly["available_at"] > target - HORIZON
            changed = self.runs_hourly.copy()
            changed.loc[late, list(PD_FIELDS.values())] = 1e6
            for runs in (changed, self.runs_hourly.loc[~late]):
                again = predispatch_features_all(pd.DatetimeIndex([target]), runs, self.realised.loc[[target]])
                pd.testing.assert_series_equal(
                    table.loc[target, EXTERNAL_PREDISPATCH_ALL].astype(float),
                    again.loc[target, EXTERNAL_PREDISPATCH_ALL].astype(float),
                    check_names=False,
                )

    def test_run_age_is_not_a_model_feature(self):
        self.assertFalse(any("run_age" in f for f in EXTERNAL_PREDISPATCH_ALL + EXTERNAL_PREDISPATCH_HALF))

    def test_half_hourly_and_revision_features_use_only_published_runs(self):
        if not (self.runs["run_label"].dt.minute == 30).any():
            self.skipTest("half-hour runs not in the consolidated file")
        table = predispatch_features_half(self.targets, self.runs, self.realised)
        origin = table.index - HORIZON
        for col in [c for c in table.columns if c.endswith("_available_at")]:
            used = table[col].notna()
            self.assertTrue((table.loc[used, col] <= origin[used]).all(), col)
        used = table["pdh_available_at"].notna()
        self.assertGreater(used.mean(), 0.9)
        self.assertGreater(table["pdh_rev30_price_t"].notna().mean(), 0.9)
        label = pd.Series(origin - pd.to_timedelta(table["pdh_run_age_min"].to_numpy(), unit="min"), index=table.index)
        issued = self.runs.groupby("run_label")["issued"].first()
        published = np.maximum(label + pd.Timedelta(minutes=40), label.map(issued).fillna(label) + pd.Timedelta(minutes=25))
        self.assertTrue((published[used] <= origin[used]).all())
        for target in self.targets[::10]:
            late = self.runs["available_at"] > target - HORIZON
            changed = self.runs.copy()
            changed.loc[late, list(PD_FIELDS.values())] = 1e6
            for runs in (changed, self.runs.loc[~late]):
                again = predispatch_features_half(pd.DatetimeIndex([target]), runs, self.realised.loc[[target]])
                pd.testing.assert_series_equal(
                    table.loc[target, EXTERNAL_PREDISPATCH_HALF].astype(float),
                    again.loc[target, EXTERNAL_PREDISPATCH_HALF].astype(float),
                    check_names=False,
                )


if __name__ == "__main__":
    unittest.main()
