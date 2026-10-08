"""Leakage-safe external features keyed to the 5-minute forecast origins.

Every value carries an availability time, and a value is used at origin
``o = T - 5 min`` only if its availability time is at or before ``o``.

Sources (see data/external/<source>/README.md):

* Open-Meteo Previous Runs API, ``previous_day1`` offset (ECMWF IFS 0.25 and
  GFS). A value valid at hour V was issued by a run initialised at least 24 h
  before V. It is treated as available at V - 12 h, which leaves at least 6 h
  for publication after the latest possible initialisation.
* AEMO WEM Reference pre-dispatch runs. A run labelled R is treated as
  available at max(R + 40 min, issue time + 25 min). The public file appears
  about 35 min after R.

All times are naive AWST (UTC+8), the same clock as the panel. The panel
labels intervals by AEMO's dispatch interval time and the project takes the
origin as that label minus 5 minutes. If the label is the interval start the
true origin is 5 minutes later, so the availability check is conservative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_design import HORIZON  # noqa: E402

WEATHER_PATH = ROOT / "data" / "external" / "open_meteo" / "previous_day1_hourly.csv.gz"
PREDISPATCH_DIR = ROOT / "data" / "external" / "aemo_predispatch" / "extract"
PREDISPATCH_CONSOLIDATED = ROOT / "data" / "external" / "aemo_predispatch" / "predispatch_runs_first9h.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "external_features.parquet"
COVERAGE_PATH = ROOT / "reports" / "forecast" / "spike_predispatch_coverage.csv"

WEATHER_LAG = pd.Timedelta(hours=12)  # value valid at V is usable from V - 12 h
PD_LABEL_LAG = pd.Timedelta(minutes=40)  # public file ~35 min after the run label
PD_ISSUE_LAG = pd.Timedelta(minutes=25)
PD_MAX_AGE = pd.Timedelta(hours=4)  # older runs are treated as missing (download gaps)
PD_REVISION_MAX_GAP = pd.Timedelta(hours=2)  # revisions only between runs at most 2 h apart
WIND_SITES = ("geraldton", "badgingarra", "merredin", "albany")
PD_FIELDS = {
    "prices.energy": "price",
    "prices.regulationRaise": "reg_raise_price",
    "prices.contingencyRaise": "cont_raise_price",
    "marketServiceRequirements.energy": "energy_req",
    "marketServiceRequirements.contingencyRaise": "cont_raise_req",
    "inServiceQuantities.energyInjectionCapacity": "in_service_cap",
    "availableQuantities.energyInjectionCapacity": "available_cap",
    "availableQuantities.contingencyRaise": "cont_raise_avail",
    "marketShortfalls.energyDeficit": "energy_deficit",
}


# --------------------------------------------------------------------------- weather


def load_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    """Hourly forecasts, model-averaged, wide by ``site_variable``, indexed by valid AWST hour."""
    raw = pd.read_csv(path, parse_dates=["valid_awst", "available_awst"])
    for col in ("valid_awst", "available_awst"):
        raw[col] = raw[col].astype("datetime64[ns]")
    if not (raw["available_awst"] <= raw["valid_awst"] - WEATHER_LAG).all():
        raise ValueError("weather file has an availability time later than the rule allows")
    mean = raw.groupby(["valid_awst", "site", "variable"])["value"].mean().unstack(["site", "variable"])
    mean.columns = [f"{site}_{var}" for site, var in mean.columns]
    full = pd.date_range(mean.index.min(), mean.index.max(), freq="h")
    return mean.reindex(full).sort_index()


def weather_at(hourly: pd.DataFrame, when: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Linear interpolation of hourly forecasts at ``when``.

    Returns the values and, per row, the latest valid hour that was read. The
    caller turns that hour into an availability time.
    """
    when = pd.DatetimeIndex(when).astype("datetime64[ns]")
    lo = when.floor("h")
    hi = when.ceil("h")
    w = ((when - lo) / pd.Timedelta(hours=1)).to_numpy()[:, None]
    a = hourly.reindex(lo).to_numpy()
    b = hourly.reindex(hi).to_numpy()
    values = np.where(w == 0, a, (1 - w) * a + w * b)
    return pd.DataFrame(values, index=when, columns=hourly.columns), pd.Series(hi, index=when)


def _wind_cf(speed: pd.DataFrame) -> pd.DataFrame:
    """Crude capacity-factor proxy for 100 m wind speed in m/s (cut-in 3, rated 12)."""
    return ((speed - 3.0) / 9.0).clip(0.0, 1.0)


def weather_features(targets: pd.DatetimeIndex, hourly: pd.DataFrame) -> pd.DataFrame:
    targets = pd.DatetimeIndex(targets).astype("datetime64[ns]")
    origin = targets - HORIZON
    leads = {"o": -HORIZON, "t": pd.Timedelta(0), "t30": pd.Timedelta(minutes=30), "t60": pd.Timedelta(minutes=60),
             "t120": pd.Timedelta(minutes=120), "t180": pd.Timedelta(minutes=180)}
    at: dict[str, pd.DataFrame] = {}
    latest = None
    for name, lead in leads.items():
        values, hour_read = weather_at(hourly, targets + lead)
        values.index = targets
        at[name] = values
        read = hour_read.to_numpy()
        latest = read if latest is None else np.maximum(latest, read)
    p = "perth_metro_"
    out = pd.DataFrame(index=targets)
    out["wx_perth_temp_t"] = at["t"][p + "temperature_2m"]
    out["wx_perth_temp_d60"] = at["t60"][p + "temperature_2m"] - at["t"][p + "temperature_2m"]
    out["wx_perth_temp_max3h"] = pd.concat([at[k][p + "temperature_2m"] for k in ("t", "t60", "t120", "t180")], axis=1).max(axis=1)
    out["wx_perth_cloud_t"] = at["t"][p + "cloud_cover"]
    out["wx_perth_cloud_d60"] = at["t60"][p + "cloud_cover"] - at["t"][p + "cloud_cover"]
    out["wx_perth_ghi_t"] = at["t"][p + "shortwave_radiation"]
    out["wx_perth_ghi_drop_5"] = at["o"][p + "shortwave_radiation"] - at["t"][p + "shortwave_radiation"]
    out["wx_perth_ghi_drop_30"] = at["o"][p + "shortwave_radiation"] - at["t30"][p + "shortwave_radiation"]
    out["wx_perth_ghi_drop_60"] = at["o"][p + "shortwave_radiation"] - at["t60"][p + "shortwave_radiation"]
    out["wx_merredin_ghi_t"] = at["t"]["merredin_shortwave_radiation"]
    for site in WIND_SITES:
        out[f"wx_wind_{site}_t"] = at["t"][f"{site}_wind_speed_100m"]
    cols = [f"{s}_wind_speed_100m" for s in WIND_SITES]
    cf_t = _wind_cf(at["t"][cols]).mean(axis=1)
    out["wx_wind_cf_t"] = cf_t
    out["wx_wind_cf_d60"] = _wind_cf(at["t60"][cols]).mean(axis=1) - cf_t
    out["wx_wind_cf_d180"] = _wind_cf(at["t180"][cols]).mean(axis=1) - cf_t
    out["wx_available_at"] = pd.DatetimeIndex(latest) - WEATHER_LAG
    out["forecast_origin"] = origin
    return out


# --------------------------------------------------------------------------- pre-dispatch


def load_predispatch(directory: Path = PREDISPATCH_DIR, consolidated: Path = PREDISPATCH_CONSOLIDATED) -> pd.DataFrame:
    """The committed consolidated file, or the per-day extracts if it has not been built.

    The consolidated file keeps the first 9 hours of every run. Features read at
    most 4 h (run age) + 2 h (look-ahead) past a run label, so it gives the same
    features as the full extracts and makes the results reproducible from git.
    """
    if consolidated.exists():
        runs = pd.read_parquet(consolidated)
    else:
        files = sorted(directory.glob("pass*/*.parquet"))
        if not files:
            return pd.DataFrame()
        runs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    runs["run_label"] = pd.to_datetime(runs["run_label"], format="%Y%m%d%H%M")
    runs["issued"] = pd.to_datetime(runs["issue_id_time"], format="%Y%m%d%H%M%S", errors="coerce")
    runs["interval"] = pd.to_datetime(runs["dispatch_interval"].str.slice(0, 19))
    for col in ("run_label", "issued", "interval"):
        runs[col] = runs[col].astype("datetime64[ns]")
    runs["available_at"] = predispatch_available_at(runs["run_label"], runs["issued"])
    keep = ["run_label", "issued", "available_at", "interval", *PD_FIELDS]
    runs = runs.loc[:, keep].rename(columns=PD_FIELDS)
    return runs.drop_duplicates(["run_label", "interval"]).sort_values(["run_label", "interval"])


def predispatch_available_at(label: pd.Series, issued: pd.Series) -> pd.Series:
    by_label = label + PD_LABEL_LAG
    by_issue = issued + PD_ISSUE_LAG
    return pd.Series(np.maximum(by_label.to_numpy(), by_issue.fillna(by_label).to_numpy()), index=label.index)


def predispatch_features(targets: pd.DatetimeIndex, runs: pd.DataFrame, realised: pd.DataFrame) -> pd.DataFrame:
    """Features from the latest pre-dispatch run available at each origin.

    ``realised`` must hold ``mcp_lag_5min`` and ``demand_lag_5min`` on the
    target index; both are already known at the origin.
    """
    targets = pd.DatetimeIndex(targets).astype("datetime64[ns]")
    out = pd.DataFrame(index=targets)
    if runs.empty:
        return out
    meta = runs.groupby("run_label", as_index=False)["available_at"].max().sort_values("available_at")
    origins = pd.DataFrame({"target": targets, "origin": targets - HORIZON}).sort_values("origin")
    chosen = pd.merge_asof(origins, meta, left_on="origin", right_on="available_at", direction="backward")
    chosen = chosen.set_index("target").reindex(targets)
    stale = (chosen["origin"] - chosen["run_label"]) > PD_MAX_AGE
    chosen.loc[stale, ["run_label", "available_at"]] = pd.NaT
    slot = targets.floor("30min")
    wide = runs.set_index(["run_label", "interval"])
    fields = list(PD_FIELDS.values())

    def lookup(offset: pd.Timedelta) -> pd.DataFrame:
        key = pd.MultiIndex.from_arrays([chosen["run_label"], slot + offset])
        got = wide.reindex(key)[fields]
        got.index = targets
        return got

    now, nxt = lookup(pd.Timedelta(0)), lookup(pd.Timedelta(minutes=30))
    ahead = [lookup(pd.Timedelta(minutes=30 * k))["price"] for k in range(0, 5)]
    out["pd_price_t"] = now["price"]
    out["pd_price_next"] = nxt["price"]
    out["pd_price_max2h"] = pd.concat(ahead, axis=1).max(axis=1)
    out["pd_price_minus_last"] = now["price"] - realised["mcp_lag_5min"]
    out["pd_energy_req_t"] = now["energy_req"]
    out["pd_energy_req_minus_last_demand"] = now["energy_req"] - realised["demand_lag_5min"]
    out["pd_energy_req_ramp30"] = nxt["energy_req"] - now["energy_req"]
    out["pd_headroom"] = now["in_service_cap"] - now["energy_req"]
    out["pd_available_cap"] = now["available_cap"]
    out["pd_cont_raise_margin"] = now["cont_raise_avail"] - now["cont_raise_req"]
    out["pd_reg_raise_price"] = now["reg_raise_price"]
    out["pd_cont_raise_price"] = now["cont_raise_price"]
    out["pd_energy_deficit"] = now["energy_deficit"]
    out["pd_run_age_min"] = (chosen["origin"] - chosen["run_label"]).dt.total_seconds() / 60.0
    out["pd_available_at"] = chosen["available_at"]
    missing = now["price"].isna()
    out.loc[missing, "pd_available_at"] = pd.NaT
    out.loc[missing, EXTERNAL_PREDISPATCH] = np.nan
    return out


def even_hour_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """The pass-1 subset (runs labelled at even hours), as used by the stage-1 ``pd_`` features."""
    label = runs["run_label"]
    return runs.loc[(label.dt.minute == 0) & (label.dt.hour % 2 == 0)]


def predispatch_features_all(targets: pd.DatetimeIndex, runs: pd.DataFrame, realised: pd.DataFrame) -> pd.DataFrame:
    """``pda_`` features from every hourly run, plus revisions between consecutive runs.

    L is the latest run with ``available_at <= origin`` (at most 4 h old). P is
    the run labelled just before L; it is used only if it too was available by
    the origin and is at most 2 h older than L. Revisions compare L and P for
    the half hour containing T. The within-run change compares L's forecast for
    that half hour with its forecast one hour earlier, or with L's first
    interval if that is later. ``pda_run_age_min`` is kept for coverage reports
    only and is not a model feature.
    """
    targets = pd.DatetimeIndex(targets).astype("datetime64[ns]")
    out = pd.DataFrame(index=targets)
    if runs.empty:
        return out
    meta = runs.groupby("run_label", as_index=False)["available_at"].max().sort_values("run_label")
    meta["prev_label"] = meta["run_label"].shift(1)
    meta["prev_available_at"] = meta["available_at"].shift(1)
    origins = pd.DataFrame({"target": targets, "origin": targets - HORIZON}).sort_values("origin")
    chosen = pd.merge_asof(origins, meta.sort_values("available_at"), left_on="origin", right_on="available_at",
                           direction="backward")
    chosen = chosen.set_index("target").reindex(targets)
    stale = (chosen["origin"] - chosen["run_label"]) > PD_MAX_AGE
    chosen.loc[stale, ["run_label", "available_at", "prev_label", "prev_available_at"]] = pd.NaT
    prev_bad = (chosen["prev_available_at"] > chosen["origin"]) | (
        (chosen["run_label"] - chosen["prev_label"]) > PD_REVISION_MAX_GAP)
    chosen.loc[prev_bad.fillna(True), ["prev_label", "prev_available_at"]] = pd.NaT
    slot = targets.floor("30min")
    wide = runs.set_index(["run_label", "interval"])
    fields = list(PD_FIELDS.values())

    def lookup(labels: pd.Series, when) -> pd.DataFrame:
        key = pd.MultiIndex.from_arrays([labels, pd.DatetimeIndex(when)])
        got = wide.reindex(key)[fields]
        got.index = targets
        return got

    label = chosen["run_label"]
    now, nxt = lookup(label, slot), lookup(label, slot + pd.Timedelta(minutes=30))
    ahead = [lookup(label, slot + pd.Timedelta(minutes=30 * k))["price"] for k in range(0, 5)]
    earlier_when = np.maximum((slot - pd.Timedelta(hours=1)).to_numpy(), label.to_numpy())
    earlier = lookup(label, earlier_when)
    prev = lookup(chosen["prev_label"], slot)
    prev_ahead = [lookup(chosen["prev_label"], slot + pd.Timedelta(minutes=30 * k))["price"] for k in range(0, 5)]
    out["pda_price_t"] = now["price"]
    out["pda_price_next"] = nxt["price"]
    out["pda_price_max2h"] = pd.concat(ahead, axis=1).max(axis=1)
    out["pda_price_minus_last"] = now["price"] - realised["mcp_lag_5min"]
    out["pda_energy_req_t"] = now["energy_req"]
    out["pda_energy_req_minus_last_demand"] = now["energy_req"] - realised["demand_lag_5min"]
    out["pda_energy_req_ramp30"] = nxt["energy_req"] - now["energy_req"]
    out["pda_headroom"] = now["in_service_cap"] - now["energy_req"]
    out["pda_available_cap"] = now["available_cap"]
    out["pda_cont_raise_margin"] = now["cont_raise_avail"] - now["cont_raise_req"]
    out["pda_reg_raise_price"] = now["reg_raise_price"]
    out["pda_cont_raise_price"] = now["cont_raise_price"]
    out["pda_energy_deficit"] = now["energy_deficit"]
    out["pda_price_change_1h"] = now["price"] - earlier["price"]
    out["pda_energy_req_change_1h"] = now["energy_req"] - earlier["energy_req"]
    out["pda_rev_price_t"] = now["price"] - prev["price"]
    out["pda_rev_price_max2h"] = out["pda_price_max2h"] - pd.concat(prev_ahead, axis=1).max(axis=1, skipna=False)
    out["pda_rev_energy_req_t"] = now["energy_req"] - prev["energy_req"]
    out["pda_rev_headroom_t"] = out["pda_headroom"] - (prev["in_service_cap"] - prev["energy_req"])
    out["pda_run_age_min"] = (chosen["origin"] - label).dt.total_seconds() / 60.0
    out["pda_available_at"] = chosen["available_at"]
    out["pda_prev_available_at"] = chosen["prev_available_at"]
    no_prev = prev["price"].isna()
    out.loc[no_prev, "pda_prev_available_at"] = pd.NaT
    out.loc[no_prev, EXTERNAL_REVISION] = np.nan
    missing = now["price"].isna()
    out.loc[missing, ["pda_available_at", "pda_prev_available_at"]] = pd.NaT
    out.loc[missing, [*EXTERNAL_PREDISPATCH_ALL, "pda_run_age_min"]] = np.nan
    return out


def on_the_hour_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Passes 1 and 2 (runs labelled at :00), as used by the stage-1b ``pda_`` features."""
    return runs.loc[runs["run_label"].dt.minute == 0]


def _previous_run(meta: pd.DataFrame, back: pd.Timedelta) -> pd.DataFrame:
    """For each run, the latest run labelled at least ``back`` earlier (label and availability)."""
    left = meta[["run_label"]].assign(key=meta["run_label"] - back).sort_values("key")
    right = meta[["run_label", "available_at"]].rename(columns={"run_label": "prev_label", "available_at": "prev_available_at"})
    got = pd.merge_asof(left, right.sort_values("prev_label"), left_on="key", right_on="prev_label", direction="backward")
    return got.set_index("run_label")[["prev_label", "prev_available_at"]]


def predispatch_features_half(targets: pd.DatetimeIndex, runs: pd.DataFrame, realised: pd.DataFrame) -> pd.DataFrame:
    """``pdh_`` features from every half-hourly run, with 30 and 60 minute revisions.

    L is the latest run with ``available_at <= origin`` (at most 4 h old). For
    the 30 minute revision P30 is the latest run labelled at least 30 min before
    L and at most 1 h before it; for the 60 minute revision P60 is the latest run
    labelled at least 60 min before L and at most 2 h before it. Each previous
    run is used only if it was itself available by the origin. Levels and the
    within-run 1 h change are as for ``pda_``. ``pdh_run_age_min`` is for
    coverage reports only.
    """
    targets = pd.DatetimeIndex(targets).astype("datetime64[ns]")
    out = pd.DataFrame(index=targets)
    if runs.empty:
        return out
    meta = runs.groupby("run_label", as_index=False)["available_at"].max().sort_values("run_label")
    origins = pd.DataFrame({"target": targets, "origin": targets - HORIZON}).sort_values("origin")
    chosen = pd.merge_asof(origins, meta.sort_values("available_at"), left_on="origin", right_on="available_at",
                           direction="backward")
    chosen = chosen.set_index("target").reindex(targets)
    stale = (chosen["origin"] - chosen["run_label"]) > PD_MAX_AGE
    chosen.loc[stale, ["run_label", "available_at"]] = pd.NaT
    label = chosen["run_label"]
    slot = targets.floor("30min")
    wide = runs.set_index(["run_label", "interval"])
    fields = list(PD_FIELDS.values())

    def lookup(labels, when) -> pd.DataFrame:
        key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(labels), pd.DatetimeIndex(when)])
        got = wide.reindex(key)[fields]
        got.index = targets
        return got

    now, nxt = lookup(label, slot), lookup(label, slot + pd.Timedelta(minutes=30))
    ahead = [lookup(label, slot + pd.Timedelta(minutes=30 * k))["price"] for k in range(0, 5)]
    earlier = lookup(label, np.maximum((slot - pd.Timedelta(hours=1)).to_numpy(), label.to_numpy()))
    p = "pdh_"
    out[p + "price_t"] = now["price"]
    out[p + "price_next"] = nxt["price"]
    out[p + "price_max2h"] = pd.concat(ahead, axis=1).max(axis=1)
    out[p + "price_minus_last"] = now["price"] - realised["mcp_lag_5min"]
    out[p + "energy_req_t"] = now["energy_req"]
    out[p + "energy_req_minus_last_demand"] = now["energy_req"] - realised["demand_lag_5min"]
    out[p + "energy_req_ramp30"] = nxt["energy_req"] - now["energy_req"]
    out[p + "headroom"] = now["in_service_cap"] - now["energy_req"]
    out[p + "available_cap"] = now["available_cap"]
    out[p + "cont_raise_margin"] = now["cont_raise_avail"] - now["cont_raise_req"]
    out[p + "reg_raise_price"] = now["reg_raise_price"]
    out[p + "cont_raise_price"] = now["cont_raise_price"]
    out[p + "energy_deficit"] = now["energy_deficit"]
    out[p + "price_change_1h"] = now["price"] - earlier["price"]
    out[p + "energy_req_change_1h"] = now["energy_req"] - earlier["energy_req"]
    out[p + "run_age_min"] = (chosen["origin"] - label).dt.total_seconds() / 60.0
    out[p + "available_at"] = chosen["available_at"]
    for tag, back, gap in (("rev30", pd.Timedelta(minutes=30), pd.Timedelta(hours=1)),
                           ("rev60", pd.Timedelta(minutes=60), pd.Timedelta(hours=2))):
        prev_meta = _previous_run(meta, back)
        prev = prev_meta.reindex(label)
        prev.index = targets
        bad = (prev["prev_available_at"] > chosen["origin"]) | ((label - prev["prev_label"]) > gap)
        prev.loc[bad.fillna(True), ["prev_label", "prev_available_at"]] = pd.NaT
        before = lookup(prev["prev_label"], slot)
        before_ahead = [lookup(prev["prev_label"], slot + pd.Timedelta(minutes=30 * k))["price"] for k in range(0, 5)]
        cols = [f"{p}{tag}_price_t", f"{p}{tag}_price_max2h", f"{p}{tag}_energy_req_t", f"{p}{tag}_headroom_t"]
        out[cols[0]] = now["price"] - before["price"]
        out[cols[1]] = out[p + "price_max2h"] - pd.concat(before_ahead, axis=1).max(axis=1, skipna=False)
        out[cols[2]] = now["energy_req"] - before["energy_req"]
        out[cols[3]] = out[p + "headroom"] - (before["in_service_cap"] - before["energy_req"])
        out[f"{p}{tag}_prev_available_at"] = prev["prev_available_at"]
        none = before["price"].isna()
        out.loc[none, f"{p}{tag}_prev_available_at"] = pd.NaT
        out.loc[none, cols] = np.nan
    missing = now["price"].isna()
    out.loc[missing, [c for c in out.columns if c.endswith("_available_at")]] = pd.NaT
    out.loc[missing, [*EXTERNAL_PREDISPATCH_HALF, p + "run_age_min"]] = np.nan
    return out


# --------------------------------------------------------------------------- table


EXTERNAL_WEATHER = [
    "wx_perth_temp_t", "wx_perth_temp_d60", "wx_perth_temp_max3h", "wx_perth_cloud_t", "wx_perth_cloud_d60",
    "wx_perth_ghi_t", "wx_perth_ghi_drop_5", "wx_perth_ghi_drop_30", "wx_perth_ghi_drop_60", "wx_merredin_ghi_t",
    *[f"wx_wind_{s}_t" for s in WIND_SITES], "wx_wind_cf_t", "wx_wind_cf_d60", "wx_wind_cf_d180",
]
EXTERNAL_PREDISPATCH = [
    "pd_price_t", "pd_price_next", "pd_price_max2h", "pd_price_minus_last", "pd_energy_req_t",
    "pd_energy_req_minus_last_demand", "pd_energy_req_ramp30", "pd_headroom", "pd_available_cap",
    "pd_cont_raise_margin", "pd_reg_raise_price", "pd_cont_raise_price", "pd_energy_deficit", "pd_run_age_min",
]
# Stage 1b: every hourly run, revision features, no run age (it partly stands in for time of day).
EXTERNAL_REVISION = ["pda_rev_price_t", "pda_rev_price_max2h", "pda_rev_energy_req_t", "pda_rev_headroom_t"]
EXTERNAL_PREDISPATCH_ALL = [
    "pda_price_t", "pda_price_next", "pda_price_max2h", "pda_price_minus_last", "pda_energy_req_t",
    "pda_energy_req_minus_last_demand", "pda_energy_req_ramp30", "pda_headroom", "pda_available_cap",
    "pda_cont_raise_margin", "pda_reg_raise_price", "pda_cont_raise_price", "pda_energy_deficit",
    "pda_price_change_1h", "pda_energy_req_change_1h", *EXTERNAL_REVISION,
]
# Stage 1c: every half-hourly run, revisions over 30 and 60 minutes.
EXTERNAL_REVISION_HALF = [
    f"pdh_{tag}_{name}" for tag in ("rev30", "rev60") for name in ("price_t", "price_max2h", "energy_req_t", "headroom_t")
]
EXTERNAL_PREDISPATCH_HALF = [
    "pdh_price_t", "pdh_price_next", "pdh_price_max2h", "pdh_price_minus_last", "pdh_energy_req_t",
    "pdh_energy_req_minus_last_demand", "pdh_energy_req_ramp30", "pdh_headroom", "pdh_available_cap",
    "pdh_cont_raise_margin", "pdh_reg_raise_price", "pdh_cont_raise_price", "pdh_energy_deficit",
    "pdh_price_change_1h", "pdh_energy_req_change_1h", *EXTERNAL_REVISION_HALF,
]


def build_external(ready: pd.DataFrame, weather: pd.DataFrame | None = None, runs: pd.DataFrame | None = None) -> pd.DataFrame:
    weather = load_weather() if weather is None else weather
    runs = load_predispatch() if runs is None else runs
    wx = weather_features(ready.index, weather)
    realised = ready[["mcp_lag_5min", "demand_lag_5min"]]
    if runs.empty:
        table = wx
    else:
        pdx = predispatch_features(ready.index, even_hour_runs(runs), realised)
        pda = predispatch_features_all(ready.index, on_the_hour_runs(runs), realised)
        table = wx.join(pdx).join(pda)
        if (runs["run_label"].dt.minute == 30).any():
            table = table.join(predispatch_features_half(ready.index, runs, realised))
    check_availability(table)
    return table


def predispatch_coverage(table: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per run set and split: origins with a usable run, run age, and runs present in the archive."""
    from scripts.forecast_design import split_mask

    split = split_mask(table.index)
    sets = (("even_hour", "pd_", even_hour_runs(runs), "2h"), ("all_hourly", "pda_", on_the_hour_runs(runs), "1h"),
            ("half_hourly", "pdh_", runs, "30min"))
    rows = []
    for set_name, prefix, subset, step in sets:
        labels = pd.Series(subset["run_label"].unique())
        run_split = split_mask(pd.DatetimeIndex(labels))
        for name in ("train", "calibration", "test"):
            block = table.loc[split.eq(name).to_numpy()]
            if block.empty or f"{prefix}price_t" not in block:
                continue
            lo, hi = block.index.min(), block.index.max()
            expected = pd.date_range(lo.ceil(step), hi.floor(step), freq=step)
            present = pd.DatetimeIndex(labels[run_split.eq(name).to_numpy()])
            age = block[f"{prefix}run_age_min"].dropna()
            row = {
                "run_set": set_name,
                "split": name,
                "first_target": lo,
                "last_target": hi,
                "origins": int(len(block)),
                "origins_with_predispatch": int(block[f"{prefix}price_t"].notna().sum()),
                "share_with_predispatch": float(block[f"{prefix}price_t"].notna().mean()),
                "runs_expected": int(len(expected)),
                "runs_present": int(present.isin(expected).sum()),
                "run_age_min_median": float(age.median()) if len(age) else np.nan,
                "run_age_min_p95": float(age.quantile(0.95)) if len(age) else np.nan,
                "run_age_min_max": float(age.max()) if len(age) else np.nan,
                "share_age_40_100": float(age.between(40, 100).mean()) if len(age) else np.nan,
                "share_age_40_70": float(age.between(40, 70).mean()) if len(age) else np.nan,
            }
            if prefix == "pda_":
                row["share_with_revision"] = float(block["pda_rev_price_t"].notna().mean())
            if prefix == "pdh_":
                row["share_with_revision"] = float(block["pdh_rev30_price_t"].notna().mean())
                row["share_with_revision60"] = float(block["pdh_rev60_price_t"].notna().mean())

            rows.append(row)
    return pd.DataFrame(rows)


def check_availability(table: pd.DataFrame) -> None:
    origin = table["forecast_origin"]
    for col in [c for c in table.columns if c.endswith("_available_at")]:
        if col in table:
            late = table[col].notna() & (table[col] > origin)
            if late.any():
                raise RuntimeError(f"{col}: {int(late.sum())} rows use data published after the origin")


def main() -> None:
    from scripts.regime_switch import load

    ready, _ = load()
    runs = load_predispatch()
    table = build_external(ready, runs=runs)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURES_PATH)
    coverage = predispatch_coverage(table, runs)
    coverage.to_csv(COVERAGE_PATH, index=False)
    print(coverage.to_string(index=False))
    print(table.describe().T.to_string())
    print("non-null share:")
    print(table.notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
