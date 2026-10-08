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

WEATHER_LAG = pd.Timedelta(hours=12)  # value valid at V is usable from V - 12 h
PD_LABEL_LAG = pd.Timedelta(minutes=40)  # public file ~35 min after the run label
PD_ISSUE_LAG = pd.Timedelta(minutes=25)
PD_MAX_AGE = pd.Timedelta(hours=4)  # older runs are treated as missing (download gaps)
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


def load_predispatch(directory: Path = PREDISPATCH_DIR) -> pd.DataFrame:
    """Per-day extracts if present, otherwise the committed consolidated file."""
    files = sorted(directory.glob("pass*/*.parquet"))
    if files:
        runs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    elif PREDISPATCH_CONSOLIDATED.exists():
        runs = pd.read_parquet(PREDISPATCH_CONSOLIDATED)
    else:
        return pd.DataFrame()
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


def build_external(ready: pd.DataFrame, weather: pd.DataFrame | None = None, runs: pd.DataFrame | None = None) -> pd.DataFrame:
    weather = load_weather() if weather is None else weather
    runs = load_predispatch() if runs is None else runs
    wx = weather_features(ready.index, weather)
    pdx = predispatch_features(ready.index, runs, ready[["mcp_lag_5min", "demand_lag_5min"]])
    table = wx.join(pdx)
    check_availability(table)
    return table


def check_availability(table: pd.DataFrame) -> None:
    origin = table["forecast_origin"]
    for col in ("wx_available_at", "pd_available_at"):
        if col in table:
            late = table[col].notna() & (table[col] > origin)
            if late.any():
                raise RuntimeError(f"{col}: {int(late.sum())} rows use data published after the origin")


def main() -> None:
    from scripts.regime_switch import load

    ready, _ = load()
    table = build_external(ready)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURES_PATH)
    print(table.describe().T.to_string())
    print("non-null share:")
    print(table.notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
