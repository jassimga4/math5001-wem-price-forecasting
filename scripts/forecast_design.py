"""5-minute-ahead WEM forecast design.

Target at dispatch-interval end t is MCP_t. The forecast origin is t - 5 minutes.
Predictors must be known by that origin. Lags are taken only after the panel is
reindexed to a complete 5-minute grid, so a lag is a timestamp offset, not a
row offset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = pd.Timedelta(minutes=5)
GRID = pd.Timedelta(minutes=5)

TRAIN_END = pd.Timestamp("2025-09-30 23:55:00")
CAL_END = pd.Timestamp("2026-03-31 23:55:00")
INNER_VAL_START = pd.Timestamp("2025-07-01 00:00:00")

TARGET = "mcp"

RIDGE_FEATURES = [
    "mcp_lag_5min",
    "mcp_lag_30min",
    "mcp_lag_60min",
    "mcp_lag_1d",
    "demand_lag_5min",
    "dpv_lag_5min",
    "withdrawal_lag_5min",
    "cr_raise_lag_5min",
    "cr_lower_lag_5min",
    "reg_raise_lag_5min",
    "reg_lower_lag_5min",
    "rtp_last_complete",
    "stem_price",
    "stem_imbalance",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

TREE_FEATURES = [
    "mcp_lag_5min",
    "mcp_lag_30min",
    "mcp_lag_60min",
    "mcp_lag_1d",
    "demand_lag_5min",
    "dpv_lag_5min",
    "withdrawal_lag_5min",
    "cr_raise_lag_5min",
    "cr_lower_lag_5min",
    "reg_raise_lag_5min",
    "reg_lower_lag_5min",
    "rtp_last_complete",
    "stem_price",
    "stem_imbalance",
    "hour",
    "dow",
    "month",
]

TREE_CATEGORICAL = ["hour", "dow", "month"]

CONTEMPORANEOUS_BANNED = {
    "mcp",
    "rtp",
    "cr_raise",
    "cr_lower",
    "reg_raise",
    "reg_lower",
    "rocof",
    "operational_demand_mw",
    "unscheduled_demand_mw",
    "operational_withdrawal_mw",
    "dpv_mw",
    "residual_demand_mw",
    "scada_mw",
    "scada_mwh",
    "scheduled_energy_mw",
    "sent_out_mw",
    "sent_out_mwh",
    "dpv_mw_30min",
    "mcp_lag1",
    "mcp_lag6",
    "mcp_lag12",
}


def reindex_5min(df: pd.DataFrame, time_col: str = "interval_end") -> pd.DataFrame:
    """Return a copy indexed by a complete 5-minute grid. Missing slots stay NaN."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.drop_duplicates(time_col, keep="last").sort_values(time_col)
    out = out.set_index(time_col)
    if not out.index.is_unique or not out.index.is_monotonic_increasing:
        raise ValueError(f"{time_col} must be unique and strictly increasing")
    full = pd.date_range(out.index.min(), out.index.max(), freq=GRID)
    out = out.reindex(full)
    out.index.name = time_col
    return out


def _require_5min_index(index: pd.DatetimeIndex) -> None:
    if len(index) < 3:
        return
    deltas = index.to_series().diff().dropna().unique()
    if len(deltas) != 1 or deltas[0] != GRID:
        raise ValueError("lags require a complete 5-minute DatetimeIndex")


def time_lag(series: pd.Series, steps: int) -> pd.Series:
    """Lag by `steps` grid intervals. Valid only on a complete 5-minute index."""
    if steps < 1:
        raise ValueError("steps must be positive")
    _require_5min_index(series.index)
    return series.shift(steps)


def last_complete_rtp(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """RTP of the latest trading interval that had ended by the forecast origin.

    RTP is constant on the six dispatch intervals of a trading interval and is
    treated as known only once that trading interval has ended. A one-row shift
    is not enough: five of the six rows would still see the same interval's RTP.
    """
    work = raw.copy()
    work["interval_end"] = pd.to_datetime(work["interval_end"])
    work["trading_interval_end"] = pd.to_datetime(work["trading_interval_end"])
    published = (
        work.dropna(subset=["trading_interval_end"])
        .groupby("trading_interval_end", sort=True)["rtp"]
        .last()
        .rename("rtp_last_complete")
        .rename_axis("published_at")
        .reset_index()
    )
    origins = pd.DataFrame({"interval_end": index, "origin": index - HORIZON})
    merged = pd.merge_asof(
        origins.sort_values("origin"),
        published.sort_values("published_at"),
        left_on="origin",
        right_on="published_at",
        direction="backward",
    )
    return merged.set_index("interval_end")["rtp_last_complete"].reindex(index)


def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = np.asarray(index.hour)
    dow = np.asarray(index.dayofweek)
    month = np.asarray(index.month)
    return pd.DataFrame(
        {
            "hour": hour,
            "dow": dow,
            "month": month,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=index,
    )


def feature_availability() -> pd.DataFrame:
    rows = [
        ("mcp_lag_5min", "t-5min", "dispatch MCP published at interval end", "yes", "persistence"),
        ("mcp_lag_30min", "t-30min", "dispatch MCP published at interval end", "yes", "30-minute seasonal persistence"),
        ("mcp_lag_60min", "t-60min", "dispatch MCP published at interval end", "yes", "hour-ago price"),
        ("mcp_lag_1d", "t-1day", "dispatch MCP published at interval end", "yes", "daily seasonal persistence"),
        ("demand_lag_5min", "t-5min", "realised operational demand; no forecast vintage in the panel", "yes", "same-interval demand removed"),
        ("dpv_lag_5min", "t-5min", "realised estimated DPV; no forecast vintage in the panel", "yes", "same-interval DPV removed"),
        ("withdrawal_lag_5min", "t-5min", "realised operational withdrawal", "yes", "same-interval withdrawal removed"),
        ("cr_raise_lag_5min", "t-5min", "FCAS price published with the dispatch interval", "yes", "same-interval FCAS removed"),
        ("cr_lower_lag_5min", "t-5min", "FCAS price published with the dispatch interval", "yes", "same-interval FCAS removed"),
        ("reg_raise_lag_5min", "t-5min", "FCAS price published with the dispatch interval", "yes", "same-interval FCAS removed"),
        ("reg_lower_lag_5min", "t-5min", "FCAS price published with the dispatch interval", "yes", "same-interval FCAS removed"),
        ("rtp_last_complete", "last trading interval ended by t-5min", "RTP treated as known only when its trading interval has ended", "yes", "not rtp.shift(1)"),
        ("stem_price", "target trading interval", "assumed day-ahead STEM, published before real-time dispatch", "yes, assumption", "not verified against an AEMO publication log"),
        ("stem_imbalance", "target trading interval", "STEM bid minus offer; same day-ahead assumption as stem_price", "yes, assumption", "not verified against an AEMO publication log"),
        ("hour/dow/month", "target interval clock", "deterministic calendar of the target timestamp", "yes", "known before the origin"),
        ("operational_demand_mw", "target interval", "realised; not in the panel as a pre-dispatch forecast", "no", "excluded"),
        ("dpv_mw", "target interval", "realised estimate for the target interval", "no", "excluded"),
        ("scada_mw", "target interval", "realised facility SCADA", "no", "excluded; also collinear with lagged demand"),
        ("scheduled_energy_mw", "target interval", "vintage (forecast vs realised) not identified", "no", "excluded"),
        ("rtp", "target trading interval", "includes prices from the interval being forecast", "no", "excluded"),
        ("rocof", "target interval", "about 99.97% zeros, not missing", "no", "excluded for near-constancy"),
        ("year", "target year", "2026 is outside the training window", "not used", "regime shift, not a linear trend"),
    ]
    return pd.DataFrame(
        rows,
        columns=["feature", "timestamp_represented", "publication_or_assumption", "valid_at_origin", "note"],
    )


def build_exante_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Reindex, then build only features that are valid at t - 5 minutes."""
    work = raw.copy()
    work["interval_end"] = pd.to_datetime(work["interval_end"])
    if "trading_interval_end" in work.columns:
        work["trading_interval_end"] = pd.to_datetime(work["trading_interval_end"])
    panel = reindex_5min(work)
    _require_5min_index(panel.index)

    panel["mcp_lag_5min"] = time_lag(panel["mcp"], 1)
    panel["mcp_lag_30min"] = time_lag(panel["mcp"], 6)
    panel["mcp_lag_60min"] = time_lag(panel["mcp"], 12)
    panel["mcp_lag_1d"] = time_lag(panel["mcp"], 288)
    panel["demand_lag_5min"] = time_lag(panel["operational_demand_mw"], 1)
    panel["dpv_lag_5min"] = time_lag(panel["dpv_mw"], 1)
    panel["withdrawal_lag_5min"] = time_lag(panel["operational_withdrawal_mw"], 1)
    panel["cr_raise_lag_5min"] = time_lag(panel["cr_raise"], 1)
    panel["cr_lower_lag_5min"] = time_lag(panel["cr_lower"], 1)
    panel["reg_raise_lag_5min"] = time_lag(panel["reg_raise"], 1)
    panel["reg_lower_lag_5min"] = time_lag(panel["reg_lower"], 1)
    panel["rtp_last_complete"] = last_complete_rtp(work, panel.index)
    panel["stem_imbalance"] = panel["stem_bid_mwh"] - panel["stem_offer_mwh"]
    panel["forecast_origin"] = panel.index - HORIZON
    calendar = _calendar(panel.index)
    panel[calendar.columns] = calendar
    leaked = sorted(CONTEMPORANEOUS_BANNED.intersection(RIDGE_FEATURES + TREE_FEATURES))
    if leaked:
        raise RuntimeError(f"banned contemporaneous columns in the feature set: {leaked}")
    return panel


def model_ready(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = [TARGET, *features, "forecast_origin"]
    return frame.loc[:, cols].dropna().sort_index()


def split_mask(index: pd.DatetimeIndex) -> pd.Series:
    labels = pd.Series("test", index=index, dtype="object")
    labels.loc[index <= TRAIN_END] = "train"
    labels.loc[(index > TRAIN_END) & (index <= CAL_END)] = "calibration"
    return labels


def slice_split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    labels = split_mask(frame.index)
    return frame.loc[labels.eq(name)].copy()
