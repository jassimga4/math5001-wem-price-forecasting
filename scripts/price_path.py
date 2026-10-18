"""Recent price-path features known at the forecast origin.

The target interval T is forecast at origin T - 5 min. The latest realised
price is the interval labelled T - 5 min (``mcp_lag_5min``). Every feature here
reads only realised prices labelled T - 5 min or earlier: on a regular 5-minute
grid that is ``shift(k)`` with k >= 1. Nothing from T or later is used.

These are summaries of the path (counts, a slope, run lengths, spread), not
extra MCP lags, which docs/spike_forecast_next_steps.md rules out. The "near
band" thresholds are fixed from the training window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PATH_FEATURES = [
    "path_near_up_30", "path_near_up_60", "path_near_down_30", "path_near_down_60",
    "path_slope_15", "path_climb_run", "path_fall_run", "path_std_30", "path_max60_minus_last",
]
MAX_RUN = 24  # run lengths are capped at 2 hours


def _run_length(step: pd.DataFrame) -> pd.Series:
    """Consecutive True values counting back from the first column (k = 1)."""
    alive = np.ones(len(step), dtype=bool)
    count = np.zeros(len(step), dtype=float)
    for col in step.columns:
        alive &= step[col].fillna(False).to_numpy(dtype=bool)
        count += alive
    return pd.Series(count, index=step.index)


def price_path_features(prices: pd.Series, targets: pd.DatetimeIndex, near_up: float, near_down: float) -> pd.DataFrame:
    """Path features for each target from realised prices labelled T - 5 min or earlier.

    ``prices`` is realised MCP indexed by interval label; it is put on a regular
    5-minute grid so that ``shift(k)`` is the interval k * 5 minutes before T.
    """
    targets = pd.DatetimeIndex(targets).astype("datetime64[ns]")
    start, end = min(prices.index.min(), targets.min()), max(prices.index.max(), targets.max())
    grid = prices.reindex(pd.date_range(start, end, freq="5min"))
    lag = {k: grid.shift(k) for k in range(1, MAX_RUN + 2)}  # k >= 1 only: T - 5 min and earlier
    last = lag[1]
    w30 = pd.concat([lag[k] for k in range(1, 7)], axis=1)
    w60 = pd.concat([lag[k] for k in range(1, 13)], axis=1)
    out = pd.DataFrame(index=grid.index)
    out["path_near_up_30"] = (w30 >= near_up).sum(axis=1).where(w30.notna().any(axis=1))
    out["path_near_up_60"] = (w60 >= near_up).sum(axis=1).where(w60.notna().any(axis=1))
    out["path_near_down_30"] = (w30 <= near_down).sum(axis=1).where(w30.notna().any(axis=1))
    out["path_near_down_60"] = (w60 <= near_down).sum(axis=1).where(w60.notna().any(axis=1))
    # least-squares slope over T-20, T-15, T-10, T-5 ($ per 5 min); x centred at 0
    x = np.array([1.5, 0.5, -0.5, -1.5])  # k = 1..4, latest has the largest x
    pts = pd.concat([lag[k] for k in range(1, 5)], axis=1).to_numpy()
    out["path_slope_15"] = (pts * x).sum(axis=1) / (x ** 2).sum()  # NaN if any point is missing
    rises = pd.DataFrame({k: lag[k] > lag[k + 1] for k in range(1, MAX_RUN + 1)}, index=grid.index)
    falls = pd.DataFrame({k: lag[k] < lag[k + 1] for k in range(1, MAX_RUN + 1)}, index=grid.index)
    out["path_climb_run"] = _run_length(rises).where(last.notna())
    out["path_fall_run"] = _run_length(falls).where(last.notna())
    out["path_std_30"] = w30.std(axis=1, ddof=0).where(w30.notna().sum(axis=1) >= 3)
    out["path_max60_minus_last"] = w60.max(axis=1) - last
    return out.reindex(targets)


def near_band(train_prices: pd.Series, upper: float = 0.90, lower: float = 0.10) -> tuple[float, float]:
    """Training-window quantiles used as the 'near the spike band' thresholds."""
    return float(train_prices.quantile(upper)), float(train_prices.quantile(lower))
