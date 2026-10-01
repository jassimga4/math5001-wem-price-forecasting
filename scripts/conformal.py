"""Sliding-window split conformal intervals for 5-minute MCP forecasts.

This is an inductive, EnbPI-style residual window, not a full transductive
conformal predictive system. The point model stays frozen. At each origin the
interval is the point forecast plus empirical residual quantiles from a recent
time window of out-of-sample residuals. In-sample training residuals are not
used.

The finite-sample rank follows split conformal: for a window of n scores the
upper quantile uses ceil((n+1)(1-alpha/2))/n, and the lower quantile uses
floor((n+1)(alpha/2))/n. A hard window is a special case of the weighted
conformal construction in Barber, Candès, Ramdas and Tibshirani (2023). It does
not restore an unconditional finite-sample coverage guarantee under serial
dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_CANDIDATES = {
    "2d": 2 * 288,
    "7d": 7 * 288,
    "14d": 14 * 288,
    "30d": 30 * 288,
}
ALPHAS = (0.50, 0.20, 0.10, 0.05)
SELECTION_ALPHA = 0.10
SCALE_WINDOW = 288
SCALE_FLOOR = 1.0
COVERAGE_FLOOR_90 = 0.89
COVERAGE_FLOOR_95 = 0.93


def lower_rank(n: int, alpha: float) -> int:
    """0-based order statistic for the 1-based rank floor((n+1) alpha/2)."""
    rank = int(np.floor((n + 1) * (alpha / 2.0)))
    return min(max(rank, 1), n) - 1


def upper_rank(n: int, alpha: float) -> int:
    """0-based order statistic for the 1-based rank ceil((n+1)(1-alpha/2))."""
    rank = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
    return min(max(rank, 1), n) - 1


def empirical_quantile(scores: np.ndarray, alpha: float, side: str) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan
    rank = upper_rank(n, alpha) if side == "upper" else lower_rank(n, alpha)
    return float(values[rank])


def fixed_split_quantiles(residuals: np.ndarray, alpha: float) -> tuple[float, float]:
    finite = np.asarray(residuals, dtype=float)
    finite = finite[np.isfinite(finite)]
    return (
        empirical_quantile(finite, alpha, "lower"),
        empirical_quantile(finite, alpha, "upper"),
    )


def sliding_residual_quantiles(
    residuals: np.ndarray,
    positions: np.ndarray,
    window: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles of residuals on (origin - window, origin], excluding the origin."""
    values = np.asarray(residuals, dtype=float)
    positions = np.asarray(positions, dtype=int)
    n = int(window)
    if n < 1:
        raise ValueError("conformal window must be positive")
    if len(positions) and np.any(positions < n):
        raise ValueError("conformal window starts before the residual history")
    lo_at = lower_rank(n, alpha)
    hi_at = upper_rank(n, alpha)
    lower = np.empty(len(positions))
    upper = np.empty(len(positions))
    offsets = np.arange(n)
    chunk = 1024
    for start in range(0, len(positions), chunk):
        pos = positions[start : start + chunk]
        sample = values[pos[:, None] - n + offsets]
        if not np.isfinite(sample).all():
            raise ValueError("conformal window contains non-finite residuals")
        ordered = np.sort(sample, axis=1)
        lower[start : start + len(pos)] = ordered[:, lo_at]
        upper[start : start + len(pos)] = ordered[:, hi_at]
    return lower, upper


def lagged_scale(residuals: pd.Series, window: int = SCALE_WINDOW, floor: float = SCALE_FLOOR) -> pd.Series:
    """Scale known at the origin: median absolute residual over the previous day."""
    sigma = residuals.abs().rolling(window, min_periods=window).median().shift(1)
    return sigma.clip(lower=floor)


def _aligned(residuals: pd.Series, yhat: pd.Series, y: pd.Series) -> pd.DataFrame:
    aligned = pd.concat({"residual": residuals, "yhat": yhat, "y": y}, axis=1).dropna()
    if aligned.empty:
        raise ValueError("no finite calibration rows")
    return aligned.sort_index()


def minimum_history(method: str, candidates: dict[str, int], scale_window: int = SCALE_WINDOW) -> int:
    """Residuals required before the first origin this method can score."""
    longest = max(candidates.values())
    if method == "absolute":
        return int(longest)
    if method == "normalized":
        return int(longest) + int(scale_window)
    raise ValueError(f"unknown conformal method: {method}")


def _mark_selected(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    eligible = table.loc[table["eligible"]]
    if eligible.empty:
        chosen = table.sort_values(["coverage_90", "window_steps"], ascending=[False, False]).iloc[0]
    else:
        chosen = eligible.sort_values(["mean_width_90", "window_steps"], ascending=[True, True]).iloc[0]
    table["selected"] = table["window"].eq(chosen["window"])
    return table


def select_window(
    residuals: pd.Series,
    yhat: pd.Series,
    y: pd.Series,
    method: str,
    candidates: dict[str, int] | None = None,
    min_history: int | None = None,
    scale_window: int = SCALE_WINDOW,
) -> pd.DataFrame:
    """Choose the window on calibration only.

    Every candidate is scored on the same late-calibration origins. Those origins
    have a full window under the longest candidate. The normalized score also
    needs a finite lagged scale on every residual inside that window, which takes
    one extra day. Pass the same `min_history` to absolute and normalized so the
    two tables share origins. Nothing in this function reads the test period.
    """
    candidates = dict(candidates or WINDOW_CANDIDATES)
    aligned = _aligned(residuals, yhat, y)
    sigma = lagged_scale(aligned["residual"], window=scale_window)
    values = aligned["residual"].to_numpy(dtype=float)
    sigma_values = sigma.to_numpy(dtype=float)
    own_start = minimum_history(method, candidates, scale_window)
    start = own_start if min_history is None else int(min_history)
    if start < own_start:
        raise ValueError("min_history is shorter than this method can score")
    if len(aligned) <= start:
        raise ValueError("calibration series is shorter than the longest conformal window")
    if method == "normalized":
        lookback = max(candidates.values())
        if not np.isfinite(sigma_values[start - lookback :]).all():
            raise ValueError("normalized conformal window contains a missing scale")
    positions = np.arange(start, len(aligned))
    y_eval = aligned["y"].to_numpy()[positions]
    yhat_eval = aligned["yhat"].to_numpy()[positions]
    rows = []
    for name, window in candidates.items():
        if method == "normalized":
            sample = values / sigma_values
            scale = sigma_values[positions]
        else:
            sample = values
            scale = None
        q_lo, q_hi = sliding_residual_quantiles(sample, positions, window, SELECTION_ALPHA)
        q_lo_95, q_hi_95 = sliding_residual_quantiles(sample, positions, window, 0.05)
        if scale is None:
            lower = yhat_eval + q_lo
            upper = yhat_eval + q_hi
            lower_95 = yhat_eval + q_lo_95
            upper_95 = yhat_eval + q_hi_95
        else:
            lower = yhat_eval + scale * q_lo
            upper = yhat_eval + scale * q_hi
            lower_95 = yhat_eval + scale * q_lo_95
            upper_95 = yhat_eval + scale * q_hi_95
        cover_90 = np.mean((y_eval >= lower) & (y_eval <= upper))
        cover_95 = np.mean((y_eval >= lower_95) & (y_eval <= upper_95))
        width_90 = np.mean(upper - lower)
        rows.append(
            {
                "method": method,
                "window": name,
                "window_steps": int(window),
                "min_history": int(start),
                "n_selection": int(len(positions)),
                "coverage_90": float(cover_90),
                "coverage_95": float(cover_95),
                "mean_width_90": float(width_90),
                "eligible": bool(cover_90 >= COVERAGE_FLOOR_90 and cover_95 >= COVERAGE_FLOOR_95),
            }
        )
    return _mark_selected(pd.DataFrame(rows))


def selection_table(
    residuals: pd.Series,
    yhat: pd.Series,
    y: pd.Series,
    methods: tuple[str, ...] = ("absolute", "normalized"),
    candidates: dict[str, int] | None = None,
    scale_window: int = SCALE_WINDOW,
) -> pd.DataFrame:
    """Score every method on one shared set of calibration origins."""
    candidates = dict(candidates or WINDOW_CANDIDATES)
    start = max(minimum_history(method, candidates, scale_window) for method in methods)
    frames = [
        select_window(
            residuals,
            yhat,
            y,
            method,
            candidates=candidates,
            min_history=start,
            scale_window=scale_window,
        )
        for method in methods
    ]
    return pd.concat(frames, ignore_index=True)


def choose_primary(selection: pd.DataFrame) -> pd.Series:
    """Pick the frozen method and window from rows already marked selected.

    Eligible selected rows win. Among them the narrowest mean 90% interval is
    kept, then the shorter window. If none are eligible, the same rule uses
    every selected row.
    """
    chosen = selection.loc[selection["selected"]]
    if chosen.empty:
        raise ValueError("selection table has no selected window")
    eligible = chosen.loc[chosen["eligible"].astype(bool)]
    pool = eligible if not eligible.empty else chosen
    return pool.sort_values(["mean_width_90", "window_steps"], ascending=[True, True]).iloc[0]


def apply_sliding(
    history: pd.Series,
    yhat: pd.Series,
    window: int,
    alpha: float,
    method: str,
) -> pd.DataFrame:
    """Intervals at `yhat.index`, using only residuals strictly before each origin.

    `history` must be the out-of-sample residual series up to and including the
    forecast timestamps. The residual at the forecast timestamp is ignored.
    """
    combined = history.copy().sort_index()
    if method == "normalized":
        sigma = lagged_scale(combined)
        scored = combined / sigma
    elif method == "absolute":
        sigma = None
        scored = combined
    else:
        raise ValueError(f"unknown conformal method: {method}")
    positions = combined.index.get_indexer(yhat.index)
    if np.any(positions < 0):
        raise ValueError("forecast timestamps are missing from the residual history")
    if np.any(positions < window):
        raise ValueError("not enough residual history for the requested window")
    q_lo, q_hi = sliding_residual_quantiles(scored.to_numpy(dtype=float), positions, window, alpha)
    yhat_values = yhat.to_numpy(dtype=float)
    if method == "normalized":
        scale = sigma.iloc[positions].to_numpy(dtype=float)
        if not np.isfinite(scale).all():
            raise ValueError("normalized scale is missing at a forecast origin")
        lower = yhat_values + scale * q_lo
        upper = yhat_values + scale * q_hi
    else:
        lower = yhat_values + q_lo
        upper = yhat_values + q_hi
    return pd.DataFrame({"lower": lower, "upper": upper, "yhat": yhat_values}, index=yhat.index)


def sliding_crps(
    history: pd.Series,
    yhat: pd.Series,
    y: pd.Series,
    window: int,
    method: str,
) -> pd.Series:
    """Per-origin CRPS of the empirical sliding residual distribution."""
    from scripts.metrics import crps_from_residual_samples

    combined = history.sort_index()
    if method == "normalized":
        sigma = lagged_scale(combined)
        scored = (combined / sigma).to_numpy(dtype=float)
        scale = sigma.to_numpy(dtype=float)
    elif method == "absolute":
        scored = combined.to_numpy(dtype=float)
        scale = None
    else:
        raise ValueError(f"unknown conformal method: {method}")
    positions = combined.index.get_indexer(yhat.index)
    if np.any(positions < 0):
        raise ValueError("forecast timestamps are missing from the residual history")
    if np.any(positions < window):
        raise ValueError("not enough residual history for the requested window")
    y_values = y.reindex(yhat.index).to_numpy(dtype=float)
    yhat_values = yhat.to_numpy(dtype=float)
    out = np.empty(len(positions))
    offsets = np.arange(window)
    for start in range(0, len(positions), 512):
        pos = positions[start : start + 512]
        samples = scored[pos[:, None] - window + offsets]
        if scale is not None:
            samples = samples * scale[pos][:, None]
        if not np.isfinite(samples).all():
            raise ValueError("conformal window contains non-finite residuals")
        out[start : start + len(pos)] = crps_from_residual_samples(
            samples,
            y_values[start : start + len(pos)],
            yhat_values[start : start + len(pos)],
        )
    return pd.Series(out, index=yhat.index, name="crps")


def mean_sliding_crps(
    history: pd.Series,
    yhat: pd.Series,
    y: pd.Series,
    window: int,
    method: str,
) -> float:
    """Mean CRPS of the empirical sliding residual distribution on `yhat.index`."""
    return float(sliding_crps(history, yhat, y, window, method).mean())


def apply_fixed(cal_residuals: np.ndarray, yhat: pd.Series, alpha: float) -> pd.DataFrame:
    q_lo, q_hi = fixed_split_quantiles(cal_residuals, alpha)
    yhat_values = yhat.to_numpy(dtype=float)
    return pd.DataFrame(
        {"lower": yhat_values + q_lo, "upper": yhat_values + q_hi, "yhat": yhat_values},
        index=yhat.index,
    )
