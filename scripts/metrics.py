"""Point and probabilistic scores for the frozen test period."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_scores(y, yhat) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    err = y - yhat
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
    }


def skill_mae(mae: float, mae_reference: float) -> float:
    if mae_reference == 0:
        return np.nan
    return float(1.0 - mae / mae_reference)


def pinball_values(y, quantile, level: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    quantile = np.asarray(quantile, dtype=float)
    error = y - quantile
    return np.maximum(level * error, (level - 1.0) * error)


def pinball_loss(y, quantile, level: float) -> float:
    return float(np.mean(pinball_values(y, quantile, level)))


def interval_group_scores(y, lower, upper, group, alpha: float, crps=None) -> pd.DataFrame:
    """Coverage, width, pinball and optional CRPS averaged within each group."""
    frame = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=float),
            "lower": np.asarray(lower, dtype=float),
            "upper": np.asarray(upper, dtype=float),
            "group": pd.Series(group).reset_index(drop=True).to_numpy(),
        }
    )
    frame["covered"] = (frame["y"] >= frame["lower"]) & (frame["y"] <= frame["upper"])
    frame["width"] = frame["upper"] - frame["lower"]
    frame["pinball_lower"] = pinball_values(frame["y"], frame["lower"], alpha / 2.0)
    frame["pinball_upper"] = pinball_values(frame["y"], frame["upper"], 1.0 - alpha / 2.0)
    aggregations = {
        "n": ("covered", "size"),
        "coverage": ("covered", "mean"),
        "mean_width": ("width", "mean"),
        "median_width": ("width", "median"),
        "pinball_lower": ("pinball_lower", "mean"),
        "pinball_upper": ("pinball_upper", "mean"),
    }
    if crps is not None:
        frame["crps"] = np.asarray(crps, dtype=float)
        aggregations["crps"] = ("crps", "mean")
    out = frame.groupby("group", sort=True).agg(**aggregations)
    return out.reset_index(names="group")


def interval_scores(y, lower, upper) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    y, lower, upper = y[ok], lower[ok], upper[ok]
    width = upper - lower
    return {
        "n": int(len(y)),
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }


def crps_from_residual_samples(residual_samples: np.ndarray, y, yhat) -> np.ndarray:
    """Exact CRPS of the empirical distribution yhat + residual sample."""
    samples = np.sort(np.asarray(residual_samples, dtype=float), axis=1)
    n, width = samples.shape
    ranks = np.arange(1, width + 1)
    pair = np.sum((2 * ranks - width - 1) * samples, axis=1) / (width ** 2)
    target = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    absolute = np.mean(np.abs(samples - target[:, None]), axis=1)
    return absolute - pair


def crps_from_quantiles(y, quantiles, levels) -> np.ndarray:
    """Trapezoidal integral of the quantile score between the first and last level.

    CRPS equals the integral from 0 to 1 of twice the pinball loss. This returns
    that integral only on the span of `levels`, which must be strictly increasing
    and inside (0, 1). Tails outside the first and last level are not included.
    """
    y = np.asarray(y, dtype=float)
    grid = np.asarray(quantiles, dtype=float)
    tau = np.asarray(levels, dtype=float)
    if grid.ndim != 2:
        raise ValueError("quantiles must be a row per origin")
    if grid.shape[0] != len(y) or grid.shape[1] != len(tau):
        raise ValueError("quantile grid does not match y and levels")
    if len(tau) < 2 or np.any(np.diff(tau) <= 0) or np.any(tau <= 0) or np.any(tau >= 1):
        raise ValueError("levels must be strictly increasing and inside (0, 1)")
    error = y[:, None] - grid
    pinball = np.maximum(tau * error, (tau - 1.0) * error)
    score = 2.0 * pinball
    pieces = 0.5 * (score[:, :-1] + score[:, 1:]) * np.diff(tau)
    return pieces.sum(axis=1)


def block_bootstrap_mae_diff(y, yhat_a, yhat_b, block: int = 288, n_boot: int = 400, seed: int = 42) -> dict[str, float]:
    """Moving-block bootstrap interval for MAE(a) - MAE(b). Negative means a is better."""
    y = np.asarray(y, dtype=float)
    abs_a = np.abs(y - np.asarray(yhat_a, dtype=float))
    abs_b = np.abs(y - np.asarray(yhat_b, dtype=float))
    diff = abs_a - abs_b
    n = len(diff)
    if n < block * 2:
        raise ValueError("series is shorter than two blocks")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([diff[s : s + block] for s in chosen])[:n]
        draws[i] = sample.mean()
    return {
        "mae_diff": float(diff.mean()),
        "boot_p05": float(np.quantile(draws, 0.05)),
        "boot_p95": float(np.quantile(draws, 0.95)),
        "block": int(block),
        "n_boot": int(n_boot),
    }


def monthly_mae(index: pd.DatetimeIndex, y, yhat) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "yhat": np.asarray(yhat, dtype=float)}, index=index)
    frame["ae"] = (frame["y"] - frame["yhat"]).abs()
    out = frame.groupby(frame.index.to_period("M"))["ae"].agg(["mean", "count"])
    out.index = out.index.astype(str)
    out = out.rename(columns={"mean": "mae", "count": "n"})
    return out.reset_index(names="month")
