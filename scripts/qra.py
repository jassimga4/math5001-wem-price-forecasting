"""Quantile regression averaging of the existing ex-ante point forecasts.

This module implements the original QRA comparison used on `main`: one linear
quantile regression of MCP on a pool of point forecasts (Nowotarski and Weron,
2015). By default the regressors are the 5-minute, 30-minute and 1-day
persistence forecasts, the ridge forecast, and the LightGBM forecast. Those
forecasts already use the 5-minute horizon and the ex-ante information set in
scripts/forecast_design.py. Quantile regression is fit on calibration origins
only. Test outcomes are used for scoring, not for the fit, and quantiles are
not shrunk to chase test coverage. Predicted quantiles are rearranged by
sorting each row so the quantile function is nondecreasing. That sort does not
use the outcome.

`fit_qra` accepts a `columns` argument so a subset of the same point forecasts
can be used. Recent calibration windows, QRM, quantile averaging, and a
native LightGBM quantile member are scored and selected in
scripts/qra_variants.py; that script never uses test outcomes for selection.

This is not a replacement for the absolute-residual sliding conformal intervals.
mapie is not used: it does not fit this regression.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.conformal import ALPHAS, apply_sliding, sliding_crps
from scripts.metrics import crps_from_quantiles, interval_group_scores, interval_scores, pinball_loss
from scripts.paths import PROJECT_ROOT

QRA_FORECASTS = (
    "persistence_5min",
    "persistence_30min",
    "persistence_1d",
    "ridge",
    "lightgbm",
)
OUT = PROJECT_ROOT / "reports" / "forecast"
COEF_INTERCEPT = "intercept"


def comparison_levels(alphas: tuple[float, ...] = ALPHAS) -> np.ndarray:
    """Interval endpoints used by the conformal tables, plus a 0.05 grid and two tails."""
    levels = {0.01, 0.99}
    for alpha in alphas:
        levels.add(float(alpha) / 2.0)
        levels.add(1.0 - float(alpha) / 2.0)
    for step in range(1, 20):
        levels.add(step / 20.0)
    return np.array(sorted(levels), dtype=float)


def level_name(level: float) -> str:
    return f"{float(level):.3f}"


def _design(forecasts: pd.DataFrame, columns: tuple[str, ...] = QRA_FORECASTS) -> np.ndarray:
    missing = [name for name in columns if name not in forecasts.columns]
    if missing:
        raise ValueError(f"QRA forecasts missing: {missing}")
    values = forecasts.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("QRA design contains a non-finite forecast")
    return sm.add_constant(values, has_constant="add")


def fit_qra(
    forecasts: pd.DataFrame,
    y: pd.Series,
    levels: np.ndarray | None = None,
    max_iter: int = 2000,
    columns: tuple[str, ...] = QRA_FORECASTS,
) -> pd.DataFrame:
    """Fit one linear quantile regression per level. `y` must be the calibration target.

    `columns` names the point forecasts used as regressors (Nowotarski and Weron
    2015). The default is the five forecasts of the original comparison.
    """
    columns = tuple(columns)
    if not columns:
        raise ValueError("QRA needs at least one forecast column")
    levels = comparison_levels() if levels is None else np.asarray(levels, dtype=float)
    if len(levels) < 2 or np.any(np.diff(levels) <= 0):
        raise ValueError("QRA levels must be strictly increasing")
    aligned = forecasts.loc[:, list(columns)].copy()
    aligned["y"] = y.reindex(forecasts.index)
    aligned = aligned.dropna()
    if len(aligned) <= len(columns) + 1:
        raise ValueError("calibration sample is shorter than the QRA design")
    design = _design(aligned, columns)
    target = aligned["y"].to_numpy(dtype=float)
    rows = []
    for level in levels:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterationLimitWarning)
            fitted = QuantReg(target, design).fit(q=float(level), max_iter=max_iter)
        params = np.asarray(fitted.params, dtype=float)
        if params.shape != (design.shape[1],) or not np.isfinite(params).all():
            raise ValueError(f"QRA failed to return finite coefficients at level {level}")
        hit_limit = any(issubclass(item.category, IterationLimitWarning) for item in caught)
        row = {
            "level": float(level),
            "converged": not hit_limit,
            COEF_INTERCEPT: float(params[0]),
        }
        for name, value in zip(columns, params[1:]):
            row[name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def coefficient_columns(coefficients: pd.DataFrame) -> tuple[str, ...]:
    """Forecast columns of a coefficient table, in the order they were fit."""
    skip = {"level", "converged", COEF_INTERCEPT}
    return tuple(name for name in coefficients.columns if name not in skip)


def _coefficient_matrix(coefficients: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    levels = coefficients["level"].to_numpy(dtype=float)
    order = np.argsort(levels)
    levels = levels[order]
    forecast_columns = coefficient_columns(coefficients)
    columns = [COEF_INTERCEPT, *forecast_columns]
    matrix = coefficients.loc[:, columns].to_numpy(dtype=float)[order]
    return levels, matrix, forecast_columns


def predict_qra(forecasts: pd.DataFrame, coefficients: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Return rearranged quantiles and the share of rows that crossed before the sort."""
    levels, matrix, forecast_columns = _coefficient_matrix(coefficients)
    raw = _design(forecasts, forecast_columns) @ matrix.T
    if raw.shape[1] > 1:
        crossed = np.any(np.diff(raw, axis=1) < 0, axis=1)
        share = float(np.mean(crossed))
    else:
        share = 0.0
    ordered = np.sort(raw, axis=1)
    frame = pd.DataFrame(ordered, index=forecasts.index, columns=[level_name(level) for level in levels])
    return frame, share


def empirical_quantile_levels(samples: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Type-1 empirical quantiles of samples sorted along axis 1."""
    ordered = np.asarray(samples, dtype=float)
    if ordered.ndim != 2:
        raise ValueError("samples must have one row per origin")
    width = ordered.shape[1]
    ranks = np.ceil(np.asarray(levels, dtype=float) * width).astype(int) - 1
    ranks = np.clip(ranks, 0, width - 1)
    return ordered[:, ranks]


def conformal_quantile_grid(
    history: pd.Series,
    yhat: pd.Series,
    window: int,
    method: str,
    levels: np.ndarray,
) -> pd.DataFrame:
    """Type-1 quantiles of the same residual window the frozen conformal interval uses.

    The origin is excluded. Samples are not retained after each chunk is reduced
    to the requested levels.
    """
    combined = history.sort_index()
    if method == "normalized":
        from scripts.conformal import lagged_scale

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
    pieces = []
    offsets = np.arange(window)
    for start in range(0, len(positions), 512):
        pos = positions[start : start + 512]
        samples = scored[pos[:, None] - window + offsets]
        if scale is not None:
            samples = samples * scale[pos][:, None]
        if not np.isfinite(samples).all():
            raise ValueError("conformal window contains non-finite residuals")
        pieces.append(empirical_quantile_levels(np.sort(samples, axis=1), levels))
    grid = np.vstack(pieces)
    return pd.DataFrame(grid, index=yhat.index, columns=[level_name(level) for level in levels])


def _residual_history(y_cal: pd.Series, yhat_cal: pd.Series, y_test: pd.Series, yhat_test: pd.Series) -> pd.Series:
    return (pd.concat([y_cal, y_test]) - pd.concat([yhat_cal, yhat_test])).sort_index()



def _group_integral(table: pd.DataFrame, group_col: str, integral: np.ndarray, groups) -> pd.DataFrame:
    frame = pd.DataFrame({"group": np.asarray(groups), "value": np.asarray(integral, dtype=float)})
    means = frame.groupby("group", sort=False)["value"].mean()
    table = table.copy()
    table["crps_quantile_integral"] = table[group_col].map(means)
    return table


def _score_interval(y: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> dict:
    scores = interval_scores(y, lower, upper)
    scores["nominal"] = 1.0 - alpha
    scores["pinball_lower"] = pinball_loss(y, lower, alpha / 2.0)
    scores["pinball_upper"] = pinball_loss(y, upper, 1.0 - alpha / 2.0)
    return scores


def _quantile_frame(forecasts: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({name: forecasts[name].reindex(y.index) for name in QRA_FORECASTS})


def run_qra_comparison(
    result: dict,
    primary_method: str,
    primary_window: int,
    frozen_source: str,
    lightgbm_source: str,
) -> dict:
    """Score QRA and the already chosen conformal method on the same test origins."""
    calibration = result["calibration"]
    test = result["test"]
    predictions = result["predictions"]
    from scripts.forecast_design import TARGET

    y_cal = calibration[TARGET]
    y_test = test[TARGET]
    cal_forecasts = _quantile_frame(predictions, y_cal)
    test_forecasts = _quantile_frame(predictions, y_test)
    if not np.isfinite(test_forecasts.to_numpy(dtype=float)).all():
        raise ValueError("test point forecasts are not all finite")
    levels = comparison_levels()
    coefficients = fit_qra(cal_forecasts, y_cal, levels)
    predicted, crossed_share = predict_qra(test_forecasts, coefficients)
    qra_integral = crps_from_quantiles(
        y_test.to_numpy(dtype=float),
        predicted.to_numpy(dtype=float),
        levels,
    )

    test_rows = []
    month_rows = []
    regime_rows = []
    crps_rows = [
        {
            "method": "qra",
            "base": "qra",
            "crps_empirical": np.nan,
            "crps_quantile_integral": float(np.mean(qra_integral)),
        }
    ]
    tail = (y_test >= result["spike_cut"]) | (y_test <= result["floor_cut"])
    labels = np.where(tail.to_numpy(), "tail", "body")
    months = test.index.to_period("M").astype(str)

    for alpha in ALPHAS:
        lower = predicted[level_name(alpha / 2.0)]
        upper = predicted[level_name(1.0 - alpha / 2.0)]
        scores = _score_interval(y_test, lower, upper, alpha)
        scores.update({"base": "qra", "method": "qra", "window_steps": np.nan, "alpha": alpha, "role": "qra"})
        test_rows.append(scores)
    regime = interval_group_scores(
        y_test,
        predicted[level_name(0.05)],
        predicted[level_name(0.95)],
        labels,
        0.10,
    ).rename(columns={"group": "regime"})
    regime = _group_integral(regime, "regime", qra_integral, labels)
    overall = _score_interval(y_test, predicted[level_name(0.05)], predicted[level_name(0.95)], 0.10)
    overall.update({"regime": "all", "crps_quantile_integral": float(np.mean(qra_integral))})
    regime = pd.concat([pd.DataFrame([overall]), regime], ignore_index=True)
    regime["base"] = "qra"
    regime["method"] = "qra"
    regime["nominal"] = 0.90
    regime_rows.append(regime)
    by_month = interval_group_scores(
        y_test,
        predicted[level_name(0.05)],
        predicted[level_name(0.95)],
        months,
        0.10,
    ).rename(columns={"group": "month"})
    by_month = _group_integral(by_month, "month", qra_integral, months)
    by_month["base"] = "qra"
    by_month["method"] = "qra"
    by_month["nominal"] = 0.90
    month_rows.append(by_month)

    for base in ("lightgbm", "persistence_5min"):
        yhat_cal = predictions[base].loc[calibration.index]
        yhat_test = predictions[base].loc[test.index]
        history = _residual_history(y_cal, yhat_cal, y_test, yhat_test)
        grid = conformal_quantile_grid(history, yhat_test, int(primary_window), primary_method, levels)
        grid = grid.add(yhat_test.to_numpy(dtype=float), axis=0)
        integral = crps_from_quantiles(y_test.to_numpy(dtype=float), grid.to_numpy(dtype=float), levels)
        empirical = sliding_crps(history, yhat_test, y_test, int(primary_window), primary_method)
        crps_rows.append(
            {
                "method": primary_method,
                "base": base,
                "crps_empirical": float(empirical.mean()),
                "crps_quantile_integral": float(np.mean(integral)),
            }
        )
        for alpha in ALPHAS:
            interval = apply_sliding(history, yhat_test, int(primary_window), alpha, primary_method)
            scores = _score_interval(y_test, interval["lower"], interval["upper"], alpha)
            scores.update(
                {
                    "base": base,
                    "method": primary_method,
                    "window_steps": int(primary_window),
                    "alpha": alpha,
                    "role": "frozen_conformal",
                }
            )
            test_rows.append(scores)
        frozen = apply_sliding(history, yhat_test, int(primary_window), 0.10, primary_method)
        regime = interval_group_scores(
            y_test, frozen["lower"], frozen["upper"], labels, 0.10, empirical
        ).rename(columns={"group": "regime"})
        regime = _group_integral(regime, "regime", integral, labels)
        overall = _score_interval(y_test, frozen["lower"], frozen["upper"], 0.10)
        overall.update(
            {
                "regime": "all",
                "crps": float(empirical.mean()),
                "crps_quantile_integral": float(np.mean(integral)),
            }
        )
        regime = pd.concat([pd.DataFrame([overall]), regime], ignore_index=True)
        regime["base"] = base
        regime["method"] = primary_method
        regime["nominal"] = 0.90
        regime_rows.append(regime)
        by_month = interval_group_scores(
            y_test, frozen["lower"], frozen["upper"], months, 0.10, empirical
        ).rename(columns={"group": "month"})
        by_month = _group_integral(by_month, "month", integral, months)
        by_month["base"] = base
        by_month["method"] = primary_method
        by_month["nominal"] = 0.90
        month_rows.append(by_month)

    limited = coefficients.loc[~coefficients["converged"].astype(bool), "level"].tolist()
    meta = {
        "method": "quantile regression averaging",
        "fit": "statsmodels QuantReg linear quantile regression with intercept, calibration origins only",
        "rearrangement": "each test row's predicted quantiles are sorted; the sort does not use the outcome",
        "forecasts": list(QRA_FORECASTS),
        "horizon": "5min",
        "levels": [float(level) for level in levels],
        "primary_conformal_method": primary_method,
        "primary_conformal_window_steps": int(primary_window),
        "frozen_source": frozen_source,
        "lightgbm_source": lightgbm_source,
        "n_calibration_fit": int((np.isfinite(cal_forecasts.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y_cal.to_numpy(dtype=float))).sum()),
        "n_test": int(len(test)),
        "share_test_rows_crossed_before_rearrangement": crossed_share,
        "levels_hit_iteration_limit": [float(level) for level in limited],
        "spike_cut_train_p95": float(result["spike_cut"]),
        "floor_cut_train_p05": float(result["floor_cut"]),
        "crps_empirical": (
            "Mean of scripts.metrics.crps_from_residual_samples on the sliding residual window. "
            "Not defined for QRA, so that column is blank on the QRA row."
        ),
        "crps_quantile_integral": (
            "Trapezoidal integral of twice the pinball loss between the lowest and highest fitted "
            "level. QRA uses the rearranged regression quantiles. The frozen conformal rows use "
            "the point forecast plus type-1 quantiles of the residual window. This is not the "
            "empirical-sample CRPS."
        ),
        "interval_note": (
            "Frozen conformal intervals use the split-conformal floor and ceil ranks in "
            "scripts/conformal.py. QRA intervals are the rearranged regression quantiles at "
            "alpha/2 and 1-alpha/2."
        ),
    }
    if "ridge_alpha" in result:
        meta["ridge_alpha"] = result["ridge_alpha"]
    return {
        "coefficients": coefficients,
        "test": pd.DataFrame(test_rows),
        "by_month": pd.concat(month_rows, ignore_index=True),
        "regime": pd.concat(regime_rows, ignore_index=True),
        "crps": pd.DataFrame(crps_rows),
        "meta": meta,
    }


def write_qra_tables(qra: dict, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    qra["coefficients"].to_csv(out / "qra_coefficients.csv", index=False)
    qra["test"].to_csv(out / "qra_test.csv", index=False)
    qra["by_month"].to_csv(out / "qra_by_month.csv", index=False)
    qra["regime"].to_csv(out / "qra_regime.csv", index=False)
    qra["crps"].to_csv(out / "qra_crps.csv", index=False)
    (out / "qra_meta.json").write_text(json.dumps(qra["meta"], indent=2), encoding="utf-8")


def _saved_point_result() -> tuple[dict, str, int]:
    """Reuse the committed LightGBM model and the frozen conformal choice.

    Ridge is refit on the training window. Persistence is recomputed from lags.
    The conformal window is not chosen again.
    """
    import joblib

    from scripts.experiment import MODEL_PATH, load_ready
    from scripts.forecast_design import RIDGE_FEATURES, TARGET, slice_split
    from scripts.point_models import persistence, predict_lgbm, predict_ridge, select_ridge

    meta_path = OUT / "experiment_meta.json"
    committed = json.loads(meta_path.read_text(encoding="utf-8"))
    ready = load_ready()
    train = slice_split(ready, "train")
    calibration = slice_split(ready, "calibration")
    test = slice_split(ready, "test")
    saved = joblib.load(MODEL_PATH)
    ridge = select_ridge(train, RIDGE_FEATURES, TARGET)
    held = pd.concat([calibration, test]).sort_index()
    predictions = {
        "persistence_5min": persistence(held, "mcp_lag_5min"),
        "persistence_30min": persistence(held, "mcp_lag_30min"),
        "persistence_1d": persistence(held, "mcp_lag_1d"),
        "ridge": predict_ridge(ridge, held, RIDGE_FEATURES),
        "lightgbm": predict_lgbm(saved["bundle"], held, saved["features"]),
    }
    point = {
        "predictions": predictions,
        "calibration": calibration,
        "test": test,
        "spike_cut": float(train[TARGET].quantile(0.95)),
        "floor_cut": float(train[TARGET].quantile(0.05)),
        "ridge_alpha": ridge["alpha"],
    }
    return (
        point,
        str(committed["conformal_primary_method"]),
        int(committed["conformal_primary_window_steps"]),
    )


def main() -> None:
    print("loading saved point forecasts", flush=True)
    point, method, window = _saved_point_result()
    print("fitting qra on calibration", method, window, flush=True)
    qra = run_qra_comparison(
        point,
        primary_method=method,
        primary_window=window,
        frozen_source="reports/forecast/experiment_meta.json; window was not chosen again",
        lightgbm_source="models/lgbm_5min_ahead.joblib; not refit",
    )
    write_qra_tables(qra)
    print(qra["test"].to_string(index=False))
    print(qra["crps"].to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
