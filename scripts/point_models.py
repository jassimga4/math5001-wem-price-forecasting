"""Point forecasts fit only on the training window.

Hyperparameters are chosen on an inner chronological slice of training
(through 2025-06-30 fit, July-September 2025 validation). Calibration and test
are not read here.

LightGBM is trained under MAE on the 5-minute price change. The issued
forecast is the lagged price plus that change, so persistence is the
zero-change forecast. MAE is the selection metric, and the price has very
heavy tails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from scripts.forecast_design import INNER_VAL_START, TREE_CATEGORICAL

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
LGBM_GRID = [
    {"learning_rate": learning_rate, "num_leaves": num_leaves, "min_child_samples": min_child}
    for learning_rate in (0.05, 0.1)
    for num_leaves in (31, 63)
    for min_child in (100, 400)
]


def persistence(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(float)


def inner_fit_val(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = train.loc[train.index < INNER_VAL_START]
    val = train.loc[train.index >= INNER_VAL_START]
    if fit.empty or val.empty:
        raise ValueError("inner training split is empty; check INNER_VAL_START")
    return fit, val


def select_ridge(train: pd.DataFrame, features: list[str], target: str) -> dict:
    fit, val = inner_fit_val(train)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit[features])
    x_val = scaler.transform(val[features])
    y_fit = fit[target].to_numpy(dtype=float)
    y_val = val[target].to_numpy(dtype=float)
    records = []
    best_alpha = None
    best_mae = np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_fit, y_fit)
        mae = float(mean_absolute_error(y_val, model.predict(x_val)))
        records.append({"alpha": alpha, "inner_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    full_scaler = StandardScaler()
    x_train = full_scaler.fit_transform(train[features])
    model = Ridge(alpha=best_alpha)
    model.fit(x_train, train[target].to_numpy(dtype=float))
    return {
        "model": model,
        "scaler": full_scaler,
        "alpha": best_alpha,
        "inner_mae": best_mae,
        "grid": pd.DataFrame(records),
    }


def predict_ridge(bundle: dict, frame: pd.DataFrame, features: list[str]) -> pd.Series:
    x = bundle["scaler"].transform(frame[features])
    pred = bundle["model"].predict(x)
    return pd.Series(pred, index=frame.index, name="yhat")


def _as_categories(fit: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = fit.copy()
    other = other.copy()
    for col in TREE_CATEGORICAL:
        fit[col] = fit[col].astype(int).astype("category")
        categories = fit[col].cat.categories
        other[col] = pd.Categorical(other[col].astype(int), categories=categories)
    return fit, other


def _lgbm(params: dict, n_estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
        **params,
    )


def _price_change(frame: pd.DataFrame, target: str, lag_column: str) -> np.ndarray:
    return frame[target].to_numpy(dtype=float) - frame[lag_column].to_numpy(dtype=float)


def select_lgbm(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    grid: list[dict] | None = None,
    n_estimators: int = 250,
    lag_column: str = "mcp_lag_5min",
) -> dict:
    """Select an MAE tree for the 5-minute price change, then refit on all training rows."""
    if lag_column not in train.columns:
        raise ValueError(f"{lag_column} is required to fit the price change")
    grid = LGBM_GRID if grid is None else grid
    fit, val = inner_fit_val(train)
    fit_x, val_x = _as_categories(fit[features], val[features])
    y_change = _price_change(fit, target, lag_column)
    y_val = val[target].to_numpy(dtype=float)
    lag_val = val[lag_column].to_numpy(dtype=float)
    records = []
    best = None
    best_mae = np.inf
    for params in grid:
        print("lightgbm inner fit", params, flush=True)
        model = _lgbm(params, n_estimators)
        model.fit(fit_x, y_change, categorical_feature=TREE_CATEGORICAL)
        mae = float(mean_absolute_error(y_val, lag_val + model.predict(val_x)))
        row = {**params, "inner_mae": mae}
        records.append(row)
        if mae < best_mae:
            best_mae = mae
            best = params
    full_x = train[features].copy()
    for col in TREE_CATEGORICAL:
        full_x[col] = full_x[col].astype(int).astype("category")
    model = _lgbm(best, n_estimators)
    model.fit(full_x, _price_change(train, target, lag_column), categorical_feature=TREE_CATEGORICAL)
    persistence_inner_mae = float(mean_absolute_error(y_val, lag_val))
    return {
        "model": model,
        "params": best,
        "inner_mae": best_mae,
        "persistence_inner_mae": persistence_inner_mae,
        "objective": "regression_l1",
        "target": "mcp_minus_lag_5min",
        "lag_column": lag_column,
        "grid": pd.DataFrame(records).sort_values("inner_mae"),
        "categories": {col: list(full_x[col].cat.categories) for col in TREE_CATEGORICAL},
    }


def predict_lgbm(bundle: dict, frame: pd.DataFrame, features: list[str]) -> pd.Series:
    x = frame[features].copy()
    for col, categories in bundle["categories"].items():
        x[col] = pd.Categorical(x[col].astype(int), categories=categories)
    lag_column = bundle["lag_column"]
    pred = frame[lag_column].to_numpy(dtype=float) + bundle["model"].predict(x)
    return pd.Series(pred, index=frame.index, name="yhat")
