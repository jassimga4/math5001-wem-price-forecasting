"""Run the frozen 5-minute forecast experiment and write tables under reports/forecast."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.conformal import (
    ALPHAS,
    WINDOW_CANDIDATES,
    apply_fixed,
    apply_sliding,
    choose_primary,
    selection_table,
    sliding_crps,
)
from scripts.forecast_design import (
    RIDGE_FEATURES,
    TARGET,
    TREE_FEATURES,
    build_exante_frame,
    feature_availability,
    model_ready,
    slice_split,
)
from scripts.metrics import (
    block_bootstrap_mae_diff,
    interval_group_scores,
    interval_scores,
    monthly_mae,
    pinball_loss,
    point_scores,
    skill_mae,
)
from scripts.paths import PROJECT_ROOT, panel_path
from scripts.point_models import persistence, predict_lgbm, predict_ridge, select_lgbm, select_ridge

OUT = PROJECT_ROOT / "reports" / "forecast"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_5min_ahead.joblib"


def load_ready() -> pd.DataFrame:
    raw = pd.read_parquet(panel_path(), engine="pyarrow")
    frame = build_exante_frame(raw)
    features = sorted(set(RIDGE_FEATURES + TREE_FEATURES))
    return model_ready(frame, features)


def _point_table(ready: pd.DataFrame, predictions: dict[str, pd.Series], split_name: str) -> pd.DataFrame:
    block = slice_split(ready, split_name)
    y = block[TARGET]
    rows = []
    persistence_mae = point_scores(y, predictions["persistence_5min"].loc[block.index])["mae"]
    for name, pred in predictions.items():
        scores = point_scores(y, pred.loc[block.index])
        scores["model"] = name
        scores["split"] = split_name
        scores["skill_vs_persistence"] = skill_mae(scores["mae"], persistence_mae)
        rows.append(scores)
    return pd.DataFrame(rows)


def fit_and_score(ready: pd.DataFrame) -> dict:
    train = slice_split(ready, "train")
    calibration = slice_split(ready, "calibration")
    test = slice_split(ready, "test")
    if train.empty or calibration.empty or test.empty:
        raise RuntimeError("one of train, calibration, test is empty")
    if not (train.index.max() < calibration.index.min() <= calibration.index.max() < test.index.min()):
        raise RuntimeError("split timestamps overlap or are out of order")

    ridge = select_ridge(train, RIDGE_FEATURES, TARGET)
    lgbm = select_lgbm(train, TREE_FEATURES, TARGET)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bundle": lgbm, "features": TREE_FEATURES}, MODEL_PATH)

    def pack(block: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "persistence_5min": persistence(block, "mcp_lag_5min"),
            "persistence_30min": persistence(block, "mcp_lag_30min"),
            "persistence_1d": persistence(block, "mcp_lag_1d"),
            "ridge": predict_ridge(ridge, block, RIDGE_FEATURES),
            "lightgbm": predict_lgbm(lgbm, block, TREE_FEATURES),
        }

    held = pd.concat([calibration, test]).sort_index()
    predictions = pack(held)
    point = pd.concat(
        [
            _point_table(ready, predictions, "calibration"),
            _point_table(ready, predictions, "test"),
        ],
        ignore_index=True,
    )
    test_y = test[TARGET]
    test_lgb = predictions["lightgbm"].loc[test.index]
    test_pers = predictions["persistence_5min"].loc[test.index]
    bootstrap = block_bootstrap_mae_diff(test_y, test_lgb, test_pers)
    spike_cut = float(train[TARGET].quantile(0.95))
    floor_cut = float(train[TARGET].quantile(0.05))
    tail = (test_y >= spike_cut) | (test_y <= floor_cut)
    regime = pd.DataFrame(
        [
            {"regime": "all", **point_scores(test_y, test_lgb)},
            {"regime": "tail", **point_scores(test_y.loc[tail], test_lgb.loc[tail])},
            {"regime": "body", **point_scores(test_y.loc[~tail], test_lgb.loc[~tail])},
            {"regime": "tail_persistence", **point_scores(test_y.loc[tail], test_pers.loc[tail])},
            {"regime": "body_persistence", **point_scores(test_y.loc[~tail], test_pers.loc[~tail])},
        ]
    )
    by_month = monthly_mae(test.index, test_y, test_lgb)
    by_month_pers = monthly_mae(test.index, test_y, test_pers).rename(columns={"mae": "persistence_mae"})
    by_month = by_month.merge(by_month_pers[["month", "persistence_mae"]], on="month")
    return {
        "ridge": ridge,
        "lgbm": lgbm,
        "predictions": predictions,
        "point": point,
        "bootstrap": bootstrap,
        "regime": regime,
        "by_month": by_month,
        "spike_cut": spike_cut,
        "floor_cut": floor_cut,
        "train": train,
        "calibration": calibration,
        "test": test,
    }


def _quantile_frame(yhat: pd.Series, residual_history: pd.Series, window: int, method: str) -> pd.DataFrame:
    frames = []
    for alpha in ALPHAS:
        interval = apply_sliding(residual_history, yhat, window, alpha, method)
        level = 1.0 - alpha
        frames.append(
            pd.DataFrame(
                {
                    f"q{level:.2f}_lower": interval["lower"],
                    f"q{level:.2f}_upper": interval["upper"],
                }
            )
        )
    out = pd.concat(frames, axis=1)
    out["yhat"] = yhat
    return out


def _history(block_y: pd.Series, block_yhat: pd.Series, test_y: pd.Series, test_yhat: pd.Series) -> pd.Series:
    return (pd.concat([block_y, test_y]) - pd.concat([block_yhat, test_yhat])).sort_index()


def _score_interval(y: pd.Series, interval: pd.DataFrame, alpha: float) -> dict:
    scores = interval_scores(y, interval["lower"], interval["upper"])
    scores["nominal"] = 1.0 - alpha
    scores["pinball_lower"] = pinball_loss(y, interval["lower"], alpha / 2.0)
    scores["pinball_upper"] = pinball_loss(y, interval["upper"], 1.0 - alpha / 2.0)
    return scores


def run_conformal(result: dict) -> dict:
    calibration = result["calibration"]
    test = result["test"]
    predictions = result["predictions"]
    tables = []
    chosen = {}
    for base in ("lightgbm", "persistence_5min"):
        residual = calibration[TARGET] - predictions[base].loc[calibration.index]
        yhat = predictions[base].loc[calibration.index]
        table = selection_table(residual, yhat, calibration[TARGET])
        table.insert(0, "base", base)
        tables.append(table)
        for method in ("absolute", "normalized"):
            pick = table.loc[table["method"].eq(method) & table["selected"]].iloc[0]
            chosen[(base, method)] = pick
    selection = pd.concat(tables, ignore_index=True)
    primary = choose_primary(selection.loc[selection["base"].eq("lightgbm")])
    primary_method = str(primary["method"])
    primary_window = int(primary["window_steps"])

    test_rows = []
    month_rows = []
    regime_rows = []
    crps_rows = []
    tail = (test[TARGET] >= result["spike_cut"]) | (test[TARGET] <= result["floor_cut"])
    for base in ("lightgbm", "persistence_5min"):
        history = _history(
            calibration[TARGET],
            predictions[base].loc[calibration.index],
            test[TARGET],
            predictions[base].loc[test.index],
        )
        yhat_test = predictions[base].loc[test.index]
        has_frozen_row = False
        for method in ("absolute", "normalized"):
            window = int(chosen[(base, method)]["window_steps"])
            is_frozen = method == primary_method and window == primary_window
            has_frozen_row = has_frozen_row or is_frozen
            for alpha in ALPHAS:
                interval = apply_sliding(history, yhat_test, window, alpha, method)
                scores = _score_interval(test[TARGET], interval, alpha)
                frozen_level = is_frozen and abs(alpha - 0.10) < 1e-12
                scores.update(
                    {
                        "base": base,
                        "method": method,
                        "window_steps": window,
                        "alpha": alpha,
                        "role": "frozen" if frozen_level else "own_selection",
                    }
                )
                test_rows.append(scores)
        fixed = apply_fixed(
            (calibration[TARGET] - predictions[base].loc[calibration.index]).to_numpy(),
            yhat_test,
            0.10,
        )
        fixed_scores = _score_interval(test[TARGET], fixed, 0.10)
        fixed_scores.update(
            {
                "base": base,
                "method": "fixed_split",
                "window_steps": int(len(calibration)),
                "alpha": 0.10,
                "role": "exchangeable_benchmark",
            }
        )
        test_rows.append(fixed_scores)

        frozen = apply_sliding(history, yhat_test, primary_window, 0.10, primary_method)
        if not has_frozen_row:
            frozen_scores = _score_interval(test[TARGET], frozen, 0.10)
            frozen_scores.update(
                {
                    "base": base,
                    "method": primary_method,
                    "window_steps": primary_window,
                    "alpha": 0.10,
                    "role": "frozen",
                }
            )
            test_rows.append(frozen_scores)
        frozen_crps = sliding_crps(history, yhat_test, test[TARGET], primary_window, primary_method)
        crps_rows.append(
            {
                "base": base,
                "method": primary_method,
                "window_steps": primary_window,
                "crps": float(frozen_crps.mean()),
            }
        )
        months = interval_group_scores(
            test[TARGET],
            frozen["lower"],
            frozen["upper"],
            test.index.to_period("M").astype(str),
            0.10,
            frozen_crps,
        ).rename(columns={"group": "month"})
        months["base"] = base
        months["method"] = primary_method
        months["nominal"] = 0.90
        month_rows.append(months)
        regime = interval_group_scores(
            test[TARGET],
            frozen["lower"],
            frozen["upper"],
            np.where(tail.to_numpy(), "tail", "body"),
            0.10,
            frozen_crps,
        )
        overall = _score_interval(test[TARGET], frozen, 0.10)
        overall.update({"group": "all", "crps": float(frozen_crps.mean())})
        regime = pd.concat([pd.DataFrame([overall]), regime], ignore_index=True)
        regime["base"] = base
        regime["method"] = primary_method
        regime["nominal"] = 0.90
        regime_rows.append(regime.rename(columns={"group": "regime"}))

    return {
        "selection": selection,
        "test": pd.DataFrame(test_rows),
        "by_month": pd.concat(month_rows, ignore_index=True),
        "regime": pd.concat(regime_rows, ignore_index=True),
        "crps": pd.DataFrame(crps_rows),
        "primary_method": primary_method,
        "primary_window": primary_window,
    }


def save_results(ready: pd.DataFrame, point: dict, conformal: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature_availability().to_csv(OUT / "feature_availability.csv", index=False)
    point["point"].to_csv(OUT / "point_metrics.csv", index=False)
    point["regime"].to_csv(OUT / "point_regime.csv", index=False)
    point["by_month"].to_csv(OUT / "point_by_month.csv", index=False)
    point["ridge"]["grid"].to_csv(OUT / "ridge_inner_grid.csv", index=False)
    point["lgbm"]["grid"].to_csv(OUT / "lgbm_inner_grid.csv", index=False)
    conformal["selection"].to_csv(OUT / "conformal_window_selection.csv", index=False)
    conformal["test"].to_csv(OUT / "conformal_test.csv", index=False)
    conformal["by_month"].to_csv(OUT / "conformal_by_month.csv", index=False)
    conformal["regime"].to_csv(OUT / "conformal_regime.csv", index=False)
    conformal["crps"].to_csv(OUT / "conformal_crps.csv", index=False)
    gaps = ready.index.to_series().diff().dropna()
    meta_gaps = {
        "max_ready_gap_minutes": float(gaps.max().total_seconds() / 60.0),
        "share_ready_gaps_over_5min": float((gaps > pd.Timedelta(minutes=5)).mean()),
    }
    meta = {
        "horizon": "5min",
        "origin": "interval_end minus 5 minutes",
        "train_end": "2025-09-30 23:55:00",
        "calibration_end": "2026-03-31 23:55:00",
        "test_start": str(point["test"].index.min()),
        "test_end": str(point["test"].index.max()),
        "n_train": int(len(point["train"])),
        "n_calibration": int(len(point["calibration"])),
        "n_test": int(len(point["test"])),
        "n_ready": int(len(ready)),
        "ridge_alpha": point["ridge"]["alpha"],
        "ridge_inner_mae": point["ridge"]["inner_mae"],
        "lgbm_params": point["lgbm"]["params"],
        "lgbm_objective": point["lgbm"]["objective"],
        "lgbm_target": point["lgbm"]["target"],
        "lgbm_inner_mae": point["lgbm"]["inner_mae"],
        "lgbm_persistence_inner_mae": point["lgbm"]["persistence_inner_mae"],
        "lgbm_inner_skill_vs_persistence": skill_mae(
            point["lgbm"]["inner_mae"], point["lgbm"]["persistence_inner_mae"]
        ),
        "spike_cut_train_p95": point["spike_cut"],
        "floor_cut_train_p05": point["floor_cut"],
        "mae_diff_lightgbm_minus_persistence": point["bootstrap"],
        "conformal_primary_method": conformal["primary_method"],
        "conformal_primary_window_steps": conformal["primary_window"],
        "window_candidates": WINDOW_CANDIDATES,
        "note": "Window choice and hyperparameters used only pre-test data. STEM contemporaneous use is an assumption.",
        **meta_gaps,
    }
    (OUT / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    print("building ex-ante frame", flush=True)
    ready = load_ready()
    print("fitting point models", flush=True)
    point = fit_and_score(ready)
    print("calibrating conformal windows", flush=True)
    conformal = run_conformal(point)
    save_results(ready, point, conformal)
    print(point["point"].to_string(index=False))
    print(conformal["selection"].to_string(index=False))
    print(conformal["test"].to_string(index=False))
    print("primary", conformal["primary_method"], conformal["primary_window"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
