"""QRA variants, recent calibration windows, and a calibration-only selection.

What is compared
----------------
All variants combine the ex-ante 5-minute-ahead forecasts already produced by
scripts/point_models.py, at the same origins and the same horizon.

* ``qra``  Quantile Regression Averaging (Nowotarski and Weron, 2015,
  Computational Statistics 30, 791-803). One linear quantile regression of MCP
  on the individual point forecasts per level. This is what scripts/qra.py
  already did with all five forecasts; it is the standard QRA.
* ``qrm``  Quantile Regression Machine (Marcjasz, Uniejewski and Weron, 2020,
  IJF 36, 466-479). Quantile regression of MCP on the simple average of the
  point forecasts.
* ``qave`` Quantile averaging (Q-Ave in Marcjasz et al., 2020). Each member is
  turned into its own probabilistic forecast first and the member quantiles are
  averaged level by level (Vincentization). A point-forecast member becomes a
  probabilistic forecast through a univariate quantile regression of MCP on that
  forecast. The ``lightgbm_quantile`` member is a native quantile forecast.
* ``lightgbm_quantile`` LightGBM with the quantile objective, one model per
  level, fit on the training window only with the frozen point-model
  hyperparameters (no new tuning). Like the point model it predicts the
  5-minute price change and adds the 5-minute lag.

Recency
-------
Every calibration-fitted variant is fit on the most recent 28, 56 or 91 days of
the available pre-test data, or on all of it. LightGBM quantile does not use
calibration data.

Selection (calibration only)
----------------------------
Inner split of calibration: fit windows end on 2026-02-28 23:55 and every
variant is scored on 1-31 March 2026. Among variants with March 90% coverage
in [0.88, 0.92], the narrowest mean 90% width is selected (ties: lower
quantile-integral CRPS). If no variant is in the band, nothing is selected.
The selected variant is then refit with the same window length ending
2026-03-31 23:55 and scored once on test. All other variants are also scored
on test for transparency, but test scores are never used to choose or tune.
No interval is shrunk, widened, or recentred using test outcomes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.conformal import ALPHAS, apply_sliding
from scripts.forecast_design import CAL_END, TARGET, TREE_CATEGORICAL
from scripts.metrics import crps_from_quantiles, interval_group_scores, interval_scores
from scripts.paths import PROJECT_ROOT
from scripts.qra import (
    QRA_FORECASTS,
    comparison_levels,
    conformal_quantile_grid,
    fit_qra,
    level_name,
    predict_qra,
)

OUT = PROJECT_ROOT / "reports" / "forecast"
INNER_FIT_END = pd.Timestamp("2026-02-28 23:55:00")
INNER_VAL_START = pd.Timestamp("2026-03-01 00:00:00")
WINDOWS: dict[str, int | None] = {"28d": 28, "56d": 56, "91d": 91, "all": None}
COVERAGE_BAND = (0.88, 0.92)
SELECTION_ALPHA = 0.10
LGBM_QUANTILE = "lightgbm_quantile"
MEMBER_MEAN = "member_mean"

SUBSETS: dict[str, tuple[str, ...]] = {
    "all5": QRA_FORECASTS,
    "lgbm_ridge_p5": ("lightgbm", "ridge", "persistence_5min"),
    "lgbm_p5": ("lightgbm", "persistence_5min"),
    "lgbm": ("lightgbm",),
}


@dataclass(frozen=True)
class Spec:
    combiner: str
    subset: str
    members: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.combiner}:{self.subset}"

    @property
    def uses_calibration(self) -> bool:
        return self.combiner != LGBM_QUANTILE


def default_specs() -> list[Spec]:
    specs = [Spec("qra", key, members) for key, members in SUBSETS.items()]
    # With one member QRA, QRM and Q-Ave are the same model, so only qra:lgbm is kept.
    for combiner in ("qrm", "qave"):
        specs.extend(Spec(combiner, key, members) for key, members in SUBSETS.items() if len(members) > 1)
    specs.append(Spec("qave", "lgbmq_lgbm_ridge_p5", (LGBM_QUANTILE, "lightgbm", "ridge", "persistence_5min")))
    specs.append(Spec(LGBM_QUANTILE, "native", (LGBM_QUANTILE,)))
    return specs


def window_rows(index: pd.DatetimeIndex, end: pd.Timestamp, days: int | None) -> np.ndarray:
    """Boolean mask of rows at or before `end` and inside the last `days` days."""
    mask = np.asarray(index <= end)
    if days is not None:
        mask &= np.asarray(index > end - pd.Timedelta(days=days))
    return mask


def _sorted_frame(values: np.ndarray, index: pd.Index, levels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.sort(values, axis=1), index=index, columns=[level_name(level) for level in levels])


def fit_spec(spec: Spec, forecasts: pd.DataFrame, y: pd.Series, levels: np.ndarray) -> dict:
    """Fit a spec on the rows given. `forecasts` and `y` must already be restricted to the fit window."""
    if not spec.uses_calibration:
        return {"spec": spec, "coefficients": {}}
    if spec.combiner == "qra":
        return {"spec": spec, "coefficients": {"qra": fit_qra(forecasts, y, levels, columns=spec.members)}}
    if spec.combiner == "qrm":
        frame = pd.DataFrame({MEMBER_MEAN: forecasts.loc[:, list(spec.members)].mean(axis=1)})
        return {"spec": spec, "coefficients": {MEMBER_MEAN: fit_qra(frame, y, levels, columns=(MEMBER_MEAN,))}}
    if spec.combiner == "qave":
        coefficients = {
            member: fit_qra(forecasts, y, levels, columns=(member,))
            for member in spec.members
            if member != LGBM_QUANTILE
        }
        return {"spec": spec, "coefficients": coefficients}
    raise ValueError(f"unknown combiner: {spec.combiner}")


def predict_spec(
    fitted: dict,
    forecasts: pd.DataFrame,
    levels: np.ndarray,
    native: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float]:
    """Return rearranged quantiles and the share of rows whose raw quantiles crossed.

    For Q-Ave each member is sorted before averaging, so the average is sorted too.
    The crossing share is the share of rows where any member crossed.
    """
    spec: Spec = fitted["spec"]
    names = [level_name(level) for level in levels]

    def native_values() -> np.ndarray:
        if native is None:
            raise ValueError("lightgbm_quantile member needs native quantiles")
        values = native.reindex(forecasts.index).loc[:, names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("native quantiles contain non-finite values")
        return values

    if spec.combiner == LGBM_QUANTILE:
        raw = native_values()
        crossed = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
        return _sorted_frame(raw, forecasts.index, levels), crossed
    if spec.combiner == "qra":
        predicted, crossed = predict_qra(forecasts, fitted["coefficients"]["qra"])
        return predicted.loc[:, names], crossed
    if spec.combiner == "qrm":
        frame = pd.DataFrame({MEMBER_MEAN: forecasts.loc[:, list(spec.members)].mean(axis=1)})
        predicted, crossed = predict_qra(frame, fitted["coefficients"][MEMBER_MEAN])
        return predicted.loc[:, names], crossed
    if spec.combiner == "qave":
        stack = []
        any_crossed = np.zeros(len(forecasts), dtype=bool)
        for member in spec.members:
            if member == LGBM_QUANTILE:
                raw = native_values()
            else:
                coefficients = fitted["coefficients"][member]
                from scripts.qra import _coefficient_matrix, _design

                order_levels, matrix, columns = _coefficient_matrix(coefficients)
                if not np.allclose(order_levels, levels):
                    raise ValueError("member levels do not match")
                raw = _design(forecasts, columns) @ matrix.T
            any_crossed |= np.any(np.diff(raw, axis=1) < 0, axis=1)
            stack.append(np.sort(raw, axis=1))
        averaged = np.mean(np.stack(stack, axis=0), axis=0)
        return _sorted_frame(averaged, forecasts.index, levels), float(np.mean(any_crossed))
    raise ValueError(f"unknown combiner: {spec.combiner}")


def score_quantiles(y: pd.Series, quantiles: pd.DataFrame, levels: np.ndarray) -> dict:
    """Coverage and width of the central intervals in ALPHAS plus quantile-integral CRPS."""
    y_values = y.reindex(quantiles.index).to_numpy(dtype=float)
    row: dict[str, float] = {"n": int(len(y_values))}
    for alpha in ALPHAS:
        tag = f"{int(round(100 * (1 - alpha)))}"
        scores = interval_scores(y_values, quantiles[level_name(alpha / 2)], quantiles[level_name(1 - alpha / 2)])
        row[f"coverage_{tag}"] = scores["coverage"]
        row[f"mean_width_{tag}"] = scores["mean_width"]
        row[f"median_width_{tag}"] = scores["median_width"]
    row["crps_quantile_integral"] = float(np.mean(crps_from_quantiles(y_values, quantiles.to_numpy(dtype=float), levels)))
    return row


def select_variant(table: pd.DataFrame, band: tuple[float, float] = COVERAGE_BAND) -> str | None:
    """Narrowest mean 90% width among rows with 90% coverage inside `band`; ties by CRPS.

    Only rows marked as candidates are considered. Returns the variant id or None.
    """
    pool = table.loc[table["candidate"].astype(bool)]
    eligible = pool.loc[pool["coverage_90"].between(band[0], band[1])]
    if eligible.empty:
        return None
    best = eligible.sort_values(["mean_width_90", "crps_quantile_integral", "variant"]).iloc[0]
    return str(best["variant"])


def lgbm_quantile_params(point_params: dict) -> dict:
    return dict(point_params)


def fit_lgbm_quantiles(
    train: pd.DataFrame,
    features: list[str],
    levels: np.ndarray,
    params: dict,
    n_estimators: int = 250,
    lag_column: str = "mcp_lag_5min",
) -> dict:
    """One LightGBM quantile model per level on the training window only."""
    from lightgbm import LGBMRegressor

    x = train[features].copy()
    for col in TREE_CATEGORICAL:
        if col in x.columns:
            x[col] = x[col].astype(int).astype("category")
    categorical = [col for col in TREE_CATEGORICAL if col in x.columns]
    change = train[TARGET].to_numpy(dtype=float) - train[lag_column].to_numpy(dtype=float)
    models = {}
    for level in levels:
        model = LGBMRegressor(
            objective="quantile",
            alpha=float(level),
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            **params,
        )
        model.fit(x, change, categorical_feature=categorical)
        models[float(level)] = model
    categories = {col: list(x[col].cat.categories) for col in categorical}
    return {"models": models, "features": list(features), "categories": categories, "lag_column": lag_column}


def predict_lgbm_quantiles(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[bundle["features"]].copy()
    for col, categories in bundle["categories"].items():
        x[col] = pd.Categorical(x[col].astype(int), categories=categories)
    lag = frame[bundle["lag_column"]].to_numpy(dtype=float)
    columns = {level_name(level): lag + model.predict(x) for level, model in bundle["models"].items()}
    return pd.DataFrame(columns, index=frame.index)


def _conformal_reference(
    y_hist: pd.Series,
    yhat_hist: pd.Series,
    y_eval: pd.Series,
    yhat_eval: pd.Series,
    window: int,
    method: str,
    levels: np.ndarray,
) -> dict:
    """Score the frozen sliding conformal interval on `y_eval`.

    The interval at each origin uses only residuals strictly before it. On the
    inner validation month those are calibration residuals (pre-March and
    earlier March origins). On test they are calibration plus earlier test
    residuals, exactly as in conformal_test.csv.
    """
    history = (pd.concat([y_hist, y_eval]) - pd.concat([yhat_hist, yhat_eval])).sort_index()
    row: dict[str, float] = {"n": int(len(y_eval))}
    for alpha in ALPHAS:
        tag = f"{int(round(100 * (1 - alpha)))}"
        interval = apply_sliding(history, yhat_eval, int(window), alpha, method)
        scores = interval_scores(y_eval, interval["lower"], interval["upper"])
        row[f"coverage_{tag}"] = scores["coverage"]
        row[f"mean_width_{tag}"] = scores["mean_width"]
        row[f"median_width_{tag}"] = scores["median_width"]
    grid = conformal_quantile_grid(history, yhat_eval, int(window), method, levels)
    grid = grid.add(yhat_eval.to_numpy(dtype=float), axis=0)
    row["crps_quantile_integral"] = float(
        np.mean(crps_from_quantiles(y_eval.to_numpy(dtype=float), grid.to_numpy(dtype=float), levels))
    )
    return row


def run_variant_study(
    point: dict,
    native_quantiles: pd.DataFrame | None,
    primary_method: str,
    primary_window: int,
    specs: list[Spec] | None = None,
    windows: dict[str, int | None] | None = None,
    inner_fit_end: pd.Timestamp = INNER_FIT_END,
    inner_val_start: pd.Timestamp = INNER_VAL_START,
    cal_end: pd.Timestamp = CAL_END,
    levels: np.ndarray | None = None,
) -> dict:
    """Score every variant on the inner calibration month, select, then score on test."""
    specs = default_specs() if specs is None else specs
    windows = WINDOWS if windows is None else windows
    levels = comparison_levels() if levels is None else np.asarray(levels, dtype=float)
    if native_quantiles is None:
        specs = [spec for spec in specs if LGBM_QUANTILE not in spec.members]
    calibration = point["calibration"]
    test = point["test"]
    predictions = point["predictions"]
    names = sorted({m for spec in specs for m in spec.members if m != LGBM_QUANTILE})
    y_cal = calibration[TARGET]
    y_test = test[TARGET]
    if not (y_cal.index.max() <= cal_end < y_test.index.min()):
        raise ValueError("calibration must end before test starts")
    if not (inner_fit_end < inner_val_start <= y_cal.index.max()):
        raise ValueError("inner validation must lie inside calibration")
    cal_frame = pd.DataFrame({n: predictions[n].reindex(y_cal.index) for n in names}, index=y_cal.index)
    test_frame = pd.DataFrame({n: predictions[n].reindex(y_test.index) for n in names}, index=y_test.index)
    val_mask = np.asarray(y_cal.index >= inner_val_start)
    val_index = y_cal.index[val_mask]

    def variants():
        for spec in specs:
            if spec.uses_calibration:
                for label, days in windows.items():
                    yield spec, label, days
            else:
                yield spec, "none", None

    calib_rows = []
    fitted_final: dict[str, dict] = {}
    for spec, label, days in variants():
        variant = f"{spec.name}@{label}"
        if spec.uses_calibration:
            fit_mask = window_rows(y_cal.index, inner_fit_end, days)
            fitted = fit_spec(spec, cal_frame.loc[fit_mask], y_cal.loc[fit_mask], levels)
            n_fit = int(fit_mask.sum())
        else:
            fitted = fit_spec(spec, cal_frame, y_cal, levels)
            n_fit = 0
        predicted, _crossed = predict_spec(fitted, cal_frame.loc[val_index], levels, native_quantiles)
        row = score_quantiles(y_cal.loc[val_index], predicted, levels)
        converged = all(bool(c["converged"].all()) for c in fitted["coefficients"].values())
        row.update(
            {
                "variant": variant,
                "combiner": spec.combiner,
                "subset": spec.subset,
                "members": "+".join(spec.members),
                "window": label,
                "window_days": np.nan if days is None else days,
                "fit_end": str(inner_fit_end) if spec.uses_calibration else "training window",
                "n_fit": n_fit,
                "eval_period": f"{val_index.min()} to {val_index.max()}",
                "all_levels_converged": converged,
                "candidate": True,
                "role": "variant",
            }
        )
        calib_rows.append(row)
        fitted_final[variant] = {"spec": spec, "days": days, "label": label}

    calib = pd.DataFrame(calib_rows)
    selected = select_variant(calib)
    calib["selected"] = calib["variant"].eq(selected) if selected is not None else False
    eligible = calib.loc[calib["coverage_90"].between(*COVERAGE_BAND)]
    crps_pick = None
    if not eligible.empty:
        crps_pick = str(eligible.sort_values(["crps_quantile_integral", "variant"]).iloc[0]["variant"])

    # Frozen conformal reference on the inner month: history is calibration only.
    ref_rows = []
    for base in ("lightgbm", "persistence_5min"):
        yhat_cal = predictions[base].reindex(y_cal.index)
        row = _conformal_reference(
            y_cal.loc[~val_mask], yhat_cal.loc[~val_mask], y_cal.loc[val_index], yhat_cal.loc[val_index],
            primary_window, primary_method, levels,
        )
        row.update({"variant": f"conformal_{primary_method}_{primary_window}:{base}", "combiner": "sliding_conformal",
                    "subset": base, "members": base, "window": f"{primary_window} steps", "candidate": False,
                    "selected": False, "role": "frozen_conformal_reference",
                    "eval_period": f"{val_index.min()} to {val_index.max()}"})
        ref_rows.append(row)
    calib = pd.concat([calib, pd.DataFrame(ref_rows)], ignore_index=True)

    # Test: refit every variant with its window ending at cal_end, score once.
    test_rows = []
    selected_quantiles = None
    baseline_quantiles = None
    selected_coefficients = None
    for variant, info in fitted_final.items():
        spec: Spec = info["spec"]
        if spec.uses_calibration:
            fit_mask = window_rows(y_cal.index, cal_end, info["days"])
            fitted = fit_spec(spec, cal_frame.loc[fit_mask], y_cal.loc[fit_mask], levels)
            n_fit = int(fit_mask.sum())
            fit_start = str(y_cal.index[fit_mask].min())
        else:
            fitted = fit_spec(spec, cal_frame, y_cal, levels)
            n_fit = 0
            fit_start = "training window"
        if not np.isfinite(test_frame.to_numpy(dtype=float)).all():
            raise ValueError("test point forecasts are not all finite")
        predicted, crossed = predict_spec(fitted, test_frame, levels, native_quantiles)
        row = score_quantiles(y_test, predicted, levels)
        converged = all(bool(c["converged"].all()) for c in fitted["coefficients"].values())
        row.update(
            {
                "variant": variant,
                "combiner": spec.combiner,
                "subset": spec.subset,
                "members": "+".join(spec.members),
                "window": info["label"],
                "window_days": np.nan if info["days"] is None else info["days"],
                "fit_start": fit_start,
                "fit_end": str(cal_end) if spec.uses_calibration else "training window",
                "n_fit": n_fit,
                "share_rows_crossed_before_rearrangement": crossed,
                "all_levels_converged": converged,
                "selected_on_calibration": variant == selected,
                "is_original_qra": variant == "qra:all5@all",
                "role": "variant",
            }
        )
        test_rows.append(row)
        if variant == selected:
            selected_quantiles = predicted
            frames = []
            for key, table in fitted["coefficients"].items():
                frame = table.copy()
                frame.insert(0, "member_fit", key)
                frames.append(frame)
            selected_coefficients = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if variant == "qra:all5@all":
            baseline_quantiles = predicted

    for base in ("lightgbm", "persistence_5min"):
        row = _conformal_reference(
            y_cal, predictions[base].reindex(y_cal.index), y_test, predictions[base].reindex(y_test.index),
            primary_window, primary_method, levels,
        )
        row.update({"variant": f"conformal_{primary_method}_{primary_window}:{base}", "combiner": "sliding_conformal",
                    "subset": base, "members": base, "window": f"{primary_window} steps",
                    "selected_on_calibration": False, "is_original_qra": False,
                    "role": "frozen_conformal_reference"})
        test_rows.append(row)
    test_table = pd.DataFrame(test_rows)

    month_rows = []
    months = y_test.index.to_period("M").astype(str)
    for label, quantiles in (("selected", selected_quantiles), ("original_qra", baseline_quantiles)):
        if quantiles is None:
            continue
        integral = crps_from_quantiles(y_test.to_numpy(dtype=float), quantiles.to_numpy(dtype=float), levels)
        table = interval_group_scores(y_test, quantiles[level_name(0.05)], quantiles[level_name(0.95)], months, 0.10)
        frame = pd.DataFrame({"g": months, "v": integral}).groupby("g")["v"].mean()
        table["crps_quantile_integral"] = table["group"].map(frame)
        table["model"] = selected if label == "selected" else "qra:all5@all"
        table["role"] = label
        month_rows.append(table.rename(columns={"group": "month"}))
    yhat_test = predictions["lightgbm"].reindex(y_test.index)
    history = (pd.concat([y_cal, y_test]) - pd.concat([predictions["lightgbm"].reindex(y_cal.index), yhat_test])).sort_index()
    frozen = apply_sliding(history, yhat_test, int(primary_window), 0.10, primary_method)
    grid = conformal_quantile_grid(history, yhat_test, int(primary_window), primary_method, levels).add(
        yhat_test.to_numpy(dtype=float), axis=0
    )
    integral = crps_from_quantiles(y_test.to_numpy(dtype=float), grid.to_numpy(dtype=float), levels)
    table = interval_group_scores(y_test, frozen["lower"], frozen["upper"], months, 0.10)
    table["crps_quantile_integral"] = table["group"].map(pd.DataFrame({"g": months, "v": integral}).groupby("g")["v"].mean())
    table["model"] = f"conformal_{primary_method}_{primary_window}:lightgbm"
    table["role"] = "frozen_conformal_reference"
    month_rows.append(table.rename(columns={"group": "month"}))

    meta = {
        "purpose": "QRA variants with recent calibration windows; selection on calibration only",
        "combiners": {
            "qra": "Nowotarski & Weron (2015): linear quantile regression of MCP on the member point forecasts, one fit per level",
            "qrm": "Marcjasz, Uniejewski & Weron (2020): quantile regression of MCP on the mean of the member point forecasts",
            "qave": "Quantile averaging (Q-Ave, Marcjasz et al. 2020): each member's quantiles (univariate QR on a point forecast, or native LightGBM quantiles) are sorted and averaged level by level",
            "lightgbm_quantile": "LightGBM quantile objective, one model per level, fit on the training window only with the frozen point-model hyperparameters; predicts the 5-minute change and adds the 5-minute lag",
        },
        "not_implemented": "Probability averaging (F-Ave) was not implemented: the member quantile grids stop at 0.01 and 0.99, so the mixture CDF tails would need an extra extrapolation assumption.",
        "subsets": {key: list(value) for key, value in SUBSETS.items()},
        "windows_days": {key: value for key, value in windows.items()},
        "levels": [float(level) for level in levels],
        "inner_fit_end": str(inner_fit_end),
        "inner_validation": f"{val_index.min()} to {val_index.max()}",
        "n_inner_validation": int(len(val_index)),
        "final_fit_end": str(cal_end),
        "test_period": f"{y_test.index.min()} to {y_test.index.max()}",
        "n_test": int(len(y_test)),
        "selection_rule": (
            "On the inner calibration month only: among candidate variants with 90% coverage in "
            f"[{COVERAGE_BAND[0]}, {COVERAGE_BAND[1]}], choose the narrowest mean 90% width; ties by lower "
            "quantile-integral CRPS. If none is in the band, nothing is selected and the original QRA stays. "
            "The selected spec is refit with the same window length ending at the calibration end and "
            "scored once on test. Test scores of the other variants are reported but not used."
        ),
        "selected_variant": selected,
        "n_variants_in_band": int(len(eligible)),
        "sensitivity_lowest_crps_in_band": crps_pick,
        "original_qra_variant": "qra:all5@all",
        "primary_conformal_method": primary_method,
        "primary_conformal_window_steps": int(primary_window),
        "rearrangement": "per-row sort of the predicted quantiles (does not use the outcome)",
        "crps_quantile_integral": "Trapezoidal integral of twice the pinball loss over the level grid (scripts.metrics.crps_from_quantiles); same estimator as qra_crps.csv",
        "conformal_reference_note": (
            "Inner month: sliding window over calibration residuals only (the window can extend into March "
            "as the month progresses, exactly as the online interval would). Test: as in conformal_test.csv."
        ),
        "test_outcomes_used_for_fit_or_selection": False,
    }
    return {
        "calib": calib,
        "test": test_table,
        "by_month": pd.concat(month_rows, ignore_index=True),
        "selected_coefficients": selected_coefficients,
        "meta": meta,
    }


def write_variant_tables(study: dict, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    study["calib"].to_csv(out / "qra_variants_calib.csv", index=False)
    study["test"].to_csv(out / "qra_variants_test.csv", index=False)
    study["by_month"].to_csv(out / "qra_variants_by_month.csv", index=False)
    if study["selected_coefficients"] is not None and not study["selected_coefficients"].empty:
        study["selected_coefficients"].to_csv(out / "qra_variants_selected_coefficients.csv", index=False)
    (out / "qra_variants_meta.json").write_text(json.dumps(study["meta"], indent=2), encoding="utf-8")


def main() -> None:
    from scripts.experiment import load_ready
    from scripts.forecast_design import TREE_FEATURES, slice_split
    from scripts.qra import _saved_point_result

    print("loading saved point forecasts", flush=True)
    point, method, window = _saved_point_result()
    committed = json.loads((OUT / "experiment_meta.json").read_text(encoding="utf-8"))
    ready = load_ready()
    train = slice_split(ready, "train")
    held = pd.concat([slice_split(ready, "calibration"), slice_split(ready, "test")]).sort_index()
    levels = comparison_levels()
    print("fitting lightgbm quantile models on the training window", flush=True)
    bundle = fit_lgbm_quantiles(train, TREE_FEATURES, levels, lgbm_quantile_params(committed["lgbm_params"]))
    native = predict_lgbm_quantiles(bundle, held)
    print("scoring variants on the inner calibration month", flush=True)
    study = run_variant_study(point, native, method, window, levels=levels)
    study["meta"]["lightgbm_quantile_params"] = {**committed["lgbm_params"], "n_estimators": 250}
    study["meta"]["lightgbm_quantile_source"] = "fit on the training window in this run; not saved"
    write_variant_tables(study)
    cols = ["variant", "coverage_90", "mean_width_90", "median_width_90", "crps_quantile_integral"]
    print(study["calib"][cols + ["selected"]].to_string(index=False))
    print("selected:", study["meta"]["selected_variant"])
    print(study["test"][cols + ["selected_on_calibration"]].to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
