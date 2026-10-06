"""Regime switch between LightGBM and 5-minute persistence.

The gate is chosen on calibration only. Ordinary origins use the saved LightGBM
forecast. Violent origins use persistence. The frozen 7-day absolute conformal
fence is then drawn around whichever centre was issued. The test set is not
read until the rule is frozen.

This is a baseline for a spike model, not an onset forecast. The gate can only
see a move that has already happened.
"""

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

from scripts.conformal import apply_sliding, sliding_crps
from scripts.forecast_design import TARGET, TREE_FEATURES, build_exante_frame, model_ready, slice_split
from scripts.metrics import point_scores
from scripts.paths import PROJECT_ROOT, panel_path
from scripts.point_models import persistence, predict_lgbm

OUT = PROJECT_ROOT / "reports" / "forecast"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_5min_ahead.joblib"
WINDOW = 2016
ALPHA = 0.10
SPIKE_Q = 0.95
FLOOR_Q = 0.05


def load():
    raw = pd.read_parquet(panel_path(), engine="pyarrow")
    for col in ("interval_end", "trading_interval_end"):
        if col in raw.columns:
            raw[col] = pd.to_datetime(raw[col]).astype("datetime64[ns]")
    ready = model_ready(build_exante_frame(raw), TREE_FEATURES)
    saved = joblib.load(MODEL_PATH)
    return ready, saved


def forecasts(block, bundle, features):
    return pd.DataFrame(
        {
            "y": block[TARGET].to_numpy(dtype=float),
            "lightgbm": predict_lgbm(bundle, block, features).to_numpy(dtype=float),
            "persistence": persistence(block, "mcp_lag_5min").to_numpy(dtype=float),
            "move_30": np.abs(block["mcp_lag_5min"].to_numpy(dtype=float) - block["mcp_lag_30min"].to_numpy(dtype=float)),
            "move_60": np.abs(block["mcp_lag_5min"].to_numpy(dtype=float) - block["mcp_lag_60min"].to_numpy(dtype=float)),
            "last": block["mcp_lag_5min"].to_numpy(dtype=float),
        },
        index=block.index,
    )


def switched(frame, mask):
    out = frame["lightgbm"].to_numpy(dtype=float).copy()
    out[np.asarray(mask)] = frame["persistence"].to_numpy(dtype=float)[np.asarray(mask)]
    return out


def mask_for(frame, rule):
    name, feature, threshold, spike, floor = rule
    if name == "already_in_train_tail":
        return (frame["last"] >= spike) | (frame["last"] <= floor)
    return frame[feature] >= threshold


def candidate_rules(train, calibration):
    spike = float(train[TARGET].quantile(SPIKE_Q))
    floor = float(train[TARGET].quantile(FLOOR_Q))
    cut = calibration.index[len(calibration) // 2]
    early = calibration.loc[calibration.index < cut]
    rules = [("already_in_train_tail", "train_p05_p95", np.nan, spike, floor)]
    for col, name in (("move_30", "abs_30min_move"), ("move_60", "abs_60min_move")):
        for q in (0.50, 0.70, 0.80, 0.90, 0.95):
            rules.append((f"{name}_q{int(q * 100)}", col, float(early[col].quantile(q)), spike, floor))
    return rules, spike, floor, cut


def select_rule(calibration, rules, cut):
    late = calibration.loc[calibration.index >= cut]
    base = float(point_scores(late["y"], late["lightgbm"])["mae"])
    rows = []
    for rule in rules:
        yhat = switched(late, mask_for(late, rule))
        tail = (late["last"] >= rule[3]) | (late["last"] <= rule[4])
        rows.append(
            {
                "rule": rule[0],
                "late_mae": point_scores(late["y"], yhat)["mae"],
                "late_tail_mae": float(np.mean(np.abs(late["y"].to_numpy()[tail] - yhat[tail]))) if tail.any() else np.nan,
                "late_switch_share": float(np.mean(mask_for(late, rule))),
                "detail": rule[1],
                "threshold": rule[2],
            }
        )
    table = pd.DataFrame(rows).sort_values(["late_tail_mae", "late_mae"])
    eligible = table.loc[table["late_mae"] <= base * 1.01]
    chosen_name = (eligible if not eligible.empty else table).iloc[0]["rule"]
    chosen = next(rule for rule in rules if rule[0] == chosen_name)
    return chosen, table, base


def score_block(y, yhat, lower, upper, crps, tail):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    rows = []
    for name, mask in (("all", np.ones(len(y), dtype=bool)), ("body", ~tail), ("tail", tail)):
        yy, yh = y[mask], yhat[mask]
        lo, hi = np.asarray(lower)[mask], np.asarray(upper)[mask]
        rows.append(
            {
                "regime": name,
                "n": int(mask.sum()),
                "mae": float(np.mean(np.abs(yy - yh))),
                "coverage_90": float(np.mean((yy >= lo) & (yy <= hi))),
                "mean_width": float(np.mean(hi - lo)),
                "median_width": float(np.median(hi - lo)),
                "crps": float(np.mean(np.asarray(crps)[mask])),
            }
        )
    return pd.DataFrame(rows)


def conformal_block(name, y_cal, yhat_cal, y_test, yhat_test, index, tail, switch_share):
    history = pd.Series(np.concatenate([y_cal - yhat_cal, y_test - yhat_test]), index=index)
    pred = pd.Series(yhat_test, index=index[len(y_cal):])
    band = apply_sliding(history, pred, WINDOW, ALPHA, "absolute")
    crps = sliding_crps(history, pred, pd.Series(y_test, index=pred.index), WINDOW, "absolute")
    block = score_block(y_test, yhat_test, band["lower"], band["upper"], crps, tail)
    block.insert(0, "model", name)
    block["switch_share"] = switch_share
    return block


def main():
    ready, saved = load()
    train = slice_split(ready, "train")
    calibration = forecasts(slice_split(ready, "calibration"), saved["bundle"], saved["features"])
    test = forecasts(slice_split(ready, "test"), saved["bundle"], saved["features"])
    rules, spike, floor, cut = candidate_rules(train, calibration)
    chosen, selection, base_mae = select_rule(calibration, rules, cut)
    test_switch = mask_for(test, chosen)
    yhat_cal = switched(calibration, mask_for(calibration, chosen))
    yhat_test = switched(test, test_switch)
    index = calibration.index.append(test.index)
    tail = (test["y"].to_numpy() >= spike) | (test["y"].to_numpy() <= floor)
    comparison = pd.concat(
        [
            conformal_block("regime_switch", calibration["y"].to_numpy(), yhat_cal, test["y"].to_numpy(), yhat_test, index, tail, float(np.mean(test_switch))),
            conformal_block("lightgbm", calibration["y"].to_numpy(), calibration["lightgbm"].to_numpy(), test["y"].to_numpy(), test["lightgbm"].to_numpy(), index, tail, 0.0),
            conformal_block("persistence", calibration["y"].to_numpy(), calibration["persistence"].to_numpy(), test["y"].to_numpy(), test["persistence"].to_numpy(), index, tail, 1.0),
        ],
        ignore_index=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    selection.to_csv(OUT / "regime_gate_selection.csv", index=False)
    comparison.to_csv(OUT / "regime_cps_comparison.csv", index=False)
    meta = {
        "rule": chosen[0],
        "feature": chosen[1],
        "threshold": None if chosen[0] == "already_in_train_tail" else float(chosen[2]),
        "train_spike": spike,
        "train_floor": floor,
        "late_lightgbm_mae": base_mae,
        "selection": "late calibration tail MAE, subject to late MAE within 1% of LightGBM",
        "window_steps": WINDOW,
        "method": "absolute",
        "alpha": ALPHA,
        "test_switch_share": float(np.mean(test_switch)),
        "n_test": int(len(test)),
        "n_test_tail": int(tail.sum()),
        "note": "Gate chosen on calibration only. Not an onset model. See docs/spike_forecast_next_steps.md.",
    }
    (OUT / "regime_cps_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(selection.to_string(index=False))
    print(comparison.to_string(index=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
