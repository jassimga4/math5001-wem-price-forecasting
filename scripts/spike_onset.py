"""Stage-1 spike-onset model with external data.

A hurdle model on origins whose last price is inside the training 5th-95th
band: a LightGBM classifier for an upward crossing and one for a downward
crossing, then a LightGBM size model (median and a quantile grid) fitted on the
training rows that crossed. Everything is fitted on the training window only
(rows from 2024-03-01, when the archived weather forecasts start; early
stopping on 2025-07-01 to 2025-09-30). Cutoffs are chosen on calibration and
frozen. The test window is scored once.

Outside the gate, and on every origin whose last price is already in the tail,
the issued forecast is the frozen regime switch (scripts/regime_switch.py).
The predictive distribution is the regime switch's 7-day absolute conformal
residual window around that centre. When a classifier fires, a probability
weight p is moved from that window onto the size model's quantile grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.external_features import (  # noqa: E402
    EXTERNAL_PREDISPATCH, EXTERNAL_PREDISPATCH_ALL, EXTERNAL_WEATHER, FEATURES_PATH, build_external,
)
from scripts.forecast_design import INNER_VAL_START, TREE_FEATURES, slice_split, split_mask  # noqa: E402
from scripts.metrics import crps_from_residual_samples  # noqa: E402
from scripts.regime_switch import WINDOW, forecasts, load, mask_for, switched  # noqa: E402

OUT = ROOT / "reports" / "forecast"
FIT_START = pd.Timestamp("2024-03-01")
SIZE_LEVELS = np.round(np.arange(0.05, 1.0, 0.10), 2)
CUTOFFS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, np.inf]
# Stage 1b: extended below 0.02 because every stage-1 model chose the lowest value.
MIX_CUTOFFS = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, np.inf]
PANEL = [*TREE_FEATURES, "move_30", "move_60"]
FEATURE_SETS = {
    "hurdle_panel": PANEL,
    "hurdle_weather": PANEL + EXTERNAL_WEATHER,
    "hurdle_weather_predispatch": PANEL + EXTERNAL_WEATHER + EXTERNAL_PREDISPATCH,  # even-hour runs (stage 1)
    "hurdle_weather_predispatch_all": PANEL + EXTERNAL_WEATHER + EXTERNAL_PREDISPATCH_ALL,  # hourly runs + revisions
}
EXTERNAL_PREFIXES = ("pd_", "pda_", "wx_")
CLF_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=7,
)
SIZE_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=7,
)


# --------------------------------------------------------------------------- data


def assemble():
    ready, saved = load()
    meta = json.loads((OUT / "regime_cps_meta.json").read_text())
    rule = (meta["rule"], meta["feature"], meta["threshold"], meta["train_spike"], meta["train_floor"])
    spike, floor = float(meta["train_spike"]), float(meta["train_floor"])
    external = build_external(ready)  # always rebuilt so new runs or weather are never stale
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    external.to_parquet(FEATURES_PATH)
    frame = forecasts(ready, saved["bundle"], saved["features"])
    frame["regime"] = switched(frame, mask_for(frame, rule))
    frame = frame.join(ready[TREE_FEATURES]).join(external.drop(columns=["forecast_origin"]))
    frame["split"] = split_mask(frame.index)
    frame["eligible"] = (frame["last"] < spike) & (frame["last"] > floor)
    frame["tail"] = (frame["y"] >= spike) | (frame["y"] <= floor)
    frame["up"] = frame["eligible"] & (frame["y"] >= spike)
    frame["down"] = frame["eligible"] & (frame["y"] <= floor)
    frame["onset"] = frame["eligible"] & frame["tail"]
    frame["rest_of_event"] = ~frame["eligible"] & frame["tail"]
    # Fresh onset: no tail price in the previous hour (all realised before the origin).
    grid = ready["mcp"].reindex(pd.date_range(ready.index.min(), ready.index.max(), freq="5min"))
    grid_tail = ((grid >= spike) | (grid <= floor)).astype(float).where(grid.notna())
    recent = pd.concat([grid_tail.shift(k) for k in range(1, 13)], axis=1).max(axis=1, skipna=True)
    frame["fresh_onset"] = frame["onset"] & recent.reindex(frame.index).eq(0).to_numpy()
    return frame, rule, spike, floor


# --------------------------------------------------------------------------- models


def _xy(frame, features):
    x = frame[features].astype(float)
    x["hour"] = x["hour"].astype(int)
    return x


def check_attached(rows, features, minimum=0.9):
    """Fail loudly if an external feature is mostly missing on the fitting rows.

    The first stage-1 run fitted the pre-dispatch set while the pull was still
    running; its pre-dispatch columns were empty on train, so LightGBM never
    split on them and the row equalled hurdle_weather.
    """
    external = [f for f in features if f.startswith(EXTERNAL_PREFIXES)]
    share = rows[external].notna().mean()
    low = share[share < minimum]
    if not low.empty:
        raise RuntimeError(f"external features mostly missing on fitting rows: {low.round(3).to_dict()}")
    return share


def fit_classifier(train, features, label):
    rows = train.loc[train["eligible"] & (train.index >= FIT_START)]
    check_attached(rows, features)
    fit, val = rows.loc[rows.index < INNER_VAL_START], rows.loc[rows.index >= INNER_VAL_START]
    probe = LGBMClassifier(**CLF_PARAMS)
    probe.fit(_xy(fit, features), fit[label].astype(int), eval_X=(_xy(val, features),), eval_y=(val[label].astype(int),),
              eval_metric="binary_logloss", callbacks=[early_stopping(200, verbose=False)])
    best = max(int(probe.best_iteration_ or 100), 50)
    model = LGBMClassifier(**{**CLF_PARAMS, "n_estimators": best})
    model.fit(_xy(rows, features), rows[label].astype(int))
    return model, best, int(rows[label].sum()), int(len(rows))


def fit_size(train, features, label):
    rows = train.loc[train[label] & (train.index >= FIT_START)]
    if len(rows) < 100:
        rows = train.loc[train[label]]
    models = {}
    for level in [0.5, *SIZE_LEVELS]:
        model = LGBMRegressor(objective="quantile", alpha=float(level), **SIZE_PARAMS)
        model.fit(_xy(rows, features), rows["y"].astype(float))
        models[float(level)] = model
    return models, int(len(rows))


def predict_size(models, frame, features):
    x = _xy(frame, features)
    median = models[0.5].predict(x)
    grid = np.column_stack([models[float(level)].predict(x) for level in SIZE_LEVELS])
    return median, np.sort(grid, axis=1)


# --------------------------------------------------------------------------- scoring


def residual_samples(history: np.ndarray, positions: np.ndarray) -> np.ndarray:
    offsets = np.arange(WINDOW)
    return history[positions[:, None] - WINDOW + offsets]


def weighted_crps(samples: np.ndarray, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact CRPS of a weighted empirical distribution (one row per origin)."""
    order = np.argsort(samples, axis=1)
    x = np.take_along_axis(samples, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    below = np.cumsum(w, axis=1) - w
    pair = 2.0 * np.sum(w * x * (2.0 * below + w - 1.0), axis=1)
    absolute = np.sum(w * np.abs(x - y[:, None]), axis=1)
    return absolute - 0.5 * pair


def mixture_crps(block, history, positions, pi_up, pi_down, plain_crps):
    """CRPS of the regime-switch residual window with weight moved onto the size grids.

    Rows where neither weight is positive keep ``plain_crps``, the regime
    switch's own empirical CRPS, so only gated rows are recomputed.
    """
    y = block["y"].to_numpy(float)
    centre = block["regime"].to_numpy(float)
    out = np.asarray(plain_crps, dtype=float).copy()
    gated = np.flatnonzero((pi_up > 0) | (pi_down > 0))
    for start in range(0, len(gated), 2048):
        g_idx = gated[start:start + 2048]
        if len(g_idx):
            base = centre[g_idx][:, None] + residual_samples(history, positions[g_idx])
            up, down = block["up_grid"].to_numpy()[g_idx], block["down_grid"].to_numpy()[g_idx]
            up, down = np.vstack(up), np.vstack(down)
            k = up.shape[1]
            samples = np.hstack([base, up, down])
            weights = np.hstack([
                np.repeat(((1 - pi_up[g_idx] - pi_down[g_idx]) / WINDOW)[:, None], WINDOW, axis=1),
                np.repeat((pi_up[g_idx] / k)[:, None], k, axis=1),
                np.repeat((pi_down[g_idx] / k)[:, None], k, axis=1),
            ])
            out[g_idx] = weighted_crps(samples, weights, y[g_idx])
    return out


def issued_point(block, c_up, c_down):
    yhat = block["regime"].to_numpy(float).copy()
    up = block["eligible"].to_numpy() & (block["p_up"].to_numpy() >= c_up) & (block["p_up"].to_numpy() >= block["p_down"].to_numpy())
    down = block["eligible"].to_numpy() & ~up & (block["p_down"].to_numpy() >= c_down)
    yhat[up] = block["up_median"].to_numpy()[up]
    yhat[down] = block["down_median"].to_numpy()[down]
    return yhat, up | down


def mixture_weights(block, c_up, c_down):
    elig = block["eligible"].to_numpy()
    p_up, p_down = block["p_up"].to_numpy(), block["p_down"].to_numpy()
    pi_up = np.where(elig & (p_up >= c_up), p_up, 0.0)
    pi_down = np.where(elig & (p_down >= c_down), p_down, 0.0)
    total = pi_up + pi_down
    scale = np.where(total > 0.95, 0.95 / np.maximum(total, 1e-12), 1.0)
    return pi_up * scale, pi_down * scale


def mae(y, yhat, mask=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if mask is not None:
        y, yhat = y[mask], yhat[mask]
    return float(np.mean(np.abs(y - yhat))) if len(y) else np.nan


def detection(pred, block):
    """Precision and recall of calling a band crossing at the first interval of a tail event."""
    elig = block["eligible"].to_numpy()
    truth = block["onset"].to_numpy()[elig]
    called = np.asarray(pred)[elig]
    tp = int(np.sum(called & truth))
    return {
        "onset_events": int(truth.sum()),
        "calls": int(called.sum()),
        "hits": tp,
        "precision": tp / called.sum() if called.sum() else np.nan,
        "recall": tp / truth.sum() if truth.sum() else np.nan,
    }


def summary(name, block, yhat, crps, called):
    y = block["y"].to_numpy(float)
    tail, onset, rest = (block[c].to_numpy() for c in ("tail", "onset", "rest_of_event"))
    row = {
        "model": name,
        "n": int(len(block)),
        "mae": mae(y, yhat),
        "crps": float(np.mean(crps)),
        "tail_mae": mae(y, yhat, tail),
        "tail_crps": float(np.mean(crps[tail])),
        "onset_mae": mae(y, yhat, onset),
        "onset_crps": float(np.mean(crps[onset])),
        "fresh_onset_n": int(block["fresh_onset"].sum()),
        "fresh_onset_mae": mae(y, yhat, block["fresh_onset"].to_numpy()),
        "fresh_onset_crps": float(np.mean(crps[block["fresh_onset"].to_numpy()])),
        "rest_of_event_mae": mae(y, yhat, rest),
        "rest_of_event_crps": float(np.mean(crps[rest])),
    }
    row.update({f"onset_{k}" if not k.startswith("onset") else k: v for k, v in detection(called, block).items()})
    fresh = detection_fresh(called, block)
    row.update({f"fresh_{k}": v for k, v in fresh.items()})
    return row


def detection_fresh(pred, block):
    """Recall on fresh onsets only; precision counts a call as a hit for any onset."""
    elig = block["eligible"].to_numpy()
    called = np.asarray(pred)[elig]
    fresh = block["fresh_onset"].to_numpy()[elig]
    return {"recall": float(np.sum(called & fresh) / fresh.sum()) if fresh.sum() else np.nan}


def block_bootstrap_mean_diff(diff, mask, block=288, n_boot=1000, seed=42):
    """Moving-block bootstrap of mean(diff[mask]) on the time-ordered test rows."""
    diff = np.where(mask, diff, 0.0)
    m = mask.astype(float)
    n = len(diff)
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block + 1)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        idx = (chosen[:, None] + np.arange(block)).ravel()[:n]
        draws[i] = diff[idx].sum() / max(m[idx].sum(), 1.0)
    return float(diff.sum() / m.sum()), float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))


BOOT_SUBSETS = ("all", "tail", "onset", "fresh_onset")


def bootstrap_table(test, per_row):
    """Model minus baseline on test, with 90% moving-block (1 day) intervals.

    Baselines are the regime switch (stage 1) and persistence (stage 1b). Each
    pair uses the same seed, so the regime-switch ranges match stage 1.
    """
    y = test["y"].to_numpy(float)
    out = []
    for baseline in ("regime_switch", "persistence"):
        base_hat, base_crps = per_row[(baseline, "test")]
        for (name, split), (yhat, crps) in per_row.items():
            if split != "test" or name == baseline:
                continue
            for subset in BOOT_SUBSETS:
                mask = np.ones(len(y), bool) if subset == "all" else test[subset].to_numpy()
                for metric, a, b in (("mae", np.abs(y - yhat), np.abs(y - base_hat)), ("crps", crps, base_crps)):
                    mean, lo, hi = block_bootstrap_mean_diff(a - b, mask)
                    out.append({"model": name, "baseline": baseline, "subset": subset, "metric": metric, "diff": mean,
                                "boot_p05": lo, "boot_p95": hi, "n": int(mask.sum())})
    return pd.DataFrame(out)


def point_called(block, yhat, spike, floor):
    return (np.asarray(yhat) >= spike) | (np.asarray(yhat) <= floor)


# --------------------------------------------------------------------------- main


def run(frame, spike, floor):
    train = frame.loc[frame["split"].eq("train")]
    cal_test = frame.loc[frame["split"].isin(["calibration", "test"])].copy()
    history = (cal_test["y"] - cal_test["regime"]).to_numpy(float)
    positions = np.arange(len(cal_test))
    is_cal = cal_test["split"].eq("calibration").to_numpy()
    selectable = is_cal & (positions >= WINDOW)
    cal = cal_test.loc[selectable].copy()
    test = cal_test.loc[~is_cal].copy()
    cal_pos, test_pos = positions[selectable], positions[~is_cal]

    rows, selections, model_meta, detail = [], [], {}, {}
    baselines, plain, per_row = {}, {}, {}
    for name, column in (("persistence", "persistence"), ("lightgbm", "lightgbm"), ("regime_switch", "regime")):
        for split, block, pos in (("calibration", cal, cal_pos), ("test", test, test_pos)):
            hist = (cal_test["y"] - cal_test[column]).to_numpy(float)
            res = residual_samples(hist, pos)
            crps = crps_from_residual_samples(res, block["y"].to_numpy(float), block[column].to_numpy(float))
            yhat = block[column].to_numpy(float)
            summ = summary(name, block, yhat, crps, point_called(block, yhat, spike, floor))
            summ.update(split=split)
            rows.append(summ)
            baselines[(name, split)] = summ
            if name == "regime_switch":
                plain[split] = crps
            per_row[(name, split)] = (yhat, crps)

    for set_name, features in FEATURE_SETS.items():
        clf_up, it_up, pos_up, n_up = fit_classifier(train, features, "up")
        clf_dn, it_dn, pos_dn, n_dn = fit_classifier(train, features, "down")
        size_up, n_size_up = fit_size(train, features, "up")
        size_dn, n_size_dn = fit_size(train, features, "down")
        for block in (cal, test):
            x = _xy(block, features)
            block["p_up"] = clf_up.predict_proba(x)[:, 1]
            block["p_down"] = clf_dn.predict_proba(x)[:, 1]
            block["up_median"], up_grid = predict_size(size_up, block, features)
            block["down_median"], down_grid = predict_size(size_dn, block, features)
            block["up_grid"] = list(up_grid)
            block["down_grid"] = list(down_grid)
        base_cal = baselines[("regime_switch", "calibration")]

        # point cutoffs: lowest calibration tail MAE with overall MAE within 1% of the regime switch
        grid = []
        for c_up in CUTOFFS:
            for c_dn in CUTOFFS:
                yhat, _ = issued_point(cal, c_up, c_dn)
                grid.append({"c_up": c_up, "c_down": c_dn, "cal_mae": mae(cal["y"], yhat),
                             "cal_tail_mae": mae(cal["y"], yhat, cal["tail"].to_numpy()),
                             "cal_onset_mae": mae(cal["y"], yhat, cal["onset"].to_numpy())})
        grid = pd.DataFrame(grid)
        ok = grid.loc[grid["cal_mae"] <= base_cal["mae"] * 1.01]
        best_point = (ok if not ok.empty else grid).sort_values(["cal_tail_mae", "cal_mae"]).iloc[0]

        # mixture cutoffs: lowest calibration tail CRPS with overall CRPS within 1% of the regime switch
        mgrid = []
        for c_up in MIX_CUTOFFS:
            for c_dn in MIX_CUTOFFS:
                pi_u, pi_d = mixture_weights(cal, c_up, c_dn)
                crps = mixture_crps(cal, history, cal_pos, pi_u, pi_d, plain["calibration"])
                mgrid.append({"c_up": c_up, "c_down": c_dn, "cal_crps": float(crps.mean()),
                              "cal_tail_crps": float(crps[cal["tail"].to_numpy()].mean())})
        mgrid = pd.DataFrame(mgrid)
        ok = mgrid.loc[mgrid["cal_crps"] <= base_cal["crps"] * 1.01]
        best_mix = (ok if not ok.empty else mgrid).sort_values(["cal_tail_crps", "cal_crps"]).iloc[0]

        # detection cutoff: best calibration F1 for calling a crossing at an eligible origin
        p_any_cal = np.maximum(cal["p_up"], cal["p_down"]).to_numpy()
        f1s = []
        for c in np.round(np.arange(0.02, 0.96, 0.02), 2):
            d = detection(p_any_cal >= c, cal)
            prec, rec = d["precision"], d["recall"]
            f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) else 0.0
            f1s.append((f1, c))
        f1_best, c_det = max(f1s)

        for split, block, pos in (("calibration", cal, cal_pos), ("test", test, test_pos)):
            yhat, _ = issued_point(block, best_point["c_up"], best_point["c_down"])
            pi_u, pi_d = mixture_weights(block, best_mix["c_up"], best_mix["c_down"])
            crps = mixture_crps(block, history, pos, pi_u, pi_d, plain[split])
            called = np.maximum(block["p_up"], block["p_down"]).to_numpy() >= c_det
            summ = summary(set_name, block, yhat, crps, called)
            elig = block["eligible"].to_numpy()
            truth = block["onset"].to_numpy()[elig]
            p_any = np.maximum(block["p_up"], block["p_down"]).to_numpy()[elig]
            summ.update(
                split=split,
                onset_average_precision=float(average_precision_score(truth, p_any)) if truth.any() else np.nan,
                onset_roc_auc=float(roc_auc_score(truth, p_any)) if truth.any() else np.nan,
                onset_base_rate=float(truth.mean()),
                point_gate_share=float(np.mean(issued_point(block, best_point["c_up"], best_point["c_down"])[1])),
                mixture_gate_share=float(np.mean((pi_u + pi_d) > 0)),
            )
            rows.append(summ)
            per_row[(set_name, split)] = (yhat, crps)
            if split == "test":
                detail[set_name] = pd.DataFrame({"y": block["y"], "yhat": yhat, "crps": crps, "p_up": block["p_up"],
                                                 "p_down": block["p_down"], "onset": block["onset"]}, index=block.index)
        model_meta[set_name] = {
            "features": features,
            "classifier_up": {"iterations": it_up, "positives": pos_up, "rows": n_up},
            "classifier_down": {"iterations": it_dn, "positives": pos_dn, "rows": n_dn},
            "size_rows": {"up": n_size_up, "down": n_size_dn},
            "point_cutoffs": {"up": float(best_point["c_up"]), "down": float(best_point["c_down"])},
            "mixture_cutoffs": {"up": float(best_mix["c_up"]), "down": float(best_mix["c_down"])},
            "detection_cutoff": float(c_det),
            "external_non_null_share": {
                split: {k: round(float(v), 4) for k, v in frame_split[[f for f in features if f.startswith(EXTERNAL_PREFIXES)]].notna().mean().items()}
                for split, frame_split in (("train_fit_rows", train.loc[train["eligible"] & (train.index >= FIT_START)]), ("calibration", cal), ("test", test))
            } if any(f.startswith(EXTERNAL_PREFIXES) for f in features) else {},
            "calibration_detection_f1": float(f1_best),
        }
        grid.insert(0, "model", set_name)
        mgrid.insert(0, "model", set_name)
        selections.append((grid, mgrid))
        importance = pd.Series(clf_up.booster_.feature_importance("gain"), index=features).sort_values(ascending=False)
        model_meta[set_name]["top_up_features_by_gain"] = importance.head(15).round(1).to_dict()
    boot = bootstrap_table(test, per_row)
    return pd.DataFrame(rows), selections, model_meta, detail, boot


def main():
    frame, rule, spike, floor = assemble()
    results, selections, meta, detail, boot = run(frame, spike, floor)
    boot.to_csv(OUT / "spike_stage1_bootstrap.csv", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "spike_stage1_results.csv", index=False)
    pd.concat([g for g, _ in selections]).to_csv(OUT / "spike_stage1_point_cutoffs.csv", index=False)
    pd.concat([m for _, m in selections]).to_csv(OUT / "spike_stage1_mixture_cutoffs.csv", index=False)
    out_meta = {
        "fit_window": f"{FIT_START.date()} to train end; early stopping on {INNER_VAL_START.date()} onwards",
        "train_spike": spike,
        "train_floor": floor,
        "regime_rule": rule[0],
        "selection": "calibration rows after the first conformal window; point: min tail MAE with overall MAE within 1% of the regime switch; mixture: min tail CRPS with overall CRPS within 1% of the regime switch; detection: max F1",
        "models": meta,
    }
    (OUT / "spike_stage1_meta.json").write_text(json.dumps(out_meta, indent=2, default=float), encoding="utf-8")
    cols = ["model", "split", "mae", "crps", "tail_mae", "tail_crps", "onset_mae", "onset_crps", "fresh_onset_mae",
            "fresh_onset_crps", "rest_of_event_mae", "onset_precision", "onset_recall", "fresh_recall", "onset_calls", "onset_events"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(results[cols].round(3).to_string(index=False))
        print(results[[c for c in results.columns if c not in cols or c == "model"]].round(4).to_string(index=False))
    print(boot.round(3).to_string(index=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in meta.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
