"""Re-choose the regime-switch threshold inside TRAINING data only (Casey comment #2), then rebuild the CPS.

Same candidate grid and criterion as scripts/regime_switch.py, moved from calibration to the inner
validation window at the end of train (2025-07-01 .. 2025-09-30):
  * LightGBM for the selection is refit with the frozen hyperparameters on train rows before 2025-07-01
    (the same inner fit used to choose those hyperparameters), so its July-September predictions are
    out of sample. It must reproduce the stored inner MAE (experiment_meta.json lgbm_inner_mae).
  * Candidate thresholds: q50/70/80/90/95 of the 30- and 60-minute absolute moves in the FIRST half of
    the window, plus the "already in train tail" rule (train 5th/95th percentile band).
  * Selection: lowest tail MAE (tail = last price outside the band) on the SECOND half, subject to MAE within
    1% of LightGBM on that half.
No calibration or test row is read for the choice. Then the saved LightGBM (fitted on all train, unchanged)
and the new rule give regime-switch forecasts for calibration and test. CPS = new regime + all calibration
residuals (scripts/spike_cps.SplitCPS, tau 0.5). Test is scored once.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.forecast_design import INNER_VAL_START, TARGET, TREE_CATEGORICAL, slice_split  # noqa: E402
from scripts.point_models import _lgbm, _price_change, predict_lgbm  # noqa: E402
from scripts.regime_switch import FLOOR_Q, SPIKE_Q, forecasts, load, mask_for, select_rule, switched  # noqa: E402
from scripts.spike_cps import SplitCPS, interval_score  # noqa: E402
from scripts.spike_onset import block_bootstrap_mean_diff  # noqa: E402

OUT = ROOT / "reports" / "forecast"


def inner_lightgbm(train, saved):
    meta = json.loads((OUT / "experiment_meta.json").read_text())
    params, feats = meta["lgbm_params"], saved["features"]
    fit = train.loc[train.index < INNER_VAL_START]
    x = fit[feats].copy()
    for col in TREE_CATEGORICAL:
        x[col] = x[col].astype(int).astype("category")
    model = _lgbm(params, 250)
    model.fit(x, _price_change(fit, TARGET, "mcp_lag_5min"), categorical_feature=TREE_CATEGORICAL)
    bundle = {"model": model, "lag_column": "mcp_lag_5min", "categories": {c: list(x[c].cat.categories) for c in TREE_CATEGORICAL}}
    return bundle, feats, meta["lgbm_inner_mae"]


def train_rules(train, val):
    spike = float(train[TARGET].quantile(SPIKE_Q)); floor = float(train[TARGET].quantile(FLOOR_Q))
    cut = val.index[len(val) // 2]
    early = val.loc[val.index < cut]
    rules = [("already_in_train_tail", "train_p05_p95", np.nan, spike, floor)]
    for col, name in (("move_30", "abs_30min_move"), ("move_60", "abs_60min_move")):
        for q in (0.50, 0.70, 0.80, 0.90, 0.95):
            rules.append((f"{name}_q{int(q * 100)}", col, float(early[col].quantile(q)), spike, floor))
    return rules, spike, floor, cut


def main():
    ready, saved = load()
    train = slice_split(ready, "train")
    inner, feats, stored_inner = inner_lightgbm(train, saved)
    val_block = train.loc[train.index >= INNER_VAL_START]
    val = forecasts(val_block, inner, feats)
    inner_mae = float(np.mean(np.abs(val["y"] - val["lightgbm"])))
    assert abs(inner_mae - stored_inner) < 1e-6, (inner_mae, stored_inner)
    rules, spike, floor, cut = train_rules(train, val)
    chosen, table, base = select_rule(val, rules, cut)
    # The earlier calibration-chosen rule, kept here so this one-off comparison stays reproducible after
    # scripts/regime_switch.py adopted the train-chosen rule (the stored CPS rows now use the new rule).
    old = {"rule": "abs_30min_move_q90", "feature": "move_30", "threshold": 32.483,
           "selection": "late calibration tail MAE, subject to late MAE within 1% of LightGBM"}
    old_rule = (old["rule"], old["feature"], old["threshold"], spike, floor)

    cal = forecasts(slice_split(ready, "calibration"), saved["bundle"], saved["features"])
    test = forecasts(slice_split(ready, "test"), saved["bundle"], saved["features"])
    rows = pd.read_parquet(OUT / "final_cps_test_rows.parquet")
    assert rows.index.equals(test.index)
    res = {}
    for tag, rule in (("old", old_rule), ("new", chosen)):
        rc, rt = switched(cal, mask_for(cal, rule)), switched(test, mask_for(test, rule))
        cps = SplitCPS(cal["y"].to_numpy() - rc)
        y = test["y"].to_numpy()
        lo, hi = cps.interval(rt)
        res[tag] = {"regime_test": rt, "crps": cps.crps(y, rt), "lo": lo, "hi": hi, "pit": cps.cdf(y, rt),
                    "is": interval_score(y, lo, hi), "switch_share_test": float(np.mean(mask_for(test, rule))),
                    "switch_share_cal": float(np.mean(mask_for(cal, rule)))}
    # the "new" rebuild must reproduce the stored CPS (scripts/spike_cps.py after the switch); before the switch the
    # "old" rebuild reproduced the stored CPS exactly (checked when this script was first run, commit c1f2da1)
    assert np.allclose(res["new"]["regime_test"], rows["regime"]) and np.allclose(res["new"]["lo"], rows["cps_split_lo"])
    assert np.allclose(res["new"]["crps"], rows["cps_split_crps"])
    q = pd.read_parquet(OUT / "qra_test_forecasts.parquet")
    y = test["y"].to_numpy(); a = 0.1
    q_is = interval_score(y, q["q0.050"].to_numpy(), q["q0.950"].to_numpy())
    masks = {"all": np.ones(len(y), bool), "tail": rows["tail"].to_numpy(bool), "onset": rows["onset"].to_numpy(bool)}

    def summary(tag):
        r = res[tag]; cov = (y >= r["lo"]) & (y <= r["hi"]); out = {"model": f"CPS ({tag} threshold)"}
        out["mae_regime"] = float(np.mean(np.abs(y - r["regime_test"])))
        for k, m in masks.items():
            out[f"crps_{k}"] = float(r["crps"][m].mean()); out[f"coverage90_{k}"] = float(cov[m].mean())
            out[f"width90_{k}"] = float((r["hi"] - r["lo"])[m].mean()); out[f"interval_score_{k}"] = float(r["is"][m].mean())
        out["persistence_share_test"] = r["switch_share_test"]; return out
    qcov = (y >= q["q0.050"].to_numpy()) & (y <= q["q0.950"].to_numpy())
    qrow = {"model": "QRA (unchanged; full-distribution flat-tail CRPS)", "mae_regime": np.nan}
    for k, m in masks.items():
        qrow[f"crps_{k}"] = float(q["crps_full"].to_numpy()[m].mean()); qrow[f"coverage90_{k}"] = float(qcov[m].mean())
        qrow[f"width90_{k}"] = float((q["q0.950"] - q["q0.050"]).to_numpy()[m].mean()); qrow[f"interval_score_{k}"] = float(q_is[m].mean())
    qrow["crps_all_quantile_integral"] = float(q["crps_quantile_integral"].mean())
    qrow["persistence_share_test"] = np.nan
    table_out = pd.DataFrame([summary("old"), summary("new"), qrow])
    boot = []
    for k in ("all", "tail", "onset"):
        for qname, qc in (("QRA full", q["crps_full"].to_numpy()), ("QRA quantile-integral", q["crps_quantile_integral"].to_numpy())):
            d, lo_, hi_ = block_bootstrap_mean_diff(res["new"]["crps"] - qc, masks[k])
            boot.append({"diff": f"new CPS - {qname}", "metric": "crps", "subset": k, "mean": d, "boot_p05": lo_, "boot_p95": hi_})
        d, lo_, hi_ = block_bootstrap_mean_diff(res["new"]["crps"] - res["old"]["crps"], masks[k])
        boot.append({"diff": "new CPS - old CPS", "metric": "crps", "subset": k, "mean": d, "boot_p05": lo_, "boot_p95": hi_})
    boot = pd.DataFrame(boot)
    table.to_csv(OUT / "regime_threshold_train_selection.csv", index=False)
    table_out.to_csv(OUT / "regime_threshold_train_comparison.csv", index=False)
    boot.to_csv(OUT / "regime_threshold_train_bootstrap.csv", index=False)
    pd.DataFrame({"y": y, "regime_new": res["new"]["regime_test"], "cps_new_lo": res["new"]["lo"], "cps_new_hi": res["new"]["hi"],
                  "cps_new_crps": res["new"]["crps"], "cps_new_pit": res["new"]["pit"], "tail": masks["tail"], "onset": masks["onset"]},
                 index=test.index).to_parquet(OUT / "regime_threshold_train_test_rows.parquet")
    meta = {"old_rule": old["rule"], "old_threshold": old["threshold"], "old_selection": old["selection"],
            "new_rule": chosen[0], "new_feature": chosen[1], "new_threshold": None if np.isnan(chosen[2]) else float(chosen[2]),
            "selection_window": "train inner validation 2025-07-01..2025-09-30; grid quantiles from first half, choice on second half",
            "selection_cut": str(cut), "selection_lightgbm": "refit with frozen params on train before 2025-07-01 (reproduces stored inner MAE)",
            "inner_lightgbm_mae": inner_mae, "selection_half_lightgbm_mae": base,
            "switch_share_cal_new": res["new"]["switch_share_cal"], "switch_share_cal_old": res["old"]["switch_share_cal"],
            "n_cal_residuals": int(len(cal)), "n_test": int(len(y)), "n_onset": int(masks["onset"].sum()),
            "old_rebuild_reproduces_stored_cps": True}
    (OUT / "regime_threshold_train_meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250)
    print(table.to_string(index=False)); print(json.dumps(meta, indent=2)); print(table_out.T.to_string()); print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
