"""LightGBM lag-feature baseline against QRA on the frozen test window.

Same data, origins, horizon and splits as scripts/qra.py:
  train        <= 2025-09-30 23:55   (LightGBM is fit here; hyperparameters frozen)
  calibration  <= 2026-03-31 23:55   (QRA is fit here, on the five point forecasts)
  test         2026-04-01 .. 2026-08-18 07:55, 40,127 origins, scored once

QRA quantiles are regenerated from the committed calibration-fitted
coefficients (reports/forecast/qra_coefficients.csv) with no refit, and the
aggregates are checked against the stored qra_test.csv / qra_regime.csv
before anything is scored.

LightGBM baseline ("lag-feature" model):
  * features: the repo's TREE_FEATURES (5/30/60 min and 1 day price lags,
    lagged demand, DPV, withdrawal, FCAS, last complete RTP, STEM, calendar)
    plus extra price lags known at the origin (10, 15, 20, 25 min, 2 h, 7 d)
    and rolling mean / std / max / min of the realised price over the last
    30 and 60 minutes (all ending at the origin, t - 5 min);
  * target: 5-minute price change, as in scripts/point_models.py; the issued
    forecast is the 5-minute lag plus the predicted change;
  * quantile models: one LightGBM quantile model per QRA level (23 levels,
    0.01 .. 0.99), rows sorted afterwards (same rearrangement as QRA);
  * point model: LightGBM under MAE (L1) on the same features;
  * hyperparameters: the frozen point-model values chosen on the inner
    training split (experiment_meta.json); no tuning on calibration or test.
Two fitting windows are reported: train only (the repo convention for
LightGBM) and train + calibration (sees the same recent data QRA is fit on).
Test outcomes are never used to fit or select anything.

Outputs: reports/forecast/lgbm_vs_qra_*.csv|json and
reports/figures/lgbm_vs_qra/*.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment import load_ready  # noqa: E402
from scripts.forecast_design import (  # noqa: E402
    CAL_END, TARGET, TREE_CATEGORICAL, TREE_FEATURES, build_exante_frame, slice_split,
)
from scripts.metrics import crps_from_quantiles, pinball_values  # noqa: E402
from scripts.paths import panel_path  # noqa: E402
from scripts.qra import OUT, QRA_FORECASTS, _saved_point_result, level_name, predict_qra  # noqa: E402

FIG = ROOT / "reports" / "figures" / "lgbm_vs_qra"
N_ESTIMATORS = 250
EXTRA_LAGS = {"mcp_lag_10min": 2, "mcp_lag_15min": 3, "mcp_lag_20min": 4, "mcp_lag_25min": 5,
              "mcp_lag_2h": 24, "mcp_lag_7d": 2016}
ROLL_WINDOWS = {"30min": 6, "60min": 12}
INTERVALS = (0.5, 0.8, 0.9, 0.95)


def lag_features_from_grid(grid: pd.Series) -> pd.DataFrame:
    """Extra price lags and rolling stats on a complete 5-minute grid of MCP.

    Row t uses prices up to t - 5 min only (the forecast origin)."""
    out = pd.DataFrame(index=grid.index)
    for name, steps in EXTRA_LAGS.items():
        out[name] = grid.shift(steps)
    known = grid.shift(1)  # last realised price at the origin
    for label, w in ROLL_WINDOWS.items():
        roll = known.rolling(w, min_periods=w)
        out[f"mcp_roll_mean_{label}"] = roll.mean()
        out[f"mcp_roll_std_{label}"] = roll.std()
        out[f"mcp_roll_max_{label}"] = roll.max()
        out[f"mcp_roll_min_{label}"] = roll.min()
    return out


def lag_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """lag_features_from_grid on the panel's complete 5-minute grid, aligned to `index`."""
    raw = pd.read_parquet(panel_path(), engine="pyarrow")
    grid = build_exante_frame(raw)["mcp"]
    return lag_features_from_grid(grid).reindex(index)


def _x(frame: pd.DataFrame, features: list[str], categories: dict | None = None):
    x = frame[features].copy()
    cats = {}
    for col in TREE_CATEGORICAL:
        if categories is None:
            x[col] = x[col].astype(int).astype("category")
            cats[col] = list(x[col].cat.categories)
        else:
            x[col] = pd.Categorical(x[col].astype(int), categories=categories[col])
    return x, (cats if categories is None else categories)


def _model(objective: str, params: dict, alpha: float | None = None):
    from lightgbm import LGBMRegressor
    extra = {"alpha": alpha} if alpha is not None else {}
    return LGBMRegressor(objective=objective, n_estimators=N_ESTIMATORS, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
                         n_jobs=-1, **params, **extra)


def fit_predict(fit: pd.DataFrame, test: pd.DataFrame, features: list[str], levels, params) -> dict:
    x_fit, cats = _x(fit, features)
    x_test, _ = _x(test, features, cats)
    change = fit[TARGET].to_numpy(float) - fit["mcp_lag_5min"].to_numpy(float)
    lag = test["mcp_lag_5min"].to_numpy(float)
    raw = np.empty((len(test), len(levels)))
    for j, level in enumerate(levels):
        t0 = time.time()
        m = _model("quantile", params, float(level))
        m.fit(x_fit, change, categorical_feature=TREE_CATEGORICAL)
        raw[:, j] = lag + m.predict(x_test)
        print(f"  q{level:.3f} fitted in {time.time() - t0:.1f}s", flush=True)
    crossed = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
    q = pd.DataFrame(np.sort(raw, axis=1), index=test.index, columns=[level_name(l) for l in levels])
    point_model = _model("regression_l1", params)
    point_model.fit(x_fit, change, categorical_feature=TREE_CATEGORICAL)
    point = pd.Series(lag + point_model.predict(x_test), index=test.index)
    gain = pd.Series(point_model.booster_.feature_importance("gain"), index=features).sort_values(ascending=False)
    return {"quantiles": q, "point": point, "crossed_share": crossed, "point_gain": gain}


def block_bootstrap_diff(diff: np.ndarray, block: int = 288, n_boot: int = 400, seed: int = 42) -> tuple[float, float]:
    """90% moving-block bootstrap range for the mean of a per-row difference."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        draws[i] = np.concatenate([diff[s:s + block] for s in chosen])[:n].mean()
    return float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))


def score(name: str, y: np.ndarray, q: pd.DataFrame, point: np.ndarray, levels, masks: dict) -> tuple[list, list, np.ndarray]:
    grid = q.to_numpy(float)
    pin = np.column_stack([pinball_values(y, grid[:, j], lv) for j, lv in enumerate(levels)])
    crps = crps_from_quantiles(y, grid, levels)
    summary, per_level = [], []
    for subset, m in masks.items():
        row = {"model": name, "subset": subset, "n": int(m.sum()),
               "avg_pinball": float(pin[m].mean()), "crps_quantile_integral": float(crps[m].mean()),
               "mae_point": float(np.mean(np.abs(y[m] - point[m]))),
               "mae_median_q": float(np.mean(np.abs(y[m] - q[level_name(0.5)].to_numpy()[m])))}
        for nominal in INTERVALS:
            a = (1 - nominal) / 2
            lo, hi = q[level_name(a)].to_numpy()[m], q[level_name(1 - a)].to_numpy()[m]
            row[f"coverage_{int(nominal*100)}"] = float(np.mean((y[m] >= lo) & (y[m] <= hi)))
            row[f"width_{int(nominal*100)}"] = float(np.mean(hi - lo))
        summary.append(row)
        for j, lv in enumerate(levels):
            per_level.append({"model": name, "subset": subset, "level": float(lv), "pinball": float(pin[m, j].mean())})
    return summary, per_level, crps


def main() -> None:
    t_start = time.time()
    meta_exp = json.loads((OUT / "experiment_meta.json").read_text())
    params = dict(meta_exp["lgbm_params"])
    print("rebuilding point forecasts (as scripts/qra.py)", flush=True)
    point, _, _ = _saved_point_result()
    test = point["test"]
    y = test[TARGET].to_numpy(float)

    # --- QRA: regenerate from committed coefficients, verify, no refit ---
    fc = pd.DataFrame({n: point["predictions"][n].reindex(test.index) for n in QRA_FORECASTS})
    coef = pd.read_csv(OUT / "qra_coefficients.csv")
    qra_q, _ = predict_qra(fc, coef)
    levels = np.array([float(c) for c in qra_q.columns])
    stored = pd.read_csv(OUT / "qra_test.csv").query("method=='qra' and alpha==0.1").iloc[0]
    stored_crps = pd.read_csv(OUT / "qra_crps.csv").query("method=='qra'").iloc[0]["crps_quantile_integral"]
    lo, hi = qra_q[level_name(0.05)].to_numpy(), qra_q[level_name(0.95)].to_numpy()
    check = {"coverage_90": float(np.mean((y >= lo) & (y <= hi))), "width_90": float(np.mean(hi - lo)),
             "crps_qi": float(crps_from_quantiles(y, qra_q.to_numpy(float), levels).mean())}
    for k, v in {"coverage_90": stored["coverage"], "width_90": stored["mean_width"], "crps_qi": stored_crps}.items():
        if abs(check[k] - v) > 1e-6 * max(1.0, abs(v)):
            raise SystemExit(f"QRA regeneration mismatch on {k}: {check[k]} vs stored {v}")
    print("QRA regenerated and verified against stored tables", check, flush=True)

    # --- spike flags: same rows as the spike/CPS evaluation ---
    rows = pd.read_parquet(OUT / "final_cps_test_rows.parquet")
    assert rows.index.equals(test.index) and np.allclose(rows["y"].to_numpy(float), y)
    masks = {"all": np.ones(len(y), bool), "body": ~rows["tail"].to_numpy(bool),
             "spike (tail)": rows["tail"].to_numpy(bool), "spike onset": rows["onset"].to_numpy(bool)}

    # --- LightGBM lag-feature baseline ---
    ready = load_ready()
    extra = lag_features(ready.index)
    ready = ready.join(extra)
    features = list(TREE_FEATURES) + list(extra.columns)
    train = slice_split(ready, "train")
    train_cal = ready.loc[ready.index <= CAL_END]
    test_x = ready.loc[test.index]
    assert len(test_x) == len(test)
    fits = {}
    for label, fit in (("LightGBM lag (train)", train), ("LightGBM lag (train+cal)", train_cal)):
        print(f"fitting {label}: {len(fit)} rows, {len(features)} features", flush=True)
        fits[label] = fit_predict(fit, test_x, features, levels, params)

    # --- score ---
    models = {
        "QRA (5 forecasts, cal-fit)": (qra_q, point["predictions"]["lightgbm"].reindex(test.index).to_numpy(float)),
    }
    for label, res in fits.items():
        models[label] = (res["quantiles"], res["point"].to_numpy(float))
    summary, per_level, crps_rows = [], [], {}
    for name, (q, pt) in models.items():
        s, p, c = score(name, y, q, pt, levels, masks)
        summary += s
        per_level += p
        crps_rows[name] = c
    # QRA has no separate point model; report its median as the point and note it.
    summary = pd.DataFrame(summary)
    qmask = summary["model"].str.startswith("QRA")
    summary.loc[qmask, "mae_point"] = summary.loc[qmask, "mae_median_q"]
    persistence = point["predictions"]["persistence_5min"].reindex(test.index).to_numpy(float)
    repo_lgbm = point["predictions"]["lightgbm"].reindex(test.index).to_numpy(float)
    ref_mae = []
    for subset, m in masks.items():
        ref_mae.append({"subset": subset, "persistence_mae": float(np.mean(np.abs(y[m] - persistence[m]))),
                        "repo_lightgbm_point_mae": float(np.mean(np.abs(y[m] - repo_lgbm[m])))})
    ref_mae = pd.DataFrame(ref_mae)

    boot = []
    base = crps_rows["QRA (5 forecasts, cal-fit)"]
    for name in fits:
        for subset, m in masks.items():
            d = crps_rows[name][m] - base[m]
            lo_b, hi_b = block_bootstrap_diff(d, block=288 if subset in ("all", "body") else 12)
            boot.append({"model": name, "subset": subset, "crps_diff_vs_qra": float(d.mean()),
                         "boot_p05": lo_b, "boot_p95": hi_b})
    boot = pd.DataFrame(boot)
    mae_boot = []
    qra_med = qra_q[level_name(0.5)].to_numpy(float)
    for name, res in fits.items():
        pt = res["point"].to_numpy(float)
        for ref_name, ref in (("persistence", persistence), ("QRA median", qra_med), ("repo LightGBM point", repo_lgbm)):
            for subset, m in masks.items():
                d = np.abs(y[m] - pt[m]) - np.abs(y[m] - ref[m])
                lo_b, hi_b = block_bootstrap_diff(d, block=288 if subset in ("all", "body") else 12)
                mae_boot.append({"model": name, "reference": ref_name, "subset": subset,
                                 "mae_diff": float(d.mean()), "boot_p05": lo_b, "boot_p95": hi_b})
    mae_boot = pd.DataFrame(mae_boot)
    mae_boot.to_csv(OUT / "lgbm_vs_qra_mae_bootstrap.csv", index=False)

    per_level = pd.DataFrame(per_level)
    summary.to_csv(OUT / "lgbm_vs_qra_summary.csv", index=False)
    per_level.to_csv(OUT / "lgbm_vs_qra_pinball_by_level.csv", index=False)
    ref_mae.to_csv(OUT / "lgbm_vs_qra_reference_mae.csv", index=False)
    boot.to_csv(OUT / "lgbm_vs_qra_bootstrap.csv", index=False)
    gain = fits["LightGBM lag (train)"]["point_gain"]
    gain.rename("gain").to_csv(OUT / "lgbm_vs_qra_point_feature_gain.csv", index_label="feature")
    preds = pd.DataFrame({"y": y, "tail": masks["spike (tail)"], "onset": masks["spike onset"]}, index=test.index)
    for name, (q, pt) in models.items():
        tag = {"QRA (5 forecasts, cal-fit)": "qra", "LightGBM lag (train)": "lgbm_train",
               "LightGBM lag (train+cal)": "lgbm_traincal"}[name]
        for lv in (0.05, 0.5, 0.95):
            preds[f"{tag}_q{lv:.2f}"] = q[level_name(lv)].to_numpy()
        preds[f"{tag}_point"] = pt
    preds.to_parquet(OUT / "lgbm_vs_qra_test_forecasts.parquet")
    meta = {
        "test_window": [str(test.index.min()), str(test.index.max())], "n_test": int(len(test)),
        "levels": [float(l) for l in levels], "lgbm_params": {**params, "n_estimators": N_ESTIMATORS},
        "features": features, "qra_source": "reports/forecast/qra_coefficients.csv (calibration fit, not refit)",
        "qra_check": check, "qra_point": "QRA median (q0.50); QRA has no separate point model",
        "lgbm_point": "LightGBM regression_l1 on the 5-minute change, same features",
        "crossed_share_before_sort": {k: v["crossed_share"] for k, v in fits.items()},
        "spike_definition": "tail: y >= train p95 (170.32 $/MWh) or <= train p05; onset: tail row whose last realised price was not in the tail (final_cps_test_rows.parquet)",
        "crps_note": "quantile-integral CRPS over 0.01-0.99, same estimator as qra_crps.csv",
        "bootstrap": "moving-block, 400 draws, block 288 rows (1 day) for all/body, 12 rows for tail/onset subsets",
        "runtime_s": round(time.time() - t_start, 1),
    }
    (OUT / "lgbm_vs_qra_meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print(ref_mae.round(3).to_string(index=False))
    print(boot.round(3).to_string(index=False))
    print(mae_boot.round(3).to_string(index=False))
    figures()


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUT / "lgbm_vs_qra_summary.csv")
    per_level = pd.read_csv(OUT / "lgbm_vs_qra_pinball_by_level.csv")
    colors = {"QRA (5 forecasts, cal-fit)": "#1f4e79", "LightGBM lag (train)": "#e07b39",
              "LightGBM lag (train+cal)": "#7aa95c"}

    # 1. pinball loss by quantile level (all test rows and spike rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, subset in zip(axes, ("all", "spike (tail)")):
        for name, g in per_level[per_level["subset"] == subset].groupby("model", sort=False):
            ax.plot(g["level"], g["pinball"], marker="o", ms=3, label=name, color=colors[name])
        ax.set_title(f"Pinball loss by quantile level - {'all test origins' if subset == 'all' else 'spike intervals'}")
        ax.set_xlabel("quantile level")
        ax.set_ylabel("mean pinball loss ($/MWh)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Test window 1 Apr - 18 Aug 2026 (lower is better)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "pinball_by_level.png", dpi=150)
    plt.close(fig)

    # 2. interval coverage vs nominal, and width (all vs spike)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    noms = [50, 80, 90, 95]
    for ax, subset in zip(axes, ("all", "spike (tail)")):
        s = summary[summary["subset"] == subset].set_index("model")
        x = np.arange(len(noms))
        for k, name in enumerate(colors):
            ax.bar(x + (k - 1) * 0.27, [s.loc[name, f"coverage_{n}"] for n in noms], width=0.27,
                   color=colors[name], label=name)
        ax.scatter(x, [n / 100 for n in noms], marker="_", s=900, color="black", zorder=5, label="nominal")
        ax.set_xticks(x, [f"{n}%" for n in noms])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("empirical coverage")
        ax.set_title("Interval coverage - " + ("all test origins" if subset == "all" else "spike intervals"))
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "interval_coverage.png", dpi=150)
    plt.close(fig)

    # 3. relative change vs QRA by period: average pinball and point MAE
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    subsets = ["all", "body", "spike (tail)", "spike onset"]
    names = {"all": "all origins", "body": "normal", "spike (tail)": "spike intervals", "spike onset": "spike onsets"}
    qra = summary[summary["model"] == "QRA (5 forecasts, cal-fit)"].set_index("subset")
    for ax, metric, label in zip(axes, ("avg_pinball", "mae_point"), ("average pinball loss", "point MAE")):
        x = np.arange(len(subsets))
        for k, name in enumerate([n for n in colors if not n.startswith("QRA")]):
            s = summary[summary["model"] == name].set_index("subset")
            vals = [100 * (s.loc[sub, metric] / qra.loc[sub, metric] - 1) for sub in subsets]
            bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.36, color=colors[name], label=name)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center",
                            va="top" if v < 0 else "bottom", fontsize=7, xytext=(0, -2 if v < 0 else 2),
                            textcoords="offset points")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [names[sub] for sub in subsets])
        ax.set_ylabel(f"% change in {label} vs QRA")
        ax.set_title(f"{label[0].upper() + label[1:]} relative to QRA (negative = better)")
        ax.grid(axis="y", alpha=0.3)
        lo = min(ax.get_ylim()[0], -30)
        ax.set_ylim(lo * 1.15, max(5, ax.get_ylim()[1]))
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Test window 1 Apr - 18 Aug 2026; QRA point = QRA median", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "relative_to_qra_by_period.png", dpi=150)
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures()
    else:
        main()
