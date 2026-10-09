"""Master comparison tables and paper figures for the final write-up.

Inputs (all already on disk; nothing is refitted here):
  reports/forecast/final_cps_metrics.csv, final_cps_bootstrap.csv, final_isotonic.csv, final_reliability.csv,
  final_cps_test_rows.parquet (scripts/spike_cps.py); spike_stage1_results.csv (scripts/spike_onset.py);
  conformal_test.csv, qra_regime.csv, qra_crps.csv (PR 2: scripts/experiment.py, scripts/qra.py).
Outputs:
  reports/forecast/final_master_table.csv and final_master_table.md; figures in reports/figures/final/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "forecast"
FIG = ROOT / "reports" / "figures" / "final"
CONFIG = ROOT / "configs" / "spike_forecaster.json"

LABELS = {
    "persistence": "Persistence + sliding residual window",
    "lightgbm_conformal": "LightGBM + absolute sliding conformal (frozen)",
    "lightgbm_normalized": "LightGBM + normalised sliding conformal",
    "lightgbm_fixed_split": "LightGBM + fixed split conformal",
    "qra": "QRA (calibration-fitted)",
    "regime_switch": "Regime switch + sliding residual window",
    "hurdle_panel": "Hurdle, panel only (LightGBM)",
    "hurdle_weather": "Hurdle + weather",
    "hurdle_weather_predispatch": "Hurdle + weather + pre-dispatch (even-hour)",
    "hurdle_weather_predispatch_all": "Hurdle + weather + pre-dispatch (hourly, revisions)",
    "hurdle_weather_predispatch_half": "Hurdle + weather + pre-dispatch (half-hourly, revisions)",
    "hurdle_panel_path": "Hurdle, panel + price path (LightGBM)",
    "hurdle_panel_path_xgboost": "Hurdle, panel + path (XGBoost)",
    "hurdle_panel_path_hist_gb": "Hurdle, panel + path (HistGradientBoosting)",
    "hurdle_panel_path_logistic": "Hurdle, panel + path (logistic)",
    "hurdle_panel_path_lgbm_wide": "Hurdle, panel + path (LightGBM, uncapped)",
    "hurdle_panel_path_hist_gb_wide": "Hurdle, panel + path (HGB, wide grid)",
    "spike_forecaster": "Spike forecaster (frozen)",
    "cps_split": "CPS: split, regime-switch point",
    "cps_mondrian": "CPS: Mondrian by spike risk (fixed calibration)",
    "cps_mondrian_sliding": "CPS: Mondrian by spike risk (sliding windows)",
    "cps_pit": "CPS: PIT-recalibrated spike forecaster",
    "cps_pit_mondrian": "CPS: PIT-recalibrated, Mondrian by spike risk",
}
SHORT = {
    "persistence": "Persistence", "lightgbm_conformal": "LightGBM + conformal", "regime_switch": "Regime switch",
    "hurdle_panel": "Hurdle (panel)", "spike_forecaster": "Spike forecaster", "cps_split": "CPS split",
    "cps_mondrian": "CPS Mondrian (fixed)", "cps_mondrian_sliding": "CPS Mondrian (sliding)",
    "cps_pit": "CPS PIT", "cps_pit_mondrian": "CPS PIT Mondrian",
}
COLORS = {
    "persistence": "#7f7f7f", "lightgbm_conformal": "#bcbd22", "regime_switch": "#ff7f0e", "hurdle_panel": "#8c564b",
    "spike_forecaster": "#1f77b4", "cps_split": "#9edae5", "cps_mondrian": "#c5b0d5", "cps_mondrian_sliding": "#9467bd",
    "cps_pit": "#2ca02c", "cps_pit_mondrian": "#98df8a",
}
STYLE = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
         "savefig.bbox": "tight"}


# --------------------------------------------------------------------------- tables


def master_table() -> pd.DataFrame:
    cps = pd.read_csv(OUT / "final_cps_metrics.csv")
    rows = []
    for (model, split), g in cps.groupby(["model", "split"]):
        g = g.set_index("regime")
        rows.append({"model": model, "split": split, "crps": g.loc["all", "crps"], "tail_crps": g.loc["tail", "crps"],
                     "onset_crps": g.loc["onset", "crps"], "coverage_90": g.loc["all", "coverage_90"],
                     "width_90": g.loc["all", "width_90"], "tail_coverage_90": g.loc["tail", "coverage_90"],
                     "tail_width_90": g.loc["tail", "width_90"], "onset_coverage_90": g.loc["onset", "coverage_90"],
                     "onset_width_90": g.loc["onset", "width_90"], "interval_score_90": g.loc["all", "interval_score_90"],
                     "onset_interval_score_90": g.loc["onset", "interval_score_90"],
                     "mae": g.loc["all", "mae"] if not model.startswith("cps_pit") else np.nan,
                     "source": "scripts/spike_cps.py (same rows and metric code)"})
    stage = pd.read_csv(OUT / "spike_stage1_results.csv")
    for _, r in stage.loc[stage["model"].str.startswith("hurdle")].iterrows():
        rows.append({"model": r["model"], "split": r["split"], "crps": r["crps"], "tail_crps": r["tail_crps"],
                     "onset_crps": r["onset_crps"], "mae": r["mae"],
                     "source": "spike_stage1_results.csv (same rows and CRPS code; intervals not stored)"})
    conf = pd.read_csv(OUT / "conformal_test.csv")
    conf = conf.loc[conf["alpha"].eq(0.1) & conf["base"].eq("lightgbm")]
    for method, name in (("normalized", "lightgbm_normalized"), ("fixed_split", "lightgbm_fixed_split")):
        r = conf.loc[conf["method"].eq(method)].iloc[0]
        rows.append({"model": name, "split": "test", "coverage_90": r["coverage"], "width_90": r["mean_width"],
                     "source": f"conformal_test.csv (PR 2; same 40,127 test rows; interval only)"})
    qra = pd.read_csv(OUT / "qra_regime.csv")
    q = qra.loc[qra["method"].eq("qra")].set_index("regime")
    rows.append({"model": "qra", "split": "test", "crps": q.loc["all", "crps_quantile_integral"],
                 "tail_crps": q.loc["tail", "crps_quantile_integral"], "coverage_90": q.loc["all", "coverage"],
                 "width_90": q.loc["all", "mean_width"], "tail_coverage_90": q.loc["tail", "coverage"],
                 "tail_width_90": q.loc["tail", "mean_width"],
                 "source": "qra_regime.csv (PR 2; same test rows; CRPS is the quantile integral over levels 0.01 to 0.99, "
                           "not the empirical CRPS; LightGBM conformal scores 4.004 overall and 15.98 tail on that measure)"})
    table = pd.DataFrame(rows)
    order = list(LABELS)
    table["order"] = table["model"].map({m: i for i, m in enumerate(order)})
    table["label"] = table["model"].map(LABELS)
    return table.sort_values(["split", "order"]).drop(columns="order")


def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def markdown(table: pd.DataFrame) -> str:
    out = []
    cols = [("crps", 3), ("tail_crps", 2), ("onset_crps", 2), ("coverage_90", 3), ("width_90", 1),
            ("tail_coverage_90", 3), ("tail_width_90", 1), ("onset_coverage_90", 3), ("onset_width_90", 1),
            ("interval_score_90", 1), ("mae", 3)]
    head = ("| Model | CRPS | Tail CRPS | 1st-int CRPS | Cov 90 | Width 90 | Tail cov | Tail width | Onset cov | "
            "Onset width | Interval score 90 | MAE |")
    for split in ("test", "calibration"):
        t = table.loc[table["split"].eq(split)]
        out.append(f"#### {split.capitalize()}\n")
        out.append(head)
        out.append("| --- |" + " ---: |" * len(cols))
        for _, r in t.iterrows():
            out.append("| " + r["label"] + " | " + " | ".join(fmt(r.get(c), d) for c, d in cols) + " |")
        out.append("")
    return "\n".join(out)


def bootstrap_markdown() -> str:
    b = pd.read_csv(OUT / "final_cps_bootstrap.csv")
    b = b.loc[b["subset"].isin(["all", "tail", "onset"])]
    lines = []
    for base in ("spike_forecaster", "persistence"):
        lines.append(f"Against {'the spike forecaster' if base == 'spike_forecaster' else 'persistence'} (test):\n")
        lines.append("| Model | CRPS | Tail CRPS | First-interval CRPS |")
        lines.append("| --- | ---: | ---: | ---: |")
        for model in [m for m in LABELS if m in set(b["model"])]:
            g = b.loc[b["model"].eq(model) & b["baseline"].eq(base)].set_index("subset")
            if g.empty:
                continue
            cell = lambda s: f"{g.loc[s, 'diff']:+.3f} [{g.loc[s, 'boot_p05']:+.3f}, {g.loc[s, 'boot_p95']:+.3f}]"
            lines.append(f"| {LABELS[model]} | {cell('all')} | {cell('tail')} | {cell('onset')} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    return FIG / f"{name}.png"


def fig_master(table):
    models = ["persistence", "lightgbm_conformal", "regime_switch", "hurdle_panel", "spike_forecaster", "cps_split",
              "cps_mondrian", "cps_mondrian_sliding", "cps_pit", "cps_pit_mondrian"]
    t = table.loc[table["split"].eq("test")].set_index("model").loc[models]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, (("crps", "CRPS (all intervals)"), ("tail_crps", "Tail CRPS"),
                                       ("onset_crps", "First-interval CRPS (403 onsets)"))):
        vals = t[col].to_numpy(float)
        ypos = np.arange(len(models))[::-1]
        ax.barh(ypos, vals, color=[COLORS[m] for m in models])
        for yv, v in zip(ypos, vals):
            ax.text(v, yv, f" {v:.2f}", va="center", fontsize=7)
        ax.set_title(title)
        ax.set_xlim(0, np.nanmax(vals) * 1.22)
        ax.set_yticks(np.arange(len(models))[::-1], [SHORT[m] for m in models])
        ax.set_xlabel("$/MWh (lower is better)")
    fig.suptitle("Test window (2026-04-01 to 2026-08-18): probabilistic scores", fontsize=10)
    fig.tight_layout()
    return save(fig, "master_comparison")


def fig_coverage():
    cps = pd.read_csv(OUT / "final_cps_metrics.csv")
    models = ["persistence", "lightgbm_conformal", "regime_switch", "spike_forecaster", "cps_split", "cps_mondrian",
              "cps_mondrian_sliding", "cps_pit", "cps_pit_mondrian"]
    regimes = ["all", "body", "tail", "onset"]
    t = cps.loc[cps["split"].eq("test")].set_index(["model", "regime"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        x = np.arange(len(regimes)) + (i - len(models) / 2) * width + width / 2
        axes[0].bar(x, [t.loc[(m, r), "coverage_90"] for r in regimes], width, color=COLORS[m], label=SHORT[m])
        axes[1].bar(x, [t.loc[(m, r), "width_90"] for r in regimes], width, color=COLORS[m])
    axes[0].axhline(0.9, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("coverage of the 90% interval")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("mean width ($/MWh, log scale)")
    for ax in axes:
        ax.set_xticks(np.arange(len(regimes)), ["all", "body", "tail", "first interval"])
    axes[0].legend(ncol=3, loc="lower left", fontsize=6)
    fig.suptitle("Test window: 90% interval coverage and width by regime (tail = realised price outside the train 5-95% band)", fontsize=10)
    fig.tight_layout()
    return save(fig, "coverage_width_by_regime")


def fig_pit(rows):
    panels = [("regime_pit", "Regime switch (sliding window)", "regime_switch"),
              ("fc_pit", "Spike forecaster", "spike_forecaster"), ("cps_split_pit", "CPS split", "cps_split"),
              ("cps_mondrian_pit", "CPS Mondrian (fixed)", "cps_mondrian"),
              ("cps_mondrian_sliding_pit", "CPS Mondrian (sliding)", "cps_mondrian_sliding"),
              ("cps_pit_pit", "CPS PIT", "cps_pit"), ("cps_pit_mondrian_pit", "CPS PIT Mondrian", "cps_pit_mondrian"),
              ("lightgbm_pit", "LightGBM + conformal", "lightgbm_conformal")]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), sharey=True)
    for ax, (col, title, key) in zip(axes.ravel(), panels):
        ax.hist(rows[col], bins=20, range=(0, 1), density=True, color=COLORS[key], edgecolor="white", lw=0.4)
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("PIT")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle("Test window: PIT histograms (mid-point PIT; flat = calibrated)", fontsize=10)
    fig.tight_layout()
    return save(fig, "pit_histograms")


def fig_reliability():
    rel = pd.read_csv(OUT / "final_reliability.csv")
    iso = pd.read_csv(OUT / "final_isotonic.csv").set_index(["split", "probability"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, split in zip(axes, ("calibration", "test")):
        ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
        for prob, color in (("raw", COLORS["spike_forecaster"]), ("isotonic", "#d62728")):
            t = rel.loc[rel["split"].eq(split) & rel["probability"].eq(prob)]
            b = iso.loc[(split, prob)]
            ax.plot(t["mean_p"], t["freq"], marker="o", color=color, ms=4,
                    label=f"{prob}: Brier {b['brier']:.5f}, log loss {b['log_loss']:.4f}")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("forecast P(spike up), bin mean")
        ax.set_ylabel("observed frequency")
        note = " (cross-fitted halves)" if split == "calibration" else ""
        ax.set_title(f"{split.capitalize()}{note}: {int(b['positives'])} upward crossings at {int(b['n'])} eligible origins")
        ax.legend(loc="upper left")
    fig.suptitle("Reliability of P(spike up) before and after isotonic recalibration (fitted on calibration)", fontsize=10)
    fig.tight_layout()
    return save(fig, "reliability_isotonic")


def fig_episode(rows, spike, floor):
    up = rows.loc[rows["onset"] & (rows["y"] >= spike)].sort_values("y", ascending=False)
    t0 = up.index[0]
    w = rows.loc[(rows.index >= t0 - pd.Timedelta(hours=1)) & (rows.index <= t0 + pd.Timedelta(hours=2))]
    panels = [("fc_q05", "fc_q95", "Spike forecaster (5-95%)", "spike_forecaster"),
              ("cps_mondrian_sliding_lo", "cps_mondrian_sliding_hi", "CPS Mondrian, sliding (90%)", "cps_mondrian_sliding"),
              ("cps_pit_lo", "cps_pit_hi", "CPS PIT-recalibrated (90%)", "cps_pit"),
              ("lightgbm_lo", "lightgbm_hi", "LightGBM + absolute sliding conformal (90%)", "lightgbm_conformal")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True, sharey=True)
    for ax, (lo, hi, title, key) in zip(axes.ravel(), panels):
        ax.fill_between(w.index, w[lo], w[hi], step="mid", color=COLORS[key], alpha=0.35, label="90% band")
        point = w["lightgbm"] if key == "lightgbm_conformal" else w["regime"]
        ax.plot(w.index, point, color=COLORS[key], lw=1.2, label="point")
        ax.plot(w.index, w["y"], color="black", lw=1.0, marker=".", ms=3, label="actual")
        ax.axhline(spike, color="#d62728", lw=0.7, ls=":")
        ax.axvline(t0, color="grey", lw=0.7, ls="--")
        ax.set_title(title)
        ax.set_ylabel("$/MWh")
    axes[0, 0].legend(loc="upper left")
    for ax in axes[1]:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Largest upward test onset: {t0:%Y-%m-%d %H:%M} AWST (interval label), actual {rows.loc[t0, 'y']:.0f} $/MWh; "
                 f"dotted line = spike threshold {spike:.1f}", fontsize=10)
    fig.tight_layout()
    return save(fig, "spike_episode")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    table = master_table()
    table.to_csv(OUT / "final_master_table.csv", index=False)
    (OUT / "final_master_table.md").write_text(markdown(table) + "\n### Bootstrap ranges\n\n" + bootstrap_markdown())
    cfg = json.loads(CONFIG.read_text())
    rows = pd.read_parquet(OUT / "final_cps_test_rows.parquet")
    for path in (fig_master(table), fig_coverage(), fig_pit(rows), fig_reliability(), fig_episode(rows, cfg["spike"], cfg["floor"])):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
