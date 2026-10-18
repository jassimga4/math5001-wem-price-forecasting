"""QRA figures from stored result csvs (nothing is refitted; test outcomes are only scored).

Inputs: reports/forecast/qra_variants_test.csv, qra_variants_calib.csv, qra_variants_meta.json, qra_test.csv,
conformal_test.csv, final_master_table.csv, spike_forecaster_test_forecasts.parquet.
Outputs: reports/figures/qra/*.png and *.pdf.
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
sys.path.insert(0, str(ROOT / "scripts"))
from final_report import STYLE  # noqa: E402
from metrics import pinball_loss  # noqa: E402

OUT = ROOT / "reports" / "forecast"
FIG = ROOT / "reports" / "figures" / "qra"
COMB = {"qra": ("QRA", "#1f77b4"), "qrm": ("QRM", "#ff7f0e"), "qave": ("Q-Ave", "#2ca02c"),
        "lightgbm_quantile": ("LightGBM quantile", "#d62728")}
WINDOWS = ["28d", "56d", "91d", "all"]
MARK = {"28d": "o", "56d": "s", "91d": "^", "all": "D", "none": "*"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    return FIG / f"{name}.png"


def references():
    t = pd.read_csv(OUT / "final_master_table.csv")
    t = t.loc[t["split"].eq("test")].set_index("model")
    return {"LightGBM + sliding conformal": t.loc["lightgbm_conformal"], "Spike forecaster": t.loc["spike_forecaster"],
            "CPS Mondrian (sliding)": t.loc["cps_mondrian_sliding"]}


def fig_coverage_width(var, meta):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    v = var.loc[var["role"].eq("variant")]
    for comb, (lab, col) in COMB.items():
        g = v.loc[v["combiner"].eq(comb)]
        for w, gw in g.groupby("window"):
            ax.scatter(gw["mean_width_90"], gw["coverage_90"], color=col, marker=MARK[w], s=26, alpha=0.8)
        ax.scatter([], [], color=col, label=lab)
    for w in WINDOWS:
        ax.scatter([], [], color="grey", marker=MARK[w], label=f"window {w}")
    sel = v.loc[v["variant"].eq(meta["selected_variant"])].iloc[0]
    ax.scatter(sel["mean_width_90"], sel["coverage_90"], s=130, facecolor="none", edgecolor="k",
               label="selected on calibration")
    for (lab, r), m in zip(references().items(), ("X", "P", "v")):
        ax.scatter(r["width_90"], r["coverage_90"], color="k", marker=m, s=60, label=lab)
    ax.axhline(0.9, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Mean 90% interval width ($/MWh), test")
    ax.set_ylabel("90% coverage, test")
    ax.set_title("QRA variants: coverage against width (test, 40,127 intervals)")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    return save(fig, "qra_coverage_vs_width")


def fig_crps(var, meta):
    v = var.sort_values("crps_quantile_integral")
    fig, ax = plt.subplots(figsize=(6.4, 8.5))
    ypos = np.arange(len(v))[::-1]
    cols = ["k" if r == "frozen_conformal_reference" else COMB[c][1] for r, c in zip(v["role"], v["combiner"].fillna(""))]
    ax.barh(ypos, v["crps_quantile_integral"], color=cols)
    labels = [s + ("  (selected)" if s == meta["selected_variant"] else "") for s in v["variant"]]
    ax.set_yticks(ypos, labels, fontsize=6)
    ax.set_xlim(3.5, v["crps_quantile_integral"].max() * 1.02)
    ax.set_xlabel("Quantile-integral CRPS, test (23 levels 0.01 to 0.99)")
    ax.set_title("CRPS by QRA variant (black: frozen sliding conformal references)")
    return save(fig, "qra_crps_by_variant")


def fig_recency(var, calib):
    v = var.loc[var["window"].isin(WINDOWS)]
    c = calib.loc[calib["window"].isin(WINDOWS) & calib["role"].eq("variant")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for col, (frame, split) in enumerate(((c, "inner calibration (March 2026)"), (v, "test"))):
        for (comb, sub), g in frame.groupby(["combiner", "subset"]):
            g = g.set_index("window").reindex(WINDOWS)
            kw = dict(color=COMB[comb][1], alpha=0.75, marker="o", ms=3, lw=1)
            axes[0, col].plot(WINDOWS, g["crps_quantile_integral"], **kw)
            axes[1, col].plot(WINDOWS, g["coverage_90"], **kw)
        axes[0, col].set_title(f"Recency window effect, {split}")
        axes[1, col].axhline(0.9, color="grey", ls="--", lw=0.8)
        axes[1, col].set_xlabel("Calibration window (most recent days)")
    axes[0, 0].set_ylabel("Quantile-integral CRPS")
    axes[1, 0].set_ylabel("90% coverage")
    for comb, (lab, colr) in COMB.items():
        if comb != "lightgbm_quantile":
            axes[0, 1].plot([], [], color=colr, label=lab)
    axes[0, 1].legend(fontsize=7)
    fig.text(0.5, -0.01, "One line per combiner and member subset. 'all' = all pre-test data available at the fit end.",
             ha="center", fontsize=7)
    return save(fig, "qra_recency_window")


def fig_pinball():
    qra = pd.read_csv(OUT / "qra_test.csv")
    conf = pd.read_csv(OUT / "conformal_test.csv")

    def levels(frame):
        lv, pb = [], []
        for _, r in frame.sort_values("alpha").iterrows():
            lv += [r["alpha"] / 2, 1 - r["alpha"] / 2]
            pb += [r["pinball_lower"], r["pinball_upper"]]
        o = np.argsort(lv)
        return np.array(lv)[o], np.array(pb)[o]

    series = {"QRA (PR 1)": levels(qra.loc[qra["method"].eq("qra")]),
              "LightGBM + sliding conformal": levels(conf.loc[conf["method"].eq("absolute") & conf["base"].eq("lightgbm")]),
              "Persistence + sliding conformal": levels(conf.loc[conf["method"].eq("absolute") & conf["base"].eq("persistence_5min")])}
    f = pd.read_parquet(OUT / "spike_forecaster_test_forecasts.parquet")
    lv = [0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99]
    series["Spike forecaster"] = (np.array(lv), np.array([pinball_loss(f["y"], f[f"q{int(round(l * 100)):02d}"], l) for l in lv]))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (lab, (x, y)), col in zip(series.items(), ("#1f77b4", "#bcbd22", "#7f7f7f", "#9467bd")):
        ax.plot(x, y, marker="o", ms=4, label=lab, color=col)
    ax.set_xlabel("Quantile level")
    ax.set_ylabel("Mean pinball loss, test ($/MWh)")
    ax.set_title("Pinball loss by quantile level (test)")
    ax.legend(fontsize=7)
    rows = [(lab, l, p) for lab, (x, y) in series.items() for l, p in zip(x, y)]
    pd.DataFrame(rows, columns=["model", "level", "pinball"]).to_csv(OUT / "qra_pinball_by_level.csv", index=False)
    return save(fig, "qra_pinball_by_quantile")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    var = pd.read_csv(OUT / "qra_variants_test.csv")
    calib = pd.read_csv(OUT / "qra_variants_calib.csv")
    meta = json.loads((OUT / "qra_variants_meta.json").read_text())
    for p in (fig_coverage_width(var, meta), fig_crps(var, meta), fig_recency(var, calib), fig_pinball()):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
