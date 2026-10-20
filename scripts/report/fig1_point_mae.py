"""Report Figure 1: test MAE of the point forecasts, from reports/forecast/point_metrics.csv. Nothing refitted.
Output folder: $REPORT_FIG_DIR (default reports/figures/report/). Run: python scripts/report/fig1_point_mae.py
"""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ROOT / "reports" / "forecast"
OUT = Path(os.environ.get("REPORT_FIG_DIR", ROOT / "reports" / "figures" / "report"))
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PALETTE = {
    "lightgbm": "#2E86AB",
    "persistence_5min": "#A23B72",
    "persistence_30min": "#F18F01",
    "persistence_1d": "#C73E1D",
    "ridge": "#6A994E",
    "qra": "#E9C46A",
}
LABELS = {
    "lightgbm": "LightGBM",
    "persistence_5min": "Persistence (5 min)",
    "persistence_30min": "Persistence (30 min)",
    "persistence_1d": "Persistence (1 day)",
    "ridge": "Ridge",
    "qra": "QRA",
}
CREATED: list[tuple[str, str, str]] = []
SKIPPED: list[str] = []


def save(fig: plt.Figure, name: str, description: str, source: str) -> None:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def chart1_point_mae() -> None:
    path = FORECAST / "point_metrics.csv"
    df = pd.read_csv(path)
    need = {"model", "split", "mae"}
    if not need.issubset(df.columns):
        raise SystemExit(f"missing columns {need - set(df.columns)}")
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise SystemExit("no test rows")
    order = ["persistence_5min", "persistence_30min", "persistence_1d", "ridge", "lightgbm"]
    order = [m for m in order if m in set(test["model"])] + [
        m for m in test["model"].unique() if m not in order
    ]
    test["model"] = pd.Categorical(test["model"], categories=order, ordered=True)
    test = test.sort_values("model")
    colors = [PALETTE.get(m, "#555555") for m in test["model"]]
    labels = [LABELS.get(m, m) for m in test["model"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, test["mae"].values, color=colors, edgecolor="white", width=0.72)
    for bar, val in zip(bars, test["mae"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("MAE ($/MWh)")
    ax.set_xlabel("Model")
    ax.set_title("Point forecast MAE on test set (5-min ahead)")
    ax.set_ylim(0, max(test["mae"].values) * 1.18)
    if "n" in test.columns:
        n = int(test["n"].iloc[0])
        ax.annotate(f"n = {n:,}", xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=8, color="#444")
    save(fig, "point_forecast_mae.png", "Test-set MAE by point-forecast model", "reports/forecast/point_metrics.csv")




if __name__ == "__main__":
    chart1_point_mae()
