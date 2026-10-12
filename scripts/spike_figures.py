"""Figures for the implemented spike forecaster (test window, scored once).

Reads reports/forecast/spike_forecaster_test_forecasts.parquet (written by
`python scripts/spike_forecaster.py evaluate`), spike_forecaster_metrics.csv,
spike_forecaster_bootstrap.csv and spike_stage1_results.csv. Writes PNGs to
reports/figures/spike/.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spike_forecaster import CONFIG_PATH, OUT, TEST_FORECASTS  # noqa: E402

FIG = ROOT / "reports" / "figures" / "spike"
BINS = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])


def reliability(fc: pd.DataFrame, spike: float, floor: float) -> Path:
    elig = fc.loc[fc["eligible"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    rows = []
    for ax, side, hit in ((axes[0], "p_up", elig["y"] >= spike), (axes[1], "p_down", elig["y"] <= floor)):
        b = pd.cut(elig[side], BINS, include_lowest=True)
        g = pd.DataFrame({"p": elig[side], "hit": hit.astype(float), "bin": b}).groupby("bin", observed=True)
        t = g.agg(mean_p=("p", "mean"), freq=("hit", "mean"), n=("hit", "size")).reset_index()
        t = t.loc[t["n"] > 0]
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        ax.scatter(t["mean_p"], t["freq"], s=np.clip(t["n"] / 40, 12, 250), zorder=3)
        ax.plot(t["mean_p"], t["freq"], lw=1)
        for _, r in t.iterrows():
            ax.annotate(f"{int(r['n'])}", (r["mean_p"], r["freq"]), fontsize=7, xytext=(4, -9), textcoords="offset points")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        label = "upward crossing (price >= %.1f)" % spike if side == "p_up" else "downward crossing (price <= %.1f)" % floor
        ax.set_title(f"P(spike {side[2:]}): {label}", fontsize=9)
        ax.set_xlabel("forecast probability (bin mean)")
        ax.set_ylabel("observed frequency")
        t.insert(0, "side", side)
        rows.append(t)
    fig.suptitle("Reliability of spike probabilities, test window (eligible origins; point labels = rows per bin)", fontsize=10)
    fig.tight_layout()
    path = FIG / "reliability_test.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    pd.concat(rows).astype({"bin": str}).to_csv(FIG / "reliability_test.csv", index=False)
    return path


def episodes(fc: pd.DataFrame, spike: float, floor: float, n_up: int = 3) -> Path:
    """Largest fresh upward onsets on distinct days plus the deepest downward onset, +-2 h around each."""
    y = fc["y"]
    onset = fc["eligible"] & ((y >= spike) | (y <= floor))
    up = fc.loc[onset & (y >= spike)].sort_values("y", ascending=False)
    picks, days = [], set()
    for t in up.index:
        if t.normalize() not in days:
            picks.append(t)
            days.add(t.normalize())
        if len(picks) == n_up:
            break
    down = fc.loc[onset & (y <= floor)].sort_values("y")
    if len(down):
        picks.append(down.index[0])
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3.0 * len(picks)))
    for ax, t in zip(np.atleast_1d(axes), picks):
        w = fc.loc[(fc.index >= t - pd.Timedelta(hours=2)) & (fc.index <= t + pd.Timedelta(hours=2))]
        ax.fill_between(w.index, w["q05"], w["q95"], color="tab:blue", alpha=0.18, step="mid", label="5-95%")
        ax.fill_between(w.index, w["q25"], w["q75"], color="tab:blue", alpha=0.35, step="mid", label="25-75%")
        ax.plot(w.index, w["point"], color="tab:blue", lw=1.2, label="point (regime switch)")
        ax.plot(w.index, w["y"], color="black", lw=1.2, marker=".", ms=3, label="actual")
        alerts = w.loc[w["alert"]]
        ax.scatter(alerts.index, alerts["y"], marker="^", color="tab:red", zorder=4, label="spike alert issued")
        ax.axhline(spike, color="tab:red", lw=0.8, ls=":")
        ax.axhline(floor, color="tab:green", lw=0.8, ls=":")
        ax.axvline(t, color="grey", lw=0.8, ls="--")
        ax.set_title(f"Onset at {t:%Y-%m-%d %H:%M} (interval label, AWST); actual {y.loc[t]:.1f} $/MWh", fontsize=9)
        ax.set_ylabel("$/MWh")
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    path = FIG / "example_episodes_test.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def first_interval_crps() -> Path:
    ours = pd.read_csv(OUT / "spike_forecaster_metrics.csv").set_index("split")
    base = pd.read_csv(OUT / "spike_stage1_results.csv").set_index(["model", "split"])
    boot = pd.read_csv(OUT / "spike_forecaster_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ("calibration", "test")):
        vals = [base.loc[("persistence", split), "onset_crps"], base.loc[("regime_switch", split), "onset_crps"],
                ours.loc[split, "onset_crps"]]
        bars = ax.bar(["persistence", "regime switch", "spike forecaster"], vals, color=["grey", "tab:orange", "tab:blue"])
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
        lines = []
        for b in ("persistence", "regime_switch"):
            r = boot.loc[(boot["split"] == split) & (boot["baseline"] == b) & (boot["subset"] == "onset") & (boot["metric"] == "crps")].iloc[0]
            lines.append(f"vs {b.replace('_', ' ')}: {r['diff']:+.2f} [{r['boot_p05']:+.2f}, {r['boot_p95']:+.2f}]")
        ax.set_title(f"{split}: first-interval CRPS (n = {int(ours.loc[split, 'onset_events'])} onsets)", fontsize=9)
        ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7)
        ax.set_ylabel("mean CRPS ($/MWh)")
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("First interval of spike events: CRPS with 90% block-bootstrap ranges of the difference", fontsize=10)
    fig.tight_layout()
    path = FIG / "first_interval_crps.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG_PATH.read_text())
    fc = pd.read_parquet(TEST_FORECASTS)
    for p in (reliability(fc, cfg["spike"], cfg["floor"]), episodes(fc, cfg["spike"], cfg["floor"]), first_interval_crps()):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
