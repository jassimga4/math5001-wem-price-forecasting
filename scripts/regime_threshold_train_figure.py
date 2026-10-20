"""Old CPS vs new CPS (train-chosen regime threshold) vs QRA on key test metrics. Reads regime_threshold_train_comparison.csv."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / "reports/forecast/regime_threshold_train_comparison.csv")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
labels = ["CPS, old threshold\n(32.48, calibration)", "CPS, new threshold\n(52.43, train)", "QRA*"]
cols = ["#9BB8D6", "#2E6FA8", "#E0A030"]
panels = [("crps_all", "CRPS, all"), ("crps_tail", "CRPS, tail"), ("crps_onset", "CRPS, first interval"),
          ("coverage90_all", "90% coverage, all"), ("coverage90_onset", "90% coverage, spike starts"), ("interval_score_all", "Interval score, all")]
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, (c, title) in zip(axes.flat, panels):
    v = t[c].to_list()
    bars = ax.bar(range(3), v, color=cols, edgecolor="k"); bars[2].set_hatch("//")
    for b, x in zip(bars, v):
        ax.annotate(f"{x:.3f}" if x < 10 else f"{x:.2f}", (b.get_x() + b.get_width() / 2, x), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3), ["old CPS", "new CPS", "QRA"]); ax.set_title(title); ax.set_ylim(0, max(v) * 1.18)
    if c.startswith("coverage"): ax.axhline(0.9, ls="--", color="k", lw=0.8)
fig.suptitle("Regime-switch threshold chosen on training data: CPS before and after, against QRA (test, 40,127 intervals)", fontsize=11)
fig.text(0.5, 0.005, "Lower is better except coverage (target 0.90, dashed). *QRA unchanged; CRPS is its full-distribution (flat-tail) score.", ha="center", fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
for p in sys.argv[1:]: fig.savefig(p, dpi=200); print(p)
