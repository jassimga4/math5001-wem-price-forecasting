"""First-interval CRPS bar chart with QRA added (test, 403 onsets). Stored results only."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "forecast"
m = pd.read_csv(OUT / "final_cps_metrics.csv").query("split=='test' and regime=='onset'").set_index("model")
q = json.loads((OUT / "qra_first_interval.json").read_text())
names = ["Persistence", "Regime switch", "QRA*", "Split CPS", "Spike forecaster"]
vals = [m.loc["persistence", "crps"], m.loc["regime_switch", "crps"], q["onset_crps_qi"], m.loc["cps_split", "crps"], m.loc["spike_forecaster", "crps"]]
cols = ["grey", "tab:orange", "#E0A030", "#2E6FA8", "tab:blue"]
fig, ax = plt.subplots(figsize=(8, 4.3))
bars = ax.bar(names, vals, color=cols, edgecolor="k")
bars[2].set_hatch("//")
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
ax.set_ylabel("mean CRPS ($/MWh), lower is better")
ax.set_ylim(0, max(vals) * 1.18)
ax.set_title(f"First interval of spike events, test period (n = {q['n_onset']} onsets)", fontsize=10)
fig.text(0.5, 0.01, f"* QRA: quantile-integral CRPS over 0.01-0.99; full-distribution (flat tails) value {q['onset_crps_full']:.2f}. Others: exact CRPS.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for p in sys.argv[1:] or [str(ROOT / "reports/figures/spike/first_interval_crps_with_qra.png")]:
    fig.savefig(p, dpi=200); print(p)
