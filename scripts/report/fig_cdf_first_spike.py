"""CDF of CPS and QRA at the first interval of the 28 Apr 2026 spike (not in the report). Stored results only, nothing refitted. Output folder: $REPORT_FIG_DIR (default reports/figures/report/).

Run from the repo root: python scripts/report/fig_cdf_first_spike.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
R = str(ROOT / "reports" / "forecast") + "/"
F = os.environ.get("REPORT_FIG_DIR", str(ROOT / "reports" / "figures" / "report")) + "/"
os.makedirs(F, exist_ok=True)
"""CDF of the CPS and QRA forecasts for the first interval of the 28 Apr 2026 spike (10:40). Stored data only, nothing refitted.

CPS: regime-switch best guess at 10:40 + all 52,397 calibration residuals (y - regime), the same set used for
cps_split in scripts/spike_cps.py; CDF uses the same tie rule (tau = 0.5). QRA: 23 rebuilt quantiles at 10:40.
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
REPO = str(ROOT)
from scripts.conformal import lower_rank, upper_rank

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
QC, CC, BAND, T = "#E0A030", "#2E6FA8", 170.318, pd.Timestamp("2026-04-28 10:40")
base = pd.read_parquet(REPO + "/data/processed/cps_base_rows.parquet")
cal = base[base.split == "calibration"]
r = np.sort((cal.y - cal.regime).to_numpy(float)); n = len(r)
row = pd.read_parquet(R + "final_cps_test_rows.parquet").loc[T]
yh, y = float(row.regime), float(row.y)
lo, hi = yh + r[lower_rank(n, .1)], yh + r[upper_rank(n, .1)]
assert n == 52397 and np.isclose(lo, row.cps_split_lo) and np.isclose(hi, row.cps_split_hi)
def cdf(x):
    z = np.asarray(x, float) - yh; a = np.searchsorted(r, z, "left"); b = np.searchsorted(r, z, "right")
    return (a + 0.5 * (b - a + 1)) / (n + 1)
assert np.isclose(cdf(y), row.cps_split_pit)
qrow = pd.read_parquet(R + "qra_test_forecasts.parquet").loc[T]
lev = np.array([float(c[1:]) for c in qrow.index if c.startswith("q0")]); qv = np.sort(qrow[[c for c in qrow.index if c.startswith("q0")]].to_numpy(float))
assert np.isclose(qrow.y, y)
def draw(ax, xmin, xmax, legend):
    xs = np.linspace(xmin, xmax, 20000)
    ax.plot(xs, cdf(xs), color=CC, lw=2, drawstyle="steps-post", label="CPS")
    ax.plot(qv, lev, color=QC, lw=2, marker="o", ms=4, label="QRA")
    ax.axvline(y, color="k", lw=1.5, label=f"Actual price ({y:.2f})")
    ax.axvline(BAND, color="r", ls=":", lw=1.2, label="Top of normal band (170.3)")
    for h in (0.05, 0.95): ax.axhline(h, color="grey", ls="--", lw=0.8)
    ax.set_xlim(xmin, xmax); ax.set_ylim(0, 1.02)
    if legend: ax.legend(loc="lower right", fontsize=9)
fig, ax = plt.subplots(figsize=(9, 5))
draw(ax, 100, 1050, True)
ax.text(105, 0.06, "0.05", ha="left", va="bottom", fontsize=8, color="grey"); ax.text(105, 0.96, "0.95", ha="left", va="bottom", fontsize=8, color="grey")
ax.set_xlabel("$/MWh"); ax.set_ylabel("Probability price is at or below x")
ax.set_title("Cumulative forecast distribution, first interval of the spike (28 Apr 2026, 10:40)\nCPS drawn as a step function (no smoothing); QRA levels joined by straight lines", fontsize=10)
ins = inset_axes(ax, width="42%", height="45%", loc="center", bbox_to_anchor=(0.12, 0.02, 1, 1), bbox_transform=ax.transAxes)
draw(ins, 160, 215, False); ins.tick_params(labelsize=8); ins.set_title("Zoom: 160 to 215 $/MWh", fontsize=8)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.92); fig.savefig(F + "cdf_first_spike_interval_cps_qra.png", dpi=200)
q = lambda l: yh + r[min(max(int(np.ceil(l * (n + 1))), 1), n) - 1]
print(f"CPS best guess {yh:.2f}; 90% range {lo:.2f}-{hi:.2f}; median {q(.5):.2f}; quantile-rule 5%/95% {q(.05):.2f}/{q(.95):.2f}")
print(f"CPS P(<=actual) {cdf(y):.6f}; P(>170.318) {1-cdf(BAND):.4f}; residuals above actual-yh: {(r > y-yh).sum()}")
print(f"QRA q05/q50/q95 {qrow['q0.050']:.2f}/{qrow['q0.500']:.2f}/{qrow['q0.950']:.2f}; q0.01 {qv[0]:.2f}; q0.99 {qv[-1]:.2f}")
