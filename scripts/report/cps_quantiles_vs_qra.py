"""Like-for-like CRPS check: the CPS against QRA, scored two ways, with block-bootstrap confidence intervals.

WHY LIKE-FOR-LIKE (Casey comment #7). CRPS scores a whole predictive distribution. The CPS gives a full
distribution, so its CRPS can be computed exactly. QRA only gives 23 quantile levels (0.01 ... 0.99). If QRA is
scored only on that 0.01-0.99 grid while the CPS is scored on its full curve, QRA is flattered: the grid score
leaves out the two outer 1% tails, and it reads about 0.03 $/MWh lower than QRA's full-curve score. That is
about as large as the whole CPS-QRA gap, so a mixed comparison could change the answer. This script therefore
scores BOTH methods the same way, twice:
  (a) full curve: CPS exact CRPS (stored cps_split_crps, re-checked here) and QRA's full reconstruction as the
      report defines it: the 23 levels joined by straight lines, held flat below 0.01 and above 0.99 (so 1% of
      the chance sits on each end level), integrated on a fine 0.0005 level grid
      (same function as scripts/qra_first_interval.full_crps; re-checked against the stored crps_full);
  (b) same grid: both read at QRA's 23 levels and scored with the same trapezoid of 2 x pinball loss over
      0.01-0.99 (scripts/metrics.crps_from_quantiles), tails outside left out for both.

HOW THE CPS QUANTILES ARE READ OFF. Nothing is refitted or tuned, and nothing is chosen on test. The CPS is the
frozen split CPS on the regime switch: its distribution is the best guess plus each of the n = 52,397 sorted
calibration errors (y - regime on the calibration split of data/processed/cps_base_rows.parquet). Its CDF is a
step function (one step per error). Level a is the error with 1-based rank floor((n+1)a) when a < 0.5
(scripts/conformal.lower_rank) and ceil((n+1)a) when a >= 0.5 (conformal.upper_rank, SplitCPS.quantile).
These are the ranks the report's CPS 90% range uses, and the script checks it reproduces that range exactly.
QRA quantiles are the stored, sorted test forecasts (reports/forecast/qra_test_forecasts.parquet).

BLOCK BOOTSTRAP. To judge whether a CPS-QRA difference is bigger than chance, the per-interval differences are
resampled in whole blocks of one day (288 five-minute intervals), 1,000 draws, seed 42, and the 5th and 95th
percentiles of the resampled mean give a 90% confidence interval (copy of scripts/spike_onset.block_bootstrap_mean_diff).
Whole days are used because neighbouring prices are strongly linked and prices follow a daily pattern; resampling
single intervals would pretend they are independent and give intervals that are too narrow.

Outputs: reports/forecast/cps_quantiles_vs_qra.csv (all metrics, subsets all / calm / tail / first_interval),
reports/forecast/cps_quantiles_vs_qra_pinball_by_level.csv, $REPORT_FIG_DIR/cps_vs_qra_matched_levels.png (default reports/figures/report/).
Run from the repo root: python scripts/report/cps_quantiles_vs_qra.py
"""
import sys, os, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
R = REPO / "reports/forecast"

def block_bootstrap_mean_diff(diff, mask, block=288, n_boot=1000, seed=42):  # copy of scripts/spike_onset.py (avoids lightgbm import)
    diff = np.where(mask, diff, 0.0); m = mask.astype(float); n = len(diff)
    rng = np.random.default_rng(seed); starts = np.arange(0, n - block + 1); nb = int(np.ceil(n / block))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.choice(starts, size=nb, replace=True)[:, None] + np.arange(block)).ravel()[:n]
        draws[i] = diff[idx].sum() / max(m[idx].sum(), 1.0)
    return float(diff.sum() / m.sum()), float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))

b = pd.read_parquet(REPO / "data/processed/cps_base_rows.parquet")
cal = b[b["split"] == "calibration"]
r = np.sort((cal.y - cal.regime).to_numpy(float)); r = r[np.isfinite(r)]; n = len(r)
c = pd.read_parquet(R / "final_cps_test_rows.parquet"); q = pd.read_parquet(R / "qra_test_forecasts.parquet")
assert c.index.equals(q.index) and np.allclose(c.y, q.y)
cols = [k for k in q.columns if k.startswith("q0.")]; lev = np.array([float(k[1:]) for k in cols])
def rank(a): k = np.floor((n + 1) * a) if a < 0.5 else np.ceil((n + 1) * a); return int(min(max(k, 1), n)) - 1
off = np.array([r[rank(a)] for a in lev])
yhat = c.regime.to_numpy(float); y = c.y.to_numpy(float)
G = {"CPS": yhat[:, None] + off[None, :], "QRA": q[cols].to_numpy(float)}
i05, i95 = list(lev).index(0.05), list(lev).index(0.95)
assert np.allclose(G["CPS"][:, i05], c.cps_split_lo) and np.allclose(G["CPS"][:, i95], c.cps_split_hi)
def pin(g): e = y[:, None] - g; return np.maximum(lev * e, (lev - 1) * e)
P = {k: pin(g) for k, g in G.items()}
def crps_grid(p): s = 2 * p; return (0.5 * (s[:, :-1] + s[:, 1:]) * np.diff(lev)).sum(axis=1)
masks = {"all": np.ones(len(y), bool), "calm": ~c["tail"].to_numpy(bool), "tail": c["tail"].to_numpy(bool), "first_interval": c["onset"].to_numpy(bool)}
pairs = [(a, round(1 - a, 3)) for a in lev if a < 0.5]
rows = []
def add(metric, xc, xq):
    for s, m in masks.items():
        d, lo, hi = block_bootstrap_mean_diff(xc - xq, m)
        rows.append(dict(metric=metric, subset=s, n=int(m.sum()), cps=float(xc[m].mean()), qra=float(xq[m].mean()), cps_minus_qra=d, ci90_lo=lo, ci90_hi=hi))
# (a) full-curve CRPS for both. CPS: exact score of the step distribution (stored, re-checked from the errors).
cum = np.concatenate([[0.0], np.cumsum(r)]); kk = np.arange(1, n + 1); pair = np.sum((2 * kk - n - 1) * r) / n ** 2
z = y - yhat; k = np.searchsorted(r, z, side="right")
cps_full = (k * z - cum[k] + (cum[-1] - cum[k]) - (n - k) * z) / n - pair
assert np.allclose(cps_full, c.cps_split_crps.to_numpy(float), atol=1e-6)
# QRA: straight lines between the 23 levels, flat beyond 0.01 / 0.99 (np.interp holds the end values), fine level grid.
tau = np.arange(0.0005 / 2, 1, 0.0005)
qra_full = np.empty(len(y))
for s0 in range(0, len(y), 4000):
    Gq = G["QRA"][s0:s0 + 4000]; qq = np.vstack([np.interp(tau, lev, row) for row in Gq]); e = y[s0:s0 + 4000, None] - qq
    qra_full[s0:s0 + 4000] = (2 * np.maximum(tau * e, (tau - 1) * e)).mean(axis=1)
assert np.allclose(qra_full, q.crps_full.to_numpy(float), atol=1e-6)
add("crps_full_curve", cps_full, qra_full)
# (b) both on the same 0.01-0.99 grid
add("crps_quantile_grid", crps_grid(P["CPS"]), crps_grid(P["QRA"]))
add("pinball_mean_23_levels", P["CPS"].mean(1), P["QRA"].mean(1))
for a, b_ in pairs:
    il, ih = list(lev).index(a), list(lev).index(b_); cov = round(b_ - a, 3); al = 1 - cov
    out = {}
    for k, g in G.items():
        lo, hi = g[:, il], g[:, ih]
        out[k] = ((y >= lo) & (y <= hi)).astype(float), hi - lo, (hi - lo) + 2 / al * np.maximum(lo - y, 0) + 2 / al * np.maximum(y - hi, 0)
    tag = f"{int(round(cov * 100))}%"
    add(f"coverage_{tag}", out["CPS"][0], out["QRA"][0]); add(f"width_{tag}", out["CPS"][1], out["QRA"][1]); add(f"interval_score_{tag}", out["CPS"][2], out["QRA"][2])
res = pd.DataFrame(rows); res.to_csv(R / "cps_quantiles_vs_qra.csv", index=False)
pl = []
for j, a in enumerate(lev):
    for s, m in masks.items():
        d, lo, hi = block_bootstrap_mean_diff(P["CPS"][:, j] - P["QRA"][:, j], m)
        pl.append(dict(level=a, subset=s, cps=float(P["CPS"][m, j].mean()), qra=float(P["QRA"][m, j].mean()), cps_minus_qra=d, ci90_lo=lo, ci90_hi=hi))
pl = pd.DataFrame(pl); pl.to_csv(R / "cps_quantiles_vs_qra_pinball_by_level.csv", index=False)
print("levels:", list(lev)); print("n calibration residuals:", n); print("CPS offsets:", dict(zip(lev, off.round(3))))
pd.set_option("display.width", 250); print(res.round(4).to_string(index=False))
print(pl[pl.subset == "all"].round(4).to_string(index=False))
F = Path(os.environ.get("REPORT_FIG_DIR", REPO / "reports/figures/report")); F.mkdir(parents=True, exist_ok=True)
if True:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cv = [int(round((b_ - a) * 100)) for a, b_ in pairs][::-1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, col in (("CPS", "tab:blue"), ("QRA", "tab:orange")):
        sub = res[res.subset == "all"]
        ax[0].plot(cv, [sub[sub.metric == f"coverage_{x}%"][k.lower()].iloc[0] for x in cv], "o-", color=col, label=k)
        ax[1].plot(cv, [sub[sub.metric == f"width_{x}%"][k.lower()].iloc[0] for x in cv], "o-", color=col, label=k)
        ps = pl[pl.subset == "all"]; ax[2].plot(ps.level, ps[k.lower()], "o-", color=col, label=k, ms=3)
    ax[0].plot([0, 100], [0, 1], "k--", lw=0.8, label="target"); ax[0].set(xlabel="Nominal central range (%)", ylabel="Coverage on test", title="Coverage")
    ax[1].set(xlabel="Nominal central range (%)", ylabel="Mean width ($/MWh)", title="Width")
    ax[2].set(xlabel="Quantile level", ylabel="Mean pinball loss ($/MWh)", title="Pinball loss per level (lower is better)")
    for a_ in ax: a_.legend(); a_.grid(alpha=0.3)
    fig.suptitle("CPS read at QRA's 23 levels vs QRA, all 40,127 test intervals"); fig.tight_layout()
    fig.savefig(F / "cps_vs_qra_matched_levels.png", dpi=200)
