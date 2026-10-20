"""Read-only support numbers for Casey's comments: CPS vs QRA paired differences on the frozen test rows,
with the repo's own 1-day moving-block bootstrap (scripts/spike_onset.block_bootstrap_mean_diff: block 288, 1,000 draws, seed 42).
Inputs: reports/forecast/final_cps_test_rows.parquet (CPS, exact CRPS) and reports/forecast/qra_test_forecasts.parquet
(QRA rebuilt from frozen coefficients, commit a9c983b). Nothing is refitted. Superseded for the report by scripts/cps_qra_comparison.py (same numbers, more subsets); kept as the check behind the supervisor-comment answers."""
import sys, numpy as np, pandas as pd
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from scripts.spike_onset import block_bootstrap_mean_diff
R = REPO + "/reports/forecast/"
c = pd.read_parquet(R + "final_cps_test_rows.parquet"); q = pd.read_parquet(R + "qra_test_forecasts.parquet")
assert c.index.equals(q.index)
y = c.y.to_numpy(float); a = 0.1
def iscore(lo, hi): return (hi - lo) + 2 / a * np.maximum(lo - y, 0) + 2 / a * np.maximum(y - hi, 0)
cps_is = iscore(c.cps_split_lo.to_numpy(), c.cps_split_hi.to_numpy())
qra_is = iscore(q["q0.050"].to_numpy(), q["q0.950"].to_numpy())
masks = {"all": np.ones(len(y), bool), "calm": ~c["tail"].to_numpy(bool), "tail": c["tail"].to_numpy(bool), "onset": c["onset"].to_numpy(bool)}
rows = []
for name, cps, qra in (("crps (CPS exact - QRA full/flat tails)", c.cps_split_crps.to_numpy(), q.crps_full.to_numpy()),
                       ("crps (CPS exact - QRA quantile-integral)", c.cps_split_crps.to_numpy(), q.crps_quantile_integral.to_numpy()),
                       ("interval score 90% (CPS - QRA)", cps_is, qra_is)):
    for sub, m in masks.items():
        d, lo, hi = block_bootstrap_mean_diff(cps - qra, m)
        rows.append({"metric": name, "subset": sub, "n": int(m.sum()), "cps": float(cps[m].mean()), "qra": float(qra[m].mean()),
                     "diff_cps_minus_qra": d, "boot_p05": lo, "boot_p95": hi})
out = pd.DataFrame(rows); out.to_csv(REPO + "/reports/forecast/casey_cps_vs_qra_bootstrap.csv", index=False)
print(out.round(3).to_string(index=False))
