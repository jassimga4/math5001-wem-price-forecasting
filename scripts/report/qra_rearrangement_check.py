"""Check for supervisor comment #10: QRA test scores before vs after sorting (rearrangement). Frozen coefficients, no refit; same inputs as scripts/qra.py."""
import sys, numpy as np, pandas as pd
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from scripts.qra import OUT, QRA_FORECASTS, _saved_point_result, _design, _coefficient_matrix
from scripts.metrics import crps_from_quantiles
from scripts.forecast_design import TARGET
point, _, _ = _saved_point_result(); test = point["test"]; y = test[TARGET].astype(float)
fc = pd.DataFrame({n: point["predictions"][n].reindex(y.index) for n in QRA_FORECASTS})
levels, matrix, cols = _coefficient_matrix(pd.read_csv(OUT / "qra_coefficients.csv"))
raw = _design(fc, cols) @ matrix.T; srt = np.sort(raw, axis=1); yv = y.to_numpy(float)
tail = pd.read_parquet(OUT / "final_cps_test_rows.parquet")["tail"].to_numpy(bool)
i05, i95 = list(levels).index(0.05), list(levels).index(0.95)
rows = []
for name, g in (("before sorting", raw), ("after sorting", srt)):
    qi = crps_from_quantiles(yv, g, levels); lo, hi = g[:, i05], g[:, i95]
    rows.append({"version": name, "crps_qi_all": qi.mean(), "crps_qi_tail": qi[tail].mean(),
                 "coverage_90": np.mean((yv >= lo) & (yv <= hi)), "width_90": np.mean(hi - lo),
                 "rows_crossed": np.mean(np.any(np.diff(g, axis=1) < 0, axis=1)),
                 "rows_where_q05_or_q95_changed": np.mean((g[:, i05] != srt[:, i05]) | (g[:, i95] != srt[:, i95]))})
d = pd.DataFrame(rows); d.to_csv(REPO + "/reports/forecast/qra_rearrangement_check.csv", index=False); print(d.to_string(index=False))
