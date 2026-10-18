"""Regenerate frozen-QRA test quantiles and score the first interval of spike events.

No refitting: QRA coefficients come from reports/forecast/qra_coefficients.csv
(fit on calibration only by scripts/qra.py). The five point forecasts are
rebuilt exactly as scripts/qra.py does (_saved_point_result). The regenerated
aggregates are checked against the stored qra_test.csv / qra_regime.csv before
anything is scored. Onset flags come from final_cps_test_rows.parquet, the same
rows used for the 403 test onsets in spike_cps.py.

CRPS is reported two ways:
- quantile_integral: scripts.metrics.crps_from_quantiles over 0.01-0.99 (same as
  the stored QRA tables; omits the tails, so slightly lenient);
- full: exact CRPS of the distribution whose quantile function is the linear
  interpolation of the 23 QRA quantiles, held flat below 0.01 and above 0.99
  (1% point mass at each end). Integrated on a 0.0005 tau grid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.metrics import crps_from_quantiles  # noqa: E402
from scripts.qra import OUT, QRA_FORECASTS, _saved_point_result, predict_qra  # noqa: E402

PARQUET = OUT / "qra_test_forecasts.parquet"


def full_crps(y, grid, levels, step=0.0005):
    tau = np.arange(step / 2, 1, step)
    q = np.vstack([np.interp(tau, levels, row) for row in grid])
    e = y[:, None] - q
    return (2 * np.maximum(tau * e, (tau - 1) * e)).mean(axis=1)


def main():
    point, _, _ = _saved_point_result()
    test = point["test"]
    y = test["mcp"] if "mcp" in test else None
    from scripts.forecast_design import TARGET
    y = test[TARGET].astype(float)
    fc = pd.DataFrame({n: point["predictions"][n].reindex(y.index) for n in QRA_FORECASTS})
    coef = pd.read_csv(OUT / "qra_coefficients.csv")
    grid, crossed = predict_qra(fc, coef)
    levels = np.array([float(c) for c in grid.columns])
    G = grid.to_numpy(float)
    yv = y.to_numpy(float)
    qi = crps_from_quantiles(yv, G, levels)
    lo, hi = grid["0.050"].to_numpy(), grid["0.950"].to_numpy()
    cov = float(np.mean((yv >= lo) & (yv <= hi)))
    width = float(np.mean(hi - lo))
    rows = pd.read_parquet(OUT / "final_cps_test_rows.parquet")
    assert rows.index.equals(y.index), "test index mismatch with final_cps_test_rows"
    assert np.allclose(rows["y"].to_numpy(float), yv)
    tail = rows["tail"].to_numpy(bool)
    onset = rows["onset"].to_numpy(bool)
    check = {"crps_qi": float(qi.mean()), "coverage_90": cov, "width_90": width,
             "tail_crps_qi": float(qi[tail].mean()), "crossed_share": crossed}
    stored_t = pd.read_csv(OUT / "qra_test.csv").query("method=='qra' and alpha==0.1").iloc[0]
    stored_r = pd.read_csv(OUT / "qra_regime.csv").query("method=='qra'").set_index("regime")
    stored = {"crps_qi": stored_r.loc["all", "crps_quantile_integral"], "coverage_90": stored_t["coverage"],
              "width_90": stored_t["mean_width"], "tail_crps_qi": stored_r.loc["tail", "crps_quantile_integral"]}
    print("regenerated", check)
    print("stored     ", stored)
    for k, v in stored.items():
        if abs(check[k] - v) > 1e-6 * max(1, abs(v)):
            raise SystemExit(f"MISMATCH {k}: regenerated {check[k]} vs stored {v}")
    full = full_crps(yv, G, levels)
    out = grid.copy()
    out.columns = [f"q{c}" for c in grid.columns]
    out.insert(0, "y", yv)
    out["tail"], out["onset"] = tail, onset
    out["crps_quantile_integral"], out["crps_full"] = qi, full
    out.to_parquet(PARQUET)
    res = {**check, "crps_full": float(full.mean()), "tail_crps_full": float(full[tail].mean()),
           "n_onset": int(onset.sum()), "onset_crps_qi": float(qi[onset].mean()),
           "onset_crps_full": float(full[onset].mean()),
           "onset_coverage_90": float(np.mean(((yv >= lo) & (yv <= hi))[onset])),
           "onset_width_90": float(np.mean((hi - lo)[onset]))}
    (OUT / "qra_first_interval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
