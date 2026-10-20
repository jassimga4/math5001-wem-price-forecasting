"""CPS vs QRA on the frozen test rows, plus the derived numbers quoted in the report.

Nothing is refitted. Inputs:
  reports/forecast/final_cps_test_rows.parquet (scripts/spike_cps.py; CPS exact CRPS, order-statistic 90% range)
  reports/forecast/qra_test_forecasts.parquet (scripts/qra_first_interval.py; frozen QRA coefficients, sorted levels)
  data/processed/cps_base_rows.parquet (fresh-onset flag, calibration residuals of the regime switch)
  reports/forecast/spike_forecaster_test_forecasts.parquet (alerts)
QRA CRPS: 'full' = exact CRPS of the 23 levels joined linearly and held flat below 0.01 / above 0.99;
'qi' = quantile integral (levels 0.01..0.99). Bootstrap: scripts/spike_onset.block_bootstrap_mean_diff
(1-day blocks of 288 intervals, 1,000 draws, seed 42), 90% interval of the paired mean difference CPS - QRA.
Writes reports/forecast/cps_qra_comparison.csv, cps_qra_bootstrap.csv, cps_qra_numbers.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.spike_onset import block_bootstrap_mean_diff  # noqa: E402

OUT = ROOT / "reports" / "forecast"
ALPHA = 0.10


def iscore(y, lo, hi):
    return (hi - lo) + 2 / ALPHA * np.maximum(lo - y, 0) + 2 / ALPHA * np.maximum(y - hi, 0)


def main():
    c = pd.read_parquet(OUT / "final_cps_test_rows.parquet")
    q = pd.read_parquet(OUT / "qra_test_forecasts.parquet")
    base = pd.read_parquet(ROOT / "data" / "processed" / "cps_base_rows.parquet")
    assert c.index.equals(q.index)
    y = c["y"].to_numpy(float)
    fresh = base.loc[c.index, "fresh_onset"].to_numpy(bool)
    masks = {"all": np.ones(len(y), bool), "calm": ~c["tail"].to_numpy(bool), "tail": c["tail"].to_numpy(bool),
             "onset": c["onset"].to_numpy(bool), "fresh_onset": fresh}
    sys_ = {
        "cps": dict(lo=c["cps_split_lo"].to_numpy(), hi=c["cps_split_hi"].to_numpy(), crps=c["cps_split_crps"].to_numpy()),
        "qra": dict(lo=q["q0.050"].to_numpy(), hi=q["q0.950"].to_numpy(), crps=q["crps_full"].to_numpy(),
                    crps_qi=q["crps_quantile_integral"].to_numpy()),
    }
    for s in sys_.values():
        s["is"] = iscore(y, s["lo"], s["hi"])
        s["cover"] = ((y >= s["lo"]) & (y <= s["hi"])).astype(float)
        s["width"] = s["hi"] - s["lo"]
    rows, boot = [], []
    for sub, m in masks.items():
        for name, s in sys_.items():
            r = {"system": name, "subset": sub, "n": int(m.sum()), "crps": s["crps"][m].mean(),
                 "coverage_90": s["cover"][m].mean(), "width_90": s["width"][m].mean(), "interval_score_90": s["is"][m].mean()}
            if "crps_qi" in s:
                r["crps_quantile_integral"] = s["crps_qi"][m].mean()
            rows.append(r)
        for metric, a, b in (("crps (QRA full)", sys_["cps"]["crps"], sys_["qra"]["crps"]),
                             ("crps (QRA quantile-integral)", sys_["cps"]["crps"], sys_["qra"]["crps_qi"]),
                             ("interval_score_90", sys_["cps"]["is"], sys_["qra"]["is"]),
                             ("coverage_90", sys_["cps"]["cover"], sys_["qra"]["cover"]),
                             ("width_90", sys_["cps"]["width"], sys_["qra"]["width"])):
            d, lo, hi = block_bootstrap_mean_diff(a - b, m)
            boot.append({"metric": metric, "subset": sub, "n": int(m.sum()), "cps": float(a[m].mean()), "qra": float(b[m].mean()),
                         "diff_cps_minus_qra": d, "boot_p05": lo, "boot_p95": hi})
    pd.DataFrame(rows).to_csv(OUT / "cps_qra_comparison.csv", index=False)
    pd.DataFrame(boot).to_csv(OUT / "cps_qra_bootstrap.csv", index=False)

    # derived numbers
    off_lo = c["cps_split_lo"] - c["regime"]; off_hi = c["cps_split_hi"] - c["regime"]
    cal = base.loc[base["split"].eq("calibration")]
    resid = (cal["y"] - cal["regime"]).to_numpy(float)
    _, counts = np.unique(resid, return_counts=True)
    fc = pd.read_parquet(OUT / "spike_forecaster_test_forecasts.parquet")
    elig = fc["eligible"].to_numpy(bool); onset = c["onset"].reindex(fc.index).to_numpy(bool)
    alert = fc["alert"].to_numpy(bool)
    meta = json.loads((OUT / "regime_cps_meta.json").read_text())
    from scripts.spike_cps import SplitCPS
    cps = SplitCPS(resid)
    ex = {}
    for t in ("2026-04-28 10:25", "2026-04-28 10:40", "2026-04-28 10:45"):
        t = pd.Timestamp(t); r = c.loc[t]; qq = q.loc[t]; f = fc.loc[t]
        ex[str(t)] = {"y": float(r["y"]), "regime": float(r["regime"]), "cps_lo": float(r["cps_split_lo"]), "cps_hi": float(r["cps_split_hi"]),
                      "cps_pit": float(r["cps_split_pit"]), "cps_crps": float(r["cps_split_crps"]),
                      "cps_prob_above_spike": float(1 - cps.cdf(np.array([170.318]), np.array([r["regime"]]))[0]),
                      "cps_prob_at_or_above_y": float(1 - cps.cdf(np.array([r["y"]]), np.array([r["regime"]]))[0]),
                      "qra_q05": float(qq["q0.050"]), "qra_q50": float(qq["q0.500"]), "qra_q95": float(qq["q0.950"]),
                      "qra_crps_full": float(qq["crps_full"]), "qra_crps_qi": float(qq["crps_quantile_integral"]),
                      "sf_p_up": float(f["p_up"]), "sf_alert": bool(f["alert"]), "sf_q05": float(f["q05"]), "sf_q50": float(f["q50"]),
                      "sf_q95": float(f["q95"]), "sf_crps": float(f["crps"])}
    numbers = {
        "regime_threshold": meta["threshold"], "regime_rule": meta["rule"], "regime_test_switch_share": meta["test_switch_share"],
        "regime_calibration_switch_share": float(np.mean(base.loc[base["split"].eq("calibration"), "regime"].to_numpy()
                                                         == base.loc[base["split"].eq("calibration"), "persistence"].to_numpy())),
        "cps_offset_lo": float(off_lo.mean()), "cps_offset_hi": float(off_hi.mean()),
        "cps_offset_constant": bool(np.allclose(off_lo, off_lo.iloc[0]) and np.allclose(off_hi, off_hi.iloc[0])),
        "n_calibration_residuals": int(len(resid)), "share_calibration_residuals_tied": float(counts[counts > 1].sum() / len(resid)),
        "n_distinct_calibration_residuals": int(len(counts)),
        "alerts": int(alert.sum()), "alerts_at_onset": int((alert & onset).sum()), "onsets": int(onset.sum()),
        "eligible": int(elig.sum()), "eligible_non_onset": int((elig & ~onset).sum()),
        "false_alerts": int((alert & ~onset).sum()), "false_alarm_rate": float((alert & ~onset).sum() / (elig & ~onset).sum()),
        "worked_example": ex,
    }
    (OUT / "cps_qra_numbers.json").write_text(json.dumps(numbers, indent=2))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(boot).round(4).to_string(index=False))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
