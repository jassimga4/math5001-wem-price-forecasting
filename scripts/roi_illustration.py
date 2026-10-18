"""Illustrative demand-response value of 5-minute spike forecasts (test window).

ILLUSTRATIVE ONLY. The load size, opportunity cost and alert rule below are
assumptions, not client data, and the rule was fixed in advance (not tuned on
test). Prices and forecasts are the real test-window values written by
scripts/lgbm_vs_qra.py (reports/forecast/lgbm_vs_qra_test_forecasts.parquet).

Strategy: a flexible industrial load of LOAD_MW that pays the 5-minute spot
price curtails for one dispatch interval whenever its alert fires. Curtailing
saves the spot price but loses production worth OPP_COST $/MWh. Value per
curtailed interval = LOAD_MW * (5/60) * (price - OPP_COST); a false alarm in a
cheap interval has negative value.

    value = sum over alerted intervals of LOAD_MW * (5/60) * (MCP_t - OPP_COST)
    annualised = value * 365 / test_days
    ROI = (annualised value of forecast rule - annualised value of reactive rule
           - annual run cost) / annual run cost
    break-even load (MW) = annual run cost / (annualised uplift per MW)

Alert rules compared:
  * reactive (no forecast): alert if the last realised price >= spike threshold
  * forecast: alert if the model's 95th-percentile forecast >= spike threshold
  * perfect foresight: curtail exactly when MCP > OPP_COST (the upper bound
    for this payoff)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "forecast"

LOAD_MW = 10.0          # assumption: curtailable load
OPP_COST = 150.0        # assumption: value of lost production, $/MWh
RUN_COST = 150_000.0    # assumption: annual cost to run the forecast + control integration, $
SPIKE = 170.318         # train p95 of MCP (reports/forecast/qra_meta.json)


def main() -> None:
    d = pd.read_parquet(OUT / "lgbm_vs_qra_test_forecasts.parquet")
    last = pd.read_parquet(OUT / "final_cps_test_rows.parquet")["persistence"].reindex(d.index)
    y = d["y"].to_numpy(float)
    days = (d.index.max() - d.index.min()).total_seconds() / 86400 + 5 / 1440
    rules = {
        "reactive (last price)": last.to_numpy(float) >= SPIKE,
        "QRA q95": d["qra_q0.95"].to_numpy(float) >= SPIKE,
        "LightGBM lag (train) q95": d["lgbm_train_q0.95"].to_numpy(float) >= SPIKE,
        "LightGBM lag (train+cal) q95": d["lgbm_traincal_q0.95"].to_numpy(float) >= SPIKE,
        "perfect foresight (price > opp. cost)": y > OPP_COST,
    }
    spike = d["tail"].to_numpy(bool)
    onset = d["onset"].to_numpy(bool)
    rows = []
    for name, alert in rules.items():
        value = float(np.sum(LOAD_MW * (5 / 60) * (y[alert] - OPP_COST)))
        rows.append({"rule": name, "alerts": int(alert.sum()),
                     "alerts_in_spikes": int((alert & spike).sum()),
                     "spike_recall": float((alert & spike).sum() / spike.sum()),
                     "onset_recall": float((alert & onset).sum() / onset.sum()),
                     "precision": float((alert & spike).sum() / max(alert.sum(), 1)),
                     "value_test_window": value, "value_annualised": value * 365 / days})
    table = pd.DataFrame(rows)
    reactive = table.set_index("rule").loc["reactive (last price)", "value_annualised"]
    table["uplift_vs_reactive_annualised"] = table["value_annualised"] - reactive
    table["uplift_per_mw_annualised"] = table["uplift_vs_reactive_annualised"] / LOAD_MW
    table["roi_vs_reactive"] = (table["uplift_vs_reactive_annualised"] - RUN_COST) / RUN_COST
    per_mw = table["uplift_per_mw_annualised"].where(table["uplift_per_mw_annualised"] > 0)
    table["break_even_load_mw"] = RUN_COST / per_mw
    table.to_csv(OUT / "roi_illustration.csv", index=False)
    meta = {"ILLUSTRATIVE": True, "load_mw": LOAD_MW, "opportunity_cost_per_mwh": OPP_COST,
            "annual_run_cost": RUN_COST, "spike_threshold": SPIKE, "test_days": days,
            "test_window": [str(d.index.min()), str(d.index.max())],
            "caveat": "Apr-Aug 2026 only (autumn/winter); annualising assumes the same spike rate all year. "
                      "One-interval curtailment, no ramp limits, no rebound, no network or retail tariff effects."}
    (OUT / "roi_illustration_meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
