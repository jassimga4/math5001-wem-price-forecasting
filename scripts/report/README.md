# Report figures and checks

Scripts that draw the figures in the MATH5001 final report and the checks behind the supervisor-comment answers. They only read stored results in `reports/forecast/` (and `data/processed/cps_base_rows.parquet`); nothing is refitted.

| Script | Output |
| --- | --- |
| `fig1_point_mae.py` | Figure 1, point-forecast MAE |
| `make_figs.py` | Figures 2, 3 and 5 (CPS vs QRA coverage/width, CRPS and interval score with bootstrap CIs, variants) |
| `fig4_worked_example.py` | Figure 4, worked example 28 Apr 2026 |
| `fig_cdf_first_spike.py` | CDF of CPS and QRA at 10:40 on 28 Apr 2026 (not in the report) |
| `qra_rearrangement_check.py` | QRA scores before and after sorting the 23 levels (`reports/forecast/qra_rearrangement_check.csv`) |
| `cps_quantiles_vs_qra.py` | Like-for-like CRPS check: CPS exact vs QRA full curve, and both on the same 0.01-0.99 grid; coverage, width, interval score and pinball per level with 90% day-block bootstrap CIs (`reports/forecast/cps_quantiles_vs_qra.csv`, `cps_quantiles_vs_qra_pinball_by_level.csv`); matched-level figure |
| `cps_vs_qra_bootstrap.py` | CPS-vs-QRA bootstrap check (`reports/forecast/casey_cps_vs_qra_bootstrap.csv`); the report uses `scripts/cps_qra_comparison.py` |

Run from the repo root, for example `REPORT_FIG_DIR=out/ python scripts/report/make_figs.py`. Default output folder: `reports/figures/report/`.
Exact package versions used: `requirements-lock.txt` (Python 3.12).
