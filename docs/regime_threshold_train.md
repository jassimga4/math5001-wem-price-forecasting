# Regime-switch threshold re-chosen on training data only (Casey comment #2)

- **Branch:** `regime-threshold-train`, off `main` at `6208b19`.
- **Script:** `scripts/regime_threshold_train.py`.
- **Outputs:** in `reports/forecast/`:
  - `regime_threshold_train_meta.json`
  - `regime_threshold_train_selection.csv`
  - `regime_threshold_train_comparison.csv`
  - `regime_threshold_train_bootstrap.csv`
  - `regime_threshold_train_test_rows.parquet`
- **Figure:** `/workspace/final_paper/figures/regime_threshold_train_comparison.png`. A copy is in the repo at `reports/figures/`.

## How the threshold was chosen before and now
- **Before** (`scripts/regime_switch.py`):
  - The candidates were the "already in the train tail" rule plus the 50/70/80/90/95% quantiles of the absolute 30- and 60-minute price moves, taken from the first half of calibration.
  - The rule was chosen on the second half of calibration, by lowest tail MAE, subject to MAE within 1% of LightGBM. Here "tail" means the last price was outside the training 5th–95th percentile band.
  - Chosen: 30-minute move ≥ **32.48 $/MWh**.
  - The CPS then used all calibration errors. So calibration was used twice.
- **Now:**
  - **Same grid and criterion**, but moved to the validation window at the end of train (1 Jul to 30 Sep 2025). The quantiles come from its first half, and the choice is made on its second half (from 15 Aug 2025 23:45).
  - **LightGBM for the selection:** refit on train rows before 1 Jul 2025 with the frozen hyperparameters, so its Jul–Sep predictions are out of sample. This is the same inner fit originally used to choose those hyperparameters. It reproduces the stored validation MAE exactly (10.007, `experiment_meta.json`).
  - **No calibration or test data** is read for the choice.
  - **Chosen:** 30-minute move ≥ **52.43 $/MWh** (the 95% quantile rule).
  - **Caveat:** on this window no rule kept MAE within 1% of LightGBM (LightGBM 11.725; the best rule was 11.930). So the code's existing fallback applied: the lowest tail MAE over all rules. Tail MAEs were very close, between 46.40 and 47.12 (`regime_threshold_train_selection.csv`).
- **After the choice:** the saved LightGBM (fitted on all of train) and the new rule give the regime-switch forecasts for calibration and test. The CPS is the new regime switch plus all 52,397 calibration errors. Test is scored once.
- **Check:** with the old rule, the same code reproduces the stored CPS test rows exactly (regime, range and CRPS).

## Test results (40,127 intervals; 3,206 tail; 403 spike starts)

| Metric | CPS, old threshold (32.48) | CPS, new threshold (52.43) | QRA (unchanged) |
|---|---:|---:|---:|
| Regime-switch MAE | 4.650 | 4.654 | – |
| Share of test intervals using persistence | 7.6% | 4.6% | – |
| CRPS, all | 4.006 | 3.995 | 3.949 (full) / 3.921 (quantile-integral) |
| 90% coverage, all | 0.926 | 0.925 | 0.930 |
| 90% width, all | 23.81 | 23.71 | 26.08 |
| Interval score, all | 53.44 | 52.90 | 47.79 |
| CRPS, tail | 14.68 | 14.81 | 15.41 (full) |
| 90% coverage, tail | 0.725 | 0.716 | 0.680 |
| Interval score, tail | 237.3 | 237.8 | 208.6 |
| CRPS, first interval | 41.58 | 41.60 | 37.67 (full) |
| 90% coverage, spike starts | 0.303 | 0.293 | 0.409 |
| Interval score, spike starts | 708.0 | 707.2 | 519.3 |

QRA doesn't use the regime switch. Its five inputs are 5-, 30- and 1-day persistence, ridge and LightGBM (`scripts/qra.py`, `QRA_FORECASTS`), so it is unchanged.

**90% one-day block bootstrap confidence intervals** (repo's `block_bootstrap_mean_diff`: 288-interval blocks, 1,000 draws, seed 42):

| Difference | Subset | Mean | 90% CI |
|---|---|---:|---|
| New CPS − QRA (full) | All | +0.045 | [−0.0001, +0.088] |
| New CPS − QRA (full) | Tail | −0.599 | [−1.131, −0.093] |
| New CPS − QRA (full) | First interval | +3.924 | [+3.022, +4.874] |
| New CPS − QRA (quantile-integral) | All | +0.074 | [+0.029, +0.118] |
| New CPS − QRA (quantile-integral) | Tail | −0.530 | [−1.065, −0.018] |
| New CPS − old CPS | All | −0.011 | [−0.020, −0.001] |
| New CPS − old CPS | Tail | +0.127 | [+0.072, +0.182] |
| New CPS − old CPS | First interval | +0.015 | [−0.081, +0.108] |

## Plain reading
Choosing the threshold on training data gives a higher threshold (52.43 against 32.48), so persistence is used less (4.6% of test intervals against 7.6%). The CPS hardly changes:
- slightly better overall (CRPS 3.995 against 4.006);
- slightly worse in the tail (14.81 against 14.68);
- no real change at spike starts.

The comparison with QRA keeps the same shape:
- QRA is a little better overall. On QRA's full-distribution score the 90% interval now just touches zero, so that gap is borderline.
- The CPS is still better in the tail.
- QRA is clearly better at spike starts.

## What depends on the regime switch and was NOT rerun
- **Spike forecaster:** its point forecast is the regime switch, and its spread is built on the regime switch's last 2,016 errors. The threshold is in `configs/spike_forecaster.json` (`regime_rule.threshold` 32.483). Its mixture and alert cutoffs were chosen on calibration with the old regime switch. Every spike-forecaster number in the report depends on it (29.08, 13.13, 3.915, 0.908, 24.4, recall 0.56, precision 0.32, 720 alerts). A clean rerun means rebuilding it with the new threshold and re-choosing those cutoffs on calibration. Not done; it isn't a one-line change.
- **CPS variants:** Mondrian (fixed and sliding), PIT and PIT-Mondrian (Table 3, Appendix B2) are built on regime-switch errors or the spike forecaster's PIT, so they also depend on it. Not rerun.
- **Regime-switch baselines:** the regime-switch CRPS (41.57 first interval) and its sliding spread, plus the bootstraps against the regime switch in `spike_forecaster_bootstrap.csv`, also depend on it. Not rerun.

## Report sentences and numbers that would change
If Jassim adopts the new threshold:
- **2.3:** "if the price moved by at least 32.48 $/MWh … I picked this threshold on calibration only" → 52.43, chosen on the training validation window (July–September 2025), with the fallback noted.
- **Table 2 and 3.3:**
  - regime-switch MAE 4.650 → 4.654;
  - the switch share "7.6% of test intervals" → 4.6%;
  - the bootstrap against persistence [−0.12, −0.07] needs recomputing;
  - calm and tail MAE of the regime switch (3.653 / 16.13) need recomputing (not computed here).
- **CPS numbers:** in the abstract, 2.4.1, 3.4.2–3.4.8, Tables 3, 4 and 5, Figures 2, 3, 5 and 6, and Chapter 4. Use the new CPS column above, for example:
  - CRPS 4.006 → 3.995;
  - coverage 0.926 → 0.925;
  - width 23.81 → 23.71;
  - tail 14.68 → 14.81 and 0.725 → 0.716;
  - first interval 41.58 → 41.60, spike-start coverage 30.3% → 29.3%;
  - interval scores 53.44 / 237.3 / 708.0 → 52.90 / 237.8 / 707.2.
- **CPS offsets** ("best guess minus 11.29 up to plus 12.53", the 3.4.1 text) and **the 28 April worked example** values (3.4.6) need recomputing from `regime_threshold_train_test_rows.parquet`. Not done here.
- **The calibration cross-fit CPS numbers** in 3.4.5 (0.867 coverage, 25.52 width, 4.737 CRPS) need a rerun.
- **All spike-forecaster and CPS-variant numbers** (see above), unless the report says they still use the old threshold.

## Adopted (rerun of everything downstream)
The train-chosen rule (30-minute move >= 52.428) is now the project's regime switch:
- `scripts/regime_switch.py` chooses it inside train and writes `regime_cps_meta.json`;
- `configs/spike_forecaster.json` uses it; `scripts/spike_onset.py`, `scripts/spike_forecaster.py fit/evaluate`, `scripts/spike_cps.py --rebuild`, `scripts/final_report.py` and `scripts/spike_figures.py` were rerun. The spike forecaster's calibration-chosen cutoffs came out unchanged (mixture 0.0025/0.2, alert 0.24);
- `scripts/cps_qra_comparison.py` writes the CPS-vs-QRA table, the 90% block-bootstrap intervals (CRPS, interval score, coverage, width) and the derived report numbers (`cps_qra_numbers.json`).
Docs written before this change (for example `docs/final_results_summary.md`, `docs/spike_stage1_results.md`) quote numbers from the earlier calibration-chosen threshold.
