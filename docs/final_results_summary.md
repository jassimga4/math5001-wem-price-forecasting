# Final results summary: 5-minute WEM price and spike forecasting

This file is for the paper write-up. Every number comes from a committed CSV in `reports/forecast/` or from `docs/spike_stage1_results.md`, and the source is given in each case. Nothing was fitted or tuned on the test window.

## Setup

- **Target:** the WEM market clearing price (MCP) for 5-minute interval T, forecast at origin T − 5 min. Inputs are used only from their publication time; `scripts/forecast_design.py` describes the ex-ante design.
- **Splits (frozen):**
  - train to 2025-09-30 23:55;
  - calibration 2025-10-01 to 2026-03-31;
  - test 2026-04-01 to 2026-08-18 07:55 (40,127 intervals).
  - Calibration is scored from its 2017th row on (50,381 intervals), because the sliding residual windows need 7 days of history.
- **Regimes:**
  - **tail:** the realised price is outside the training 5–95% band (−54.19 to 170.318 $/MWh). There are 3,206 tail intervals in test and 491 in calibration.
  - **first interval (onset):** the last price was inside the band and the target is outside it. There are 403 in test, all upward, and 125 in calibration.
  - **fresh onset:** an onset with no tail price in the previous hour. There are 233 in test and 84 in calibration.
  - **body:** everything else.
- **Metrics:**
  - CRPS (exact, of each method's predictive distribution);
  - coverage and mean width of the central 90% interval;
  - interval (Winkler) score at 90%: width + (2/α)·undershoot + (2/α)·overshoot, with α = 0.1;
  - MAE of the point forecast.
  - Bootstrap ranges are 90% moving-block bootstrap intervals with 1-day (288-interval) blocks, 1,000 draws and seed 42, applied to the per-interval differences.

## Methods

**Persistence + sliding residual window.** The point is the last realised price. Its predictive distribution is the point plus the last 2,016 realised residuals (7 days, before the origin). This is the project's absolute-residual sliding conformal construction (`scripts/conformal.py`). Nothing is fitted.

**LightGBM + absolute sliding conformal (frozen, PR 2).** A LightGBM absolute-error point model is fitted on train (`models/lgbm_5min_ahead.joblib`). The 90% interval uses the conformal order statistics of the last 2,016 residuals; the window was chosen on calibration. Its test coverage is 0.898 with width 20.89, the same as `conformal_test.csv`. CRPS is taken from the same residual window.

**LightGBM + normalised sliding conformal, and fixed split conformal (PR 2).** These are interval-only variants taken from `reports/forecast/conformal_test.csv`. The normalised variant divides the residuals by a lagged 1-day scale. The fixed split variant uses all 52,397 calibration residuals.

**QRA (PR 2).** Linear quantile regression averaging of the persistence (5 min, 30 min, 1 day), ridge and LightGBM forecasts, fitted on calibration (`scripts/qra.py`). The numbers come from `qra_regime.csv`. Its CRPS is a quantile-score integral over levels 0.01 to 0.99, not an exact CRPS. On that measure the LightGBM conformal scores 4.004 overall and 15.98 on the tail (exact: 4.033 and 16.11). First-interval scores were not stored.

**Regime switch.** The point is persistence when the absolute 30-minute move is at least 32.48 $/MWh, and LightGBM otherwise; the rule was chosen on calibration (`scripts/regime_switch.py`). Its distribution is the point plus the last 2,016 residuals of the switch.

**Hurdle models (spike stage 1 to B).** These apply at origins whose last price is inside the training band:

- an up-crossing classifier and a down-crossing classifier;
- LightGBM quantile models of the price size, fitted on the training crossings;
- the regime-switch residual window, with probability weight p moved onto the size grid when a classifier passes a mixture cutoff.

Classifier capacity is chosen on an inner validation window inside train. The mixture, point and detection cutoffs are chosen on calibration. The variants differ in features and classifier: panel only; + weather; + pre-dispatch at even-hour, hourly or half-hourly frequency; + price path; and LightGBM, XGBoost, HistGradientBoosting (HGB) or logistic classifiers (`scripts/spike_onset.py`). Only CRPS and MAE were stored per row, so these rows have no interval columns.

**Spike forecaster (frozen).** This is the hurdle chosen on calibration (`configs/spike_forecaster.json`, `scripts/spike_forecaster.py`):

- 28 panel and price-path features;
- an HGB up classifier (learning rate 0.05, 31 leaves, 100 iterations) and an HGB down classifier (learning rate 0.05, 15 leaves, 100 iterations);
- LightGBM quantile size models;
- mixture cutoffs 0.0025 (up) and 0.2 (down);
- a spike alert when max(P up, P down) ≥ 0.24;
- the regime switch as the point forecast.

The module reproduces the frozen comparison row to 1.1e-16. Its 90% interval is the 5% to 95% quantiles of the mixture.

**CPS: split, around the regime-switch point (`cps_split`).** This is a split conformal predictive system (Vovk et al., 2019) on additive residuals. With n residuals r_i = y_i − ŷ_i from calibration, Q(y) = (#{r_i < y − ŷ} + τ(#{r_i = y − ŷ} + 1)) / (n + 1), with τ = 0.5. The 90% interval uses the conformal ranks floor((n+1)α/2) and ceil((n+1)(1 − α/2)). It is built on all calibration rows and frozen for test. Its sliding counterpart, the last 2,016 residuals before the origin, is the regime-switch row.

**CPS: Mondrian by spike risk (`cps_mondrian`, fixed calibration).** The same CPS is fitted separately within spike-risk categories that are known at the origin:

- the last price at or above the spike threshold (544 calibration rows);
- the last price at or below the floor (4 rows, so it uses the pooled residuals);
- eligible origins split by the forecaster's P(spike up) into [0, 0.01), [0.01, 0.05), [0.05, 0.2) and ≥ 0.2 (49,741, 1,369, 508 and 231 rows).

The edges were fixed before scoring, and the 231-row top bin joins the 0.05–0.2 bin because it has fewer than 300 rows (`final_cps_categories.csv`). This is the textbook way for a CPS to "predict spikes": a high-risk origin gets the heavy right tail of the residuals seen at high-risk origins.

**CPS: Mondrian by spike risk, sliding windows (`cps_mondrian_sliding`).** This uses the same categories, but each origin uses the last K realised residuals of its own category from before the origin: K = 2,016 for the low-risk bin and K = 300 for the others. It follows the project's frozen sliding conformal convention, which uses realised test-period residuals once they are past. A category with fewer than 30 earlier residuals falls back to the pooled last 2,016. It adapts to a change in spike size between calibration and test.

**CPS: PIT-recalibrated spike forecaster (`cps_pit`, `cps_pit_mondrian`).** This is a CPS on the forecaster's own predictive distribution F, with conformity score F(y) (conformal recalibration). The recalibrated CDF is G(F(y)), where G is the empirical CDF of the calibration PIT values. It is computed globally for `cps_pit` and per risk category for `cps_pit_mondrian`. The 90% interval is [F⁻¹(u_lo), F⁻¹(u_hi)], where u_lo and u_hi are conformal-rank order statistics of the calibration PITs. CRPS is exact on the recalibrated atoms.

**Isotonic recalibration of P(spike up).** Isotonic regression of the up-crossing label on the forecaster's P(up), fitted on eligible calibration origins (`final_isotonic_map.json`).

**Calibration-period scores of the CPS variants are cross-fitted.** Calibration is split in two halves by time, and each half is scored with a CPS built on the other half. Test uses the CPS built on all calibration. The spike forecaster's calibration scores are not cross-fitted, because its cutoffs were chosen on calibration. Code: `scripts/spike_cps.py`. Tests: `tests/test_spike_cps.py`.

## Master tables

Missing values (–) mean the metric was not stored for that method. Hurdle rows other than the frozen forecaster come from `spike_stage1_results.csv`, which has the same rows and CRPS code but no stored intervals. The normalised and fixed split conformal rows come from `conformal_test.csv` (interval only). QRA comes from `qra_regime.csv`: it covers the same 40,127 test intervals, but its CRPS is the quantile integral (see Methods). QRA and the PR 2 interval variants have no calibration rows, because they were fitted or selected on calibration. MAE is not given for the PIT variants, because their point is unchanged from the forecaster. Full long-format tables: `reports/forecast/final_master_table.csv` and `final_cps_metrics.csv`.

#### Test

| Model | CRPS | Tail CRPS | 1st-int CRPS | Cov 90 | Width 90 | Tail cov | Tail width | Onset cov | Onset width | Interval score 90 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence + sliding residual window | 4.176 | 14.77 | 42.50 | 0.898 | 21.8 | 0.738 | 24.8 | 0.308 | 26.0 | 55.7 | 4.745 |
| LightGBM + absolute sliding conformal (frozen) | 4.033 | 16.11 | 41.29 | 0.898 | 20.9 | 0.679 | 24.6 | 0.315 | 25.1 | 52.1 | 4.753 |
| LightGBM + normalised sliding conformal | – | – | – | 0.898 | 22.1 | – | – | – | – | – | – |
| LightGBM + fixed split conformal | – | – | – | 0.923 | 23.9 | – | – | – | – | – | – |
| QRA (calibration-fitted) | 3.921 | 15.34 | – | 0.930 | 26.1 | 0.680 | 35.3 | – | – | – | – |
| Regime switch + sliding residual window | 3.975 | 14.70 | 41.57 | 0.898 | 20.2 | 0.719 | 23.2 | 0.293 | 24.1 | 52.0 | 4.650 |
| Hurdle, panel only (LightGBM) | 3.917 | 13.29 | 30.37 | – | – | – | – | – | – | – | 4.640 |
| Hurdle + weather | 3.921 | 13.32 | 30.55 | – | – | – | – | – | – | – | 4.709 |
| Hurdle + weather + pre-dispatch (even-hour) | 3.925 | 13.35 | 30.80 | – | – | – | – | – | – | – | 4.831 |
| Hurdle + weather + pre-dispatch (hourly, revisions) | 3.926 | 13.28 | 30.28 | – | – | – | – | – | – | – | 4.869 |
| Hurdle + weather + pre-dispatch (half-hourly, revisions) | 3.926 | 13.33 | 30.68 | – | – | – | – | – | – | – | 4.854 |
| Hurdle, panel + price path (LightGBM) | 3.914 | 13.20 | 29.66 | – | – | – | – | – | – | – | 4.689 |
| Hurdle, panel + path (XGBoost) | 3.912 | 13.18 | 29.44 | – | – | – | – | – | – | – | 4.650 |
| Hurdle, panel + path (HistGradientBoosting) | 3.915 | 13.13 | 29.08 | – | – | – | – | – | – | – | 4.695 |
| Hurdle, panel + path (logistic) | 3.947 | 13.82 | 34.54 | – | – | – | – | – | – | – | 4.635 |
| Hurdle, panel + path (LightGBM, uncapped) | 3.914 | 13.20 | 29.66 | – | – | – | – | – | – | – | 4.689 |
| Hurdle, panel + path (HGB, wide grid) | 3.911 | 13.12 | 29.01 | – | – | – | – | – | – | – | 4.650 |
| Spike forecaster (frozen) | 3.915 | 13.13 | 29.08 | 0.908 | 24.4 | 0.785 | 38.0 | 0.826 | 143.0 | 49.6 | 4.650 |
| CPS: split, regime-switch point | 4.006 | 14.68 | 41.58 | 0.926 | 23.8 | 0.725 | 23.8 | 0.303 | 23.8 | 53.4 | 4.650 |
| CPS: Mondrian by spike risk (fixed calibration) | 4.388 | 19.51 | 34.63 | 0.960 | 52.2 | 0.966 | 246.7 | 0.893 | 145.7 | 60.9 | 4.650 |
| CPS: Mondrian by spike risk (sliding windows) | 3.885 | 14.29 | 37.78 | 0.900 | 25.4 | 0.891 | 86.8 | 0.623 | 54.9 | 46.4 | 4.650 |
| CPS: PIT-recalibrated spike forecaster | 3.909 | 13.16 | 29.15 | 0.905 | 23.5 | 0.779 | 36.2 | 0.811 | 133.4 | 49.3 | – |
| CPS: PIT-recalibrated, Mondrian by spike risk | 4.452 | 21.04 | 29.96 | 0.931 | 55.6 | 0.964 | 358.8 | 0.841 | 153.3 | 65.7 | – |

#### Calibration

| Model | CRPS | Tail CRPS | 1st-int CRPS | Cov 90 | Width 90 | Tail cov | Tail width | Onset cov | Onset width | Interval score 90 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence + sliding residual window | 4.794 | 62.76 | 102.97 | 0.902 | 28.1 | 0.462 | 28.5 | 0.016 | 33.5 | 61.2 | 5.562 |
| LightGBM + absolute sliding conformal (frozen) | 4.381 | 62.16 | 99.48 | 0.902 | 24.8 | 0.385 | 25.4 | 0.008 | 29.5 | 53.6 | 5.268 |
| Regime switch + sliding residual window | 4.550 | 62.35 | 100.39 | 0.902 | 25.3 | 0.450 | 25.7 | 0.016 | 30.0 | 57.2 | 5.422 |
| Hurdle, panel only (LightGBM) | 4.511 | 55.97 | 75.34 | – | – | – | – | – | – | – | 5.437 |
| Hurdle + weather | 4.513 | 56.42 | 77.10 | – | – | – | – | – | – | – | 5.448 |
| Hurdle + weather + pre-dispatch (even-hour) | 4.508 | 56.27 | 76.52 | – | – | – | – | – | – | – | 5.455 |
| Hurdle + weather + pre-dispatch (hourly, revisions) | 4.509 | 56.17 | 76.11 | – | – | – | – | – | – | – | 5.463 |
| Hurdle + weather + pre-dispatch (half-hourly, revisions) | 4.508 | 56.32 | 76.70 | – | – | – | – | – | – | – | 5.459 |
| Hurdle, panel + price path (LightGBM) | 4.507 | 55.91 | 75.09 | – | – | – | – | – | – | – | 5.445 |
| Hurdle, panel + path (XGBoost) | 4.508 | 55.67 | 74.16 | – | – | – | – | – | – | – | 5.422 |
| Hurdle, panel + path (HistGradientBoosting) | 4.508 | 55.47 | 73.36 | – | – | – | – | – | – | – | 5.458 |
| Hurdle, panel + path (logistic) | 4.530 | 58.41 | 84.92 | – | – | – | – | – | – | – | 5.434 |
| Hurdle, panel + path (LightGBM, uncapped) | 4.507 | 55.91 | 75.08 | – | – | – | – | – | – | – | 5.447 |
| Hurdle, panel + path (HGB, wide grid) | 4.512 | 56.01 | 75.49 | – | – | – | – | – | – | – | 5.423 |
| Spike forecaster (frozen) | 4.508 | 55.47 | 73.37 | 0.905 | 26.5 | 0.593 | 67.5 | 0.576 | 194.8 | 55.0 | 5.422 |
| CPS: split, regime-switch point | 4.737 | 62.81 | 101.84 | 0.867 | 25.5 | 0.446 | 24.5 | 0.040 | 21.5 | 68.7 | 5.422 |
| CPS: Mondrian by spike risk (fixed calibration) | 4.659 | 60.28 | 91.00 | 0.868 | 27.2 | 0.760 | 213.9 | 0.360 | 91.4 | 62.2 | 5.422 |
| CPS: Mondrian by spike risk (sliding windows) | 4.440 | 59.77 | 82.14 | 0.902 | 27.5 | 0.823 | 231.7 | 0.672 | 137.1 | 50.5 | 5.422 |
| CPS: PIT-recalibrated spike forecaster | 4.506 | 55.57 | 73.69 | 0.901 | 25.8 | 0.587 | 65.1 | 0.560 | 187.1 | 54.9 | – |
| CPS: PIT-recalibrated, Mondrian by spike risk | 4.429 | 57.32 | 76.20 | 0.899 | 29.9 | 0.798 | 416.2 | 0.528 | 203.1 | 51.2 | – |

## Bootstrap ranges (test; model minus baseline; 90%, 1-day blocks)

Against the spike forecaster (test):

| Model | CRPS | Tail CRPS | First-interval CRPS |
| --- | ---: | ---: | ---: |
| CPS: split, regime-switch point | +0.091 [+0.066, +0.117] | +1.552 [+1.231, +1.932] | +12.503 [+10.034, +15.522] |
| CPS: Mondrian by spike risk (fixed calibration) | +0.473 [+0.377, +0.580] | +6.383 [+5.996, +6.752] | +5.546 [+3.573, +7.962] |
| CPS: Mondrian by spike risk (sliding windows) | -0.029 [-0.051, -0.008] | +1.164 [+0.894, +1.462] | +8.698 [+6.387, +11.527] |
| CPS: PIT-recalibrated spike forecaster | -0.006 [-0.006, -0.005] | +0.024 [+0.020, +0.030] | +0.073 [+0.043, +0.109] |
| CPS: PIT-recalibrated, Mondrian by spike risk | +0.537 [+0.368, +0.745] | +7.913 [+6.678, +9.072] | +0.882 [+0.714, +1.058] |

Against persistence (test):

| Model | CRPS | Tail CRPS | First-interval CRPS |
| --- | ---: | ---: | ---: |
| Spike forecaster (frozen) | -0.261 [-0.299, -0.221] | -1.637 [-2.103, -1.251] | -13.417 [-16.717, -10.696] |
| CPS: split, regime-switch point | -0.170 [-0.197, -0.143] | -0.084 [-0.242, +0.067] | -0.914 [-1.324, -0.579] |
| CPS: Mondrian by spike risk (fixed calibration) | +0.212 [+0.110, +0.328] | +4.746 [+4.138, +5.306] | -7.871 [-9.081, -6.792] |
| CPS: Mondrian by spike risk (sliding windows) | -0.290 [-0.325, -0.253] | -0.473 [-0.753, -0.218] | -4.719 [-5.341, -4.160] |
| CPS: PIT-recalibrated spike forecaster | -0.267 [-0.304, -0.227] | -1.612 [-2.077, -1.229] | -13.344 [-16.630, -10.640] |
| CPS: PIT-recalibrated, Mondrian by spike risk | +0.276 [+0.096, +0.493] | +6.276 [+4.789, +7.609] | -12.534 [-15.744, -9.902] |

Earlier comparisons (test, from `spike_stage1_bootstrap.csv`), as overall CRPS / tail CRPS / first-interval CRPS / overall MAE:

- Panel-only hurdle against persistence: −0.26 [−0.29, −0.22] / −1.48 [−1.90, −1.12] / −12.13 [−14.99, −9.77] / −0.10 [−0.14, −0.06].
- Against the panel-only hurdle:
  - weather: +0.00 [+0.00, +0.01] / +0.02 [−0.02, +0.07] / +0.19 [−0.12, +0.52] / +0.07 [+0.04, +0.10];
  - pre-dispatch even-hour: +0.01 [+0.00, +0.01] / +0.05 [+0.01, +0.11] / +0.44 [+0.07, +0.90] / +0.19 [+0.14, +0.24];
  - pre-dispatch hourly with revisions: +0.01 [+0.00, +0.01] / −0.01 [−0.06, +0.04] / −0.09 [−0.49, +0.35] / +0.23 [+0.17, +0.29];
  - pre-dispatch half-hourly with revisions: +0.01 [+0.00, +0.01] / +0.04 [−0.01, +0.10] / +0.32 [−0.09, +0.83] / +0.21 [+0.15, +0.28];
  - price path (LightGBM): −0.00 [−0.01, +0.00] / −0.09 [−0.14, −0.04] / −0.70 [−1.14, −0.33] / +0.05 [+0.02, +0.08].
- Against the LightGBM panel + path hurdle:
  - HGB: +0.00 [−0.00, +0.00] / −0.07 [−0.11, −0.04] / −0.58 [−0.90, −0.30] / +0.01 [−0.01, +0.02];
  - XGBoost: −0.00 [−0.01, +0.00] / −0.03 [−0.07, +0.01] / −0.22 [−0.52, +0.08] / −0.04 [−0.08, +0.00];
  - logistic: +0.03 [+0.02, +0.04] / +0.61 [+0.47, +0.79] / +4.88 [+3.84, +6.08] / −0.05 [−0.09, −0.02].

## Isotonic recalibration of P(spike up) (`final_isotonic.csv`)

| Split | Probability | Eligible origins | Up-crossings | Brier | Log loss | Average precision | Mean P | Base rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration (cross-fitted) | raw | 49,890 | 123 | 0.001863 | 0.00779 | 0.389 | 0.0035 | 0.0025 |
| Calibration (cross-fitted) | isotonic | 49,890 | 123 | 0.002017 | 0.00911 | 0.301 | 0.0028 | 0.0025 |
| Test | raw | 36,921 | 403 | 0.008523 | 0.03240 | 0.346 | 0.0169 | 0.0109 |
| Test | isotonic | 36,921 | 403 | 0.008532 | 0.03117 | 0.327 | 0.0141 | 0.0109 |

Isotonic recalibration brings the low and middle probability bins closer to the diagonal on test (`final_reliability.csv`). For example, the raw 0.02–0.05 bin had a mean P of 0.032 against an observed 0.014; after recalibration the bin had a mean of 0.029 against 0.023. Log loss improves (0.0324 to 0.0312), but the Brier score is unchanged to 4 decimal places (0.00852 to 0.00853). Average precision falls (0.346 to 0.327), because the step function ties many origins. On cross-fitted calibration every score is worse. The base rate of upward crossings at eligible origins is 4.4 times higher in test (1.09%) than in calibration (0.25%).

## Figures (`reports/figures/final/`, PNG and PDF; `python scripts/final_report.py`)

- `master_comparison.png`: test CRPS, tail CRPS and first-interval CRPS for the baselines, the panel-only hurdle, the spike forecaster and the five CPS variants.
- `coverage_width_by_regime.png`: test coverage and mean width of the 90% interval for all intervals, body, tail and first interval (width on a log scale).
- `pit_histograms.png`: test PIT histograms for the regime switch, the spike forecaster, the five CPS variants and the LightGBM conformal.
- `reliability_isotonic.png`: reliability of P(spike up) before and after isotonic recalibration, on cross-fitted calibration and on test, with Brier score and log loss in the legend.
- `spike_episode.png`: the largest upward test onset (2026-04-28 10:40, 976 $/MWh), with the 90% bands of the spike forecaster, the sliding Mondrian CPS, the PIT CPS and the LightGBM conformal.

Earlier figures: `reports/figures/spike/` (forecaster reliability, three episodes, first-interval CRPS).

## Findings

1. **First interval of a spike: the spike forecaster wins.** On test it scores 29.08 first-interval CRPS, against 42.50 for persistence (−13.42 [−16.72, −10.70]) and 41.57 for the regime switch (−12.49 [−15.53, −9.91]). It covers 82.6% of onsets with its 90% interval; the residual-window methods cover 29–32%.
2. **No CPS variant beats the spike forecaster at the first interval.**
   - The PIT-recalibrated forecaster is practically identical: first-interval CRPS +0.07 [+0.04, +0.11], tail CRPS +0.02 [+0.02, +0.03], overall CRPS −0.006 [−0.006, −0.005].
   - The Mondrian CPS variants are clearly worse at onset: +5.5 (fixed) and +8.7 (sliding).
   - The split CPS is far worse: +12.5, close to the residual-window baselines, because it has no spike information.
3. **Overall CRPS and calibration: the sliding Mondrian CPS has the best overall scores.** It has the lowest overall CRPS (3.885; −0.029 [−0.051, −0.008] against the forecaster, −0.290 [−0.325, −0.253] against persistence) and the lowest interval score (46.4 against 49.6). Its tail coverage (0.891) is the closest to nominal. Its tail intervals are wide (86.8 $/MWh), and it is worse on tail CRPS (+1.16) and first-interval CRPS (+8.70) than the forecaster. Its gain comes from the body: body CRPS is 2.982, against 3.114 for the forecaster and 3.044 for the regime switch. Its low-risk category uses only residuals from calm origins, so body intervals are tighter.
4. **Calibration-period CPS does not carry over to test unchanged.** Spikes were much larger in calibration than in test: tail MAE of persistence was 65.35 in calibration against 16.00 in test. Fixed-calibration Mondrian categories, built from residuals of those larger spikes, give over-wide tail intervals on test (tail width 246.7, coverage 0.966) and the worst tail CRPS (19.51 fixed; 21.04 for the PIT Mondrian). The fixed split CPS over-covers on test (0.926) and under-covers on cross-fitted calibration (0.867). Sliding windows, which use realised residuals before each origin, adapt and stay near 0.90 overall.
5. **Body intervals.** Every residual-window method and the CPS variants cover 0.90–0.96 in the body. The spike forecaster's 90% interval is wider overall (24.4 against 20.2 for the regime switch) and covers 0.908, because its mixture adds spike mass at risky origins.
6. **Point accuracy.** The regime switch has the lowest test MAE among the issued points (4.650; the panel-only hurdle's override gives 4.640, which is not significantly different, −0.01 [−0.03, +0.01] in stage 1c). The spike forecaster and the residual CPS variants issue the regime-switch point.
7. **External data.**
   - Weather (17 features) and AEMO pre-dispatch at even-hour, hourly with revisions, and half-hourly with 30/60-minute revisions never beat the panel-only hurdle on any probabilistic metric beyond noise. Their overall MAE was significantly worse (+0.07 to +0.23).
   - Pre-dispatch revision features did not rank in the up classifier's top 15.
   - Pre-dispatch is solved half-hourly on forecast inputs, while spikes start inside the half hour.
8. **Price-path features help.** Nine features from prices realised by the origin (counts near the band, the 15-minute slope, run lengths, the 30-minute spread, the 60-minute high minus the last price) improve tail CRPS (−0.09 [−0.14, −0.04]) and first-interval CRPS (−0.70 [−1.14, −0.33]) over the panel only, and raise AP from 0.330 to 0.364. Five of them rank in the up classifier's top 15 (the near-up count over 30 minutes is 3rd).
9. **Classifiers.** HGB beats LightGBM on tail CRPS (−0.07 [−0.11, −0.04]) and first-interval CRPS (−0.58 [−0.90, −0.30]). XGBoost is level with LightGBM, and logistic regression is clearly worse (+4.88 first-interval CRPS). The default inner validation window had no downward crossings, so down-classifier capacity was chosen on 2024 Q4 instead.
10. **Probabilities.** The forecaster's raw P(spike up) overstates low risks on test. Isotonic recalibration on calibration improves log loss slightly, leaves the Brier score unchanged and lowers AP. It is not a clear improvement, because the spike base rate in test was 4.4 times the calibration rate.

## Limitations

- **No downward spikes in test and 2 in calibration.** Every down-side component (classifier, categories, recalibration) is effectively untested. Downward crossings almost stopped after 2024.
- **Distribution shift.** Calibration spikes were larger and rarer than test spikes. Methods frozen on calibration, such as fixed-calibration CPS and isotonic recalibration, carry that shift into test. Sliding-window methods adapt but use realised test-period residuals (only before each origin), as the frozen PR 2 conformal does.
- **Serial dependence.** Conformal coverage guarantees assume exchangeability, which 5-minute prices do not satisfy. Coverage here is empirical. Bootstrap ranges use 1-day blocks to respect within-day dependence.
- **Coverage of the hurdle rows.** Hurdle rows other than the frozen forecaster have no stored intervals, so only their CRPS and MAE are compared.
- **QRA's CRPS is a truncated quantile integral, not an exact CRPS.**
- **The calibration scores of the spike forecaster are not out of sample**, because its cutoffs were chosen there. The calibration scores of the CPS variants are cross-fitted halves.
