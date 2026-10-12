# Next steps for spike-price forecasting

The average error is not the gap to close. On the frozen test window LightGBM and 5-minute persistence are tied (MAE 4.75 against 4.74). LightGBM is better on ordinary prices (3.60 against 3.77). Persistence is better on the tails (16.00 against 18.04). Those tails are about 8% of test intervals and are the consequential miss.

The conformal band does not fix this. It is the point forecast plus two residual quantiles. Swapping the residual fence moved CRPS by about 0.04. Swapping the centre moved it by about 0.14 to 0.18. Coverage falls to zero once the point error is above $50/MWh. Report the 89.8% hit rate as calibration, not as a better price forecast.

QRA is the blend benchmark, not the spike method. It wins the shared quantile-integral CRPS (3.92 against 4.00 and 4.14) and still loses the tail to persistence (15.34 against 14.63), because a blend cannot hand the violent interval entirely to the last price.

## Regime-switch baseline

`scripts/regime_switch.py` chooses a gate on calibration only. If the absolute move over the previous 30 minutes is at least $32.48, the 90th percentile of that move in early calibration, the issued forecast is persistence. Otherwise it is the saved LightGBM model. The frozen 7-day absolute conformal fence is drawn around whichever centre was issued. The rule fires on 7.6% of test intervals.

| Model | MAE | CRPS | 90% coverage | Mean width | Tail MAE | Tail CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regime switch | 4.65 | 3.98 | 89.8% | 20.20 | 16.13 | 14.70 |
| LightGBM | 4.75 | 4.03 | 89.8% | 20.89 | 18.04 | 16.11 |
| Persistence | 4.74 | 4.18 | 89.8% | 21.83 | 16.00 | 14.77 |

This is not an onset forecast. The gate only sees a move after it has happened. A better spike model has to beat this row on calibration, then on the untouched test window.

## Data still to add

Do not add more lags of MCP, and do not add anything realised in the target interval. Same-interval demand, DPV and SCADA leak. Each new series needs a publication time, and a value cannot be used before that time.

- Pre-dispatch demand, from the WEM market data site or Dispatch API. Feature: forecast for the target interval minus last realised demand.
- Projected DPV. Feature: forecast drop over the next 5 to 30 minutes, not the level.
- Outages and not-in-service capacity known by the origin. Feature: available capacity minus forecast demand. A daily reserve-capacity number is too slow.
- Bureau temperature and cloud forecast for Perth and the SWIS, issued before the origin. Not the weather observed in the interval.
- Pre-dispatch price only if it was published before the origin. Do not take it from the same row as realised MCP.
- FCESS requirement against available raise, as a later trial. Lagged FCAS prices are already in the panel and are the lagging piece.

## Model to fit once that data is attached

Keep the absolute-error LightGBM for ordinary prices. Do not refit it and expect a tail.

- Hurdle model: a classifier for crossing the training 5th–95th band, then a size model on those rows. The probability cutoff is chosen on calibration and frozen.
- Or a second LightGBM with pinball loss at 0.95 or 0.99, used only when the gate says the interval is violent.
- Leave persistence in charge after the price is already in the tail, unless the new model beats it on that bucket.

Judge the result on the first interval of a tail event, the rest of the tail event, and tail CRPS. Not on overall MAE. Leave the conformal window frozen until the centre improves.

---

*Added 2026-10-12, after the spike forecaster, the CPS and the QRA variants. The plan above is kept as written.*

## Next steps for spike prediction forecasting

Each item says why it is worth doing (from our results and `/workspace/spike_forecasting_literature.md`), how to do it within the frozen design, and how to evaluate it. Some rules apply to every item:

- fit on train;
- choose cutoffs and windows on calibration;
- score test once;
- use each series only from its publication time;
- report CRPS, tail CRPS, first-interval CRPS, 90% coverage and width, with 1-day block bootstrap ranges against the frozen spike forecaster and against persistence.

### 1. Hazard (ACH-style) features and a regime-switching logit benchmark

- **Why:** Recent price-path features carried the only robust signal we found:
  - first-interval CRPS −0.70 [−1.14, −0.33] against the panel-only hurdle;
  - average precision 0.330 → 0.364;
  - the near-up count over 30 minutes ranked 3rd in importance.

  Hazard and self-exciting models formalise the same spike-clustering effect (Christensen, Hurn & Lindsay 2012; Clements et al. 2015). Eichler et al. (2014) recommend a regime-switching logit out of sample.
- **How:** Add features computed from prices realised by the origin:
  - exponentially decayed counts of past band crossings, with half-lives of 30 minutes, 2 hours and 1 day (decays fixed in advance, or chosen on calibration);
  - time since the last up or down spike;
  - the duration of the current spell.

  Refit the HGB classifier with the frozen capacity grid. Separately, fit a small benchmark (a logit with an ACH intensity term, or a two-state regime-switching logit) on train as a classifier.
- **Evaluate:** Average precision and Brier score on calibration and test, and first-interval CRPS once the benchmark is plugged into the same mixture. Bootstrap the differences against the frozen forecaster.

### 2. EVT (generalised Pareto) for spike size, and the down side

- **Why:** The forecaster's size models are LightGBM quantiles, fitted on few crossings. Test has 403 upward onsets and no downward ones; calibration has 2 downward onsets. Downward crossings per quarter since 2025 were 7, 1, 0, 2, 0, 0, 0. The fixed-calibration CPS variants show how unstable empirical tails are across periods (test tail width 246.7 for the fixed Mondrian CPS). Chan & Gray (2006) fit a peaks-over-threshold tail to electricity residuals.
- **How:** Fit a generalised Pareto distribution to exceedances above the training spike threshold (170.318) and below the floor (−54.19) on train. Let the scale depend on one or two origin features, or fit it unconditionally, and use it in place of the LightGBM size grid in the mixture. Choose the threshold from a mean-excess plot on train.
- **Evaluate:**
  - tail CRPS and first-interval CRPS against the frozen forecaster;
  - test coverage of upper quantiles (95%, 99%) on onsets;
  - for the down side, report only calibration and the 2 events, and state that test cannot evaluate it.

### 3. Reserve-margin and wind-forecast-error features published before the origin

- **Why:** The reserve margin is the strongest spike driver in Mount, Ning & Cai (2006), and supply-demand indices matter in Lu et al. (2005) and Liu et al. (2022). The EPWA (2025) WEM review names wind forecast error as a leading cause of unforecast high prices. Our pre-dispatch features (even-hour, hourly and half-hourly, with revisions) gave no gain beyond noise: for example, first-interval CRPS −0.09 [−0.49, +0.35] for the hourly runs. Those features are half-hourly forecasts, so a real-time margin is the untested version.
- **How:** Build, from WEM real-time data published by T−5:
  - the latest dispatch headroom (available capacity minus dispatched demand);
  - the SCADA wind output minus the most recent published wind forecast for that interval.

  Record the publication timestamp of each series and add a leakage test like `tests/test_external_features.py`. Add no new MCP lags.
- **Evaluate:** As for stage 1: bootstrap against the panel + path hurdle on overall, tail and first-interval CRPS and MAE. A null result is reported as one.

### 4. Model combination across training windows (QRA of forecasters)

- **Why:** The QRA variants show strong recency effects. On test, the 28- and 56-day windows under-cover:
  - the selected `qra:lgbm_p5@28d` covers 0.868 on test against 0.915 in the inner calibration month;
  - all-calibration windows cover 0.93.

  The calibration-to-test shift also hurt every fixed-calibration CPS variant. Cornell et al. (2023) found that combining quantile models across training-window lengths (Q-QRA) gave the lowest CRPS for 5-minute SA prices.
- **How:** Fit the spike forecaster's classifier and size models on several training windows (all training data, the last 12 months, the last 6 months). Combine their predictive quantiles with QRA or quantile averaging fitted on calibration only, by the existing `scripts/qra_variants.py` rules, with the spike forecaster and sliding conformal as members.
- **Evaluate:** Exact CRPS (from each mixture), quantile-integral CRPS for comparison with the QRA rows, 90% coverage and width on test, and first-interval CRPS against the frozen forecaster.

### 5. Probability recalibration under regime shift

- **Why:** Isotonic recalibration fitted on calibration left the test Brier score unchanged (0.008523 → 0.008532), improved log loss slightly (0.0324 → 0.0312) and lowered average precision (0.346 → 0.327). The test spike base rate was 4.4 times the calibration rate. A fixed map cannot follow this shift.
- **How:** Use a sliding or exponentially weighted recalibration on realised outcomes before the origin. Options: Platt scaling on a 7- or 28-day window, or a logistic intercept shift updated on realised base rates, like the sliding conformal residual window. Choose the window on calibration only.
- **Evaluate:**
  - Brier score, log loss and reliability diagrams on test;
  - the effect on the alert cutoff (precision and recall at the frozen 0.24);
  - first-interval CRPS when the recalibrated P(up) feeds the mixture.

### 6. More test windows and rolling-origin evaluation

- **Why:** The test window covers 4.5 months (40,127 intervals, 403 onsets, no downward spikes). Weron (2014) and Lago et al. (2021) recommend at least a year of test data, and Sheybanivaziri et al. (2024) flag short windows as a common weakness. Spike frequency differed by a factor of 4 between calibration and test, so one split is fragile.
- **How:** Re-run the whole frozen pipeline on rolling origins: refit on an expanding training window, calibrate on the next 6 months, and test on the following quarter, for every quarter with enough history. Keep the current test as one fold. Choose nothing on any test fold.
- **Evaluate:** Report per-fold CRPS, tail CRPS, first-interval CRPS and coverage, and test the pooled differences with Diebold–Mariano-type tests or the existing block bootstrap.

### 7. Live deployment notes

- **Why:** The forecaster is meant to issue P(spike) and quantiles at T−5. In production, timing and drift matter more than small score gains.
- **How:**
  - Run `scripts/spike_forecaster.py forecast` on each new dispatch interval.
  - Rebuild the residual history with `set_history` from the latest 2,016 realised intervals.
  - Log every forecast with its origin time and input publication times.
  - Monitor rolling 7-day coverage, the Brier score and the spike base rate, with an alert when coverage of the 90% interval leaves [0.85, 0.95].
  - Refit the classifier and size models quarterly, and re-select cutoffs on the latest calibration months only.
  - Treat the down side as unvalidated.
- **Evaluate:** Shadow-run it against persistence and the regime switch for at least one quarter before relying on alerts, and report the same metrics on that live period.
