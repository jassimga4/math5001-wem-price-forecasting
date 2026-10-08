# Spike stage 1: external data and an onset model

Stage 1 asks one question: does data published before the origin help call the first interval of a price spike, where persistence and the regime switch can only react afterwards? Rules are in `docs/spike_forecast_next_steps.md`. No extra MCP lags, nothing realised in the target interval, train on train, choose cutoffs on calibration, score test once.

Scripts: `scripts/pull_open_meteo.py`, `scripts/pull_aemo_predispatch.py`, `scripts/external_features.py`, `scripts/spike_onset.py`. Outputs: `reports/forecast/spike_stage1_*.csv`, `spike_stage1_meta.json`, `spike_predispatch_coverage.csv`.

## Sources and availability rules

All times are naive AWST (UTC+8), the same clock as the panel. The origin of a target interval T is T minus 5 minutes. A value is used at an origin only if its availability time is at or before that origin.

| Source | What is used | Availability rule |
| --- | --- | --- |
| Open-Meteo Previous Runs API, `previous_day1`, ECMWF IFS 0.25° and GFS (mean of the two) | Hourly temperature, cloud, shortwave radiation, 100 m wind at Perth metro and four wind regions | Value valid at V usable from V − 12 h. It comes from a run initialised 24–29 h before V, which reaches the API 4–6 h after initialisation. |
| AEMO WEM Reference pre-dispatch, even-hour runs | System-level fields: energy and FCESS prices, energy requirement, in-service and available capacity, contingency-raise requirement and availability, energy deficit | Run labelled R usable from max(R + 40 min, issue time + 25 min). On the live folder the file appears about 35 min after R and 20 min after `dispatchDataIssueID`. The latest usable run is taken; runs older than 4 h count as missing. |

Other sources that were checked and left out (Bureau ADFD, outage files, load forecasts, market schedule CSVs) are listed with reasons in `data/external/README.md`. The schedule CSVs carry realised same-interval values and would leak.

## Date coverage

Weather: GFS from the start of the panel, ECMWF from February–March 2024; both complete from March 2024 to 2026-08-19. Models are fitted from 2024-03-01, so the fitting rows and calibration and test are 100% covered (see `spike_stage1_meta.json`, `external_non_null_share`).

Pre-dispatch: 1,053 trading-day ZIPs from 2023-09-30 to 2026-08-17, no missing day. 12,561 of 12,636 even-hour runs were extracted. The 75 missing runs are in `data/external/aemo_predispatch/missing_runs.csv`. The pull logged no download failures, so these runs are not in the archive ZIPs. 69 fall in train (including 2023-12-19 16:00 to 2023-12-21 08:00 and 2024-02-22 10:00 to 2024-02-23 12:00), 6 in calibration, none in test.

| Split | Targets | Origins with a usable pre-dispatch run | Even-hour runs present / expected |
| --- | --- | ---: | ---: |
| Train | 2023-10-02 08:00 to 2025-09-30 23:55 | 99.28% | 8,687 / 8,756 |
| Calibration | 2025-10-01 00:00 to 2026-03-31 23:55 | 99.84% | 2,178 / 2,184 |
| Test | 2026-04-01 00:00 to 2026-08-18 07:55 | 100.00% | 1,672 / 1,672 |

On the hurdle fitting rows (from 2024-03-01) pre-dispatch is 99.7% non-null. The run used is 40 to 240 minutes old at the origin, median 105 minutes.

The committed `data/external/aemo_predispatch/predispatch_runs_first9h.parquet` (about 11 MB) holds the first 9 hours of each run. Features read at most 6 hours past a run label, so it gives the same features as the full per-day extracts (81 MB, gitignored). `python scripts/pull_aemo_predispatch.py --pass 1` then `--consolidate` rebuilds it.

### Why the first pre-dispatch row equalled the weather row

The first run was made while the pre-dispatch pull was still going. Only 3% of origins had a pre-dispatch value then, none of them in the training window. LightGBM never split on an all-missing column, so `hurdle_weather_predispatch` came out identical to `hurdle_weather`. `spike_onset.py` now raises an error if any external feature is under 90% non-null on the fitting rows. The rerun reproduces the panel and weather rows exactly, and the pre-dispatch row now differs.

## Features

- Panel (`hurdle_panel`): the existing tree features (MCP lags at 5, 30, 60 min and 1 day, lagged demand, DPV, withdrawal, FCESS prices, last complete real-time price, STEM price and imbalance, hour, day of week, month) plus the absolute 30 and 60 minute price moves.
- Weather (`wx_*`, 17 features): Perth temperature at T, its 60 min change and 3 h max; Perth cloud and its 60 min change; Perth radiation at T and its drop from the origin to T, T + 30 and T + 60 min; Merredin radiation; 100 m wind at four sites; a crude wind capacity-factor proxy at T and its change over 60 and 180 min. Hourly forecasts are interpolated linearly to 5 minutes.
- Pre-dispatch (`pd_*`, 14 features): forecast energy price for the half hour containing T, the next half hour and the max over the next 2 hours; forecast price minus the last realised price; energy requirement, requirement minus last realised demand, and its 30 min ramp; in-service capacity minus requirement (headroom); available capacity; contingency-raise margin; regulation and contingency raise prices; energy deficit; age of the run used.

## Models

- Persistence: the last price.
- LightGBM: the saved absolute-error point model.
- Regime switch: persistence when the absolute 30 min move is at least $32.48, otherwise LightGBM (`scripts/regime_switch.py`).
- Hurdle: at origins where the last price is inside the training 5th–95th band (−$54.19 to $170.32), a LightGBM classifier for an upward crossing and one for a downward crossing, then LightGBM quantile models of the price size, fitted on training rows that crossed. Fitted on train from 2024-03-01, with early stopping on 2025-07-01 to 2025-09-30. Everywhere else the issued forecast is the regime switch.
- Point forecast: the size model's median replaces the regime switch when a classifier is above its cutoff.
- Predictive distribution: the regime switch's 7-day absolute-residual window, with probability p moved onto the size model's quantile grid when a classifier is above its mixture cutoff. Baseline CRPS uses each model's own 7-day residual window.

Three cutoffs were chosen on calibration only (rows after the first 7-day window) and frozen:

- Point: the lowest calibration tail MAE whose overall MAE is within 1% of the regime switch.
- Mixture: the lowest calibration tail CRPS whose overall CRPS is within 1% of the regime switch.
- Detection: the best calibration F1 for calling a crossing.

| Model | Point cutoffs (up, down) | Mixture cutoffs | Detection cutoff | Calibration F1 |
| --- | --- | --- | ---: | ---: |
| Hurdle, panel only | 0.5, 0.2 | 0.02, 0.02 | 0.24 | 0.408 |
| Hurdle + weather | 0.4, 0.2 | 0.02, 0.02 | 0.22 | 0.401 |
| Hurdle + weather + pre-dispatch | 0.3, 0.2 | 0.02, 0.02 | 0.22 | 0.416 |

The mixture cutoff was the lowest value on the grid (0.02) for all three. A lower one was not tried.

## Results

Definitions:

- Tail: the realised price is outside the training band.
- First interval of event (onset): the last price was inside the band and the target is outside it.
- Rest of event: the last price was already outside the band and the target is too.
- Precision and recall: for the hurdle models, a call means the larger classifier probability is at or above the detection cutoff, scored on eligible origins. For the baselines, a call means the point forecast itself is outside the band.
- AP: average precision of the classifier probability on eligible origins.

The rest-of-event columns are the same for the regime switch and all three hurdle models by construction. Those origins are not eligible, so the regime switch is issued.

### Calibration (2025-10-08 to 2026-03-31; 50,381 intervals, 125 onsets)

| Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Rest-of-event MAE | Onset precision | Onset recall | Onset AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 5.562 | 4.794 | 65.35 | 62.76 | 108.26 | 102.97 | 50.69 | – (0 calls) | 0.00 | – |
| LightGBM | 5.268 | 4.381 | 65.34 | 62.16 | 105.00 | 99.48 | 51.80 | 1.00 (4 calls) | 0.03 | – |
| Regime switch | 5.422 | 4.550 | 65.00 | 62.35 | 105.76 | 100.39 | 51.09 | 1.00 (1 call) | 0.01 | – |
| Hurdle, panel only | 5.437 | 4.511 | 64.08 | 56.00 | 102.14 | 75.46 | 51.09 | 0.36 | 0.46 | 0.338 |
| Hurdle + weather | 5.448 | 4.513 | 63.54 | 56.46 | 100.02 | 77.26 | 51.09 | 0.35 | 0.46 | 0.326 |
| Hurdle + weather + pre-dispatch | 5.455 | 4.508 | 61.63 | 56.32 | 92.51 | 76.69 | 51.09 | 0.36 | 0.50 | 0.333 |

### Test (2026-04-01 to 2026-08-18 07:55; 40,127 intervals, 403 onsets), scored once

| Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Rest-of-event MAE | Onset precision | Onset recall | Onset AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 4.745 | 4.176 | 16.00 | 14.77 | 45.83 | 42.50 | 11.71 | – (0 calls) | 0.00 | – |
| LightGBM | 4.753 | 4.033 | 18.04 | 16.11 | 45.29 | 41.29 | 14.13 | 0.41 | 0.03 | – |
| Regime switch | 4.650 | 3.975 | 16.13 | 14.70 | 45.22 | 41.57 | 11.95 | 0.41 | 0.02 | – |
| Hurdle, panel only | 4.640 | 3.915 | 15.94 | 13.30 | 43.73 | 30.39 | 11.95 | 0.32 | 0.56 | 0.330 |
| Hurdle + weather | 4.709 | 3.919 | 16.06 | 13.32 | 44.64 | 30.58 | 11.95 | 0.33 | 0.60 | 0.336 |
| Hurdle + weather + pre-dispatch | 4.831 | 3.923 | 16.07 | 13.35 | 44.76 | 30.83 | 11.95 | 0.31 | 0.59 | 0.317 |

Test fresh onsets (no tail price in the previous hour, 233 events): first-interval MAE is 52.99 for persistence, 52.12 for the regime switch, 51.66 for panel only, 52.37 for + weather and 51.69 for + pre-dispatch. CRPS is 49.69, 48.50, 35.51, 35.62 and 36.01. Recall is 0.00, 0.01, 0.52, 0.55 and 0.55.

The point override fires on 0.08% of test intervals for panel only, 0.37% for + weather and 0.96% for + pre-dispatch. The mixture fires on 10.7%, 10.8% and 11.9%.

### Test differences against the regime switch, with 90% bootstrap ranges

Each row is the model minus the regime switch on test, so negative is better. The ranges come from a moving-block bootstrap with 1-day blocks and 1,000 draws (`spike_stage1_bootstrap.csv`). The file has no ranges for rest of event, precision, recall or AP, or for differences against persistence.

| Model | Subset (n) | MAE difference [5%, 95%] | CRPS difference [5%, 95%] |
| --- | --- | ---: | ---: |
| Persistence | all (40,127) | +0.09 [+0.07, +0.12] | +0.20 [+0.18, +0.22] |
| Persistence | tail (3,206) | −0.13 [−0.30, +0.03] | +0.07 [−0.08, +0.21] |
| Persistence | onset (403) | +0.61 [+0.38, +0.86] | +0.93 [+0.72, +1.18] |
| LightGBM | all | +0.10 [+0.06, +0.16] | +0.06 [+0.02, +0.10] |
| LightGBM | tail | +1.91 [+1.40, +2.55] | +1.41 [+0.93, +2.00] |
| LightGBM | onset | +0.06 [−0.05, +0.17] | −0.27 [−0.39, −0.16] |
| Hurdle, panel only | all | −0.01 [−0.03, +0.01] | −0.06 [−0.08, −0.04] |
| Hurdle, panel only | tail | −0.19 [−0.52, +0.06] | −1.41 [−1.76, −1.11] |
| Hurdle, panel only | onset | −1.49 [−4.07, +0.49] | −11.18 [−13.81, −8.93] |
| Hurdle, panel only | fresh onset (233) | −0.46 [−2.72, +1.40] | −13.00 [−15.89, −10.34] |
| Hurdle + weather | all | +0.06 [+0.02, +0.10] | −0.06 [−0.08, −0.03] |
| Hurdle + weather | tail | −0.07 [−0.48, +0.31] | −1.38 [−1.73, −1.09] |
| Hurdle + weather | onset | −0.59 [−3.72, +2.52] | −10.98 [−13.69, −8.75] |
| Hurdle + weather | fresh onset | +0.24 [−3.69, +3.37] | −12.88 [−15.81, −10.21] |
| Hurdle + weather + pre-dispatch | all | +0.18 [+0.12, +0.24] | −0.05 [−0.07, −0.03] |
| Hurdle + weather + pre-dispatch | tail | −0.06 [−0.49, +0.34] | −1.35 [−1.66, −1.09] |
| Hurdle + weather + pre-dispatch | onset | −0.46 [−3.91, +2.70] | −10.74 [−13.04, −8.75] |
| Hurdle + weather + pre-dispatch | fresh onset | −0.43 [−4.10, +2.52] | −12.49 [−14.99, −10.22] |

## What helped and what did not

What helped:

- The hurdle structure, as a distribution. Moving probability onto a spike-size grid when the classifier fires cuts test CRPS on the first interval of an event from 41.6 (regime switch) and 42.5 (persistence) to 30.4. Tail CRPS falls from 14.7 to 13.3. Both bootstrap ranges are clearly below zero. Overall CRPS also improves slightly: 3.915 against 3.975 for the regime switch and 4.176 for persistence. This is the one result that beats persistence on its own ground, the tail.
- The classifiers see onsets that the baselines cannot. Test recall is 0.56–0.60 against 0.02 for the regime switch, at precision of about 0.32, with ROC AUC 0.976. Onsets are 1.1% of eligible test origins, so an AP of 0.33 is about 30 times the base rate.

What did not help:

- The point forecast at onset. The hurdle's median rarely replaces the regime switch, since the 1% MAE guard keeps the cutoffs high. First-interval MAE moves from 45.2 to 43.7–44.8, and every bootstrap range crosses zero. No model beats persistence on tail MAE by more than noise.
- Day-ahead weather. On test it changes nothing that the bootstrap can separate from panel only. AP is 0.336 against 0.330, recall 0.60 against 0.56, onset CRPS 30.6 against 30.4. Overall MAE is worse (+0.06 against the regime switch, range +0.02 to +0.10) because the calibration-chosen point cutoff fires more often.
- Pre-dispatch, on test. It looked useful on calibration: first-interval MAE fell from 100.0 to 92.5, tail MAE from 63.5 to 61.6, recall rose from 0.46 to 0.50, and two pre-dispatch fields ranked in the up-classifier's top ten by gain (requirement minus last demand 6th, headroom 9th; contingency-raise margin 12th). None of that carried to test. AP fell to 0.317, the lowest of the three. First-interval CRPS was 30.8 against 30.4. Overall MAE rose to 4.831, worse than the regime switch by 0.18 (range +0.12 to +0.24) and worse than persistence. The calibration-chosen point cutoff (0.3) overrides 0.96% of test intervals, and those overrides cost more than they save.

Why calibration did not transfer:

- The two windows are very different. Calibration has 125 onsets in 50,381 intervals, a base rate of 0.25% on eligible origins, and very large spikes (tail MAE about 65). Test has 403 onsets, a base rate of 1.09%, and smaller ones (tail MAE about 16).
- Cutoffs and the pre-dispatch gain rest on 125 calibration events, so they are noisy.
- The run used has a median age of 105 minutes. Even-hour runs only give a forecast made up to 4 hours before the origin, which is too stale to see a 5-minute onset.
- `pd_run_age_min` ranks high by gain. It is partly a time-of-day proxy, not market information.

In short, stage 1 shows that a calibrated spike probability with a spike-size distribution beats persistence and the regime switch on onset and tail CRPS. It does not yet give a better point price at onset. On this test window, neither external source adds anything measurable beyond the panel.

## Next steps

1. Pull the odd-hour pre-dispatch runs (`--pass 2`). That halves the run age to 40–100 minutes. Better still, take the 5-minute WEMDE dispatch or short-horizon pre-dispatch, if an archive with publication times exists.
2. Turn the pre-dispatch fields into onset-specific signals: forecast price change inside the next hour, headroom against the realised last demand rather than the forecast requirement, and the change between consecutive runs (revision). Drop `pd_run_age_min`, or replace it with run hour.
3. Make cutoff selection less fragile. Use a rolling-origin or blocked choice across the end of train and calibration, rather than one window with 125 events, and widen the mixture grid below 0.02. Report the precision-recall curve rather than one F1 point.
4. Keep the hurdle as a distribution model only. Issue the regime switch as the point forecast, and use the hurdle's probability and size grid for CRPS, intervals and spike alerts. Its point override does not pay.
5. Add bootstrap ranges against persistence, and for rest of event, precision, recall and AP, before any stage-2 claims.
