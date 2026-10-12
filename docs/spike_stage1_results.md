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

The stage 1 tables below are from commit 8a07e63, when the mixture grid started at 0.02. Stage 1b (below) re-chose the mixture cutoffs on calibration from a wider grid. That changes only the CRPS columns of these rows, by at most 0.002 overall, 0.04 on the tail and 0.23 on fresh onsets. MAE, precision, recall and AP are unchanged. The current numbers for every row are in the stage 1b section and in `reports/forecast/spike_stage1_results.csv`.

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

## Stage 1b: hourly pre-dispatch runs and revision features

### Data and availability

The odd-hour runs (`--pass 2`) were pulled for 2023-09-30 to 2026-08-17, the same trading days as pass 1. The consolidated file now holds 25,120 hourly runs, 12,561 even and 12,559 odd. It is 18 MB and is committed. 152 on-the-hour runs are absent from the archive: 75 even and 77 odd. 141 are in train, 11 in calibration and none in test. Neither pull logged a download failure, and the 2023-12-20 ZIP has no on-the-hour runs at all. The list is in `data/external/aemo_predispatch/missing_runs.csv`.

The availability rule is unchanged and holds for odd-hour runs:

- Issue lag: odd runs have the same lag after the label as even runs, median 15.15 minutes (92% within 15.5 minutes).
- Posting time: on the live folder, odd-hour, even-hour and half-hour runs were all posted 35 minutes after the label (`live_posting_check_20261008.csv`).

| Run set | Split | Origins covered | Runs present / expected | Run age median / p95 / max (min) | Age 40–100 min | Previous run for revisions |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Even-hour (stage 1) | train | 99.28% | 8,687 / 8,756 | 105 / 160 / 240 | 48.2% | – |
| Even-hour | calibration | 99.84% | 2,178 / 2,184 | 105 / 155 / 240 | 49.9% | – |
| Even-hour | test | 100.00% | 1,672 / 1,672 | 105 / 155 / 205 | 50.0% | – |
| All hourly (stage 1b) | train | 99.49% | 17,371 / 17,512 | 75 / 100 / 240 | 96.3% | 99.42% |
| All hourly | calibration | 99.90% | 4,357 / 4,368 | 75 / 100 / 240 | 99.75% | 99.87% |
| All hourly | test | 99.97% | 3,344 / 3,344 | 75 / 100 / 145 | 99.95% | 99.94% |

The run used at the origin is now 40–100 minutes old, median 75, against a median of 105 with even-hour runs only.

### Features (`pda_`, 19)

- The 13 stage 1 pre-dispatch features, recomputed from the latest hourly run. This includes the spread between the latest run's forecast and the last realised price (`pda_price_minus_last`).
- Within-run change: forecast price and energy requirement for the half hour containing T, minus the same run's forecast one hour earlier. If one hour earlier is before the run's first interval, the run's first interval is used instead, so the gap is 30–60 minutes.
- Revisions: the latest usable run minus the run labelled before it, for forecast price at T, maximum forecast price over the next 2 hours, energy requirement at T, and headroom at T. The previous run is used only if it was itself published by the origin under the same rule and is at most 2 hours older. Otherwise the revisions are missing.
- `pd_run_age_min` is not used. Run age is kept in the table for the coverage report only.

The stage 1 even-hour features (`pd_`) are still computed from the even-hour runs alone, so that row is reproduced. Its point, detection and MAE figures are identical to stage 1.

Leakage tests (`tests/test_external_features.py`) cover the new features in two ways:

- **Synthetic runs:**
  - corrupting or dropping any run published after the origin leaves every `pda_` feature, revisions included, unchanged;
  - a previous run published late gives no revision;
  - the revision compares exactly the latest and previous published runs.
- **Real hourly runs:** the latest and previous runs both have `available_at` at or before the origin, the label + 40 min and issue + 25 min rule holds when recomputed from the raw label and issue time, and corrupting or dropping later runs changes nothing.

I checked that the late-previous-run test fails when the availability check on the previous run is removed.

### Cutoff grid

The mixture grid now runs 0.0025, 0.005, 0.01, 0.02, 0.05, ... It is still chosen on calibration only, by the lowest tail CRPS with overall CRPS within 1% of the regime switch.

Every model again chose the lowest up-cutoff, 0.0025. The calibration surface is almost flat there: tail CRPS at (0.02, 0.02) is only 0.03–0.04 above the chosen point for every model. So the edge choice amounts to always mixing in the spike grid with weight p. Going lower would change little.

The new model's cutoffs, all chosen on calibration:

- point: up 0.3, down 0.1;
- mixture: up 0.0025, down 0.01;
- detection: 0.20 (calibration F1 0.398).

### Calibration (2025-10-08 to 2026-03-31; 50,381 intervals, 125 onsets)

| Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Rest-of-event MAE | Onset precision | Onset recall | Onset AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 5.562 | 4.794 | 65.35 | 62.76 | 108.26 | 102.97 | 50.69 | – (0 calls) | 0.00 | – |
| LightGBM | 5.268 | 4.381 | 65.34 | 62.16 | 105.00 | 99.48 | 51.80 | 1.00 (4 calls) | 0.03 | – |
| Regime switch | 5.422 | 4.550 | 65.00 | 62.35 | 105.76 | 100.39 | 51.09 | 1.00 (1 call) | 0.01 | – |
| Hurdle, panel only | 5.437 | 4.511 | 64.08 | 55.97 | 102.14 | 75.34 | 51.09 | 0.36 | 0.46 | 0.338 |
| Hurdle + weather | 5.448 | 4.513 | 63.54 | 56.42 | 100.02 | 77.10 | 51.09 | 0.35 | 0.46 | 0.326 |
| Hurdle + weather + pre-dispatch (even-hour) | 5.455 | 4.508 | 61.63 | 56.27 | 92.51 | 76.52 | 51.09 | 0.36 | 0.50 | 0.333 |
| **Hurdle + weather + pre-dispatch (hourly + revisions)** | 5.463 | 4.509 | 61.47 | 56.17 | 91.87 | 76.11 | 51.09 | 0.32 | 0.53 | 0.325 |

### Test (2026-04-01 to 2026-08-18 07:55; 40,127 intervals, 403 onsets), scored once

| Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Rest-of-event MAE | Onset precision | Onset recall | Onset AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 4.745 | 4.176 | 16.00 | 14.77 | 45.83 | 42.50 | 11.71 | – (0 calls) | 0.00 | – |
| LightGBM | 4.753 | 4.033 | 18.04 | 16.11 | 45.29 | 41.29 | 14.13 | 0.41 | 0.03 | – |
| Regime switch | 4.650 | 3.975 | 16.13 | 14.70 | 45.22 | 41.57 | 11.95 | 0.41 | 0.02 | – |
| Hurdle, panel only | 4.640 | 3.917 | 15.94 | 13.29 | 43.73 | 30.37 | 11.95 | 0.32 | 0.56 | 0.330 |
| Hurdle + weather | 4.709 | 3.921 | 16.06 | 13.32 | 44.64 | 30.55 | 11.95 | 0.33 | 0.60 | 0.336 |
| Hurdle + weather + pre-dispatch (even-hour) | 4.831 | 3.925 | 16.07 | 13.35 | 44.76 | 30.80 | 11.95 | 0.31 | 0.59 | 0.317 |
| **Hurdle + weather + pre-dispatch (hourly + revisions)** | 4.869 | 3.926 | 15.82 | 13.28 | 42.71 | 30.28 | 11.95 | 0.30 | 0.66 | 0.308 |

Test fresh onsets (233): the new row has first-interval MAE 49.27, CRPS 35.19 and recall 0.65. The even-hour row has 51.69, 35.97 and 0.55. Panel only has 51.66, 35.47 and 0.52. Persistence has 52.99 and 49.69.

The new model's point override fires on 1.09% of test intervals (0.96% for even-hour). Its spike-grid mixture is active on 38.4% of test intervals, against 35.1–39.0% for the other hurdle rows at the new cutoffs.

### Test bootstrap ranges (model minus baseline, moving-block bootstrap with 1-day blocks, 90%, 1,000 draws)

Against persistence:

| Model | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | +0.01 [−0.05, +0.07] | +1.34 [+0.83, +1.92] | −0.55 [−0.84, −0.28] | −1.20 [−1.49, −0.97] |
| Regime switch | −0.09 [−0.12, −0.07] | −0.07 [−0.21, +0.08] | −0.61 [−0.86, −0.38] | −0.93 [−1.18, −0.72] |
| Hurdle, panel only | −0.10 [−0.14, −0.06] | −1.48 [−1.90, −1.12] | −2.10 [−4.82, −0.08] | −12.13 [−14.99, −9.77] |
| Hurdle + weather | −0.04 [−0.09, +0.02] | −1.45 [−1.87, −1.10] | −1.19 [−4.55, +1.90] | −11.94 [−14.83, −9.58] |
| Hurdle + wx + pre-dispatch (even-hour) | +0.09 [+0.02, +0.15] | −1.42 [−1.81, −1.10] | −1.07 [−4.64, +2.22] | −11.70 [−14.20, −9.57] |
| Hurdle + wx + pre-dispatch (hourly + revisions) | +0.12 [+0.05, +0.20] | −1.49 [−1.90, −1.14] | −3.12 [−6.86, +0.32] | −12.22 [−14.97, −9.89] |

Against the regime switch:

| Model | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| Persistence | +0.09 [+0.07, +0.12] | +0.07 [−0.08, +0.21] | +0.61 [+0.38, +0.86] | +0.93 [+0.72, +1.18] |
| LightGBM | +0.10 [+0.06, +0.16] | +1.41 [+0.93, +2.00] | +0.06 [−0.05, +0.17] | −0.27 [−0.39, −0.16] |
| Hurdle, panel only | −0.01 [−0.03, +0.01] | −1.41 [−1.76, −1.12] | −1.49 [−4.07, +0.49] | −11.20 [−13.83, −8.95] |
| Hurdle + weather | +0.06 [+0.02, +0.10] | −1.38 [−1.73, −1.09] | −0.59 [−3.72, +2.52] | −11.01 [−13.72, −8.79] |
| Hurdle + wx + pre-dispatch (even-hour) | +0.18 [+0.12, +0.24] | −1.35 [−1.66, −1.09] | −0.46 [−3.91, +2.70] | −10.77 [−13.06, −8.77] |
| Hurdle + wx + pre-dispatch (hourly + revisions) | +0.22 [+0.15, +0.29] | −1.42 [−1.76, −1.14] | −2.51 [−6.20, +0.89] | −11.29 [−13.86, −9.08] |

Fresh-onset CRPS ranges and every subset are in `spike_stage1_bootstrap.csv`, with a `baseline` column. The file has no ranges between two hurdle rows, so the gap between the new row and panel only is not tested directly.

### Do fresher runs and revisions help at spike onset?

Only a little, and not separably from noise. Against the even-hour pre-dispatch row on test:

- first-interval MAE fell from 44.76 to 42.71, and fresh-onset MAE from 51.69 to 49.27;
- recall rose from 0.59 to 0.66;
- first-interval CRPS fell from 30.80 to 30.28.

That makes the new row the best of all rows on first-interval MAE, first-interval CRPS, tail MAE and tail CRPS. But its first-interval MAE range crosses zero against both persistence (−3.12 [−6.86, +0.32]) and the regime switch (−2.51 [−6.20, +0.89]). Its CRPS edge over panel only (30.28 against 30.37) is too small to matter.

The costs are also real:

- precision fell to 0.30 and AP to 0.308, the lowest of the hurdle rows;
- overall MAE is the worst of all rows (4.869), significantly worse than persistence (+0.12 [+0.05, +0.20]) and the regime switch (+0.22 [+0.15, +0.29]).

The extra point overrides at calibration-chosen cutoffs cost more on ordinary intervals than they save at onsets.

The revision and within-run-change features are not among the up-classifier's 15 largest by gain. The pre-dispatch fields that rank (5th to 12th) are the level features: forecast price, headroom, requirement minus last demand, the spread to the last price, and the contingency margin. Calibration moved the same way as test, but by less: first-interval MAE fell from 92.51 to 91.87 and recall rose from 0.50 to 0.53, while AP fell from 0.333 to 0.325.

The robust result is still the one from stage 1. The hurdle's spike-probability mixture beats persistence by about 12 on first-interval CRPS and 1.5 on tail CRPS, with ranges well clear of zero, and panel features alone already get it. External data, fresher or not, has not yet added a gain that the bootstrap can separate.

Next:

- Issue the regime switch as the point forecast and keep the hurdle for the distribution, spike probability and alerts. Every pre-dispatch row loses on overall MAE through its point override.
- Revisions between 60-minute-apart runs are probably too coarse. The half-hour runs (posted 35 minutes after the label) would give 30-minute revisions and a run 40–70 minutes old.
- Choose cutoffs on more than one window. The calibration window has 125 onsets against 403 in test.

## Stage 1c: half-hourly pre-dispatch runs and 30-minute revisions

### Data and availability

The half-hour runs (`--pass 3`, labelled :30) were pulled for 2023-09-30 to 2026-08-17. The consolidated file now holds 50,229 runs: 12,561 even-hour, 12,559 odd-hour and 25,109 half-hour. It is 31 MB and committed.

315 runs are absent from the archive (75 even-hour, 77 odd-hour, 163 half-hour). No pull logged a download failure. By split:

| Run type | Train | Calibration | Test |
| --- | ---: | ---: | ---: |
| Even-hour | 69 | 6 | 0 |
| Odd-hour | 72 | 5 | 0 |
| Half-hour | 153 | 9 | 1 |
| **All** | **294** | **20** | **1** |

Most missing runs fall on 2023-12-19 to 2023-12-21 and 2024-02-22 to 2024-02-23 (`missing_runs.csv`).

Publication rule for :30 runs. The rule is unchanged: available from max(label + 40 min, issue + 25 min), and it holds.

- **Issue times in the archive:** the median lag after the label is 15.15 minutes. 92.0% are within 15.5 minutes. 119 of 25,109 are more than 40 minutes late, and the rule delays those runs. On-the-hour runs look the same: median 15.15, 91.7%, 113 late.
- **Live folder:** 13 half-hour runs between 07:30 and 19:30 on 2026-10-08 were all posted exactly 35 minutes after the label, the same as even- and odd-hour runs (`live_posting_check_20261008.csv`, 25 runs).

| Run set | Split | Origins covered | Runs present / expected | Run age median / p95 / max (min) | Age 40–70 min | 30 min revision available |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Even-hour (stage 1) | test | 100.00% | 1,672 / 1,672 | 105 / 155 / 205 | 25.0% | – |
| Hourly (stage 1b) | test | 99.97% | 3,344 / 3,344 | 75 / 100 / 145 | 50.0% | 99.94% (60 min) |
| Half-hourly (stage 1c) | train | 99.52% | 34,732 / 35,024 | 60 / 75 / 240 | 93.6% | 99.44% |
| Half-hourly | calibration | 99.91% | 8,716 / 8,736 | 60 / 70 / 240 | 99.6% | 99.88% |
| Half-hourly | test | 99.99% | 6,687 / 6,688 | 60 / 70 / 145 | 99.9% | 99.96% |

The run in use is now 40–70 minutes old at almost every origin. In practice it is 45–70 minutes, because issue time + 25 min usually lands seconds after label + 40 min, so the first origin that can use a run is label + 45 min.

### Features (`pdh_`, 23)

- The 15 stage 1b level and within-run features (forecast price and requirement levels, spread to the last price, headroom, margins, 1 h within-run change), recomputed from the latest half-hourly run.
- 30 minute revisions. The latest run minus the latest run labelled at least 30 minutes and at most 1 hour earlier, for:
  - forecast price at T;
  - maximum forecast price over the next 2 hours;
  - energy requirement at T;
  - headroom at T.
- 60 minute revisions. The same four fields against a run at least 60 minutes and at most 2 hours earlier (kept because they are cheap).
- No run-age feature.

Every previous run must itself have been available by the origin. Otherwise its revisions are missing.

The stage 1 (`pd_`, even-hour) and stage 1b (`pda_`, now explicitly on-the-hour runs only) features are bit-for-bit unchanged. All earlier rows reproduce to within 2e-12.

Leakage tests cover `pdh_` features as follows:

- **Synthetic runs:**
  - corrupting or dropping any run published after the origin leaves every `pdh_` feature unchanged;
  - the 30 and 60 minute revisions compare exactly the runs 30 and 60 minutes before the latest;
  - a previous run published late gives no 30 minute revision, while the 60 minute revision falls back to an older run that was published on time.
- **Real half-hourly runs:** every latest and previous run used has `available_at` at or before the origin, the label + 40 / issue + 25 rule holds when recomputed from raw label and issue time, and corrupting or dropping later runs changes nothing.

Removing the availability check on the previous run makes the late-run test fail.

### Cutoffs (calibration only, same rules and grids as stage 1b)

- Point: up 0.3, down 0.1.
- Mixture: up 0.0025, down 0.005.
- Detection: 0.20 (calibration F1 0.404).

### Calibration (125 onsets)

| Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Precision | Recall | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 5.562 | 4.794 | 65.35 | 62.76 | 108.26 | 102.97 | – | 0.00 | – |
| LightGBM | 5.268 | 4.381 | 65.34 | 62.16 | 105.00 | 99.48 | 1.00 | 0.03 | – |
| Regime switch | 5.422 | 4.550 | 65.00 | 62.35 | 105.76 | 100.39 | 1.00 | 0.01 | – |
| Hurdle, panel only | 5.437 | 4.511 | 64.08 | 55.97 | 102.14 | 75.34 | 0.36 | 0.46 | 0.338 |
| Hurdle + weather | 5.448 | 4.513 | 63.54 | 56.42 | 100.02 | 77.10 | 0.35 | 0.46 | 0.326 |
| + pre-dispatch, even-hour | 5.455 | 4.508 | 61.63 | 56.27 | 92.51 | 76.52 | 0.36 | 0.50 | 0.333 |
| + pre-dispatch, hourly + revisions | 5.463 | 4.509 | 61.47 | 56.17 | 91.87 | 76.11 | 0.32 | 0.53 | 0.325 |
| **+ pre-dispatch, half-hourly + 30/60 min revisions** | 5.459 | 4.508 | 61.95 | 56.32 | 93.77 | 76.70 | 0.33 | 0.51 | 0.345 |

### Test (403 onsets), scored once

| Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Precision | Recall | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 4.745 | 4.176 | 16.00 | 14.77 | 45.83 | 42.50 | – | 0.00 | – |
| LightGBM | 4.753 | 4.033 | 18.04 | 16.11 | 45.29 | 41.29 | 0.41 | 0.03 | – |
| Regime switch | 4.650 | 3.975 | 16.13 | 14.70 | 45.22 | 41.57 | 0.41 | 0.02 | – |
| Hurdle, panel only | 4.640 | 3.917 | 15.94 | 13.29 | 43.73 | 30.37 | 0.32 | 0.56 | 0.330 |
| Hurdle + weather | 4.709 | 3.921 | 16.06 | 13.32 | 44.64 | 30.55 | 0.33 | 0.60 | 0.336 |
| + pre-dispatch, even-hour | 4.831 | 3.925 | 16.07 | 13.35 | 44.76 | 30.80 | 0.31 | 0.59 | 0.317 |
| + pre-dispatch, hourly + revisions | 4.869 | 3.926 | 15.82 | 13.28 | 42.71 | 30.28 | 0.30 | 0.66 | 0.308 |
| **+ pre-dispatch, half-hourly + 30/60 min revisions** | 4.854 | 3.926 | 16.12 | 13.33 | 45.11 | 30.68 | 0.30 | 0.65 | 0.309 |

Test fresh onsets (233) for the new row: first-interval MAE 52.83, CRPS 35.97, recall 0.61. The point override fires on 0.93% of test intervals and the mixture on 42.0%.

### Test bootstrap ranges (model minus baseline; moving-block bootstrap, 1-day blocks, 90%, 1,000 draws)

Against persistence:

| Model | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| Regime switch | −0.09 [−0.12, −0.07] | −0.07 [−0.21, +0.08] | −0.61 [−0.86, −0.38] | −0.93 [−1.18, −0.72] |
| Hurdle, panel only | −0.10 [−0.14, −0.06] | −1.48 [−1.90, −1.12] | −2.10 [−4.82, −0.08] | −12.13 [−14.99, −9.77] |
| + pre-dispatch, hourly + revisions | +0.12 [+0.05, +0.20] | −1.49 [−1.90, −1.14] | −3.12 [−6.86, +0.32] | −12.22 [−14.97, −9.89] |
| **+ pre-dispatch, half-hourly + 30/60 min** | +0.11 [+0.03, +0.19] | −1.44 [−1.83, −1.10] | −0.72 [−4.81, +3.41] | −11.82 [−14.39, −9.64] |

Against the regime switch:

| Model | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| Hurdle, panel only | −0.01 [−0.03, +0.01] | −1.41 [−1.76, −1.12] | −1.49 [−4.07, +0.49] | −11.20 [−13.83, −8.95] |
| + pre-dispatch, hourly + revisions | +0.22 [+0.15, +0.29] | −1.42 [−1.76, −1.14] | −2.51 [−6.20, +0.89] | −11.29 [−13.86, −9.08] |
| **+ pre-dispatch, half-hourly + 30/60 min** | +0.20 [+0.14, +0.27] | −1.37 [−1.68, −1.09] | −0.11 [−4.15, +3.91] | −10.89 [−13.29, −8.84] |

Against the panel-only hurdle:

| Model | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| Hurdle + weather | +0.07 [+0.04, +0.10] | +0.02 [−0.02, +0.07] | +0.90 [−0.63, +2.50] | +0.19 [−0.12, +0.52] |
| + pre-dispatch, even-hour | +0.19 [+0.14, +0.24] | +0.05 [+0.01, +0.11] | +1.03 [−1.75, +3.71] | +0.44 [+0.07, +0.90] |
| + pre-dispatch, hourly + revisions | +0.23 [+0.17, +0.29] | −0.01 [−0.06, +0.04] | −1.02 [−4.43, +2.17] | −0.09 [−0.49, +0.35] |
| **+ pre-dispatch, half-hourly + 30/60 min** | +0.21 [+0.15, +0.28] | +0.04 [−0.01, +0.10] | +1.38 [−1.90, +4.84] | +0.32 [−0.09, +0.83] |

All other models and subsets, including fresh onsets and LightGBM, are in `spike_stage1_bootstrap.csv` under the `baseline` column.

### Feature importance

`reports/forecast/spike_stage1_feature_gain.csv` holds the full gain ranking for both classifiers.

- **Up classifier** (59 features): no revision feature is in the top 15. The highest is the 30 min headroom revision at 20th (0.44% of gain), and the 30 min price revision is 36th. The within-run 1 h price change is 14th (0.7%). The pre-dispatch features that do rank are levels, at 5th to 13th: requirement minus last demand, headroom, forecast price, contingency margin, spread to the last price. The last complete real-time price takes 66% of the gain.
- **Down classifier:** one revision feature makes the top 15. The 60 minute energy-requirement revision is 15th (1.2%), and the best 30 minute revision is 21st.

### Do fresher runs and 30-minute revisions help at spike onset?

No.

- **First interval:** the half-hourly row is worse than the hourly row on test first-interval MAE (45.11 against 42.71) and CRPS (30.68 against 30.28). It no longer beats persistence or the regime switch on first-interval MAE: −0.72 [−4.81, +3.41] and −0.11 [−4.15, +3.91]. Against panel only it is +1.38 [−1.90, +4.84] on MAE and +0.32 [−0.09, +0.83] on CRPS, so no gain.
- **Recall:** 0.65 is about the same as hourly (0.66). Precision (0.30) and AP (0.309) are no better.
- **Overall MAE:** still significantly worse than persistence (+0.11 [+0.03, +0.19]), the regime switch (+0.20) and panel only (+0.21), because of the point override.
- **Calibration:** it had the best AP (0.345) but a worse first-interval MAE than the hourly row (93.77 against 91.87). That AP edge did not carry to test.

The hourly row's onset MAE gain in stage 1b looks like noise: a finer, fresher version of the same information did not reproduce it.

The pattern from stages 1 and 1b holds. The hurdle's spike-probability mixture beats persistence on first-interval CRPS by about 12 and on tail CRPS by about 1.5, with ranges clear of zero. Panel-only gets that already. No pre-dispatch variant beats panel only on any of the four metrics by more than noise: the hourly row has slightly better point estimates on three of them, but every range crosses zero.

Fresher pre-dispatch information does not appear to carry the 5-minute onset signal. Pre-dispatch is solved half-hourly on forecast inputs, and spikes start inside the half hour. Further pre-dispatch work is unlikely to pay. Next steps:

- issue the regime switch as the point forecast, and use the panel-only hurdle for the distribution and spike alerts;
- look for 5-minute inputs published before the origin, for example the dispatch-interval outcome of the previous interval beyond MCP: binding constraints, FCESS shortfalls, and the change in available capacity between consecutive dispatch runs.

## Stage A: recent price-path features

### Features (`path_`, 9; `scripts/price_path.py`)

All are built from 5-minute prices labelled T−5 or earlier, so only prices realised by the origin are used: lags k = 1 and up, where k = 1 is `mcp_lag_5min`. There are no new MCP lags and nothing from the target interval. The near-spike band comes from train only: q90 = 143.11 and q10 = −33.14.

- `path_near_up_30` and `path_near_up_60`: the number of prices at or above the train q90 in the last 30 and 60 minutes. `path_near_down_30` and `path_near_down_60` count prices at or below q10.
- `path_slope_15`: the least-squares slope of the last four prices, in $ per 5 minutes.
- `path_climb_run` and `path_fall_run`: the number of consecutive rises or falls ending at the last price (capped at 24).
- `path_std_30`: the standard deviation over the last 30 minutes.
- `path_max60_minus_last`: the highest price in the last 60 minutes minus the last price.

`tests/test_price_path.py` has 5 leakage and behaviour tests:

- setting the target-interval price and every later price to 1e6 leaves all features unchanged;
- dropping prices after the origin changes nothing;
- a known ramp gives the expected slope, run and counts;
- the feature names avoid the banned contemporaneous list and `mcp_lag`.

As a mutation check, shifting by k − 1 instead of k makes 3 of the tests fail.

### Row and cutoffs

`hurdle_panel_path` is the panel-only hurdle plus the 9 path features. It uses the same LightGBM classifiers and size models, and the same grids and rules. Earlier rows are unchanged to within 5e-9.

Cutoffs were chosen on calibration only:

- point: up 0.5, down 0.2;
- mixture: up 0.0025, down 0.02;
- detection: 0.22 (calibration F1 0.426, against 0.24 for panel only).

| Split | Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Precision | Recall | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration | Persistence | 5.562 | 4.794 | 65.35 | 62.76 | 108.26 | 102.97 | – | 0.00 | – |
| Calibration | Hurdle, panel only | 5.437 | 4.511 | 64.08 | 55.97 | 102.14 | 75.34 | 0.36 | 0.46 | 0.338 |
| Calibration | **Hurdle, panel + path** | 5.445 | 4.507 | 64.40 | 55.91 | 103.39 | 75.09 | 0.37 | 0.50 | 0.373 |
| Test | Persistence | 4.745 | 4.176 | 16.00 | 14.77 | 45.83 | 42.50 | – | 0.00 | – |
| Test | Hurdle, panel only | 4.640 | 3.917 | 15.94 | 13.29 | 43.73 | 30.37 | 0.32 | 0.56 | 0.330 |
| Test | **Hurdle, panel + path** | 4.689 | 3.914 | 16.01 | 13.20 | 44.27 | 29.66 | 0.32 | 0.61 | 0.364 |

Test bootstrap ranges for panel + path (model minus baseline; 1-day blocks, 90%, 1,000 draws):

| Baseline | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| Persistence | −0.06 [−0.11, −0.01] | −1.56 [−2.02, −1.19] | −1.57 [−4.81, +1.16] | −12.84 [−15.83, −10.24] |
| Regime switch | +0.04 [−0.00, +0.08] | −1.50 [−1.88, −1.18] | −0.96 [−4.04, +1.71] | −11.91 [−14.72, −9.40] |
| Hurdle, panel only | +0.05 [+0.02, +0.08] | −0.09 [−0.14, −0.04] | +0.53 [−1.20, +2.23] | −0.70 [−1.14, −0.33] |

Against panel only, fresh-onset CRPS improves by −1.01 [−1.74, −0.37].

### Feature gain

The path features rank in the top 15.

- **Up classifier:** `path_near_up_30` is 3rd (3.4% of gain), `path_std_30` 4th, `path_slope_15` 5th, `path_max60_minus_last` 10th and `path_near_up_60` 13th. `path_climb_run` is 19th.
- **Down classifier:** `path_slope_15` is 5th, `path_std_30` 6th and `path_max60_minus_last` 15th.

The last complete real-time price still dominates.

### Verdict

Keep the path features. The decision was made on calibration: they improved CRPS, tail CRPS, first-interval CRPS, recall and AP (0.338 to 0.373), while overall and tail MAE were slightly worse.

Test agrees on the probabilistic side. Tail CRPS and first-interval CRPS improve over panel only with ranges clear of zero, and AP rises from 0.330 to 0.364. The cost is a small, significant rise in overall MAE (+0.05) as the point override fires more often (0.19% of test intervals against 0.08%). The gains are small next to the hurdle's gain over persistence.

The set used in stage B is panel + path.

## Stage B: other spike classifiers

All Stage B rows use the panel + path features. Only the up and down crossing classifiers change. The LightGBM quantile size models, the mixture, the conformal residuals and the cutoff rules and grids are kept as they were. No adapter was needed, because every classifier exposes `predict_proba`.

Capacity is chosen on the inner validation window (2025-07-01 to 2025-09-30, inside train) by log loss, then each model is refitted on all train rows from 2024-03-01, as for LightGBM.

- **XGBoost** (`xgboost>=3.4,<4`, pinned in `requirements.txt`): hist trees with depth 6, learning rate 0.03, subsample and colsample 0.8, L2 1. Early stopping after 200 rounds chose 143 trees for up and 1,493 for down.
- **HistGradientBoosting** (scikit-learn): 31 leaves, minimum leaf 200, learning rate 0.05, L2 1. Iterations come from {50, 100, 200, 400, 800}: 100 for up and 800 for down. 800 is the top of the grid.
- **Logistic regression:** median imputation with missing-value indicators, standardisation, and C from {0.01, 0.1, 1}. Up chose 0.1 and down chose 1.

Earlier rows, including the LightGBM panel + path row, are unchanged to within 3e-11. Their bootstrap pairs are identical.

### Cutoffs (calibration only)

| Classifier | Point (up, down) | Mixture (up, down) | Detection | Calibration F1 |
| --- | --- | --- | ---: | ---: |
| LightGBM | 0.5, 0.2 | 0.0025, 0.02 | 0.22 | 0.426 |
| XGBoost | 0.9, 0.2 | 0.0025, 0.02 | 0.24 | 0.458 |
| HistGradientBoosting | 0.5, 0.1 | 0.0025, 0.05 | 0.24 | 0.429 |
| Logistic | 0.6, 0.1 | 0.0025, 0.01 | 0.16 | 0.316 |

The XGBoost point override never fires on calibration or test, so its point forecast equals the regime switch.

### Results

| Split | Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Precision | Recall | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration | Persistence | 5.562 | 4.794 | 65.35 | 62.76 | 108.26 | 102.97 | – | 0.00 | – |
| Calibration | Hurdle, panel only (LightGBM) | 5.437 | 4.511 | 64.08 | 55.97 | 102.14 | 75.34 | 0.36 | 0.46 | 0.338 |
| Calibration | Panel + path, LightGBM | 5.445 | 4.507 | 64.40 | 55.91 | 103.39 | 75.09 | 0.37 | 0.50 | 0.373 |
| Calibration | Panel + path, XGBoost | 5.422 | 4.508 | 64.96 | 55.67 | 105.57 | 74.16 | 0.42 | 0.50 | 0.391 |
| Calibration | **Panel + path, HistGradientBoosting** | 5.458 | 4.508 | 64.93 | 55.47 | 105.46 | 73.36 | 0.38 | 0.50 | 0.389 |
| Calibration | Panel + path, logistic | 5.434 | 4.530 | 64.03 | 58.41 | 101.91 | 84.92 | 0.38 | 0.27 | 0.247 |
| Test | Persistence | 4.745 | 4.176 | 16.00 | 14.77 | 45.83 | 42.50 | – | 0.00 | – |
| Test | Hurdle, panel only (LightGBM) | 4.640 | 3.917 | 15.94 | 13.29 | 43.73 | 30.37 | 0.32 | 0.56 | 0.330 |
| Test | Panel + path, LightGBM | 4.689 | 3.914 | 16.01 | 13.20 | 44.27 | 29.66 | 0.32 | 0.61 | 0.364 |
| Test | Panel + path, XGBoost | 4.650 | 3.912 | 16.13 | 13.18 | 45.22 | 29.44 | 0.33 | 0.54 | 0.354 |
| Test | **Panel + path, HistGradientBoosting** | 4.695 | 3.915 | 15.81 | 13.13 | 42.68 | 29.08 | 0.32 | 0.56 | 0.346 |
| Test | Panel + path, logistic | 4.635 | 3.947 | 15.94 | 13.82 | 43.67 | 34.54 | 0.29 | 0.29 | 0.250 |

### Test bootstrap ranges (model minus baseline; 1-day blocks, 90%, 1,000 draws)

Against the LightGBM panel + path row:

| Classifier | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | −0.04 [−0.08, +0.00] | −0.03 [−0.07, +0.01] | +0.96 [−1.71, +4.04] | −0.22 [−0.52, +0.08] |
| HistGradientBoosting | +0.01 [−0.01, +0.03] | −0.07 [−0.11, −0.04] | −1.58 [−3.57, −0.17] | −0.58 [−0.90, −0.30] |
| Logistic | −0.05 [−0.09, −0.02] | +0.61 [+0.47, +0.79] | −0.60 [−2.12, +1.05] | +4.88 [+3.84, +6.08] |

Against persistence:

| Classifier | Overall MAE | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | −0.06 [−0.11, −0.01] | −1.56 [−2.02, −1.19] | −1.57 [−4.81, +1.16] | −12.84 [−15.83, −10.24] |
| XGBoost | −0.10 [−0.12, −0.07] | −1.59 [−2.05, −1.22] | −0.61 [−0.86, −0.38] | −13.06 [−16.07, −10.50] |
| HistGradientBoosting | −0.05 [−0.11, +0.01] | −1.64 [−2.10, −1.25] | −3.15 [−6.63, +0.02] | −13.42 [−16.72, −10.70] |
| Logistic | −0.11 [−0.15, −0.07] | −0.95 [−1.40, −0.60] | −2.16 [−4.86, −0.15] | −7.96 [−11.09, −5.50] |

Ranges against the regime switch and against panel only are in `spike_stage1_bootstrap.csv`.

### Verdict

HistGradientBoosting is the best classifier, by a small margin.

- **Calibration:** it had the lowest tail CRPS (55.47) and first-interval CRPS (73.36).
- **Test:** it is the only classifier that beats the LightGBM row with ranges clear of zero on tail CRPS (−0.07), first-interval CRPS (−0.58), first-interval MAE (−1.58) and tail MAE (−0.20 [−0.45, −0.02]). Overall MAE and CRPS are no different.
- **XGBoost** is level with LightGBM. It has the best calibration AP and F1, but on test its differences from LightGBM all cross zero except fresh-onset CRPS (−0.49 [−0.88, −0.12]). Its point forecast is the regime switch, because the calibrated point cutoff never fires.
- **Logistic regression** is clearly worse on the distribution (first-interval CRPS +4.9) and on detection (AP 0.25). The ranking is a non-linear problem.

All of these gains are small next to the gap between any tree hurdle and persistence (first-interval CRPS about −13). The test AP order (LightGBM 0.364, XGBoost 0.354, HistGradientBoosting 0.346) does not match calibration, so the detection differences are noise.

HistGradientBoosting's down classifier chose 800 iterations, the top of its grid. A wider grid, chosen on the inner validation window, is a cheap follow-up.

## Stage B follow-up: wider HGB grid and uncapped LightGBM

Two new rows search classifier capacity more widely. Every choice is still made inside train.

- **HGB, wide grid:** learning rate {0.05, 0.1}, leaves {15, 31, 63}, and up to 3,200 warm-started iterations. Each path stops after two consecutive rises in validation loss.
- **LightGBM, uncapped:** the tree cap goes from 1,500 to 8,000, with the same early stopping.

**Finding: the down classifier's validation window was degenerate.** The default inner window (2025-07-01 to 2025-09-30) has 416 upward crossings but no downward ones. Downward crossings almost stopped after 2024; by quarter from 2025 Q1 they number 7, 1, 0, 2, 0, 0, 0. On a window with no positives, log loss keeps falling as the model pushes every probability to zero. That is why stage B's down classifiers ran to the top of their grids (LightGBM 1,500 trees, HGB 800 iterations). In a first run with the wider grids they did it again (LightGBM 7,998 of 8,000; HGB 3,200).

The wide rows now step the validation window back a quarter at a time, inside train, until it holds at least 50 positives. For down this gives 2024-10-01 to 2024-12-31 (213 crossings), with the probe fitted on earlier train rows. The up window is unchanged.

With a usable window, the down classifiers choose small models: LightGBM 112 trees, HGB 15 leaves and 100 iterations at learning rate 0.05. The up choices are LightGBM 150 trees (as before) and HGB 15 leaves, 100 iterations (stage B used 31 leaves).

| Split | Row | MAE | CRPS | Tail CRPS | First-interval CRPS | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | HGB (stage B) | 5.458 | 4.5075 | 55.47 | 73.36 | 0.389 |
| Calibration | HGB, wide grid | 5.423 | 4.5125 | 56.01 | 75.49 | 0.382 |
| Calibration | LightGBM (stage A) | 5.445 | 4.5072 | 55.91 | 75.09 | 0.373 |
| Calibration | LightGBM, uncapped | 5.447 | 4.5072 | 55.91 | 75.08 | 0.382 |
| Test | HGB (stage B) | 4.695 | 3.9148 | 13.13 | 29.08 | 0.346 |
| Test | HGB, wide grid | 4.650 | 3.9106 | 13.12 | 29.01 | 0.348 |
| Test | LightGBM (stage A) | 4.689 | 3.9143 | 13.20 | 29.66 | 0.364 |
| Test | LightGBM, uncapped | 4.689 | 3.9143 | 13.20 | 29.66 | 0.364 |

Test barely changes.

- Uncapped LightGBM is identical to the stage A row on test, because no test origin passes its down mixture cutoff and the up classifier is the same.
- The wide-grid HGB is level with the stage B HGB on test. Against the LightGBM stage A row it is −0.65 [−1.15, −0.21] on first-interval CRPS. On calibration it is worse (tail CRPS 56.01 against 55.47).

## Implemented spike forecaster

### Frozen configuration (`configs/spike_forecaster.json`)

The configuration was chosen on calibration only. Test was not used for any choice.

- **Features:** panel plus price-path features (28).
- **Classifier:** HistGradientBoosting.
  - Up: learning rate 0.05, 31 leaves, 100 iterations (the stage B choice on 2025-07 to 2025-09).
  - Down: learning rate 0.05, 15 leaves, 100 iterations, chosen on 2024-10 to 2024-12. This replaces stage B's 800 iterations, which came from the degenerate window. Calibration has only 2 downward onsets, so it cannot choose this.
  - Both: minimum leaf 200, L2 1. Refitted on eligible train rows from 2024-03-01 to 2025-09-30 23:55.
- **Size models, mixture and residual window:** unchanged.
  - LightGBM quantile models (median plus 10 levels), fitted on train crossings.
  - A 2016-interval (7-day) residual window of the regime switch.
  - Mixture weight capped at 0.95.
- **Cutoffs (calibration):**
  - mixture up 0.0025, down 0.2;
  - spike alert when max(P up, P down) ≥ 0.24 (calibration F1 0.428).
- **Point forecast: the regime switch, with no hurdle override.**
  - Why: on calibration the HGB override improved tail MAE by only 0.07 (65.00 to 64.93) and first-interval MAE by 0.30 (105.76 to 105.46). It worsened overall MAE by 0.036 (5.422 to 5.458, +0.7%).
  - Spike probabilities are rarely above 0.5, so an MAE-optimal point stays in the base distribution. The spike information is carried by the distribution and the alert.
- **Why HGB:** it had the lowest calibration tail CRPS (55.47) and first-interval CRPS (73.36) of every candidate. The others were LightGBM 55.91 / 75.09, uncapped LightGBM 55.91 / 75.08, XGBoost 55.67 / 74.16, wide-grid HGB 56.01 / 75.49 and logistic 58.41 / 84.92.
- **Windows:**
  - train to 2025-09-30 23:55;
  - calibration 2025-10-01 to 2026-03-31, scored after its first 2016 rows;
  - test 2026-04-01 to 2026-08-18 07:55.
- **Comparison row:** `scripts/spike_onset.py` builds the frozen row `spike_forecaster` from the same config file.

### Results (calibration 125 onsets; test 403 onsets, all upward), with 90% moving-block bootstrap ranges (1-day blocks, 1,000 draws)

| Split | Model | MAE | CRPS | Tail MAE | Tail CRPS | First-interval MAE | First-interval CRPS | Precision | Recall | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration | Persistence | 5.562 | 4.794 | 65.35 | 62.76 | 108.26 | 102.97 | – | 0.00 | – |
| Calibration | Regime switch | 5.422 | 4.550 | 65.00 | 62.35 | 105.76 | 100.39 | 1.00 | 0.01 | – |
| Calibration | **Spike forecaster** | 5.422 | 4.508 | 65.00 | 55.47 | 105.76 | 73.37 | 0.38 | 0.50 | 0.391 |
| Test | Persistence | 4.745 | 4.176 | 16.00 | 14.77 | 45.83 | 42.50 | – | 0.00 | – |
| Test | Regime switch | 4.650 | 3.975 | 16.13 | 14.70 | 45.22 | 41.57 | 0.41 | 0.02 | – |
| Test | **Spike forecaster** | 4.650 | 3.915 | 16.13 | 13.13 | 45.22 | 29.08 | 0.32 | 0.56 | 0.346 |

Each range is the spike forecaster minus the baseline:

| Split | Baseline | Overall MAE | Overall CRPS | Tail CRPS | First-interval MAE | First-interval CRPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | Persistence | −0.14 [−0.19, −0.10] | −0.29 [−0.33, −0.25] | −7.29 [−10.18, −4.91] | −2.50 [−3.21, −1.76] | −29.60 [−38.49, −22.11] |
| Calibration | Regime switch | 0 | −0.04 [−0.06, −0.02] | −6.88 [−9.79, −4.66] | 0 | −27.02 [−35.89, −19.94] |
| Test | Persistence | −0.10 [−0.12, −0.07] | −0.26 [−0.30, −0.22] | −1.64 [−2.10, −1.25] | −0.61 [−0.86, −0.38] | −13.42 [−16.72, −10.70] |
| Test | Regime switch | 0 | −0.06 [−0.09, −0.04] | −1.57 [−1.97, −1.23] | 0 | −12.49 [−15.53, −9.91] |

The point equals the regime switch, so its MAE differences against the switch are exactly zero.

- Fresh onsets on test: first-interval CRPS 33.89 against 49.69 for persistence (−15.80 [−19.52, −12.48]).
- Test spike alerts: 720 of 40,127 intervals.
- Empirical coverage of the central intervals on test: 90.8% (5–95%), 81.1% (10–90%) and 50.8% (25–75%). At the 403 onsets the 5–95% interval covers 82.6%.

The module (`python scripts/spike_forecaster.py evaluate`) reproduces the frozen comparison row on calibration and test to 1.1e-16 on MAE, CRPS, tail, first-interval and fresh-onset metrics, precision, recall, AP and ROC AUC (`reports/forecast/spike_forecaster_validation.json`). Its test bootstrap ranges are identical to the ones from `spike_onset.py`.

### Figures (`reports/figures/spike/`, made by `python scripts/spike_figures.py`)

- `reliability_test.png` (bins in `reliability_test.csv`): P(spike up) on test at eligible origins.
  - Above 0.3 it is close to calibrated: bins with means 0.38 and 0.56 saw 0.35 and 0.50.
  - Below that it overstates the chance: the 0.07 bin saw 0.035, the 0.03 bin 0.014, and the 0.007 bin 0.001.
  - The test window has no downward crossings, so the P(spike down) panel only shows that P(down) stayed below 0.01.
- `example_episodes_test.png`: the three largest upward test onsets on separate days, with the 25–75% and 5–95% bands, the point, the actual price and the alerts. The test window has no downward onset.
- `first_interval_crps.png`: first-interval CRPS against persistence and the regime switch on calibration and test, with bootstrap ranges.

### Using it

```bash
python scripts/spike_forecaster.py fit        # train fit, calibration cutoffs (checked against the config), saves models/spike_forecaster.joblib
python scripts/spike_forecaster.py evaluate   # test forecasts -> reports/forecast/spike_forecaster_test_forecasts.parquet, metrics, bootstrap, validation
python scripts/spike_forecaster.py forecast --start "2026-08-17 00:00" --end "2026-08-18 07:55" --out forecasts.csv
```

```python
from scripts.spike_forecaster import SpikeForecaster, build_frame, read_panel
model = SpikeForecaster.load()                 # models/spike_forecaster.joblib (2.8 MB)
frame = build_frame(read_panel(), model.config)  # origin-time inputs; the target price may be missing
model.set_history(frame)                       # realised regime-switch residuals of earlier intervals
fc = model.forecast(frame, targets)            # targets = interval labels; origin = target - 5 min
```

Each target gets:

- the origin and the point forecast;
- `p_up` and `p_down` (NaN when the last price is already outside the band, where the regime switch handles the tail);
- `alert`;
- the mixture weights;
- the probabilities at or above the spike threshold and at or below the floor;
- the quantiles q01, q05, q10, q25, q50, q75, q90 and q95, plus q99.

`tests/test_spike_forecaster.py` covers:

- weighted-quantile correctness and monotonicity;
- no leakage at the origin: everything realised at or after the target, and later STEM, is perturbed and the forecast is unchanged;
- monotone quantiles and probabilities in [0, 1];
- the save and load round trip;
- reproduction of the frozen row on a test day, row by row and in metrics.

As a mutation check, feeding the target's own price into the path features makes the leakage and reproduction tests fail.

### Caveats

- The test window has no downward crossings, and calibration has 2. The down side is effectively untested; with mixture cutoff 0.2 it moved no weight on test.
- Low spike probabilities are overstated on test, so read the alert and the mixture weights as rankings, not exact frequencies. A calibration-fitted recalibration (isotonic regression on calibration) would be the next step.
- The regime-switch point and the first-interval point MAE gain over the regime switch are both zero by construction. The gain is in the distribution and the alert.
