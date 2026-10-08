# AEMO WEM Reference pre-dispatch (system-level extract)

Pulled by `scripts/pull_aemo_predispatch.py` on 2026-10-08.

- Source: `https://data.wa.aemo.com.au/public/market-data/wemde/dispatchSolution/preDispatchData/previous/DispatchSolutionPreDispatch_YYYYMMDD.zip`
- One ZIP per trading day (08:00 to 07:30 next day, AWST). Each holds 48 half-hourly runs `ReferencePre-DispatchSolution_YYYYMMDDHHMM.json`. Each run solves 96 half-hourly future dispatch intervals (48 h ahead).
- Time zone: AWST (UTC+8). The JSON carries `+08:00` offsets. Stored times are naive AWST.

## What was pulled

The archive is ~160-250 MB per day (~250 GB from October 2023), and the server gave about 0.15-0.25 MB/s per connection. Only selected runs were read with HTTP range requests, and only system-level fields were kept. Per-facility schedules and constraints were dropped.

- `extract/pass1/YYYYMMDD.parquet`: runs labelled at even hours (00:00, 02:00, ..., 22:00), 12 per day
- `extract/pass2/YYYYMMDD.parquet`: runs labelled at odd hours (01:00, 03:00, ..., 23:00), 12 per day (stage 1b, pulled 2026-10-08)
- Columns: `run_label`, `issue_id_time` (from `dispatchDataIssueID`), `dispatch_interval`, `solve_status`, and per interval: `prices.*` (energy and FCESS), `marketServiceRequirements.*` (energy requirement, contingency and regulation raise), `inServiceQuantities.energyInjectionCapacity`, `availableQuantities.*`, `dispatchTotal.*`, `marketShortfalls.*`, `contingencySolution.demandLevel/dpvLevel/largestContingency`

- `predispatch_runs_first9h.parquet` (committed, ~18 MB): pass-1 and pass-2 extracts merged, first 18 half-hourly intervals (9 h) of each run. 25,120 hourly runs (12,561 even, 12,559 odd) from 2023-09-30 08:00 to 2026-08-18 07:00 AWST, 452,151 rows. The extracts themselves (81 MB per pass) stay out of git. Features read at most 4 h (run age) + 2 h (look-ahead) past a run label, so this file gives the same features as the full extracts. `scripts/external_features.py` reads it first.
- `missing_runs.csv` (committed): the 152 on-the-hour runs not in the extracts (75 even-hour, 77 odd-hour), with `run_type`. Neither pull logged a download failure, so these runs are absent from the archive ZIPs. 141 fall in train (69 even, 72 odd; mostly 2023-12-19 to 2023-12-21, where the 2023-12-20 ZIP has no on-the-hour runs at all, and 2024-02-22 to 2024-02-23), 11 in calibration (6 even, 5 odd), none in test.

Rebuild: `python scripts/pull_aemo_predispatch.py --pass 1` and `--pass 2` (each re-reads only missing days into the gitignored `extract/`; about an hour each with two processes over halves of the date range), then `python scripts/pull_aemo_predispatch.py --consolidate`.

## Coverage of forecast origins

From `reports/forecast/spike_predispatch_coverage.csv` (an origin is covered if a run no older than 4 h was published by then):

| Run set | Split | Covered | Runs present / expected | Run age median / p95 / max (min) | Age 40-100 min |
| --- | --- | ---: | ---: | --- | ---: |
| Even-hour (stage 1 `pd_`) | train | 99.28% | 8,687 / 8,756 | 105 / 160 / 240 | 48.2% |
| Even-hour | calibration | 99.84% | 2,178 / 2,184 | 105 / 155 / 240 | 49.9% |
| Even-hour | test | 100.00% | 1,672 / 1,672 | 105 / 155 / 205 | 50.0% |
| All hourly (stage 1b `pda_`) | train | 99.49% | 17,371 / 17,512 | 75 / 100 / 240 | 96.3% |
| All hourly | calibration | 99.90% | 4,357 / 4,368 | 75 / 100 / 240 | 99.75% |
| All hourly | test | 99.97% | 3,344 / 3,344 | 75 / 100 / 145 | 99.95% |

Targets: train 2023-10-02 08:00 to 2025-09-30 23:55, calibration 2025-10-01 00:00 to 2026-03-31 23:55, test 2026-04-01 00:00 to 2026-08-18 07:55. With hourly runs the run used is 40 to 100 minutes old for 96-100% of origins; older only next to a missing or late-issued run. A previous run for the revision features exists at 99.4% (train), 99.9% (calibration) and 99.9% (test) of origins.

## Availability rule

On the live `current/` folder (checked 2026-10-08), each run file is posted 35 minutes after its label: run 07:30 at 08:05, 08:00 at 08:35, and so on. `dispatchDataIssueID` shows issue about 15 minutes after the label. Historical ZIP member timestamps are packing times, not publication times.

In the runs pulled, the issue time is 15 minutes after the label for most runs (median 15.15 min for both even and odd runs; 91% of even and 92% of odd runs within 15.5 min), 30 to 45 minutes after it for about 8%, and more than 40 minutes after it for 58 even and 55 odd runs (up to about a day). On the live folder the file appeared 20 minutes after the issue time.

Odd-hour runs follow the same schedule: on the live folder (`live_posting_check_20261008.csv`, 15 runs from 07:30 to 14:30 on 2026-10-08) odd-hour, even-hour and half-hour runs were all posted exactly 35 minutes after the label, so the same rule applies to every run.

Rule used in `scripts/external_features.py`: **available_at = max(label + 40 min, issue time + 25 min)**. At an origin, the latest run with `available_at <= origin` is used. Runs older than 4 hours are treated as missing. With even-hour runs only, the run used is 40 to 160 minutes old at the origin, or up to 240 minutes next to a missing or late-issued run; with all hourly runs, 40 to 100 minutes.

Revision features (stage 1b) compare the latest usable run with the run labelled before it. That previous run is used only if it was itself available by the origin under the same rule and is at most 2 hours older.

Pre-dispatch intervals are half-hourly. The target interval T reads the pre-dispatch interval labelled `floor(T, 30 min)`, plus the next one, from the same run. Both are forecasts issued before the origin.

## Leaving out

- `contingencySolution.dpvLevel` is missing in 2023 runs and is a coarse bucket. It is kept in the extract but not used as a feature.
- `marketShortfalls.energyDeficit` is almost always zero (172 non-zero interval rows in the runs from Oct 2023 to Jul 2024). It is kept as a feature.
