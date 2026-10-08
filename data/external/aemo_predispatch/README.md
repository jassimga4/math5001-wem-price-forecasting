# AEMO WEM Reference pre-dispatch (system-level extract)

Pulled by `scripts/pull_aemo_predispatch.py` on 2026-10-08.

- Source: `https://data.wa.aemo.com.au/public/market-data/wemde/dispatchSolution/preDispatchData/previous/DispatchSolutionPreDispatch_YYYYMMDD.zip`
- One ZIP per trading day (08:00 to 07:30 next day, AWST). Each holds 48 half-hourly runs `ReferencePre-DispatchSolution_YYYYMMDDHHMM.json`. Each run solves 96 half-hourly future dispatch intervals (48 h ahead).
- Time zone: AWST (UTC+8). The JSON carries `+08:00` offsets. Stored times are naive AWST.

## What was pulled

The archive is ~160-250 MB per day (~250 GB from October 2023), and the server gave about 0.15-0.25 MB/s per connection. Only selected runs were read with HTTP range requests, and only system-level fields were kept. Per-facility schedules and constraints were dropped.

- `extract/pass1/YYYYMMDD.parquet`: runs labelled at even hours (00:00, 02:00, ..., 22:00), 12 per day
- Columns: `run_label`, `issue_id_time` (from `dispatchDataIssueID`), `dispatch_interval`, `solve_status`, and per interval: `prices.*` (energy and FCESS), `marketServiceRequirements.*` (energy requirement, contingency and regulation raise), `inServiceQuantities.energyInjectionCapacity`, `availableQuantities.*`, `dispatchTotal.*`, `marketShortfalls.*`, `contingencySolution.demandLevel/dpvLevel/largestContingency`

- `predispatch_runs_first9h.parquet` (committed, ~11 MB): all pass-1 extracts merged, first 18 half-hourly intervals (9 h) of each run. 12,561 runs from 2023-09-30 08:00 to 2026-08-18 06:00 AWST, 226,098 rows. Features read at most 4 h (run age) + 2 h (look-ahead) past a run label, so this file gives the same features as the full extracts. `scripts/external_features.py` reads it first.
- `missing_runs.csv` (committed): the 75 even-hour runs not in the extracts. Every trading day from 2023-09-30 to 2026-08-17 has a ZIP; the pull logged no download failures, so these runs are absent from the archive ZIPs. Most fall in train (69: a block from 2023-12-19 16:00 to 2023-12-21 08:00, another from 2024-02-22 10:00 to 2024-02-23 12:00, and single runs); 6 fall in calibration; none in test.

Rebuild: `python scripts/pull_aemo_predispatch.py --pass 1` (re-reads only missing days into the gitignored `extract/`), then `python scripts/pull_aemo_predispatch.py --consolidate`.

## Coverage of forecast origins

From `reports/forecast/spike_predispatch_coverage.csv` (an origin is covered if a run no older than 4 h was published by then):

| Split | Targets | Covered | Even-hour runs present / expected |
| --- | --- | ---: | ---: |
| train | 2023-10-02 08:00 to 2025-09-30 23:55 | 99.28% | 8,687 / 8,756 |
| calibration | 2025-10-01 00:00 to 2026-03-31 23:55 | 99.84% | 2,178 / 2,184 |
| test | 2026-04-01 00:00 to 2026-08-18 07:55 | 100.00% | 1,672 / 1,672 |

The run used is 40 to 240 minutes old at the origin (median 105); above 160 minutes (1.4% of origins) only next to a missing run or a run issued late.

Odd-hour runs (`--pass 2`) were not pulled. Run it to halve the lead time; the feature code uses whatever runs exist.

## Availability rule

On the live `current/` folder (checked 2026-10-08), each run file is posted 35 minutes after its label: run 07:30 at 08:05, 08:00 at 08:35, and so on. `dispatchDataIssueID` shows issue about 15 minutes after the label. Historical ZIP member timestamps are packing times, not publication times.

In the runs pulled, the issue time is 15 minutes after the label for most runs, and 30 to 45 minutes after it for a few. On the live folder the file appeared 20 minutes after the issue time.

Rule used in `scripts/external_features.py`: **available_at = max(label + 40 min, issue time + 25 min)**. At an origin, the latest run with `available_at <= origin` is used. Runs older than 4 hours are treated as missing. With even-hour runs only, the run used is 40 to 160 minutes old at the origin, or up to 240 minutes next to a missing or late-issued run.

Pre-dispatch intervals are half-hourly. The target interval T reads the pre-dispatch interval labelled `floor(T, 30 min)`, plus the next one, from the same run. Both are forecasts issued before the origin.

## Leaving out

- `contingencySolution.dpvLevel` is missing in 2023 runs and is a coarse bucket. It is kept in the extract but not used as a feature.
- `marketShortfalls.energyDeficit` is almost always zero (172 non-zero interval rows in the runs from Oct 2023 to Jul 2024). It is kept as a feature.
