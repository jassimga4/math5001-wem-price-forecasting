# AEMO WEM Reference pre-dispatch (system-level extract)

Pulled by `scripts/pull_aemo_predispatch.py` on 2026-10-08.

- Source: `https://data.wa.aemo.com.au/public/market-data/wemde/dispatchSolution/preDispatchData/previous/DispatchSolutionPreDispatch_YYYYMMDD.zip`
- One ZIP per trading day (08:00 to 07:30 next day, AWST). Each holds 48 half-hourly runs `ReferencePre-DispatchSolution_YYYYMMDDHHMM.json`. Each run solves 96 half-hourly future dispatch intervals (48 h ahead).
- Time zone: AWST (UTC+8). The JSON carries `+08:00` offsets. Stored times are naive AWST.

## What was pulled

The archive is ~160-250 MB per day (~250 GB from October 2023), and the server gave about 0.15-0.25 MB/s per connection. Only selected runs were read with HTTP range requests, and only system-level fields were kept. Per-facility schedules and constraints were dropped.

- `extract/pass1/YYYYMMDD.parquet`: runs labelled at even hours (00:00, 02:00, ..., 22:00), 12 per day
- Columns: `run_label`, `issue_id_time` (from `dispatchDataIssueID`), `dispatch_interval`, `solve_status`, and per interval: `prices.*` (energy and FCESS), `marketServiceRequirements.*` (energy requirement, contingency and regulation raise), `inServiceQuantities.energyInjectionCapacity`, `availableQuantities.*`, `dispatchTotal.*`, `marketShortfalls.*`, `contingencySolution.demandLevel/dpvLevel/largestContingency`

Odd-hour runs (`--pass 2`) were not pulled. Run it to halve the lead time; the feature code uses whatever runs exist.

## Availability rule

On the live `current/` folder (checked 2026-10-08), each run file is posted 35 minutes after its label: run 07:30 at 08:05, 08:00 at 08:35, and so on. `dispatchDataIssueID` shows issue about 15 minutes after the label. Historical ZIP member timestamps are packing times, not publication times.

In the runs pulled, the issue time is 15 minutes after the label for most runs, and 30 to 45 minutes after it for a few. On the live folder the file appeared 20 minutes after the issue time.

Rule used in `scripts/external_features.py`: **available_at = max(label + 40 min, issue time + 25 min)**. At an origin, the latest run with `available_at <= origin` is used. Runs older than 4 hours are treated as missing. With even-hour runs only, the run used is 40 to 160 minutes old at the origin.

Pre-dispatch intervals are half-hourly. The target interval T reads the pre-dispatch interval labelled `floor(T, 30 min)`, plus the next one, from the same run. Both are forecasts issued before the origin.

## Leaving out

- `contingencySolution.dpvLevel` is missing in 2023 runs and is a coarse bucket. It is kept in the extract but not used as a feature.
- `marketShortfalls.energyDeficit` is almost always zero (172 non-zero interval rows in the runs from Oct 2023 to Jul 2024). It is kept as a feature.
