# External data for the spike-onset work

Each source folder has its own README with URL, parameters, time zone and availability rule.

| Folder | Source | Used | Availability rule |
| --- | --- | --- | --- |
| `open_meteo/` | Open-Meteo Previous Runs API, `previous_day1` (ECMWF IFS 0.25°, GFS) | yes | value valid at V usable from V - 12 h |
| `aemo_predispatch/` | AEMO WEM Reference pre-dispatch runs (even hours in stage 1, every hour in stage 1b, every half hour in stage 1c) | yes | run label + 40 min, or issue time + 25 min if later |

## Checked and not used

| Source | Why it is left out |
| --- | --- |
| Bureau of Meteorology forecasts (ADFD) | Real-time only, through the paid Registered User service. The BoM charges page says these "are real-time data only" and points to Climate Data Services for history (https://reg.bom.gov.au/other/charges.shtml). No free archive of issued forecasts was found. BoM ACCESS-G is in the Open-Meteo archive and could be added the same way. |
| WEMDE `notInServiceCapacity` (5-minute JSON per dispatch interval) | Sampled six days between Oct 2023 and Apr 2026. The total was 1-31 MW per interval against ~4,000 MW of in-service capacity, so it is not an outage series. Pre-dispatch `inServiceQuantities` is used instead. |
| `public-data/datafiles/outages`, `realtime-outages` | Last files are 2023 (`outages-2023.csv` dated 25 Jan 2024; `realtime-outages.csv` dated 29 Nov 2023). Nothing for the modelling window. |
| `public-data/datafiles/load-forecast`, `extended-load-forecast` | Single files dated 3 Oct 2023 (pre-reform). |
| `market-schedule-csv`, `market-requirements-csv` | Monthly CSVs of the dispatch solution for each interval (energy schedule, energy requirement). They are realised same-interval values with only an `Extracted At` stamp, not a forecast vintage. Using them at the target interval would leak. |
| `dsp/scheduleInformation/DSPPre-Dispatch` | Demand-side programme facilities only (three facilities). Not a system forecast. |
| WEMDE `preDispatchData-AvailableCapacity` | Starts 13 Dec 2024 in the archive and is as large as the main pre-dispatch files. The main runs already carry available and in-service capacity. |
