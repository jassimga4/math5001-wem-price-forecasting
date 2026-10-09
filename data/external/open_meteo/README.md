# Open-Meteo archived day-ahead weather forecasts

Pulled by `scripts/pull_open_meteo.py` on 2026-10-08.

- Endpoint: `https://previous-runs-api.open-meteo.com/v1/forecast` (Previous Runs API, docs at https://open-meteo.com/en/docs/previous-runs-api)
- Parameters: `hourly=<variable>_previous_day1`, `models=ecmwf_ifs025` and `models=gfs_seamless`, `timezone=GMT`, `wind_speed_unit=ms`, five half-year chunks from `start_date=2023-09-25` to `end_date=2026-08-19`
- Variables: `temperature_2m` (°C), `cloud_cover` (%), `shortwave_radiation` (W/m², mean over the preceding hour), `wind_speed_100m` (m/s)
- Sites (requested lat, lon; the response gives the model grid point):

| Site | Lat | Lon | Why |
| --- | ---: | ---: | --- |
| perth_metro | -31.95 | 115.86 | demand, temperature, cloud, rooftop PV |
| geraldton | -28.78 | 114.61 | northern wind farms (Walkaway, Greenough) |
| badgingarra | -30.40 | 115.50 | Badgingarra, Warradarge, Yandin wind |
| merredin | -31.48 | 118.28 | Collgar wind, Merredin solar (Wheatbelt) |
| albany | -35.03 | 117.88 | Albany and Grasmere wind |

## Files

- `responses/*.json`: API responses, unchanged (one list entry per site, same order as above)
- `previous_day1_hourly.csv.gz`: tidy long table. `valid_utc` is the API time; `valid_awst = valid_utc + 8 h`
- `coverage_by_month.csv`: non-null hourly values per model, variable and month

## Coverage

GFS 2 m temperature is filled from the start. GFS cloud and radiation start in January 2024, GFS 100 m wind in February 2024. ECMWF IFS 0.25° starts February (temperature, cloud) to March 2024 (radiation, wind). From March 2024 to 2026-08-19 both models are complete. The feature code averages whichever models are present at each hour.

## Issue time and availability rule

`_previous_day1` is not the latest forecast. An Open-Meteo maintainer describes it (github.com/open-meteo/open-meteo/issues/1750) as forecast hours 24-29 of the 00z run for valid hours 00-05 the next day, hours 30-35 of the 06z run, and so on. A value valid at V was therefore issued by a run initialised between 24 and 29 hours before V. Global runs reach the API about 4-6 hours after initialisation (openmeteo.substack.com/p/single-runs-api).

Rule used in `scripts/external_features.py`: a value valid at V is available from **V - 12 h**. That leaves at least 6 hours of publication delay after the latest possible initialisation. Features read hours up to 3 h past the target interval, so every value used was available at least 9 hours before the origin. `available_awst` in the tidy file is this rule.

This is a forecast archive, not observed weather. Nothing from the target interval is observed.

## Licence

Open-Meteo data is CC BY 4.0. The free API is for non-commercial use.
