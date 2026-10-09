"""Pull archived day-ahead weather forecasts from the Open-Meteo Previous Runs API.

Only the ``_previous_day1`` offset is requested. Per the Open-Meteo maintainers
(github.com/open-meteo/open-meteo/issues/1750), ``previous_day1`` for a
6-hourly global model uses forecast hours 24-29 of the run initialised at the
latest 00/06/12/18 UTC cycle at least 24 hours before the valid time. Global
runs are typically published 4-6 hours after initialisation
(openmeteo.substack.com/p/single-runs-api). The availability rule used
downstream is therefore conservative: a value valid at hour V (UTC) is treated
as available from V - 12 hours, and never earlier.

Raw responses are written unchanged as JSON under data/external/open_meteo/responses/
and a tidy long CSV is written beside them. Times in the API response are UTC
(``timezone=GMT``); they are converted to Perth (AWST, UTC+8, no DST) in the
tidy file.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "open_meteo"
RAW = OUT / "responses"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

SITES = {
    "perth_metro": (-31.95, 115.86),
    "geraldton": (-28.78, 114.61),
    "badgingarra": (-30.40, 115.50),
    "merredin": (-31.48, 118.28),
    "albany": (-35.03, 117.88),
}
MODELS = ("ecmwf_ifs025", "gfs_seamless")
VARIABLES = (
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_100m",
)
OFFSET = "previous_day1"
CHUNKS = [
    ("2023-09-25", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-08-19"),
]


def fetch(model: str, start: str, end: str) -> list[dict]:
    params = {
        "latitude": ",".join(str(v[0]) for v in SITES.values()),
        "longitude": ",".join(str(v[1]) for v in SITES.values()),
        "hourly": ",".join(f"{v}_{OFFSET}" for v in VARIABLES),
        "models": model,
        "start_date": start,
        "end_date": end,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            body = r.json()
            return body if isinstance(body, list) else [body]
        print(f"{model} {start}: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"failed {model} {start}-{end}")


def tidy(bodies: list[dict], model: str) -> pd.DataFrame:
    frames = []
    for site, body in zip(SITES, bodies):
        hourly = pd.DataFrame(body["hourly"])
        hourly["valid_utc"] = pd.to_datetime(hourly.pop("time"))
        long = hourly.melt(id_vars="valid_utc", var_name="variable", value_name="value")
        long["variable"] = long["variable"].str.replace(f"_{OFFSET}", "", regex=False)
        long["site"] = site
        long["model"] = model
        long["grid_lat"] = body["latitude"]
        long["grid_lon"] = body["longitude"]
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    parts = []
    for model in MODELS:
        for start, end in CHUNKS:
            path = RAW / f"{model}_{OFFSET}_{start}_{end}.json"
            if path.exists():
                bodies = json.loads(path.read_text())
            else:
                bodies = fetch(model, start, end)
                path.write_text(json.dumps(bodies))
                time.sleep(5)
            parts.append(tidy(bodies, model))
            print(model, start, end, "ok", flush=True)
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["model", "site", "variable", "valid_utc"])
    data["valid_awst"] = data["valid_utc"] + pd.Timedelta(hours=8)
    data["issued_utc_upper_bound"] = data["valid_utc"] - pd.Timedelta(hours=24)
    data["available_awst"] = data["valid_awst"] - pd.Timedelta(hours=12)
    data = data.sort_values(["model", "site", "variable", "valid_utc"])
    data.to_csv(OUT / "previous_day1_hourly.csv.gz", index=False)
    cover = (
        data.dropna(subset=["value"])
        .assign(month=lambda d: d["valid_awst"].dt.to_period("M").astype(str))
        .groupby(["model", "variable", "month"])["value"].size()
        .unstack("month")
    )
    cover.to_csv(OUT / "coverage_by_month.csv")
    print(cover.T.to_string())


if __name__ == "__main__":
    main()
