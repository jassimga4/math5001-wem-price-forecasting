# MATH5001 — WEM electricity price forecasting

Curtin MATH5001 project: probabilistic electricity price forecasting for the Western Australia Wholesale Electricity Market (WEM), using conformal prediction.

Processed modelling files live in this repository. Raw AEMO CSVs (~2.5 GB) stay local and are not committed.

## Layout

```
data/processed/wem_5min_panel.parquet   # 5-minute modelling panel
data/processed/wem_5min_panel_sample.csv
scripts/build_panel.py                  # rebuild the panel from raw AEMO CSVs
scripts/inspect_panel.py                # print panel shape / coverage
scripts/paths.py                        # POSIX path + .env helpers
notebooks/                              # analysis notebooks
.env.example                            # copy to .env
Dockerfile
docker-compose.yml
```

The panel is a 5-minute table (price as the spine) covering 1 Oct 2023 onwards: market clearing price, demand, distributed PV, STEM, RTP, SCADA, and calendar features.

## Configuration

Copy the example env file and edit POSIX paths as needed:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `PANEL_PATH` | `data/processed/wem_5min_panel.parquet` | Processed modelling panel (POSIX path, relative to the project root) |
| `RAW_DATA_DIR` | `raw` | Directory of downloaded AEMO CSVs for `scripts/build_panel.py` |
| `JUPYTER_TOKEN` | `math5001` | JupyterLab token |
| `JUPYTER_PORT` | `8888` | Host port for JupyterLab |
| `TZ` | `Australia/Perth` | Container timezone |

Relative paths in `.env` use forward slashes and are resolved from the project root.

## Docker

Docker Desktop must be running. The image installs packages into `/opt/venv` (not bind-mounted), so the container does not use a host virtualenv.

```bash
docker compose build
```

JupyterLab (token from `JUPYTER_TOKEN` in `.env`):

```bash
docker compose up lab
```

Then open http://localhost:8888 and enter the token.

Interactive shell:

```bash
docker compose run --rm shell
```

Confirm the panel loads:

```bash
docker compose run --rm inspect
```

## Rebuild the panel

Put raw AEMO CSVs under `RAW_DATA_DIR` (default `raw/`), or set that variable in `.env` to a POSIX path visible in the container.

```bash
docker compose run --rm shell python scripts/build_panel.py
```

## Local Python (without Docker)

Create a virtualenv at `.venv` if you do not already have one:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/inspect_panel.py
```

The project virtualenv includes **pyarrow**, which pandas needs to read `.parquet` files. Point the notebook kernel at `.venv/bin/python`.

If you still see `Missing optional dependency 'pyarrow'`, the notebook or terminal is using a different Python:

```bash
python -m pip install pyarrow
```
