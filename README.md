# MATH5001 — WEM electricity price forecasting

Curtin MATH5001 project: probabilistic electricity price forecasting for the Western Australia Wholesale Electricity Market (WEM), using conformal prediction.

Processed modelling files live in this repository. Raw AEMO CSVs (~2.5 GB) stay in the original download folder and are not committed.

## Layout

```
data/processed/wem_5min_panel.parquet   # 5-minute modelling panel
data/processed/wem_5min_panel_sample.csv
scripts/build_panel.py                  # rebuild the panel from raw AEMO CSVs
scripts/inspect_panel.py                # print panel shape / coverage
notebooks/                              # analysis notebooks
Dockerfile
docker-compose.yml
```

The panel is a 5-minute table (price as the spine) covering 1 Oct 2023 onwards: market clearing price, demand, distributed PV, STEM, RTP, SCADA, and calendar features.

## Docker

Docker Desktop must be running.

```bash
cd "/Users/jg/Documents/MATH5001 Project"
docker compose build
```

JupyterLab (token `math5001`):

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

Raw AEMO files remain at:

`/Users/jg/Desktop/Demand Forecasting AEMO/data`

```bash
docker compose run --rm shell python scripts/build_panel.py \
  --data-dir "/Users/jg/Desktop/Demand Forecasting AEMO/data" \
  --out data/processed/wem_5min_panel.parquet
```

## Local Python (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/inspect_panel.py
```
