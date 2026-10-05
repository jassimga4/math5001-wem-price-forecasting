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
scripts/forecast_design.py              # ex-ante 5-minute design and frozen splits
scripts/point_models.py                 # persistence, ridge, and LightGBM
scripts/conformal.py                    # sliding-window and fixed split conformal intervals
scripts/qra.py                          # quantile regression averaging comparison
scripts/qra_variants.py                 # recent-window QRA / QRM / Q-Ave / LightGBM quantile study
scripts/experiment.py                   # fit point models, conformal tables, and QRA
scripts/metrics.py                      # MAE, pinball, interval scores, and CRPS
notebooks/00_load_panel.ipynb            # load the processed panel
notebooks/01_eda_correlation.ipynb       # descriptive EDA (not used to tune the test set)
notebooks/02_baseline_models.ipynb       # ex-ante 5-minute baselines
notebooks/03_final_point_model.ipynb     # LightGBM fit on the training window only
notebooks/04_sliding_conformal.ipynb     # sliding-window conformal intervals
models/lgbm_5min_ahead.joblib            # LightGBM saved by the experiment
reports/forecast/                        # point, conformal, and QRA tables
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

The project virtualenv includes **pyarrow**, which pandas needs to read `.parquet` files. Point the notebook kernel at `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows).

## Forecast experiment

The forecast is 5-minute-ahead MCP. The origin is the end of the previous dispatch interval. `scripts/forecast_design.py` reindexes to a complete 5-minute grid before lagging, and it does not use same-interval demand, DPV, SCADA, RTP, or FCAS.

Frozen periods:

| Split | End |
| --- | --- |
| Train | 2025-09-30 23:55 |
| Calibration | 2026-03-31 23:55 |
| Test | after the calibration end, untouched by tuning |

```bash
python -m unittest tests/test_forecast_design.py tests/test_point_models.py tests/test_conformal.py tests/test_qra.py tests/test_qra_variants.py
python scripts/experiment.py
```

Tables are written to `reports/forecast/`. Hyperparameters and the conformal window are chosen on pre-test data only. The LightGBM forecast is the 5-minute lag plus a price change fit under MAE. Conformal coverage, width, pinball and CRPS are reported for the test period and split by month and by tails defined from training quantiles of MCP. Contemporaneous STEM is an explicit day-ahead assumption, not a verified publication timestamp.

Quantile regression averaging is a comparison, not a replacement for the absolute-residual conformal intervals. `scripts/qra.py` implements standard QRA (Nowotarski and Weron, 2015): one linear quantile regression of MCP on a pool of point forecasts. By default those are the 5-minute, 30-minute and 1-day persistence forecasts, the ridge forecast, and the LightGBM forecast. Those are the same origins and the same 5-minute horizon as the conformal experiment. The fit does not see the test set. Test scores are written to `reports/forecast/qra_test.csv`, `qra_by_month.csv`, `qra_regime.csv`, and `qra_crps.csv`. `crps_empirical` is the sliding residual-sample CRPS and is only filled for the frozen conformal rows. `crps_quantile_integral` uses the same quantile grid for QRA and for the frozen conformal residual window; it is not the same estimator as `crps_empirical`.

`python scripts/qra.py` does not refit LightGBM. It reloads `models/lgbm_5min_ahead.joblib`, refits ridge on the training window, recomputes persistence from lags, and reads the frozen conformal method and window from `reports/forecast/experiment_meta.json` without choosing them again. `python scripts/experiment.py` refits the point models, selects the conformal window on calibration, and then writes the same QRA tables.

### QRA variants and recent calibration

`scripts/qra_variants.py` compares the same ex-ante forecasts under three combiners from the electricity-price forecasting literature, plus a native LightGBM quantile forecast:

| Combiner | What it does |
| --- | --- |
| `qra` | Standard QRA: QuantReg of MCP on the member point forecasts (Nowotarski and Weron, 2015) |
| `qrm` | QuantReg of MCP on the mean of the member point forecasts (Marcjasz, Uniejewski and Weron, 2020) |
| `qave` | Quantile averaging: each member first becomes a quantile forecast (univariate QuantReg of MCP on that point forecast, or native LightGBM quantiles), then the sorted member quantiles are averaged level by level |
| `lightgbm_quantile` | LightGBM with the quantile objective, one model per level, fit on the training window only with the frozen point-model hyperparameters |

Member subsets include all five forecasts, LightGBM only, LightGBM + 5-minute persistence, LightGBM + ridge + 5-minute persistence, and (for `qave`) those three plus native LightGBM quantiles. Each calibration-fitted variant is fit on the most recent 28, 56 or 91 days of available pre-test data, or on all of it. Selection uses only an inner split of calibration: windows end on 2026-02-28 23:55 and are scored on March 2026. Among variants with March 90% coverage in [0.88, 0.92], the narrowest mean 90% width is selected (ties: lower quantile-integral CRPS). The selected variant is refit with the same window length ending 2026-03-31 23:55 and scored once on test. Test outcomes are never used to fit, select, or shrink intervals. Tables: `reports/forecast/qra_variants_calib.csv`, `qra_variants_test.csv`, `qra_variants_by_month.csv`, `qra_variants_meta.json`.

```bash
python -m unittest tests/test_forecast_design.py tests/test_point_models.py tests/test_conformal.py tests/test_qra.py tests/test_qra_variants.py
python scripts/qra_variants.py
```

If you still see `Missing optional dependency 'pyarrow'`, the notebook or terminal is using a different Python:

```bash
python -m pip install pyarrow
```
