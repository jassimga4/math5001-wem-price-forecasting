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
scripts/regime_switch.py                # calibration-chosen LightGBM / persistence gate
scripts/pull_open_meteo.py              # archived day-ahead weather forecasts (Open-Meteo Previous Runs)
scripts/pull_aemo_predispatch.py        # system-level fields from AEMO WEM pre-dispatch runs
scripts/external_features.py            # leakage-safe external features keyed to forecast origins
scripts/spike_onset.py                  # stage-1 hurdle onset model vs persistence, LightGBM, regime switch
scripts/price_path.py                   # recent price-path features known at the origin
scripts/spike_forecaster.py             # implemented spike forecaster: fit, save/load, forecast CLI
scripts/spike_figures.py                # reliability, episode and first-interval CRPS figures
configs/spike_forecaster.json           # frozen spike forecaster config (chosen on calibration)
models/spike_forecaster.joblib          # fitted spike forecaster
data/external/                          # external pulls, one README per source
scripts/experiment.py                   # fit point models, conformal tables, and QRA
scripts/metrics.py                      # MAE, pinball, interval scores, and CRPS
docs/spike_forecast_next_steps.md       # onset-model plan
docs/spike_stage1_results.md            # stage-1 sources, availability rules and results
notebooks/00_load_panel.ipynb            # load the processed panel
notebooks/01_eda_correlation.ipynb       # descriptive EDA (not used to tune the test set)
notebooks/02_baseline_models.ipynb       # ex-ante 5-minute baselines
notebooks/03_final_point_model.ipynb     # LightGBM fit on the training window only
notebooks/04_sliding_conformal.ipynb     # sliding-window conformal intervals
models/lgbm_5min_ahead.joblib            # LightGBM saved by the experiment
reports/forecast/                        # point, conformal, QRA, regime-switch and spike tables
reports/figures/spike/                   # spike forecaster figures
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
python -m unittest tests/test_forecast_design.py tests/test_point_models.py tests/test_conformal.py tests/test_qra.py
python scripts/experiment.py
```

Tables are written to `reports/forecast/`. Hyperparameters and the conformal window are chosen on pre-test data only. The LightGBM forecast is the 5-minute lag plus a price change fit under MAE. Conformal coverage, width, pinball and CRPS are reported for the test period and split by month and by tails defined from training quantiles of MCP. Contemporaneous STEM is an explicit day-ahead assumption, not a verified publication timestamp.

Quantile regression averaging is a comparison, not a replacement for the absolute-residual conformal intervals. `scripts/qra.py` regresses calibration MCP on the 5-minute, 30-minute and 1-day persistence forecasts, the ridge forecast, and the LightGBM forecast. Those are the same origins and the same 5-minute horizon as the conformal experiment. The fit does not see the test set. Test scores are written to `reports/forecast/qra_test.csv`, `qra_by_month.csv`, `qra_regime.csv`, and `qra_crps.csv`. `crps_empirical` is the sliding residual-sample CRPS and is only filled for the frozen conformal rows. `crps_quantile_integral` uses the same quantile grid for QRA and for the frozen conformal residual window; it is not the same estimator as `crps_empirical`.

`python scripts/qra.py` does not refit LightGBM. It reloads `models/lgbm_5min_ahead.joblib`, refits ridge on the training window, recomputes persistence from lags, and reads the frozen conformal method and window from `reports/forecast/experiment_meta.json` without choosing them again. `python scripts/experiment.py` refits the point models, selects the conformal window on calibration, and then writes the same QRA tables.

## Regime switch and spike next steps

`python scripts/regime_switch.py` does not refit LightGBM. On late calibration it chooses a gate: persistence if the absolute 30-minute move is at least $32.48, otherwise the saved LightGBM forecast. The frozen 7-day absolute conformal fence is applied to that switched centre. Test scores are in `reports/forecast/regime_cps_comparison.csv`. The switch improves test MAE from 4.75 to 4.65 and empirical CRPS from 4.03 to 3.98, and it closes most of the tail gap to persistence. It is not an onset model.

`docs/spike_forecast_next_steps.md` is the plan for the spike piece: pre-dispatch demand, projected DPV, outages known by the origin, and a Bureau forecast, then a hurdle or pinball model beside the current tree. Those series are not in the panel yet. QRA stays the blend benchmark, not the spike method.

## Spike stage 1: external data and an onset model

`data/external/` holds archived day-ahead weather forecasts (Open-Meteo, ECMWF and GFS) and an extract of AEMO WEM pre-dispatch runs. Each folder has a README with the URL, time zone and availability rule. `scripts/external_features.py` uses a value only if it was published by the forecast origin. `tests/test_external_features.py` checks that, including that changing anything published after the origin cannot change a feature. `scripts/spike_onset.py` fits a hurdle model on train, picks its cutoffs on calibration and scores test once. Results and caveats are in `docs/spike_stage1_results.md`.

```bash
python scripts/pull_open_meteo.py                  # weather (cached responses are reused)
python scripts/pull_aemo_predispatch.py --pass 1   # slow: about 1 h for the even-hour runs
python scripts/pull_aemo_predispatch.py --pass 2   # odd-hour runs (stage 1b), about 1 h
python scripts/pull_aemo_predispatch.py --pass 3   # half-hour runs (stage 1c), about 1.5 h
python scripts/pull_aemo_predispatch.py --consolidate
python -m unittest tests/test_external_features.py
python scripts/spike_onset.py
```

The onset model needs `remotezip` for the pre-dispatch pull only. Features are built from the committed `data/external/aemo_predispatch/predispatch_runs_first9h.parquet` (about 31 MB, all half-hourly runs), so a fresh clone reproduces the results without the pull; the per-day extracts are used only if that file is absent.

## Spike forecaster

The implemented spike forecaster is frozen in `configs/spike_forecaster.json`. It uses panel and price-path features and a HistGradientBoosting spike classifier, LightGBM quantile size models, a mixture around the regime switch, and the regime switch as the point forecast. For each target interval, at origin T − 5 min, it gives the point forecast, P(spike up), P(spike down), quantiles (q01 to q99) and a spike alert. See "Implemented spike forecaster" in `docs/spike_stage1_results.md`.

```bash
python scripts/spike_forecaster.py fit        # fit on train, cutoffs on calibration; saves models/spike_forecaster.joblib
python scripts/spike_forecaster.py evaluate   # test-window forecasts and metrics; checks they reproduce the frozen row
python scripts/spike_forecaster.py forecast --start "2026-08-17 00:00" --end "2026-08-18 07:55" --out forecasts.csv
python scripts/spike_figures.py               # figures in reports/figures/spike/
python -m pytest tests/test_spike_forecaster.py
```

If you still see `Missing optional dependency 'pyarrow'`, the notebook or terminal is using a different Python:

```bash
python -m pip install pyarrow
```
