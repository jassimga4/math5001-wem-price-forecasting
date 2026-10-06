# Next steps for spike-price forecasting

The average error is not the gap to close. On the frozen test window LightGBM and 5-minute persistence are tied (MAE 4.75 against 4.74). LightGBM is better on ordinary prices (3.60 against 3.77). Persistence is better on the tails (16.00 against 18.04). Those tails are about 8% of test intervals and are the consequential miss.

The conformal band does not fix this. It is the point forecast plus two residual quantiles. Swapping the residual fence moved CRPS by about 0.04. Swapping the centre moved it by about 0.14 to 0.18. Coverage falls to zero once the point error is above $50/MWh. Report the 89.8% hit rate as calibration, not as a better price forecast.

QRA is the blend benchmark, not the spike method. It wins the shared quantile-integral CRPS (3.92 against 4.00 and 4.14) and still loses the tail to persistence (15.34 against 14.63), because a blend cannot hand the violent interval entirely to the last price.

## Regime-switch baseline

`scripts/regime_switch.py` chooses a gate on calibration only. If the absolute move over the previous 30 minutes is at least $32.48, the 90th percentile of that move in early calibration, the issued forecast is persistence. Otherwise it is the saved LightGBM model. The frozen 7-day absolute conformal fence is drawn around whichever centre was issued. The rule fires on 7.6% of test intervals.

| Model | MAE | CRPS | 90% coverage | Mean width | Tail MAE | Tail CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regime switch | 4.65 | 3.98 | 89.8% | 20.20 | 16.13 | 14.70 |
| LightGBM | 4.75 | 4.03 | 89.8% | 20.89 | 18.04 | 16.11 |
| Persistence | 4.74 | 4.18 | 89.8% | 21.83 | 16.00 | 14.77 |

This is not an onset forecast. The gate only sees a move after it has happened. A better spike model has to beat this row on calibration, then on the untouched test window.

## Data still to add

Do not add more lags of MCP, and do not add anything realised in the target interval. Same-interval demand, DPV and SCADA leak. Each new series needs a publication time, and a value cannot be used before that time.

- Pre-dispatch demand, from the WEM market data site or Dispatch API. Feature: forecast for the target interval minus last realised demand.
- Projected DPV. Feature: forecast drop over the next 5 to 30 minutes, not the level.
- Outages and not-in-service capacity known by the origin. Feature: available capacity minus forecast demand. A daily reserve-capacity number is too slow.
- Bureau temperature and cloud forecast for Perth and the SWIS, issued before the origin. Not the weather observed in the interval.
- Pre-dispatch price only if it was published before the origin. Do not take it from the same row as realised MCP.
- FCESS requirement against available raise, as a later trial. Lagged FCAS prices are already in the panel and are the lagging piece.

## Model to fit once that data is attached

Keep the absolute-error LightGBM for ordinary prices. Do not refit it and expect a tail.

- Hurdle model: a classifier for crossing the training 5th–95th band, then a size model on those rows. The probability cutoff is chosen on calibration and frozen.
- Or a second LightGBM with pinball loss at 0.95 or 0.99, used only when the gate says the interval is violent.
- Leave persistence in charge after the price is already in the tail, unless the new model beats it on that bucket.

Judge the result on the first interval of a tail event, the rest of the tail event, and tail CRPS. Not on overall MAE. Leave the conformal window frozen until the centre improves.
