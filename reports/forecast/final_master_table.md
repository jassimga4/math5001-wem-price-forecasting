#### Test

| Model | CRPS | Tail CRPS | 1st-int CRPS | Cov 90 | Width 90 | Tail cov | Tail width | Onset cov | Onset width | Interval score 90 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence + sliding residual window | 4.176 | 14.77 | 42.50 | 0.898 | 21.8 | 0.738 | 24.8 | 0.308 | 26.0 | 55.7 | 4.745 |
| LightGBM + absolute sliding conformal (frozen) | 4.033 | 16.11 | 41.29 | 0.898 | 20.9 | 0.679 | 24.6 | 0.315 | 25.1 | 52.1 | 4.753 |
| LightGBM + normalised sliding conformal | – | – | – | 0.898 | 22.1 | – | – | – | – | – | – |
| LightGBM + fixed split conformal | – | – | – | 0.923 | 23.9 | – | – | – | – | – | – |
| QRA, PR 1: all five members, all calibration (calibration-fitted) † | 3.921 | 15.34 | – | 0.930 | 26.1 | 0.680 | 35.3 | – | – | – | – |
| QRA variant, PR 2: LightGBM + 5-min persistence, 28-day window (selected on calibration) † | 3.916 | – | – | 0.868 | 16.7 | – | – | – | – | – | – |
| QRA variant, PR 2: LightGBM + ridge + 5-min persistence, 28-day window (lowest-CRPS calibration alternative) † | 3.856 | – | – | 0.857 | 16.4 | – | – | – | – | – | – |
| QRM variant, PR 2: LightGBM + ridge + 5-min persistence, 28-day window (not selected) † | 4.126 | – | – | 0.860 | 16.4 | – | – | – | – | – | – |
| Q-Ave variant, PR 2: LightGBM quantiles + LightGBM + ridge + 5-min persistence, 28-day window (not selected) † | 3.868 | – | – | 0.913 | 22.0 | – | – | – | – | – | – |
| LightGBM quantile regression, PR 2: train only (not selected) † | 3.744 | – | – | 0.927 | 34.3 | – | – | – | – | – | – |
| Regime switch + sliding residual window | 3.965 | 14.82 | 41.52 | 0.897 | 20.2 | 0.711 | 23.4 | 0.288 | 24.2 | 51.5 | 4.654 |
| Hurdle, panel only (LightGBM) | 3.907 | 13.42 | 30.38 | – | – | – | – | – | – | – | 4.644 |
| Hurdle + weather | 3.911 | 13.44 | 30.57 | – | – | – | – | – | – | – | 4.712 |
| Hurdle + weather + pre-dispatch (even-hour) | 3.915 | 13.47 | 30.81 | – | – | – | – | – | – | – | 4.835 |
| Hurdle + weather + pre-dispatch (hourly, revisions) | 3.916 | 13.40 | 30.28 | – | – | – | – | – | – | – | 4.873 |
| Hurdle + weather + pre-dispatch (half-hourly, revisions) | 3.917 | 13.45 | 30.69 | – | – | – | – | – | – | – | 4.858 |
| Hurdle, panel + price path (LightGBM) | 3.905 | 13.33 | 29.68 | – | – | – | – | – | – | – | 4.692 |
| Hurdle, panel + path (XGBoost) | 3.903 | 13.30 | 29.47 | – | – | – | – | – | – | – | 4.654 |
| Hurdle, panel + path (HistGradientBoosting) | 3.905 | 13.26 | 29.11 | – | – | – | – | – | – | – | 4.654 |
| Hurdle, panel + path (logistic) | 3.937 | 13.94 | 34.54 | – | – | – | – | – | – | – | 4.639 |
| Hurdle, panel + path (LightGBM, uncapped) | 3.905 | 13.33 | 29.68 | – | – | – | – | – | – | – | 4.692 |
| Hurdle, panel + path (HGB, wide grid) | 3.901 | 13.25 | 29.03 | – | – | – | – | – | – | – | 4.654 |
| Spike forecaster (frozen) | 3.905 | 13.26 | 29.11 | 0.908 | 24.5 | 0.778 | 38.2 | 0.826 | 142.9 | 49.2 | 4.654 |
| CPS: split, regime-switch point | 3.995 | 14.81 | 41.60 | 0.925 | 23.7 | 0.716 | 23.7 | 0.293 | 23.7 | 52.9 | 4.654 |
| CPS: Mondrian by spike risk (fixed calibration) | 4.355 | 19.25 | 34.40 | 0.959 | 51.3 | 0.966 | 246.2 | 0.893 | 141.6 | 60.1 | 4.654 |
| CPS: Mondrian by spike risk (sliding windows) | 3.876 | 14.37 | 37.67 | 0.900 | 25.3 | 0.890 | 85.8 | 0.620 | 54.5 | 46.1 | 4.654 |
| CPS: PIT-recalibrated spike forecaster | 3.899 | 13.28 | 29.18 | 0.904 | 23.6 | 0.770 | 36.4 | 0.811 | 133.3 | 48.8 | – |
| CPS: PIT-recalibrated, Mondrian by spike risk | 4.412 | 20.61 | 29.60 | 0.931 | 55.0 | 0.964 | 356.7 | 0.844 | 153.2 | 65.0 | – |

† The CRPS is the quantile-score integral over the stored levels (0.01 to 0.99), not the exact or empirical CRPS used in the other rows. The frozen LightGBM conformal scores 4.004 on that measure, against 4.033 exact.

#### Calibration

| Model | CRPS | Tail CRPS | 1st-int CRPS | Cov 90 | Width 90 | Tail cov | Tail width | Onset cov | Onset width | Interval score 90 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence + sliding residual window | 4.794 | 62.76 | 102.97 | 0.902 | 28.1 | 0.462 | 28.5 | 0.016 | 33.5 | 61.2 | 5.562 |
| LightGBM + absolute sliding conformal (frozen) | 4.381 | 62.16 | 99.48 | 0.902 | 24.8 | 0.385 | 25.4 | 0.008 | 29.5 | 53.6 | 5.268 |
| Regime switch + sliding residual window | 4.478 | 62.22 | 99.59 | 0.902 | 24.8 | 0.448 | 25.3 | 0.008 | 29.3 | 55.7 | 5.357 |
| Hurdle, panel only (LightGBM) | 4.439 | 56.01 | 75.23 | – | – | – | – | – | – | – | 5.373 |
| Hurdle + weather | 4.441 | 56.46 | 76.99 | – | – | – | – | – | – | – | 5.385 |
| Hurdle + weather + pre-dispatch (even-hour) | 4.436 | 56.32 | 76.43 | – | – | – | – | – | – | – | 5.392 |
| Hurdle + weather + pre-dispatch (hourly, revisions) | 4.438 | 56.20 | 75.97 | – | – | – | – | – | – | – | 5.399 |
| Hurdle + weather + pre-dispatch (half-hourly, revisions) | 4.437 | 56.36 | 76.59 | – | – | – | – | – | – | – | 5.396 |
| Hurdle, panel + price path (LightGBM) | 4.436 | 55.95 | 74.99 | – | – | – | – | – | – | – | 5.382 |
| Hurdle, panel + path (XGBoost) | 4.437 | 55.73 | 74.10 | – | – | – | – | – | – | – | 5.357 |
| Hurdle, panel + path (HistGradientBoosting) | 4.436 | 55.52 | 73.31 | – | – | – | – | – | – | – | 5.357 |
| Hurdle, panel + path (logistic) | 4.459 | 58.47 | 84.88 | – | – | – | – | – | – | – | 5.370 |
| Hurdle, panel + path (LightGBM, uncapped) | 4.436 | 55.95 | 74.98 | – | – | – | – | – | – | – | 5.383 |
| Hurdle, panel + path (HGB, wide grid) | 4.441 | 56.06 | 75.42 | – | – | – | – | – | – | – | 5.358 |
| Spike forecaster (frozen) | 4.436 | 55.53 | 73.32 | 0.905 | 26.0 | 0.593 | 67.0 | 0.576 | 193.6 | 53.5 | 5.357 |
| CPS: split, regime-switch point | 4.657 | 62.66 | 101.00 | 0.868 | 25.0 | 0.438 | 24.1 | 0.040 | 21.3 | 66.4 | 5.357 |
| CPS: Mondrian by spike risk (fixed calibration) | 4.580 | 59.83 | 89.99 | 0.868 | 26.8 | 0.762 | 213.7 | 0.368 | 90.7 | 60.2 | 5.357 |
| CPS: Mondrian by spike risk (sliding windows) | 4.370 | 59.24 | 81.18 | 0.901 | 27.1 | 0.825 | 231.0 | 0.680 | 134.7 | 49.2 | 5.357 |
| CPS: PIT-recalibrated spike forecaster | 4.435 | 55.64 | 73.66 | 0.901 | 25.4 | 0.582 | 64.6 | 0.560 | 185.9 | 53.4 | – |
| CPS: PIT-recalibrated, Mondrian by spike risk | 4.362 | 57.01 | 75.38 | 0.900 | 29.6 | 0.802 | 416.1 | 0.544 | 205.3 | 49.8 | – |

### Bootstrap ranges

Against the spike forecaster (test):

| Model | CRPS | Tail CRPS | First-interval CRPS |
| --- | ---: | ---: | ---: |
| CPS: split, regime-switch point | +0.090 [+0.065, +0.115] | +1.555 [+1.235, +1.926] | +12.491 [+10.034, +15.478] |
| CPS: Mondrian by spike risk (fixed calibration) | +0.450 [+0.357, +0.550] | +5.995 [+5.603, +6.354] | +5.289 [+3.354, +7.671] |
| CPS: Mondrian by spike risk (sliding windows) | -0.029 [-0.052, -0.008] | +1.113 [+0.841, +1.402] | +8.566 [+6.296, +11.375] |
| CPS: PIT-recalibrated spike forecaster | -0.006 [-0.007, -0.005] | +0.027 [+0.022, +0.033] | +0.076 [+0.046, +0.114] |
| CPS: PIT-recalibrated, Mondrian by spike risk | +0.507 [+0.348, +0.701] | +7.350 [+6.202, +8.410] | +0.497 [+0.282, +0.681] |

Against persistence (test):

| Model | CRPS | Tail CRPS | First-interval CRPS |
| --- | ---: | ---: | ---: |
| Spike forecaster (frozen) | -0.271 [-0.312, -0.226] | -1.513 [-1.982, -1.123] | -13.391 [-16.664, -10.654] |
| CPS: split, regime-switch point | -0.181 [-0.207, -0.151] | +0.043 [-0.121, +0.203] | -0.899 [-1.338, -0.542] |
| CPS: Mondrian by spike risk (fixed calibration) | +0.179 [+0.080, +0.293] | +4.482 [+3.884, +5.042] | -8.102 [-9.326, -7.005] |
| CPS: Mondrian by spike risk (sliding windows) | -0.300 [-0.334, -0.258] | -0.400 [-0.677, -0.133] | -4.824 [-5.475, -4.246] |
| CPS: PIT-recalibrated spike forecaster | -0.277 [-0.316, -0.232] | -1.486 [-1.955, -1.093] | -13.314 [-16.564, -10.612] |
| CPS: PIT-recalibrated, Mondrian by spike risk | +0.236 [+0.066, +0.434] | +5.837 [+4.435, +7.081] | -12.894 [-16.158, -10.190] |
