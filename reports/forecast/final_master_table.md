#### Test

| Model | CRPS | Tail CRPS | 1st-int CRPS | Cov 90 | Width 90 | Tail cov | Tail width | Onset cov | Onset width | Interval score 90 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence + sliding residual window | 4.176 | 14.77 | 42.50 | 0.898 | 21.8 | 0.738 | 24.8 | 0.308 | 26.0 | 55.7 | 4.745 |
| LightGBM + absolute sliding conformal (frozen) | 4.033 | 16.11 | 41.29 | 0.898 | 20.9 | 0.679 | 24.6 | 0.315 | 25.1 | 52.1 | 4.753 |
| LightGBM + normalised sliding conformal | – | – | – | 0.898 | 22.1 | – | – | – | – | – | – |
| LightGBM + fixed split conformal | – | – | – | 0.923 | 23.9 | – | – | – | – | – | – |
| QRA (calibration-fitted) | 3.921 | 15.34 | – | 0.930 | 26.1 | 0.680 | 35.3 | – | – | – | – |
| Regime switch + sliding residual window | 3.975 | 14.70 | 41.57 | 0.898 | 20.2 | 0.719 | 23.2 | 0.293 | 24.1 | 52.0 | 4.650 |
| Hurdle, panel only (LightGBM) | 3.917 | 13.29 | 30.37 | – | – | – | – | – | – | – | 4.640 |
| Hurdle + weather | 3.921 | 13.32 | 30.55 | – | – | – | – | – | – | – | 4.709 |
| Hurdle + weather + pre-dispatch (even-hour) | 3.925 | 13.35 | 30.80 | – | – | – | – | – | – | – | 4.831 |
| Hurdle + weather + pre-dispatch (hourly, revisions) | 3.926 | 13.28 | 30.28 | – | – | – | – | – | – | – | 4.869 |
| Hurdle + weather + pre-dispatch (half-hourly, revisions) | 3.926 | 13.33 | 30.68 | – | – | – | – | – | – | – | 4.854 |
| Hurdle, panel + price path (LightGBM) | 3.914 | 13.20 | 29.66 | – | – | – | – | – | – | – | 4.689 |
| Hurdle, panel + path (XGBoost) | 3.912 | 13.18 | 29.44 | – | – | – | – | – | – | – | 4.650 |
| Hurdle, panel + path (HistGradientBoosting) | 3.915 | 13.13 | 29.08 | – | – | – | – | – | – | – | 4.695 |
| Hurdle, panel + path (logistic) | 3.947 | 13.82 | 34.54 | – | – | – | – | – | – | – | 4.635 |
| Hurdle, panel + path (LightGBM, uncapped) | 3.914 | 13.20 | 29.66 | – | – | – | – | – | – | – | 4.689 |
| Hurdle, panel + path (HGB, wide grid) | 3.911 | 13.12 | 29.01 | – | – | – | – | – | – | – | 4.650 |
| Spike forecaster (frozen) | 3.915 | 13.13 | 29.08 | 0.908 | 24.4 | 0.785 | 38.0 | 0.826 | 143.0 | 49.6 | 4.650 |
| CPS: split, regime-switch point | 4.006 | 14.68 | 41.58 | 0.926 | 23.8 | 0.725 | 23.8 | 0.303 | 23.8 | 53.4 | 4.650 |
| CPS: Mondrian by spike risk (fixed calibration) | 4.388 | 19.51 | 34.63 | 0.960 | 52.2 | 0.966 | 246.7 | 0.893 | 145.7 | 60.9 | 4.650 |
| CPS: Mondrian by spike risk (sliding windows) | 3.885 | 14.29 | 37.78 | 0.900 | 25.4 | 0.891 | 86.8 | 0.623 | 54.9 | 46.4 | 4.650 |
| CPS: PIT-recalibrated spike forecaster | 3.909 | 13.16 | 29.15 | 0.905 | 23.5 | 0.779 | 36.2 | 0.811 | 133.4 | 49.3 | – |
| CPS: PIT-recalibrated, Mondrian by spike risk | 4.452 | 21.04 | 29.96 | 0.931 | 55.6 | 0.964 | 358.8 | 0.841 | 153.3 | 65.7 | – |

#### Calibration

| Model | CRPS | Tail CRPS | 1st-int CRPS | Cov 90 | Width 90 | Tail cov | Tail width | Onset cov | Onset width | Interval score 90 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence + sliding residual window | 4.794 | 62.76 | 102.97 | 0.902 | 28.1 | 0.462 | 28.5 | 0.016 | 33.5 | 61.2 | 5.562 |
| LightGBM + absolute sliding conformal (frozen) | 4.381 | 62.16 | 99.48 | 0.902 | 24.8 | 0.385 | 25.4 | 0.008 | 29.5 | 53.6 | 5.268 |
| Regime switch + sliding residual window | 4.550 | 62.35 | 100.39 | 0.902 | 25.3 | 0.450 | 25.7 | 0.016 | 30.0 | 57.2 | 5.422 |
| Hurdle, panel only (LightGBM) | 4.511 | 55.97 | 75.34 | – | – | – | – | – | – | – | 5.437 |
| Hurdle + weather | 4.513 | 56.42 | 77.10 | – | – | – | – | – | – | – | 5.448 |
| Hurdle + weather + pre-dispatch (even-hour) | 4.508 | 56.27 | 76.52 | – | – | – | – | – | – | – | 5.455 |
| Hurdle + weather + pre-dispatch (hourly, revisions) | 4.509 | 56.17 | 76.11 | – | – | – | – | – | – | – | 5.463 |
| Hurdle + weather + pre-dispatch (half-hourly, revisions) | 4.508 | 56.32 | 76.70 | – | – | – | – | – | – | – | 5.459 |
| Hurdle, panel + price path (LightGBM) | 4.507 | 55.91 | 75.09 | – | – | – | – | – | – | – | 5.445 |
| Hurdle, panel + path (XGBoost) | 4.508 | 55.67 | 74.16 | – | – | – | – | – | – | – | 5.422 |
| Hurdle, panel + path (HistGradientBoosting) | 4.508 | 55.47 | 73.36 | – | – | – | – | – | – | – | 5.458 |
| Hurdle, panel + path (logistic) | 4.530 | 58.41 | 84.92 | – | – | – | – | – | – | – | 5.434 |
| Hurdle, panel + path (LightGBM, uncapped) | 4.507 | 55.91 | 75.08 | – | – | – | – | – | – | – | 5.447 |
| Hurdle, panel + path (HGB, wide grid) | 4.512 | 56.01 | 75.49 | – | – | – | – | – | – | – | 5.423 |
| Spike forecaster (frozen) | 4.508 | 55.47 | 73.37 | 0.905 | 26.5 | 0.593 | 67.5 | 0.576 | 194.8 | 55.0 | 5.422 |
| CPS: split, regime-switch point | 4.737 | 62.81 | 101.84 | 0.867 | 25.5 | 0.446 | 24.5 | 0.040 | 21.5 | 68.7 | 5.422 |
| CPS: Mondrian by spike risk (fixed calibration) | 4.659 | 60.28 | 91.00 | 0.868 | 27.2 | 0.760 | 213.9 | 0.360 | 91.4 | 62.2 | 5.422 |
| CPS: Mondrian by spike risk (sliding windows) | 4.440 | 59.77 | 82.14 | 0.902 | 27.5 | 0.823 | 231.7 | 0.672 | 137.1 | 50.5 | 5.422 |
| CPS: PIT-recalibrated spike forecaster | 4.506 | 55.57 | 73.69 | 0.901 | 25.8 | 0.587 | 65.1 | 0.560 | 187.1 | 54.9 | – |
| CPS: PIT-recalibrated, Mondrian by spike risk | 4.429 | 57.32 | 76.20 | 0.899 | 29.9 | 0.798 | 416.2 | 0.528 | 203.1 | 51.2 | – |

### Bootstrap ranges

Against the spike forecaster (test):

| Model | CRPS | Tail CRPS | First-interval CRPS |
| --- | ---: | ---: | ---: |
| CPS: split, regime-switch point | +0.091 [+0.066, +0.117] | +1.552 [+1.231, +1.932] | +12.503 [+10.034, +15.522] |
| CPS: Mondrian by spike risk (fixed calibration) | +0.473 [+0.377, +0.580] | +6.383 [+5.996, +6.752] | +5.546 [+3.573, +7.962] |
| CPS: Mondrian by spike risk (sliding windows) | -0.029 [-0.051, -0.008] | +1.164 [+0.894, +1.462] | +8.698 [+6.387, +11.527] |
| CPS: PIT-recalibrated spike forecaster | -0.006 [-0.006, -0.005] | +0.024 [+0.020, +0.030] | +0.073 [+0.043, +0.109] |
| CPS: PIT-recalibrated, Mondrian by spike risk | +0.537 [+0.368, +0.745] | +7.913 [+6.678, +9.072] | +0.882 [+0.714, +1.058] |

Against persistence (test):

| Model | CRPS | Tail CRPS | First-interval CRPS |
| --- | ---: | ---: | ---: |
| Spike forecaster (frozen) | -0.261 [-0.299, -0.221] | -1.637 [-2.103, -1.251] | -13.417 [-16.717, -10.696] |
| CPS: split, regime-switch point | -0.170 [-0.197, -0.143] | -0.084 [-0.242, +0.067] | -0.914 [-1.324, -0.579] |
| CPS: Mondrian by spike risk (fixed calibration) | +0.212 [+0.110, +0.328] | +4.746 [+4.138, +5.306] | -7.871 [-9.081, -6.792] |
| CPS: Mondrian by spike risk (sliding windows) | -0.290 [-0.325, -0.253] | -0.473 [-0.753, -0.218] | -4.719 [-5.341, -4.160] |
| CPS: PIT-recalibrated spike forecaster | -0.267 [-0.304, -0.227] | -1.612 [-2.077, -1.229] | -13.344 [-16.630, -10.640] |
| CPS: PIT-recalibrated, Mondrian by spike risk | +0.276 [+0.096, +0.493] | +6.276 [+4.789, +7.609] | -12.534 [-15.744, -9.902] |
