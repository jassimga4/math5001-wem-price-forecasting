# One-page summary: early warning for electricity price spikes in WA

**Bottom line.** A machine-learning forecast gives a calibrated range of what the WA wholesale electricity price will do over the next 5 minutes. It flags most price spikes as they start: 89% of spike onsets in a 4.5-month blind test. A flexible industrial load or a battery can use that warning to act before it pays the spike price. On illustrative assumptions, the business case breaks even at about **26–31 MW of flexible load**.

### The problem
- **How prices behave:** in the Wholesale Electricity Market (WEM), the price is set every 5 minutes. Most of the time it sits between about $97 and $133/MWh. In about 8% of intervals it spikes above $170/MWh, and in the test period it reached $984/MWh.
- **Who it costs:** those spikes drive a large share of the cost and risk for anyone buying at spot prices, including mine sites, processing plants, water utilities, retailers and battery operators.
- **What decisions need:** a single price guess isn't enough. Operators need to know how high the price **could** go, and how likely that is.

### What was built
- **Forecasts:** a forecast of the full price range (from the 1st to the 99th percentile) for every 5-minute interval, made 5 minutes ahead, using only public AEMO data that is available at that moment.
- **Benchmark:** it was compared against a standard industry method, Quantile Regression Averaging (QRA), on exactly the same test period of 1 Apr – 18 Aug 2026. That period was never used to build or tune either model.

### Results (blind test, 40,127 five-minute intervals)

| | Standard method (QRA) | New model (LightGBM) |
| --- | ---: | ---: |
| Overall forecast error (average pinball loss, lower is better) | 1.82 | **1.69** (−7%) |
| Error at the start of a spike | 17.4 | **13.2** (−24%) |
| Spike prices inside the model's 90% range | 68% | **94%** |
| Spike starts flagged (95th-percentile forecast above $170) | 75% | **89%** |

- **Where the value comes from:** the gain is in warning about spikes, not in day-to-day price accuracy. For an ordinary interval, the best guess improves only about 2%.
- **The trade-off:** the new model raises more false alarms. 58% of its alerts fall in a spike interval, against 66% for QRA. That is why the alert threshold has to be set against the real cost of acting.

### What it's worth (illustrative)

These figures rest on assumptions: a 10 MW load that can pause for 5 minutes, lost production valued at $150/MWh, and a $150k/yr running cost. Real test-period prices were used.

- **Value of a warning:** compared with simply reacting after a spike appears, the forecast-driven alert adds about **$4,900–5,800 per MW of flexible load per year**. Perfect foresight would add $9,100.
- **Break-even:** that pays back from roughly **26–31 MW**. A small 10 MW site would not cover the running cost.
- **Formula:** value = Σ alerted intervals × MW × 5/60 h × (price − value of lost production), annualised. The full workings are in the README, §5.

### Uses
- **Battery operators:** keep charge in reserve when a spike is likely.
- **Large loads:** automatically shed flexible load for an interval.
- **Retailers:** size their spot-price risk and hedges. This needs longer horizons, which is the next step.

### Next steps
1. Tune alert thresholds to each site's real cost of curtailing.
2. Extend the horizon to 30 minutes and day-ahead using pre-dispatch and weather forecasts.
3. Test across more seasons.
4. Pilot with one battery or flexible load.

*Source: Curtin MATH5001 project, repository `math5001-wem-price-forecasting`. All figures come from `reports/forecast/lgbm_vs_qra_summary.csv` and `roi_illustration.csv`.*
