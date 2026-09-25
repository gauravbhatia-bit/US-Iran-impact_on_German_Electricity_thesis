# Enhanced RQ1 probabilistic scenarios

The 90-day structural exercise now forecasts **weekly-average** German day-ahead prices. It is separate from the 14-day operational forecast.

## Model

- Conditional mean-reverting weekly ARX coefficient: 0.262
- Drivers: TTF gas, renewable generation, load, net cross-border flow and annual seasonality
- Uncertainty: joint coefficient draws, stochastic gas paths, seasonally matched system paths, and two-week moving blocks of historical residuals
- SMARD/Energy-Charts consistency: correlation 1.00000, MAE 0.000 EUR/MWh

## Scenario specification

- Origin TTF: 67.08 EUR/MWh; recent four-week mean: 61.95.
- Renewed restriction target: 82.79 EUR/MWh (origin plus estimated closure premium 15.71).
- Status quo target: 67.08 EUR/MWh (origin level).
- Normalisation target: 32.76 EUR/MWh (pre-conflict benchmark 32.76).

## Week 13 results

- **renewed-restriction**: median 173.57 EUR/MWh (90% interval 110.89 to 221.55)
- **status-quo**: median 147.58 EUR/MWh (90% interval 85.53 to 194.93)
- **normalisation**: median 90.11 EUR/MWh (90% interval 24.67 to 140.33)

Separation: 98.1% of renewed-restriction draws exceed the normalisation median.

## Historical pseudo-scenario validation

- Pseudo-origins: 12
- Weekly forecasts evaluated: 156
- Empirical 90% coverage: 87.2%
- Mean CRPS: 16.92
- Median 90% interval width: 92.68 EUR/MWh

The backtest conditions on realised gas and system paths. It evaluates calibration of the price-response distribution, not the probability assigned to a political scenario.

Generated-driver sensitivity: 90% coverage 97.4%, mean CRPS 19.19. This generates gas and seasonally matched system paths from the training data rather than using realised future drivers.

## Separate short-term product

`short_term_14d_forecast.csv` contains a price-only 1-14 day forecast. It must not be presented as the 90-day conflict scenario.

Chart: `scenario_weekly_fan_chart.png`