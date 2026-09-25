# RQ4 compact forecast-origin feature model

The final model tests only economically motivated predictors: price/calendar
baseline, forecast residual load, TTF, EUA and coal where auditable files are
available, expected gas marginal cost, Hormuz traffic shortfall and the two
pre-specified interactions.  Date-only market inputs enter with a conservative
two-day lag; the
residual-load forecast is a seven-day seasonal-naive proxy, not realised same-day
load.  This prevents the old same-day leakage problem.

## Feature availability
- EUA source: eua_daily.csv (1928 lagged rows).
- Coal source: not supplied (0 lagged rows).
- Missing EUA/coal prices are left missing and excluded from the corresponding fit; bounded carry-forward is accompanied by staleness flags; no synthetic prices are created.

## Nested out-of-sample comparison

| Set | Features | Mean MAE (all folds) | Mean MAE (shock folds) |
|---|---:|---:|---:|
| A_price_calendar | 11 | 22.924 | 32.175 |
| B_plus_fundamentals | 16 | 21.841 | 29.506 |
| C_plus_hormuz | 17 | 21.807 | 29.466 |
| D_plus_interactions | 19 | 21.792 | 29.309 |

Lower MAE is better.  A gain from C to D means the Hormuz exposure and its
interactions add predictive information beyond the baseline and fuel/system
variables.  It is not evidence that the feature is causal; use the RQ2a event
study for the transmission claim.

## Permutation importance

The permutation output is an out-of-sample diagnostic for the final compact model
on shock folds.  Positive values indicate that shuffling a feature worsened
negative-MAE score; rankings are descriptive and should be accompanied by the
nested accuracy comparison.

Outputs: `rq4_compact_results.csv`, `rq4_compact_daily_errors.csv`,
`rq4_compact_feature_importance.csv` and `rq4_compact_diagnostics.json`.
