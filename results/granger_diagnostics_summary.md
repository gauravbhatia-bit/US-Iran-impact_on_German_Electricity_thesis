# Granger Diagnostics

Stress-tests of the observed news-volume Granger association before it is interpreted as an independent leading signal.

All date-only external series are shifted by two calendar days before the diagnostic calculations. A reported Granger lag *L* therefore uses raw external information from *t-(L+2)*; realised electricity price history remains in its normal lagged form. The checks are retrospective predictive-precedence diagnostics, not operational same-day forecasts.

Every Granger design is built on a complete daily calendar. A row is retained only when its target and all required calendar-day lags are observed; missing Guardian dates are not compressed, zero-filled or interpolated.

| Check | Question | Result |
|---|---|---|
| A | Does causality run both ways? | BIDIRECTIONAL |
| B | What is the full-window TTF relationship in stationary models? | cointegrated; ECM supports long-run adjustment |
| C | Does it survive controlling for GPR, Brent and gas? | survives controls |
| D | Is it robust to transformation and outliers? | robust |

## Note on interpretation

Granger causality tests predictive precedence, not causation. A significant result means one series helps forecast another; it does not establish a causal mechanism.