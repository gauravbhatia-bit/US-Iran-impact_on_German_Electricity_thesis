# Granger Causality (RQ5)

Does past news coverage help predict German electricity prices beyond what past prices already provide?

## Method

F-test on nested models, lags 1 to 7, Holm-Bonferroni adjusted within each variable. All series first-differenced for stationarity (ADF results reported in the script output).

Every date-only external series (Guardian variables, GPR and TTF) is shifted by two calendar days before differencing and Granger-lag construction because the source exports do not establish a pre-auction publication time. Thus a reported Granger lag *L* uses raw external information from *t-(L+2)*. The realised electricity target remains unshifted and enters only through historical target lags. This is a retrospective predictive-precedence diagnostic, not an operational same-day forecasting result.

Lags are built on a complete daily calendar. Rows with any missing required target or external lag are excluded, so a one-day lag always means one calendar day; missing Guardian dates are neither compressed nor replaced with invented values.

TTF gas is included as a diagnostic comparator. Its relationship is not assumed from an earlier levels model.

## Results

| Variable | Period | Significant lags |
|---|---|---|
| Geopolitical risk index | CONFLICT PERIOD ONLY | [6] |
| Geopolitical risk index | NEWS-AVAILABLE WINDOW | [3, 4, 5, 6, 7] |
| News sentiment (tone) | CONFLICT PERIOD ONLY | none |
| News sentiment (tone) | NEWS-AVAILABLE WINDOW | none |
| News volume (coverage intensity) | CONFLICT PERIOD ONLY | [3] |
| News volume (coverage intensity) | NEWS-AVAILABLE WINDOW | [3, 6, 7] |
| TTF gas price  [DIAGNOSTIC COMPARATOR] | CONFLICT PERIOD ONLY | none |
| TTF gas price  [DIAGNOSTIC COMPARATOR] | NEWS-AVAILABLE WINDOW | none |

## Interpretation

- Diagnostic comparator (gas): 0 significant combinations
- News coverage: 4 significant combinations
- Geopolitical risk: 6 significant combinations

Granger causality tests predictive precedence, not causation. A significant result means one series helps forecast another; it does not establish a causal mechanism.

## Relation to prior work

The author's M508 project tested a related relationship over 2016-2020 using All the News 2.0 and FinBERT, on price volatility rather than price level, and found no Granger causality at lags 1 to 5 (p = 0.43 to 0.86). That null contrasts with the current article-volume result. The difference may reflect the period, corpus, or target, and the current bidirectional diagnostic prevents treating it as a clean leading signal. It must be cited explicitly as the author's own prior work rather than presented as an independent replication.