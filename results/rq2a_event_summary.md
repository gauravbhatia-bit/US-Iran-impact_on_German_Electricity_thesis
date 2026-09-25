# RQ2a Event-table Hormuz transmission

The primary exposure has three states -- open, restricted and severely
restricted -- set by a measurable rule: daily IMF PortWatch transit volume
against the 81-vessel-per-day pre-conflict median (open at or above 70% of
baseline, restricted 20-70%, severely restricted below 20%).  The sourced
event table supplies the dated transitions; the threshold rule decides which
state they carry.  Brent is intentionally excluded from the primary model
because the thesis mechanism is European gas pass-through to German
electricity; oil is discussed as context rather than treated as a direct
German power-price input.

## Data and identification

The 28 February conflict-start row is a restricted warning state and the 2
March closure row is severely restricted.  The 15 June agreement-to-open is
a diplomatic annotation only: transits on 15-17 June ran at 9-12% of
baseline, so the severely restricted state continues through 17 June.  The
18 June JMIC advisory marks a partial, contested corridor (20 transits,
24.7% of baseline), coded restricted rather than open.  Ship attacks on 6
July, the memorandum collapse on 8 July and the IRGC closure declaration on
12 July return the state to severely restricted through the data cutoff.
No verified open state exists anywhere after the conflict's start, so the
analysis reports re-closure-date sensitivity rather than reopening-date
sensitivity.

## Descriptive TTF levels
- open: n=1881, mean=57.60, median=38.54 EUR/MWh
- restricted: n=20, mean=41.19, median=42.04 EUR/MWh
- severely_restricted: n=166, mean=52.16, median=50.11 EUR/MWh

## Event windows and tests
- Seven-day event-window placebo p-value for the 2 March transition: 0.0562.
- Severe-state coefficient in a HAC daily TTF-return regression: +0.437% TTF return (HAC p=0.3181).
- Distributed lag from previous-day TTF returns to German daily price changes (lags 1-14): +2.707 EUR/MWh per 1% TTF return (p=0.0000).
- Re-closure sensitivity candidates evaluated (1-15 July): 15 dates.

The event study is exploratory: there is one main closure episode, event
dates are daily rather than exact hours, and state labels are not a causal
estimate by themselves.  Results should be read as evidence on whether a
source-verified state change coincided with a TTF movement and whether TTF
movements subsequently passed through to German electricity.

## Outputs

- `rq2a_hormuz_daily_exposure.csv`: daily state timeline and price returns.
- `rq2a_ttf_event_windows.csv`: pre/during/post event-window changes.
- `rq2a_ttf_event_study.csv`: HAC state regression coefficients.
- `rq2a_ttf_power_distributed_lag.csv`: TTF-to-power lags 1-14.
- `rq2a_ttf_placebos.csv` and `rq2a_reclosure_sensitivity.csv`.
- `rq2a_traffic_robustness.csv`: optional PortWatch check, not primary exposure.
- `rq2a_ttf_event_chart.png`: shaded three-state TTF chart.
