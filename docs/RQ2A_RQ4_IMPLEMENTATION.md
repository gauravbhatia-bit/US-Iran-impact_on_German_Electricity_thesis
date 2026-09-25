# Implemented RQ2a and RQ4 specification

## RQ2a : source-verified Hormuz event table

The primary exposure is now `data/raw/hormuz_event_table.csv`.  It contains
three explicitly labelled states:

| State | Meaning in this study |
|---|---|
| `open` | Source says the Strait is open or an agreement-to-open is recorded |
| `restricted` | Conflict-start warning state; not presented as proof of a physical closure |
| `severely_restricted` | Source-labelled restriction/closure episode |

The event table keeps source URLs, scope, verification flags and the exact-time
flag.  Iranian-port blockades are retained as source-verified sensitivity rows
but are excluded from the general-Strait state.  Because no source verifies an
exact reopening hour, the analysis is daily and reports reopening sensitivity.

The primary channel is **Hormuz state -> TTF -> German electricity**.  Brent is
not included in the primary RQ2a model; it remains contextual because Germany's
short-run power price is more directly linked to the European gas marginal-cost
channel.  PortWatch vessel shortfall is retained only as `rq2a_traffic_robustness.csv`.

Main outputs:

- `results/rq2a_event_summary.md`
- `results/rq2a_ttf_event_chart.png`
- `results/rq2a_ttf_event_windows.csv`
- `results/rq2a_ttf_event_study.csv`
- `results/rq2a_ttf_power_distributed_lag.csv`
- `results/rq2a_ttf_placebos.csv`
- `results/rq2a_reopening_sensitivity.csv`

The sourced 2 March transition is followed by a seven-day TTF increase of
about **EUR 20.17/MWh**.  The HAC daily state coefficient is positive but not
statistically significant (p about 0.54), while the cumulative lag-1-to-14
TTF-to-power response is positive (p about 0.00001).  The event result is
therefore exploratory, not a confirmed causal closure effect.

## RQ4 : compact forecast-origin model

`src/modeling/22_rq4_compact_features.py` evaluates nested sets on the same 244
rolling one-day-ahead folds:

1. Price lags and calendar baseline.
2. Forecast residual load (seven-day seasonal-naive forecast), TTF, EUA, coal
   and expected gas marginal cost.
3. Hormuz traffic shortfall as an additional predictor.
4. `TTF x high residual load` and `traffic shortfall x TTF` interactions.

All fuel/market inputs enter with a one-day lag.  Realised same-day residual
load is never used as a predictor.  In the latest clean rerun, mean MAE is
22.409 for the final interaction set versus 22.924 for the price/calendar
baseline; shock-fold MAE is 31.793 versus 32.175.  The fundamentals-only set is
best across all folds (22.500), while the interaction set is best on the full
sample and remains effectively tied on shock folds.  These are predictive
gains, not causal estimates.

The EUA input `data/raw/eua_daily.csv` is built from the supplied EEX auction
workbooks by `src/pipeline/prepare_eua_daily.py`.  It keeps successful T3PA
general-allowance auctions, excludes EAA3 aviation allowances, and uses a
volume-weighted price when needed.  The resulting 1,234 observations cover
2021-01-29 to 2026-09-16, with 1,929 lagged rows available on the project
calendar after the model's bounded forward-fill.  The expected gas marginal
cost converts TTF from EUR/MWh thermal gas to EUR/MWh electricity before adding
the EUA emissions cost.

Daily coal prices remain unavailable.  They are omitted rather than replaced
with a non-comparable or low-frequency proxy.  The templates
`data/templates/eua_daily_template.csv` and `data/templates/coal_daily_template.csv`
document the input schemas for future updates.

Main outputs:

- `results/rq4_compact_summary.md`
- `results/rq4_compact_results.csv`
- `results/rq4_compact_daily_errors.csv`
- `results/rq4_compact_feature_importance.csv`
- `results/rq4_compact_diagnostics.json`

## EUA contribution: block-length sensitivity and feature importance

The paired bootstrap compares the absolute errors from otherwise identical
models with and without EUA.  A positive `observed_delta_mae` means that adding
EUA lowers the forecast error.  Because the validation origins are weekly, the
moving blocks are defined in consecutive weekly origins rather than by a
calendar-daily contiguity rule.  The sensitivity run uses 1, 2, 4 and 8
origins (approximately 1, 2, 4 and 8 weeks), with 5,000 replicates per cell.

At the four-origin (approximately 28-day) setting, the paired improvement is
0.576 EUR/MWh for the fundamentals set (B) and 0.781 EUR/MWh for the final
interaction set (D) across all 244 folds.  The 95% bootstrap intervals are
[-0.138, 1.314] for B and [0.092, 1.553] for D; the two-sided p-values are
0.112 and 0.023, respectively.  On the 79 shock folds the observed gains are
1.129 (B) and 1.360 (D) EUR/MWh, but both intervals include zero (p=0.185 and
0.128), so the crisis-only evidence is directionally positive but imprecise.

The interaction-set (D) all-fold interval excludes zero at every tested block
length: [0.108, 1.473] (1 origin), [0.141, 1.484] (2), [0.092, 1.557] (4), and
[0.041, 1.620] (8).  The fundamentals-set (B) interval includes zero at all
four lengths.  This supports a robust all-period predictive contribution of
EUA in the final compact model, while avoiding an overclaim about a separate
crisis-only effect.

The shock-fold pooled out-of-sample permutation diagnostic ranks EUA's lagged
price (`eua_eur_tco2_lag1`) at about **0.89 EUR/MWh** of MAE deterioration when
shuffled (standard deviation about 0.42), behind the price lag, expected gas
marginal cost and TTF.  In plain terms, EUA provides a measurable but smaller
incremental signal after the model already knows recent electricity prices and
gas costs.  This ranking is predictive, not causal.  EUA is also included in
the expected gas marginal-cost calculation, so collinearity means the
permutation value should not be interpreted as the full economic importance of
carbon pricing.  The corrected pooling evaluates each feature on the shock
folds where it is available; the PortWatch traffic terms therefore have only
27 valid shock folds because that series starts in 2025.

The per-fold warning cleanup explicitly drops predictors with no historical
observations at a forecast origin before median imputation (for example, early
Hormuz traffic lags).  This removes the all-missing-column warnings without
inventing data or changing the intended feature sets.  The compact model was
rerun after this change; the reported MAE values above are the current outputs.

Sensitivity output: `results/rq4_eua_block_length_sensitivity.csv`.
The corrected default bootstrap output is `results/rq4_eua_block_bootstrap.csv`,
with paired fold losses in `results/rq4_eua_block_bootstrap_daily.csv`.

The new stages are registered in `run_pipeline.py` as `rq2a_event`,
`rq4_compact`, `rq4_eua_bootstrap` and `rq4_eua_sensitivity`; the default end
stage is now `rq4_eua_sensitivity`.

## Hormuz recode (adopted as primary on 2026-09-21)

A review of this table's "open" state found that it forward-fills from the
15/18 June 2026 rows to the 1 September data cutoff (79 days) on a diplomatic
agreement and a JMIC advisory headline, while IMF PortWatch traffic already in
this repo shows 0 of those 79 days clear a measurable 70%-of-baseline open
threshold, and a 6-12 July attack / MOU-collapse / IRGC-closure-declaration
sequence -- independently verified against external reporting (Al Jazeera,
Tasnim News, Wikipedia's sourced ship-attack list) -- was missing from every
event table in this project.

The corrected tables are now the **primary** inputs.
`data/raw/hormuz_event_table.csv` and `data/raw/event_table.csv` hold the
recoded content, with the added rows carrying their own `source_url`, and the
primary scripts `21_rq2a_event_table_study.py` and `12_magnitude_analysis.py`
regenerate the `rq2a_*` and `magnitude_*` artifacts from them. Reopening-date
sensitivity (`results/rq2a_reopening_sensitivity.csv`) is replaced by
re-closure-date sensitivity (`results/rq2a_reclosure_sensitivity.csv`).

Environment note: reproducing `12_magnitude_analysis.py` requires the pinned
`prophet==1.4.0` from `requirements.txt`. An earlier 1.1.5 install in this
environment failed to initialise its Stan backend; with the pinned version the
counterfactual reproduces exactly (+20.65 EUR/MWh, and +90.09 for the matched
2022 window), which also confirms the earlier figures were reproducible.

What the correction changed, established by running both codings side by side
before the recode was adopted:

| Analysis | Headline change |
|---|---|
| RQ2a event-table study | No verified open day remains 15 Jun-1 Sep; severely-restricted grows 105->166 days; severe-state HAC coefficient +0.384% (p=0.54) -> +0.437% (p=0.32), still not significant. |
| RQ2 magnitude / CAR | Adds the missing 6/8/12 July events (merged with the existing 13 July row into one CAR episode). Core finding **unchanged**: Brent significant in 3/4 episodes, German power and TTF in 0/4, before and after. |
| Legacy binary-state transmission script (not used for final reporting) | Pass-through during closure moves from not significant (p=0.0857, n=105) to significant (p=0.0417, n=166), and its joint Hormuz-vs-GPR test flips: GPR stays significant (p=0.0008) while Hormuz does not (p=0.1774), i.e. broad geopolitical risk rather than a chokepoint-specific effect. That script's framing is not cited in the dissertation, which reports the event-table specification instead. If it is ever reintroduced, the corrected reading above is the supported one. |

See also `VALIDATED_RELEASE.md`'s "Corrections incorporated" item 6 for the
RQ1 scenario-anchor caveat this same finding implies (the scenario engine's
"stalemate"/recent-30-day anchor overlaps a period now understood to be
severely restricted, not de-escalated).
