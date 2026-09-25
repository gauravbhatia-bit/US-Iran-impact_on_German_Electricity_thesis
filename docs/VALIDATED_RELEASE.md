# Validated dissertation release: reporting source

This note is the reporting source for the validated dissertation release, dated
18 September 2026 and amended on 21 September 2026 for the Hormuz recode. It
supersedes all earlier numerical summaries produced during the recovery and
enhancement runs.

The run records are `../results/run_manifest.json` (baseline stages) and
`../results/enhanced_run_manifest.json` (enhanced RQ1/RQ2/RQ2a stages). Each
holds the Python and package versions, the stage list, and SHA-256 hashes of
the scripts and artifacts. They are separate because the stages were executed
in separate sessions, and must not be described as one monolithic online run.

## Final reporting conclusions

| Question | Final conclusion | Main evidence |
|---|---|---|
| RQ1 | Conditional weekly risk is highest under renewed restriction, but scenario bands overlap. | Week-13 medians: EUR 173.57/MWh renewed restriction, EUR 147.58/MWh status quo and EUR 90.11/MWh normalisation; nominal-90% realised-driver coverage 87.2%; CRPS 16.92. |
| RQ2 | The data do not establish a positive Germany-specific average causal effect. | Total model deviation +20.65 EUR/MWh; controlled-direct deviation -27.86 EUR/MWh; full-post synthetic-control gap -2.32 EUR/MWh and full-post placebo p=0.7355. |
| RQ2a | The state-to-TTF link is suggestive, while the TTF-to-power result is a pooled observational association, not Hormuz-specific causal proof. | 2 March TTF-window placebo p=0.0562; severe-state return coefficient is not significant; 1-14-day TTF-to-power cumulative association +2.707 EUR/MWh per 1% TTF return. |
| RQ3 | Chronos has the lowest observed 2026 error, but no statistically proven universal winner exists. | Chronos 2026 MAE EUR 21.89/MWh; the small 2026 comparison does not yield a corrected pairwise winner. |
| RQ4 | Fundamentals improve MAE; EUA is economically justified but its separate incremental gain is not statistically reliable. | All-fold MAE falls from 22.924 (A) to 21.792 (D); all 7-56 day block-bootstrap EUA intervals include zero. |
| RQ5 | GPR and article volume have timing-sensitive in-sample associations, but neither is an actionable leading signal. | Volume is bidirectional; each reported external lag uses D-2 data and true calendar-day windows; neither the GPR nor joint Guardian block has a Holm-significant out-of-sample gain. |

## Corrections incorporated in this release

1. **RQ2 source-resolution harmonisation.** Quarter-hour source observations
   are averaged to actual delivery hours before volatility, negative-price-share
   and high-tail calculations. Mean, peak and off-peak results are unchanged.
   After harmonisation, negative-price share remains Holm-significant (+2.16
   percentage points; Holm p=0.0052); volatility is borderline (Holm p=0.0575)
   and high-tail share is not significant (Holm p=0.3063).

2. **RQ4 canonical bootstrap alignment.** The 28-day sensitivity calculation
   now uses the same fixed seed as the canonical paired moving-block bootstrap.
   This removes an avoidable simulation-seed discrepancy while leaving the
   substantive inference unchanged.

3. **RQ5 availability and calendar lags.** Date-only external inputs are
   shifted two days before differencing and lag construction. Missing Guardian
   days are excluded from affected windows rather than compressed, zero-filled
   or interpolated. A reported model lag `L` therefore uses raw external
   information from `t-(L+2)`.

4. **RQ1 scenario transparency.** The final report gives the numerical TTF
   anchors, targets, reversion parameter and simulation design. Scenario labels
   are analyst-defined conditional assumptions, not estimates of political
   probability.

5. **Report and provenance clarity.** The report now uses the harmonised RQ2
   and D-2/full-calendar RQ5 numbers, avoids causal overstatement of RQ2a,
   states the RQ4 compound-specification caveat, and identifies the frozen
   release record instead of implying all results came from one run.

6. **RQ1 scenario-anchor caveat (applied to the report on 2026-09-21).** A
   review of the RQ2a Hormuz event table found that the "open"
   state it forward-fills from 15/18 June 2026 to the 1 September data cutoff
   does not survive a measurable traffic-volume test (0 of 79 "open"-coded
   days clear a 70%-of-baseline threshold against the 81-vessel-per-day
   pre-conflict median in `../data/raw/imf_portwatch_hormuz.csv`), and that a 6-12 July
   attack/collapse/closure-declaration sequence, independently verified
   against external reporting, was missing from every event table in this
   project. The scenario engine's stalemate anchor (`15_scenario_engine.py`,
   `recent_mean` = the most recent 30 observed TTF days) and its de-escalation
   framing implicitly treat the most recent data as a calmer, post-agreement
   period. Under the corrected Hormuz timeline that period is severely
   restricted, not de-escalated, and remained so through at least
   19 September 2026 per external tracking. The week-13 medians and coverage
   figures above are unaffected -- they are not derived from the Hormuz state
   label -- but the "normalisation"/de-escalation scenario should be read as a
   scenario that has not yet begun to occur, rather than as one already under
   way.

7. **Hormuz exposure recoded and adopted as primary (2026-09-21).** The RQ2a
   state is now set by a measurable rule -- IMF PortWatch transit volume against
   the 81-vessel-per-day pre-conflict median -- with the sourced event table
   supplying dated transitions rather than determining the state. Three
   independently sourced July events (6 July ship attacks, 8 July memorandum
   collapse, 12 July IRGC closure declaration) were added. State counts change
   from 1,960/2/105 days to 1,881/20/166, the severe-state HAC coefficient from
   +0.384 (p=0.5387) to +0.437 (p=0.3181), and reopening-date sensitivity is
   replaced by re-closure-date sensitivity. The RQ2 magnitude conclusions are
   unchanged by the correction (Brent significant in 3/4 episodes, German power
   and TTF in 0/4, before and after). The recoded inputs are now the primary
   `data/raw/hormuz_event_table.csv` and `data/raw/event_table.csv`, read by
   `21_rq2a_event_table_study.py`. The reasoning is recorded in
   `RQ2A_RQ4_IMPLEMENTATION.md`.

## Authoritative artifact paths

- `../results/rq2_resolution_harmonisation.json`
- `../results/rq4_eua_block_bootstrap.csv`
- `../results/rq4_eua_block_length_sensitivity.csv`
- `../results/granger_availability_metadata.json`
- `../results/granger_diagnostics_availability_metadata.json`

