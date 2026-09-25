# Enhanced RQ2 analysis

## Estimands

- Total conflict-associated forecast deviation (TTF excluded as a post-treatment mediator): **+20.65 EUR/MWh**; 14-day block 95% interval +13.37 to +27.66.
- Controlled direct forecast deviation conditional on the realised lagged TTF path: **-27.86 EUR/MWh**; 14-day block 95% interval -33.33 to -21.42.
- These are model-based deviations. The synthetic-control section supplies a comparative design, but one conflict still limits causal identification.

## European synthetic control

- Mean-price post-treatment gap, adjusted for the last 180 pre-treatment days: **-2.32 EUR/MWh**.
- Matched 90-day mean gap used for time-placebo inference: **-4.61 EUR/MWh**.
- HAC p-value: 0.0547.
- 7-day block interval: -4.29 to -0.44; 14-day block interval: -4.62 to -0.00.
- Matched 90-day time-placebo p-value (120 pseudo-starts): 0.3967.
- Full-post-length time-placebo p-value (120 pseudo-starts): 0.7355; all pseudo-windows end before the 2026 treatment.
The placebo windows are spaced weekly but overlap; their effective number of independent windows is substantially smaller than the row count.

Donors were declared before fitting as France, Switzerland, Norway NO2, Sweden SE4, Denmark DK1 and DK2. They are comparatively less directly tied to Middle Eastern gas, but they are not assumed to be completely unaffected by a European energy shock.

## Distributional outcomes

- Within-day volatility, negative-price share and high-tail share are calculated from a canonical hourly series. From 1 October 2025, four 15-minute source observations are averaged within each actual local delivery hour before those metrics and the pre-conflict high-tail threshold are calculated. Mean, peak and off-peak remain direct interval means because they are invariant to complete hourly-to-quarter-hourly resampling. The UTC hour is retained to preserve both 02:00 DST hours. See `rq2_resolution_harmonisation.json`.

- mean: adjusted gap -2.321, HAC p=0.0547, Holm-adjusted p=0.2190
- peak: adjusted gap -0.644, HAC p=0.5278, Holm-adjusted p=0.5278
- offpeak: adjusted gap +1.520, HAC p=0.1276, Holm-adjusted p=0.3063
- volatility: adjusted gap +4.087, HAC p=0.0115, Holm-adjusted p=0.0575
- negative: adjusted gap +0.022, HAC p=0.0009, Holm-adjusted p=0.0052
- high_tail: adjusted gap +0.001, HAC p=0.1021, Holm-adjusted p=0.3063

## Hourly, timestamped events

- 2026-04-13 14:00:00+00:00: mean abnormal Germany-minus-control gap +8.46 EUR/MWh; 24-hour block interval +6.75 to +10.27; Holm-adjusted time-placebo p=0.4234.
- 2026-07-14 20:00:00+00:00: mean abnormal Germany-minus-control gap -5.96 EUR/MWh; 24-hour block interval -7.89 to -3.48; Holm-adjusted time-placebo p=0.4234.

The two timestamped events are Iranian-port blockade operations, not a blanket Strait closure. This distinction corrects the earlier binary-state interpretation.

## 2022 comparison

- 2022: mean deviation +90.09; +1.49 pre-event standard deviations; pre-event TTF mean 83.53.
- 2026: mean deviation +20.65; +0.83 pre-event standard deviations; pre-event TTF mean 33.29.

Chart: `rq2_synthetic_control.png`