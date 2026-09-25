# Enhanced RQ2a physical transmission analysis

Primary exposure is continuous IMF PortWatch vessel-traffic shortfall, not a binary closure date and not GPR.

## Physical measure

- Pre-conflict median: 81 vessel transits/day
- Maximum measured traffic shortfall: 100.0%
- PortWatch is AIS-derived; jamming, spoofing and dark vessels remain measurement risks.
- Joint Maritime Information Center via IMO independently recorded the Strait as open with moderate threat on 2026-06-18; this verifies status by that date, not the exact reopening instant.

## Distributed-lag chain

- Traffic shortfall → TTF, cumulative lags 0-7: +16.118 (SE 8.203, p=0.0494, Holm p=0.0989)
- Previous-day TTF → German power, cumulative lags 1-7: +0.982 (SE 0.713, p=0.1684, Holm p=0.1684)

The two coefficients are reported separately. Their product is not labelled a causal mediation effect because sequential ignorability is not credible in one conflict.

## Independent energy-flow validation

- EIA Hormuz LNG flow changed -92.4% and oil flow changed -77.3% in 2026-Q2 versus 2025-Q4.
- These quarterly observations validate direction and scale only; they are not copied into daily regressions.
- Daily war-risk insurance costs remain omitted because no auditable open series was available; a source-labelled import template is provided.

## Local projections

Responses for horizons 0-14 days are in `rq2a_local_projections.csv`; they allow delayed transmission rather than forcing a same-day relationship.
After Holm correction, 0/30 local-projection coefficients are significant.

## Gas-likely marginal hours

- Base previous-day gas-change coefficient: +0.027 (p=0.2705)
- Additional coefficient in high-gas/high-residual-load hours: +0.036 (p=0.6760)
- Total in gas-likely hours: +0.063
- Observations: 49,559 hours in 2,065 day clusters

## Sensitivity and GPR

- Continuous-shortfall thresholds from 50% to 90% were tested; 9/9 price-level specifications remain significant after Holm correction. These level regressions are vulnerable to period confounding.
- Joint comparator: traffic shortfall p=0.0000; GPR p=0.0175; correlation=0.376.
- GPR is retained only as a broad-risk comparator, never interpreted as physical closure.

The earlier reopening-date sensitivity remains available in `hormuz_transmission.csv`; the continuous specification no longer requires one reopening date to define exposure.

Chart: `rq2a_local_projections.png`