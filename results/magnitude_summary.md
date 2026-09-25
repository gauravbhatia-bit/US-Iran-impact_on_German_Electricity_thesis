# Magnitude Analysis

How much did the 2026 Iran-US conflict move German electricity prices?

## Counterfactual

- Mean deviation from the pre-conflict forecast: **+20.65 EUR/MWh**
- Cumulative excess: +3,842 EUR/MWh-days
- Days above the counterfactual: 80.1%
- Days outside the 90% interval: 1.1% (~10% expected by chance)

The benchmark is built with Prophet trained only on pre-conflict data and 
conditions on lagged realised renewable output. It is a model-based forecast 
deviation, not an identified causal effect. 
The hybrid model is deliberately not used: it depends on conflict features, 
which by construction do not exist in a no-conflict world.
A matched-length 2022 Ukraine-shock comparison has mean forecast deviation **+90.09 EUR/MWh** over 186 days.

## Event study

- **smard_mean**: mean |CAR| 50.05, largest 92.01
- **ttf_eur_mwh**: mean |CAR| 8.85, largest 18.05
- **DCOILBRENTEU**: mean |CAR| 11.61, largest 15.87

## Placebo significance

A single conflict provides no cross-sectional sample, so standard event-study 
t-tests do not apply. Each event's response is instead compared against the 
distribution of responses from 500 random pre-conflict dates.

- **smard_mean**: 0/4 event episodes significant at Holm-adjusted p<0.05
- **ttf_eur_mwh**: 0/4 event episodes significant at Holm-adjusted p<0.05
- **DCOILBRENTEU**: 3/4 event episodes significant at Holm-adjusted p<0.05

## Interpretation

Forecast accuracy and price impact are separate questions. An ablation result 
about predictive value does not imply that the 
conflict had no effect. Conversely, a small measured effect here would 
explain the ablation result: features cannot predict a movement that did not 
occur.

Cross-market comparison matters for the interpretation. If Brent shows a large 
response while German electricity does not, the finding is one of 
**transmission**: the shock hit global oil but did not propagate to German 
power, plausibly reflecting post-2022 LNG diversification.