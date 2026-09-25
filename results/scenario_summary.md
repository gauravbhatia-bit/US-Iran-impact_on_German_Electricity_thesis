# Scenario Projections (RQ1)

German day-ahead electricity prices, 90-day horizon, 4000 simulations per scenario.

## Method

Scenarios are conditional sensitivities propagated through the GAS CHANNEL rather than through news or geopolitical risk features. The gas-to-electricity link is re-estimated in first differences to avoid a spurious levels relationship.

- Pass-through: +1.1892 EUR/MWh per EUR/MWh of gas (se 0.3752, p = 0.001525)
- Model R2: 0.129, residual sd 36.84, n = 2065

Three uncertainty sources are propagated: parameter uncertainty in the pass-through, path uncertainty in the gas trajectory, and residual uncertainty in what gas does not explain.

## Scenario definitions

- **escalation**: gas to 77.52 EUR/MWh (current level (62.53) plus the observed closure premium (+14.99, measured as closure mean 48.41 minus the 60 days before closure 33.42))
- **stalemate**: gas to 62.53 EUR/MWh (mean of the most recent 30 observed days)
- **de-escalation**: gas to 33.42 EUR/MWh (immediate 60-day pre-closure mean (pre-closure 33.42, recent 62.53))

## Results

| Scenario | Median at day 90 | 90% interval |
|---|---|---|
| escalation | 231.93 | -308.76 to 762.44 |
| stalemate | 219.19 | -310.65 to 768.54 |
| de-escalation | 192.61 | -351.24 to 732.71 |

## Limitations

- Driver levels are anchored to observed values, so the model is never extrapolated into unseen gas price territory. The cost is that escalation is capped at the severity actually observed in 2026; a more severe or prolonged closure would push prices higher, and a single episode cannot say how much higher.
- Severity is not identified. One closure episode provides no basis for scaling to different closure durations or completeness.
- The projection is a conditional sensitivity, not a causal estimate or trained forecast. It composes two separately estimated links rather than learning the full chain, because one episode cannot support end-to-end training.
- Weather is set to its historical expected rate rather than a forecast. Actual outcomes will diverge with weather.