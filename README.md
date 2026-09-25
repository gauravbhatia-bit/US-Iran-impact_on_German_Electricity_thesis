# Iran-Energy Impact Analyser

Analysis of German day-ahead electricity prices during the 2026 Iran/Hormuz
conflict, with the 2022 European energy shock as a comparison.

Raw-data cutoff `2026-09-12`; the common processed window ends `2026-09-01`.

**[Results, charts and headline findings →](results/README.md)**

## Research questions

| RQ | Question |
|---|---|
| RQ1 | What price ranges follow under escalation, stalemate or de-escalation? |
| RQ2 | How large was the conflict's impact on German prices? |
| RQ2a | Through which channel did it transmit, and what role did Hormuz play? |
| RQ3 | Which model forecasts best during a shock? |
| RQ4 | Which feature groups contribute to forecast accuracy? |
| RQ5 | Does news coverage lead prices? |

## Setup

Python 3.11, packages pinned in `requirements.txt`.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproducing from this repository

`data/processed/master_features.csv` is committed, so the modelling stages rerun
without API keys. From the repository root:

```powershell
python run_pipeline.py --from-stage folds --through-stage scenario --end-date 2026-09-12
```

That rebuilds the CV folds and reruns the forecast arms, ablation, significance
tests, magnitude, transmission, Granger diagnostics and scenarios. The only
network access it needs is the Chronos model weights, which Hugging Face
downloads on first use (about 800 MB).

The enhanced RQ1, RQ2, RQ2a and RQ4 stages additionally read the bulk
Energy-Charts exports, which are too large to commit. Refetch them once, then
run those stages:

```powershell
python src/pipeline/01b_enhancement_collection.py --end-date 2026-09-12
python run_pipeline.py --from-stage rq1_enhanced --end-date 2026-09-12
```

To check the committed outputs are present and non-empty instead of recomputing
them, add `--verify-existing` to either command.

### Rebuilding from source data

Collection and preprocessing need inputs that are not redistributed here:

| Input | How to obtain |
|---|---|
| `data/raw/Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv` | Manual Investing.com export of Dutch TTF futures. Set `TTF_DAYFIRST=1` if its dates are DD/MM/YYYY. |
| Guardian article text | Set `GUARDIAN_API_KEY`; `01_data_collection.py` paginates within the developer-key rate model. The derived daily sentiment series is committed. |
| `data/raw/energy_charts_*.csv` | Fraunhofer ISE Energy-Charts, CC BY 4.0. Refetched by `01b_enhancement_collection.py`; excluded here only for size. |
| `data/raw/eua_daily.csv` | Committed. Rebuilt with `prepare_eua_daily.py --historical-zip <zip> --current-xlsx <xlsx>` from EEX exports. |

With those in place, a full run is:

```powershell
$env:GUARDIAN_API_KEY = "your-key"
python run_pipeline.py --end-date 2026-09-12
```

The runner stops at the first failed source, validation check, model fold or
scenario consistency check, and writes `results/run_manifest.json` recording the
stages, Python and package versions, script hashes and artifact hashes.

## Data provenance

Committed inputs and their sources are documented in
[`docs/Data_Sources_Justification_Memo.md`](docs/Data_Sources_Justification_Memo.md).
The hand-built event tables (`data/raw/event_table.csv`,
`data/raw/hormuz_event_table.csv`, `data/raw/hourly_event_timestamps.csv`,
`data/raw/shipping_status_evidence.csv`) carry a source URL and a verification
flag on every row. SMARD prices and generation are published by
Bundesnetzagentur under CC BY; Brent and EUR/USD come from FRED; the GPR index
comes from Caldara and Iacoviello; Hormuz transit counts come from IMF
PortWatch.

## Method safeguards

- All required sources fail closed: a partial collection produces no completion
  manifest and cannot enter preprocessing.
- SMARD hours are interpreted as Europe/Berlin delivery time, including 23- and
  25-hour daylight-saving days; incomplete internal days are rejected.
- Target and renewable observations are never forward-filled; market fills are
  capped and marked.
- Rolling statistics and all market, GPR, news, renewable and Dunkelflaute
  signals are lagged, so day *t* never contains day-*t* outcomes.
- Cross-validation is true one-day-ahead evaluation at weekly origins, with
  `SHOCK_2022` and `SHOCK_2026` reported separately.
- The scenario engine estimates gas-to-power pass-through in first differences,
  not in non-stationary price levels.
- Event analysis accepts only sourced events, removes market-outcome-selected
  dates, clusters overlapping windows, and applies plus-one placebo p-values
  with Holm correction.

The validation gate (`src/pipeline/validation_gate.py`) independently checks
hashes, required sources and fields, daily continuity, a non-null target, lag
alignment and shifted rolling volatility, and exits non-zero on any failure.

## Layout

```text
run_pipeline.py   pipeline runner
src/pipeline/     collection, preprocessing, validation, CV folds
src/modeling/     models, tests, magnitude, transmission, scenarios
data/raw/         source inputs and sourced event tables
data/processed/   intermediate datasets and descriptive artifacts
results/          final tables, summaries, charts, run manifests
docs/             final results, RQ2a/RQ4 specification, data sources
```

The dissertation report itself is not published in this repository.
