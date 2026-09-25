# Data Sources: Justification & Provenance Memo

## Purpose
For each dataset in the pipeline: what it is, why it's needed, what it feeds into,
and how its authenticity/provenance can be checked (i.e. why it is NOT synthetic data).

---

## 1. SMARD Day-Ahead Electricity Prices (Germany/Luxembourg)

**What it is:** Hourly wholesale electricity prices, published by Bundesnetzagentur
(Germany's Federal Network Agency) via their public SMARD platform.

**Why we need it:** This is one of our three **target variables** - the thing we're
actually forecasting. German electricity prices are the practical, policy-relevant
endpoint of the whole causal chain: conflict → gas supply shock → gas-fired
generation cost → electricity price. Without this, the thesis has no direct link to
"impact on German energy markets" as stated in the title.

**Provenance / why it's not synthetic:** SMARD is a primary regulatory data source -
the underlying data legally must be reported to ENTSO-E under EU Regulation
543/2013, and SMARD republishes it under a Creative Commons license with the
required attribution "Bundesnetzagentur | SMARD.de". This is about as close to
ground-truth as energy data gets. To verify authenticity yourself: cross-check a
handful of your downloaded prices against SMARD's own visual charts on smard.de for
the same dates - they should match exactly since it's the same underlying database.

**Improvement idea:** Also pull *intraday* prices (a separate SMARD filter) as a
robustness check - day-ahead is your primary target, but intraday reaction speed
around the conflict's exact strike dates could be a nice secondary finding for the
event study.

---

## 2. Dutch TTF Natural Gas Futures

**What it is:** Daily front-month futures prices for TTF, the European benchmark
gas hub, sourced from Investing.com.

**Why we need it:** Second target variable, and arguably the most directly exposed
to the conflict - Qatar's LNG force majeure and Strait of Hormuz disruption hit gas
supply first and hardest. TTF is also the price German electricity is most sensitive
to on the margin (gas-fired plants set price in many hours).

**Provenance:** Investing.com aggregates from ICE (Intercontinental Exchange), the
actual exchange TTF futures trade on. It's a secondary aggregator rather than the
exchange itself, which is worth naming honestly in your methodology chapter - for
a stronger provenance chain, ICE itself or Refinitiv/LSEG publish the same data
if your university library gives you access to a financial database (check this -
many universities have a Refinitiv or Bloomberg terminal in the library).

**Improvement idea:** If you can get library access to ICE/Refinitiv directly, switch
to that as your primary source and keep Investing.com only as a cross-check -
strengthens the "not synthetic, not scraped from a blog" argument in your data
chapter.

---

## 3. Brent Crude Oil

**What it is:** Daily Brent spot price in USD/barrel.

**Why we need it:** Third target/control variable. Brent is the most liquid, most
closely watched global oil benchmark, and it's what the CRS report and most news
coverage of the conflict actually quote (the "$71 to $100" figures). Including it
lets you directly validate your event-study dates against a widely reported,
independently verifiable price series.

**Provenance:** Recommend pulling `DCOILBRENTEU` from FRED (Federal Reserve
Economic Data) rather than a futures contract like `BZ=F` from Yahoo Finance -
FRED's series is sourced directly from the U.S. EIA, a federal statistical agency,
and is spot price rather than a rolling futures contract (avoids roll-over
artifacts that could look like fake price jumps in your event study).

---

## 4. EUR/USD Exchange Rate

**What it is:** Daily EUR/USD reference rate.

**Why we need it:** A control variable, not a target. Brent is priced in USD, TTF
and SMARD prices in EUR - without controlling for FX movement, a chunk of what
looks like "conflict impact" on European prices could actually just be dollar
strength/weakness. This is a standard control in cross-currency commodity research,
not something specific to your topic - but omitting it is a common and easily
critiqued gap in weaker versions of this kind of thesis.

**Provenance:** FRED's `DEXUSEU`, sourced from the Federal Reserve's own H.10
release - a primary central-bank source.

---

## 5. Geopolitical Risk (GPR) Index: Caldara & Iacoviello

**What it is:** A published academic index quantifying geopolitical tension by
counting war/terrorism/military-buildup keywords in major newspapers.

**Why we need it:** This is the **benchmark control feature** the literature
already uses (Chowdhury et al. 2025 use it directly). Including it does two things:
(1) it's a sanity check - if your custom sentiment score doesn't correlate at all
with an established, peer-reviewed index, that's a red flag worth investigating,
not ignoring; (2) it lets you argue your daily custom sentiment adds value *beyond*
what the existing monthly GPR index already captures, which is one of your stated
contributions (Gap 2 in your lit review).

**Provenance:** Published directly by the original authors (Dario Caldara, Federal
Reserve Board; Matteo Iacoviello, Federal Reserve Board) on Iacoviello's own
academic website, updated regularly. About as authoritative as a research dataset
gets - it's the index the whole geopolitical-risk-in-finance literature cites.

---

## 6. EU Sanctions Timeline (binary event flags)

**What it is:** A small, manually compiled dataset of sanctions-related
announcement dates from the EU Official Journal.

**Why we need it:** Sanctions are discrete policy events, not something you can
capture from continuous price or sentiment data alone - you need explicit dates
to test whether announcements cause measurable price reactions (this feeds directly
into your event study).

**Provenance:** The EU Official Journal is the EU's legally authoritative
publication record - every sanctions regulation is published there with an exact
date, making this the most verifiable dataset in the whole pipeline, if the most
manual to build.

---

## 7. News Sentiment (headlines → VADER score)

**What it is:** Daily aggregated sentiment computed from news headlines, using
VADER plus your custom energy-geopolitics lexicon.

**Why we need it:** This is your **novel exogenous feature** - the thing that
lets the model react to conflict developments faster than price data alone would
reveal them, and it's central to RQ4 and H3.

**Provenance:** Real news headlines from an aggregator (GDELT or NewsAPI, see
options below) - VADER computes a score *from* real text, it doesn't generate any
text itself, so as long as your headline source is real, the sentiment scores
are a genuine derived measurement, not synthetic data. Worth explicitly stating
this distinction in your methodology chapter: **the input is real, the sentiment
score is a computed feature, not fabricated data.**

---

# Overall data authenticity strategy for your advisor meeting

State plainly: every dataset here is either (a) published by a regulator/central
bank/exchange with legal reporting obligations, (b) a peer-reviewed academic index
from named, identifiable authors, or (c) a manually-sourced record from an official
government publication. Nothing is simulated, bootstrapped, or synthetically generated. The
one place synthetic data legitimately enters the thesis is later, in the Monte
Carlo scenario simulation (Phase 6) - and that's clearly framed as a *forecasting
tool built from real model residuals*, not a dataset presented as historical fact.
That distinction is worth stating explicitly and early, since it heads off any
confusion about what's observed vs. simulated.

---

# News Sentiment Sourcing: Options Compared

| Option | Historical depth | Cost | Effort | Verdict |
|---|---|---|---|---|
| **GDELT (DOC 2.0 API)** | Full history via BigQuery; REST API only ~3 months | Free | Medium-high (BigQuery setup) | Best long-term option, more setup work |
| **NewsAPI** | Free tier: last 30 days only | Free tier / paid for history | Low | Useless for your Jan 2023 start date on the free tier - would need a paid plan for backfill |
| **X/Twitter API** | Full history technically possible | Expensive - API access is now a paid tier, pricing has increased significantly since 2023 | High | Cost alone likely rules this out for a student thesis budget |
| **Truth Social (Trump specifically)** | No official API at all | N/A | Very high (scraping only) | Scraping violates ToS - your own proposal already flagged this correctly |
| **Reddit (r/geopolitics, r/energy)** | Full history via Pushshift-style archives | Free | Medium | Interesting secondary/exploratory source, not a primary one |
| **Licensed news APIs (Reuters/Bloomberg via university library)** | Full history, high quality | Free if your library has it | Low if access exists | Worth checking - ask your library before building anything |

## On using Trump's tweets specifically

Your own proposal already reasoned through this correctly, and I'd stand by that
reasoning rather than reverse it:

- **Frequency problem:** a handful of relevant posts per day, sometimes zero,
  isn't enough to build a genuinely daily time series - you'd have long gaps
  needing interpolation, and interpolated sentiment isn't real signal.
- **Access problem:** Truth Social has no public API. Scraping it would violate
  its terms of service, which is a real methodological and potentially ethical
  risk to flag to your supervisor before attempting it - not something to route
  around quietly.
- **Signal problem:** a single political figure's posts are a noisy, biased proxy
  for "market-relevant sentiment" - broad news aggregation is a more defensible
  measure of what's actually moving markets, and it's what the literature you're
  citing (Dragomirescu-Gaina et al.) treats as the interesting *finding* (tweets
  move markets briefly) rather than the *recommended data source*.

**Where a Trump/political-leader signal could still add value:** not as a
continuous sentiment series, but as a small, manually curated table of specific,
high-impact statements tied to your event-study dates - e.g. "Trump gives 48-hour
ultimatum, March 23" as a discrete event flag, similar to your sanctions timeline.
That's tractable, defensible, and avoids all of the above problems, while still
letting you test H3 (leading-signal hypothesis) on a small, high-confidence sample
rather than a noisy scraped one.

**My recommendation:** GDELT as primary (free, broad, defensible), with a small
manually-curated table of named political-leader statements as a secondary event
feature - not X/Twitter API, not Truth Social scraping.
