import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "smard_mean"
SECONDARY = ["ttf_eur_mwh", "DCOILBRENTEU"]   # for cross-market comparison

# Conflict start: first event in the timeline (US/Israel strikes).
CONFLICT_START = pd.Timestamp("2026-02-28")

# Event-study windows (trading-day counts, here calendar days since the
# series is daily and continuous).
ESTIMATION_WINDOW = 60    # days before the event used to learn "normal"
ESTIMATION_GAP = 5        # days skipped immediately before the event, so the
                          # estimation window is not contaminated by pre-event drift
EVENT_WINDOW = (-1, 5)    # from 1 day before to 5 days after

N_PLACEBO = 500
RANDOM_SEED = 42


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "master_features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ev_path = RAW_DIR / "event_table.csv"
    if not ev_path.exists():
        raise FileNotFoundError(f"{ev_path} not found")
    ev = pd.read_csv(ev_path, parse_dates=["date"])
    required = {"date", "event", "category", "verified", "source_url"}
    if missing := required - set(ev.columns):
        raise ValueError(f"Event table lacks provenance columns: {sorted(missing)}")
    truth = ev["verified"].astype(str).str.lower().map({"true": True, "false": False})
    if truth.isna().any() or not truth.all():
        raise ValueError("Every included event must be explicitly verified")
    if ev["source_url"].isna().any() or not ev["source_url"].str.startswith("http").all():
        raise ValueError("Every included event must have a traceable source URL")
    if ev["category"].eq("market_peak").any():
        raise ValueError("Outcome-selected market peaks cannot be used as events")
    return df, ev


def cluster_overlapping_events(events):
    ordered = events.sort_values("date").reset_index(drop=True)
    clusters = []
    for _, row in ordered.iterrows():
        if not clusters or row["date"] > clusters[-1]["last_date"] + pd.Timedelta(days=6):
            clusters.append({"date": row["date"], "last_date": row["date"],
                             "event": row["event"], "category": row["category"]})
        else:
            clusters[-1]["last_date"] = row["date"]
            clusters[-1]["event"] += " | " + row["event"]
            clusters[-1]["category"] += "+" + row["category"]
    return pd.DataFrame(clusters).drop(columns="last_date")


# ---------------------------------------------------------------- PART A ----
def counterfactual(df, conflict_start=CONFLICT_START, horizon=None, episode="2026 Iran"):
    from prophet import Prophet

    pre = df[df["date"] < conflict_start].copy()
    post = df[df["date"] >= conflict_start].copy()
    if horizon is not None:
        post = post.head(horizon)
    if post.empty:
        raise ValueError("No post-conflict data found.")

    print(f"  training on {len(pre)} pre-conflict days "
          f"({pre['date'].min().date()} to {pre['date'].max().date()})")
    print(f"  projecting {len(post)} conflict-period days")

    regressors = [c for c in ["renewable_output_mwh_lag1",
                               "dunkelflaute_flag_lag1", "is_holiday"]
                  if c in df.columns]
    if regressors:
        pre = pre.dropna(subset=regressors)
        post = post.dropna(subset=regressors)
    # The matched 2022 pre-event window contains less than two full years;
    # let Prophet disable the otherwise under-identified annual term there.
    m = Prophet(yearly_seasonality="auto", weekly_seasonality=True,
                daily_seasonality=False, changepoint_prior_scale=0.05,
                interval_width=0.90)
    for col in regressors:
        m.add_regressor(col)
    fit_cols = ["ds", "y"] + regressors
    m.fit(pre.rename(columns={"date": "ds", TARGET: "y"})[fit_cols])

    # Prophet draws its interval by Monte Carlo. Seeding keeps cf_lower/cf_upper
    # reproducible; the point forecast is deterministic either way.
    np.random.seed(RANDOM_SEED)
    fc = m.predict(post[["date"] + regressors].rename(columns={"date": "ds"}))

    out = pd.DataFrame({
        "date": post["date"].values,
        "actual": post[TARGET].values,
        "counterfactual": fc["yhat"].values,
        "cf_lower": fc["yhat_lower"].values,
        "cf_upper": fc["yhat_upper"].values,
        "episode": episode,
    })
    out["excess"] = out["actual"] - out["counterfactual"]
    # "Outside the band" flags days the pre-conflict model genuinely failed to
    # anticipate -- a stricter criterion than simply being above the mean.
    out["outside_band"] = ((out["actual"] > out["cf_upper"]) |
                            (out["actual"] < out["cf_lower"])).astype(int)
    return out


def report_counterfactual(cf):
    mean_excess = cf["excess"].mean()
    total_excess = cf["excess"].sum()
    pct_outside = 100 * cf["outside_band"].mean()
    pct_above = 100 * (cf["excess"] > 0).mean()

    print(f"\n  Mean excess over counterfactual : {mean_excess:+.2f} EUR/MWh")
    print(f"  Cumulative excess               : {total_excess:+,.0f} EUR/MWh-days")
    print(f"  Days above counterfactual       : {pct_above:.1f}%")
    print(f"  Days outside the 90% band       : {pct_outside:.1f}%")
    print(f"    (about 10% is expected by chance -- materially more than that")
    print(f"     indicates the pre-conflict model genuinely failed to anticipate")
    print(f"     what happened)")

    if abs(mean_excess) < 3:
        print(f"\n  INTERPRETATION: the forecast deviation is small. German electricity prices")
        print(f"  stayed close to the path implied by pre-conflict dynamics. This")
        print(f"  is a substantive finding -- see the header note on LNG")
        print(f"  diversification -- and it explains why conflict features add no")
        print(f"  forecasting value.")
    else:
        direction = "above" if mean_excess > 0 else "below"
        print(f"\n  INTERPRETATION: prices ran {abs(mean_excess):.1f} EUR/MWh "
              f"{direction} the counterfactual on average.")
    return {"mean_excess": mean_excess, "total_excess": total_excess,
            "pct_outside": pct_outside, "pct_above": pct_above}


def plot_counterfactual(cf, events):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(cf["date"], cf["cf_lower"], cf["cf_upper"],
                    alpha=0.2, color="tab:blue", label="Counterfactual 90% interval")
    ax.plot(cf["date"], cf["counterfactual"], "--", color="tab:blue",
            lw=1.6, label="Pre-conflict forecast benchmark")
    ax.plot(cf["date"], cf["actual"], color="tab:red", lw=1.4, label="Actual")

    for _, e in events.iterrows():
        if e["date"] >= cf["date"].min() and e["date"] <= cf["date"].max():
            ax.axvline(e["date"], color="grey", alpha=0.35, lw=0.8)

    ax.set_title("German day-ahead electricity price: actual vs. pre-conflict forecast benchmark\n"
                 "(model trained only on pre-conflict data; grey lines mark conflict events)")
    ax.set_ylabel("EUR/MWh")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = RESULTS_DIR / "magnitude_counterfactual.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------- PART B ----
def compute_car(series_df, event_date, col):
    s = series_df[["date", col]].dropna().sort_values("date").reset_index(drop=True)
    s["chg"] = s[col].diff()

    est_end = event_date - pd.Timedelta(days=ESTIMATION_GAP)
    est_start = est_end - pd.Timedelta(days=ESTIMATION_WINDOW)
    est = s[(s["date"] >= est_start) & (s["date"] < est_end)]["chg"].dropna()
    if len(est) < ESTIMATION_WINDOW * 0.6:
        return None

    normal, sd = est.mean(), est.std()
    if not np.isfinite(sd) or sd == 0:
        return None

    ev_start = event_date + pd.Timedelta(days=EVENT_WINDOW[0])
    ev_end = event_date + pd.Timedelta(days=EVENT_WINDOW[1])
    ev = s[(s["date"] >= ev_start) & (s["date"] <= ev_end)]
    required_days = EVENT_WINDOW[1] - EVENT_WINDOW[0] + 1
    if len(ev) < required_days:
        return None

    abnormal = ev["chg"].dropna() - normal
    car = abnormal.sum()
    return {"car": car, "car_std": car / (sd * np.sqrt(len(abnormal))),
            "n_days": len(abnormal), "normal_drift": normal, "est_sd": sd}


def event_study(df, events):
    rows = []
    for _, e in events.iterrows():
        for col in [TARGET] + [c for c in SECONDARY if c in df.columns]:
            r = compute_car(df, e["date"], col)
            if r is None:
                continue
            rows.append({
                "event_date": e["date"].date(),
                "event": str(e.get("event", ""))[:70],
                "series": col,
                "CAR": round(r["car"], 2),
                "CAR_standardised": round(r["car_std"], 3),
                "n_days": r["n_days"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- PART C ----
def placebo_test(df, events, col, n=N_PLACEBO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    valid = df[(df["date"] < CONFLICT_START - pd.Timedelta(days=30)) &
               (df["date"] > df["date"].min() + pd.Timedelta(days=ESTIMATION_WINDOW + 10))]
    if valid.empty:
        return None
    exclusion_dates = list(pd.to_datetime(events["date"]))
    ukraine_start, ukraine_end = pd.Timestamp("2022-02-24"), pd.Timestamp("2023-02-23")
    candidates = valid.loc[
        (~valid["date"].between(ukraine_start - pd.Timedelta(days=14),
                                ukraine_end + pd.Timedelta(days=14))) &
        (~valid["date"].apply(
            lambda d: any(abs((d - event_date).days) <= 14
                          for event_date in exclusion_dates)
        )), "date"
    ].values

    placebo = []
    attempts = 0
    while len(placebo) < n and attempts < n * 6:
        attempts += 1
        d = pd.Timestamp(rng.choice(candidates))
        r = compute_car(df, d, col)
        if r is not None:
            placebo.append(r["car"])
    if len(placebo) < 50:
        return None
    placebo = np.array(placebo)

    results = []
    for _, e in events.iterrows():
        r = compute_car(df, e["date"], col)
        if r is None:
            continue
        p = float((1 + np.sum(np.abs(placebo) >= abs(r["car"]))) /
                  (len(placebo) + 1))
        results.append({
            "event_date": e["date"].date(),
            "event": str(e.get("event", ""))[:60],
            "CAR": round(r["car"], 2),
            "placebo_p": p,
            "significant": p < 0.05,
        })
    return pd.DataFrame(results), placebo


# ------------------------------------------------------------------ MAIN ----
if __name__ == "__main__":
    df, raw_events = load_data()
    events = cluster_overlapping_events(raw_events)
    print(f"Loaded {len(df)} rows, {len(raw_events)} verified events "
          f"({len(events)} non-overlapping episodes)")
    print(f"Conflict start: {CONFLICT_START.date()}\n")

    print("=" * 78)
    print("PART A -- COUNTERFACTUAL FORECAST")
    print("=" * 78)
    cf = counterfactual(df)
    cf_stats = report_counterfactual(cf)
    print("\n  Matched 2022 Ukraine-shock comparison:")
    cf_2022 = counterfactual(
        df, pd.Timestamp("2022-02-24"), horizon=len(cf), episode="2022 Ukraine"
    )
    cf_2022_stats = report_counterfactual(cf_2022)
    pd.concat([cf, cf_2022], ignore_index=True).to_csv(
        RESULTS_DIR / "magnitude_counterfactual.csv", index=False
    )
    plot_counterfactual(cf, events)

    print("\n" + "=" * 78)
    print("PART B -- EVENT STUDY (cumulative abnormal response)")
    print("=" * 78)
    car = event_study(df, events)
    if car.empty:
        print("  No events had sufficient data coverage.")
    else:
        for col in car["series"].unique():
            sub = car[car["series"] == col].sort_values("CAR", key=abs, ascending=False)
            print(f"\n  {col} -- largest absolute responses:")
            for _, r in sub.head(5).iterrows():
                print(f"    {r['event_date']}  CAR {r['CAR']:>8.2f}  "
                      f"(std {r['CAR_standardised']:>6.2f})  {r['event'][:48]}")
        car.to_csv(RESULTS_DIR / "magnitude_event_car.csv", index=False)
        print(f"\n  Saved: {RESULTS_DIR / 'magnitude_event_car.csv'}")

    print("\n" + "=" * 78)
    print("PART C -- PLACEBO SIGNIFICANCE")
    print("=" * 78)
    print(f"  Comparing real-event responses against {N_PLACEBO} random "
          f"pre-conflict dates.\n")

    placebo_summary = {}
    placebo_frames = []
    for col in [TARGET] + [c for c in SECONDARY if c in df.columns]:
        res = placebo_test(df, events, col)
        if res is None:
            print(f"  {col}: insufficient data for placebo test")
            continue
        pl_df, pl_dist = res
        # Holm adjustment within each market controls family-wise error across
        # the several event episodes tested against the same placebo sample.
        order = np.argsort(pl_df["placebo_p"].values)
        m_tests = len(pl_df)
        adjusted = np.empty(m_tests)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m_tests - rank) * pl_df.iloc[idx]["placebo_p"])
            adjusted[idx] = min(running, 1.0)
        pl_df["placebo_p_holm"] = adjusted
        pl_df["significant"] = pl_df["placebo_p_holm"] < 0.05
        pl_df["series"] = col
        placebo_frames.append(pl_df)
        n_sig = int(pl_df["significant"].sum())
        placebo_summary[col] = (pl_df, n_sig)
        print(f"  {col}:")
        print(f"    placebo CAR distribution: mean {pl_dist.mean():+.2f}, "
              f"sd {pl_dist.std():.2f}, 95th pct of |CAR| "
              f"{np.percentile(np.abs(pl_dist), 95):.2f}")
        print(f"    events significant at Holm-adjusted p<0.05: {n_sig}/{len(pl_df)}")
        for _, r in pl_df[pl_df["significant"]].iterrows():
            print(f"      * {r['event_date']}  CAR {r['CAR']:+.2f}  "
                  f"p={r['placebo_p']:.3f}  {r['event'][:42]}")

    # ---------------------------------------------------------- summary ----
    lines = [
        "# Magnitude Analysis",
        "",
        "How much did the 2026 Iran-US conflict move German electricity prices?",
        "",
        "## Counterfactual",
        "",
        f"- Mean deviation from the pre-conflict forecast: **{cf_stats['mean_excess']:+.2f} EUR/MWh**",
        f"- Cumulative excess: {cf_stats['total_excess']:+,.0f} EUR/MWh-days",
        f"- Days above the counterfactual: {cf_stats['pct_above']:.1f}%",
        f"- Days outside the 90% interval: {cf_stats['pct_outside']:.1f}% "
        f"(~10% expected by chance)",
        "",
        "The benchmark is built with Prophet trained only on pre-conflict data and ",
        "conditions on lagged realised renewable output. It is a model-based forecast ",
        "deviation, not an identified causal effect. ",
        "The hybrid model is deliberately not used: it depends on conflict features, ",
        "which by construction do not exist in a no-conflict world.",
        f"A matched-length 2022 Ukraine-shock comparison has mean forecast "
        f"deviation **{cf_2022_stats['mean_excess']:+.2f} EUR/MWh** over "
        f"{len(cf_2022)} days.",
        "",
        "## Event study",
        "",
    ]
    if not car.empty:
        for col in car["series"].unique():
            sub = car[car["series"] == col]
            lines.append(f"- **{col}**: mean |CAR| {sub['CAR'].abs().mean():.2f}, "
                          f"largest {sub['CAR'].abs().max():.2f}")
    lines += [
        "",
        "## Placebo significance",
        "",
        "A single conflict provides no cross-sectional sample, so standard event-study ",
        "t-tests do not apply. Each event's response is instead compared against the ",
        f"distribution of responses from {N_PLACEBO} random pre-conflict dates.",
        "",
    ]
    for col, (pl_df, n_sig) in placebo_summary.items():
        lines.append(f"- **{col}**: {n_sig}/{len(pl_df)} event episodes significant at Holm-adjusted p<0.05")
    lines += [
        "",
        "## Interpretation",
        "",
        "Forecast accuracy and price impact are separate questions. An ablation result ",
        "about predictive value does not imply that the ",
        "conflict had no effect. Conversely, a small measured effect here would ",
        "explain the ablation result: features cannot predict a movement that did not ",
        "occur.",
        "",
        "Cross-market comparison matters for the interpretation. If Brent shows a large ",
        "response while German electricity does not, the finding is one of ",
        "**transmission**: the shock hit global oil but did not propagate to German ",
        "power, plausibly reflecting post-2022 LNG diversification.",
    ]
    out_md = RESULTS_DIR / "magnitude_summary.md"
    if placebo_frames:
        pd.concat(placebo_frames, ignore_index=True).to_csv(
            RESULTS_DIR / "magnitude_placebo_tests.csv", index=False
        )
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved summary: {out_md}")
    print("\nNext: 13_scenario_engine.py, once the RQ1 framing decision is made.")
