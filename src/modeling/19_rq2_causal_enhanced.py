from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TREATED = "DE-LU"
DONORS = ["FR", "CH", "NO2", "SE4", "DK1", "DK2"]
CONFLICT_START = pd.Timestamp("2026-02-28")
UKRAINE_START = pd.Timestamp("2022-02-24")
SEED = 42
N_BOOT = 3000
N_TIME_PLACEBOS = 120
QUARTER_HOUR_SWITCH = pd.Timestamp("2025-10-01")


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(values) - rank) * values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def block_bootstrap_mean(values, block_length=7, n_boot=N_BOOT, seed=SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < block_length * 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        sample = []
        while len(sample) < len(values):
            start = int(rng.integers(0, len(values)))
            sample.extend(values[(start + j) % len(values)] for j in range(block_length))
        means[b] = np.mean(sample[:len(values)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def load_hourly_prices():
    path = RAW / "energy_charts_prices.csv"
    frame = pd.read_csv(path)
    frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True)
    frame = frame[frame["bidding_zone"].isin([TREATED, *DONORS])].copy()
    frame["datetime_local"] = frame["datetime_utc"].dt.tz_convert("Europe/Berlin")
    frame["delivery_date"] = frame["datetime_local"].dt.tz_localize(None).dt.normalize()
    frame["hour_local"] = frame["datetime_local"].dt.hour
    if set([TREATED, *DONORS]) - set(frame["bidding_zone"]):
        raise ValueError("Enhanced RQ2 donor pool is incomplete")
    return frame


def canonical_hourly_intervals(price_intervals):
    columns = ["delivery_date", "bidding_zone", "hour_local", "datetime_utc", "price_eur_mwh"]
    d = price_intervals.loc[:, columns].copy()
    d["utc_hour"] = d["datetime_utc"].dt.floor("h")
    canonical = (
        d.groupby(
            ["delivery_date", "bidding_zone", "hour_local", "utc_hour"],
            as_index=False,
        )
        .agg(
            price_eur_mwh=("price_eur_mwh", "mean"),
            source_intervals=("price_eur_mwh", "size"),
        )
    )
    canonical["source_resolution"] = np.select(
        [canonical["source_intervals"].eq(1), canonical["source_intervals"].eq(4)],
        ["hourly", "quarter_hour"],
        default="incomplete_or_mixed",
    )
    source_rows = canonical.groupby("source_resolution")["source_intervals"].sum()
    diagnostics = {
        "switch_date": str(QUARTER_HOUR_SWITCH.date()),
        "method": (
            "Quarter-hour source observations are averaged to one value per "
            "local delivery date, bidding zone and actual delivery hour. "
            "UTC hour is retained to preserve both repeated 02:00 DST hours."
        ),
        "metrics_harmonised": ["volatility", "negative", "high_tail"],
        "metrics_retaining_direct_interval_means": ["mean", "peak", "offpeak"],
        "high_tail_threshold_basis": "95th percentile of canonical hourly pre-conflict prices by zone",
        "source_rows": {
            str(key): int(value)
            for key, value in source_rows.sort_index().items()
        },
        "canonical_hour_rows": {
            str(key): int(value)
            for key, value in canonical["source_resolution"].value_counts().sort_index().items()
        },
        "hours_with_unexpected_source_interval_count": int(
            canonical["source_resolution"].eq("incomplete_or_mixed").sum()
        ),
    }
    return canonical, diagnostics


def daily_outcomes(price_intervals):
    grouped = price_intervals.groupby(["delivery_date", "bidding_zone"])
    base = grouped["price_eur_mwh"].mean().rename("mean").reset_index()
    peak = price_intervals[price_intervals["hour_local"].between(17, 20)].groupby(
        ["delivery_date", "bidding_zone"]
    )["price_eur_mwh"].mean().rename("peak").reset_index()
    offpeak = price_intervals[price_intervals["hour_local"].between(0, 5)].groupby(
        ["delivery_date", "bidding_zone"]
    )["price_eur_mwh"].mean().rename("offpeak").reset_index()

    hourly, diagnostics = canonical_hourly_intervals(price_intervals)
    def resampling_check(name, raw_subset, hourly_subset):
        raw_daily = raw_subset.groupby(["delivery_date", "bidding_zone"])["price_eur_mwh"].mean()
        hourly_daily = hourly_subset.groupby(["delivery_date", "bidding_zone"])["price_eur_mwh"].mean()
        difference = raw_daily.sub(hourly_daily, fill_value=np.nan).dropna()
        diagnostics[f"max_abs_{name}_resampling_difference"] = float(difference.abs().max())
        diagnostics[f"n_{name}_resampling_invariance_failures"] = int(
            (difference.abs() > 1e-9).sum()
        )

    resampling_check("mean", price_intervals, hourly)
    resampling_check(
        "peak",
        price_intervals[price_intervals["hour_local"].between(17, 20)],
        hourly[hourly["hour_local"].between(17, 20)],
    )
    resampling_check(
        "offpeak",
        price_intervals[price_intervals["hour_local"].between(0, 5)],
        hourly[hourly["hour_local"].between(0, 5)],
    )
    pre = hourly[hourly["delivery_date"] < CONFLICT_START]
    thresholds = pre.groupby("bidding_zone")["price_eur_mwh"].quantile(.95)
    hourly["high_tail"] = hourly["price_eur_mwh"] > hourly["bidding_zone"].map(thresholds)
    hourly["negative"] = hourly["price_eur_mwh"] < 0
    intraday = (
        hourly.groupby(["delivery_date", "bidding_zone"])
        .agg(
            volatility=("price_eur_mwh", "std"),
            negative=("negative", "mean"),
            high_tail=("high_tail", "mean"),
        )
        .reset_index()
    )
    return base.merge(peak).merge(offpeak).merge(intraday), diagnostics


def fit_weights(pivot, start, end, treated=TREATED, donors=DONORS, ridge=1e-4):
    train = pivot.loc[(pivot.index >= start) & (pivot.index < end), [treated, *donors]].dropna()
    if len(train) < 365:
        raise ValueError(f"Synthetic-control pre-period has only {len(train)} days")
    y = train[treated].to_numpy(dtype=float)
    x = train[donors].to_numpy(dtype=float)
    scale = max(float(np.std(y)), 1.0)

    def objective(w):
        return np.mean(((y - x @ w) / scale) ** 2) + ridge * np.sum(w ** 2)

    result = minimize(
        objective, np.full(len(donors), 1 / len(donors)), method="SLSQP",
        bounds=[(0.0, 1.0)] * len(donors),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Synthetic-control optimisation failed: {result.message}")
    weights = pd.Series(result.x, index=donors)
    fitted = x @ result.x
    return weights, float(np.sqrt(np.mean((y - fitted) ** 2))), len(train)


def synthetic_outcome(outcomes, outcome, end_date):
    pivot = outcomes.pivot(index="delivery_date", columns="bidding_zone", values=outcome)
    pivot = pivot.sort_index()
    weights, pre_rmse, n_pre = fit_weights(
        pivot, pd.Timestamp("2023-01-01"), CONFLICT_START
    )
    valid = pivot[[TREATED, *DONORS]].dropna().copy()
    valid["synthetic"] = valid[DONORS].to_numpy() @ weights.to_numpy()
    valid["gap"] = valid[TREATED] - valid["synthetic"]
    pre = valid.loc[(valid.index >= CONFLICT_START - pd.Timedelta(days=180)) &
                    (valid.index < CONFLICT_START), "gap"]
    post = valid.loc[(valid.index >= CONFLICT_START) & (valid.index <= end_date), "gap"]
    baseline_gap = float(pre.mean())
    adjusted = post - baseline_gap
    adjusted_90 = adjusted.iloc[:90]
    ci7 = block_bootstrap_mean(adjusted, 7)
    ci14 = block_bootstrap_mean(adjusted, 14, seed=SEED + 1)
    hac = sm.OLS(adjusted.to_numpy(), np.ones((len(adjusted), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 14}
    )
    stats = {
        "outcome": outcome,
        "pre_rmse": pre_rmse,
        "n_pre": n_pre,
        "pre_gap": baseline_gap,
        "post_effect": float(adjusted.mean()),
        "post_effect_90d": float(adjusted_90.mean()),
        "hac_se": float(hac.bse[0]),
        "hac_p": float(hac.pvalues[0]),
        "block7_ci_low": ci7[0], "block7_ci_high": ci7[1],
        "block14_ci_low": ci14[0], "block14_ci_high": ci14[1],
        "n_post": len(adjusted),
    }
    path = valid.reset_index().rename(columns={TREATED: "germany"})
    path["outcome"] = outcome
    return stats, weights, path


def placebo_candidate_dates(horizon=90):
    latest_candidate = min(
        pd.Timestamp("2025-10-01"),
        CONFLICT_START - pd.Timedelta(days=horizon),
    )
    candidates = pd.date_range("2023-03-01", latest_candidate, freq="7D")
    return [d for d in candidates
            if not pd.Timestamp("2022-02-01") <= d <= pd.Timestamp("2023-03-01")]


def time_placebos(outcomes, outcome, observed_effect, horizon=90):
    pivot = outcomes.pivot(index="delivery_date", columns="bidding_zone", values=outcome).sort_index()
    candidates = placebo_candidate_dates(horizon)
    rng = np.random.default_rng(SEED)
    if len(candidates) > N_TIME_PLACEBOS:
        candidates = list(rng.choice(candidates, N_TIME_PLACEBOS, replace=False))
    rows = []
    for date in sorted(pd.to_datetime(candidates)):
        try:
            weights, pre_rmse, _ = fit_weights(
                pivot, date - pd.Timedelta(days=730), date
            )
        except (ValueError, RuntimeError):
            continue
        window = pivot.loc[(pivot.index >= date) &
                           (pivot.index < date + pd.Timedelta(days=horizon)),
                           [TREATED, *DONORS]].dropna()
        before = pivot.loc[(pivot.index >= date - pd.Timedelta(days=90)) &
                           (pivot.index < date), [TREATED, *DONORS]].dropna()
        if len(window) < horizon * .85 or len(before) < 75:
            continue
        post_gap = window[TREATED] - window[DONORS].to_numpy() @ weights.to_numpy()
        pre_gap = before[TREATED] - before[DONORS].to_numpy() @ weights.to_numpy()
        rows.append({"placebo_start": date.date(), "effect": float(post_gap.mean() - pre_gap.mean()),
                     "pre_rmse": pre_rmse})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, np.nan
    p = float((1 + np.sum(np.abs(frame["effect"]) >= abs(observed_effect))) /
              (len(frame) + 1))
    return frame, p


def prophet_estimands(master):
    from prophet import Prophet

    def fit_one(regressors, label, start=CONFLICT_START, horizon=None):
        pre = master[master["date"] < start].copy()
        post = master[master["date"] >= start].copy()
        if horizon is not None:
            post = post.head(horizon)
        needed = ["date", "smard_mean", *regressors]
        pre = pre[needed].dropna()
        post = post[needed].dropna()
        model = Prophet(yearly_seasonality="auto", weekly_seasonality=True,
                        daily_seasonality=False, interval_width=.90,
                        changepoint_prior_scale=.05)
        for col in regressors:
            model.add_regressor(col)
        model.fit(pre.rename(columns={"date": "ds", "smard_mean": "y"}))
        fc = model.predict(post[["date", *regressors]].rename(columns={"date": "ds"}))
        out = pd.DataFrame({"date": post["date"].to_numpy(),
                            "actual": post["smard_mean"].to_numpy(),
                            "counterfactual": fc["yhat"].to_numpy(),
                            "lower": fc["yhat_lower"].to_numpy(),
                            "upper": fc["yhat_upper"].to_numpy(),
                            "estimand": label})
        out["deviation"] = out["actual"] - out["counterfactual"]
        return out

    common = [c for c in ["renewable_output_mwh_lag1", "dunkelflaute_flag_lag1",
                            "is_holiday"] if c in master]
    total = fit_one(common, "total_associated_deviation_no_post_treatment_ttf")
    direct = fit_one([*common, "ttf_eur_mwh_lag2"],
                     "controlled_direct_deviation_conditional_on_realised_ttf")
    comparison_2022 = fit_one(common, "2022_total_associated_deviation",
                              UKRAINE_START, horizon=len(total))
    return pd.concat([total, direct, comparison_2022], ignore_index=True)


def standardised_episode_comparison(master, estimands):
    rows = []
    for label, start in [("2022", UKRAINE_START), ("2026", CONFLICT_START)]:
        pre = master[(master["date"] >= start - pd.Timedelta(days=60)) &
                     (master["date"] < start)]
        est_label = ("2022_total_associated_deviation" if label == "2022" else
                     "total_associated_deviation_no_post_treatment_ttf")
        dev = estimands[estimands["estimand"] == est_label]["deviation"]
        rows.append({
            "episode": label, "start": start.date(),
            "calendar_match": "late February",
            "pre_price_mean": pre["smard_mean"].mean(),
            "pre_price_volatility": pre["smard_mean"].std(),
            "pre_ttf_mean": pre["ttf_eur_mwh"].mean(),
            "pre_renewable_mean": pre["renewable_output_mwh"].mean(),
            "mean_deviation": dev.mean(),
            "standardised_deviation_pre_price_sd": dev.mean() / pre["smard_mean"].std(),
        })
    return pd.DataFrame(rows)


def hourly_event_analysis(hourly, mean_weights):
    event_path = RAW / "hourly_event_timestamps.csv"
    events = pd.read_csv(event_path)
    events["effective_at_utc"] = pd.to_datetime(events["effective_at_utc"], utc=True)
    h = hourly.copy()
    h["hour_utc"] = h["datetime_utc"].dt.floor("h")
    h = h.groupby(["hour_utc", "bidding_zone"], as_index=False)["price_eur_mwh"].mean()
    pivot = h.pivot(index="hour_utc", columns="bidding_zone", values="price_eur_mwh")
    pivot = pivot[[TREATED, *DONORS]].dropna()
    pivot["synthetic"] = pivot[DONORS].to_numpy() @ mean_weights.to_numpy()
    pivot["gap"] = pivot[TREATED] - pivot["synthetic"]
    rows = []
    for _, event in events.iterrows():
        t = event["effective_at_utc"]
        estimation = pivot.loc[(pivot.index >= t - pd.Timedelta(days=30)) &
                               (pivot.index < t - pd.Timedelta(days=2)), "gap"]
        window = pivot.loc[(pivot.index >= t - pd.Timedelta(hours=24)) &
                           (pivot.index <= t + pd.Timedelta(hours=72)), "gap"]
        if len(estimation) < 24 * 20 or len(window) < 80:
            continue
        abnormal = window - estimation.mean()
        low, high = block_bootstrap_mean(abnormal, 24, seed=SEED + len(rows))
        pseudo = []
        candidates = pd.date_range("2023-03-01", "2025-10-01", freq="7D", tz="UTC")
        for candidate in candidates:
            est_p = pivot.loc[(pivot.index >= candidate - pd.Timedelta(days=30)) &
                              (pivot.index < candidate - pd.Timedelta(days=2)), "gap"]
            win_p = pivot.loc[(pivot.index >= candidate - pd.Timedelta(hours=24)) &
                              (pivot.index <= candidate + pd.Timedelta(hours=72)), "gap"]
            if len(est_p) >= 24 * 20 and len(win_p) >= 80:
                pseudo.append(float((win_p - est_p.mean()).mean()))
        placebo_p = (np.nan if not pseudo else
                     float((1 + np.sum(np.abs(pseudo) >= abs(abnormal.mean()))) /
                           (len(pseudo) + 1)))
        rows.append({
            "effective_at_utc": t, "event": event["event"],
            "treatment_scope": event["treatment_scope"],
            "n_hours": len(abnormal), "mean_abnormal_gap": abnormal.mean(),
            "cumulative_abnormal_gap_hours": abnormal.sum(),
            "block24_ci_low": low, "block24_ci_high": high,
            "time_placebo_p": placebo_p,
            "source_url": event["source_url"],
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["time_placebo_p_holm"] = holm_adjust(result["time_placebo_p"])
    return result


def plot_control(path):
    s = path[path["outcome"] == "mean"]
    s = s[s["delivery_date"] >= pd.Timestamp("2025-09-01")]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(s["delivery_date"], s["germany"], lw=1.1, label="Germany")
    ax.plot(s["delivery_date"], s["synthetic"], lw=1.1, label="Synthetic control")
    ax.axvline(CONFLICT_START, color="tab:red", ls="--", label="Conflict start")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("RQ2 enhanced: Germany and pre-treatment-fitted synthetic control")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    output = RESULTS / "rq2_synthetic_control.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def main():
    hourly = load_hourly_prices()
    outcomes, resolution_diagnostics = daily_outcomes(hourly)
    end_date = min(outcomes["delivery_date"].max(), pd.Timestamp("2026-09-01"))
    summaries, weights_rows, paths = [], [], []
    mean_weights = None
    placebo = None
    placebo_p = np.nan
    placebo_full = None
    placebo_full_p = np.nan
    for outcome in ["mean", "peak", "offpeak", "volatility", "negative", "high_tail"]:
        stats, weights, path = synthetic_outcome(outcomes, outcome, end_date)
        summaries.append(stats)
        paths.append(path)
        weights_rows.extend({"outcome": outcome, "donor": donor, "weight": weight}
                            for donor, weight in weights.items())
        if outcome == "mean":
            mean_weights = weights
            placebo, placebo_p = time_placebos(
                outcomes, outcome, stats["post_effect_90d"], horizon=90
            )
            placebo_full, placebo_full_p = time_placebos(
                outcomes, outcome, stats["post_effect"], horizon=int(stats["n_post"])
            )
    summary = pd.DataFrame(summaries)
    summary["hac_p_holm"] = holm_adjust(summary["hac_p"])
    summary["significant_holm"] = summary["hac_p_holm"] < .05
    summary["time_placebo_p"] = np.where(summary["outcome"].eq("mean"), placebo_p, np.nan)
    summary.to_csv(RESULTS / "rq2_outcome_summary.csv", index=False)
    (RESULTS / "rq2_resolution_harmonisation.json").write_text(
        json.dumps(resolution_diagnostics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(weights_rows).to_csv(RESULTS / "rq2_synthetic_weights.csv", index=False)
    path_frame = pd.concat(paths, ignore_index=True)
    path_frame.to_csv(RESULTS / "rq2_synthetic_control.csv", index=False)
    if placebo is not None:
        placebo.to_csv(RESULTS / "rq2_time_placebos.csv", index=False)
    if placebo_full is not None:
        placebo_full.to_csv(RESULTS / "rq2_time_placebos_full_post.csv", index=False)

    master = pd.read_csv(PROCESSED / "master_features.csv", parse_dates=["date"])
    estimands = prophet_estimands(master)
    estimands.to_csv(RESULTS / "rq2_estimands.csv", index=False)
    episodes = standardised_episode_comparison(master, estimands)
    episodes.to_csv(RESULTS / "rq2_episode_comparison.csv", index=False)
    hourly_events = hourly_event_analysis(hourly, mean_weights)
    hourly_events.to_csv(RESULTS / "rq2_hourly_events.csv", index=False)
    chart = plot_control(path_frame)

    total = estimands[estimands["estimand"] ==
                      "total_associated_deviation_no_post_treatment_ttf"]["deviation"]
    direct = estimands[estimands["estimand"] ==
                       "controlled_direct_deviation_conditional_on_realised_ttf"]["deviation"]
    total_ci = block_bootstrap_mean(total, 14)
    direct_ci = block_bootstrap_mean(direct, 14, seed=SEED + 5)
    mean_row = summary[summary["outcome"] == "mean"].iloc[0]

    lines = [
        "# Enhanced RQ2 analysis", "", "## Estimands", "",
        f"- Total conflict-associated forecast deviation (TTF excluded as a post-treatment mediator): "
        f"**{total.mean():+.2f} EUR/MWh**; 14-day block 95% interval "
        f"{total_ci[0]:+.2f} to {total_ci[1]:+.2f}.",
        f"- Controlled direct forecast deviation conditional on the realised lagged TTF path: "
        f"**{direct.mean():+.2f} EUR/MWh**; 14-day block 95% interval "
        f"{direct_ci[0]:+.2f} to {direct_ci[1]:+.2f}.",
        "- These are model-based deviations. The synthetic-control section supplies a comparative "
        "design, but one conflict still limits causal identification.", "",
        "## European synthetic control", "",
        f"- Mean-price post-treatment gap, adjusted for the last 180 pre-treatment days: "
        f"**{mean_row['post_effect']:+.2f} EUR/MWh**.",
        f"- Matched 90-day mean gap used for time-placebo inference: "
        f"**{mean_row['post_effect_90d']:+.2f} EUR/MWh**.",
        f"- HAC p-value: {mean_row['hac_p']:.4f}.",
        f"- 7-day block interval: {mean_row['block7_ci_low']:+.2f} to "
        f"{mean_row['block7_ci_high']:+.2f}; 14-day block interval: "
        f"{mean_row['block14_ci_low']:+.2f} to {mean_row['block14_ci_high']:+.2f}.",
        f"- Matched 90-day time-placebo p-value ({0 if placebo is None else len(placebo)} pseudo-starts): "
        f"{placebo_p:.4f}.",
        f"- Full-post-length time-placebo p-value ({0 if placebo_full is None else len(placebo_full)} pseudo-starts): "
        f"{placebo_full_p:.4f}; all pseudo-windows end before the 2026 treatment.",
        "The placebo windows are spaced weekly but overlap; their effective number "
        "of independent windows is substantially smaller than the row count.", "",
        "Donors were declared before fitting as France, Switzerland, Norway NO2, Sweden SE4, "
        "Denmark DK1 and DK2. They are comparatively less directly tied to Middle Eastern gas, "
        "but they are not assumed to be completely unaffected by a European energy shock.", "",
        "## Distributional outcomes", "",
        "- Within-day volatility, negative-price share and high-tail share are calculated "
        "from a canonical hourly series. From 1 October 2025, four 15-minute source "
        "observations are averaged within each actual local delivery hour before those "
        "metrics and the pre-conflict high-tail threshold are calculated. Mean, peak and "
        "off-peak remain direct interval means because they are invariant to complete "
        "hourly-to-quarter-hourly resampling. The UTC hour is retained to preserve both "
        "02:00 DST hours. See `rq2_resolution_harmonisation.json`.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(f"- {row['outcome']}: adjusted gap {row['post_effect']:+.3f}, "
                     f"HAC p={row['hac_p']:.4f}, Holm-adjusted p={row['hac_p_holm']:.4f}")
    lines += ["", "## Hourly, timestamped events", ""]
    if hourly_events.empty:
        lines.append("No event had both an exact independently sourced operational timestamp and sufficient data.")
    else:
        for _, row in hourly_events.iterrows():
            lines.append(f"- {row['effective_at_utc']}: mean abnormal Germany-minus-control gap "
                         f"{row['mean_abnormal_gap']:+.2f} EUR/MWh; 24-hour block interval "
                         f"{row['block24_ci_low']:+.2f} to {row['block24_ci_high']:+.2f}; "
                         f"Holm-adjusted time-placebo p={row['time_placebo_p_holm']:.4f}.")
    lines += ["", "The two timestamped events are Iranian-port blockade operations, not a blanket "
              "Strait closure. This distinction corrects the earlier binary-state interpretation.", "",
              "## 2022 comparison", ""]
    for _, row in episodes.iterrows():
        lines.append(f"- {row['episode']}: mean deviation {row['mean_deviation']:+.2f}; "
                     f"{row['standardised_deviation_pre_price_sd']:+.2f} pre-event standard deviations; "
                     f"pre-event TTF mean {row['pre_ttf_mean']:.2f}.")
    lines += ["", f"Chart: `{chart.name}`"]
    (RESULTS / "rq2_enhanced_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Enhanced RQ2 complete")
    print(json.dumps({"total_deviation": total.mean(), "direct_deviation": direct.mean(),
                      "synthetic_mean_effect": mean_row["post_effect"],
                      "synthetic_hac_p": mean_row["hac_p"],
                       "time_placebo_p": placebo_p,
                       "full_post_time_placebo_p": placebo_full_p}, indent=2))


if __name__ == "__main__":
    main()
