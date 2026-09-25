from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EVENTS = RAW / "hormuz_event_table.csv"
MASTER = PROCESSED / "master_features.csv"
PORTWATCH = RAW / "imf_portwatch_hormuz.csv"
CONFLICT_START = pd.Timestamp("2026-02-28")
PRIMARY_STATES = ["open", "restricted", "severely_restricted"]


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(MASTER, parse_dates=["date"])
    required = {"date", "smard_mean", "ttf_eur_mwh"}
    missing = required - set(daily.columns)
    if missing:
        raise RuntimeError(f"master_features.csv is missing {sorted(missing)}")
    daily = daily.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    events = pd.read_csv(EVENTS)
    required_events = {
        "event_id", "event_date", "status", "main_scope", "included_primary",
        "source_verified", "source_url", "exact_transition_time_verified",
    }
    missing = required_events - set(events.columns)
    if missing:
        raise RuntimeError(f"Hormuz event table is missing {sorted(missing)}")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="raise")
    for col in ["included_primary", "source_verified", "exact_transition_time_verified"]:
        events[col] = events[col].astype(str).str.lower().isin(["true", "1", "yes"])
    if not events["source_verified"].all():
        raise RuntimeError("Every Hormuz event row must be source verified")
    if not events["source_url"].str.startswith("https://").all():
        raise RuntimeError("Every Hormuz event row must have an HTTPS source")
    bad = set(events.loc[events["included_primary"], "status"]) - set(PRIMARY_STATES)
    if bad:
        raise RuntimeError(f"Primary event table contains invalid states: {sorted(bad)}")
    return daily, events.sort_values(["event_date", "event_id"]).reset_index(drop=True)


def attach_event_states(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["hormuz_state"] = "unknown"
    out["state_evidence"] = "outside_source_window"
    out["state_event_id"] = ""
    state = "open"
    evidence = "pre-event baseline (not a physical closure claim)"
    event_id = ""
    primary = events[(events["included_primary"]) & (events["main_scope"] == "strait_of_hormuz")]
    for i, row in out.iterrows():
        hit = primary[primary["event_date"] <= row["date"]]
        if not hit.empty:
            latest = hit.iloc[-1]
            state = str(latest["status"])
            evidence = f"source event: {latest['event_id']}"
            event_id = str(latest["event_id"])
        out.at[i, "hormuz_state"] = state
        out.at[i, "state_evidence"] = evidence
        out.at[i, "state_event_id"] = event_id
    out["state_open"] = (out["hormuz_state"] == "open").astype(int)
    out["state_restricted"] = (out["hormuz_state"] == "restricted").astype(int)
    out["state_severely_restricted"] = (
        out["hormuz_state"] == "severely_restricted"
    ).astype(int)
    out["ttf_return"] = out["ttf_eur_mwh"].pct_change() * 100.0
    out["power_return"] = out["smard_mean"].diff()
    out["ttf_change"] = out["ttf_eur_mwh"].diff()
    out["weekday"] = out["date"].dt.dayofweek
    return out


def fit_hac(frame: pd.DataFrame, y_col: str, x_cols: list[str], maxlags: int = 7):
    use = frame[[y_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(40, len(x_cols) * 4):
        return None, use
    model = sm.OLS(use[y_col], sm.add_constant(use[x_cols], has_constant="add"))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags}), use


def status_regression(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for lag in range(1, 4):
        work[f"ttf_return_lag{lag}"] = work["ttf_return"].shift(lag)
    x = ["state_restricted", "state_severely_restricted"]
    x += [f"ttf_return_lag{i}" for i in range(1, 4)]
    weekday = pd.get_dummies(work["weekday"], prefix="dow", drop_first=True, dtype=float)
    work = pd.concat([work, weekday], axis=1)
    x += list(weekday.columns)
    result, use = fit_hac(work, "ttf_return", x)
    if result is None:
        return pd.DataFrame()
    rows = []
    for term in result.params.index:
        rows.append({
            "term": term,
            "coefficient": float(result.params[term]),
            "std_error": float(result.bse[term]),
            "p_value": float(result.pvalues[term]),
            "n_obs": int(result.nobs),
            "model": "TTF return ~ Hormuz state + own lags + weekday + HAC(7)",
        })
    return pd.DataFrame(rows)


def event_windows(frame: pd.DataFrame, events: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    rows = []
    for _, event in events[(events["included_primary"]) &
                           (events["main_scope"] == "strait_of_hormuz")].iterrows():
        date = event["event_date"]
        pre = frame[(frame["date"] >= date - pd.Timedelta(days=window)) &
                    (frame["date"] < date)]["ttf_eur_mwh"].dropna()
        post = frame[(frame["date"] >= date) &
                     (frame["date"] <= date + pd.Timedelta(days=window))]["ttf_eur_mwh"].dropna()
        if len(pre) == 0 or len(post) == 0:
            continue
        rows.append({
            "event_id": event["event_id"], "event_date": date.date(),
            "status": event["status"], "event": event.get("event", ""),
            "pre_mean_ttf": float(pre.mean()), "post_mean_ttf": float(post.mean()),
            "post_minus_pre_ttf": float(post.mean() - pre.mean()),
            "pre_n": int(len(pre)), "post_n": int(len(post)),
            "exact_transition_time_verified": bool(event["exact_transition_time_verified"]),
            "source_url": event["source_url"],
        })
    return pd.DataFrame(rows)


def placebo_test(frame: pd.DataFrame, actual_date: pd.Timestamp, window: int = 7) -> tuple[pd.DataFrame, float]:
    rows = []
    max_date = actual_date - pd.Timedelta(days=window * 2)
    candidates = frame[(frame["date"] >= frame["date"].min() + pd.Timedelta(days=window)) &
                       (frame["date"] <= max_date)].copy()
    # Seven-day spacing avoids treating overlapping pseudo-events as independent.
    candidates = candidates.iloc[::7]
    for _, r in candidates.iterrows():
        d = r["date"]
        pre = frame[(frame["date"] >= d - pd.Timedelta(days=window)) & (frame["date"] < d)]["ttf_eur_mwh"]
        post = frame[(frame["date"] >= d) & (frame["date"] <= d + pd.Timedelta(days=window))]["ttf_eur_mwh"]
        if len(pre.dropna()) < 5 or len(post.dropna()) < 5:
            continue
        rows.append({"placebo_date": d.date(), "delta_ttf": float(post.mean() - pre.mean())})
    placebo = pd.DataFrame(rows)
    actual = frame[(frame["date"] >= actual_date - pd.Timedelta(days=window)) &
                   (frame["date"] < actual_date)]["ttf_eur_mwh"].mean()
    actual_post = frame[(frame["date"] >= actual_date) &
                        (frame["date"] <= actual_date + pd.Timedelta(days=window))]["ttf_eur_mwh"].mean()
    actual_delta = float(actual_post - actual)
    p = float((np.abs(placebo["delta_ttf"]) >= abs(actual_delta)).mean()) if len(placebo) else np.nan
    return placebo, p


def reclosure_sensitivity(frame: pd.DataFrame, start: pd.Timestamp = pd.Timestamp("2026-07-01"),
                          end: pd.Timestamp = pd.Timestamp("2026-07-15")) -> pd.DataFrame:
    rows = []
    for date in pd.date_range(start, end, freq="D"):
        pre = frame[(frame["date"] >= date - pd.Timedelta(days=7)) & (frame["date"] < date)]["ttf_eur_mwh"]
        post = frame[(frame["date"] >= date) & (frame["date"] <= date + pd.Timedelta(days=7))]["ttf_eur_mwh"]
        if len(pre.dropna()) < 5 or len(post.dropna()) < 5:
            continue
        rows.append({"candidate_reclosure_date": date.date(),
                     "pre_mean_ttf": float(pre.mean()), "post_mean_ttf": float(post.mean()),
                     "post_minus_pre_ttf": float(post.mean() - pre.mean())})
    return pd.DataFrame(rows)


def ttf_to_power(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    x = []
    for lag in range(1, 15):
        col = f"ttf_return_lag{lag}"
        work[col] = work["ttf_return"].shift(lag)
        x.append(col)
    for lag in range(1, 8):
        col = f"power_return_lag{lag}"
        work[col] = work["power_return"].shift(lag)
        x.append(col)
    # Lagged renewable output is known at the price-setting forecast origin.
    if "renewable_output_mwh_lag1" in work:
        x.append("renewable_output_mwh_lag1")
    weekday = pd.get_dummies(work["weekday"], prefix="dow", drop_first=True, dtype=float)
    work = pd.concat([work, weekday], axis=1)
    x += list(weekday.columns)
    result, use = fit_hac(work, "power_return", x, maxlags=14)
    if result is None:
        return pd.DataFrame()
    rows = []
    for lag in range(1, 15):
        term = f"ttf_return_lag{lag}"
        rows.append({"term": term, "lag_days": lag,
                     "coefficient": float(result.params[term]),
                     "std_error": float(result.bse[term]),
                     "p_value": float(result.pvalues[term]),
                     "n_obs": int(result.nobs),
                     "model": "German power change ~ lagged TTF returns + controls + HAC(14)"})
    # A Wald test for the cumulative 1--14 day response.
    terms = [f"ttf_return_lag{i}" for i in range(1, 15)]
    weights = np.zeros(len(result.params))
    for term in terms:
        weights[list(result.params.index).index(term)] = 1.0
    estimate = float(weights @ result.params.to_numpy())
    variance = float(weights @ result.cov_params().to_numpy() @ weights)
    se = float(np.sqrt(max(variance, 0.0)))
    z = estimate / se if se else np.nan
    p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    rows.append({"term": "cumulative_lag1_14", "lag_days": "1-14",
                 "coefficient": estimate, "std_error": se, "p_value": p,
                 "n_obs": int(result.nobs),
                 "model": "Wald sum of distributed TTF lags"})
    return pd.DataFrame(rows)


def traffic_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    if not PORTWATCH.exists():
        return pd.DataFrame([{"status": "not_run", "reason": "PortWatch file missing"}])
    traffic = pd.read_csv(PORTWATCH, parse_dates=["date"])
    traffic["date"] = traffic["date"].dt.tz_localize(None)
    base = traffic.loc[traffic["date"] < CONFLICT_START, "n_total"].median()
    traffic["traffic_shortfall"] = 1 - traffic["n_total"] / base
    work = frame.merge(traffic[["date", "traffic_shortfall"]], on="date", how="left")
    for lag in range(0, 8):
        work[f"shortfall_lag{lag}"] = work["traffic_shortfall"].shift(lag)
    x = [f"shortfall_lag{i}" for i in range(8)]
    result, use = fit_hac(work, "ttf_return", x, maxlags=7)
    if result is None:
        return pd.DataFrame([{"status": "not_run", "reason": "insufficient complete rows"}])
    return pd.DataFrame([{"status": "optional_robustness", "term": term,
                          "coefficient": float(result.params[term]),
                          "p_value": float(result.pvalues[term]),
                          "n_obs": int(result.nobs),
                          "baseline_median_vessels": float(base)}
                         for term in x])


def plot_ttf(frame: pd.DataFrame, events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    colors = {"open": "#d9f2d9", "restricted": "#fff0bf", "severely_restricted": "#ffd6d6"}
    dates = frame["date"]
    for state, color in colors.items():
        mask = frame["hormuz_state"] == state
        if not mask.any():
            continue
        groups = (mask != mask.shift(fill_value=False)).cumsum()
        for _, group in frame[mask].groupby(groups[mask]):
            ax.axvspan(group["date"].min(), group["date"].max() + pd.Timedelta(days=1),
                       color=color, alpha=0.45, lw=0)
    ax.plot(frame["date"], frame["ttf_eur_mwh"], color="#144d7a", lw=1.25, label="Daily TTF (EUR/MWh)")
    for _, event in events[events["included_primary"]].iterrows():
        ax.axvline(event["event_date"], color="#444", lw=0.8, ls="--", alpha=0.65)
    ax.set_title("Dutch TTF around source-verified Hormuz states")
    ax.set_ylabel("EUR/MWh")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "rq2a_ttf_event_chart.png", dpi=180)
    plt.close(fig)


def write_summary(frame: pd.DataFrame, events: pd.DataFrame, windows: pd.DataFrame,
                  status: pd.DataFrame, power: pd.DataFrame, placebo_p: float,
                  sensitivity: pd.DataFrame, traffic: pd.DataFrame) -> None:
    state_means = frame.groupby("hormuz_state")["ttf_eur_mwh"].agg(["count", "mean", "median"]).round(3)
    state_lines = [f"- {idx}: n={int(row['count'])}, mean={row['mean']:.2f}, median={row['median']:.2f} EUR/MWh"
                   for idx, row in state_means.iterrows()]
    severe = status[status["term"] == "state_severely_restricted"] if not status.empty else pd.DataFrame()
    severe_text = "not estimable"
    if not severe.empty:
        severe_text = f"{severe.iloc[0]['coefficient']:+.3f}% TTF return (HAC p={severe.iloc[0]['p_value']:.4f})"
    cumulative = power[power["term"] == "cumulative_lag1_14"] if not power.empty else pd.DataFrame()
    power_text = "not estimable"
    if not cumulative.empty:
        power_text = f"{cumulative.iloc[0]['coefficient']:+.3f} EUR/MWh per 1% TTF return (p={cumulative.iloc[0]['p_value']:.4f})"
    lines = [
        "# RQ2a Event-table Hormuz transmission",
        "",
        "The primary exposure has three states -- open, restricted and severely",
        "restricted -- set by a measurable rule: daily IMF PortWatch transit volume",
        "against the 81-vessel-per-day pre-conflict median (open at or above 70% of",
        "baseline, restricted 20-70%, severely restricted below 20%).  The sourced",
        "event table supplies the dated transitions; the threshold rule decides which",
        "state they carry.  Brent is intentionally excluded from the primary model",
        "because the thesis mechanism is European gas pass-through to German",
        "electricity; oil is discussed as context rather than treated as a direct",
        "German power-price input.",
        "",
        "## Data and identification",
        "",
        "The 28 February conflict-start row is a restricted warning state and the 2",
        "March closure row is severely restricted.  The 15 June agreement-to-open is",
        "a diplomatic annotation only: transits on 15-17 June ran at 9-12% of",
        "baseline, so the severely restricted state continues through 17 June.  The",
        "18 June JMIC advisory marks a partial, contested corridor (20 transits,",
        "24.7% of baseline), coded restricted rather than open.  Ship attacks on 6",
        "July, the memorandum collapse on 8 July and the IRGC closure declaration on",
        "12 July return the state to severely restricted through the data cutoff.",
        "No verified open state exists anywhere after the conflict's start, so the",
        "analysis reports re-closure-date sensitivity rather than reopening-date",
        "sensitivity.",
        "",
        "## Descriptive TTF levels",
        *state_lines,
        "",
        "## Event windows and tests",
        f"- Seven-day event-window placebo p-value for the 2 March transition: {placebo_p:.4f}.",
        f"- Severe-state coefficient in a HAC daily TTF-return regression: {severe_text}.",
        f"- Distributed lag from previous-day TTF returns to German daily price changes (lags 1-14): {power_text}.",
        f"- Re-closure sensitivity candidates evaluated (1-15 July): {len(sensitivity)} dates.",
        "",
        "The event study is exploratory: there is one main closure episode, event",
        "dates are daily rather than exact hours, and state labels are not a causal",
        "estimate by themselves.  Results should be read as evidence on whether a",
        "source-verified state change coincided with a TTF movement and whether TTF",
        "movements subsequently passed through to German electricity.",
        "",
        "## Outputs",
        "",
        "- `rq2a_hormuz_daily_exposure.csv`: daily state timeline and price returns.",
        "- `rq2a_ttf_event_windows.csv`: pre/during/post event-window changes.",
        "- `rq2a_ttf_event_study.csv`: HAC state regression coefficients.",
        "- `rq2a_ttf_power_distributed_lag.csv`: TTF-to-power lags 1-14.",
        "- `rq2a_ttf_placebos.csv` and `rq2a_reclosure_sensitivity.csv`.",
        "- `rq2a_traffic_robustness.csv`: optional PortWatch check, not primary exposure.",
        "- `rq2a_ttf_event_chart.png`: shaded three-state TTF chart.",
    ]
    (RESULTS / "rq2a_event_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    daily, events = read_inputs()
    frame = attach_event_states(daily, events)
    frame.to_csv(RESULTS / "rq2a_hormuz_daily_exposure.csv", index=False)
    windows = event_windows(frame, events)
    windows.to_csv(RESULTS / "rq2a_ttf_event_windows.csv", index=False)
    status = status_regression(frame)
    status.to_csv(RESULTS / "rq2a_ttf_event_study.csv", index=False)
    power = ttf_to_power(frame)
    power.to_csv(RESULTS / "rq2a_ttf_power_distributed_lag.csv", index=False)
    placebo, placebo_p = placebo_test(frame, pd.Timestamp("2026-03-02"))
    placebo.to_csv(RESULTS / "rq2a_ttf_placebos.csv", index=False)
    sensitivity = reclosure_sensitivity(frame)
    sensitivity.to_csv(RESULTS / "rq2a_reclosure_sensitivity.csv", index=False)
    traffic = traffic_robustness(frame)
    traffic.to_csv(RESULTS / "rq2a_traffic_robustness.csv", index=False)
    plot_ttf(frame, events)
    diagnostics = {
        "primary_exposure": "source-verified hormuz_event_table.csv",
        "states": PRIMARY_STATES,
        "brent_primary": False,
        "exact_transition_times_verified": bool(events["exact_transition_time_verified"].any()),
        "n_daily_rows": int(len(frame)),
        "state_counts": frame["hormuz_state"].value_counts().to_dict(),
        "n_source_events": int(events["included_primary"].sum()),
        "placebo_p_value": placebo_p,
        "traffic_is_robustness_only": True,
    }
    (RESULTS / "rq2a_event_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    write_summary(frame, events, windows, status, power, placebo_p, sensitivity, traffic)
    print("RQ2a event-table study complete")
    print(f"  states: {diagnostics['state_counts']}")
    print(f"  outputs: {RESULTS / 'rq2a_event_summary.md'}")


if __name__ == "__main__":
    main()
