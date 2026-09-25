import hashlib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "smard_mean"
MAX_LAG = 7
ALPHA = 0.05
CONFLICT_START = pd.Timestamp("2026-02-28")
DATE_ONLY_AVAILABILITY_LAG = 2

# Gas is a diagnostic comparator, not a guaranteed positive control. Its
# short-run differenced relationship must be established on this run.
CANDIDATES = [
    ("guardian_sentiment_mean", "News sentiment (tone)"),
    ("guardian_article_count", "News volume (coverage intensity)"),
    ("gpr_daily", "Geopolitical risk index"),
    ("ttf_eur_mwh", "TTF gas price  [DIAGNOSTIC COMPARATOR]"),
]


def available_name(raw_name):
    return f"{raw_name}__available_d2"


def apply_date_only_availability(frame, raw_columns):
    out = frame.copy()
    metadata = []
    for raw in raw_columns:
        if raw not in out.columns:
            continue
        safe = available_name(raw)
        out[safe] = out[raw].shift(DATE_ONLY_AVAILABILITY_LAG)
        raw_observed = out.loc[out[raw].notna(), "date"]
        safe_observed = out.loc[out[safe].notna(), "date"]
        safe_start = safe_observed.min() if not safe_observed.empty else None
        safe_end = safe_observed.max() if not safe_observed.empty else None
        metadata.append({
            "raw_column": raw,
            "available_column": safe,
            "availability_lag_days": DATE_ONLY_AVAILABILITY_LAG,
            "raw_start": str(raw_observed.min().date()) if not raw_observed.empty else None,
            "raw_end": str(raw_observed.max().date()) if not raw_observed.empty else None,
            "usable_start": str(safe_start.date()) if safe_start is not None else None,
            "usable_end": str(safe_end.date()) if safe_end is not None else None,
            "raw_date_at_usable_start": (
                str((safe_start - pd.Timedelta(days=DATE_ONLY_AVAILABILITY_LAG)).date())
                if safe_start is not None else None
            ),
            "raw_date_at_usable_end": (
                str((safe_end - pd.Timedelta(days=DATE_ONLY_AVAILABILITY_LAG)).date())
                if safe_end is not None else None
            ),
        })
    return out, metadata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_stationarity(series, name):
    s = series.dropna()
    if len(s) < 20:
        return None
    try:
        stat, p, *_ = adfuller(s, autolag="AIC")
        return {"series": name, "adf_stat": stat, "p": p,
                "stationary": p < 0.05}
    except Exception:
        return None


def align_daily_calendar(frame, columns):
    work = frame[["date", *columns]].copy().sort_values("date")
    if work["date"].duplicated().any():
        raise ValueError("Granger input contains duplicate dates")
    calendar = pd.date_range(work["date"].min(), work["date"].max(), freq="D")
    return (work.set_index("date").reindex(calendar)
            .rename_axis("date").reset_index())


def run_granger(df, cause_col, effect_col, max_lag=MAX_LAG):
    d = align_daily_calendar(df, [effect_col, cause_col])
    d[effect_col] = d[effect_col].diff()
    d[cause_col] = d[cause_col].diff()
    rows = []
    min_required = max_lag * 10

    for lag in range(1, max_lag + 1):
        design = pd.DataFrame({"y": d[effect_col]})
        restricted_cols = []
        cause_lag_cols = []
        for offset in range(1, lag + 1):
            effect_lag = f"effect_lag{offset}"
            cause_lag = f"cause_lag{offset}"
            design[effect_lag] = d[effect_col].shift(offset)
            design[cause_lag] = d[cause_col].shift(offset)
            restricted_cols.append(effect_lag)
            cause_lag_cols.append(cause_lag)
        design = design.dropna()
        n_complete = len(design)
        if n_complete < min_required:
            continue

        try:
            restricted = sm.OLS(
                design["y"], sm.add_constant(design[restricted_cols])
            ).fit()
            unrestricted = sm.OLS(
                design["y"],
                sm.add_constant(design[restricted_cols + cause_lag_cols]),
            ).fit()
            df_num = lag
            df_den = int(unrestricted.df_resid)
            if df_den <= 0 or unrestricted.ssr <= 0:
                continue
            f_stat = ((restricted.ssr - unrestricted.ssr) / df_num) / (
                unrestricted.ssr / df_den
            )
            p_val = float(stats.f.sf(f_stat, df_num, df_den))
            rows.append({
                "lag": lag,
                "F": float(f_stat),
                "p": p_val,
                "significant": p_val < ALPHA,
                "n_calendar_complete": n_complete,
                "calendar_days_in_subperiod": len(d),
            })
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}", len(d)

    if not rows:
        return None, len(d)
    return pd.DataFrame(rows), len(d)


def holm_bonferroni(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


if __name__ == "__main__":
    input_path = PROCESSED_DIR / "master_features.csv"
    raw_df = pd.read_csv(input_path, parse_dates=["date"])
    raw_df = raw_df.sort_values("date").reset_index(drop=True)
    raw_columns = [col for col, _ in CANDIDATES if col in raw_df.columns]
    df, availability_metadata = apply_date_only_availability(raw_df, raw_columns)
    print(f"Loaded {len(df)} days")
    print("Applied a conservative D-2 availability treatment to all date-only "
          "external predictors before Granger lag construction.\n")

    # ------------------------------------------------- stationarity ----
    print("=" * 76)
    print("STATIONARITY CHECK (ADF).  H0: series has a unit root")
    print("=" * 76)
    print("  Granger causality requires stationary series. Levels are tested")
    print("  first; if non-stationary, first differences are used throughout.\n")
    print(f"  {'series':<34} {'ADF p (levels)':>15} {'ADF p (diff)':>14}")
    print("  " + "-" * 66)
    for col, label in [(TARGET, "German electricity price")] + \
                       [(available_name(c), l.replace("  [DIAGNOSTIC COMPARATOR]", ""))
                        for c, l in CANDIDATES]:
        if col not in df.columns:
            continue
        lv = check_stationarity(df[col], col)
        dv = check_stationarity(df[col].diff(), col)
        lv_s = f"{lv['p']:.4f}" if lv else "  --  "
        dv_s = f"{dv['p']:.4f}" if dv else "  --  "
        flag = "" if (dv and dv["stationary"]) else "  <- still non-stationary"
        print(f"  {label:<34} {lv_s:>15} {dv_s:>14}{flag}")
    print("\n  All tests below use FIRST DIFFERENCES. External-series ADF "
          "tests use their D-2 available versions.")

    # ---------------------------------------------------- main tests ----
    news_available = available_name("guardian_sentiment_mean")
    news_window = df[df[news_available].notna()] \
        if news_available in df.columns else df
    if not news_window.empty:
        win_start, win_end = news_window["date"].min(), news_window["date"].max()
    else:
        win_start = win_end = None

    all_rows = []
    for period_name, sub in [
        ("NEWS-AVAILABLE WINDOW", df[df["date"] >= win_start] if win_start is not None else df),
        ("CONFLICT PERIOD ONLY", df[df["date"] >= CONFLICT_START]),
    ]:
        print("\n" + "=" * 76)
        print(f"GRANGER CAUSALITY -- {period_name}")
        if not sub.empty:
            print(f"  {sub['date'].min().date()} to {sub['date'].max().date()} "
                  f"({len(sub)} days)")
        print("=" * 76)

        for raw_col, label in CANDIDATES:
            col = available_name(raw_col)
            if col not in sub.columns:
                continue
            res, n_used = run_granger(sub, col, TARGET)
            if isinstance(res, str):
                print(f"\n  {label}")
                print(f"    {res}")
                continue
            if res is None:
                print(f"\n  {label}")
                print(f"    insufficient data ({n_used} usable observations; "
                      f"need at least {MAX_LAG * 10})")
                continue

            res["p_adj"] = holm_bonferroni(res["p"].values)
            n_sig = int((res["p_adj"] < ALPHA).sum())
            best = res.loc[res["p_adj"].idxmin()]
            n_min = int(res["n_calendar_complete"].min())
            n_max = int(res["n_calendar_complete"].max())

            print(f"\n  {label}   (complete calendar windows per lag: "
                  f"n = {n_min} to {n_max}; subperiod days = {n_used})")
            print(f"    {'lag':>4} {'F':>9} {'p raw':>9} {'p adj':>9}")
            for _, r in res.iterrows():
                star = " *" if r["p_adj"] < ALPHA else ""
                print(f"    {int(r['lag']):>4} {r['F']:>9.3f} "
                      f"{r['p']:>9.4f} {r['p_adj']:>9.4f}{star}")
            if n_sig:
                print(f"    -> Granger-causes prices at {n_sig} of {MAX_LAG} lags "
                      f"(strongest: lag {int(best['lag'])}, adj p = {best['p_adj']:.4f})")
            else:
                print(f"    -> NO Granger causality at any lag "
                      f"(smallest adj p = {res['p_adj'].min():.4f})")

            for _, r in res.iterrows():
                all_rows.append({
                    "period": period_name,
                    "cause": raw_col,
                    "cause_available_column": col,
                    "availability_lag_days": DATE_ONLY_AVAILABILITY_LAG,
                    "granger_lag_interpretation": (
                        "reported lag L uses raw external information at t-(L+2)"
                    ),
                    "label": label,
                    "lag": int(r["lag"]), "F": round(r["F"], 4),
                    "p_raw": round(r["p"], 4), "p_adj": round(r["p_adj"], 4),
                    "significant": bool(r["p_adj"] < ALPHA),
                    "n": int(r["n_calendar_complete"]),
                    "n_calendar_complete": int(r["n_calendar_complete"]),
                    "calendar_days_in_subperiod": int(r["calendar_days_in_subperiod"]),
                })

    if not all_rows:
        raise SystemExit("No tests could be run.")

    out = pd.DataFrame(all_rows)
    out.to_csv(RESULTS_DIR / "granger_results.csv", index=False)

    availability_trace = {
        "schema_version": 1,
        "analysis": "RQ5 Granger predictive-precedence test",
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "input_sha256": sha256(input_path),
        "input_date_start": str(raw_df["date"].min().date()),
        "input_date_end": str(raw_df["date"].max().date()),
        "date_only_external_availability_rule": (
            "Every date-only external predictor is shifted two calendar days "
            "before differencing and Granger-lag construction."
        ),
        "effective_external_timing": (
            "For target day t, a reported Granger lag L uses raw external "
            "information from t-(L+2)."
        ),
        "calendar_lag_rule": (
            "Each test is constructed on a complete daily calendar. A row is "
            "retained only when its differenced target and every required "
            "calendar-day target/cause lag are observed; no missing Guardian "
            "days are compressed, zero-filled or interpolated."
        ),
        "target_history_rule": (
            "The realised electricity target is not availability-shifted; "
            "the restricted model uses its historical lags only."
        ),
        "interpretation_scope": (
            "This is a retrospective predictive-precedence diagnostic with "
            "conservative external-data availability, not an operational "
            "same-day forecasting claim."
        ),
        "series": availability_metadata,
    }
    with (RESULTS_DIR / "granger_availability_metadata.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(availability_trace, fh, indent=2)

    # ------------------------------------------------ interpretation ----
    print("\n" + "=" * 76)
    print("INTERPRETATION")
    print("=" * 76)

    gas = out[out["cause"] == "ttf_eur_mwh"]
    gas_sig = int(gas["significant"].sum()) if not gas.empty else 0
    print(f"\n  DIAGNOSTIC COMPARATOR -- TTF gas: significant at "
          f"{gas_sig} of {len(gas)} lag/period combinations")
    if gas_sig == 0:
        print("    The short-run differenced gas comparator is unsupported.")
        print("    Do not validate news results using an old levels regression.")
    else:
        print("    The test detects a gas lead-lag relationship in this sample.")

    news_rows = out[out["cause"].str.contains("guardian")]
    news_sig = int(news_rows["significant"].sum()) if not news_rows.empty else 0
    print(f"\n  RQ5 -- news coverage: significant at {news_sig} of "
          f"{len(news_rows)} lag/period combinations")
    if news_sig == 0:
        print("    NO Granger causality from news coverage to German electricity")
        print("    prices at any lag from 1 to 7 days.")
        print("    Past news carries no information about future prices beyond")
        print("    what past prices already contain. This is the direct test of")
        print("    the leading-signal claim, and it is a FIFTH independent")
        print("    confirmation of the null.")
        print("\n    Consistent with the author's M508 project (2016-2020, All the")
        print("    News 2.0, FinBERT), which found no Granger causality at lags 1-5")
        print("    (p = 0.43 to 0.86) using a different corpus, sentiment model and")
        print("    target variable. Cite explicitly as own prior work.")
    else:
        sig = news_rows[news_rows["significant"]]
        print("    Some evidence of Granger causality:")
        for _, r in sig.iterrows():
            print(f"      {r['label']} at lag {r['lag']} ({r['period']}), "
                  f"adj p = {r['p_adj']:.4f}")
        print("    NOTE: Granger causality is predictive precedence, not causation.")

    gpr = out[out["cause"] == "gpr_daily"]
    gpr_sig = int(gpr["significant"].sum()) if not gpr.empty else 0
    print(f"\n  Geopolitical risk index: significant at {gpr_sig} of "
          f"{len(gpr)} combinations")

    # ------------------------------------------------------ summary ----
    lines = [
        "# Granger Causality (RQ5)", "",
        "Does past news coverage help predict German electricity prices beyond "
        "what past prices already provide?", "",
        "## Method", "",
        f"F-test on nested models, lags 1 to {MAX_LAG}, Holm-Bonferroni adjusted "
        f"within each variable. All series first-differenced for stationarity "
        f"(ADF results reported in the script output).", "",
        "Every date-only external series (Guardian variables, GPR and TTF) is "
        "shifted by two calendar days before differencing and Granger-lag "
        "construction because the source exports do not establish a pre-auction "
        "publication time. Thus a reported Granger lag *L* uses raw external "
        "information from *t-(L+2)*. The realised electricity target remains "
        "unshifted and enters only through historical target lags. This is a "
        "retrospective predictive-precedence diagnostic, not an operational "
        "same-day forecasting result.", "",
        "Lags are built on a complete daily calendar. Rows with any missing "
        "required target or external lag are excluded, so a one-day lag always "
        "means one calendar day; missing Guardian dates are neither compressed "
        "nor replaced with invented values.", "",
        "TTF gas is included as a diagnostic comparator. Its relationship is "
        "not assumed from an earlier levels model.", "",
        "## Results", "",
        "| Variable | Period | Significant lags |", "|---|---|---|",
    ]
    for (cause, period), g in out.groupby(["label", "period"]):
        sig_lags = g[g["significant"]]["lag"].tolist()
        lines.append(f"| {cause} | {period} | "
                      f"{sig_lags if sig_lags else 'none'} |")
    prior_work = (
        "The author's M508 project tested a related relationship over 2016-2020 "
        "using All the News 2.0 and FinBERT, on price volatility rather than "
        "price level, and found no Granger causality at lags 1 to 5 "
        "(p = 0.43 to 0.86). "
    )
    if news_sig == 0:
        prior_work += (
            "That result is directionally consistent with this study, although "
            "the corpus, period, sentiment method, and target differ. "
        )
    else:
        prior_work += (
            "That null contrasts with the current article-volume result. The "
            "difference may reflect the period, corpus, or target, and the "
            "current bidirectional diagnostic prevents treating it as a clean "
            "leading signal. "
        )
    prior_work += (
        "It must be cited explicitly as the author's own prior work rather than "
        "presented as an independent replication."
    )

    lines += [
        "", "## Interpretation", "",
        f"- Diagnostic comparator (gas): {gas_sig} significant combinations",
        f"- News coverage: {news_sig} significant combinations",
        f"- Geopolitical risk: {gpr_sig} significant combinations", "",
        "Granger causality tests predictive precedence, not causation. A "
        "significant result means one series helps forecast another; it does not "
        "establish a causal mechanism.", "",
        "## Relation to prior work", "",
        prior_work,
    ]
    with (RESULTS_DIR / "granger_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {RESULTS_DIR / 'granger_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'granger_summary.md'}")
    print(f"Saved: {RESULTS_DIR / 'granger_availability_metadata.json'}")
