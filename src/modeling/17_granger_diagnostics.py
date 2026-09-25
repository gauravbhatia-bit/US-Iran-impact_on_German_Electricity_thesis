import hashlib
import json
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import coint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "smard_mean"
VOLUME = "guardian_article_count"
GAS = "ttf_eur_mwh"
GPR = "gpr_daily"
BRENT = "DCOILBRENTEU"

MAX_LAG = 7
ALPHA = 0.05
CONFLICT_START = pd.Timestamp("2026-02-28")
DATE_ONLY_AVAILABILITY_LAG = 2


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


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def align_daily_calendar(frame, columns):
    work = frame[["date", *columns]].copy().sort_values("date")
    if work["date"].duplicated().any():
        raise ValueError("Granger diagnostic input contains duplicate dates")
    calendar = pd.date_range(work["date"].min(), work["date"].max(), freq="D")
    return (work.set_index("date").reindex(calendar)
            .rename_axis("date").reset_index())


def granger(d, effect, cause, max_lag=MAX_LAG, difference=True):
    calendar = align_daily_calendar(d, [effect, cause])
    x = calendar[[effect, cause]].copy()
    if difference:
        x = x.diff()

    rows = []
    for lag in range(1, max_lag + 1):
        design = pd.DataFrame({"y": x[effect]})
        restricted_cols = []
        cause_cols = []
        for offset in range(1, lag + 1):
            y_col = f"y_l{offset}"
            cause_col = f"cause_l{offset}"
            design[y_col] = x[effect].shift(offset)
            design[cause_col] = x[cause].shift(offset)
            restricted_cols.append(y_col)
            cause_cols.append(cause_col)
        design = design.dropna()
        if len(design) < max_lag * 10:
            continue
        try:
            restricted = sm.OLS(
                design["y"], sm.add_constant(design[restricted_cols])
            ).fit()
            unrestricted = sm.OLS(
                design["y"],
                sm.add_constant(design[restricted_cols + cause_cols]),
            ).fit()
            df_num = lag
            df_den = int(unrestricted.df_resid)
            if df_den <= 0 or unrestricted.ssr <= 0:
                continue
            f_stat = ((restricted.ssr - unrestricted.ssr) / df_num) / (
                unrestricted.ssr / df_den
            )
            rows.append({
                "lag": lag,
                "F": float(f_stat),
                "p": float(stats.f.sf(f_stat, df_num, df_den)),
                "n_calendar_complete": len(design),
                "calendar_days_in_subperiod": len(calendar),
            })
        except Exception as exc:
            print(f"      [error] {type(exc).__name__}: {exc}")
            return None
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["p_adj"] = holm(out["p"].values)
    out["significant"] = out["p_adj"] < ALPHA
    return out


def show(res, label, n=None):
    if res is None:
        print(f"    {label}: insufficient data")
        return 0
    sig = int(res["significant"].sum())
    if n is None and "n_calendar_complete" in res:
        n_min = int(res["n_calendar_complete"].min())
        n_max = int(res["n_calendar_complete"].max())
        tag = f" (complete calendar windows n={n_min} to {n_max})"
    else:
        tag = f" (n={n})" if n else ""
    print(f"    {label}{tag}: significant at {sig}/{MAX_LAG} lags, "
          f"smallest adj p = {res['p_adj'].min():.4f}")
    return sig


def calendar_sample_fields(prefix, res):
    if res is None or "n_calendar_complete" not in res:
        return {}
    return {
        f"{prefix}_n_calendar_complete_min": int(res["n_calendar_complete"].min()),
        f"{prefix}_n_calendar_complete_max": int(res["n_calendar_complete"].max()),
        f"{prefix}_calendar_days_in_subperiod": int(
            res["calendar_days_in_subperiod"].max()
        ),
    }


def conditional_granger(d, effect, cause, controls, max_lag=MAX_LAG):
    cols = [effect, cause] + controls
    calendar = align_daily_calendar(d, cols)
    x = calendar[cols].diff()
    rows = []
    for L in range(1, max_lag + 1):
        frame = pd.DataFrame({"y": x[effect]})
        restricted_cols = []
        cause_cols = []
        for lag in range(1, L + 1):
            y_lag = f"y_l{lag}"
            frame[y_lag] = x[effect].shift(lag)
            restricted_cols.append(y_lag)
            for c in controls:
                control_lag = f"{c}_l{lag}"
                frame[control_lag] = x[c].shift(lag)
                restricted_cols.append(control_lag)
            cause_lag = f"cause_l{lag}"
            frame[cause_lag] = x[cause].shift(lag)
            cause_cols.append(cause_lag)
        frame = frame.dropna()
        if len(frame) < max_lag * 12:
            continue

        yr = frame["y"]
        Xr = sm.add_constant(frame[restricted_cols])
        Xu = sm.add_constant(frame[restricted_cols + cause_cols])
        try:
            mr = sm.OLS(yr, Xr).fit()
            mu = sm.OLS(yr, Xu).fit()
            ssr_r, ssr_u = mr.ssr, mu.ssr
            df_num = L
            df_den = int(mu.df_resid)
            if df_den <= 0 or ssr_u <= 0:
                continue
            F = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den)
            p = float(stats.f.sf(F, df_num, df_den))
            rows.append({
                "lag": L,
                "F": F,
                "p": p,
                "n_calendar_complete": len(frame),
                "calendar_days_in_subperiod": len(calendar),
            })
        except Exception:
            continue

    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["p_adj"] = holm(out["p"].values)
    out["significant"] = out["p_adj"] < ALPHA
    return out


if __name__ == "__main__":
    input_path = PROCESSED_DIR / "master_features.csv"
    raw_df = pd.read_csv(input_path, parse_dates=["date"])
    raw_df = raw_df.sort_values("date").reset_index(drop=True)
    df, availability_metadata = apply_date_only_availability(
        raw_df, [VOLUME, GAS, GPR, BRENT]
    )
    volume = available_name(VOLUME)
    gas = available_name(GAS)
    gpr = available_name(GPR)
    brent = available_name(BRENT)
    usable_news_dates = df.loc[df[volume].notna(), "date"]
    if usable_news_dates.empty:
        raise ValueError("No D-2-available Guardian volume observations")
    news = df[(df["date"] >= usable_news_dates.min()) &
              (df["date"] <= usable_news_dates.max())].copy()
    conflict = df[df["date"] >= CONFLICT_START].copy()
    print(f"Loaded {len(df)} days; D-2 usable news window {len(news)} days")
    print("Applied a conservative D-2 availability treatment to Guardian, GPR, "
          "Brent and TTF before all diagnostic differences and Granger lags.\n")

    records = []

    # ---------------------------------------------------- CHECK A ----
    print("=" * 76)
    print("CHECK A -- REVERSE CAUSALITY")
    print("=" * 76)
    print("  If price also Granger-causes news volume, the relationship is")
    print("  bidirectional and a common driver is more likely than a leading")
    print("  signal. This is the first thing an examiner will ask.\n")

    fwd = granger(news, TARGET, volume)
    rev = granger(news, volume, TARGET)
    f_sig = show(fwd, "news volume -> price")
    r_sig = show(rev, "price -> news volume")

    print()
    if f_sig and r_sig:
        print("  BIDIRECTIONAL. Causality runs both ways, so this cannot be")
        print("  reported as news leading prices. Most likely both respond to a")
        print("  common driver (market stress, or conflict events driving both).")
        print("  Report as a bidirectional dynamic association, not a leading signal.")
        verdict_a = "BIDIRECTIONAL"
    elif f_sig and not r_sig:
        print("  UNIDIRECTIONAL, news -> price. This is the pattern a genuine")
        print("  leading signal would produce.")
        verdict_a = "UNIDIRECTIONAL (news leads)"
    elif r_sig and not f_sig:
        print("  REVERSED. Price leads news, which would mean coverage responds")
        print("  to market moves rather than anticipating them.")
        verdict_a = "REVERSED (price leads)"
    else:
        print("  Neither direction significant.")
        verdict_a = "NEITHER"
    records.append({"check": "A_reverse_causality", "result": verdict_a,
                    "forward_sig_lags": f_sig, "reverse_sig_lags": r_sig,
                    "external_availability_lag_days": DATE_ONLY_AVAILABILITY_LAG,
                    **calendar_sample_fields("forward", fwd),
                    **calendar_sample_fields("reverse", rev)})

    # ---------------------------------------------------- CHECK B ----
    print("\n" + "=" * 76)
    print("CHECK B -- SPECIFICATION (TTF relationship in stationary models)")
    print("=" * 76)
    print("  A levels regression alone is unsafe for non-stationary prices.")
    print("  This check evaluates the full available analysis window with")
    print("  cointegration and a stationary error-correction model (ECM).")
    print("  It is a specification diagnostic, not a positive-control claim")
    print("  for the shorter news-available Granger window.\n")

    gas_diff = granger(df, TARGET, gas)
    show(gas_diff, "gas -> price (differences, full analysis window)")

    # Engle-Granger test plus an ECM: all regression outcomes are stationary.
    d = align_daily_calendar(df, [TARGET, gas, "dunkelflaute_flag"])
    d = d.dropna().copy()
    _, coint_p, _ = coint(d[TARGET], d[gas])
    long_run = sm.OLS(d[TARGET], sm.add_constant(d[[gas]])).fit()
    d["ec_lag1"] = long_run.resid.shift(1)
    d["d_price"] = d[TARGET].diff()
    d["d_price_lag1"] = d["d_price"].shift(1)
    d["d_gas_lag1"] = d[gas].diff().shift(1)
    d["dunkelflaute_lag1"] = d["dunkelflaute_flag"].shift(1)
    d = d.dropna()
    X = sm.add_constant(d[["d_gas_lag1", "d_price_lag1", "ec_lag1",
                           "dunkelflaute_lag1"]])
    m = sm.OLS(d["d_price"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    print(f"    Engle-Granger cointegration p = {coint_p:.4g}")
    print(f"    ECM short-run gas coefficient {m.params['d_gas_lag1']:+.4f}, "
          f"p = {m.pvalues['d_gas_lag1']:.4g}")
    print(f"    ECM adjustment coefficient {m.params['ec_lag1']:+.4f}, "
          f"p = {m.pvalues['ec_lag1']:.4g}")

    gas_diff_sig = int(gas_diff["significant"].sum()) if gas_diff is not None else 0
    short_run_sig = m.pvalues["d_gas_lag1"] < ALPHA
    cointegrated = coint_p < ALPHA
    adjustment_sig = m.pvalues["ec_lag1"] < ALPHA
    print()
    if cointegrated and adjustment_sig:
        print("  A long-run equilibrium relation is supported; the ECM, not a raw")
        print("  levels regression, is the defensible specification.")
        verdict_b = "cointegrated; ECM supports long-run adjustment"
    elif short_run_sig or gas_diff_sig:
        print("  Gas has a short-run differenced relationship without evidence for")
        print("  a stable levels equilibrium.")
        verdict_b = "short-run differenced relationship only"
    else:
        print("  The full-window gas relationship is unsupported in stationary")
        print("  models. Do not use an old levels result to validate the news test.")
        verdict_b = "full-window gas relationship unsupported"
    records.append({"check": "B_specification", "result": verdict_b,
                    "gas_diff_sig_lags": gas_diff_sig,
                    "cointegration_p": float(coint_p),
                    "ecm_gas_p": float(m.pvalues["d_gas_lag1"]),
                    "ecm_adjustment_p": float(m.pvalues["ec_lag1"]),
                    "ecm_n_calendar_complete": int(len(d)),
                    "external_availability_lag_days": DATE_ONLY_AVAILABILITY_LAG,
                    **calendar_sample_fields("gas_difference", gas_diff)})

    # ---------------------------------------------------- CHECK C ----
    print("\n" + "=" * 76)
    print("CHECK C -- CONFOUNDING")
    print("=" * 76)
    corr_gpr = news[[volume, gpr]].corr().iloc[0, 1] if gpr in news else np.nan
    corr_brent = news[[volume, brent]].corr().iloc[0, 1] if brent in news else np.nan
    print(f"  news volume vs GPR   : r = {corr_gpr:.3f}")
    print(f"  news volume vs Brent : r = {corr_brent:.3f}")
    print("  The unconditional correlations do not identify an independent")
    print("  news-volume effect. If volume proxies these conditions, controlling")
    print("  for them should remove its association with price.\n")

    controls = [c for c in [gpr, brent, gas] if c in news.columns]
    control_labels = {
        gpr: "GPR (D-2)",
        brent: "Brent (D-2)",
        gas: "TTF (D-2)",
    }
    cond = conditional_granger(news, TARGET, volume, controls)
    if cond is None:
        print("    conditional test: insufficient data")
        verdict_c = "insufficient data"
        cond_sig = 0
    else:
        cond_sig = int(cond["significant"].sum())
        cond_n_min = int(cond["n_calendar_complete"].min())
        cond_n_max = int(cond["n_calendar_complete"].max())
        print(f"    news volume -> price, controlling for "
              f"{', '.join(control_labels[c] for c in controls)}: "
              f"significant at {cond_sig}/{MAX_LAG} lags "
              f"(complete calendar windows n={cond_n_min} to {cond_n_max}), "
              f"smallest adj p = {cond['p_adj'].min():.4f}")
        print()
        if cond_sig:
            print("  SURVIVES THIS SPECIFICATION. The association is not fully")
            print("  absorbed by the included commodity-price and risk controls,")
            print("  but this is not evidence of an independent causal mechanism.")
            verdict_c = "survives controls"
        else:
            print("  DOES NOT SURVIVE. Once commodity prices and geopolitical risk")
            print("  are controlled for, news volume adds no separately detectable")
            print("  information. The unconditional association may be picking up")
            print("  market conditions already represented elsewhere.")
            verdict_c = "does not survive controls"
    records.append({"check": "C_confounding", "result": verdict_c,
                    "conditional_sig_lags": cond_sig,
                    "corr_with_gpr": float(corr_gpr) if np.isfinite(corr_gpr) else None,
                    "external_availability_lag_days": DATE_ONLY_AVAILABILITY_LAG,
                    **calendar_sample_fields("conditional", cond)})

    # ---------------------------------------------------- CHECK D ----
    print("\n" + "=" * 76)
    print("CHECK D -- ROBUSTNESS")
    print("=" * 76)
    print("  Article counts are skewed and non-stationary in levels. If the")
    print("  result depends on transformation or on a few")
    print("  extreme days, it is not a stable relationship.\n")

    n2 = news.copy()
    n2["log_volume"] = np.log1p(n2[volume])
    log_res = granger(n2, TARGET, "log_volume")
    log_sig = show(log_res, "log(1 + volume) -> price")

    cutoff = n2[volume].quantile(0.95)
    trimmed = n2.copy()
    trimmed.loc[trimmed[volume] > cutoff, volume] = np.nan
    trim_res = granger(trimmed, TARGET, volume)
    trim_sig = show(trim_res, f"volume -> price, top 5% days removed "
                               f"(> {cutoff:.0f} articles)")

    conf_res = granger(conflict, TARGET, volume)
    conf_sig = show(conf_res, "volume -> price, conflict period only")

    print()
    stable = sum([log_sig > 0, trim_sig > 0, conf_sig > 0])
    if stable == 3:
        print("  ROBUST across all three variants.")
        verdict_d = "robust"
    elif stable == 0:
        print("  FRAGILE: the result disappears under every variant. Most likely")
        print("  driven by extreme coverage days rather than a stable relationship.")
        verdict_d = "fragile"
    else:
        print(f"  PARTIALLY ROBUST: holds in {stable} of 3 variants. Report the")
        print("  sensitivity explicitly rather than the headline result alone.")
        verdict_d = f"partially robust ({stable}/3)"
    records.append({"check": "D_robustness", "result": verdict_d,
                    "log_sig": log_sig, "trimmed_sig": trim_sig,
                    "conflict_sig": conf_sig,
                    "external_availability_lag_days": DATE_ONLY_AVAILABILITY_LAG,
                    **calendar_sample_fields("log", log_res),
                    **calendar_sample_fields("trimmed", trim_res),
                    **calendar_sample_fields("conflict", conf_res)})

    # ---------------------------------------------------- VERDICT ----
    print("\n" + "=" * 76)
    print("OVERALL VERDICT")
    print("=" * 76)
    print(f"  A reverse causality : {verdict_a}")
    print(f"  B specification     : {verdict_b}")
    print(f"  C confounding       : {verdict_c}")
    print(f"  D robustness        : {verdict_d}")
    print()

    survives = (verdict_a.startswith("UNIDIRECTIONAL")
                and verdict_c == "survives controls"
                and verdict_d == "robust")
    if survives:
        print("  The association SURVIVES all checks. It is evidence consistent")
        print("  with news-volume predictive precedence, but remains observational")
        print("  and does not itself establish an operational forecasting gain.")
    else:
        print("  The association does NOT survive all checks. Do not report it as")
        print("  a clean leading signal; report the unconditional timing result")
        print("  alongside the reverse-causality and confounding qualifications.")

    pd.DataFrame(records).to_csv(
        RESULTS_DIR / "granger_diagnostics.csv", index=False)

    availability_trace = {
        "schema_version": 1,
        "analysis": "RQ5 Granger diagnostic checks",
        "input_path": str(input_path.relative_to(PROJECT_ROOT)),
        "input_sha256": sha256(input_path),
        "input_date_start": str(raw_df["date"].min().date()),
        "input_date_end": str(raw_df["date"].max().date()),
        "date_only_external_availability_rule": (
            "Guardian article volume, TTF, GPR and Brent are shifted two "
            "calendar days before differences, correlations, ECM construction "
            "and Granger-lag construction."
        ),
        "effective_external_timing": (
            "For target day t, a reported Granger lag L uses raw external "
            "information from t-(L+2)."
        ),
        "calendar_lag_rule": (
            "All Granger designs are built on a complete daily calendar. A "
            "row is retained only if its differenced target and every required "
            "calendar-day target/cause/control lag are observed; missing "
            "Guardian days are not compressed, zero-filled or interpolated."
        ),
        "target_history_rule": (
            "The realised electricity price is retained as the target; only "
            "historical target lags enter the restricted/unrestricted models."
        ),
        "interpretation_scope": (
            "These are retrospective robustness diagnostics with conservative "
            "external-data availability, not operational same-day forecasts."
        ),
        "series": availability_metadata,
    }
    with (RESULTS_DIR / "granger_diagnostics_availability_metadata.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(availability_trace, fh, indent=2)

    lines = [
        "# Granger Diagnostics", "",
        "Stress-tests of the observed news-volume Granger association before it "
        "is interpreted as an independent leading signal.", "",
        "All date-only external series are shifted by two calendar days before "
        "the diagnostic calculations. A reported Granger lag *L* therefore "
        "uses raw external information from *t-(L+2)*; realised electricity "
        "price history remains in its normal lagged form. The checks are "
        "retrospective predictive-precedence diagnostics, not operational "
        "same-day forecasts.", "",
        "Every Granger design is built on a complete daily calendar. A row is "
        "retained only when its target and all required calendar-day lags are "
        "observed; missing Guardian dates are not compressed, zero-filled or "
        "interpolated.", "",
        "| Check | Question | Result |", "|---|---|---|",
        f"| A | Does causality run both ways? | {verdict_a} |",
        f"| B | What is the full-window TTF relationship in stationary models? | {verdict_b} |",
        f"| C | Does it survive controlling for GPR, Brent and gas? | {verdict_c} |",
        f"| D | Is it robust to transformation and outliers? | {verdict_d} |",
        "", "## Note on interpretation", "",
        "Granger causality tests predictive precedence, not causation. A "
        "significant result means one series helps forecast another; it does not "
        "establish a causal mechanism.",
    ]
    with (RESULTS_DIR / "granger_diagnostics_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {RESULTS_DIR / 'granger_diagnostics.csv'}")
    print(f"Saved: {RESULTS_DIR / 'granger_diagnostics_summary.md'}")
    print(f"Saved: {RESULTS_DIR / 'granger_diagnostics_availability_metadata.json'}")
