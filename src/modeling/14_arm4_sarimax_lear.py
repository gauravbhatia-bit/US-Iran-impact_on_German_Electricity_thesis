import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "smard_mean"

# SARIMAX is slow (state-space fit per fold). Set True for a fast run on the
# shock folds only, which is where the research question lives.
SHOCK_ONLY = False

# SARIMAX exogenous set: small and interpretable on purpose.
SARIMAX_EXOG = [
    "ttf_eur_mwh_lag2",
    "DCOILBRENTEU_lag2",
    "gpr_daily_lag2",
    "guardian_sentiment_mean_lag2",
]
SARIMAX_ORDER = (1, 0, 1)  # target is explicitly differenced below
SARIMAX_SEASONAL = (1, 0, 1, 7)


def build_feature_sets(df):
    price_lags = [c for c in df.columns if c.startswith("smard_mean_lag")]
    calendar = [c for c in ["day_of_week", "month", "is_weekend", "is_holiday"]
                if c in df.columns]
    market = [c for c in df.columns
              if any(c.startswith(p) for p in
              ["DCOILBRENTEU", "DEXUSEU", "ttf_eur_mwh"])
              and ("_lag" in c or "_vol" in c)
              and not c.endswith("_lag1")]
    weather = [c for c in df.columns if
               c.startswith("renewable_output_mwh_lag") or
               c.startswith("dunkelflaute_flag_lag")]
    geo = [c for c in df.columns if c.startswith("gpr_daily_lag") and not c.endswith("_lag1")]
    news = [c for c in df.columns if
            (("guardian" in c) or c.startswith("sentiment_available"))
             and "_lag" in c and not c.endswith("_lag1")]

    return {
        "A_price_only": price_lags + calendar,
        "E_all_features": price_lags + calendar + market + weather + geo + news,
    }, {"geo": geo, "news": news}


def prepare_lasso_matrix(train, test, feats):
    Xtr, Xte = train[feats].copy(), test[feats].copy()

    high_miss = [c for c in feats if Xtr[c].isna().mean() > 0.2]
    for c in high_miss:
        Xtr[f"{c}_missing"] = Xtr[c].isna().astype(int)
        Xte[f"{c}_missing"] = Xte[c].isna().astype(int)

    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med).fillna(0)
    Xte = Xte.fillna(med).fillna(0)

    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    return Xtr_s, Xte_s, list(Xtr.columns)


def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))


def fit_sarimax_with_retries(mod):
    attempts = (("lbfgs", 200), ("powell", 300))
    failures = []
    for method, maxiter in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = mod.fit(method=method, disp=False, maxiter=maxiter)
            if bool(res.mle_retvals.get("converged", False)):
                return res
            failures.append(f"{method}: did not converge")
        except Exception as exc:
            failures.append(f"{method}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(failures))


def run_arm4(master, folds, feature_sets, groups):
    rows, coef_rows, sel_rows = [], [], []
    total = len(folds)

    for i, (_, fold) in enumerate(folds.iterrows(), 1):
        train = master[master["date"] <= fold["train_end"]]
        test = master[(master["date"] >= fold["test_start"]) &
                      (master["date"] <= fold["test_end"])]
        if len(train) < 120 or test.empty:
            continue

        actual = test[TARGET].values
        row = {"fold_id": fold["fold_id"], "period": fold["period"]}

        # ---------------------------------------------------------- LEAR ----
        for set_name, feats in feature_sets.items():
            feats = [f for f in feats if f in master.columns]
            if not feats:
                continue
            try:
                Xtr, Xte, names = prepare_lasso_matrix(train, test, feats)
                ytr = train[TARGET].values
                ok = ~np.isnan(ytr)
                model = LassoCV(cv=5, max_iter=4000, random_state=42, n_jobs=-1)
                model.fit(Xtr[ok], ytr[ok])
                row[f"LEAR_{set_name}_MAE"] = mae(actual, model.predict(Xte))

                # Which features does LASSO actually keep? This is the direct
                # feature-selection answer, independent of accuracy.
                if set_name == "E_all_features":
                    nz = {n: c for n, c in zip(names, model.coef_) if abs(c) > 1e-8}
                    geo_kept = sum(1 for n in nz if n in groups["geo"])
                    news_kept = sum(1 for n in nz if n in groups["news"])
                    sel_rows.append({
                        "fold_id": fold["fold_id"], "period": fold["period"],
                        "n_selected": len(nz), "n_total": len(names),
                        "geo_features_kept": geo_kept,
                        "news_features_kept": news_kept,
                        "alpha": model.alpha_,
                    })
            except Exception as e:
                row[f"LEAR_{set_name}_MAE"] = np.nan

        # ------------------------------------------- XGBoost univariate ----
        try:
            uni = [c for c in master.columns if c.startswith("smard_mean_lag")]
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbosity=0)
            m.fit(train[uni], train[TARGET])
            row["XGB_univariate_MAE"] = mae(actual, m.predict(test[uni]))
        except Exception:
            row["XGB_univariate_MAE"] = np.nan

        # ------------------------------------------------------- SARIMAX ----
        exog_source = [c for c in SARIMAX_EXOG if c in master.columns]
        try:
            exog_changes = master[exog_source].diff().rename(
                columns={c: f"d_{c}" for c in exog_source}
            )
            exog_cols = list(exog_changes.columns)
            y = train[TARGET].diff().dropna()
            etr = exog_changes.loc[y.index].copy()
            ete = exog_changes.loc[test.index].copy()
            med = etr.median()
            etr, ete = etr.fillna(med).fillna(0), ete.fillna(med).fillna(0)

            y = y.reset_index(drop=True)
            mod = sm.tsa.SARIMAX(y, exog=etr.reset_index(drop=True),
                                 order=SARIMAX_ORDER,
                                 seasonal_order=SARIMAX_SEASONAL,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)
            res = fit_sarimax_with_retries(mod)
            pred_change = np.asarray(
                res.forecast(steps=len(test), exog=ete.reset_index(drop=True))
            )
            pred = float(train[TARGET].iloc[-1]) + np.cumsum(pred_change)
            row["SARIMAX_MAE"] = mae(actual, pred)

            # Coefficients are the point of this arm. Store per fold so the
            # distribution of p-values across folds can be summarised.
            params, pvals = res.params, res.pvalues
            for col in exog_cols:
                if col in params.index:
                    coef_rows.append({
                        "fold_id": fold["fold_id"], "period": fold["period"],
                        "variable": col,
                        "coef": float(params[col]),
                        "pvalue": float(pvals[col]),
                    })
        except Exception as e:
            row["SARIMAX_MAE"] = np.nan
            if i == 1:
                # Report the first failure rather than hiding every one.
                print(f"  [SARIMAX] first-fold failure: {type(e).__name__}: {e}")

        rows.append(row)
        if i % 20 == 0:
            print(f"  ...{i}/{total} folds done")

    return (pd.DataFrame(rows), pd.DataFrame(coef_rows), pd.DataFrame(sel_rows))


if __name__ == "__main__":
    master = pd.read_csv(PROCESSED_DIR / "master_features.csv", parse_dates=["date"])
    folds = pd.read_csv(PROCESSED_DIR / "cv_folds.csv",
                        parse_dates=["train_end", "test_start", "test_end"])
    if SHOCK_ONLY:
        folds = folds[folds["period"].str.startswith("SHOCK")]
        print("SHOCK_ONLY is set -- running shock folds only.\n")

    print(f"Loaded {len(master)} rows, {len(folds)} folds")
    feature_sets, groups = build_feature_sets(master)

    ALLOWED_SAMEDAY = {"day_of_week", "month", "is_weekend", "is_holiday"}
    leaks = []
    for name, feats in feature_sets.items():
        for f in feats:
            if f not in master.columns:
                continue
            if ("_lag" in f) or ("_vol" in f) or (f in ALLOWED_SAMEDAY):
                continue
            leaks.append(f"{name}: {f}")
    if leaks:
        raise RuntimeError("same-day leakage detected: " + "; ".join(leaks))
    else:
        print("  leakage check: passed (all features lagged or calendar)")

    for k, v in feature_sets.items():
        print(f"  {k}: {len([f for f in v if f in master.columns])} features")
    print(f"  SARIMAX exog: {[c for c in SARIMAX_EXOG if c in master.columns]}")
    print(f"\nRunning arm 4 (SARIMAX is the slow part)...")

    res, coefs, sel = run_arm4(master, folds, feature_sets, groups)
    print(f"Completed {len(res)} folds\n")

    # ------------------------------------------------------- accuracy ----
    for period in sorted(res["period"].dropna().unique()):
        s = res[res["period"] == period]
        if s.empty:
            continue
        print("=" * 74)
        print(f"ARM 4 -- {period} ({len(s)} folds)   MAE, lower = better")
        print("=" * 74)
        for col, lab in [("LEAR_A_price_only_MAE", "LEAR (price lags only)"),
                          ("LEAR_E_all_features_MAE", "LEAR (all features)"),
                          ("XGB_univariate_MAE", "XGBoost univariate"),
                          ("SARIMAX_MAE", "SARIMAX")]:
            if col in s and s[col].notna().any():
                print(f"  {lab:<26} {s[col].mean():7.2f}")

        a = s.get("LEAR_A_price_only_MAE")
        e = s.get("LEAR_E_all_features_MAE")
        if a is not None and e is not None and a.notna().any() and e.notna().any():
            d = 100 * (a.mean() - e.mean()) / a.mean()
            print(f"\n  LEAR: adding all features changes MAE by {d:+.2f}%")
            print("  -> " + ("features help under LASSO as well"
                              if d > 1 else
                              "consistent with the null: features add nothing "
                              "even under LASSO"))
        print()

    # ------------------------------------------ LASSO feature selection ----
    if not sel.empty:
        print("=" * 74)
        print("LASSO FEATURE SELECTION  (does it retain conflict features at all?)")
        print("=" * 74)
        for period in sorted(sel["period"].dropna().unique()):
            g = sel[sel["period"] == period]
            if g.empty:
                continue
            print(f"\n  {period} ({len(g)} folds)")
            print(f"    features retained : {g['n_selected'].mean():.1f} "
                  f"of {g['n_total'].iloc[0]}")
            print(f"    GPR lags retained : {g['geo_features_kept'].mean():.2f} "
                  f"(of {len(groups['geo'])})")
            print(f"    news lags retained: {g['news_features_kept'].mean():.2f} "
                  f"(of {len(groups['news'])})")
            if g["news_features_kept"].mean() < 0.5:
                print("    -> LASSO discards news coverage almost entirely.")
                print("       Independent confirmation of the null, from a method")
                print("       that selects features rather than weighting them.")
        sel.to_csv(RESULTS_DIR / "arm4_lasso_selection.csv", index=False)

    # ------------------------------------------- SARIMAX coefficients ----
    if not coefs.empty:
        print("\n" + "=" * 74)
        print("SARIMAX COEFFICIENTS  (direct significance test on conflict drivers)")
        print("=" * 74)
        print("  Each fold refits the model, so this reports the DISTRIBUTION of")
        print("  coefficients and p-values across folds rather than a single fit.\n")
        for period in sorted(coefs["period"].dropna().unique()):
            g = coefs[coefs["period"] == period]
            if g.empty:
                continue
            print(f"  {period}:")
            print(f"    {'variable':<32} {'mean coef':>11} {'median p':>10} "
                  f"{'% folds p<0.05':>15}")
            for v in g["variable"].unique():
                sub = g[g["variable"] == v]
                pct = 100 * (sub["pvalue"] < 0.05).mean()
                print(f"    {v:<32} {sub['coef'].mean():>11.4f} "
                      f"{sub['pvalue'].median():>10.4f} {pct:>14.1f}%")
            print()

        news_var = "d_guardian_sentiment_mean_lag2"
        for period in [p for p in coefs["period"].unique() if str(p).startswith("SHOCK")]:
            gs = coefs[(coefs["variable"] == news_var) & (coefs["period"] == period)]
            if gs.empty:
                continue
            pct = 100 * (gs["pvalue"] < 0.05).mean()
            print(f"  RQ4 direct test -- news coefficient, {period}:")
            print(f"    significant in {pct:.1f}% of folds "
                  f"(median p = {gs['pvalue'].median():.4f})")
            if pct < 20:
                print("    -> The news coefficient is not distinguishable from zero.")
                print("       This is INDEPENDENT evidence for the null: it tests the")
                print("       coefficient directly rather than inferring from accuracy.")
        coefs.to_csv(RESULTS_DIR / "arm4_coefficients.csv", index=False)

    required_metrics = ["LEAR_A_price_only_MAE", "LEAR_E_all_features_MAE",
                        "XGB_univariate_MAE", "SARIMAX_MAE"]
    failure_rates = {c: float(res[c].isna().mean()) for c in required_metrics if c in res}
    bad = {c: rate for c, rate in failure_rates.items() if rate > 0.05}
    if bad:
        raise RuntimeError(f"More than 5% model fits failed; results withheld: {bad}")
    out = RESULTS_DIR / "arm4_results.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("Compare against chronos_zeroshot_results.csv and "
          "prophet_xgboost_results.csv (same folds).")
