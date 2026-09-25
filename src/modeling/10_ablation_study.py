import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)   # Prophet/Stan emit hundreds of identical
for _n in ("prophet", "cmdstanpy", "prophet.models", "prophet.forecaster"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)
    logging.getLogger(_n).propagate = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COL = "smard_mean"
MAX_FOLDS = None       # e.g. 25 for a quick trial
SHOCK_ONLY = False     # True = only run SHOCK folds (much faster, but loses
                       # the baseline comparison; keep False for the real run)


def load_data():
    master = pd.read_csv(PROCESSED_DIR / "master_features.csv")
    master["date"] = pd.to_datetime(master["date"])
    folds = pd.read_csv(PROCESSED_DIR / "cv_folds.csv")
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        folds[c] = pd.to_datetime(folds[c])
    return master, folds


def build_feature_sets(master):
    cols = set(master.columns)

    def lags_of(base, first_lag=1):
        return [f"{base}_lag{i}" for i in range(first_lag, 8)
                if f"{base}_lag{i}" in cols]

    price = lags_of("smard_mean")
    calendar = [c for c in ["day_of_week", "month", "is_weekend", "is_holiday"] if c in cols]
    price_vol = [c for c in ["smard_mean_vol7d", "smard_mean_vol30d"] if c in cols]

    # Daily close/reference exports have no pre-auction timestamp. Use D-2
    # onward; yesterday's electricity target history remains available at lag 1.
    market = (lags_of("DCOILBRENTEU", first_lag=2)
              + lags_of("ttf_eur_mwh", first_lag=2)
              + lags_of("DEXUSEU", first_lag=2)
              + [c for c in ["DCOILBRENTEU_vol7d", "DCOILBRENTEU_vol30d",
                              "ttf_eur_mwh_vol7d", "ttf_eur_mwh_vol30d"] if c in cols])

    weather = lags_of("dunkelflaute_flag") + lags_of("renewable_output_mwh")

    geopolitical = lags_of("gpr_daily", first_lag=2)

    sentiment = (lags_of("guardian_sentiment_mean", first_lag=2)
                 + lags_of("guardian_sentiment_min", first_lag=2)
                 + lags_of("guardian_sentiment_max", first_lag=2)
                 + lags_of("guardian_article_count", first_lag=2)
                 + lags_of("sentiment_available", first_lag=2))

    sets = {}
    sets["A_price_only"] = price + calendar + price_vol
    sets["B_plus_market"] = sets["A_price_only"] + market
    sets["C_plus_weather"] = sets["B_plus_market"] + weather
    sets["D_plus_geopolitical"] = sets["C_plus_weather"] + geopolitical
    sets["E_plus_sentiment"] = sets["D_plus_geopolitical"] + sentiment

    diagnostics = {
        "price lags": len(price), "calendar": len(calendar),
        "market lags/vol": len(market), "weather": len(weather),
        "geopolitical (GPR lags)": len(geopolitical),
        "sentiment lags": len(sentiment),
    }
    return sets, diagnostics


def metrics(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    m = ~np.isnan(a) & ~np.isnan(p)
    a, p = a[m], p[m]
    if len(a) == 0:
        return {"MAE": np.nan, "RMSE": np.nan}
    e = a - p
    return {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2))}


def run_ablation(master, folds, feature_sets):
    from prophet import Prophet
    import xgboost as xgb

    fold_iter = folds if MAX_FOLDS is None else folds.head(MAX_FOLDS)
    if SHOCK_ONLY:
        fold_iter = fold_iter[fold_iter["period"].str.startswith("SHOCK")]

    rows = []
    daily_rows = []   # per-observation errors, for properly powered DM tests
    total = len(fold_iter)
    for i, (_, fold) in enumerate(fold_iter.iterrows(), 1):
        train = master[master["date"] <= fold["train_end"]]
        test = master[(master["date"] >= fold["test_start"]) &
                      (master["date"] <= fold["test_end"])]
        if len(train) < 60 or test.empty:
            continue

        try:
            m = Prophet(yearly_seasonality="auto", weekly_seasonality=True,
                        daily_seasonality=False, changepoint_prior_scale=0.05)
            m.fit(train.rename(columns={"date": "ds", TARGET_COL: "y"})[["ds", "y"]])
        except Exception as e:
            print(f"  [fold {fold['fold_id']}] Prophet failed: {e}")
            continue

        p_train = m.predict(train[["date"]].rename(columns={"date": "ds"}))["yhat"].values
        p_test = m.predict(test[["date"]].rename(columns={"date": "ds"}))["yhat"].values
        train_resid = train[TARGET_COL].values - p_train
        actual = test[TARGET_COL].values

        row = {"fold_id": fold["fold_id"], "period": fold["period"]}
        row["prophet_MAE"] = metrics(actual, p_test)["MAE"]

        # One observation per true one-day-ahead origin, retained for paired
        # loss-differential tests.
        fold_daily = pd.DataFrame({
            "fold_id": fold["fold_id"],
            "period": fold["period"],
            "date": test["date"].values,
            "actual": actual,
            "prophet_abs_err": np.abs(actual - p_test),
        })

        for set_name, feats in feature_sets.items():
            if not feats:
                row[f"{set_name}_MAE"] = np.nan
                fold_daily[f"{set_name}_abs_err"] = np.nan
                continue
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
            model.fit(train[feats], train_resid)   # XGBoost handles NaN natively
            pred = p_test + model.predict(test[feats])
            row[f"{set_name}_MAE"] = metrics(actual, pred)["MAE"]
            fold_daily[f"{set_name}_abs_err"] = np.abs(actual - pred)

        rows.append(row)
        daily_rows.append(fold_daily)
        if i % 20 == 0:
            print(f"  ...{i}/{total} folds done")

    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return pd.DataFrame(rows), daily


def report(results, feature_sets):
    order = ["A_price_only", "B_plus_market", "C_plus_weather",
             "D_plus_geopolitical", "E_plus_sentiment"]
    labels = {
        "A_price_only": "A. Price lags + calendar",
        "B_plus_market": "B.  + market (Brent/TTF/FX)",
        "C_plus_weather": "C.  + weather (Dunkelflaute)",
        "D_plus_geopolitical": "D.  + geopolitical (GPR)",
        "E_plus_sentiment": "E.  + news sentiment",
    }

    for period in sorted(results["period"].dropna().unique()):
        sub = results[results["period"] == period]
        if sub.empty:
            continue
        print("\n" + "=" * 74)
        print(f"ABLATION -- {period} ({len(sub)} folds)   MAE, lower = better")
        print("=" * 74)
        print(f"{'Prophet alone (no XGBoost)':<38} {sub['prophet_MAE'].mean():8.2f}")
        print("-" * 74)

        prev = None
        for s in order:
            col = f"{s}_MAE"
            if col not in sub or sub[col].isna().all():
                print(f"{labels[s]:<38} {'--':>8}   (no features available)")
                continue
            mae = sub[col].mean()
            n_feats = len(feature_sets[s])
            if prev is None:
                print(f"{labels[s]:<38} {mae:8.2f}   ({n_feats} features)")
            else:
                delta = prev - mae
                pct = 100 * delta / prev if prev else 0
                arrow = "improves" if delta > 0 else "WORSENS"
                print(f"{labels[s]:<38} {mae:8.2f}   ({n_feats} features)  "
                      f"-> {arrow} {abs(pct):.1f}%")
            prev = mae

    print("\n" + "-" * 74)
    print("HOW TO READ THIS:")
    print("  A->B  = value of energy commodity prices")
    print("  C->D  = value of the geopolitical risk index")
    print("  D->E  = value of news sentiment  <-- this is RQ4/H3")
    print("A gain near zero for D->E means sentiment adds nothing beyond what")
    print("price and market variables already capture. That is a legitimate")
    print("negative result and should be reported, not buried.")


if __name__ == "__main__":
    master, folds = load_data()
    feature_sets, diag = build_feature_sets(master)

    print(f"Loaded {len(master)} rows, {len(folds)} folds")
    print("\nFeature availability check:")
    for k, v in diag.items():
        flag = "" if v > 0 else "   <-- MISSING, this ablation step will be uninformative"
        print(f"  {k:<28} {v:>3} columns{flag}")

    if diag["geopolitical (GPR lags)"] == 0 or diag["sentiment lags"] == 0:
        raise RuntimeError(
            "GPR and/or sentiment lags are absent. Re-run 02_preprocessing.py; "
            "the ablation cannot answer the research question without them."
        )

    print("\nFeature set sizes:")
    for k, v in feature_sets.items():
        print(f"  {k:<24} {len(v):>3}")

    print(f"\nRunning ablation (5 XGBoost fits per fold -- slower than arm 1)...")
    results, daily = run_ablation(master, folds, feature_sets)
    print(f"Completed {len(results)} folds")
    expected_folds = folds if MAX_FOLDS is None else folds.head(MAX_FOLDS)
    if SHOCK_ONLY:
        expected_folds = expected_folds[expected_folds["period"].str.startswith("SHOCK")]
    if len(results) != len(expected_folds):
        raise RuntimeError(
            f"Only {len(results)}/{len(expected_folds)} folds completed; partial results withheld"
        )

    report(results, feature_sets)

    out = PROCESSED_DIR / "ablation_results.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved per-fold results: {out}")

    if not daily.empty:
        daily_out = PROCESSED_DIR / "ablation_daily_errors.csv"
        daily.to_csv(daily_out, index=False)
        n_shock = daily["period"].str.startswith("SHOCK").sum()
        print(f"Saved per-day errors:   {daily_out} "
              f"({len(daily)} rows; {n_shock} in the shock period)")
        print("  -> 11_significance_tests.py will use these automatically, giving the")
        print(f"     Diebold-Mariano test n={n_shock} instead of n={results['period'].str.startswith('SHOCK').sum()} "
              f"for the shock period.")
