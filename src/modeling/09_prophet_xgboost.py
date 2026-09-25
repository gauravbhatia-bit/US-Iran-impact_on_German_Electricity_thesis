import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore")
for _name in ("prophet", "cmdstanpy", "prophet.models", "prophet.forecaster"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
    logging.getLogger(_name).propagate = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COL = "smard_mean"
MAX_FOLDS = None                    # e.g. 20 for a quick trial
ALLOW_SAMEDAY_FEATURES = False      # see leakage note above

# Features excluded from XGBoost regardless: the target, its own direct
# transforms, and anything that would leak the answer.
ALWAYS_EXCLUDE = {
    "date", TARGET_COL,
    "smard_min", "smard_max", "smard_std",          # same-day target aggregates = leakage
    "smard_mean_outlier_flag",                       # computed using full-series stats
    "DCOILBRENTEU_outlier_flag",
}

# Same-day (unlagged) market/news values -- excluded unless explicitly allowed
SAMEDAY_FEATURES = {
    "DCOILBRENTEU", "DEXUSEU", "ttf_eur_mwh", "gpr_daily",
    "guardian_sentiment_mean", "guardian_sentiment_min",
    "guardian_sentiment_max", "guardian_article_count",
    "wind_offshore_mwh", "wind_onshore_mwh", "solar_mwh",
    "renewable_output_mwh", "dunkelflaute_flag", "sentiment_available",
}

DAILY_CLOSE_BASES = {
    "DCOILBRENTEU", "DEXUSEU", "ttf_eur_mwh", "gpr_daily",
    "guardian_sentiment_mean", "guardian_sentiment_min",
    "guardian_sentiment_max", "guardian_article_count", "sentiment_available",
}


def load_data():
    master = pd.read_csv(PROCESSED_DIR / "master_features.csv")
    master["date"] = pd.to_datetime(master["date"])
    folds = pd.read_csv(PROCESSED_DIR / "cv_folds.csv")
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        folds[c] = pd.to_datetime(folds[c])
    return master, folds


def select_features(master):
    excluded = set(ALWAYS_EXCLUDE)
    if not ALLOW_SAMEDAY_FEATURES:
        excluded |= SAMEDAY_FEATURES
    calendar = {"day_of_week", "month", "is_weekend", "is_holiday"}
    feats = [c for c in master.columns
             if c not in excluded
             and pd.api.types.is_numeric_dtype(master[c])
             and (ALLOW_SAMEDAY_FEATURES or c in calendar
                  or "_lag" in c or "_vol" in c)
             and not (any(c.startswith(base + "_lag") for base in DAILY_CLOSE_BASES)
                      and c.endswith("_lag1"))]
    return feats


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if len(actual) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "n": 0}
    err = actual - predicted
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "n": len(actual)}
    # MAPE omitted deliberately: prices cross zero, percentage error explodes.


def fit_prophet(train_df):
    from prophet import Prophet
    m = Prophet(
        yearly_seasonality="auto",
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m.fit(train_df.rename(columns={"date": "ds", TARGET_COL: "y"})[["ds", "y"]])
    return m


def run_folds(master, folds, features, verbose_every=20):
    import xgboost as xgb

    results = []
    fold_iter = folds if MAX_FOLDS is None else folds.head(MAX_FOLDS)
    total = len(fold_iter)

    for i, (_, fold) in enumerate(fold_iter.iterrows(), 1):
        train = master[master["date"] <= fold["train_end"]].copy()
        test = master[(master["date"] >= fold["test_start"]) &
                      (master["date"] <= fold["test_end"])].copy()
        if len(train) < 60 or test.empty:
            continue

        # --- ARM 1a: Prophet alone ---
        try:
            m = fit_prophet(train)
        except Exception as e:
            print(f"  [fold {fold['fold_id']}] Prophet failed: {e}")
            continue

        future_test = test[["date"]].rename(columns={"date": "ds"})
        prophet_test_pred = m.predict(future_test)["yhat"].values

        # --- ARM 1b: XGBoost on Prophet's in-sample residuals ---
        future_train = train[["date"]].rename(columns={"date": "ds"})
        prophet_train_pred = m.predict(future_train)["yhat"].values
        train_resid = train[TARGET_COL].values - prophet_train_pred

        X_train = train[features]
        X_test = test[features]

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        # XGBoost handles NaN natively -- important, since sentiment is
        # genuinely absent before Sept 2025 and must NOT be imputed.
        model.fit(X_train, train_resid)
        resid_pred = model.predict(X_test)
        hybrid_test_pred = prophet_test_pred + resid_pred

        actual = test[TARGET_COL].values
        m_prophet = metrics(actual, prophet_test_pred)
        m_hybrid = metrics(actual, hybrid_test_pred)

        results.append({
            "fold_id": fold["fold_id"], "period": fold["period"],
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "prophet_MAE": m_prophet["MAE"], "prophet_RMSE": m_prophet["RMSE"],
            "hybrid_MAE": m_hybrid["MAE"], "hybrid_RMSE": m_hybrid["RMSE"],
            "n_test": m_prophet["n"],
        })

        if verbose_every and i % verbose_every == 0:
            print(f"  ...{i}/{total} folds done")

    return pd.DataFrame(results)


def report(results):
    print("\n" + "=" * 78)
    print("ARM 1 RESULTS BY PERIOD  (lower = better)")
    print("=" * 78)

    for period in sorted(results["period"].dropna().unique()):
        sub = results[results["period"] == period]
        if sub.empty:
            continue
        p_mae, h_mae = sub["prophet_MAE"].mean(), sub["hybrid_MAE"].mean()
        p_rmse = np.sqrt(np.mean(np.square(sub["prophet_RMSE"])))
        h_rmse = np.sqrt(np.mean(np.square(sub["hybrid_RMSE"])))
        gain = 100 * (p_mae - h_mae) / p_mae if p_mae else np.nan

        print(f"\n{period}  ({len(sub)} folds)")
        print(f"  Prophet alone            MAE {p_mae:8.2f}   RMSE {p_rmse:8.2f}")
        print(f"  Prophet + XGBoost hybrid MAE {h_mae:8.2f}   RMSE {h_rmse:8.2f}")
        if np.isfinite(gain):
            verdict = "IMPROVES on" if gain > 0 else "is WORSE than"
            print(f"  -> Adding conflict features {verdict} Prophet alone "
                  f"by {abs(gain):.1f}% on MAE")

    print("\n" + "-" * 78)
    print("The SHOCK_2022 and SHOCK_2026 lines answer the research question:")
    print("if the hybrid beats Prophet alone there, the full feature set carries")
    print("information that price history does not. If it does not, that is a")
    print("genuine negative result and should be reported as such.")
    print("\nCompare these numbers against chronos_zeroshot_results.csv (same folds).")


if __name__ == "__main__":
    master, folds = load_data()
    features = select_features(master)

    print(f"Loaded {len(master)} rows, {len(folds)} CV folds")
    print(f"Target: {TARGET_COL}")
    print(f"Same-day features allowed: {ALLOW_SAMEDAY_FEATURES} "
          f"({'NOWCAST mode' if ALLOW_SAMEDAY_FEATURES else 'true FORECAST mode'})")
    print(f"XGBoost features used: {len(features)}")
    print(f"  e.g. {features[:6]}{' ...' if len(features) > 6 else ''}")

    print(f"\nRunning folds (Prophet refits each fold -- this takes a while)...")
    results = run_folds(master, folds, features)
    print(f"Completed {len(results)} folds")
    expected = len(folds if MAX_FOLDS is None else folds.head(MAX_FOLDS))
    if len(results) != expected:
        raise RuntimeError(
            f"Only {len(results)}/{expected} folds completed; partial results withheld"
        )

    report(results)

    out = PROCESSED_DIR / "prophet_xgboost_results.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved per-fold results: {out}")
