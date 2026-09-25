import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COL = "smard_mean"
CONTEXT_DAYS = 365      # how much price history Chronos sees per forecast
MODEL_NAME = "amazon/chronos-bolt-base"   # bolt = faster/lighter than chronos-t5
MAX_FOLDS = None        # set to an int (e.g. 20) for a quick trial run


def load_data():
    master = pd.read_csv(PROCESSED_DIR / "master_features.csv")
    master["date"] = pd.to_datetime(master["date"])
    folds = pd.read_csv(PROCESSED_DIR / "cv_folds.csv")
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        folds[c] = pd.to_datetime(folds[c])
    return master, folds


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if len(actual) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "n": 0}
    err = actual - predicted
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "n": len(actual),
    }
    # NOTE: MAPE deliberately omitted -- German electricity prices cross zero
    # and go negative, so percentage error is undefined/explosive here.


def naive_persistence_forecast(history_values, horizon):
    return np.repeat(history_values[-1], horizon)


def run_chronos_folds(master, folds, pipeline=None, forecast_fn=None):
    results = []
    fold_iter = folds if MAX_FOLDS is None else folds.head(MAX_FOLDS)

    for _, fold in fold_iter.iterrows():
        hist = master.loc[master["date"] <= fold["train_end"], TARGET_COL].dropna().values
        if len(hist) < 30:
            continue
        context = hist[-CONTEXT_DAYS:]

        test_mask = (master["date"] >= fold["test_start"]) & (master["date"] <= fold["test_end"])
        actual = master.loc[test_mask, TARGET_COL].values
        horizon = len(actual)
        if horizon == 0:
            continue

        # --- Chronos forecast ---
        if forecast_fn is not None:
            chronos_pred = forecast_fn(context, horizon)
        else:
            import torch
            quantiles, mean = pipeline.predict_quantiles(
                torch.tensor(context, dtype=torch.float32),
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            chronos_pred = mean[0].numpy()
            q_lo = quantiles[0, :, 0].numpy()
            q_hi = quantiles[0, :, 2].numpy()

        naive_pred = naive_persistence_forecast(context, horizon)

        m_chronos = metrics(actual, chronos_pred)
        m_naive = metrics(actual, naive_pred)

        results.append({
            "fold_id": fold["fold_id"], "period": fold["period"],
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "chronos_MAE": m_chronos["MAE"], "chronos_RMSE": m_chronos["RMSE"],
            "naive_MAE": m_naive["MAE"], "naive_RMSE": m_naive["RMSE"],
            "n_test": m_chronos["n"],
        })

    return pd.DataFrame(results)


def report(results: pd.DataFrame):
    print("\n" + "=" * 78)
    print("RESULTS BY PERIOD  (lower = better)")
    print("=" * 78)

    for period in sorted(results["period"].dropna().unique()):
        sub = results[results["period"] == period]
        if sub.empty:
            continue
        c_mae, n_mae = sub["chronos_MAE"].mean(), sub["naive_MAE"].mean()
        c_rmse = np.sqrt(np.mean(np.square(sub["chronos_RMSE"])))
        n_rmse = np.sqrt(np.mean(np.square(sub["naive_RMSE"])))
        improvement = 100 * (n_mae - c_mae) / n_mae if n_mae else np.nan

        print(f"\n{period}  ({len(sub)} folds)")
        print(f"  Chronos (zero-shot)   MAE {c_mae:8.2f}   RMSE {c_rmse:8.2f}")
        print(f"  Naive persistence     MAE {n_mae:8.2f}   RMSE {n_rmse:8.2f}")
        if np.isfinite(improvement):
            verdict = "BEATS naive" if improvement > 0 else "LOSES to naive"
            print(f"  -> Chronos {verdict} by {abs(improvement):.1f}% on MAE")

    b = results[results["period"] == "BASELINE"]["chronos_MAE"].mean()
    for shock in [p for p in results["period"].unique() if str(p).startswith("SHOCK")]:
        s = results[results["period"] == shock]["chronos_MAE"].mean()
        if np.isfinite(b) and np.isfinite(s):
            print(f"\nDegradation BASELINE -> {shock}: {b:.2f} -> {s:.2f} "
                  f"({100*(s-b)/b:+.1f}%)")


if __name__ == "__main__":
    master, folds = load_data()
    print(f"Loaded {len(master)} rows, {len(folds)} CV folds")
    print(f"Target: {TARGET_COL} | Context: {CONTEXT_DAYS}d | Model: {MODEL_NAME}")

    print("\nLoading Chronos (first run downloads weights from HuggingFace)...")
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    print(f"  Loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    print(f"\nRunning {len(folds)} folds...")
    results = run_chronos_folds(master, folds, pipeline=pipeline)

    report(results)

    out = PROCESSED_DIR / "chronos_zeroshot_results.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved per-fold results: {out}")
    print("\nNext: run your Prophet/XGBoost model on these SAME folds and compare")
    print("against both columns. Your model must beat naive to be useful at all,")
    print("and should beat Chronos to justify the added feature complexity.")
