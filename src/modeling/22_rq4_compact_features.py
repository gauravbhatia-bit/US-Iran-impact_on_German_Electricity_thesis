from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_PERMUTATION_REPEATS = 20

TARGET = "smard_mean"
CONFLICT_START = pd.Timestamp("2026-02-28")
GAS_EFFICIENCY = 0.55
GAS_EMISSIONS_T_PER_MWH = 0.202
DAILY_CLOSE_SAFE_LAG = 2


def _first_existing(names: list[str]) -> Path | None:
    for name in names:
        path = RAW / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def load_market_file(candidates: list[str], value_name: str) -> tuple[pd.DataFrame, str | None]:
    path = _first_existing(candidates)
    if path is None:
        return pd.DataFrame(columns=["date", value_name]), None
    raw = pd.read_csv(path)
    date_col = next((c for c in ["date", "Date", "observation_date", "datetime", "delivery_date"] if c in raw.columns), None)
    value_col = next((c for c in [value_name, "price", "Price", "value", "close", "Close"] if c in raw.columns), None)
    if date_col is None or value_col is None:
        raise RuntimeError(f"{path.name} must contain a date column and a price/value column")
    out = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="coerce"),
                       value_name: pd.to_numeric(raw[value_col], errors="coerce")})
    out = out.dropna().assign(date=lambda x: x["date"].dt.tz_localize(None).dt.normalize())
    out = out.groupby("date", as_index=False)[value_name].mean()
    return out, path.name


def load_frame() -> tuple[pd.DataFrame, dict]:
    master = pd.read_csv(PROCESSED / "master_features.csv", parse_dates=["date"])
    master["date"] = master["date"].dt.normalize()
    if TARGET not in master or "ttf_eur_mwh" not in master:
        raise RuntimeError("master_features.csv must contain smard_mean and ttf_eur_mwh")
    system = pd.read_csv(RAW / "energy_charts_german_system.csv")
    system["datetime_utc"] = pd.to_datetime(system["datetime_utc"], utc=True, errors="raise")
    system["date"] = system["datetime_utc"].dt.tz_convert("Europe/Berlin").dt.normalize().dt.tz_localize(None)
    if "Residual load" not in system:
        raise RuntimeError("German system data lacks Residual load")
    daily_system = system.groupby("date", as_index=False).agg(
        residual_load_actual=("Residual load", "mean"),
        load_actual=("Load", "mean"),
        gas_generation=("Fossil gas", "mean"),
    )
    frame = master.merge(daily_system, on="date", how="left")
    # Seven-day seasonal-naive forecast: known on date t from the previous
    # week's same weekday.  It is deliberately not the realised t-day load.
    frame["forecast_residual_load"] = frame["residual_load_actual"].shift(7)
    frame["forecast_load"] = frame["load_actual"].shift(7)
    # The manual TTF export has dates but no pre-auction publication time.
    # D-2 is therefore the conservative forecast-origin lag.
    frame["ttf_lag2"] = frame["ttf_eur_mwh"].shift(DAILY_CLOSE_SAFE_LAG)
    for lag in range(1, 8):
        frame[f"price_lag{lag}"] = frame[TARGET].shift(lag)

    eua, eua_source = load_market_file(["eua_daily.csv", "eua_eex_daily.csv", "eua_price.csv"], "eua_eur_tco2")
    coal, coal_source = load_market_file(["coal_daily.csv", "coal_ara_daily.csv", "coal_price.csv"], "coal_eur_tonne")
    frame = frame.merge(eua, on="date", how="left").merge(coal, on="date", how="left")
    # Weekend/holiday prices can be carried forward, but only before lagging;
    # this does not use a future observation.
    for col in ["eua_eur_tco2", "coal_eur_tonne"]:
        observed = frame[col].notna()
        frame[col] = frame[col].ffill(limit=3)
        frame[f"{col}_lag2"] = frame[col].shift(DAILY_CLOSE_SAFE_LAG)
        last_observed = frame["date"].where(observed).ffill()
        frame[f"{col}_age_days"] = (frame["date"] - last_observed).dt.days
        frame[f"{col}_stale_flag"] = np.where(
            frame[col].notna(), (frame[f"{col}_age_days"] > 3).astype(float), np.nan
        )
    frame["expected_gas_marginal_cost"] = frame["ttf_lag2"] / GAS_EFFICIENCY
    if frame["eua_eur_tco2_lag2"].notna().any():
        frame["expected_gas_marginal_cost"] = (
            frame["ttf_lag2"] / GAS_EFFICIENCY
            + GAS_EMISSIONS_T_PER_MWH / GAS_EFFICIENCY * frame["eua_eur_tco2_lag2"]
        )

    # PortWatch is a robustness input to the compact model, not a definition
    # of the RQ2a primary event exposure.
    portwatch = RAW / "imf_portwatch_hormuz.csv"
    traffic_source = None
    if portwatch.exists():
        traffic = pd.read_csv(portwatch, parse_dates=["date"])
        traffic["date"] = traffic["date"].dt.normalize()
        baseline = traffic.loc[traffic["date"] < CONFLICT_START, "n_total"].median()
        traffic["traffic_shortfall"] = 1 - traffic["n_total"] / baseline
        frame = frame.merge(traffic[["date", "traffic_shortfall"]], on="date", how="left")
        frame["traffic_shortfall_lag2"] = frame["traffic_shortfall"].shift(DAILY_CLOSE_SAFE_LAG)
        traffic_source = "imf_portwatch_hormuz.csv"

    frame["high_residual_load"] = np.nan
    frame["ttf_x_high_residual"] = np.nan
    frame["traffic_x_ttf"] = np.nan
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["is_holiday"] = frame.get("is_holiday", 0)
    diagnostics = {
        "forecast_residual_load_source": "7-day seasonal-naive proxy from energy_charts_german_system.csv",
        "forecast_residual_load_threshold_policy": "training-fold q75",
        "eua_source": eua_source,
        "coal_source": coal_source,
        "traffic_source": traffic_source,
        "eua_available_rows": int(frame["eua_eur_tco2_lag2"].notna().sum()),
        "coal_available_rows": int(frame["coal_eur_tonne_lag2"].notna().sum()),
        "daily_close_safe_lag_days": DAILY_CLOSE_SAFE_LAG,
        "note": "EUA/coal are not imputed or synthetically generated; short market gaps are bounded and carry staleness flags.",
    }
    return frame, diagnostics


def feature_sets(frame: pd.DataFrame) -> tuple[dict[str, list[str]], dict]:
    calendar = ["day_of_week", "month", "is_weekend", "is_holiday"]
    baseline = [f"price_lag{i}" for i in range(1, 8)] + calendar
    fundamentals = ["forecast_residual_load", "ttf_lag2", "eua_eur_tco2_lag2",
                    "coal_eur_tonne_lag2", "expected_gas_marginal_cost",
                    "eua_eur_tco2_stale_flag", "coal_eur_tonne_stale_flag"]
    hormuz = ["traffic_shortfall_lag2"]
    interactions = ["ttf_x_high_residual", "traffic_x_ttf"]
    sets = {
        "A_price_calendar": baseline,
        "B_plus_fundamentals": baseline + fundamentals,
        "C_plus_hormuz": baseline + fundamentals + hormuz,
        "D_plus_interactions": baseline + fundamentals + hormuz + interactions,
    }
    available = {}
    for name, cols in sets.items():
        fold_constructed = {"ttf_x_high_residual", "traffic_x_ttf"}
        available[name] = [
            c for c in cols
            if c in frame.columns and (frame[c].notna().any() or c in fold_constructed)
        ]
    return available, {name: len(cols) for name, cols in available.items()}


def metrics(y, pred):
    err = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def run_models(frame: pd.DataFrame, folds: pd.DataFrame, sets: dict[str, list[str]]):
    rows, daily_rows = [], []
    imp_rows = []
    shock_models = []
    for _, fold in folds.iterrows():
        train = frame[frame["date"] <= fold["train_end"]].copy()
        test = frame[(frame["date"] >= fold["test_start"]) & (frame["date"] <= fold["test_end"])].copy()
        if len(train) < 100 or test.empty:
            continue
        threshold = train["forecast_residual_load"].dropna().quantile(0.75)
        if not np.isfinite(threshold):
            threshold = 0.0
        for subset in (train, test):
            subset["high_residual_load"] = (
                subset["forecast_residual_load"] >= threshold
            ).astype(float)
            subset["ttf_x_high_residual"] = (
                subset["ttf_lag2"] * subset["high_residual_load"]
            )
            subset["traffic_x_ttf"] = (
                subset.get("traffic_shortfall_lag2", np.nan) * subset["ttf_lag2"]
            )
        row = {"fold_id": fold["fold_id"], "period": fold["period"],
               "residual_load_threshold": float(threshold)}
        preds = {}
        for name, cols in sets.items():
            fit_cols = [col for col in cols if train[col].notna().any()]
            if not fit_cols:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(max_iter=60, learning_rate=0.07,
                                              max_leaf_nodes=11, l2_regularization=1.0,
                                              random_state=42),
            )
            model.fit(train[fit_cols], train[TARGET])
            pred = model.predict(test[fit_cols])
            preds[name] = pred
            mae, rmse = metrics(test[TARGET], pred)
            row[f"{name}_MAE"] = mae
            row[f"{name}_RMSE"] = rmse
            if name == "D_plus_interactions" and str(fold["period"]).startswith("SHOCK"):
                shock_models.append((model, fit_cols, test[fit_cols].copy(), test[TARGET].to_numpy()))
        if preds:
            daily = pd.DataFrame({"fold_id": fold["fold_id"], "period": fold["period"],
                                  "date": test["date"].values, "actual": test[TARGET].values})
            for name, pred in preds.items():
                daily[f"{name}_abs_err"] = np.abs(test[TARGET].values - pred)
            daily_rows.append(daily)
            rows.append(row)
    # Pooled OOS replacement importance for the final set.  This is a
    # predictive diagnostic, not a causal attribution.
    if shock_models:
        all_cols = sorted({feature for _, cols, _, _ in shock_models for feature in cols})
        for feature_index, feature in enumerate(all_cols):
            compatible = [
                (model, cols, test_x, y)
                for model, cols, test_x, y in shock_models
                if feature in cols
            ]
            if not compatible:
                continue
            baseline_losses = []
            for model, cols, test_x, y in compatible:
                baseline_losses.extend(np.abs(y - model.predict(test_x)))
            baseline = float(np.mean(baseline_losses))
            draws = []
            feature_values = pd.concat(
                [test_x[[feature]] for _, _, test_x, _ in compatible],
                ignore_index=True,
            )[feature]
            dist = pd.to_numeric(feature_values, errors="coerce").dropna().to_numpy()
            if len(dist) == 0:
                continue
            # A feature-specific seed keeps the diagnostic reproducible even
            # when the union of available columns changes across reruns.
            feature_rng = np.random.default_rng(42 + feature_index)
            for _ in range(N_PERMUTATION_REPEATS):
                losses = []
                for model, cols, test_x, y in compatible:
                    altered = test_x.copy()
                    altered[feature] = feature_rng.choice(dist, size=len(altered), replace=True)
                    losses.extend(np.abs(y - model.predict(altered)))
                draws.append(float(np.mean(losses) - baseline))
            imp_rows.append({"feature": feature, "importance_neg_mae": float(np.mean(draws)),
                             "importance_std": float(np.std(draws)),
                             "n_shock_rows": len(compatible), "baseline_shock_mae": baseline,
                             "n_permutation_repeats": N_PERMUTATION_REPEATS})
    return pd.DataFrame(rows), pd.concat(daily_rows, ignore_index=True), pd.DataFrame(imp_rows)


def write_summary(results: pd.DataFrame, importance: pd.DataFrame, diagnostics: dict, sizes: dict) -> None:
    lines = [
        "# RQ4 compact forecast-origin feature model",
        "",
        "The final model tests only economically motivated predictors: price/calendar",
        "baseline, forecast residual load, TTF, EUA and coal where auditable files are",
        "available, expected gas marginal cost, Hormuz traffic shortfall and the two",
        "pre-specified interactions.  Date-only market inputs enter with a conservative",
        "two-day lag; the",
        "residual-load forecast is a seven-day seasonal-naive proxy, not realised same-day",
        "load.  This prevents the old same-day leakage problem.",
        "",
        "## Feature availability",
        f"- EUA source: {diagnostics['eua_source'] or 'not supplied'} ({diagnostics['eua_available_rows']} lagged rows).",
        f"- Coal source: {diagnostics['coal_source'] or 'not supplied'} ({diagnostics['coal_available_rows']} lagged rows).",
        "- Missing EUA/coal prices are left missing and excluded from the corresponding fit; bounded carry-forward is accompanied by staleness flags; no synthetic prices are created.",
        "",
        "## Nested out-of-sample comparison",
        "",
        "| Set | Features | Mean MAE (all folds) | Mean MAE (shock folds) |",
        "|---|---:|---:|---:|",
    ]
    for name, n in sizes.items():
        col = f"{name}_MAE"
        if col not in results:
            continue
        all_mae = results[col].mean()
        shock = results.loc[results["period"].astype(str).str.startswith("SHOCK"), col].mean()
        lines.append(f"| {name} | {n} | {all_mae:.3f} | {shock:.3f} |")
    lines += [
        "",
        "Lower MAE is better.  A gain from C to D means the Hormuz exposure and its",
        "interactions add predictive information beyond the baseline and fuel/system",
        "variables.  It is not evidence that the feature is causal; use the RQ2a event",
        "study for the transmission claim.",
        "",
        "## Permutation importance",
        "",
        "The permutation output is an out-of-sample diagnostic for the final compact model",
        "on shock folds.  Positive values indicate that shuffling a feature worsened",
        "negative-MAE score; rankings are descriptive and should be accompanied by the",
        "nested accuracy comparison.",
        "",
        "Outputs: `rq4_compact_results.csv`, `rq4_compact_daily_errors.csv`,",
        "`rq4_compact_feature_importance.csv` and `rq4_compact_diagnostics.json`.",
    ]
    (RESULTS / "rq4_compact_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    frame, diagnostics = load_frame()
    sets, sizes = feature_sets(frame)
    folds = pd.read_csv(PROCESSED / "cv_folds.csv")
    for c in ["train_end", "test_start", "test_end"]:
        folds[c] = pd.to_datetime(folds[c]).dt.normalize()
    results, daily, importance = run_models(frame, folds, sets)
    if results.empty:
        raise RuntimeError("No valid RQ4 folds completed")
    results.to_csv(RESULTS / "rq4_compact_results.csv", index=False)
    daily.to_csv(RESULTS / "rq4_compact_daily_errors.csv", index=False)
    importance.to_csv(RESULTS / "rq4_compact_feature_importance.csv", index=False)
    diagnostics.update({"feature_counts": sizes, "n_completed_folds": int(len(results)),
                        "n_shock_folds": int(results["period"].astype(str).str.startswith("SHOCK").sum()),
                        "model": "HistGradientBoostingRegressor with fold-trained median imputation"})
    (RESULTS / "rq4_compact_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    write_summary(results, importance, diagnostics, sizes)
    print("RQ4 compact feature model complete")
    print(f"  folds: {len(results)} (shock={diagnostics['n_shock_folds']})")
    print(f"  feature counts: {sizes}")
    print(f"  summary: {RESULTS / 'rq4_compact_summary.md'}")


if __name__ == "__main__":
    main()
