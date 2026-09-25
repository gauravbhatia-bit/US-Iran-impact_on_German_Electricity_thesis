from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BLOCK_LENGTH_FOLDS = 4
N_BOOT = 5000
RANDOM_SEED = 42


def load_rq4_module():
    import importlib.util

    path = ROOT / "src" / "modeling" / "22_rq4_compact_features.py"
    spec = importlib.util.spec_from_file_location("rq4_compact", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_sets(frame: pd.DataFrame, module) -> dict[str, list[str]]:
    calendar = ["day_of_week", "month", "is_weekend", "is_holiday"]
    baseline = [f"price_lag{i}" for i in range(1, 8)] + calendar
    no_eua_fundamentals = [
        "forecast_residual_load",
        "ttf_lag2",
        "expected_gas_marginal_cost_no_eua",
    ]
    with_eua_fundamentals = [
        "forecast_residual_load",
        "ttf_lag2",
        "eua_eur_tco2_lag2",
        "expected_gas_marginal_cost",
    ]
    extra = ["traffic_shortfall_lag2", "ttf_x_high_residual", "traffic_x_ttf"]
    return {
        "B_no_EUA": baseline + no_eua_fundamentals,
        "B_with_EUA": baseline + with_eua_fundamentals,
        "D_no_EUA": baseline + no_eua_fundamentals + extra,
        "D_with_EUA": baseline + with_eua_fundamentals + extra,
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], target: str) -> np.ndarray:
    fit_columns = [column for column in columns if train[column].notna().any()]
    if not fit_columns:
        raise RuntimeError("A requested RQ4 specification has no available features")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=60,
            learning_rate=0.07,
            max_leaf_nodes=11,
            l2_regularization=1.0,
            random_state=42,
        ),
    )
    model.fit(train[fit_columns], train[target])
    return model.predict(test[fit_columns])


def build_paired_errors(module) -> pd.DataFrame:
    frame, diagnostics = module.load_frame()
    frame["expected_gas_marginal_cost_no_eua"] = frame["ttf_lag2"] / module.GAS_EFFICIENCY
    sets = feature_sets(frame, module)
    folds = pd.read_csv(PROCESSED / "cv_folds.csv")
    for column in ["train_end", "test_start", "test_end"]:
        folds[column] = pd.to_datetime(folds[column]).dt.normalize()

    rows: list[dict[str, object]] = []
    for _, fold in folds.iterrows():
        train = frame[frame["date"] <= fold["train_end"]].copy()
        test = frame[
            (frame["date"] >= fold["test_start"])
            & (frame["date"] <= fold["test_end"])
        ].copy()
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
        row: dict[str, object] = {
            "fold_id": fold["fold_id"],
            "period": fold["period"],
            "date": test["date"].iloc[0],
        }
        target = module.TARGET
        actual = test[target].to_numpy(dtype=float)
        for name, columns in sets.items():
            prediction = fit_predict(train, test, columns, target)
            row[f"{name}_abs_error"] = float(np.mean(np.abs(actual - prediction)))
        rows.append(row)
    if not rows:
        raise RuntimeError("No paired RQ4 folds were completed")
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    result.to_csv(RESULTS / "rq4_eua_block_bootstrap_daily.csv", index=False)
    (RESULTS / "rq4_eua_block_bootstrap_input_diagnostics.json").write_text(
        json.dumps(
            {
                "eua_source": diagnostics.get("eua_source"),
                "eua_available_rows": diagnostics.get("eua_available_rows"),
                "n_completed_folds": len(result),
                "n_shock_folds": int(result["period"].astype(str).str.startswith("SHOCK").sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def contiguous_blocks(dates: pd.Series, block_length_folds: int) -> list[np.ndarray]:
    if block_length_folds < 1:
        raise ValueError("block_length_folds must be at least one")
    ordered = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    if len(ordered) == 1:
        return [np.asarray([0], dtype=int)]
    gaps = ordered.diff().dropna().dt.days.to_numpy(dtype=float)
    positive = gaps[gaps > 0]
    cadence = float(np.median(positive)) if len(positive) else 1.0
    # A gap more than 1.5 times the normal fold cadence starts a new segment.
    max_contiguous_gap = max(1.0, 1.5 * cadence)
    segments: list[list[int]] = []
    current: list[int] = []
    for index, value in enumerate(ordered):
        if current and (value - ordered.iloc[current[-1]]).days > max_contiguous_gap:
            segments.append(current)
            current = []
        current.append(index)
    if current:
        segments.append(current)

    blocks: list[np.ndarray] = []
    for segment in segments:
        if len(segment) <= block_length_folds:
            blocks.append(np.asarray(segment, dtype=int))
        else:
            for start in range(0, len(segment) - block_length_folds + 1):
                blocks.append(np.asarray(segment[start : start + block_length_folds], dtype=int))
    if not blocks:
        raise RuntimeError("No contiguous bootstrap blocks were available")
    return blocks


def bootstrap_summary(
    values: np.ndarray,
    dates: pd.Series,
    group: str,
    block_length_folds: int = BLOCK_LENGTH_FOLDS,
    n_boot: int = N_BOOT,
    random_seed: int = RANDOM_SEED,
) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    blocks = contiguous_blocks(dates, block_length_folds)
    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        selected: list[int] = []
        while len(selected) < len(values):
            selected.extend(blocks[int(rng.integers(len(blocks)))].tolist())
        bootstrap_means[iteration] = values[np.asarray(selected[: len(values)], dtype=int)].mean()

    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    # Re-centre the paired loss differences to obtain an approximate null
    # distribution for H0: mean difference == 0.
    centered = values - observed
    null_means = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        selected = []
        while len(selected) < len(values):
            selected.extend(blocks[int(rng.integers(len(blocks)))].tolist())
        null_means[iteration] = centered[np.asarray(selected[: len(values)], dtype=int)].mean()
    p_one_sided = float(np.mean(null_means >= observed))
    p_two_sided = float(2 * min(np.mean(null_means >= abs(observed)), np.mean(null_means <= -abs(observed))))
    return {
        "comparison": "no_EUA_error_minus_with_EUA_error",
        "group": group,
        "n_observations": int(len(values)),
        "block_length_folds": int(block_length_folds),
        "nominal_block_length_days": int(block_length_folds * 7),
        "block_length_days": int(block_length_folds * 7),
        "n_blocks": len(blocks),
        "n_bootstrap_replicates": int(n_boot),
        "observed_delta_mae": observed,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "p_value_one_sided_improvement": p_one_sided,
        "p_value_two_sided": p_two_sided,
        "eua_improves_if_delta_positive": True,
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def main() -> None:
    module = load_rq4_module()
    daily = build_paired_errors(module)
    summaries: list[dict[str, object]] = []
    groups = {
        "all_folds": pd.Series(True, index=daily.index),
        "shock_folds": daily["period"].astype(str).str.startswith("SHOCK"),
    }
    for label, mask in groups.items():
        subset = daily.loc[mask].copy()
        for prefix in ["B", "D"]:
            delta = subset[f"{prefix}_no_EUA_abs_error"].to_numpy() - subset[f"{prefix}_with_EUA_abs_error"].to_numpy()
            summary = bootstrap_summary(delta, subset["date"], label)
            summary["nested_set"] = prefix
            summary["mean_no_eua_mae"] = float(subset[f"{prefix}_no_EUA_abs_error"].mean())
            summary["mean_with_eua_mae"] = float(subset[f"{prefix}_with_EUA_abs_error"].mean())
            summaries.append(summary)
    output = pd.DataFrame(summaries)
    output.to_csv(RESULTS / "rq4_eua_block_bootstrap.csv", index=False)
    print(output.to_string(index=False))
    print(f"Saved: {RESULTS / 'rq4_eua_block_bootstrap.csv'}")
    print(f"Saved: {RESULTS / 'rq4_eua_block_bootstrap_daily.csv'}")


if __name__ == "__main__":
    main()
