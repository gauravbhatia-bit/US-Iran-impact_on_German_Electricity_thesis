from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DAILY_PATH = RESULTS / "rq4_eua_block_bootstrap_daily.csv"
OUTPUT_PATH = RESULTS / "rq4_eua_block_length_sensitivity.csv"
BLOCK_LENGTHS_FOLDS = (1, 2, 4, 8)
N_BOOT = 5000


def load_bootstrap_module():
    path = ROOT / "src" / "modeling" / "23_rq4_eua_block_bootstrap.py"
    spec = importlib.util.spec_from_file_location("rq4_eua_bootstrap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not DAILY_PATH.exists() or DAILY_PATH.stat().st_size == 0:
        raise RuntimeError(
            f"Missing paired losses at {DAILY_PATH}; run 23_rq4_eua_block_bootstrap.py first"
        )
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    module = load_bootstrap_module()
    rows: list[dict[str, object]] = []
    groups = {
        "all_folds": pd.Series(True, index=daily.index),
        "shock_folds": daily["period"].astype(str).str.startswith("SHOCK"),
    }
    for block_length in BLOCK_LENGTHS_FOLDS:
        for label, mask in groups.items():
            subset = daily.loc[mask].copy()
            for prefix in ["B", "D"]:
                delta = (
                    subset[f"{prefix}_no_EUA_abs_error"].to_numpy()
                    - subset[f"{prefix}_with_EUA_abs_error"].to_numpy()
                )
                random_seed = (
                    module.RANDOM_SEED
                    if block_length == module.BLOCK_LENGTH_FOLDS
                    else module.RANDOM_SEED + block_length
                )
                summary = module.bootstrap_summary(
                    delta,
                    subset["date"],
                    label,
                    block_length_folds=block_length,
                    n_boot=N_BOOT,
                    random_seed=random_seed,
                )
                summary["nested_set"] = prefix
                summary["mean_no_eua_mae"] = float(
                    subset[f"{prefix}_no_EUA_abs_error"].mean()
                )
                summary["mean_with_eua_mae"] = float(
                    subset[f"{prefix}_with_EUA_abs_error"].mean()
                )
                rows.append(summary)
    output = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(output.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
