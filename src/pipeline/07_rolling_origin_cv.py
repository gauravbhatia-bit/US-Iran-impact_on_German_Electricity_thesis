import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANALYSIS_START = "2021-01-01"
SHOCK_PERIODS = {
    "SHOCK_2022": (pd.Timestamp("2022-02-24"), pd.Timestamp("2023-02-23")),
    "SHOCK_2026": (pd.Timestamp("2026-02-28"), pd.Timestamp.max.normalize()),
}

MIN_TRAIN_DAYS = 365     # Prophet needs at least a full year for seasonality
HORIZON_DAYS = 1         # true one-step-ahead forecast; prevents observed-test-lag leakage
STEP_DAYS = 7            # how far the origin moves between folds (weekly)


def classify_period(test_start, test_end):
    matches = []
    for label, (period_start, period_end) in SHOCK_PERIODS.items():
        if test_start >= period_start and test_end <= period_end:
            matches.append(label)
        elif test_start <= period_end and test_end >= period_start:
            return f"STRADDLES_{label.removeprefix('SHOCK_')}"
    return matches[0] if matches else "BASELINE"


def build_folds(dates: pd.Series, min_train_days=MIN_TRAIN_DAYS,
                 horizon_days=HORIZON_DAYS, step_days=STEP_DAYS):
    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    n = len(dates)
    if dates.duplicated().any():
        raise ValueError("CV dates contain duplicates")
    if len(dates) > 1 and not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("CV requires a complete daily date index")

    folds = []
    origin_idx = min_train_days
    fold_id = 1
    while origin_idx + horizon_days <= n:
        train_dates = dates.iloc[:origin_idx]
        test_dates = dates.iloc[origin_idx: origin_idx + horizon_days]

        test_start, test_end = test_dates.iloc[0], test_dates.iloc[-1]
        period = classify_period(test_start, test_end)

        folds.append({
            "fold_id": fold_id,
            "train_start": train_dates.iloc[0], "train_end": train_dates.iloc[-1],
            "n_train_days": len(train_dates),
            "test_start": test_start, "test_end": test_end,
            "n_test_days": len(test_dates),
            "period": period,
        })
        fold_id += 1
        origin_idx += step_days

    return pd.DataFrame(folds)


def plot_fold_timeline(folds: pd.DataFrame, out_path: str, max_folds_shown=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    period_colors = {
        "BASELINE": "#1f77b4", "SHOCK_2022": "#9467bd",
        "SHOCK_2026": "#d62728", "STRADDLES_2022": "#e0a300",
        "STRADDLES_2026": "#ff7f0e",
    }

    # If there are many folds, show every Nth one so the plot stays legible
    step = max(1, len(folds) // max_folds_shown)
    shown = folds.iloc[::step].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, max(4, len(shown) * 0.18)))
    for i, row in shown.iterrows():
        ax.barh(i, (row["train_end"] - row["train_start"]).days, left=row["train_start"],
                height=0.6, color="#d9d9d9", label="train" if i == 0 else "")
        ax.barh(i, (row["test_end"] - row["test_start"]).days + 1, left=row["test_start"],
                height=0.6, color=period_colors[row["period"]],
                label=row["period"] if row["period"] not in ax.get_legend_handles_labels()[1] else "")

    for label, (start, _) in SHOCK_PERIODS.items():
        ax.axvline(start, color="black", linewidth=1.0, linestyle="--")
        ax.text(start, len(shown), label, fontsize=8, rotation=90,
                va="bottom", ha="right")

    ax.set_yticks([])
    ax.set_xlabel("date")
    ax.set_title(f"Rolling-origin CV fold scheme ({len(folds)} total folds, "
                 f"{len(shown)} shown for legibility)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    master = pd.read_csv(f"{PROCESSED_DIR}/master_features.csv")
    master["date"] = pd.to_datetime(master["date"])

    folds = build_folds(master["date"])

    print(f"Total folds: {len(folds)}")
    for period, count in folds["period"].value_counts().sort_index().items():
        print(f"  {period}: {count} one-day test folds")
    print(f"\nFirst fold:  train {folds.iloc[0]['train_start'].date()} -> "
          f"{folds.iloc[0]['train_end'].date()} ({folds.iloc[0]['n_train_days']} days), "
          f"test {folds.iloc[0]['test_start'].date()} -> {folds.iloc[0]['test_end'].date()}")
    print(f"Last fold:   train {folds.iloc[-1]['train_start'].date()} -> "
          f"{folds.iloc[-1]['train_end'].date()} ({folds.iloc[-1]['n_train_days']} days), "
          f"test {folds.iloc[-1]['test_start'].date()} -> {folds.iloc[-1]['test_end'].date()}")

    print("\nEach origin evaluates exactly one unseen day. Origins are sampled weekly; "
          "test windows do not overlap and no test-day actual can leak through a lag.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(PROCESSED_DIR / "cv_folds.csv", index=False)
    print(f"\nSaved fold definitions: {PROCESSED_DIR / 'cv_folds.csv'}")

    plot_fold_timeline(folds, PROCESSED_DIR / "cv_fold_timeline.png")

    print("\nNext: for each fold, train on master[master.date <= train_end] and "
          "evaluate on master[(master.date >= test_start) & (master.date <= test_end)]. "
          "Report metrics separately for BASELINE, SHOCK_2022, and SHOCK_2026; "
          "do not blend them into one overall score.")
