import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROCESSED_DIR = "data/processed"
DIRECTION_COLOR = {"up": "#d62728", "down": "#2ca02c", "ambiguous": "#7f7f7f"}


def find_event_table():
    for path in ["event_table.csv", "data/raw/event_table.csv", "data/event_table.csv"]:
        if os.path.exists(path):
            return path
    return None


def load_data():
    master = pd.read_csv(f"{PROCESSED_DIR}/master_features.csv")
    master["date"] = pd.to_datetime(master["date"])

    event_path = find_event_table()
    if event_path is None:
        print("[warn] event_table.csv not found in the working directory, data/raw/, "
              "or data/ -- place it in one of these and re-run for the event-alignment plots.")
        events = None
    else:
        events = pd.read_csv(event_path)
        events["date"] = pd.to_datetime(events["date"])
        print(f"Loaded {len(events)} events from {event_path}")

    return master, events


# PART A: EVENT ALIGNMENT PLOTS

def plot_series_with_events(master, events, cols, labels, date_range, title, out_path):
    sub = master[(master["date"] >= date_range[0]) & (master["date"] <= date_range[1])]
    if events is not None:
        ev_sub = events[(events["date"] >= date_range[0]) & (events["date"] <= date_range[1])]
    else:
        ev_sub = pd.DataFrame()

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col, label in zip(axes, cols, labels):
        if col not in sub.columns:
            ax.text(0.5, 0.5, f"[column '{col}' not found]", ha="center", va="center")
            continue
        ax.plot(sub["date"], sub[col], linewidth=0.9, color="#1f77b4")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)

        for _, ev in ev_sub.iterrows():
            color = DIRECTION_COLOR.get(ev.get("expected_price_direction", "ambiguous"), "#7f7f7f")
            ax.axvline(ev["date"], color=color, linewidth=1, alpha=0.6, linestyle="--")

    # Event labels only on the top subplot, to avoid repeating 15x
    if not ev_sub.empty:
        for _, ev in ev_sub.iterrows():
            color = DIRECTION_COLOR.get(ev.get("expected_price_direction", "ambiguous"), "#7f7f7f")
            axes[0].annotate(
                str(ev.get("category", ""))[:14], xy=(ev["date"], 1.02), xycoords=("data", "axes fraction"),
                rotation=60, fontsize=6.5, color=color, ha="left", va="bottom"
            )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.suptitle(title, fontsize=12, y=1.02 if n > 1 else 1.08)

    legend_handles = [plt.Line2D([0], [0], color=c, linestyle="--", label=d)
                       for d, c in DIRECTION_COLOR.items()]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, title="expected direction",
               bbox_to_anchor=(0.99, 1.0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# PART B: FULL DATASET OVERVIEW

def describe_column(col: str) -> str:
    base_descriptions = {
        "date": "calendar date",
        "smard_mean": "German day-ahead electricity price, daily mean (EUR/MWh)",
        "smard_min": "German day-ahead electricity price, daily minimum (EUR/MWh)",
        "smard_max": "German day-ahead electricity price, daily maximum (EUR/MWh)",
        "smard_std": "German day-ahead electricity price, daily std dev within the day",
        "DCOILBRENTEU": "Brent crude oil price (USD/barrel)",
        "DEXUSEU": "EUR/USD exchange rate",
        "ttf_eur_mwh": "Dutch TTF natural gas price (EUR/MWh)",
        "gpr_daily": "Geopolitical Risk index, daily (Caldara & Iacoviello)",
        "renewable_output_mwh": "combined wind offshore + onshore + solar output (MWh)",
        "wind_offshore_mwh": "offshore wind generation, Germany (MWh/day)",
        "wind_onshore_mwh": "onshore wind generation, Germany (MWh/day)",
        "solar_mwh": "solar (photovoltaic) generation, Germany (MWh/day)",
        "dunkelflaute_flag": "1 if renewable output was in the bottom 15% for that period (weather-driven low-output day)",
        "guardian_sentiment_mean": "mean VADER sentiment of Guardian articles that day (Iran/Hormuz-scoped, -1 to +1)",
        "guardian_sentiment_min": "most negative article sentiment that day",
        "guardian_sentiment_max": "most positive article sentiment that day",
        "guardian_article_count": "number of clean (non-noise) Guardian articles that day",
        "sentiment_available": "1 if Guardian sentiment exists for this date, 0 if outside the Sept 2025+ coverage window",
        "day_of_week": "0=Monday ... 6=Sunday",
        "month": "calendar month, 1-12",
        "is_weekend": "1 if Saturday/Sunday",
        "is_holiday": "1 if a German public holiday",
    }
    if col in base_descriptions:
        return base_descriptions[col]
    if "_lag" in col:
        base, n = col.rsplit("_lag", 1)
        return f"value of '{base}' from {n} day(s) earlier"
    if "_vol7d" in col:
        return f"7-day rolling volatility (std dev) of '{col.replace('_vol7d','')}'"
    if "_vol30d" in col:
        return f"30-day rolling volatility (std dev) of '{col.replace('_vol30d','')}'"
    if "_outlier_flag" in col:
        return f"1 if '{col.replace('_outlier_flag','')}' was a statistical outlier that day (z-score > 3)"
    return "(no description available -- check column name)"


def build_data_dictionary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in master.columns:
        n_missing = master[col].isna().sum()
        pct_missing = 100 * n_missing / len(master)
        is_numeric = pd.api.types.is_numeric_dtype(master[col])
        rows.append({
            "column": col,
            "description": describe_column(col),
            "dtype": str(master[col].dtype),
            "pct_missing": round(pct_missing, 1),
            "min": round(master[col].min(), 3) if is_numeric else "",
            "max": round(master[col].max(), 3) if is_numeric else "",
            "mean": round(master[col].mean(), 3) if is_numeric else "",
        })
    return pd.DataFrame(rows)


# MAIN

if __name__ == "__main__":
    master, events = load_data()
    print(f"Loaded master_features.csv: {master.shape[0]} rows x {master.shape[1]} columns\n")

    # -------------------- PART A --------------------
    print("=== PART A: EVENT ALIGNMENT PLOTS ===\n")

    core_cols = [c for c in ["smard_mean", "DCOILBRENTEU", "ttf_eur_mwh"] if c in master.columns]
    core_labels = ["SMARD price (EUR/MWh)", "Brent (USD/bbl)", "TTF gas (EUR/MWh)"]
    core_labels = core_labels[:len(core_cols)]

    full_range = (master["date"].min(), master["date"].max())
    plot_series_with_events(master, events, core_cols, core_labels, full_range,
                             "Full range: price series vs. conflict events",
                             f"{PROCESSED_DIR}/event_alignment_full_range.png")

    conflict_start = pd.Timestamp("2026-02-01")
    conflict_end = master["date"].max()
    if conflict_start < master["date"].max():
        plot_series_with_events(master, events, core_cols, core_labels, (conflict_start, conflict_end),
                                 "Zoomed: Feb 2026 onward (the actual conflict window)",
                                 f"{PROCESSED_DIR}/event_alignment_conflict_window.png")

    # -------------------- PART B --------------------
    print("\n=== PART B: FULL DATASET OVERVIEW ===\n")

    data_dict = build_data_dictionary(master)
    print(data_dict.to_string(index=False))
    data_dict.to_csv(f"{PROCESSED_DIR}/data_dictionary.csv", index=False)
    print(f"\nSaved data dictionary: {PROCESSED_DIR}/data_dictionary.csv")

    print("\n--- Sample row: BEFORE the conflict (2024-06-15 or nearest) ---")
    before = master.iloc[(master["date"] - pd.Timestamp("2024-06-15")).abs().argsort()[:1]]
    print(before.T.to_string())

    print("\n--- Sample row: DURING the conflict (2026-03-15 or nearest) ---")
    during = master.iloc[(master["date"] - pd.Timestamp("2026-03-15")).abs().argsort()[:1]]
    print(during.T.to_string())

    master.to_csv(f"{PROCESSED_DIR}/master_features_preview.csv", index=False)
    print(f"\nFull dataset also re-saved (unchanged) at "
          f"{PROCESSED_DIR}/master_features_preview.csv for easy viewing "
          f"in Excel/Sheets if you want to scroll through it yourself.")
