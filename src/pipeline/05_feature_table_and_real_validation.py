import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

PROCESSED_DIR = "data/processed"

EVENT_COLORS = {
    "strike": "#b30000", "closure": "#b30000", "blockade": "#b30000",
    "military_campaign": "#e07b00", "military_operation": "#e07b00",
    "political_statement": "#7a4fbf", "ceasefire": "#1a7a3c",
    "de-escalation": "#1a7a3c", "diplomatic": "#1a7a3c", "market_peak": "#b30000",
}


def find_event_table():
    for path in ["event_table.csv", "data/raw/event_table.csv", "data/event_table.csv"]:
        if os.path.exists(path):
            return path
    return None


# PART A: POLISHED FEATURE TABLE

def describe_column(col: str) -> str:
    base = {
        "date": "calendar date",
        "smard_mean": "electricity price, daily mean (EUR/MWh)",
        "smard_min": "electricity price, daily min (EUR/MWh)",
        "smard_max": "electricity price, daily max (EUR/MWh)",
        "smard_std": "electricity price, intraday std dev",
        "DCOILBRENTEU": "Brent crude oil (USD/barrel)",
        "DEXUSEU": "EUR/USD exchange rate",
        "ttf_eur_mwh": "Dutch TTF natural gas (EUR/MWh)",
        "gpr_daily": "Geopolitical Risk index, daily",
        "renewable_output_mwh": "wind + solar output, combined (MWh)",
        "wind_offshore_mwh": "offshore wind generation (MWh)",
        "wind_onshore_mwh": "onshore wind generation (MWh)",
        "solar_mwh": "solar generation (MWh)",
        "dunkelflaute_flag": "1 = low renewable-output day (weather-driven)",
        "guardian_sentiment_mean": "mean news sentiment, Iran/Hormuz-scoped (-1 to +1)",
        "guardian_sentiment_min": "most negative article sentiment that day",
        "guardian_sentiment_max": "most positive article sentiment that day",
        "guardian_article_count": "count of clean (non-noise) articles that day",
        "sentiment_available": "1 = sentiment exists for this date",
        "day_of_week": "0=Mon ... 6=Sun", "month": "calendar month",
        "is_weekend": "1 = Sat/Sun", "is_holiday": "1 = German public holiday",
    }
    if col in base:
        return base[col]
    if "_lag" in col:
        b, n = col.rsplit("_lag", 1)
        return f"'{b}', {n} day(s) earlier"
    if "_vol7d" in col:
        return f"7-day rolling volatility of '{col.replace('_vol7d','')}'"
    if "_vol30d" in col:
        return f"30-day rolling volatility of '{col.replace('_vol30d','')}'"
    if "_outlier_flag" in col:
        return f"1 = '{col.replace('_outlier_flag','')}' was an outlier (z>3)"
    return "-"


def build_feature_table_image(master: pd.DataFrame, out_path: str, max_rows_per_image: int = 30):
    rows = []
    for col in master.columns:
        n_missing = master[col].isna().sum()
        pct_missing = 100 * n_missing / len(master)
        is_numeric = pd.api.types.is_numeric_dtype(master[col])
        rows.append([
            col, describe_column(col), str(master[col].dtype),
            f"{pct_missing:.1f}%",
            f"{master[col].min():.2f}" if is_numeric else "",
            f"{master[col].max():.2f}" if is_numeric else "",
            f"{master[col].mean():.2f}" if is_numeric else "",
            pct_missing,
        ])

    col_labels = ["Column", "Description", "Type", "% Missing", "Min", "Max", "Mean"]
    n_chunks = int(np.ceil(len(rows) / max_rows_per_image))
    saved_paths = []

    for chunk_i in range(n_chunks):
        chunk = rows[chunk_i * max_rows_per_image: (chunk_i + 1) * max_rows_per_image]
        fig, ax = plt.subplots(figsize=(13, max(3, len(chunk) * 0.4)))
        ax.axis("off")

        table = ax.table(
            cellText=[r[:7] for r in chunk], colLabels=col_labels,
            cellLoc="left", loc="center", colWidths=[0.16, 0.30, 0.09, 0.09, 0.09, 0.09, 0.09]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", weight="bold")

        for i, row in enumerate(chunk, start=1):
            pct = row[7]
            color = "#ffffff" if pct == 0 else (
                "#fff3cd" if pct < 25 else ("#ffe0b3" if pct < 60 else "#ffc9a3")
            )
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(color)

        suffix = f" (part {chunk_i+1}/{n_chunks})" if n_chunks > 1 else ""
        ax.set_title(f"master_features.csv -- column overview{suffix}", fontsize=11, pad=14)
        fig.tight_layout()

        path = out_path if n_chunks == 1 else out_path.replace(".png", f"_part{chunk_i+1}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)
        print(f"  Saved: {path}")

    return saved_paths


# PART B: REAL EVENT-WINDOW VALIDATION (same logic as validate_sentiment_against_events.py)

def plot_sentiment_vs_events(sentiment_df, event_df, price_df=None, save_path=None):
    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    event_df = event_df.copy()
    event_df["date"] = pd.to_datetime(event_df["date"])

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(sentiment_df["date"], sentiment_df["sentiment"], color="#2b2b2b",
              linewidth=1.2, label="Daily sentiment (VADER, custom lexicon)")
    ax1.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("Sentiment score (-1 to +1)")
    ax1.set_ylim(-1, 1)

    if price_df is not None:
        price_df = price_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"])
        ax2 = ax1.twinx()
        ax2.plot(price_df["date"], price_df["price"], color="#1f77b4",
                  linewidth=1.0, alpha=0.6, label="Price (secondary axis)")
        ax2.set_ylabel("Electricity price EUR/MWh (secondary axis)", color="#1f77b4")

    for _, row in event_df.iterrows():
        color = EVENT_COLORS.get(row["category"], "#555555")
        ax1.axvline(row["date"], color=color, linewidth=0.9, alpha=0.7, linestyle=":")

    present_categories = event_df["category"].unique()
    legend_elements = [Line2D([0], [0], color=EVENT_COLORS.get(c, "#555555"), linestyle=":", label=c)
                        for c in present_categories]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8, ncol=3,
               title="Event category (dotted lines)")

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax1.set_title("REAL sentiment vs. curated conflict events (Sept 2025 onward)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    return fig


def compute_event_window_check(sentiment_df, event_df, window_days=2, baseline_days=21):
    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    event_df = event_df.copy()
    event_df["date"] = pd.to_datetime(event_df["date"])

    results = []
    for _, ev in event_df.iterrows():
        window = sentiment_df[
            (sentiment_df["date"] >= ev["date"] - pd.Timedelta(days=window_days))
            & (sentiment_df["date"] <= ev["date"] + pd.Timedelta(days=window_days))
        ]
        baseline = sentiment_df[
            (sentiment_df["date"] >= ev["date"] - pd.Timedelta(days=window_days + baseline_days))
            & (sentiment_df["date"] < ev["date"] - pd.Timedelta(days=window_days))
        ]
        if window.empty or baseline.empty:
            results.append({"event": ev["event"], "date": ev["date"],
                             "avg_sentiment_in_window": None, "baseline_avg": None,
                             "match": None, "note": "insufficient data (window or baseline empty)"})
            continue

        avg_sent = window["sentiment"].mean()
        baseline_avg = baseline["sentiment"].mean()
        expected_negative = ev["category"] in ("strike", "closure", "blockade", "market_peak")
        expected_positive = ev["category"] in ("ceasefire", "de-escalation", "diplomatic")
        if expected_negative:
            match = avg_sent < baseline_avg
        elif expected_positive:
            match = avg_sent > baseline_avg
        else:
            match = None
        results.append({"event": ev["event"], "date": ev["date"],
                         "avg_sentiment_in_window": round(avg_sent, 3),
                         "baseline_avg": round(baseline_avg, 3), "match": match, "note": ""})
    return pd.DataFrame(results)


# MAIN

if __name__ == "__main__":
    master = pd.read_csv(f"{PROCESSED_DIR}/master_features.csv")
    master["date"] = pd.to_datetime(master["date"])
    print(f"Loaded master_features.csv: {master.shape[0]} rows x {master.shape[1]} columns")

    event_path = find_event_table()
    if event_path is None:
        raise FileNotFoundError("event_table.csv not found in ., data/raw/, or data/")
    events = pd.read_csv(event_path)
    print(f"Loaded {len(events)} events from {event_path}")

    # -------------------- PART A --------------------
    print("\n=== PART A: FEATURE TABLE IMAGE ===")
    build_feature_table_image(master, f"{PROCESSED_DIR}/feature_table.png")

    # -------------------- PART B --------------------
    print("\n=== PART B: REAL EVENT-WINDOW VALIDATION ===")

    if "guardian_sentiment_mean" not in master.columns:
        raise ValueError("guardian_sentiment_mean not found in master_features.csv")

    sentiment_df = master.loc[master["guardian_sentiment_mean"].notna(), ["date", "guardian_sentiment_mean"]].copy()
    sentiment_df.columns = ["date", "sentiment"]
    print(f"  Real sentiment data: {len(sentiment_df)} days")

    price_df = master[["date", "smard_mean"]].copy()
    price_df.columns = ["date", "price"]
    plot_start = sentiment_df["date"].min() - pd.Timedelta(days=30)
    price_df_windowed = price_df[price_df["date"] >= plot_start]

    plot_sentiment_vs_events(sentiment_df, events, price_df_windowed,
                              save_path=f"{PROCESSED_DIR}/real_sentiment_validation.png")

    check_df = compute_event_window_check(sentiment_df, events)
    print("\nEvent-window sentiment check (REAL data):")
    print(check_df.to_string(index=False))

    n_checked = check_df["match"].notna().sum()
    n_matched = (check_df["match"] == True).sum()
    n_outside_coverage = check_df["note"].str.len().gt(0).sum() if "note" in check_df else 0

    print(f"\nRESULT: sentiment matched the expected direction in "
          f"{n_matched}/{n_checked} checkable events "
          f"({n_outside_coverage} event(s) had no sentiment coverage in their window, "
          f"excluded from this count; ambiguous-category events are also excluded).")

    check_df.to_csv(f"{PROCESSED_DIR}/event_window_check_results.csv", index=False)
    print(f"\nSaved: {PROCESSED_DIR}/event_window_check_results.csv")
