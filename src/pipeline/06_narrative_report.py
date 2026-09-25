import pandas as pd
import numpy as np
import json
import os

PROCESSED_DIR = "data/processed"


def find_event_table():
    for path in ["event_table.csv", "data/raw/event_table.csv", "data/event_table.csv"]:
        if os.path.exists(path):
            return path
    return None


# PART A: SPIKE EXPLANATIONS

def find_nearest_event(date, events, max_days=5):
    diffs = (events["date"] - date).abs().dt.days
    idx = diffs.idxmin()
    if diffs[idx] <= max_days:
        return events.loc[idx], diffs[idx]
    return None, None


def explain_spikes(master, col, label, events, n=3, max_days=5):
    lines = []
    top = master.nlargest(n, col)
    bottom = master.nsmallest(n, col)

    for direction, subset, verb in [("high", top, "spiked to"), ("low", bottom, "dropped to")]:
        for _, row in subset.iterrows():
            ev, diff = find_nearest_event(row["date"], events, max_days)
            date_str = row["date"].date()
            if ev is not None:
                short_event = str(ev["event"])[:70]
                lines.append(
                    f"- {label} {verb} {row[col]:.1f} on {date_str} -- "
                    f"{diff} day(s) from event: \"{short_event}...\" "
                    f"(expected direction: {ev['expected_price_direction']})"
                )
            else:
                dunk = row.get("dunkelflaute_flag", None)
                if dunk == 1:
                    reason = "no conflict event nearby, but flagged as a Dunkelflaute (low renewable output) day -- likely weather-driven, not conflict-driven"
                else:
                    reason = "no conflict event within 5 days and not a Dunkelflaute day -- unexplained by this dataset, worth a manual look"
                lines.append(f"- {label} {verb} {row[col]:.1f} on {date_str} -- {reason}")
    return lines


# PART B: SENTIMENT VALIDATION SUMMARY (relative-baseline check, same as script 05)

def compute_event_window_check(sentiment_df, event_df, window_days=2, baseline_days=21):
    sentiment_df = sentiment_df.copy()
    event_df = event_df.copy()
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
            results.append({"event": ev["event"], "date": ev["date"], "match": None})
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
        results.append({"event": ev["event"], "date": ev["date"], "match": match})
    return pd.DataFrame(results)


def summarize_sentiment_validation(master, events):
    sentiment_df = master.loc[master["guardian_sentiment_mean"].notna(), ["date", "guardian_sentiment_mean"]].copy()
    sentiment_df.columns = ["date", "sentiment"]
    overall_mean = sentiment_df["sentiment"].mean()

    result = compute_event_window_check(sentiment_df, events)
    n_checked = result["match"].notna().sum()
    n_matched = (result["match"] == True).sum()

    bias_note = (
        f"Overall mean sentiment across the coverage window is {overall_mean:.2f} -- "
        f"consistently negative, which is expected for coverage of an active war "
        f"(articles about even positive developments like a ceasefire still reference "
        f"the surrounding conflict). Because of this, the check compares each event's "
        f"sentiment against its own 21-day rolling baseline rather than against zero."
    )
    result_line = (
        f"Using relative-baseline matching, sentiment moved in the expected direction "
        f"for {n_matched}/{n_checked} checkable events."
    )
    return bias_note, result_line, result


# PART C: FEATURE CATEGORY BREAKDOWN

def categorize_column(col: str) -> str:
    if col == "date":
        return "Identifier"
    if col.startswith("smard_") and "_lag" not in col and "_vol" not in col and "_outlier" not in col:
        return "Target (electricity price)"
    if col in ("DCOILBRENTEU", "DEXUSEU", "ttf_eur_mwh", "gpr_daily") and "_lag" not in col:
        return "Macro / geopolitical driver"
    if col in ("wind_offshore_mwh", "wind_onshore_mwh", "solar_mwh", "renewable_output_mwh", "dunkelflaute_flag"):
        return "Weather / renewable generation"
    if col.startswith("guardian_") or col == "sentiment_available":
        return "News sentiment"
    if "_lag" in col:
        return "Engineered: lag feature"
    if "_vol7d" in col or "_vol30d" in col:
        return "Engineered: rolling volatility"
    if "_outlier_flag" in col:
        return "Engineered: outlier flag"
    if col in ("day_of_week", "month", "is_weekend", "is_holiday"):
        return "Calendar"
    return "Other / uncategorized"


def build_feature_category_summary(master):
    cats = pd.Series([categorize_column(c) for c in master.columns])
    counts = cats.value_counts()
    total = len(master.columns)
    return counts, total


# MAIN

if __name__ == "__main__":
    master = pd.read_csv(f"{PROCESSED_DIR}/master_features.csv")
    master["date"] = pd.to_datetime(master["date"])

    event_path = find_event_table()
    events = pd.read_csv(event_path)
    events["date"] = pd.to_datetime(events["date"])

    report_lines = ["# Pipeline Narrative Report\n"]
    captions = {}

    # -------------------- PART A --------------------
    print("=== PART A: SPIKE EXPLANATIONS ===\n")
    report_lines.append("## Price spike explanations\n")

    full_range_lines = ["**SMARD electricity price -- notable highs/lows:**"]
    if "smard_mean" in master.columns:
        full_range_lines += explain_spikes(master, "smard_mean", "SMARD price", events)
    else:
        full_range_lines.append("- [skipped] 'smard_mean' column not found in master_features.csv")

    full_range_lines.append("\n**Brent crude -- notable highs/lows:**")
    if "DCOILBRENTEU" in master.columns:
        full_range_lines += explain_spikes(master, "DCOILBRENTEU", "Brent", events, n=2)
    else:
        full_range_lines.append("- [skipped] 'DCOILBRENTEU' column not found in master_features.csv")

    full_range_lines.append("\n**TTF gas -- notable highs/lows:**")
    if "ttf_eur_mwh" in master.columns:
        full_range_lines += explain_spikes(master, "ttf_eur_mwh", "TTF gas", events, n=2)
    else:
        full_range_lines.append(
            "- [skipped] 'ttf_eur_mwh' column not found in master_features.csv -- "
            "this usually means the manual TTF CSV wasn't present in data/raw/ when "
            "01_data_collection.py and 02_preprocessing.py last ran. Re-upload "
            "Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv, re-run those two "
            "steps, then re-run this script."
        )

    for line in full_range_lines:
        print(line)
    report_lines += full_range_lines

    captions["event_alignment_full_range.png"] = "\n".join(full_range_lines)
    captions["event_alignment_conflict_window.png"] = (
        "Same spikes as above, zoomed to the Feb 2026+ conflict window where the "
        "event lines are actually legible. See explanations above for what's "
        "driving each visible move."
    )

    # -------------------- PART B --------------------
    print("\n=== PART B: SENTIMENT VALIDATION SUMMARY ===\n")
    if "guardian_sentiment_mean" in master.columns:
        bias_note, result_line, result_df = summarize_sentiment_validation(master, events)
        print(bias_note)
        print(result_line)
        report_lines.append("\n## Sentiment validation\n")
        report_lines.append(bias_note)
        report_lines.append(result_line)
        captions["real_sentiment_validation.png"] = f"{bias_note}\n\n{result_line}"
    else:
        captions["real_sentiment_validation.png"] = "Sentiment column not found in master_features.csv."

    # -------------------- PART C --------------------
    print("\n=== PART C: FEATURE CATEGORY BREAKDOWN ===\n")
    counts, total = build_feature_category_summary(master)
    print(counts.to_string())
    print(f"\nTOTAL FEATURES: {total}")

    report_lines.append("\n## Feature category breakdown\n")
    report_lines.append(counts.to_string())
    report_lines.append(f"\n**Total features: {total}**")

    feature_table_caption = (
        f"master_features.csv has {total} columns total, grouped as: "
        + ", ".join(f"{v} {k.lower()}" for k, v in counts.items())
        + ". See the printed breakdown for the full category list."
    )
    captions["feature_table_part1.png"] = feature_table_caption
    captions["feature_table_part2.png"] = feature_table_caption

    # -------------------- SAVE --------------------
    with open(f"{PROCESSED_DIR}/plot_captions.json", "w") as f:
        json.dump(captions, f, indent=2)
    with open(f"{PROCESSED_DIR}/NARRATIVE_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved captions: {PROCESSED_DIR}/plot_captions.json")
    print(f"Saved full report: {PROCESSED_DIR}/NARRATIVE_REPORT.md")
