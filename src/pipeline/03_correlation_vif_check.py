import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
OUT_DIR = "data/processed"

# Core explanatory drivers, no sentiment -- available across the full range.
CORE_FEATURES = [
    "DCOILBRENTEU",       # Brent crude
    "DEXUSEU",             # EUR/USD
    "ttf_eur_mwh",          # TTF gas
    "gpr_daily",            # Geopolitical Risk index
    "renewable_output_mwh", # Dunkelflaute driver
    "dunkelflaute_flag",
]

# Sentiment-window features, added on top of the core set for the
# restricted-window check.
SENTIMENT_FEATURES = [
    "guardian_sentiment_mean",
    "guardian_article_count",
]


def load_master():
    df = pd.read_csv(f"{PROCESSED_DIR}/master_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def correlation_check(df: pd.DataFrame, features: list, label: str):
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [note] not in master_features.csv, skipping: {missing}")

    corr = df[available].corr()

    print(f"\n=== CORRELATION MATRIX ({label}) ===")
    print(corr.round(2).to_string())

    # Flag high-correlation pairs (upper triangle only, avoid duplicates/self)
    print(f"\n  Pairs with |r| > 0.7 (redundancy candidates):")
    found_any = False
    for i, col_i in enumerate(available):
        for col_j in available[i + 1:]:
            r = corr.loc[col_i, col_j]
            if abs(r) > 0.7:
                print(f"    {col_i} <-> {col_j}: r = {r:.2f}")
                found_any = True
    if not found_any:
        print("    none -- no pair exceeds |r| > 0.7")

    return corr


def vif_check(df: pd.DataFrame, features: list, label: str):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    available = [f for f in features if f in df.columns]
    subset = df[available].dropna()

    print(f"\n=== VIF CHECK ({label}) ===")
    print(f"  Using {len(subset)}/{len(df)} rows (after dropping any row "
          f"with a missing value in this feature set)")

    if len(subset) < len(available) + 10:
        print("  [warn] very few complete rows relative to feature count -- "
              "VIF estimates here are unreliable. Treat as directional only.")

    if subset.empty or len(available) < 2:
        print("  [skipped] not enough data/features to compute VIF")
        return None

    subset_with_const = add_constant(subset)

    vif_data = pd.DataFrame({
        "feature": subset_with_const.columns,
        "VIF": [variance_inflation_factor(subset_with_const.values, i)
                for i in range(subset_with_const.shape[1])],
    })
    vif_data = vif_data[vif_data["feature"] != "const"]
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    print(vif_data.to_string(index=False))

    concerning = vif_data[vif_data["VIF"] > 5]
    if len(concerning) > 0:
        print(f"\n  [flag] {len(concerning)} feature(s) with VIF > 5 -- moderate "
              f"multicollinearity, worth discussing in your methodology chapter:")
        for _, row in concerning.iterrows():
            severity = "SEVERE" if row["VIF"] > 10 else "moderate"
            print(f"    {row['feature']}: VIF = {row['VIF']:.2f} ({severity})")
    else:
        print("\n  No feature exceeds VIF > 5 -- no strong multicollinearity "
              "in this set.")

    return vif_data


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 1.1), max(5, len(corr) * 0.9)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            # dark text on light cells, light text on dark (saturated) cells
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=text_color, fontsize=8)

    ax.set_title(title, fontsize=11, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation (r)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap: {out_path}")


if __name__ == "__main__":
    master = load_master()
    print(f"Loaded master_features.csv: {master.shape[0]} rows, {master.shape[1]} cols")

    # --- Check 1: core drivers, full date range, no sentiment ---
    corr_full = correlation_check(master, CORE_FEATURES, "full range, core drivers")
    plot_correlation_heatmap(corr_full, "Correlation matrix -- full range, core drivers",
                              f"{OUT_DIR}/correlation_heatmap_full_range.png")
    vif_full = vif_check(master, CORE_FEATURES, "full range, core drivers")

    # --- Check 2: core drivers + sentiment, restricted to sentiment window ---
    sentiment_window = master[master["sentiment_available"] == 1] if "sentiment_available" in master.columns else pd.DataFrame()
    if not sentiment_window.empty:
        combined_features = CORE_FEATURES + SENTIMENT_FEATURES
        corr_sent = correlation_check(sentiment_window, combined_features, "sentiment window only")
        plot_correlation_heatmap(corr_sent, "Correlation matrix -- sentiment window only",
                                  f"{OUT_DIR}/correlation_heatmap_sentiment_window.png")
        vif_sent = vif_check(sentiment_window, combined_features, "sentiment window only")
    else:
        print("\n[note] no sentiment_available column found or no rows with sentiment -- "
              "skipping the sentiment-window check.")

    # --- Save outputs ---
    os.makedirs(OUT_DIR, exist_ok=True)
    corr_full.to_csv(f"{OUT_DIR}/correlation_matrix_full_range.csv")
    if vif_full is not None:
        vif_full.to_csv(f"{OUT_DIR}/vif_full_range.csv", index=False)
    if not sentiment_window.empty:
        corr_sent.to_csv(f"{OUT_DIR}/correlation_matrix_sentiment_window.csv")
        if vif_sent is not None:
            vif_sent.to_csv(f"{OUT_DIR}/vif_sentiment_window.csv", index=False)

    print(f"\nSaved correlation matrices and VIF tables to {OUT_DIR}/")
    print("\nNext: if anything shows VIF > 10 or |r| > 0.85, decide whether to "
          "drop one of the pair, combine them, or keep both with the overlap "
          "explicitly acknowledged as a limitation -- this is also a natural "
          "place to point to your planned SHAP/ablation analysis later.")
