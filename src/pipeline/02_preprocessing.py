import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_START = "2021-01-01"
LOCAL_TZ = "Europe/Berlin"
DAILY_CLOSE_SAFE_LAG = 2


# 1. LOAD RAW DATA

def load_smard():
    df = pd.read_csv(f"{RAW_DIR}/smard_day_ahead_price.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def load_fred_series(filename, value_col):
    df = pd.read_csv(f"{RAW_DIR}/{filename}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={value_col: value_col})  # keep as-is, just confirming
    return df[["date", value_col]]


def load_ttf():
    df = pd.read_csv(f"{RAW_DIR}/ttf_gas_price.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "ttf_eur_mwh"]]


def load_gpr():
    df = pd.read_csv(f"{RAW_DIR}/gpr_index.csv")
    print(f"  GPR file columns found: {list(df.columns)}")
    exact_date = [c for c in df.columns if c.lower() == "date"]
    date_col_candidates = exact_date or [c for c in df.columns if c.lower() in ("day", "dates")]
    value_col_candidates = [c for c in df.columns if "gprd" in c.lower() and "ma" not in c.lower()]

    if not date_col_candidates or not value_col_candidates:
        raise ValueError(
            "Could not confidently identify GPR date/value columns automatically. "
            f"Actual columns are: {list(df.columns)}. "
            "Open the CSV, find the daily GPR index column (likely named "
            "something like 'GPRD'), and update load_gpr() with the exact name."
        )

    date_col, value_col = date_col_candidates[0], value_col_candidates[0]
    print(f"  Using date column '{date_col}' and value column '{value_col}'")

    other_date_like_cols = [c for c in df.columns
                             if c != date_col and c.lower() in ("date", "day", "dates")]
    if other_date_like_cols:
        df = df.drop(columns=other_date_like_cols)

    df = df.rename(columns={date_col: "date", value_col: "gpr_daily"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "gpr_daily"]]
    df = df[df["date"] >= ANALYSIS_START].reset_index(drop=True)

    if df.empty or df["date"].max() < pd.Timestamp(ANALYSIS_START):
        raise ValueError(
            f"GPR date column parsed but produced no rows overlapping "
            f"{ANALYSIS_START} onward (parsed range: "
            f"{df['date'].min() if not df.empty else 'empty'} to "
            f"{df['date'].max() if not df.empty else 'empty'}). "
            "The wrong column was likely used as the date -- check "
            "load_gpr() against your actual file's columns."
        )
    print(f"  GPR date range after filtering: {df['date'].min().date()} to {df['date'].max().date()}")

    return df


def load_guardian_sentiment():
    df = pd.read_csv(f"{RAW_DIR}/guardian_daily_sentiment_clean.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


# 2. AGGREGATE SMARD TO DAILY

def load_smard_generation():
    df = pd.read_csv(f"{RAW_DIR}/smard_generation.csv")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def expected_local_hours(date_value) -> int:
    start = pd.Timestamp(date_value).tz_localize(LOCAL_TZ)
    end = (pd.Timestamp(date_value) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    return int((end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1))


def validate_complete_smard_days(df, value_cols, label):
    out = df.copy().sort_values("datetime")
    if out["datetime"].duplicated().any():
        raise ValueError(f"{label} contains duplicate hourly timestamps")
    out["local_datetime"] = out["datetime"].dt.tz_convert(LOCAL_TZ)
    out["date"] = out["local_datetime"].dt.tz_localize(None).dt.normalize()
    counts = out.groupby("date")[value_cols].count()
    expected = pd.Series({d: expected_local_hours(d) for d in counts.index})
    incomplete = counts.ne(expected, axis=0).any(axis=1)

    last_date = out["date"].max()
    if last_date in incomplete.index and incomplete.loc[last_date]:
        print(f"  [note] dropping incomplete final {label} day: {last_date.date()}")
        out = out[out["date"] < last_date]
        incomplete = incomplete.drop(index=last_date)
    bad_dates = list(incomplete[incomplete].index)
    if bad_dates:
        preview = ", ".join(str(d.date()) for d in bad_dates[:5])
        raise ValueError(f"{label} has incomplete internal delivery days: {preview}")
    return out


def aggregate_generation_daily(gen_df):
    gen_df = validate_complete_smard_days(
        gen_df,
        ["wind_offshore_mwh", "wind_onshore_mwh", "solar_mwh"],
        "SMARD generation",
    )
    daily = gen_df.groupby("date")[
        ["wind_offshore_mwh", "wind_onshore_mwh", "solar_mwh"]
    ].sum(min_count=1).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["renewable_output_mwh"] = (
        daily["wind_offshore_mwh"] + daily["wind_onshore_mwh"] + daily["solar_mwh"]
    )
    return daily


def add_dunkelflaute_flag(df, col="renewable_output_mwh", window_days=90, percentile=0.15):
    df = df.copy()
    # Estimate the threshold from information available before day t. The
    # resulting flag describes day t and may only enter forecast models lagged.
    rolling_threshold = df[col].shift(1).rolling(
        window=window_days, min_periods=30
    ).quantile(percentile)
    df["dunkelflaute_flag"] = (df[col] < rolling_threshold).astype(int)
    return df


def aggregate_smard_daily(smard_df):
    smard_df = validate_complete_smard_days(smard_df, ["price_eur_mwh"], "SMARD price")
    daily = smard_df.groupby("date")["price_eur_mwh"].agg(
        smard_mean="mean", smard_min="min", smard_max="max", smard_std="std"
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# 3. MISSING VALUE HANDLING (forward-fill, per methodology)

def forward_fill_on_full_range(df, date_col, start, end, limit, label):
    full_range = pd.date_range(start, end, freq="D")
    df = df.set_index(date_col).reindex(full_range)
    original_missing = df.isna()
    df = df.ffill(limit=limit)
    remaining = df.isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        raise ValueError(
            f"{label} still has missing values after a maximum {limit}-day fill: "
            f"{remaining.to_dict()}"
        )
    for col in [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]:
        df[f"{col}_imputed"] = (original_missing[col] & df[col].notna()).astype(int)
    df.index.name = date_col
    return df.reset_index()


# 4. OUTLIER FLAGGING (z-score > 3, FLAG not remove)

def flag_outliers(df, cols, z_thresh=3.0):
    df = df.copy()
    for col in cols:
        # Expanding past-only moments avoid using future observations to decide
        # whether an earlier value was unusual.
        history = df[col].shift(1).expanding(min_periods=30)
        z = (df[col] - history.mean()) / history.std()
        df[f"{col}_outlier_flag"] = (z.abs() > z_thresh).astype(int)
    return df


# 5. FEATURE ENGINEERING

def add_lag_features(df, cols, lags=range(1, 8)):
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_volatility(df, cols, windows=(7, 30), source_lags=None):
    df = df.copy()
    source_lags = source_lags or {}
    for col in cols:
        source_lag = int(source_lags.get(col, 1))
        for w in windows:
            df[f"{col}_vol{w}d"] = df[col].shift(source_lag).rolling(window=w).std()
    return df


def add_calendar_features(df, date_col):
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Basic DE public holiday flag -- not exhaustive, refine with a proper
    # holiday calendar package (e.g. `holidays` library) before final use.
    try:
        import holidays
        years = range(df[date_col].dt.year.min(), df[date_col].dt.year.max() + 1)
        de_holidays = holidays.Germany(years=years)
        df["is_holiday"] = df[date_col].dt.date.isin(de_holidays).astype(int)
    except ImportError:
        print("  [note] 'holidays' package not installed -- is_holiday set to 0 "
              "for all rows. Run: pip install holidays, then re-run for real flags.")
        df["is_holiday"] = 0
    return df


# VALIDATION AND MAIN

REQUIRED_RAW_FILES = [
    "manifest.json",
    "smard_day_ahead_price.csv",
    "smard_generation.csv",
    "brent_fred.csv",
    "eurusd_fred.csv",
    "ttf_gas_price.csv",
    "gpr_index.csv",
    "guardian_daily_sentiment_clean.csv",
    "event_table.csv",
]


def require_collection_complete():
    missing = [name for name in REQUIRED_RAW_FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Raw collection is incomplete. Missing: " + ", ".join(missing) +
            ". Run 01_data_collection.py successfully; do not continue with "
            "a partial source set."
        )
    with (RAW_DIR / "manifest.json").open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "complete":
        raise ValueError("data/raw/manifest.json does not certify a complete collection")
    manifested = manifest.get("files", {})
    required_manifested = set(REQUIRED_RAW_FILES) - {"manifest.json"}
    if missing_entries := required_manifested - set(manifested):
        raise ValueError(f"Raw manifest omits required files: {sorted(missing_entries)}")
    for filename, metadata in manifested.items():
        path = RAW_DIR / filename
        if not path.exists() or sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Raw file does not match collection manifest: {filename}")
    return manifest


def last_observed_date(df, value_col):
    observed = df.loc[df[value_col].notna(), "date"]
    if observed.empty:
        raise ValueError(f"{value_col} contains no observed values")
    return observed.max().normalize()


def first_observed_date(df, value_col):
    observed = df.loc[df[value_col].notna(), "date"]
    if observed.empty:
        raise ValueError(f"{value_col} contains no observed values")
    return observed.min().normalize()


def reindex_required_daily(df, date_col, start, end, required_cols, label):
    if df[date_col].duplicated().any():
        raise ValueError(f"{label} contains duplicate dates")
    full_range = pd.date_range(start, end, freq="D")
    out = df.set_index(date_col).reindex(full_range)
    missing = out[required_cols].isna().any(axis=1)
    if missing.any():
        preview = ", ".join(str(d.date()) for d in out.index[missing][:5])
        raise ValueError(f"{label} has missing required daily observations: {preview}")
    out.index.name = date_col
    return out.reset_index()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

if __name__ == "__main__":
    raw_manifest = require_collection_complete()
    # A failed rebuild must not leave a stale success certificate.
    (PROCESSED_DIR / "manifest.json").unlink(missing_ok=True)
    print("Loading raw data...")
    smard_raw = load_smard()
    brent = load_fred_series("brent_fred.csv", "DCOILBRENTEU")
    eurusd = load_fred_series("eurusd_fred.csv", "DEXUSEU")
    gpr = load_gpr()

    ttf = load_ttf()
    guardian = load_guardian_sentiment()
    print(f"  TTF: {len(ttf)} rows loaded")

    print("\nAggregating SMARD to daily...")
    smard_daily = aggregate_smard_daily(smard_raw)
    print(f"  {len(smard_daily)} daily rows")

    print("\nLoading and aggregating SMARD renewable generation "
          "(wind offshore/onshore, solar) for Dunkelflaute control...")
    gen_raw = load_smard_generation()
    gen_daily = aggregate_generation_daily(gen_raw)
    print(f"  {len(gen_daily)} daily rows")

    print("\nBuilding a common complete daily window...")
    start_date = max(
        pd.Timestamp(ANALYSIS_START),
        first_observed_date(smard_daily, "smard_mean"),
        first_observed_date(gen_daily, "renewable_output_mwh"),
        first_observed_date(brent, "DCOILBRENTEU"),
        first_observed_date(eurusd, "DEXUSEU"),
        first_observed_date(ttf, "ttf_eur_mwh"),
        first_observed_date(gpr, "gpr_daily"),
    )
    end_date = min(
        last_observed_date(smard_daily, "smard_mean"),
        last_observed_date(gen_daily, "renewable_output_mwh"),
        last_observed_date(brent, "DCOILBRENTEU"),
        last_observed_date(eurusd, "DEXUSEU"),
        last_observed_date(ttf, "ttf_eur_mwh"),
        last_observed_date(gpr, "gpr_daily"),
    )
    if end_date < start_date:
        raise ValueError(f"No common source coverage after {ANALYSIS_START}")
    print(f"  Common analysis range: {start_date.date()} to {end_date.date()}")

    smard_daily = reindex_required_daily(
        smard_daily, "date", start_date, end_date,
        ["smard_mean", "smard_min", "smard_max", "smard_std"], "SMARD price",
    )
    gen_daily = reindex_required_daily(
        gen_daily, "date", start_date, end_date,
        ["wind_offshore_mwh", "wind_onshore_mwh", "solar_mwh",
         "renewable_output_mwh"], "SMARD generation",
    )
    # Market closures create short, legitimate gaps; cap carry-forward so a
    # stale quote cannot silently stand in for weeks of missing data.
    brent = forward_fill_on_full_range(
        brent, "date", start_date, end_date, 4, "Brent"
    )
    eurusd = forward_fill_on_full_range(
        eurusd, "date", start_date, end_date, 4, "EUR/USD"
    )
    ttf = forward_fill_on_full_range(
        ttf, "date", start_date, end_date, 4, "TTF"
    )
    gpr = forward_fill_on_full_range(
        gpr, "date", start_date, end_date, 7, "daily GPR"
    )

    print("\nMerging into master table...")
    master = smard_daily.merge(brent, on="date", how="left")
    master = master.merge(eurusd, on="date", how="left")
    master = master.merge(gpr, on="date", how="left")
    master = master.merge(ttf, on="date", how="left")
    master = master.merge(gen_daily, on="date", how="left")
    master = add_dunkelflaute_flag(master)
    n_flagged = master["dunkelflaute_flag"].sum()
    print(f"  Added TTF ({master['ttf_eur_mwh'].notna().sum()} rows) and "
          f"renewables ({n_flagged} prior-threshold low-output days)")

    print("\nMerging Guardian sentiment (partial coverage, NOT forward-filled)...")
    master = master.merge(guardian, on="date", how="left")
    # Missing means unavailable, not neutral. Models receive lagged values and
    # a lagged availability indicator; they never receive today's news status.
    master["sentiment_available"] = master["guardian_sentiment_mean"].notna().astype(int)
    n_available = master["sentiment_available"].sum()
    print(f"  Sentiment available on {n_available}/{len(master)} days "
          f"({100*n_available/len(master):.1f}%)")

    print(f"  Master table: {master.shape[0]} rows, {master.shape[1]} columns")

    print("\nFlagging outliers (z > 3, not removed)...")
    master = flag_outliers(master, ["smard_mean", "DCOILBRENTEU"])

    print("\nAdding lag features (t-1 to t-7)...")
    lag_cols = ["smard_mean", "DCOILBRENTEU", "DEXUSEU"]
    lag_cols.extend(["ttf_eur_mwh", "renewable_output_mwh",
                     "dunkelflaute_flag", "sentiment_available"])

    for conflict_col in ["gpr_daily", "guardian_sentiment_mean",
                          "guardian_sentiment_min", "guardian_sentiment_max",
                          "guardian_article_count"]:
        if conflict_col in master.columns:
            lag_cols.append(conflict_col)
        else:
            print(f"  [note] '{conflict_col}' not in master table -- no lags created for it")

    master = add_lag_features(master, lag_cols)
    print(f"  Lagged: {lag_cols}")

    print("\nAdding rolling volatility (7d, 30d)...")
    vol_cols = ["smard_mean", "DCOILBRENTEU"]
    vol_cols.append("ttf_eur_mwh")
    master = add_rolling_volatility(
        master,
        vol_cols,
        source_lags={"DCOILBRENTEU": DAILY_CLOSE_SAFE_LAG,
                     "ttf_eur_mwh": DAILY_CLOSE_SAFE_LAG},
    )

    print("\nAdding calendar features...")
    master = add_calendar_features(master, "date")

    print("\nChecking for silent gaps from the merge...")
    n_missing = master.isna().sum()
    cols_with_gaps = n_missing[n_missing > 0]
    if len(cols_with_gaps) > 0:
        print("  Columns with missing values (expected for early lag rows):")
        print(cols_with_gaps.to_string())
    else:
        print("  No missing values.")

    if master["date"].duplicated().any() or master["smard_mean"].isna().any():
        raise ValueError("Final master table violates unique-date/non-null-target contract")
    out_path = PROCESSED_DIR / "master_features.csv"
    atomic_write_csv(master, out_path)
    processed_manifest = {
        "schema_version": 1,
        "status": "complete",
        "analysis_start": str(master["date"].min().date()),
        "analysis_end": str(master["date"].max().date()),
        "rows": int(len(master)),
        "columns": int(master.shape[1]),
        "source_manifest_sha256": sha256(RAW_DIR / "manifest.json"),
        "master_features_sha256": sha256(out_path),
        "forecast_feature_policy": (
            "electricity target history enters at lag 1; daily closing-market, "
            "geopolitical, news and auction features use a conservative D-2 "
            "availability lag unless a timestamped pre-cutoff observation exists; "
            "renewable/Dunkelflaute signals enter only lagged and rolling statistics "
            "are shifted according to the same policy"
        ),
        "daily_close_safe_lag_days": DAILY_CLOSE_SAFE_LAG,
        "raw_analysis_end": raw_manifest.get("analysis_end"),
    }
    manifest_path = PROCESSED_DIR / "manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    with tmp_manifest.open("w", encoding="utf-8") as fh:
        json.dump(processed_manifest, fh, indent=2)
    tmp_manifest.replace(manifest_path)
    print(f"\nSaved {master.shape[0]} rows x {master.shape[1]} cols to {out_path}")
    print("\nNext: inspect master_features.csv manually before modeling --")
    print("plot smard_mean and DCOILBRENTEU over time and confirm the shape")
    print("looks right around your event dates.")
