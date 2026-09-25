import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2021-01-01"
END_DATE = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")            # not actually needed for the CSV endpoint used below
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY", "")    # https://open-platform.theguardian.com/access/

GUARDIAN_FROM_DATE = "2025-09-01"
GUARDIAN_TO_DATE = END_DATE

SOURCE_URLS = {
    "smard": "https://www.smard.de/app/chart_data",
    "fred": "https://fred.stlouisfed.org/graph/fredgraph.csv",
    "gpr": "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
    "guardian": "https://content.guardianapis.com/search",
    "ttf": "manual Investing.com export",
}


def request_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "iran-energy-thesis/1.0 (academic research)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


HTTP = request_session()


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def require_nonempty(df: pd.DataFrame, label: str, required_columns: tuple[str, ...]) -> None:
    if df.empty:
        raise ValueError(f"{label} returned zero rows")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 1. SMARD -- Day-ahead electricity price (Germany/Luxembourg)

def fetch_smard_series(filter_code: str, region: str = "DE-LU", resolution: str = "hour") -> pd.DataFrame:
    base = "https://www.smard.de/app/chart_data"
    idx_url = f"{base}/{filter_code}/{region}/index_{resolution}.json"
    resp = HTTP.get(idx_url, timeout=45)
    resp.raise_for_status()
    timestamps = resp.json()["timestamps"]

    start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    relevant_timestamps = [ts for ts in timestamps if ts >= start_ms - 1000 * 60 * 60 * 24 * 90]
    # ^ pad backwards ~90 days since chunk boundaries don't align exactly to START_DATE

    all_rows = []
    failed_chunks = []
    for ts in relevant_timestamps:
        chunk_url = f"{base}/{filter_code}/{region}/{filter_code}_{region}_{resolution}_{ts}.json"
        try:
            r = HTTP.get(chunk_url, timeout=45)
            r.raise_for_status()
            series = r.json().get("series", [])
            all_rows.extend(series)
        except requests.RequestException as e:
            print(f"    [warn] failed chunk at ts={ts}: {e}")
            failed_chunks.append(ts)
        time.sleep(0.3)  # be polite to the API

    if failed_chunks:
        raise RuntimeError(
            f"SMARD {filter_code} was incomplete: {len(failed_chunks)} chunk(s) failed. "
            "No partial file was written; rerun collection."
        )

    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "value"])
    require_nonempty(df, f"SMARD {filter_code}", ("timestamp_ms", "value"))
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df["datetime"].duplicated().any():
        raise ValueError(f"SMARD {filter_code} contains duplicate timestamps")
    # SMARD values are German delivery hours. Bound the request in local time,
    # then convert to UTC; otherwise the first local day loses its 00:00 hour.
    start = pd.Timestamp(START_DATE).tz_localize("Europe/Berlin").tz_convert("UTC")
    end_exclusive = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).tz_localize(
        "Europe/Berlin"
    ).tz_convert("UTC")
    df = df[(df["datetime"] >= start) & (df["datetime"] < end_exclusive)]
    df = df.sort_values("datetime").reset_index(drop=True)
    require_nonempty(df, f"SMARD {filter_code} in requested date range", ("datetime", "value"))
    return df[["datetime", "value"]]


def fetch_smard_day_ahead_price(region: str = "DE-LU") -> pd.DataFrame:
    df = fetch_smard_series("4169", region=region)  # 4169 = Marktpreis DE/LU
    return df.rename(columns={"value": "price_eur_mwh"})


# Wind offshore, wind onshore, solar -- summed later into a single
# renewable-output feature for the Dunkelflaute control.
GENERATION_FILTERS = {
    "wind_offshore_mwh": "1225",
    "wind_onshore_mwh": "4067",
    "solar_mwh": "4068",
}


def fetch_smard_generation() -> pd.DataFrame:
    merged = None
    for col_name, filter_code in GENERATION_FILTERS.items():
        print(f"    fetching {col_name} (filter {filter_code})...")
        series_df = fetch_smard_series(filter_code)
        series_df = series_df.rename(columns={"value": col_name})
        merged = series_df if merged is None else merged.merge(series_df, on="datetime", how="outer")
    return merged.sort_values("datetime").reset_index(drop=True)


# 2. FRED -- Brent crude and EUR/USD

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={START_DATE}&coed={END_DATE}"
    )
    response = HTTP.get(url, timeout=45)
    response.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")  # FRED uses '.' for missing days
    require_nonempty(df, f"FRED {series_id}", ("date", series_id))
    return df


def fetch_brent() -> pd.DataFrame:
    return fetch_fred_series("DCOILBRENTEU")  # Brent, Europe, daily, $/barrel


def fetch_eurusd() -> pd.DataFrame:
    return fetch_fred_series("DEXUSEU")  # EUR/USD daily reference rate


def fetch_ttf(path: str = f"{RAW_DIR}/Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy your downloaded "
            "Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv into data/raw/ "
            "(or update `path` above) before running this."
        )
    path = Path(path)
    df = pd.read_csv(path)
    require_nonempty(df, "TTF manual export", ("Date", "Price"))
    df = df.rename(columns={"Date": "date", "Price": "ttf_eur_mwh"})
    dayfirst = os.environ.get("TTF_DAYFIRST", "0") == "1"
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=dayfirst, errors="coerce")
    cleaned = (df["ttf_eur_mwh"].astype(str)
               .str.replace("\u202f", "", regex=False)
               .str.replace(" ", "", regex=False)
               .str.replace(",", "", regex=False))
    df["ttf_eur_mwh"] = pd.to_numeric(cleaned, errors="coerce")
    if df[["date", "ttf_eur_mwh"]].isna().any().any():
        bad = int(df[["date", "ttf_eur_mwh"]].isna().any(axis=1).sum())
        raise ValueError(
            f"TTF parsing failed for {bad} row(s). Check TTF_DAYFIRST and the export's decimal format."
        )
    df = df[["date", "ttf_eur_mwh"]].sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("TTF export contains duplicate dates")
    if (df["ttf_eur_mwh"] <= 0).any() or (df["ttf_eur_mwh"] > 1000).any():
        raise ValueError(
            "TTF export contains implausible prices; verify decimal separators, units, and parsing"
        )
    return df


# 4. GPR Index (Caldara & Iacoviello)

def fetch_gpr_index() -> pd.DataFrame:
    GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    response = HTTP.get(GPR_URL, timeout=60)
    response.raise_for_status()
    from io import BytesIO
    df = pd.read_excel(BytesIO(response.content))
    require_nonempty(df, "GPR daily dataset", tuple())
    if not any(c.lower() == "date" for c in df.columns):
        raise ValueError(f"GPR daily dataset has no exact date column: {list(df.columns)}")
    if not any("gprd" in c.lower() and "ma" not in c.lower() for c in df.columns):
        raise ValueError(f"GPR daily dataset has no GPRD value column: {list(df.columns)}")
    return df


GUARDIAN_BASE_URL = "https://content.guardianapis.com/search"
GUARDIAN_QUERY = '(Iran OR "Strait of Hormuz" OR Hormuz) AND (oil OR gas OR sanctions OR energy)'

SENTIMENT_LEXICON = {
    "strait of hormuz": -3.0, "force majeure": -2.5, "supply disruption": -2.0,
    "missile strike": -3.0, "air raid": -3.0, "sanctions": -1.5,
    "ceasefire": 2.0, "peace deal": 2.5, "diplomatic": 1.5,
    "de-escalation": 2.0, "lng shortage": -2.5, "oil embargo": -2.0, "blockade": -2.5,
}

# Sections where a genuine Iran/Hormuz-conflict story is essentially never
# going to appear.
OFF_TOPIC_SECTIONS = {
    "Music", "Film", "Games", "Fashion", "Sport", "Football", "Life and style",
    "Food", "Books", "Stage", "Television & radio", "Culture", "Travel",
    "Crossword", "Recipes", "Art and design",
}

# Broader than literal "Iran"/"Hormuz" -- live blogs and follow-on coverage
# often use these instead of naming the country/strait directly.
CORE_TERMS = (
    "iran", "hormuz", "tehran", "khamenei", "gulf of oman",
    "middle east crisis", "middle east war", "persian gulf",
)


def fetch_all_guardian_articles(query: str, from_date: str, to_date: str) -> list:
    all_results = []
    page = 1
    while True:
        params = {
            "q": query, "from-date": from_date, "to-date": to_date,
            "page-size": 200, "page": page, "order-by": "oldest",
            "show-fields": "trailText,body", "api-key": GUARDIAN_API_KEY,
        }
        resp = HTTP.get(GUARDIAN_BASE_URL, params=params, timeout=45)
        resp.raise_for_status()
        payload = resp.json()["response"]
        results = payload.get("results", [])
        all_results.extend(results)
        total_pages = payload.get("pages", 1)
        print(f"    fetched page {page}/{total_pages} ({len(results)} articles)")
        if page >= total_pages:
            break
        page += 1
        time.sleep(1)  # stay clear of any per-second limit
    return all_results


def flag_article(art: dict) -> tuple:
    section = art.get("sectionName", "")
    title = art.get("webTitle", "") or ""
    trail = art.get("fields", {}).get("trailText", "") or ""
    body = art.get("fields", {}).get("body", "") or ""

    headline_haystack = f"{title} {trail}".lower()
    body_text = body.lower()

    has_term_in_headline = any(term in headline_haystack for term in CORE_TERMS)
    has_term_in_body = any(term in body_text for term in CORE_TERMS)

    reasons = []
    if section in OFF_TOPIC_SECTIONS:
        reasons.append(f"off-topic section ({section})")
    if not has_term_in_headline and not has_term_in_body:
        reasons.append("no core term anywhere in article (title, trailText, or body)")

    return (len(reasons) > 0, "; ".join(reasons) if reasons else "")


def build_guardian_dataframe(articles: list) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(SENTIMENT_LEXICON)

    rows = []
    tag_re = re.compile(r"<[^>]+>")
    for art in articles:
        likely_noise, reason = flag_article(art)
        title = art.get("webTitle", "")
        trail = art.get("fields", {}).get("trailText", "") or ""
        text = tag_re.sub(" ", f"{title}. {trail}")
        sentiment = analyzer.polarity_scores(text)["compound"]
        rows.append({
            "date": art.get("webPublicationDate", "")[:10],
            "article_id": art.get("id", ""),
            "section": art.get("sectionName", ""),
            "title": title,
            "trailText": trail,
            "sentiment": sentiment,
            "likely_noise": likely_noise,
            "noise_reason": reason,
        })
    out = pd.DataFrame(rows)
    if not out.empty and out["article_id"].astype(bool).any():
        out = out.drop_duplicates(subset="article_id", keep="last")
    return out


def coverage_density(df: pd.DataFrame, from_date: str, to_date: str, label: str) -> pd.Series:
    full_range = pd.date_range(from_date, to_date, freq="D")
    dates = pd.to_datetime(df["date"])
    daily_counts = dates.groupby(dates.dt.floor("D")).size().reindex(full_range, fill_value=0)
    n_total = len(full_range)
    n_zero = (daily_counts == 0).sum()
    print(f"  [{label}] {n_total - n_zero}/{n_total} days covered "
          f"({100*(n_total - n_zero)/n_total:.1f}%), mean {daily_counts.mean():.2f} articles/day")
    return daily_counts


def fetch_guardian_sentiment() -> tuple:
    print(f"  Query: {GUARDIAN_QUERY}")
    articles = fetch_all_guardian_articles(GUARDIAN_QUERY, GUARDIAN_FROM_DATE, GUARDIAN_TO_DATE)
    print(f"  Total articles retrieved: {len(articles)}")

    df = build_guardian_dataframe(articles)
    require_nonempty(df, "Guardian search", ("date", "article_id", "sentiment", "likely_noise"))
    coverage_density(df, GUARDIAN_FROM_DATE, GUARDIAN_TO_DATE, "before noise filtering")

    n_flagged = df["likely_noise"].sum()
    print(f"  Flagged as noise: {n_flagged}/{len(df)} ({100*n_flagged/len(df):.1f}%)")

    clean_df = df[~df["likely_noise"]].copy()
    coverage_density(clean_df, GUARDIAN_FROM_DATE, GUARDIAN_TO_DATE, "after noise filtering")

    daily = clean_df.groupby("date")["sentiment"].agg(
        guardian_sentiment_mean="mean", guardian_sentiment_min="min",
        guardian_sentiment_max="max", guardian_article_count="count"
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    return df, clean_df, daily


# MAIN

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and validate all thesis data sources")
    parser.add_argument("--start-date", default=START_DATE, help="inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", default=END_DATE, help="inclusive YYYY-MM-DD; freeze this in reported runs")
    return parser.parse_args()


def run_collection(start_date: str, end_date: str) -> None:
    global START_DATE, END_DATE, GUARDIAN_TO_DATE
    START_DATE, END_DATE, GUARDIAN_TO_DATE = start_date, end_date, end_date
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start-date must not be after end-date")
    if pd.Timestamp(end_date) >= pd.Timestamp(datetime.now(timezone.utc).date()):
        raise ValueError("end-date must be no later than yesterday; today's SMARD day is incomplete")
    # Invalidate any prior success certificate before touching source files.
    # A failed refresh must never leave an old manifest looking current.
    (RAW_DIR / "manifest.json").unlink(missing_ok=True)

    failures = []
    written: dict[str, Path] = {}

    def collect(label, filename, producer):
        print(f"\n{label}...")
        try:
            frame = producer()
            require_nonempty(frame, label, tuple())
            path = RAW_DIR / filename
            atomic_to_csv(frame, path)
            written[filename] = path
            print(f"  saved {len(frame)} rows -> {path}")
            return frame
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  [FAILED] {failures[-1]}")
            return None

    collect("[1/6] SMARD day-ahead prices", "smard_day_ahead_price.csv", fetch_smard_day_ahead_price)
    collect("[2/6] SMARD renewable generation", "smard_generation.csv", fetch_smard_generation)
    collect("[3a/6] FRED Brent crude", "brent_fred.csv", fetch_brent)
    collect("[3b/6] FRED EUR/USD", "eurusd_fred.csv", fetch_eurusd)
    collect("[4/6] Dutch TTF manual export", "ttf_gas_price.csv", fetch_ttf)
    collect("[5/6] GPR daily index", "gpr_index.csv", fetch_gpr_index)

    if not GUARDIAN_API_KEY:
        failures.append("[6/6] Guardian: GUARDIAN_API_KEY is not set")
        print(f"\n  [FAILED] {failures[-1]}")
    else:
        try:
            print("\n[6/6] Guardian Iran/Hormuz coverage...")
            all_article_df, article_df, daily_df = fetch_guardian_sentiment()
            for filename, frame in (
                ("guardian_sentiment_articles_all.csv", all_article_df),
                ("guardian_sentiment_articles_clean.csv", article_df),
                ("guardian_daily_sentiment_clean.csv", daily_df),
            ):
                path = RAW_DIR / filename
                atomic_to_csv(frame, path)
                written[filename] = path
            print(f"  saved {len(article_df)} clean articles and {len(daily_df)} daily rows")
        except Exception as exc:
            failures.append(f"[6/6] Guardian: {type(exc).__name__}: {exc}")
            print(f"  [FAILED] {failures[-1]}")

    # Preserve the exact manual input in the provenance manifest as well as its
    # standardised derivative.
    ttf_source = RAW_DIR / "Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv"
    if ttf_source.exists():
        written[ttf_source.name] = ttf_source
    event_table = RAW_DIR / "event_table.csv"
    if event_table.exists():
        events = pd.read_csv(event_table)
        required_event_cols = {"date", "event", "category", "verified", "source_url"}
        if missing := required_event_cols - set(events.columns):
            failures.append(f"event table missing columns: {sorted(missing)}")
        elif not events["verified"].astype(str).str.lower().eq("true").all():
            failures.append("event table contains unverified included events")
        else:
            written[event_table.name] = event_table
    else:
        failures.append("verified event_table.csv is missing")

    if failures:
        print("\nCOLLECTION FAILED. No downstream stage should run:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "retrieved_at_utc": retrieved_at,
        "analysis_start": START_DATE,
        "analysis_end": END_DATE,
        "sources": SOURCE_URLS,
        "files": {},
    }
    for filename, path in sorted(written.items()):
        frame = pd.read_csv(path)
        manifest["files"][filename] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "sha256": file_sha256(path),
        }
    manifest_path = RAW_DIR / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    print(f"\nCollection complete. Frozen manifest: {manifest_path}")


if __name__ == "__main__":
    args = parse_args()
    run_collection(args.start_date, args.end_date)
