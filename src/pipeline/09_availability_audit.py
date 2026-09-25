from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


POLICY = [
    ("electricity_target", "smard_mean", "historical delivery outcome", 1, "lag1"),
    ("ttf_close", "ttf_eur_mwh", "manual daily close; no intraday timestamp", 2, "lag2"),
    ("brent_close", "DCOILBRENTEU", "FRED daily close; no pre-auction timestamp", 2, "lag2"),
    ("eurusd_close", "DEXUSEU", "FRED daily reference; no pre-auction timestamp", 2, "lag2"),
    ("gpr_daily", "gpr_daily", "daily index; publication time unavailable", 2, "lag2"),
    ("guardian_news", "guardian_sentiment_mean", "date-only article export", 2, "lag2"),
    ("eua_auction", "eua_eur_tco2", "auction result date; publication time unavailable", 2, "lag2"),
    ("hormuz_traffic", "traffic_shortfall", "daily observed traffic; post-cutoff timing unavailable", 2, "lag2"),
]


def main() -> None:
    master = pd.read_csv(PROCESSED / "master_features.csv", nrows=5)
    rows = []
    for source, raw_name, evidence, safe_lag, selected in POLICY:
        lag_columns = [f"{raw_name}_lag{i}" for i in range(1, 8)]
        existing = [c for c in lag_columns if c in master.columns]
        rows.append({
            "source": source,
            "raw_feature": raw_name,
            "availability_evidence": evidence,
            "safe_lag_days": safe_lag,
            "selected_forecast_lag": selected,
            "lag_columns_present": ";".join(existing),
            "source_file_present": bool(
                raw_name != "eua_eur_tco2" or (RAW / "eua_daily.csv").exists()
            ),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(PROCESSED / "availability_audit.csv", index=False)
    diagnostics = {
        "policy": "daily date-only closes use lag 2; electricity history uses lag 1",
        "auction_cutoff": "EPEX day-ahead auction is treated as a pre-noon D-1 cutoff",
        "safe_lag_features": {row[0]: row[4] for row in POLICY},
        "raw_manifest_present": (RAW / "manifest.json").exists(),
    }
    (PROCESSED / "availability_audit.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"Availability audit complete: {len(audit)} source policies")


if __name__ == "__main__":
    main()
