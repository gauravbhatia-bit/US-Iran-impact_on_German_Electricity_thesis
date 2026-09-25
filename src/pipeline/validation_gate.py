import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate():
    errors = []
    raw_manifest_path = RAW_DIR / "manifest.json"
    processed_manifest_path = PROCESSED_DIR / "manifest.json"
    master_path = PROCESSED_DIR / "master_features.csv"

    for path in (raw_manifest_path, processed_manifest_path, master_path):
        if not path.exists():
            errors.append(f"missing required artifact: {path}")
    if errors:
        return errors, {}

    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    processed_manifest = json.loads(processed_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("status") != "complete":
        errors.append("raw manifest is not complete")
    if processed_manifest.get("status") != "complete":
        errors.append("processed manifest is not complete")

    for filename, metadata in raw_manifest.get("files", {}).items():
        path = RAW_DIR / filename
        if not path.exists():
            errors.append(f"manifested raw file is missing: {filename}")
        elif sha256(path) != metadata.get("sha256"):
            errors.append(f"raw file changed after collection: {filename}")

    if processed_manifest.get("source_manifest_sha256") != sha256(raw_manifest_path):
        errors.append("processed table was not built from the current raw manifest")
    if processed_manifest.get("master_features_sha256") != sha256(master_path):
        errors.append("master_features.csv changed after preprocessing")

    df = pd.read_csv(master_path, parse_dates=["date"])
    required = {
        "date", "smard_mean", "ttf_eur_mwh", "gpr_daily",
        "renewable_output_mwh", "dunkelflaute_flag_lag1",
        "sentiment_available_lag1", "smard_mean_vol7d",
    }
    if missing := required - set(df.columns):
        errors.append(f"master table lacks required fields: {sorted(missing)}")
    if df["date"].duplicated().any():
        errors.append("master table contains duplicate dates")
    if len(df) > 1 and not df["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        errors.append("master table is not a complete daily calendar")
    if df["smard_mean"].isna().any():
        errors.append("target contains missing or fabricated gaps")

    # Recompute a representative lag and volatility field to catch accidental
    # reintroduction of the two leakage bugs that invalidated earlier results.
    if "smard_mean_lag1" in df:
        observed = df["smard_mean_lag1"].iloc[1:].to_numpy(float)
        expected = df["smard_mean"].shift(1).iloc[1:].to_numpy(float)
        if not np.allclose(observed, expected, equal_nan=True):
            errors.append("smard_mean_lag1 is not a strict one-day lag")
    if "smard_mean_vol7d" in df:
        expected = df["smard_mean"].shift(1).rolling(7).std()
        if not np.allclose(df["smard_mean_vol7d"], expected, equal_nan=True):
            errors.append("smard_mean_vol7d contains current/future target information")

    stats = {
        "rows": len(df), "columns": len(df.columns),
        "start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
    }
    return errors, stats


def main():
    errors, stats = validate()
    report = ["# Pipeline Validation Gate", ""]
    if errors:
        report += ["Status: **FAILED**", ""] + [f"- {item}" for item in errors]
    else:
        report += ["Status: **PASSED**", "",
                   f"Validated {stats['rows']} rows and {stats['columns']} columns, "
                   f"{stats['start']} through {stats['end']}."]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROCESSED_DIR / "PIPELINE_AUDIT_REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"Report: {report_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
