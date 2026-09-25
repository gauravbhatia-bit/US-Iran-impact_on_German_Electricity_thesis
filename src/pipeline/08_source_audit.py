from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def audit_ttf() -> tuple[pd.DataFrame, dict]:
    path = RAW / "Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    required = {"Date", "Price"}
    if not required.issubset(raw.columns):
        raise ValueError(f"TTF export is missing {sorted(required - set(raw.columns))}")
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="mixed", errors="raise"),
        "price": pd.to_numeric(raw["Price"], errors="raise"),
    }).sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any() or (out["price"] <= 0).any():
        raise ValueError("TTF dates must be unique and prices must be positive")
    out["previous_date"] = out["date"].shift(1)
    out["previous_price"] = out["price"].shift(1)
    out["calendar_gap_days"] = out["date"].diff().dt.days
    out["return_pct"] = out["price"].pct_change() * 100.0
    out["sep_2026_roll_window"] = out["date"].isin(
        pd.to_datetime(["2026-08-27", "2026-08-28"])
    )
    out["uk_bank_holiday_candidate"] = out["date"].isin(
        pd.to_datetime(["2026-08-31"])
    )
    window = out[out["date"].between("2026-08-25", "2026-09-01")].copy()
    diagnostics = {
        "source": path.name,
        "rows": int(len(out)),
        "first_date": str(out["date"].min().date()),
        "last_date": str(out["date"].max().date()),
        "august_31_2026_present": bool((out["date"] == pd.Timestamp("2026-08-31")).any()),
        "august_31_2026_price": (
            float(out.loc[out["date"].eq(pd.Timestamp("2026-08-31")), "price"].iloc[0])
            if (out["date"] == pd.Timestamp("2026-08-31")).any() else None
        ),
        "roll_window_max_abs_return_pct": (
            float(window["return_pct"].abs().max()) if not window.empty else None
        ),
        "roll_window": window[["date", "price", "return_pct"]].to_dict("records"),
        "contract_identifier_available": False,
        "interpretation": (
            "The manual export does not contain contract identifiers. The roll window "
            "is flagged for sensitivity analysis; no discontinuity is assumed."
        ),
    }
    return out, diagnostics


def audit_price_resolution() -> tuple[pd.DataFrame, dict]:
    path = RAW / "energy_charts_prices.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = ["datetime_utc", "price_eur_mwh", "bidding_zone", "delivery_date"]
    raw = pd.read_csv(path, usecols=usecols)
    raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"], utc=True, errors="raise")
    raw["delivery_date"] = pd.to_datetime(raw["delivery_date"], errors="raise").dt.normalize()
    raw["price_eur_mwh"] = pd.to_numeric(raw["price_eur_mwh"], errors="coerce")
    raw = raw.dropna(subset=["price_eur_mwh"])
    raw["local_datetime"] = raw["datetime_utc"].dt.tz_convert("Europe/Berlin")
    raw["local_date"] = raw["local_datetime"].dt.tz_localize(None).dt.normalize()
    raw["local_hour"] = raw["local_datetime"].dt.hour
    raw["utc_hour"] = raw["datetime_utc"].dt.floor("h")
    raw["period"] = np.where(raw["delivery_date"] < pd.Timestamp("2025-10-01"), "hourly", "quarter_hour")

    counts = (raw.groupby(["bidding_zone", "delivery_date"], as_index=False)
              .agg(interval_count=("price_eur_mwh", "size"),
                   direct_daily_mean=("price_eur_mwh", "mean")))
    hourly = (raw.groupby(["bidding_zone", "delivery_date", "utc_hour"], as_index=False)
              .agg(hourly_mean=("price_eur_mwh", "mean")))
    hourly_daily = (hourly.groupby(["bidding_zone", "delivery_date"], as_index=False)
                    .agg(hourly_daily_mean=("hourly_mean", "mean")))
    checks = counts.merge(
        hourly_daily, on=["bidding_zone", "delivery_date"], how="left",
    )
    checks["mean_difference"] = checks["direct_daily_mean"] - checks["hourly_daily_mean"]
    checks["complete_pre_switch"] = checks["interval_count"].isin([23, 24, 25])
    checks["complete_post_switch"] = checks["interval_count"].isin([92, 96, 100])
    checks["within_mean_tolerance"] = checks["mean_difference"].abs() <= 1e-9
    checks.to_csv(PROCESSED / "quarter_hour_resolution_audit.csv", index=False)
    de = checks[checks["bidding_zone"].eq("DE-LU")]
    diagnostics = {
        "source": path.name,
        "rows": int(len(raw)),
        "zones": sorted(raw["bidding_zone"].dropna().unique().tolist()),
        "pre_switch_counts": de.loc[de["delivery_date"] < "2025-10-01", "interval_count"].value_counts().to_dict(),
        "post_switch_counts": de.loc[de["delivery_date"] >= "2025-10-01", "interval_count"].value_counts().to_dict(),
        "max_abs_mean_difference": float(de["mean_difference"].abs().max()),
        "n_mean_invariance_failures": int((~de["within_mean_tolerance"]).sum()),
        "post_switch_indicator_required": True,
        "interpretation": (
            "Daily means are arithmetically invariant only when intervals are complete "
            "and local-day/DST handling is consistent. The hourly-to-quarter-hour market "
            "design change remains a sensitivity limitation."
        ),
    }
    return checks, diagnostics


def main() -> None:
    ttf, ttf_diag = audit_ttf()
    ttf.to_csv(PROCESSED / "ttf_roll_audit.csv", index=False)
    _, resolution_diag = audit_price_resolution()
    diagnostics = {"ttf": ttf_diag, "electricity_resolution": resolution_diag}
    (PROCESSED / "source_audit.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )
    print("Source audit complete")
    print(f"  TTF rows: {ttf_diag['rows']} ({ttf_diag['first_date']} to {ttf_diag['last_date']})")
    print(f"  TTF 25 Aug-1 Sep max absolute move: {ttf_diag['roll_window_max_abs_return_pct']:.3f}%")
    print(f"  Quarter-hour mean invariance failures: {resolution_diag['n_mean_invariance_failures']}")


if __name__ == "__main__":
    main()
