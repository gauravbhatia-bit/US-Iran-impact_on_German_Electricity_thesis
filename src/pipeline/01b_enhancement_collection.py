from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

ENERGY_CHARTS = "https://api.energy-charts.info"
PORTWATCH = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)

PRICE_ZONES = [
    "DE-LU", "FR", "CH", "NO2", "SE4", "DK1", "DK2",
    "AT", "BE", "NL", "CZ", "PL", "IT-NORTH", "SI",
]
LOWER_DIRECT_GAS_DONORS = ["FR", "CH", "NO2", "SE4", "DK1", "DK2"]
USER_AGENT = "iran-energy-thesis/2.0 (academic replication)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(session: requests.Session, url: str, params: dict, attempts: int = 4):
    error = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=120)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 10))
                time.sleep(max(retry_after, 10) * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            return payload
        except Exception as exc:  # pragma: no cover - network failure path
            error = exc
            if attempt + 1 < attempts:
                time.sleep(max(3, 2 ** attempt))
    raise RuntimeError(f"Failed to fetch {url}: {error}")


def energy_price_frame(payload: dict, zone: str) -> pd.DataFrame:
    required = {"unix_seconds", "price", "unit", "license_info"}
    if missing := required - set(payload):
        raise ValueError(f"Energy-Charts {zone} response missing {sorted(missing)}")
    if len(payload["unix_seconds"]) != len(payload["price"]):
        raise ValueError(f"Energy-Charts {zone} timestamps/prices are misaligned")
    frame = pd.DataFrame({
        "datetime_utc": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True),
        "price_eur_mwh": pd.to_numeric(payload["price"], errors="coerce"),
    })
    frame["bidding_zone"] = zone
    frame["delivery_date"] = frame["datetime_utc"].dt.tz_convert(
        "Europe/Berlin"
    ).dt.date
    frame["license_info"] = payload["license_info"]
    if frame["datetime_utc"].duplicated().any():
        raise ValueError(f"Energy-Charts {zone} returned duplicate timestamps")
    if frame["price_eur_mwh"].notna().mean() < 0.90:
        raise ValueError(f"Energy-Charts {zone} has more than 10% missing prices")
    return frame


def named_series_frame(payload: dict, key: str, value_name: str) -> pd.DataFrame:
    if "unix_seconds" not in payload or key not in payload:
        raise ValueError(f"Energy-Charts response lacks unix_seconds/{key}")
    times = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    out = pd.DataFrame({"datetime_utc": times})
    for item in payload[key]:
        name = str(item.get("name", "")).strip()
        values = item.get("data", [])
        if not name or len(values) != len(times):
            raise ValueError(f"Misaligned {key} series {name!r}")
        out[name] = pd.to_numeric(pd.Series(values), errors="coerce")
    out["delivery_date"] = out["datetime_utc"].dt.tz_convert(
        "Europe/Berlin"
    ).dt.date
    if out["datetime_utc"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {value_name}")
    return out


def fetch_energy_charts(session: requests.Session, start: str, end: str):
    price_frames = []
    failures = {}
    for zone in PRICE_ZONES:
        try:
            payload = get_json(
                session, f"{ENERGY_CHARTS}/price",
                {"bzn": zone, "start": start, "end": end},
            )
            frame = energy_price_frame(payload, zone)
            if frame["delivery_date"].nunique() < 365:
                raise ValueError("less than one year of daily coverage")
            price_frames.append(frame)
            print(f"  price {zone:<8} {len(frame):>7,} intervals")
            time.sleep(2.0)
        except Exception as exc:
            failures[zone] = str(exc)
            print(f"  price {zone:<8} unavailable: {exc}")

    available = {f["bidding_zone"].iloc[0] for f in price_frames}
    lower_available = set(LOWER_DIRECT_GAS_DONORS) & available
    if "DE-LU" not in available or len(lower_available) < 4:
        raise RuntimeError(
            "The treated zone and at least four declared lower-direct-gas "
            "donors are required; available lower-exposure donors are "
            f"{sorted(lower_available)}; failures={failures}"
        )
    prices = pd.concat(price_frames, ignore_index=True)
    prices.to_csv(RAW / "energy_charts_prices.csv", index=False)

    power_payload = get_json(
        session, f"{ENERGY_CHARTS}/public_power",
        {"country": "de", "start": start, "end": end},
    )
    power = named_series_frame(power_payload, "production_types", "German system data")
    required_power = {"Fossil gas", "Load", "Residual load", "Wind offshore",
                      "Wind onshore", "Solar"}
    if missing := required_power - set(power):
        raise ValueError(f"German system data missing {sorted(missing)}")
    power.to_csv(RAW / "energy_charts_german_system.csv", index=False)

    flow_payload = get_json(
        session, f"{ENERGY_CHARTS}/cbpf",
        {"country": "de", "start": start, "end": end},
    )
    flows = named_series_frame(flow_payload, "countries", "German cross-border flows")
    if "sum" not in flows:
        raise ValueError("German cross-border response lacks aggregate sum")
    flows.to_csv(RAW / "energy_charts_cross_border.csv", index=False)
    return prices, power, flows, failures


def fetch_portwatch(session: requests.Session, start: str, end: str) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    where = (
        "portname='Strait of Hormuz' AND "
        f"date>='{start_ts.date()}' AND date<='{end_ts.date()}'"
    )
    payload = get_json(session, PORTWATCH, {
        "where": where,
        "outFields": "*",
        "orderByFields": "year ASC, month ASC, day ASC",
        "resultRecordCount": "5000",
        "returnGeometry": "false",
        "f": "json",
    })
    rows = [feature.get("attributes", {}) for feature in payload.get("features", [])]
    if not rows:
        raise RuntimeError("IMF PortWatch returned no Strait of Hormuz observations")
    frame = pd.DataFrame(rows)
    required = {"date", "n_total", "n_tanker", "n_cargo", "capacity"}
    if missing := required - set(frame):
        raise ValueError(f"IMF PortWatch response missing {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    numeric = ["n_total", "n_tanker", "n_cargo", "capacity"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if frame["n_total"].isna().any() or (frame["n_total"] < 0).any():
        raise ValueError("IMF PortWatch total transit counts are invalid")
    if frame["date"].max() < pd.Timestamp("2026-08-01"):
        raise ValueError("IMF PortWatch series is stale for the 2026 analysis")
    keep = ["date", "portid", "portname", "n_total", "n_tanker", "n_cargo",
            "capacity", "capacity_tanker", "capacity_cargo"]
    frame[[c for c in keep if c in frame]].to_csv(
        RAW / "imf_portwatch_hormuz.csv", index=False
    )
    print(f"  PortWatch {len(frame):,} daily observations through "
          f"{frame['date'].max().date()}")
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-09-12")
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="revalidate existing downloaded files and rebuild their manifest",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.manifest_only:
        prices = pd.read_csv(RAW / "energy_charts_prices.csv")
        power = pd.read_csv(RAW / "energy_charts_german_system.csv")
        flows = pd.read_csv(RAW / "energy_charts_cross_border.csv")
        portwatch = pd.read_csv(RAW / "imf_portwatch_hormuz.csv")
        failures = {"IT-NORTH": "unsupported historical request"}
        print("Revalidating existing enhancement inputs without network access")
    else:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        print("Collecting Fraunhofer Energy-Charts inputs")
        prices, power, flows, failures = fetch_energy_charts(
            session, args.start_date, args.end_date
        )
        print("Collecting IMF PortWatch Strait of Hormuz inputs")
        portwatch = fetch_portwatch(session, "2025-01-01", args.end_date)

    exact_events = RAW / "hourly_event_timestamps.csv"
    event_frame = pd.read_csv(exact_events)
    required_event = {"effective_at_utc", "event", "treatment_scope", "source_url"}
    if missing := required_event - set(event_frame):
        raise ValueError(f"Exact event table missing {sorted(missing)}")
    pd.to_datetime(event_frame["effective_at_utc"], utc=True, errors="raise")
    if not event_frame["source_url"].str.startswith("https://").all():
        raise ValueError("Every exact hourly event requires an HTTPS source")

    eia_flows = RAW / "eia_hormuz_quarterly.csv"
    eia_frame = pd.read_csv(eia_flows)
    required_eia = {
        "period", "oil_million_barrels_per_day",
        "lng_billion_cubic_feet_per_day", "source_url",
    }
    if missing := required_eia - set(eia_frame):
        raise ValueError(f"EIA quarterly flow table missing {sorted(missing)}")
    if not {"2025-Q4", "2026-Q2"}.issubset(set(eia_frame["period"])):
        raise ValueError("EIA quarterly flow table lacks validation endpoints")
    if not eia_frame["source_url"].str.startswith("https://www.eia.gov/").all():
        raise ValueError("EIA flow rows require an official EIA source URL")

    shipping_status = RAW / "shipping_status_evidence.csv"
    shipping_frame = pd.read_csv(shipping_status)
    required_shipping = {
        "observed_at_utc", "status", "scope", "exact_transition_time_verified",
        "source_org", "source_url", "source_note",
    }
    if missing := required_shipping - set(shipping_frame):
        raise ValueError(f"Shipping-status evidence missing {sorted(missing)}")
    pd.to_datetime(shipping_frame["observed_at_utc"], utc=True, errors="raise")
    if not shipping_frame["source_url"].str.startswith("https://").all():
        raise ValueError("Shipping-status evidence requires HTTPS source URLs")

    event_table = RAW / "hormuz_event_table.csv"
    event_frame = pd.read_csv(event_table)
    required_hormuz_event = {
        "event_id", "event_date", "status", "main_scope", "included_primary",
        "source_verified", "source_url", "exact_transition_time_verified",
    }
    if missing := required_hormuz_event - set(event_frame):
        raise ValueError(f"Hormuz event table missing {sorted(missing)}")
    pd.to_datetime(event_frame["event_date"], errors="raise")
    if not event_frame["source_verified"].astype(str).str.lower().isin(["true", "1", "yes"]).all():
        raise ValueError("Every Hormuz event row must be source verified")
    if not event_frame["source_url"].str.startswith("https://").all():
        raise ValueError("Hormuz event rows require HTTPS source URLs")
    allowed_states = {"open", "restricted", "severely_restricted"}
    primary_states = set(event_frame.loc[
        event_frame["included_primary"].astype(str).str.lower().isin(["true", "1", "yes"]),
        "status"
    ])
    if not primary_states.issubset(allowed_states):
        raise ValueError(f"Invalid primary Hormuz states: {sorted(primary_states - allowed_states)}")

    paths = [
        RAW / "energy_charts_prices.csv",
        RAW / "energy_charts_german_system.csv",
        RAW / "energy_charts_cross_border.csv",
        RAW / "imf_portwatch_hormuz.csv",
        exact_events,
        eia_flows,
        shipping_status,
        event_table,
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_start": args.start_date,
        "analysis_end": args.end_date,
        "sources": {
            "energy_charts": ENERGY_CHARTS,
            "portwatch": PORTWATCH,
            "eia_hormuz_flows": "https://www.eia.gov/outlooks/steo/report/energysecurity/article.php",
            "shipping_status": "Joint Maritime Information Center advisory hosted by IMO",
            "hormuz_event_table": "Source-labelled event table compiled from the validated event and shipping evidence files",
        },
        "declared_lower_direct_gas_donors": LOWER_DIRECT_GAS_DONORS,
        "unavailable_optional_price_zones": failures,
        "files": {
            path.name: {
                "rows": int(len(pd.read_csv(path))),
                "sha256": sha256(path),
            }
            for path in paths
        },
        "notes": [
            "Energy-Charts responses report CC BY 4.0 source licensing.",
            "PortWatch is AIS-derived; GPS jamming, spoofing and dark vessels can bias counts.",
            "The donor declaration is design metadata, not an empirical claim that controls are unaffected.",
            "Quarterly EIA oil/LNG flows are external validation and are not upsampled into daily regressions.",
            "The JMIC advisory verifies the Strait was open by 18 June 2026, not the exact reopening instant.",
            "The primary RQ2a exposure is the three-state source event table; PortWatch traffic is robustness-only.",
        ],
    }
    tmp = RAW / "enhancement_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(RAW / "enhancement_manifest.json")
    print(f"Enhancement source manifest: {RAW / 'enhancement_manifest.json'}")
    print(f"Rows: prices={len(prices):,}, system={len(power):,}, "
          f"flows={len(flows):,}, PortWatch={len(portwatch):,}")


if __name__ == "__main__":
    main()
