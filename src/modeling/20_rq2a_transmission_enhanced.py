from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CONFLICT_START = pd.Timestamp("2026-02-28")
TARGET = "smard_mean"
GAS = "ttf_eur_mwh"
MAX_LAG = 7
EIA_FLOWS = RAW / "eia_hormuz_quarterly.csv"
INSURANCE = RAW / "hormuz_insurance_daily.csv"
SHIPPING_STATUS = RAW / "shipping_status_evidence.csv"


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(values) - rank) * values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def hac_ols(frame, y, x, maxlags=7):
    data = frame[[y, *x]].dropna()
    if len(data) < max(50, len(x) * 5):
        raise ValueError(f"Insufficient rows for {y} on {x}: {len(data)}")
    return sm.OLS(data[y], sm.add_constant(data[x], has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )


def linear_combination(model, names):
    params = model.params.loc[names].to_numpy(dtype=float)
    cov = model.cov_params().loc[names, names].to_numpy(dtype=float)
    coef = float(params.sum())
    se = float(np.sqrt(np.ones(len(names)) @ cov @ np.ones(len(names))))
    z = coef / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return coef, se, p


def load_daily():
    master = pd.read_csv(PROCESSED / "master_features.csv", parse_dates=["date"])
    transits = pd.read_csv(RAW / "imf_portwatch_hormuz.csv", parse_dates=["date"])
    pre = transits[transits["date"] < CONFLICT_START]
    if len(pre) < 200:
        raise ValueError("PortWatch needs at least 200 pre-conflict days")
    baseline = float(pre["n_total"].median())
    transits["transit_ratio"] = transits["n_total"] / baseline
    transits["transit_shortfall"] = np.clip(1 - transits["transit_ratio"], 0, 1)
    transits["d_shortfall"] = transits["transit_shortfall"].diff()
    d = master.merge(
        transits[["date", "n_total", "n_tanker", "n_cargo", "capacity",
                  "transit_ratio", "transit_shortfall", "d_shortfall"]],
        on="date", how="inner",
    ).sort_values("date").reset_index(drop=True)

    system = pd.read_csv(RAW / "energy_charts_german_system.csv")
    flows = pd.read_csv(RAW / "energy_charts_cross_border.csv")
    system["delivery_date"] = pd.to_datetime(system["delivery_date"])
    flows["delivery_date"] = pd.to_datetime(flows["delivery_date"])
    system_daily = system.groupby("delivery_date", as_index=False).agg({
        "Load": "mean", "Residual load": "mean",
        "Wind offshore": "mean", "Wind onshore": "mean", "Solar": "mean",
    })
    system_daily["load_gw"] = system_daily["Load"] / 1000.0
    system_daily["residual_load_gw"] = system_daily["Residual load"] / 1000.0
    flow_daily = flows.groupby("delivery_date", as_index=False)["sum"].mean()
    flow_daily = flow_daily.rename(columns={"sum": "net_import_gw"})
    d = d.merge(system_daily[["delivery_date", "load_gw", "residual_load_gw"]],
                left_on="date", right_on="delivery_date", how="left").drop(columns="delivery_date")
    d = d.merge(flow_daily, left_on="date", right_on="delivery_date", how="left").drop(
        columns="delivery_date"
    )
    if d["date"].max() < pd.Timestamp("2026-08-01"):
        raise ValueError("Merged physical-disruption series is stale")
    return d, baseline


def distributed_lag_chain(d):
    frame = d.copy()
    frame["d_gas"] = frame[GAS].diff()
    frame["d_power"] = frame[TARGET].diff()
    frame["d_renew"] = frame["renewable_output_mwh"].diff() / 100000.0
    frame["d_brent"] = frame["DCOILBRENTEU"].diff()
    frame["d_load"] = frame["load_gw"].diff()
    frame["d_import"] = frame["net_import_gw"].diff()
    for lag in range(MAX_LAG + 1):
        frame[f"shortfall_L{lag}"] = frame["d_shortfall"].shift(lag)
        frame[f"gas_L{lag}"] = frame["d_gas"].shift(lag)
    for lag in range(1, MAX_LAG + 1):
        frame[f"gas_own_L{lag}"] = frame["d_gas"].shift(lag)
        frame[f"power_own_L{lag}"] = frame["d_power"].shift(lag)

    short_names = [f"shortfall_L{x}" for x in range(MAX_LAG + 1)]
    gas_own = [f"gas_own_L{x}" for x in range(1, MAX_LAG + 1)]
    # Brent is a parallel post-treatment energy-market response, so the
    # primary shortfall-to-TTF link does not condition on it.
    stage1_x = [*short_names, *gas_own]
    m1 = hac_ols(frame, "d_gas", stage1_x, maxlags=14)
    stage1 = linear_combination(m1, short_names)

    # German day-ahead prices for delivery day t are fixed before the same-day
    # TTF close. Start at lag 1 to preserve temporal ordering.
    gas_names = [f"gas_L{x}" for x in range(1, MAX_LAG + 1)]
    power_own = [f"power_own_L{x}" for x in range(1, MAX_LAG + 1)]
    stage2_x = [*gas_names, *power_own, "d_renew", "d_load", "d_import"]
    m2 = hac_ols(frame, "d_power", stage2_x, maxlags=14)
    stage2 = linear_combination(m2, gas_names)
    return frame, m1, m2, stage1, stage2


def local_projections(frame, exposure, outcome, horizons=range(15)):
    rows = []
    for horizon in horizons:
        work = frame.copy()
        # Cumulative change from the end of t-1 through t+h.
        work["future_change"] = work[outcome].shift(-horizon) - work[outcome].shift(1)
        controls = [exposure]
        for lag in [1, 2, 3, 7]:
            col = f"outcome_change_L{lag}"
            work[col] = work[outcome].diff().shift(lag)
            controls.append(col)
        model = hac_ols(work, "future_change", controls, maxlags=14)
        rows.append({
            "outcome": outcome,
            "exposure": exposure,
            "horizon_days": horizon,
            "coefficient": model.params[exposure],
            "se": model.bse[exposure],
            "p_value": model.pvalues[exposure],
            "ci_low": model.conf_int().loc[exposure, 0],
            "ci_high": model.conf_int().loc[exposure, 1],
            "n": int(model.nobs),
        })
    return pd.DataFrame(rows)


def disruption_threshold_sensitivity(d):
    work = d.copy()
    work["renew_scaled"] = work["renewable_output_mwh"] / 100000.0
    work["sin12"] = np.sin(2 * np.pi * work["month"] / 12)
    work["cos12"] = np.cos(2 * np.pi * work["month"] / 12)
    rows = []
    for threshold in np.arange(.50, .91, .05):
        work["disrupted"] = (work["transit_shortfall"] >= threshold).astype(int)
        model = hac_ols(
            work, TARGET,
            ["disrupted", "renew_scaled", "load_gw", "net_import_gw",
             "is_weekend", "sin12", "cos12"],
            maxlags=14,
        )
        rows.append({
            "shortfall_threshold": threshold,
            "affected_days": int(work["disrupted"].sum()),
            "adjusted_price_difference": model.params["disrupted"],
            "se": model.bse["disrupted"],
            "p_value": model.pvalues["disrupted"],
        })
    return pd.DataFrame(rows)


def gpr_comparator(d):
    work = d.copy()
    work["gpr_z"] = (work["gpr_daily"] - work["gpr_daily"].mean()) / work["gpr_daily"].std()
    work["renew_scaled"] = work["renewable_output_mwh"] / 100000.0
    work["sin12"] = np.sin(2 * np.pi * work["month"] / 12)
    work["cos12"] = np.cos(2 * np.pi * work["month"] / 12)
    x = ["transit_shortfall", "gpr_z", "renew_scaled", "load_gw",
         "net_import_gw", "is_weekend", "sin12", "cos12"]
    model = hac_ols(work, TARGET, x, maxlags=14)
    return {
        "transit_shortfall_coef": model.params["transit_shortfall"],
        "transit_shortfall_p": model.pvalues["transit_shortfall"],
        "gpr_coef": model.params["gpr_z"],
        "gpr_p": model.pvalues["gpr_z"],
        "correlation": work[["transit_shortfall", "gpr_z"]].dropna().corr().iloc[0, 1],
        "n": int(model.nobs),
        "interpretation": "GPR is a comparator for broad risk, never a physical-closure proxy",
    }


def hourly_marginality():
    prices = pd.read_csv(RAW / "energy_charts_prices.csv")
    prices = prices[prices["bidding_zone"] == "DE-LU"].copy()
    prices["datetime_utc"] = pd.to_datetime(prices["datetime_utc"], utc=True)
    prices["hour_utc"] = prices["datetime_utc"].dt.floor("h")
    prices = prices.groupby("hour_utc", as_index=False)["price_eur_mwh"].mean()

    system = pd.read_csv(RAW / "energy_charts_german_system.csv")
    system["datetime_utc"] = pd.to_datetime(system["datetime_utc"], utc=True)
    system["hour_utc"] = system["datetime_utc"].dt.floor("h")
    keep = ["Fossil gas", "Residual load", "Wind offshore", "Wind onshore", "Solar"]
    system = system.groupby("hour_utc", as_index=False)[keep].mean()
    hourly = prices.merge(system, on="hour_utc", how="inner")
    hourly["date"] = hourly["hour_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None).dt.normalize()
    ttf = pd.read_csv(PROCESSED / "master_features.csv", parse_dates=["date"])[["date", GAS]]
    hourly = hourly.merge(ttf, on="date", how="inner").sort_values("hour_utc")
    pre = hourly[hourly["date"] < CONFLICT_START]
    gas_cut = pre["Fossil gas"].quantile(.75)
    load_cut = pre["Residual load"].quantile(.75)
    hourly["gas_likely_marginal"] = (
        (hourly["Fossil gas"] >= gas_cut) & (hourly["Residual load"] >= load_cut)
    ).astype(int)
    hourly["d_price"] = hourly["price_eur_mwh"].diff()
    daily_gas = ttf.sort_values("date").copy()
    daily_gas["d_gas_lag1"] = daily_gas[GAS].diff().shift(1)
    hourly = hourly.merge(daily_gas[["date", "d_gas_lag1"]], on="date", how="left")
    hourly["interaction"] = hourly["d_gas_lag1"] * hourly["gas_likely_marginal"]
    hourly["d_renew_gw"] = (
        hourly[["Wind offshore", "Wind onshore", "Solar"]].sum(axis=1).diff() / 1000.0
    )
    local = hourly["hour_utc"].dt.tz_convert("Europe/Berlin")
    hour_dummies = pd.get_dummies(local.dt.hour, prefix="hour", drop_first=True, dtype=float)
    model_frame = pd.concat([hourly.reset_index(drop=True), hour_dummies.reset_index(drop=True)], axis=1)
    x = ["d_gas_lag1", "gas_likely_marginal", "interaction", "d_renew_gw",
         *hour_dummies.columns]
    data = model_frame[["d_price", *x, "date"]].dropna()
    model = sm.OLS(data["d_price"], sm.add_constant(data[x])).fit(
        cov_type="cluster", cov_kwds={"groups": data["date"]}
    )
    return {
        "gas_base_coef": model.params["d_gas_lag1"],
        "gas_base_p": model.pvalues["d_gas_lag1"],
        "gas_likely_increment": model.params["interaction"],
        "gas_likely_increment_p": model.pvalues["interaction"],
        "gas_likely_total": model.params["d_gas_lag1"] + model.params["interaction"],
        "gas_generation_p75_mw": gas_cut,
        "residual_load_p75_mw": load_cut,
        "gas_likely_share": data["gas_likely_marginal"].mean(),
        "n_hours": int(model.nobs),
        "n_day_clusters": int(data["date"].nunique()),
    }


def external_flow_validation():
    flows = pd.read_csv(EIA_FLOWS)
    required = {
        "period", "oil_million_barrels_per_day",
        "lng_billion_cubic_feet_per_day", "source_url",
    }
    if missing := required - set(flows):
        raise ValueError(f"EIA quarterly validation file missing {sorted(missing)}")
    indexed = flows.set_index("period")
    for period in ["2025-Q4", "2026-Q2"]:
        if period not in indexed.index:
            raise ValueError(f"EIA quarterly validation lacks {period}")
    q4, q2 = indexed.loc["2025-Q4"], indexed.loc["2026-Q2"]
    return {
        "comparison": "2026-Q2 versus 2025-Q4",
        "lng_change_percent": float(
            100 * (q2["lng_billion_cubic_feet_per_day"] /
                   q4["lng_billion_cubic_feet_per_day"] - 1)
        ),
        "oil_change_percent": float(
            100 * (q2["oil_million_barrels_per_day"] /
                   q4["oil_million_barrels_per_day"] - 1)
        ),
        "source_url": str(q2["source_url"]),
        "use": "external quarterly validation only; not expanded into daily rows",
    }


def insurance_data_status():
    if not INSURANCE.exists():
        return {
            "available": False,
            "reason": (
                "No auditable daily war-risk premium series was available; "
                "the analysis omits it rather than fabricating or interpolating values."
            ),
            "expected_path": str(INSURANCE.relative_to(ROOT)),
            "template": "data/templates/hormuz_insurance_daily_template.csv",
        }
    insurance = pd.read_csv(INSURANCE)
    required = {"date", "war_risk_premium_usd", "source_url", "source_note"}
    if missing := required - set(insurance):
        raise ValueError(f"Insurance series missing {sorted(missing)}")
    insurance["date"] = pd.to_datetime(insurance["date"], errors="raise")
    insurance["war_risk_premium_usd"] = pd.to_numeric(
        insurance["war_risk_premium_usd"], errors="raise"
    )
    if insurance["date"].duplicated().any() or (insurance["war_risk_premium_usd"] < 0).any():
        raise ValueError("Insurance series has duplicate dates or negative premiums")
    return {
        "available": True,
        "rows": int(len(insurance)),
        "first_date": str(insurance["date"].min().date()),
        "last_date": str(insurance["date"].max().date()),
        "use": "validated optional context; not included automatically in the primary estimand",
    }


def shipping_status_validation():
    evidence = pd.read_csv(SHIPPING_STATUS)
    required = {
        "observed_at_utc", "status", "scope", "exact_transition_time_verified",
        "source_org", "source_url", "source_note",
    }
    if missing := required - set(evidence):
        raise ValueError(f"Shipping-status evidence missing {sorted(missing)}")
    evidence["observed_at_utc"] = pd.to_datetime(
        evidence["observed_at_utc"], utc=True, errors="raise"
    )
    latest = evidence.sort_values("observed_at_utc").iloc[-1]
    exact_value = str(latest["exact_transition_time_verified"]).strip().lower()
    return {
        "observed_at_utc": latest["observed_at_utc"].isoformat(),
        "status": str(latest["status"]),
        "scope": str(latest["scope"]),
        "exact_transition_time_verified": exact_value in {"true", "1", "yes"},
        "source_org": str(latest["source_org"]),
        "source_url": str(latest["source_url"]),
        "interpretation": (
            "independent status observation only; not treated as an exact reopening timestamp"
        ),
    }


def plot_local_projections(lp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, outcome, title in zip(
        axes, [GAS, TARGET], ["TTF response", "German power response"]
    ):
        s = lp[lp["outcome"] == outcome]
        ax.axhline(0, color="black", lw=.8)
        ax.fill_between(s["horizon_days"], s["ci_low"], s["ci_high"], alpha=.2)
        ax.plot(s["horizon_days"], s["coefficient"], marker="o", ms=3)
        ax.set_title(title)
        ax.set_xlabel("Days after a one-unit increase in traffic shortfall")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Cumulative EUR/MWh change")
    fig.suptitle("Local projections: physical Hormuz traffic disruption")
    fig.tight_layout()
    path = RESULTS / "rq2a_local_projections.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    daily, baseline = load_daily()
    frame, m1, m2, stage1, stage2 = distributed_lag_chain(daily)
    lp_gas = local_projections(frame, "d_shortfall", GAS)
    lp_power = local_projections(frame, "d_shortfall", TARGET)
    lp = pd.concat([lp_gas, lp_power], ignore_index=True)
    lp["p_value_holm"] = holm_adjust(lp["p_value"])
    lp["significant_holm"] = lp["p_value_holm"] < .05
    lp.to_csv(RESULTS / "rq2a_local_projections.csv", index=False)
    sensitivity = disruption_threshold_sensitivity(daily)
    sensitivity["p_value_holm"] = holm_adjust(sensitivity["p_value"])
    sensitivity["significant_holm"] = sensitivity["p_value_holm"] < .05
    sensitivity.to_csv(RESULTS / "rq2a_disruption_sensitivity.csv", index=False)
    comparator = gpr_comparator(daily)
    marginal = hourly_marginality()
    flow_validation = external_flow_validation()
    insurance_status = insurance_data_status()
    shipping_status = shipping_status_validation()

    stage_rows = pd.DataFrame([
        {"stage": "PortWatch shortfall to TTF", "lags": f"0-{MAX_LAG}",
         "cumulative_coefficient": stage1[0], "se": stage1[1], "p_value": stage1[2],
         "n": int(m1.nobs)},
        {"stage": "TTF to German power", "lags": f"1-{MAX_LAG}",
         "cumulative_coefficient": stage2[0], "se": stage2[1], "p_value": stage2[2],
         "n": int(m2.nobs)},
    ])
    stage_rows["p_value_holm"] = holm_adjust(stage_rows["p_value"])
    stage_rows["significant_holm"] = stage_rows["p_value_holm"] < .05
    stage_rows.to_csv(RESULTS / "rq2a_distributed_lag_chain.csv", index=False)
    diagnostics = {
        "pre_conflict_median_daily_transits": baseline,
        "observed_min_daily_transits": int(daily["n_total"].min()),
        "observed_max_shortfall": float(daily["transit_shortfall"].max()),
        "gpr_comparator": {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in comparator.items()},
        "hourly_marginality": {k: float(v) if isinstance(v, (np.floating, float)) else v
                               for k, v in marginal.items()},
        "external_eia_flow_validation": flow_validation,
        "insurance_cost_measure": insurance_status,
        "shipping_status_validation": shipping_status,
        "limitations": [
            "One conflict episode: estimates are associative, not structurally causal.",
            "AIS jamming, spoofing and dark vessels can bias PortWatch traffic counts.",
            "Gas-likely marginal hours are a transparent proxy, not plant-level bid-stack proof.",
        ],
    }
    (RESULTS / "rq2a_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    chart = plot_local_projections(lp)

    significant_thresholds = int(sensitivity["significant_holm"].sum())
    significant_lp = int(lp["significant_holm"].sum())
    lines = [
        "# Enhanced RQ2a physical transmission analysis", "",
        "Primary exposure is continuous IMF PortWatch vessel-traffic shortfall, not a binary "
        "closure date and not GPR.", "", "## Physical measure", "",
        f"- Pre-conflict median: {baseline:.0f} vessel transits/day",
        f"- Maximum measured traffic shortfall: {100 * daily['transit_shortfall'].max():.1f}%",
        "- PortWatch is AIS-derived; jamming, spoofing and dark vessels remain measurement risks.",
        f"- {shipping_status['source_org']} independently recorded the Strait as "
        f"{shipping_status['status'].replace('_', ' ')} on "
        f"{shipping_status['observed_at_utc'][:10]}; this verifies status by that date, "
        "not the exact reopening instant.",
        "", "## Distributed-lag chain", "",
        f"- Traffic shortfall → TTF, cumulative lags 0-{MAX_LAG}: {stage1[0]:+.3f} "
        f"(SE {stage1[1]:.3f}, p={stage1[2]:.4f}, "
        f"Holm p={stage_rows.iloc[0]['p_value_holm']:.4f})",
        f"- Previous-day TTF → German power, cumulative lags 1-{MAX_LAG}: {stage2[0]:+.3f} "
        f"(SE {stage2[1]:.3f}, p={stage2[2]:.4f}, "
        f"Holm p={stage_rows.iloc[1]['p_value_holm']:.4f})", "",
        "The two coefficients are reported separately. Their product is not labelled a causal "
        "mediation effect because sequential ignorability is not credible in one conflict.", "",
        "## Independent energy-flow validation", "",
        f"- EIA Hormuz LNG flow changed {flow_validation['lng_change_percent']:.1f}% and oil flow "
        f"changed {flow_validation['oil_change_percent']:.1f}% in "
        f"{flow_validation['comparison']}.",
        "- These quarterly observations validate direction and scale only; they are not copied "
        "into daily regressions.",
        ("- Daily insurance costs are available and schema-validated."
         if insurance_status["available"] else
         "- Daily war-risk insurance costs remain omitted because no auditable open series was "
         "available; a source-labelled import template is provided."), "",
        "## Local projections", "",
        "Responses for horizons 0-14 days are in `rq2a_local_projections.csv`; they allow delayed "
        "transmission rather than forcing a same-day relationship.",
        f"After Holm correction, {significant_lp}/{len(lp)} local-projection coefficients are significant.", "",
        "## Gas-likely marginal hours", "",
        f"- Base previous-day gas-change coefficient: {marginal['gas_base_coef']:+.3f} "
        f"(p={marginal['gas_base_p']:.4f})",
        f"- Additional coefficient in high-gas/high-residual-load hours: "
        f"{marginal['gas_likely_increment']:+.3f} "
        f"(p={marginal['gas_likely_increment_p']:.4f})",
        f"- Total in gas-likely hours: {marginal['gas_likely_total']:+.3f}",
        f"- Observations: {marginal['n_hours']:,} hours in "
        f"{marginal['n_day_clusters']:,} day clusters", "",
        "## Sensitivity and GPR", "",
        f"- Continuous-shortfall thresholds from 50% to 90% were tested; "
        f"{significant_thresholds}/{len(sensitivity)} price-level specifications remain significant "
        "after Holm correction. These level regressions are vulnerable to period confounding.",
        f"- Joint comparator: traffic shortfall p={comparator['transit_shortfall_p']:.4f}; "
        f"GPR p={comparator['gpr_p']:.4f}; correlation={comparator['correlation']:.3f}.",
        "- GPR is retained only as a broad-risk comparator, never interpreted as physical closure.",
        "", "The earlier reopening-date sensitivity remains available in `hormuz_transmission.csv`; "
        "the continuous specification no longer requires one reopening date to define exposure.",
        "", f"Chart: `{chart.name}`",
    ]
    (RESULTS / "rq2a_enhanced_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Enhanced RQ2a complete")
    print(json.dumps({"baseline_transits": baseline, "stage1": stage1,
                      "stage2": stage2, "marginality": marginal,
                      "thresholds_significant_holm": significant_thresholds,
                      "local_projections_significant_holm": significant_lp}, indent=2))


if __name__ == "__main__":
    main()
