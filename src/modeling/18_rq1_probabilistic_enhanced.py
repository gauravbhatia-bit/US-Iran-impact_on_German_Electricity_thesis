from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = "smard_mean"
GAS = "ttf_eur_mwh"
HORIZON_WEEKS = 13
N_SIMS = 4000
BACKTEST_SIMS = 600
SEED = 42
CONFLICT_START = pd.Timestamp("2026-02-28")

SYSTEM_FEATURES = ["renew_gw", "load_gw", "net_import_gw"]
MODEL_FEATURES = ["gas", *SYSTEM_FEATURES, "sin52", "cos52"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def load_inputs():
    daily = pd.read_csv(PROCESSED / "master_features.csv", parse_dates=["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    system = pd.read_csv(RAW / "energy_charts_german_system.csv")
    flows = pd.read_csv(RAW / "energy_charts_cross_border.csv")
    prices = pd.read_csv(RAW / "energy_charts_prices.csv")
    for frame in (system, flows, prices):
        frame["delivery_date"] = pd.to_datetime(frame["delivery_date"])

    system_daily = system.groupby("delivery_date", as_index=False).agg({
        "Wind offshore": "mean",
        "Wind onshore": "mean",
        "Solar": "mean",
        "Load": "mean",
    })
    system_daily["renew_gw"] = (
        system_daily["Wind offshore"] + system_daily["Wind onshore"]
        + system_daily["Solar"]
    ) / 1000.0
    system_daily["load_gw"] = system_daily["Load"] / 1000.0
    flow_daily = flows.groupby("delivery_date", as_index=False)["sum"].mean()
    flow_daily = flow_daily.rename(columns={"sum": "net_import_gw"})

    daily = daily.merge(
        system_daily[["delivery_date", "renew_gw", "load_gw"]],
        left_on="date", right_on="delivery_date", how="left",
    ).drop(columns="delivery_date")
    daily = daily.merge(
        flow_daily, left_on="date", right_on="delivery_date", how="left",
    ).drop(columns="delivery_date")

    # A source-consistency gate prevents a treated series from one provider and
    # donor series from another being mixed without verification.
    de = prices[prices["bidding_zone"] == "DE-LU"].groupby(
        "delivery_date", as_index=False
    )["price_eur_mwh"].mean()
    check = daily[["date", TARGET]].merge(
        de, left_on="date", right_on="delivery_date", how="inner"
    )
    corr = float(check[TARGET].corr(check["price_eur_mwh"]))
    mae = float(np.mean(np.abs(check[TARGET] - check["price_eur_mwh"])))
    if corr < 0.995 or mae > 2.0:
        raise ValueError(
            f"SMARD/Energy-Charts treated-price mismatch: corr={corr:.4f}, MAE={mae:.3f}"
        )
    return daily, {"correlation": corr, "mae": mae, "n": len(check)}


def weekly_frame(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.set_index("date")
    required = [TARGET, GAS, *SYSTEM_FEATURES]
    weekly = d[required].resample("W-SUN").agg(["mean", "count"])
    means = weekly.xs("mean", axis=1, level=1)
    counts = weekly.xs("count", axis=1, level=1)
    complete = counts[[TARGET, GAS]].min(axis=1) >= 7
    means = means.loc[complete].dropna().reset_index().rename(columns={
        TARGET: "power", GAS: "gas",
    })
    means["weekofyear"] = means["date"].dt.isocalendar().week.astype(int)
    means["sin52"] = np.sin(2 * np.pi * means["weekofyear"] / 52.18)
    means["cos52"] = np.cos(2 * np.pi * means["weekofyear"] / 52.18)
    return means


def fit_weekly_arx(weekly: pd.DataFrame):
    d = weekly.copy()
    d["power_lag1"] = d["power"].shift(1)
    cols = ["power", "power_lag1", *MODEL_FEATURES]
    d = d.dropna(subset=cols)
    if len(d) < 80:
        raise ValueError("At least 80 complete historical weeks are required")

    scale_cols = ["gas", *SYSTEM_FEATURES]
    means = d[scale_cols].mean()
    stds = d[scale_cols].std().replace(0, 1.0)
    x = pd.DataFrame(index=d.index)
    x["power_lag1"] = d["power_lag1"]
    for col in scale_cols:
        x[col] = (d[col] - means[col]) / stds[col]
    x[["sin52", "cos52"]] = d[["sin52", "cos52"]]
    model = sm.OLS(d["power"], sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4}
    )
    rho = float(model.params["power_lag1"])
    if not (-0.25 < rho < 0.995):
        raise ValueError(f"Weekly ARX is not mean reverting: rho={rho:.4f}")
    return {
        "model": model,
        "data": d,
        "scale_mean": means,
        "scale_std": stds,
        "rho": rho,
        "residuals": np.asarray(model.resid, dtype=float),
    }


def sample_blocks(residuals, n_sims, horizon, block_length, rng):
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < block_length * 3:
        raise ValueError("Too few residuals for moving-block bootstrap")
    values = values - values.mean()
    out = np.empty((n_sims, horizon))
    for i in range(n_sims):
        seq = []
        while len(seq) < horizon:
            start = int(rng.integers(0, len(values)))
            seq.extend(values[(start + j) % len(values)] for j in range(block_length))
        out[i] = seq[:horizon]
    return out


def crps_ensemble(draws, observed):
    x = np.sort(np.asarray(draws, dtype=float))
    n = len(x)
    first = np.mean(np.abs(x - observed))
    weights = 2 * np.arange(1, n + 1) - n - 1
    second = np.sum(weights * x) / (n * n)
    return float(first - second)


def scenario_anchors(weekly: pd.DataFrame):
    recent = weekly["gas"].tail(4)
    before = weekly.loc[weekly["date"] < CONFLICT_START, "gas"].tail(9)
    closed = weekly.loc[weekly["date"].between("2026-03-02", "2026-06-17"), "gas"]
    current = float(weekly["gas"].iloc[-1])
    recent_mean = float(recent.mean())
    baseline = float(before.mean())
    closed_mean = float(closed.mean())
    premium = max(closed_mean - baseline, 0.0)
    volatility = {
        "renewed_restriction": float(closed.diff().std()),
        "status_quo": float(recent.diff().std()),
        "normalisation": float(before.diff().std()),
    }
    fallback = float(weekly["gas"].diff().tail(52).std())
    volatility = {k: (v if np.isfinite(v) and v > 0 else fallback)
                  for k, v in volatility.items()}
    return {
        "current": current,
        "recent": recent_mean,
        "baseline": baseline,
        "closed_mean": closed_mean,
        "closure_premium": premium,
        "targets": {
            "renewed_restriction": current + premium,
            "status_quo": current,
            "normalisation": min(baseline, current),
        },
        "scenario_labels": {
            "renewed_restriction": "Renewed restriction",
            "status_quo": "Status quo",
            "normalisation": "Normalisation",
        },
        "volatility": volatility,
    }


def simulate_gas(start, target, volatility, n_sims, horizon, rng):
    paths = np.zeros((n_sims, horizon))
    value = np.full(n_sims, start, dtype=float)
    kappa = 0.25
    for h in range(horizon):
        value += kappa * (target - value) + rng.normal(0, volatility, n_sims)
        value = np.maximum(value, 0.0)
        paths[:, h] = value
    return paths


def future_weeks(last_date, horizon=HORIZON_WEEKS):
    first = last_date + pd.offsets.Week(weekday=6)
    return pd.date_range(first, periods=horizon, freq="W-SUN")


def seasonal_system_paths(history, dates, n_sims, rng):
    result = {col: np.empty((n_sims, len(dates))) for col in SYSTEM_FEATURES}
    week_numbers = history["weekofyear"].to_numpy()
    for h, date in enumerate(dates):
        target_week = int(date.isocalendar().week)
        circular_distance = np.minimum(
            np.abs(week_numbers - target_week), 52 - np.abs(week_numbers - target_week)
        )
        pool = history.loc[circular_distance <= 3, SYSTEM_FEATURES].dropna()
        if len(pool) < 10:
            pool = history[SYSTEM_FEATURES].dropna()
        sampled = pool.iloc[rng.integers(0, len(pool), n_sims)]
        for col in SYSTEM_FEATURES:
            result[col][:, h] = sampled[col].to_numpy()
    return result


def parameter_draws(fit, n_sims, rng):
    model = fit["model"]
    cov = np.asarray(model.cov_params(), dtype=float)
    cov = (cov + cov.T) / 2
    eigval, eigvec = np.linalg.eigh(cov)
    cov = eigvec @ np.diag(np.maximum(eigval, 0)) @ eigvec.T
    draws = rng.multivariate_normal(np.asarray(model.params), cov, size=n_sims)
    out = {name: draws[:, i] for i, name in enumerate(model.params.index)}
    out["power_lag1"] = np.clip(out["power_lag1"], -0.20, 0.98)
    return out


def simulate_power(fit, last_power, dates, gas_paths, system_paths, rng,
                   n_sims, residual_block=2):
    params = parameter_draws(fit, n_sims, rng)
    residuals = sample_blocks(
        fit["residuals"], n_sims, len(dates), residual_block, rng
    )
    power = np.zeros((n_sims, len(dates)))
    previous = np.full(n_sims, last_power, dtype=float)
    means, stds = fit["scale_mean"], fit["scale_std"]
    for h, date in enumerate(dates):
        value = params["const"] + params["power_lag1"] * previous
        raw = {"gas": gas_paths[:, h]}
        raw.update({col: system_paths[col][:, h] for col in SYSTEM_FEATURES})
        for col in ["gas", *SYSTEM_FEATURES]:
            value += params[col] * ((raw[col] - means[col]) / stds[col])
        week = int(date.isocalendar().week)
        value += params["sin52"] * np.sin(2 * np.pi * week / 52.18)
        value += params["cos52"] * np.cos(2 * np.pi * week / 52.18)
        value += residuals[:, h]
        power[:, h] = value
        previous = value
    return power


def backtest(weekly, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    # Quarterly, non-overlapping 13-week pseudo-scenarios; all estimation for
    # an origin uses only data at or before that origin.
    last_pre_conflict = weekly.index[weekly["date"] < CONFLICT_START][-1]
    origins = range(104, last_pre_conflict - HORIZON_WEEKS + 1, 13)
    for origin in origins:
        train = weekly.iloc[:origin + 1].copy()
        actual = weekly.iloc[origin + 1: origin + 1 + HORIZON_WEEKS].copy()
        if len(actual) < HORIZON_WEEKS:
            continue
        fit = fit_weekly_arx(train)
        dates = pd.DatetimeIndex(actual["date"])
        gas = np.tile(actual["gas"].to_numpy(), (BACKTEST_SIMS, 1))
        systems = {
            col: np.tile(actual[col].to_numpy(), (BACKTEST_SIMS, 1))
            for col in SYSTEM_FEATURES
        }
        sims = simulate_power(
            fit, float(train["power"].iloc[-1]), dates, gas, systems, rng,
            BACKTEST_SIMS,
        )
        for h, observed in enumerate(actual["power"].to_numpy()):
            q5, q50, q95 = np.quantile(sims[:, h], [0.05, 0.50, 0.95])
            rows.append({
                "origin": train["date"].iloc[-1].date(),
                "horizon_week": h + 1,
                "target_week": dates[h].date(),
                "actual": observed,
                "q05": q5,
                "q50": q50,
                "q95": q95,
                "covered_90": q5 <= observed <= q95,
                "crps": crps_ensemble(sims[:, h], observed),
                "backtest_type": "conditional_on_realised_gas_and_system_paths",
            })
    return pd.DataFrame(rows)


def backtest_generated_drivers(weekly, seed=SEED + 101):
    rng = np.random.default_rng(seed)
    rows = []
    last_pre_conflict = weekly.index[weekly["date"] < CONFLICT_START][-1]
    origins = range(104, last_pre_conflict - HORIZON_WEEKS + 1, 13)
    for origin in origins:
        train = weekly.iloc[:origin + 1].copy()
        actual = weekly.iloc[origin + 1: origin + 1 + HORIZON_WEEKS].copy()
        if len(actual) < HORIZON_WEEKS:
            continue
        fit = fit_weekly_arx(train)
        dates = pd.DatetimeIndex(actual["date"])
        gas_start = float(train["gas"].iloc[-1])
        gas_target = float(train["gas"].tail(9).mean())
        gas_vol = float(train["gas"].diff().tail(52).std())
        if not np.isfinite(gas_vol) or gas_vol <= 0:
            gas_vol = float(train["gas"].diff().std())
        gas_paths = simulate_gas(
            gas_start, gas_target, gas_vol, BACKTEST_SIMS, HORIZON_WEEKS, rng
        )
        systems = seasonal_system_paths(train, dates, BACKTEST_SIMS, rng)
        sims = simulate_power(
            fit, float(train["power"].iloc[-1]), dates, gas_paths, systems, rng,
            BACKTEST_SIMS,
        )
        for h, observed in enumerate(actual["power"].to_numpy()):
            q5, q50, q95 = np.quantile(sims[:, h], [0.05, 0.50, 0.95])
            rows.append({
                "origin": train["date"].iloc[-1].date(),
                "horizon_week": h + 1,
                "target_week": dates[h].date(),
                "actual": observed,
                "q05": q5, "q50": q50, "q95": q95,
                "covered_90": q5 <= observed <= q95,
                "crps": crps_ensemble(sims[:, h], observed),
                "backtest_type": "generated_gas_and_seasonal_system_paths",
            })
    return pd.DataFrame(rows)


def short_term_forecast(daily, rng, n_sims=N_SIMS):
    d = daily[["date", TARGET]].dropna().copy()
    for lag in [1, 2, 7, 14]:
        d[f"lag{lag}"] = d[TARGET].shift(lag)
    d["trend"] = np.arange(len(d), dtype=float) / len(d)
    d["sin7"] = np.sin(2 * np.pi * d["date"].dt.dayofweek / 7)
    d["cos7"] = np.cos(2 * np.pi * d["date"].dt.dayofweek / 7)
    cols = [f"lag{x}" for x in [1, 2, 7, 14]] + ["trend", "sin7", "cos7"]
    fit_data = d.dropna(subset=cols)
    model = sm.OLS(fit_data[TARGET], sm.add_constant(fit_data[cols])).fit()
    residuals = sample_blocks(model.resid, n_sims, 14, 7, rng)
    history = list(d[TARGET].to_numpy())
    paths = np.zeros((n_sims, 14))
    sim_history = np.tile(np.asarray(history[-14:], dtype=float), (n_sims, 1))
    dates = pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=14)
    for h, date in enumerate(dates):
        x = {
            "lag1": sim_history[:, -1], "lag2": sim_history[:, -2],
            "lag7": sim_history[:, -7], "lag14": sim_history[:, -14],
            "trend": np.full(n_sims, (len(d) + h) / len(d)),
            "sin7": np.full(n_sims, np.sin(2 * np.pi * date.dayofweek / 7)),
            "cos7": np.full(n_sims, np.cos(2 * np.pi * date.dayofweek / 7)),
        }
        value = np.full(n_sims, model.params["const"])
        for col in cols:
            value += model.params[col] * x[col]
        value += residuals[:, h]
        paths[:, h] = value
        sim_history = np.column_stack([sim_history[:, 1:], value])
    rows = []
    for h, date in enumerate(dates):
        qs = np.quantile(paths[:, h], QUANTILES)
        rows.append({"date": date.date(), "horizon_day": h + 1,
                     **{f"q{int(q*100):02d}": v for q, v in zip(QUANTILES, qs)}})
    return pd.DataFrame(rows)


def plot_weekly(projections, historical):
    colors = {"renewed_restriction": "tab:red", "status_quo": "tab:orange",
              "normalisation": "tab:green"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, sharey=True)
    hist = historical.tail(26)
    for ax, scenario in zip(axes, colors):
        s = projections[projections["scenario"] == scenario]
        ax.plot(hist["date"], hist["power"], color="black", lw=1, label="Observed weekly mean")
        ax.fill_between(s["week_end"], s["q05"], s["q95"], color=colors[scenario],
                        alpha=.18, label="90% interval")
        ax.fill_between(s["week_end"], s["q25"], s["q75"], color=colors[scenario],
                        alpha=.32, label="50% interval")
        ax.plot(s["week_end"], s["q50"], color=colors[scenario], lw=1.8,
                label="Median weekly mean")
        ax.set_title(scenario.replace("_", "-").title(), loc="left")
        ax.set_ylabel("EUR/MWh")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, ncol=4)
    fig.suptitle("RQ1 enhanced: weekly-average German day-ahead price scenarios")
    fig.tight_layout()
    path = RESULTS / "scenario_weekly_fan_chart.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    rng = np.random.default_rng(SEED)
    daily, source_check = load_inputs()
    weekly = weekly_frame(daily)
    fit = fit_weekly_arx(weekly)
    anchors = scenario_anchors(weekly)
    dates = future_weeks(weekly["date"].max())
    systems = seasonal_system_paths(weekly, dates, N_SIMS, rng)

    projection_rows = []
    end_draws = {}
    scenario_order = ["renewed_restriction", "status_quo", "normalisation"]
    for scenario in scenario_order:
        gas = simulate_gas(
            anchors["current"], anchors["targets"][scenario],
            anchors["volatility"][scenario], N_SIMS, HORIZON_WEEKS, rng,
        )
        power = simulate_power(
            fit, float(weekly["power"].iloc[-1]), dates, gas, systems, rng, N_SIMS
        )
        end_draws[scenario] = power[:, -1]
        for h, date in enumerate(dates):
            qs = np.quantile(power[:, h], QUANTILES)
            projection_rows.append({
                "scenario": scenario, "week_end": date,
                "horizon_week": h + 1,
                **{f"q{int(q*100):02d}": v for q, v in zip(QUANTILES, qs)},
                "mean": float(power[:, h].mean()),
            })
    projections = pd.DataFrame(projection_rows)
    projections.to_csv(RESULTS / "scenario_weekly_projections.csv", index=False)

    medians = {k: float(np.median(v)) for k, v in end_draws.items()}
    if not (medians["renewed_restriction"] >= medians["status_quo"] >= medians["normalisation"]):
        raise RuntimeError(f"Scenario ordering failed: {medians}")
    separation = float(np.mean(
        end_draws["renewed_restriction"] > np.median(end_draws["normalisation"])
    ))

    bt = backtest(weekly)
    if bt.empty:
        raise RuntimeError("No valid historical pseudo-scenario backtests")
    bt.to_csv(RESULTS / "scenario_weekly_backtest.csv", index=False)
    generated_bt = backtest_generated_drivers(weekly)
    if generated_bt.empty:
        raise RuntimeError("No valid generated-driver pseudo-scenario backtests")
    generated_bt.to_csv(RESULTS / "scenario_weekly_backtest_generated.csv", index=False)
    validation = {
        "n_pseudo_origins": int(bt["origin"].nunique()),
        "n_weekly_forecasts": int(len(bt)),
        "coverage_90": float(bt["covered_90"].mean()),
        "mean_crps": float(bt["crps"].mean()),
        "median_interval_width": float((bt["q95"] - bt["q05"]).median()),
        "backtest_type": str(bt["backtest_type"].iloc[0]),
        "generated_driver_coverage_90": float(generated_bt["covered_90"].mean()),
        "generated_driver_mean_crps": float(generated_bt["crps"].mean()),
        "generated_driver_n_weekly_forecasts": int(len(generated_bt)),
        "warning": (
            "This validates price-model uncertainty conditional on realised gas and system paths; "
            "it does not validate the political scenario probabilities."
        ),
    }
    (RESULTS / "scenario_weekly_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    short = short_term_forecast(daily, rng)
    short.to_csv(RESULTS / "short_term_14d_forecast.csv", index=False)
    chart = plot_weekly(projections, weekly)

    lines = [
        "# Enhanced RQ1 probabilistic scenarios", "",
        "The 90-day structural exercise now forecasts **weekly-average** German day-ahead prices. "
        "It is separate from the 14-day operational forecast.", "",
        "## Model", "",
        f"- Conditional mean-reverting weekly ARX coefficient: {fit['rho']:.3f}",
        "- Drivers: TTF gas, renewable generation, load, net cross-border flow and annual seasonality",
        "- Uncertainty: joint coefficient draws, stochastic gas paths, seasonally matched system paths, "
        "and two-week moving blocks of historical residuals",
        f"- SMARD/Energy-Charts consistency: correlation {source_check['correlation']:.5f}, "
        f"MAE {source_check['mae']:.3f} EUR/MWh", "",
        "## Scenario specification", "",
        f"- Origin TTF: {anchors['current']:.2f} EUR/MWh; recent four-week mean: {anchors['recent']:.2f}.",
        f"- Renewed restriction target: {anchors['targets']['renewed_restriction']:.2f} EUR/MWh "
        f"(origin plus estimated closure premium {anchors['closure_premium']:.2f}).",
        f"- Status quo target: {anchors['targets']['status_quo']:.2f} EUR/MWh (origin level).",
        f"- Normalisation target: {anchors['targets']['normalisation']:.2f} EUR/MWh "
        f"(pre-conflict benchmark {anchors['baseline']:.2f}).", "",
        "## Week 13 results", "",
    ]
    for scenario in scenario_order:
        draws = end_draws[scenario]
        lines.append(
            f"- **{scenario.replace('_', '-')}**: median {np.median(draws):.2f} EUR/MWh "
            f"(90% interval {np.quantile(draws, .05):.2f} to {np.quantile(draws, .95):.2f})"
        )
    lines += [
        "", f"Separation: {100 * separation:.1f}% of renewed-restriction draws exceed the "
        "normalisation median.", "",
        "## Historical pseudo-scenario validation", "",
        f"- Pseudo-origins: {validation['n_pseudo_origins']}",
        f"- Weekly forecasts evaluated: {validation['n_weekly_forecasts']}",
        f"- Empirical 90% coverage: {100 * validation['coverage_90']:.1f}%",
        f"- Mean CRPS: {validation['mean_crps']:.2f}",
        f"- Median 90% interval width: {validation['median_interval_width']:.2f} EUR/MWh", "",
        "The backtest conditions on realised gas and system paths. It evaluates calibration of the "
        "price-response distribution, not the probability assigned to a political scenario.", "",
        f"Generated-driver sensitivity: 90% coverage {100 * validation['generated_driver_coverage_90']:.1f}%, "
        f"mean CRPS {validation['generated_driver_mean_crps']:.2f}. This generates gas and "
        "seasonally matched system paths from the training data rather than using realised future drivers.", "",
        "## Separate short-term product", "",
        "`short_term_14d_forecast.csv` contains a price-only 1-14 day forecast. It must not be "
        "presented as the 90-day conflict scenario.", "",
        f"Chart: `{chart.name}`",
    ]
    (RESULTS / "scenario_weekly_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Enhanced RQ1 complete")
    print(json.dumps({"week13_medians": medians, "separation": separation,
                      "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
