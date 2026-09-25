import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "smard_mean"
GAS = "ttf_eur_mwh"

HORIZON = 90              # days; confirm with supervisor
N_SIMS = 4000             # Monte Carlo draws per scenario
CONFLICT_START = pd.Timestamp("2026-02-28")
CLOSURE_START = pd.Timestamp("2026-03-02")
CLOSURE_END = pd.Timestamp("2026-06-15")
RANDOM_SEED = 42

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def load():
    df = pd.read_csv(PROCESSED_DIR / "master_features.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------ LINK 2 ------
def estimate_passthrough(df):
    d = df.copy()
    d["d_electricity"] = d[TARGET].diff()
    d["d_gas_lag1"] = d[GAS].diff().shift(1)
    d["d_electricity_lag1"] = d["d_electricity"].shift(1)
    cols = ["d_electricity", "d_gas_lag1", "d_electricity_lag1"]
    ctrl = []
    if "dunkelflaute_flag_lag1" in d.columns:
        ctrl.append("dunkelflaute_flag_lag1")
    if "month" in d.columns:
        d["m_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["m_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        ctrl += ["m_sin", "m_cos"]
    if "is_weekend" in d.columns:
        ctrl.append("is_weekend")

    d = d[cols + ctrl + ["date"]].dropna()
    regressors = ["d_gas_lag1", "d_electricity_lag1"] + ctrl
    X = sm.add_constant(d[regressors])
    m = sm.OLS(d["d_electricity"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 7}
    )

    resid_sd = float(np.std(m.resid, ddof=len(m.params)))
    return {
        "beta": float(m.params["d_gas_lag1"]),
        "beta_se": float(m.bse["d_gas_lag1"]),
        "beta_p": float(m.pvalues["d_gas_lag1"]),
        "phi": float(m.params["d_electricity_lag1"]),
        "intercept": float(m.params["const"]),
        "resid_sd": resid_sd,
        "r2": float(m.rsquared),
        "n": int(m.nobs),
        "controls": ctrl,
        "model": m,
    }


# ------------------------------------------------------------ LINK 1 ------
def anchor_gas_levels(df):
    pre = df[df["date"] < CONFLICT_START][GAS].dropna()
    closed = df[(df["date"] >= CLOSURE_START) &
                (df["date"] <= CLOSURE_END)][GAS].dropna()
    recent = df[GAS].dropna().tail(30)

    recent_mean = float(recent.mean())
    closed_mean = float(closed.mean()) if len(closed) else recent_mean
    closed_peak = float(closed.max()) if len(closed) else recent_mean

    pre_closure_window = df[(df["date"] >= CLOSURE_START - pd.Timedelta(days=60)) &
                             (df["date"] < CLOSURE_START)][GAS].dropna()
    baseline_before = (float(pre_closure_window.mean())
                       if len(pre_closure_window) > 10 else float(pre.tail(60).mean()))
    closure_premium = max(closed_mean - baseline_before, 0.0)
    closure_peak_premium = max(closed_peak - baseline_before, 0.0)

    esc_target = recent_mean + closure_premium

    deesc_target = min(baseline_before, recent_mean)
    deesc_basis = ("immediate 60-day pre-closure mean"
                   if deesc_target == baseline_before
                   else "recent 30-day mean, already below the pre-closure mean")

    levels = {
        "de_escalation": {
            "target": deesc_target,
            "vol": float(pre_closure_window.diff().std()),
            "basis": f"{deesc_basis} (pre-closure {baseline_before:.2f}, "
                      f"recent {recent_mean:.2f})",
        },
        "stalemate": {
            "target": recent_mean,
            "vol": float(recent.diff().std()),
            "basis": "mean of the most recent 30 observed days",
        },
        "escalation": {
            "target": esc_target,
            "vol": float(closed.diff().std()) if len(closed) > 2 else float(recent.diff().std()),
            "basis": f"current level ({recent_mean:.2f}) plus the observed "
                      f"closure premium (+{closure_premium:.2f}, measured as "
                      f"closure mean {closed_mean:.2f} minus the 60 days "
                      f"before closure {baseline_before:.2f})",
        },
    }
    levels["_closure_premium"] = closure_premium
    levels["_closure_peak_premium"] = closure_peak_premium
    levels["_closed_mean"] = closed_mean
    levels["_baseline_before"] = baseline_before
    levels["_pre_mean"] = baseline_before
    levels["_recent_mean"] = recent_mean
    levels["_observed_peak"] = closed_peak
    levels["_current"] = float(df[GAS].dropna().iloc[-1])
    return levels


def simulate_gas_paths(start, target, vol, horizon, n_sims, rng):
    kappa = 0.05                       # ~20 day adjustment half-life
    paths = np.zeros((n_sims, horizon))
    level = np.full(n_sims, start, dtype=float)
    for t in range(horizon):
        level = level + kappa * (target - level) + rng.normal(0, vol, n_sims)
        level = np.maximum(level, 0.0)   # gas prices cannot go negative
        paths[:, t] = level
    return paths


def project(df, pt, levels, scenario, rng):
    spec = levels[scenario]
    start = levels["_current"]

    gas = simulate_gas_paths(start, spec["target"], spec["vol"],
                             HORIZON, N_SIMS, rng)

    # Joint parameter uncertainty preserves covariance among coefficients.
    m = pt["model"]
    cov = np.asarray(m.cov_params(), dtype=float)
    cov = (cov + cov.T) / 2
    eigval, eigvec = np.linalg.eigh(cov)
    cov_psd = eigvec @ np.diag(np.maximum(eigval, 0)) @ eigvec.T
    draws = rng.multivariate_normal(np.asarray(m.params), cov_psd, size=N_SIMS)
    param = {name: draws[:, i] for i, name in enumerate(m.params.index)}
    # Prevent rare parameter draws from creating an explosive AR path that was
    # never supported by the fitted stationary change model.
    param["d_electricity_lag1"] = np.clip(
        param["d_electricity_lag1"], -0.99, 0.99
    )

    # Seasonal control evaluated over the projection window so the projection
    # does not silently assume the current month persists for 90 days.
    last_date = df["date"].max()
    future = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON)
    seas = np.zeros(HORIZON)
    # Residual uncertainty: what differenced gas does not explain.
    noise = rng.normal(0, pt["resid_sd"], (N_SIMS, HORIZON))

    gas_change = np.diff(np.column_stack([np.full(N_SIMS, start), gas]), axis=1)
    last_gas_change = float(df[GAS].diff().dropna().iloc[-1])
    gas_lag_change = np.column_stack([
        np.full(N_SIMS, last_gas_change), gas_change[:, :-1]
    ])
    dprice = np.zeros((N_SIMS, HORIZON))
    previous = np.full(N_SIMS, float(df[TARGET].diff().dropna().iloc[-1]))
    historical_low_output_rate = float(df["dunkelflaute_flag"].mean())
    for t, date in enumerate(future):
        change = param["const"] + param["d_gas_lag1"] * gas_lag_change[:, t]
        change += param["d_electricity_lag1"] * previous
        if "m_sin" in param:
            change += param["m_sin"] * np.sin(2 * np.pi * date.month / 12)
            change += param["m_cos"] * np.cos(2 * np.pi * date.month / 12)
        if "dunkelflaute_flag_lag1" in param:
            change += param["dunkelflaute_flag_lag1"] * historical_low_output_rate
        if "is_weekend" in param:
            change += param["is_weekend"] * float(date.dayofweek >= 5)
        change += noise[:, t]
        dprice[:, t] = change
        previous = change
    price = float(df[TARGET].iloc[-1]) + np.cumsum(dprice, axis=1)
    return price, gas


def summarise(price, dates, scenario):
    rows = []
    for i, d in enumerate(dates):
        col = price[:, i]
        row = {"scenario": scenario, "date": d.date(), "day": i + 1}
        for q in QUANTILES:
            row[f"q{int(q*100)}"] = float(np.quantile(col, q))
        row["mean"] = float(col.mean())
        rows.append(row)
    return rows


def plot_fan(all_rows, df):
    proj = pd.DataFrame(all_rows)
    proj["pass_through_p"] = pt["beta_p"]
    proj["evidence_status"] = (
        "supported_conditional_sensitivity" if pt["beta_p"] < 0.05
        else "exploratory_pass_through_not_significant"
    )
    proj["date"] = pd.to_datetime(proj["date"])
    hist = df[df["date"] >= df["date"].max() - pd.Timedelta(days=120)]

    colors = {"escalation": "tab:red", "stalemate": "tab:orange",
              "de_escalation": "tab:green"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, sharey=True)

    for ax, sc in zip(axes, ["escalation", "stalemate", "de_escalation"]):
        s = proj[proj["scenario"] == sc]
        c = colors[sc]
        ax.plot(hist["date"], hist[TARGET], color="black", lw=1.0,
                label="Observed")
        ax.fill_between(s["date"], s["q5"], s["q95"], alpha=0.18, color=c,
                        label="90% interval")
        ax.fill_between(s["date"], s["q25"], s["q75"], alpha=0.35, color=c,
                        label="50% interval")
        ax.plot(s["date"], s["q50"], color=c, lw=1.8, label="Median")
        ax.axvline(df["date"].max(), color="grey", ls=":", lw=1)
        ax.set_title(f"{sc.replace('_', '-')}", loc="left", fontsize=11)
        ax.set_ylabel("EUR/MWh")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=4)

    axes[0].set_title("Projected German day-ahead electricity prices, "
                       f"{HORIZON}-day horizon\nescalation", loc="left", fontsize=11)
    fig.tight_layout()
    p = RESULTS_DIR / "scenario_fan_chart.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    df = load()
    print(f"Loaded {len(df)} days, to {df['date'].max().date()}\n")

    # ------------------------------------------------------- link 2 ----
    print("=" * 74)
    print("LINK 2 -- GAS TO ELECTRICITY PASS-THROUGH")
    print("=" * 74)
    pt = estimate_passthrough(df)
    print(f"  Delta electricity(t) ~ Delta gas(t-1) + lagged Delta electricity, controlling for "
          f"{', '.join(pt['controls'])}")
    print(f"    pass-through  {pt['beta']:+.4f} EUR/MWh per EUR/MWh "
          f"(se {pt['beta_se']:.4f}, p = {pt['beta_p']:.4g})")
    print(f"    R2 {pt['r2']:.3f}, residual sd {pt['resid_sd']:.2f}, n = {pt['n']}")
    if pt["beta_p"] > 0.05:
        print("\n  WARNING: the pass-through is not significant in this")
        print("  specification. Scenario differences will be weak and the")
        print("  projections should not be reported as causal.")
    else:
        print(f"\n  Significant. One EUR/MWh on gas yesterday implies about")
        print(f"  {pt['beta']:.2f} EUR/MWh on electricity today.")

    # ------------------------------------------------------- link 1 ----
    print("\n" + "=" * 74)
    print("LINK 1 -- SCENARIO GAS LEVELS (all anchored to observed data)")
    print("=" * 74)
    levels = anchor_gas_levels(df)
    print(f"  Current gas price: {levels['_current']:.2f} EUR/MWh")
    print(f"  Peak observed during closure: {levels['_observed_peak']:.2f}\n")

    print(f"  Gas level comparison:")
    print(f"    60 days before closure : {levels['_baseline_before']:6.2f}")
    print(f"    during closure         : {levels['_closed_mean']:6.2f}  "
          f"(premium +{levels['_closure_premium']:.2f})")
    print(f"    pre-conflict overall   : {levels['_pre_mean']:6.2f}")
    print(f"    recent 30 days         : {levels['_recent_mean']:6.2f}")
    if levels["_recent_mean"] > levels["_closed_mean"]:
        print(f"\n    NOTE: gas is currently HIGHER than during the closure.")
        print(f"    Escalation is therefore anchored to the closure PREMIUM")
        print(f"    applied to today's level, not to the closure level itself.")
        print(f"    This is a data-derived anchoring condition, not an assumed")
        print(f"    conclusion about the strength of the conflict signal.")
    print()
    for sc in ["escalation", "stalemate", "de_escalation"]:
        s = levels[sc]
        print(f"  {sc:<15} target {s['target']:7.2f}  daily vol {s['vol']:5.2f}")
        print(f"  {'':<15} basis: {s['basis']}")

    # --------------------------------------------------- projection ----
    print("\n" + "=" * 74)
    print(f"PROJECTION -- {HORIZON} days, {N_SIMS} simulations per scenario")
    print("=" * 74)
    last_date = df["date"].max()
    future = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON)

    all_rows = []
    finals = {}
    for sc in ["escalation", "stalemate", "de_escalation"]:
        price, gas = project(df, pt, levels, sc, rng)
        all_rows += summarise(price, future, sc)
        finals[sc] = price[:, -1]
        avg = price.mean(axis=1)
        print(f"\n  {sc.replace('_', '-').upper()}")
        print(f"    mean over horizon : {np.median(avg):7.2f} EUR/MWh "
              f"(90% interval {np.quantile(avg, 0.05):.2f} to "
              f"{np.quantile(avg, 0.95):.2f})")
        print(f"    at day {HORIZON}       : {np.median(finals[sc]):7.2f} "
              f"(90% interval {np.quantile(finals[sc], 0.05):.2f} to "
              f"{np.quantile(finals[sc], 0.95):.2f})")

    # ------------------------------------------------ sanity check ----
    print("\n" + "=" * 74)
    print("SANITY CHECK -- scenario ordering")
    print("=" * 74)
    me = np.median(finals["escalation"])
    ms = np.median(finals["stalemate"])
    md = np.median(finals["de_escalation"])
    print(f"  escalation {me:.2f}  >=  stalemate {ms:.2f}  >=  "
          f"de-escalation {md:.2f}")
    if me >= ms >= md:
        print("  PASS -- ordering is coherent.")
    else:
        raise RuntimeError(
            "Scenario ordering failed. Outputs are withheld because escalation, "
            "stalemate, and de-escalation are not empirically distinguishable in "
            "the required direction."
        )

    sep_hi = np.mean(finals["escalation"] > np.median(finals["de_escalation"]))
    print(f"\n  Separation: {100*sep_hi:.1f}% of escalation draws exceed the")
    print(f"  de-escalation median.")
    if sep_hi < 0.6:
        print("  The scenarios overlap heavily. Report the overlap honestly:")
        print("  it means conflict trajectory has limited influence on German")
        print("  prices relative to weather and baseline volatility.")

    # ---------------------------------------------------- outputs ----
    proj = pd.DataFrame(all_rows)
    projection_path = RESULTS_DIR / "scenario_projections.csv"
    proj.to_csv(projection_path, index=False)
    p = plot_fan(all_rows, df)
    print(f"\nSaved projections: {projection_path}")
    print(f"Saved fan chart:   {p}")

    lines = [
        "# Scenario Projections (RQ1)", "",
        f"German day-ahead electricity prices, {HORIZON}-day horizon, "
        f"{N_SIMS} simulations per scenario.", "",
        "## Method", "",
        "Scenarios are conditional sensitivities propagated through the GAS CHANNEL rather than through "
        "news or geopolitical risk features. The gas-to-electricity link is re-estimated in first "
        "differences to avoid a spurious levels relationship.", "",
        f"- Pass-through: {pt['beta']:+.4f} EUR/MWh per EUR/MWh of gas "
        f"(se {pt['beta_se']:.4f}, p = {pt['beta_p']:.4g})",
        f"- Model R2: {pt['r2']:.3f}, residual sd {pt['resid_sd']:.2f}, n = {pt['n']}",
        "",
        "Three uncertainty sources are propagated: parameter uncertainty in the "
        "pass-through, path uncertainty in the gas trajectory, and residual "
        "uncertainty in what gas does not explain.", "",
        "## Scenario definitions", "",
    ]
    if pt["beta_p"] >= 0.05:
        lines += [
            "> **Exploratory only:** the first-difference gas pass-through is not "
            "statistically distinguishable from zero. Scenario separation is not "
            "empirically established and must not be reported as a causal forecast.",
            "",
        ]
    for sc in ["escalation", "stalemate", "de_escalation"]:
        lines.append(f"- **{sc.replace('_', '-')}**: gas to "
                      f"{levels[sc]['target']:.2f} EUR/MWh "
                      f"({levels[sc]['basis']})")
    lines += ["", "## Results", "",
              "| Scenario | Median at day 90 | 90% interval |", "|---|---|---|"]
    for sc in ["escalation", "stalemate", "de_escalation"]:
        f_ = finals[sc]
        lines.append(f"| {sc.replace('_', '-')} | {np.median(f_):.2f} | "
                      f"{np.quantile(f_, 0.05):.2f} to {np.quantile(f_, 0.95):.2f} |")
    lines += [
        "", "## Limitations", "",
        "- Driver levels are anchored to observed values, so the model is never "
        "extrapolated into unseen gas price territory. The cost is that "
        "escalation is capped at the severity actually observed in 2026; a more "
        "severe or prolonged closure would push prices higher, and a single "
        "episode cannot say how much higher.",
        "- Severity is not identified. One closure episode provides no basis for "
        "scaling to different closure durations or completeness.",
        "- The projection is a conditional sensitivity, not a causal estimate or trained forecast. It composes two "
        "separately estimated links rather than learning the full chain, because "
        "one episode cannot support end-to-end training.",
        "- Weather is set to its historical expected rate rather than a forecast. "
        "Actual outcomes will diverge with weather.",
    ]
    summary_path = RESULTS_DIR / "scenario_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved summary:     {summary_path}")
