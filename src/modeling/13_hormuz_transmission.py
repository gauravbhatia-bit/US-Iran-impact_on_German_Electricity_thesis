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
GPR = "gpr_daily"

CLOSURE_START = pd.Timestamp("2026-03-02")
REOPEN_DATE = pd.Timestamp("2026-06-15")
CONTESTED_START = pd.Timestamp("2026-07-13")  # verified blockade resumption
CONTESTED_END = pd.Timestamp("2026-09-30")    # safely clipped to observed data


def add_hormuz_state(df, reopen=REOPEN_DATE):
    d = df.copy()
    d["hormuz_closed"] = ((d["date"] >= CLOSURE_START) & (d["date"] < reopen)).astype(int)
    contested = (d["date"] >= CONTESTED_START) & (d["date"] <= CONTESTED_END)
    d["hormuz_stress"] = (d["hormuz_closed"].astype(bool) | contested).astype(int)
    return d


def fit_ols(y, X, label):
    X = sm.add_constant(X, has_constant="add")
    ok = y.notna() & X.notna().all(axis=1)
    if ok.sum() < 40:
        return None
    m = sm.OLS(y[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return m


# ---------------------------------------------------------------- PART A ----
def pass_through(df):
    d = df[["date", TARGET, GAS, "hormuz_closed", "dunkelflaute_flag"]].dropna().copy()
    d["d_elec"] = d[TARGET].diff()
    d["d_gas"] = d[GAS].diff()
    d = d.dropna()

    results = {}

    m_all = fit_ols(d["d_elec"], d[["d_gas"]], "full")
    if m_all is not None:
        results["full_period"] = {
            "coef": m_all.params["d_gas"], "se": m_all.bse["d_gas"],
            "p": m_all.pvalues["d_gas"], "n": int(m_all.nobs),
            "r2": m_all.rsquared,
        }

    for name, mask in [("hormuz_open", d["hormuz_closed"] == 0),
                        ("hormuz_closed", d["hormuz_closed"] == 1)]:
        sub = d[mask]
        if len(sub) < 40:
            continue
        m = fit_ols(sub["d_elec"], sub[["d_gas"]], name)
        if m is not None:
            results[name] = {
                "coef": m.params["d_gas"], "se": m.bse["d_gas"],
                "p": m.pvalues["d_gas"], "n": int(m.nobs), "r2": m.rsquared,
            }

    # Interaction: does pass-through DIFFER by regime? The interaction term's
    # p-value is the direct test, rather than eyeballing two separate coefs.
    d["gas_x_closed"] = d["d_gas"] * d["hormuz_closed"]
    m_int = fit_ols(d["d_elec"], d[["d_gas", "hormuz_closed", "gas_x_closed"]], "interaction")
    if m_int is not None:
        results["interaction"] = {
            "coef": m_int.params["gas_x_closed"], "se": m_int.bse["gas_x_closed"],
            "p": m_int.pvalues["gas_x_closed"], "n": int(m_int.nobs),
        }
    return results


# ---------------------------------------------------------------- PART B ----
def regime_comparison(df):
    d = df.copy()
    ctrl = ["dunkelflaute_flag"]
    if "renewable_output_mwh" in d.columns:
        d["renew_scaled"] = d["renewable_output_mwh"] / 1000.0
        ctrl.append("renew_scaled")
    if "is_weekend" in d.columns:
        ctrl.append("is_weekend")
    if "month" in d.columns:
        d["m_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["m_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        ctrl += ["m_sin", "m_cos"]

    cols = [TARGET, "hormuz_closed"] + ctrl
    d = d[cols].dropna()
    if d.empty:
        return None

    m = fit_ols(d[TARGET], d[["hormuz_closed"] + ctrl], "regime")
    if m is None:
        return None

    raw_closed = df.loc[df["hormuz_closed"] == 1, TARGET].mean()
    raw_open = df.loc[df["hormuz_closed"] == 0, TARGET].mean()
    return {
        "raw_closed": raw_closed, "raw_open": raw_open,
        "raw_diff": raw_closed - raw_open,
        "adj_coef": m.params["hormuz_closed"],
        "adj_se": m.bse["hormuz_closed"],
        "adj_p": m.pvalues["hormuz_closed"],
        "n": int(m.nobs), "controls": ctrl,
    }


# ---------------------------------------------------------------- PART C ----
def head_to_head(df):
    d = df.copy()
    ctrl = ["dunkelflaute_flag"]
    if "renewable_output_mwh" in d.columns:
        d["renew_scaled"] = d["renewable_output_mwh"] / 1000.0
        ctrl.append("renew_scaled")
    if "is_weekend" in d.columns:
        ctrl.append("is_weekend")

    # GPR is on a very different scale from a 0/1 dummy; standardising makes
    # the coefficients comparable in magnitude.
    d["gpr_z"] = (d[GPR] - d[GPR].mean()) / d[GPR].std()

    need = [TARGET, "gpr_z", "hormuz_closed"] + ctrl
    d = d[need].dropna()
    if d.empty:
        return None

    out = {}
    specs = {
        "gpr_only": ["gpr_z"] + ctrl,
        "hormuz_only": ["hormuz_closed"] + ctrl,
        "both": ["gpr_z", "hormuz_closed"] + ctrl,
    }
    for name, X in specs.items():
        m = fit_ols(d[TARGET], d[X], name)
        if m is None:
            continue
        rec = {"n": int(m.nobs), "r2": m.rsquared}
        for v in ["gpr_z", "hormuz_closed"]:
            if v in m.params.index:
                rec[f"{v}_coef"] = m.params[v]
                rec[f"{v}_p"] = m.pvalues[v]
        out[name] = rec

    # How correlated are they? If very high, the head-to-head cannot separate
    # them and that itself is the answer.
    out["correlation"] = float(d["gpr_z"].corr(d["hormuz_closed"]))
    return out


# ---------------------------------------------------------------- PART D ----
def sensitivity(df_raw):
    rows = []
    for wk in range(-6, 7):
        rd = REOPEN_DATE + pd.Timedelta(weeks=wk)
        d = add_hormuz_state(df_raw, reopen=rd)
        r = regime_comparison(d)
        if r is None:
            continue
        rows.append({
            "reopen_date": rd.date(),
            "weeks_offset": wk,
            "adjusted_effect": round(r["adj_coef"], 2),
            "p_value": round(r["adj_p"], 4),
            "significant": r["adj_p"] < 0.05,
        })
    return pd.DataFrame(rows)


def plot_regimes(df):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(df["date"], df[TARGET], lw=0.9, color="tab:blue", label="German electricity")
    closed = df[df["hormuz_closed"] == 1]
    if not closed.empty:
        ax.axvspan(closed["date"].min(), closed["date"].max(),
                   alpha=0.18, color="tab:red", label="Hormuz closed")
    ax.axvspan(CONTESTED_START, CONTESTED_END, alpha=0.12, color="tab:orange",
               label="Contested transit")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("German day-ahead electricity price and Strait of Hormuz state")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = RESULTS_DIR / "hormuz_transmission.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ MAIN ----
if __name__ == "__main__":
    raw = pd.read_csv(PROCESSED_DIR / "master_features.csv", parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    df = add_hormuz_state(raw)

    n_closed = int(df["hormuz_closed"].sum())
    print(f"Loaded {len(df)} days")
    print(f"Hormuz closed: {n_closed} days "
          f"({CLOSURE_START.date()} to {REOPEN_DATE.date()})")
    print(f"NOTE: the reopening date is UNVERIFIED. Part D tests sensitivity.\n")

    print("=" * 78)
    print("PART A -- GAS TO ELECTRICITY PASS-THROUGH")
    print("=" * 78)
    print("  Estimated in first differences (both series are non-stationary in")
    print("  levels, which would give a spurious relationship).\n")
    pt = pass_through(df)
    for k in ["full_period", "hormuz_open", "hormuz_closed"]:
        if k in pt:
            r = pt[k]
            sig = "*" if r["p"] < 0.05 else " "
            print(f"  {k:<16} {r['coef']:+.4f} EUR elec per EUR gas  "
                  f"(se {r['se']:.4f}, p {r['p']:.4f}){sig}  n={r['n']}")
    if "interaction" in pt:
        r = pt["interaction"]
        print(f"\n  Difference in pass-through between regimes:")
        print(f"    interaction coef {r['coef']:+.4f} (p = {r['p']:.4f})")
        print("    -> " + ("pass-through DIFFERS significantly by regime"
                            if r["p"] < 0.05 else
                            "no significant difference in pass-through by regime"))

    print("\n" + "=" * 78)
    print("PART B -- PRICE LEVEL BY REGIME (with controls)")
    print("=" * 78)
    rc = regime_comparison(df)
    if rc:
        print(f"  Raw mean, closed : {rc['raw_closed']:.2f} EUR/MWh")
        print(f"  Raw mean, open   : {rc['raw_open']:.2f} EUR/MWh")
        print(f"  Raw difference   : {rc['raw_diff']:+.2f}")
        print(f"\n  After controlling for {', '.join(rc['controls'])}:")
        print(f"    adjusted effect {rc['adj_coef']:+.2f} EUR/MWh "
              f"(se {rc['adj_se']:.2f}, p = {rc['adj_p']:.4f})")
        gap = rc["raw_diff"] - rc["adj_coef"]
        print(f"    controls absorb {gap:+.2f} EUR/MWh of the raw difference")
        if abs(gap) > abs(rc["raw_diff"]) * 0.4:
            print("    -> a large share of the raw gap is weather/seasonal, not Hormuz")

    print("\n" + "=" * 78)
    print("PART C -- HORMUZ STATE vs GPR  (the central question)")
    print("=" * 78)
    hh = head_to_head(df)
    if hh:
        print(f"  Correlation between GPR and Hormuz state: {hh['correlation']:.3f}")
        if abs(hh["correlation"]) > 0.8:
            print("  WARNING: very highly correlated. The two variables cannot be")
            print("  cleanly separated, and that near-redundancy is itself the answer.")
        print()
        for spec in ["gpr_only", "hormuz_only", "both"]:
            if spec not in hh:
                continue
            r = hh[spec]
            print(f"  {spec}:  (R2 {r['r2']:.3f}, n {r['n']})")
            for v, lab in [("gpr_z", "GPR (standardised)"),
                            ("hormuz_closed", "Hormuz closed")]:
                if f"{v}_coef" in r:
                    star = "*" if r[f"{v}_p"] < 0.05 else " "
                    print(f"      {lab:<22} {r[f'{v}_coef']:+8.2f}  "
                          f"p = {r[f'{v}_p']:.4f}{star}")
            print()

        if "both" in hh:
            b = hh["both"]
            g_sig = b.get("gpr_z_p", 1) < 0.05
            h_sig = b.get("hormuz_closed_p", 1) < 0.05
            print("  READING:")
            if h_sig and not g_sig:
                print("    Hormuz survives, GPR does not. The closure state carries")
                print("    information the diffuse risk index misses. This supports")
                print("    building RQ1 scenarios on the chokepoint mechanism.")
            elif g_sig and not h_sig:
                print("    GPR survives, Hormuz does not. The effect is broad")
                print("    geopolitical risk rather than chokepoint-specific.")
            elif g_sig and h_sig:
                print("    Both remain significant -- they capture partly separate")
                print("    information. Worth reporting both.")
            else:
                print("    Neither is significant once controls are included. No")
                print("    detectable chokepoint transmission at the price-level.")
                print("    This is consistent with the ablation null and is solid")
                print("    evidence that the signal is not being missed.")

    print("\n" + "=" * 78)
    print("PART D -- SENSITIVITY TO THE UNVERIFIED REOPENING DATE")
    print("=" * 78)
    sens = sensitivity(raw)
    if not sens.empty:
        n_sig = int(sens["significant"].sum())
        print(f"  Tested {len(sens)} reopening dates "
              f"({sens['reopen_date'].min()} to {sens['reopen_date'].max()})")
        print(f"  Significant in {n_sig}/{len(sens)} specifications")
        print(f"  Effect ranges {sens['adjusted_effect'].min():+.2f} to "
              f"{sens['adjusted_effect'].max():+.2f} EUR/MWh\n")
        for _, r in sens.iterrows():
            mark = "*" if r["significant"] else " "
            print(f"    {r['reopen_date']}  ({r['weeks_offset']:+d}w)  "
                  f"effect {r['adjusted_effect']:+7.2f}  p {r['p_value']:.4f}{mark}")
        if 0 < n_sig < len(sens):
            print("\n  CAUTION: the conclusion depends on which reopening date is")
            print("  assumed. Verify the MoU date before reporting this result.")
        elif n_sig == len(sens):
            print("\n  Robust: significant across every plausible reopening date.")
        else:
            print("\n  Robust: non-significant across every plausible reopening date.")
        sens.to_csv(RESULTS_DIR / "hormuz_transmission.csv", index=False)

    p = plot_regimes(df)
    print(f"\nSaved plot: {p}")

    # ---------------------------------------------------------- summary ----
    lines = ["# Hormuz Transmission Analysis (RQ2a)", "",
             "**Exploratory.** No success threshold was specified in advance; ",
             "results are reported as exploratory rather than confirmatory.", "",
             f"Closure period analysed: {CLOSURE_START.date()} to {REOPEN_DATE.date()} "
             f"({n_closed} days). The reopening date is timing-sensitive.", "",
             "## Pass-through (gas to electricity)", ""]
    for k in ["full_period", "hormuz_open", "hormuz_closed"]:
        if k in pt:
            r = pt[k]
            lines.append(f"- {k}: {r['coef']:+.4f} EUR/EUR (p = {r['p']:.4f}, n = {r['n']})")
    if rc:
        lines += ["", "## Price level by regime", "",
                  f"- Raw difference: {rc['raw_diff']:+.2f} EUR/MWh",
                  f"- After controls: {rc['adj_coef']:+.2f} EUR/MWh (p = {rc['adj_p']:.4f})"]
    if hh and "both" in hh:
        b = hh["both"]
        lines += ["", "## Hormuz state vs GPR", "",
                  f"- Correlation: {hh['correlation']:.3f}"]
        if "gpr_z_p" in b:
            lines.append(f"- GPR in joint model: p = {b['gpr_z_p']:.4f}")
        if "hormuz_closed_p" in b:
            lines.append(f"- Hormuz in joint model: p = {b['hormuz_closed_p']:.4f}")
    if not sens.empty:
        lines += ["", "## Sensitivity to reopening date", "",
                  f"- Significant in {int(sens['significant'].sum())} of {len(sens)} "
                  f"specifications tested"]
    out_md = RESULTS_DIR / "hormuz_summary.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved summary: {out_md}")
