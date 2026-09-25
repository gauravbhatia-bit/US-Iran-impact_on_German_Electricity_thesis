import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ALPHA = 0.05


def diebold_mariano(loss_a, loss_b, h=1, power=1):
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 8:
        return np.nan, np.nan, np.nan

    d = (a ** power) - (b ** power)
    d_bar = d.mean()

    # Long-run variance using autocovariances up to lag h-1
    gamma0 = np.sum((d - d_bar) ** 2) / n
    v = gamma0
    for lag in range(1, h):
        gl = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / n
        v += 2 * gl
    if v <= 0:
        return np.nan, np.nan, d_bar

    dm = d_bar / np.sqrt(v / n)

    # HLN correction -- materially important at small n
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corrected = dm * hln

    p = 2 * (1 - stats.t.cdf(abs(dm_corrected), df=n - 1))
    return dm_corrected, p, d_bar


def holm_bonferroni(pvals):
    p = np.asarray(pvals, float)
    valid = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return out
    idx = np.argsort(p[valid])
    m = valid.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * p[valid][i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(running, 1.0)
    out[valid] = adj
    return out


def run_comparisons(df, pairs, period, label_map, comparison_type, h=1):
    rows = []
    sub = df[df["period"] == period]
    if sub.empty:
        return rows

    for col_a, col_b in pairs:
        if col_a not in sub.columns or col_b not in sub.columns:
            continue
        dm, p, d_bar = diebold_mariano(sub[col_a], sub[col_b], h=h)
        mae_a, mae_b = sub[col_a].mean(), sub[col_b].mean()
        rows.append({
            "comparison_type": comparison_type,
            "period": period,
            "model_a": label_map.get(col_a, col_a),
            "model_b": label_map.get(col_b, col_b),
            "n_obs": len(sub),
            "n_folds": len(sub),
            "mae_a": round(mae_a, 3),
            "mae_b": round(mae_b, 3),
            "diff_pct": round(100 * (mae_a - mae_b) / mae_a, 2) if mae_a else np.nan,
            "dm_stat": round(dm, 3) if np.isfinite(dm) else np.nan,
            "p_raw": p if np.isfinite(p) else np.nan,
        })
    return rows


def verdict(row):
    p = row["p_adj"]
    if not np.isfinite(p):
        return "insufficient data"
    if p >= ALPHA:
        return "no significant difference"
    return f"{row['model_b']} better" if row["mae_a"] > row["mae_b"] else f"{row['model_a']} better"


if __name__ == "__main__":
    all_rows = []

    # ---------- Ablation: consecutive blocks (the marginal contributions) ----------
    ab_path = PROCESSED_DIR / "ablation_results.csv"
    daily_path = PROCESSED_DIR / "ablation_daily_errors.csv"

    blocks = ["A_price_only", "B_plus_market", "C_plus_weather",
               "D_plus_geopolitical", "E_plus_sentiment"]
    block_names = ["A price+calendar", "B +market", "C +weather",
                    "D +geopolitical", "E +news"]

    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        labels = {f"{b}_abs_err": n for b, n in zip(blocks, block_names)}
        labels["prophet_abs_err"] = "Prophet alone"

        step_pairs = [(f"{blocks[i]}_abs_err", f"{blocks[i+1]}_abs_err")
                       for i in range(len(blocks) - 1)]
        step_pairs.append(("A_price_only_abs_err", "E_plus_sentiment_abs_err"))

        print(f"Using one-day-ahead errors ({len(daily)} observations), h=1.\n")
        for period in sorted(daily["period"].dropna().unique()):
            all_rows += run_comparisons(daily, step_pairs, period, labels,
                                         "ablation", h=1)

    elif ab_path.exists():
        # FALLBACK: per-fold MAE only. Conservative -- see the header note.
        ab = pd.read_csv(ab_path)
        labels = {f"{b}_MAE": n for b, n in zip(blocks, block_names)}
        labels["prophet_MAE"] = "Prophet alone"

        step_pairs = [(f"{blocks[i]}_MAE", f"{blocks[i+1]}_MAE")
                       for i in range(len(blocks) - 1)]
        step_pairs.append(("A_price_only_MAE", "E_plus_sentiment_MAE"))

        print("Using PER-FOLD MAE -- CONSERVATIVE. Re-run 10_ablation_study.py to\n"
              "generate per-day errors for a properly powered test.\n")
        for period in sorted(ab["period"].dropna().unique()):
            all_rows += run_comparisons(ab, step_pairs, period, labels, "ablation")
    else:
        print("[skip] no ablation results found")

    # ---------- Cross-arm: model against model ----------
    ch_path = PROCESSED_DIR / "chronos_zeroshot_results.csv"
    px_path = PROCESSED_DIR / "prophet_xgboost_results.csv"
    if ch_path.exists() and px_path.exists():
        ch = pd.read_csv(ch_path)
        px = pd.read_csv(px_path)
        merged = ch.merge(px, on=["fold_id", "period"], suffixes=("_c", "_p"))

        arm_labels = {"naive_MAE": "Naive", "chronos_MAE": "Chronos",
                       "prophet_MAE": "Prophet", "hybrid_MAE": "Prophet+XGB"}
        arm_pairs = [
            ("naive_MAE", "chronos_MAE"),
            ("naive_MAE", "hybrid_MAE"),
            ("prophet_MAE", "hybrid_MAE"),
            ("chronos_MAE", "hybrid_MAE"),
        ]
        for period in sorted(merged["period"].dropna().unique()):
            all_rows += run_comparisons(merged, arm_pairs, period, arm_labels, "cross-arm")
    else:
        print(f"[skip] cross-arm files not found")

    if not all_rows:
        raise SystemExit("No results files found -- run the modeling scripts first.")

    res = pd.DataFrame(all_rows)

    # Holm-Bonferroni within each comparison family, per period
    res["p_adj"] = np.nan
    for (ctype, period), grp in res.groupby(["comparison_type", "period"]):
        res.loc[grp.index, "p_adj"] = holm_bonferroni(grp["p_raw"].values)
    res["verdict"] = res.apply(verdict, axis=1)

    # ---------- Report ----------
    for ctype in ["ablation", "cross-arm"]:
        for period in sorted(res["period"].dropna().unique()):
            grp = res[(res["comparison_type"] == ctype) & (res["period"] == period)]
            if grp.empty:
                continue
            print("\n" + "=" * 92)
            print(f"{ctype.upper()}  --  {period}  (n={grp['n_obs'].iloc[0]} observations)")
            print("=" * 92)
            print(f"{'comparison':<38} {'MAE a':>8} {'MAE b':>8} {'p raw':>8} {'p adj':>8}  verdict")
            print("-" * 92)
            for _, r in grp.iterrows():
                comp = f"{r['model_a']} vs {r['model_b']}"
                praw = f"{r['p_raw']:.4f}" if np.isfinite(r["p_raw"]) else "  --  "
                padj = f"{r['p_adj']:.4f}" if np.isfinite(r["p_adj"]) else "  --  "
                star = " *" if np.isfinite(r["p_adj"]) and r["p_adj"] < ALPHA else ""
                print(f"{comp:<38} {r['mae_a']:>8.2f} {r['mae_b']:>8.2f} "
                      f"{praw:>8} {padj:>8}  {r['verdict']}{star}")

    n_sig = (res["p_adj"] < ALPHA).sum()
    print("\n" + "=" * 92)
    print(f"{n_sig} of {len(res)} comparisons significant at adjusted p < {ALPHA}")
    print("=" * 92)

    shock_abl = res[(res["comparison_type"] == "ablation") &
                    res["period"].str.startswith("SHOCK")]
    rq4_rows = shock_abl[(shock_abl["model_a"] == "D +geopolitical") &
                         (shock_abl["model_b"] == "E +news")]
    for _, r in rq4_rows.iterrows():
        print(f"\nRQ4 (news coverage as a leading signal), {r['period']}:")
        if not np.isfinite(r["p_adj"]):
            print("  Test could not be computed.")
        elif r["p_adj"] >= ALPHA:
            print(f"  NO SIGNIFICANT DIFFERENCE (adjusted p = {r['p_adj']:.3f}).")
            print("  The correct claim is that news coverage has no MEASURABLE effect on")
            print("  forecast accuracy -- not that it worsens it. Report the null result.")
        else:
            direction = "worsens" if r["mae_b"] > r["mae_a"] else "improves"
            print(f"  SIGNIFICANT (adjusted p = {r['p_adj']:.3f}): adding news coverage")
            print(f"  {direction} accuracy.")

    print("\nInference uses one-day-ahead loss differentials sampled at weekly origins.")

    out = PROCESSED_DIR / "significance_tests.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved: {out}")
