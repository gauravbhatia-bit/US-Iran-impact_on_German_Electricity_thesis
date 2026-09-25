import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

STAGES = [
    ("collect", ["src/pipeline/01_data_collection.py"]),
    ("preprocess", ["src/pipeline/02_preprocessing.py"]),
    ("validate", ["src/pipeline/validation_gate.py"]),
    ("descriptives", [
        "src/pipeline/03_correlation_vif_check.py",
        "src/pipeline/04_event_alignment_and_dataset_overview.py",
        "src/pipeline/05_feature_table_and_real_validation.py",
        "src/pipeline/06_narrative_report.py",
    ]),
    ("folds", ["src/pipeline/07_rolling_origin_cv.py"]),
    ("chronos", ["src/modeling/08_chronos_zeroshot_baseline.py"]),
    ("hybrid", ["src/modeling/09_prophet_xgboost.py"]),
    ("ablation", ["src/modeling/10_ablation_study.py"]),
    ("significance", ["src/modeling/11_significance_tests.py"]),
    ("magnitude", ["src/modeling/12_magnitude_analysis.py"]),
    ("transmission", ["src/modeling/13_hormuz_transmission.py"]),
    ("robustness", ["src/modeling/14_arm4_sarimax_lear.py"]),
    ("granger", ["src/modeling/16_granger_causality.py"]),
    ("granger_diagnostics", ["src/modeling/17_granger_diagnostics.py"]),
    ("scenario", ["src/modeling/15_scenario_engine.py"]),
    ("enhancement_data", ["src/pipeline/01b_enhancement_collection.py"]),
    ("source_audit", ["src/pipeline/08_source_audit.py"]),
    ("availability_audit", ["src/pipeline/09_availability_audit.py"]),
    ("rq1_enhanced", ["src/modeling/18_rq1_probabilistic_enhanced.py"]),
    ("rq2_enhanced", ["src/modeling/19_rq2_causal_enhanced.py"]),
    ("rq2a_enhanced", ["src/modeling/20_rq2a_transmission_enhanced.py"]),
    ("rq2a_event", ["src/modeling/21_rq2a_event_table_study.py"]),
    ("rq4_compact", ["src/modeling/22_rq4_compact_features.py"]),
    ("rq4_eua_bootstrap", ["src/modeling/23_rq4_eua_block_bootstrap.py"]),
    ("rq4_eua_sensitivity", ["src/modeling/24_rq4_eua_block_length_sensitivity.py"]),
]

EXPECTED_OUTPUTS = {
    "collect": ["data/raw/manifest.json"],
    "preprocess": ["data/processed/master_features.csv", "data/processed/manifest.json"],
    "validate": ["data/processed/PIPELINE_AUDIT_REPORT.md"],
    "descriptives": ["data/processed/correlation_matrix_full_range.csv",
                     "data/processed/event_window_check_results.csv",
                     "data/processed/data_dictionary.csv",
                     "data/processed/NARRATIVE_REPORT.md"],
    "folds": ["data/processed/cv_folds.csv"],
    "chronos": ["data/processed/chronos_zeroshot_results.csv"],
    "hybrid": ["data/processed/prophet_xgboost_results.csv"],
    "ablation": ["data/processed/ablation_results.csv",
                 "data/processed/ablation_daily_errors.csv"],
    "significance": ["data/processed/significance_tests.csv"],
    "magnitude": ["results/magnitude_counterfactual.csv",
                  "results/magnitude_event_car.csv",
                  "results/magnitude_placebo_tests.csv",
                  "results/magnitude_summary.md"],
    "transmission": ["results/hormuz_transmission.csv", "results/hormuz_summary.md"],
    "robustness": ["results/arm4_results.csv", "results/arm4_coefficients.csv",
                   "results/arm4_lasso_selection.csv"],
    "granger": ["results/granger_results.csv", "results/granger_summary.md"],
    "granger_diagnostics": ["results/granger_diagnostics.csv",
                            "results/granger_diagnostics_summary.md"],
    "scenario": ["results/scenario_projections.csv", "results/scenario_summary.md",
                 "results/scenario_fan_chart.png"],
    "enhancement_data": ["data/raw/enhancement_manifest.json",
                         "data/raw/energy_charts_prices.csv",
                         "data/raw/energy_charts_german_system.csv",
                         "data/raw/energy_charts_cross_border.csv",
                         "data/raw/imf_portwatch_hormuz.csv",
                         "data/raw/hourly_event_timestamps.csv",
                         "data/raw/eia_hormuz_quarterly.csv",
                         "data/raw/shipping_status_evidence.csv",
                         "data/raw/hormuz_event_table.csv"],
    "source_audit": ["data/processed/ttf_roll_audit.csv",
                     "data/processed/quarter_hour_resolution_audit.csv",
                     "data/processed/source_audit.json"],
    "availability_audit": ["data/processed/availability_audit.csv",
                            "data/processed/availability_audit.json"],
    "rq1_enhanced": ["results/scenario_weekly_projections.csv",
                     "results/scenario_weekly_backtest.csv",
                     "results/scenario_weekly_backtest_generated.csv",
                     "results/scenario_weekly_validation.json",
                     "results/short_term_14d_forecast.csv",
                     "results/scenario_weekly_summary.md",
                     "results/scenario_weekly_fan_chart.png"],
    "rq2_enhanced": ["results/rq2_estimands.csv",
                     "results/rq2_episode_comparison.csv",
                     "results/rq2_outcome_summary.csv",
                     "results/rq2_synthetic_weights.csv",
                     "results/rq2_synthetic_control.csv",
                     "results/rq2_time_placebos.csv",
                     "results/rq2_time_placebos_full_post.csv",
                     "results/rq2_hourly_events.csv",
                     "results/rq2_enhanced_summary.md",
                     "results/rq2_synthetic_control.png"],
    "rq2a_enhanced": ["results/rq2a_distributed_lag_chain.csv",
                      "results/rq2a_local_projections.csv",
                      "results/rq2a_disruption_sensitivity.csv",
                      "results/rq2a_diagnostics.json",
                      "results/rq2a_enhanced_summary.md",
                      "results/rq2a_local_projections.png"],
    "rq2a_event": ["results/rq2a_hormuz_daily_exposure.csv",
                    "results/rq2a_ttf_event_windows.csv",
                    "results/rq2a_ttf_event_study.csv",
                    "results/rq2a_ttf_power_distributed_lag.csv",
                    "results/rq2a_ttf_placebos.csv",
                    "results/rq2a_reclosure_sensitivity.csv",
                    "results/rq2a_traffic_robustness.csv",
                    "results/rq2a_event_diagnostics.json",
                    "results/rq2a_event_summary.md",
                    "results/rq2a_ttf_event_chart.png"],
    "rq4_compact": ["results/rq4_compact_results.csv",
                     "results/rq4_compact_daily_errors.csv",
                     "results/rq4_compact_feature_importance.csv",
                     "results/rq4_compact_diagnostics.json",
                     "results/rq4_compact_summary.md"],
    "rq4_eua_bootstrap": ["results/rq4_eua_block_bootstrap.csv",
                          "results/rq4_eua_block_bootstrap_daily.csv",
                          "results/rq4_eua_block_bootstrap_input_diagnostics.json"],
    "rq4_eua_sensitivity": ["results/rq4_eua_block_length_sensitivity.csv"],
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(status, args, selected, completed, error=None,
                   execution_mode="executed"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ["pandas", "numpy", "scipy", "statsmodels", "prophet",
                 "xgboost", "scikit-learn", "chronos-forecasting"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    scripts = [script for stage, items in STAGES if stage in selected for script in items]
    outputs = [relative for stage in selected
               for relative in EXPECTED_OUTPUTS.get(stage, [])]
    payload = {
        "schema_version": 1,
        "status": status,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "analysis_end": args.end_date,
        "execution_mode": execution_mode,
        "selected_stages": selected,
        "completed_stages": completed,
        "error": error,
        "packages": packages,
        "script_sha256": {script: file_hash(ROOT / script) for script in scripts},
        "artifact_sha256": {
            relative: file_hash(ROOT / relative)
            for relative in outputs
            if (ROOT / relative).exists() and (ROOT / relative).stat().st_size > 0
        },
    }
    # Partial enhancement- and Phase 1-only runs must not overwrite the
    # complete recovered-run provenance record. Full runs use run_manifest.json.
    manifest_name = (
        "enhanced_run_manifest.json"
        if selected and selected[0] == "enhancement_data"
        else "phase1_run_manifest.json"
        if selected and selected[0] in {"source_audit", "availability_audit"}
        else "run_manifest.json"
    )
    path = RESULTS_DIR / manifest_name
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_args():
    names = [name for name, _ in STAGES]
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    parser = argparse.ArgumentParser(
        description="Run the thesis pipeline from a chosen stage, stopping at the first failure."
    )
    parser.add_argument("--from-stage", choices=names, default="collect")
    parser.add_argument("--through-stage", choices=names, default="rq4_eua_sensitivity")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default=yesterday,
                        help="freeze an inclusive data cutoff (default: yesterday UTC)")
    parser.add_argument(
        "--verify-existing", action="store_true",
        help="verify and hash existing outputs instead of executing stages",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    names = [name for name, _ in STAGES]
    start = names.index(args.from_stage)
    end = names.index(args.through_stage)
    if start > end:
        raise SystemExit("--from-stage must not come after --through-stage")
    selected = names[start:end + 1]
    completed = []

    if args.verify_existing:
        missing = [relative for stage in selected
                   for relative in EXPECTED_OUTPUTS.get(stage, [])
                   if not (ROOT / relative).exists()
                   or (ROOT / relative).stat().st_size == 0]
        if missing:
            write_manifest("failed", args, selected, completed,
                           f"missing existing outputs: {missing}",
                           execution_mode="verified_staged_recovery")
            raise SystemExit(f"Cannot verify incomplete recovery: {missing}")
        completed = selected.copy()
        write_manifest("complete", args, selected, completed,
                       execution_mode="verified_staged_recovery")
        manifest_name = (
            "enhanced_run_manifest.json"
            if selected and selected[0] == "enhancement_data"
            else "phase1_run_manifest.json"
            if selected and selected[0] in {"source_audit", "availability_audit"}
            else "run_manifest.json"
        )
        print(f"Verified {len(selected)} completed stages. "
              f"Reproducibility record: {RESULTS_DIR / manifest_name}")
        return

    try:
        for stage, scripts in STAGES[start:end + 1]:
            print(f"\n{'=' * 78}\nSTAGE: {stage}\n{'=' * 78}", flush=True)
            # Exact, generated artifacts only. Removing them first prevents a
            # stale result from satisfying the completion contract.
            if stage not in {"collect", "preprocess"}:
                for relative in EXPECTED_OUTPUTS.get(stage, []):
                    (ROOT / relative).unlink(missing_ok=True)
            for script in scripts:
                command = [sys.executable, str(ROOT / script)]
                if stage in {"collect", "enhancement_data"}:
                    command += ["--start-date", args.start_date,
                                "--end-date", args.end_date]
                subprocess.run(command, cwd=ROOT, check=True)
            missing = [relative for relative in EXPECTED_OUTPUTS.get(stage, [])
                       if not (ROOT / relative).exists()
                       or (ROOT / relative).stat().st_size == 0]
            if missing:
                raise RuntimeError(f"stage {stage} did not create required outputs: {missing}")
            completed.append(stage)
        write_manifest("complete", args, selected, completed)
    except Exception as exc:
        write_manifest("failed", args, selected, completed,
                       f"{type(exc).__name__}: {exc}")
        raise

    manifest_name = (
        "enhanced_run_manifest.json"
        if selected and selected[0] == "enhancement_data"
        else "phase1_run_manifest.json"
        if selected and selected[0] in {"source_audit", "availability_audit"}
        else "run_manifest.json"
    )
    print(f"\nPipeline completed. Reproducibility record: {RESULTS_DIR / manifest_name}")


if __name__ == "__main__":
    main()
