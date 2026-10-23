"""Energy-proxy analysis using job-level grouped-test predictions.

This script reads the new prediction CSVs that include id_job, node_id, and gpu_index.
It does not reconstruct a random row-level split. It verifies prediction identity by joining
against the corresponding early-window dataset, then compares the predictive PWM policy
with stepped and always-100%-PWM reference policies.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\datacenter_project")
OUTPUT_DIR = PROJECT_DIR / "energy_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
RUNTIME_FILE = PROJECT_DIR / "runtime_per_job.csv"
PWM_FILE = PROJECT_DIR / "pwm_lookup_table.csv"
WINDOWS = [10, 30, 60, 120, 180, 240, 300, 360, 600]
KEYS = ["id_job", "node_id", "gpu_index"]
TARGET = "full_temperature_gpu_peak"

# Must match the controller's tier logic.
def temp_to_tier(temp: float) -> str:
    if temp < 35:
        return "Low"
    if temp < 45:
        return "Medium"
    if temp < 71:
        return "High"
    return "Critical"


def read_pwm_lookup() -> dict[str, float]:
    if not PWM_FILE.is_file():
        raise FileNotFoundError(f"PWM lookup table not found: {PWM_FILE}")
    table = pd.read_csv(PWM_FILE)
    required = {"thermal_region", "pwm_pct_clipped"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"PWM lookup table missing columns: {sorted(missing)}")
    table["thermal_region"] = table["thermal_region"].astype(str).str.strip()
    table["pwm_pct_clipped"] = pd.to_numeric(table["pwm_pct_clipped"], errors="raise")
    lookup = dict(zip(table["thermal_region"], table["pwm_pct_clipped"].astype(float)))
    tiers = ["Low", "Medium", "High", "Critical"]
    absent = [tier for tier in tiers if tier not in lookup]
    if absent:
        raise ValueError(f"PWM lookup table missing tiers: {absent}")
    for tier, pct in lookup.items():
        if not 0 <= pct <= 100:
            raise ValueError(f"PWM for {tier} must be between 0 and 100; found {pct}")
    return {tier: lookup[tier] for tier in tiers}


def load_runtime() -> pd.DataFrame:
    if not RUNTIME_FILE.is_file():
        raise FileNotFoundError(f"Runtime file not found: {RUNTIME_FILE}")
    runtime = pd.read_csv(RUNTIME_FILE)
    required = {"id_job", "runtime_seconds"}
    missing = required - set(runtime.columns)
    if missing:
        raise ValueError(f"Runtime file missing columns: {sorted(missing)}")
    runtime = runtime[["id_job", "runtime_seconds"]].copy()
    runtime["id_job"] = pd.to_numeric(runtime["id_job"], errors="raise").astype("int64")
    runtime["runtime_seconds"] = pd.to_numeric(runtime["runtime_seconds"], errors="coerce")
    runtime = runtime.dropna(subset=["runtime_seconds"])
    if runtime["id_job"].duplicated().any():
        raise ValueError("runtime_per_job.csv has multiple rows for the same id_job.")
    if (runtime["runtime_seconds"] < 0).any():
        raise ValueError("Negative job runtime was found.")
    return runtime


def read_predictions_and_validate(window: int, pwm_mapping: dict[str, float], runtime: pd.DataFrame) -> pd.DataFrame:
    data_path = PROJECT_DIR / f"{window}s" / f"model_a_early_{window}sec_vs_full_v4.csv"
    pred_path = PROJECT_DIR / f"{window}s" / "XGBOOST" / f"model_a_{window}s_test_predictions.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Early-window data file not found: {data_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}\nRun retrain_grouped_models.py first.")

    data = pd.read_csv(data_path, usecols=KEYS + [TARGET])
    pred = pd.read_csv(pred_path)
    required_pred = set(KEYS + ["actual_peak_temp", "predicted_peak_temp", "error"])
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(
            f"{pred_path} is missing {sorted(missing)}. "
            "These must be the new prediction CSVs produced by retrain_grouped_models.py."
        )

    for frame in (data, pred):
        frame["id_job"] = pd.to_numeric(frame["id_job"], errors="raise").astype("int64")
        frame["gpu_index"] = pd.to_numeric(frame["gpu_index"], errors="raise").astype("int64")
        frame["node_id"] = frame["node_id"].astype(str).str.strip()
    if data.duplicated(KEYS).any() or pred.duplicated(KEYS).any():
        raise ValueError(f"Duplicate job-node-GPU keys found for {window}s data or predictions.")

    # Confirm all predictions map to the correct source record.
    merged = pred.merge(
        data,
        how="left",
        on=KEYS,
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"{window}s: {unmatched} prediction rows did not match the source dataset.")
    merged = merged.drop(columns=["_merge"])

    stored_actual = pd.to_numeric(merged["actual_peak_temp"], errors="raise").to_numpy()
    source_actual = pd.to_numeric(merged[TARGET], errors="raise").to_numpy()
    if not np.allclose(stored_actual, source_actual, rtol=0, atol=1e-8):
        raise ValueError(f"{window}s: prediction targets do not match the corresponding source rows.")

    merged["actual_peak_temp"] = source_actual
    merged["predicted_peak_temp"] = pd.to_numeric(merged["predicted_peak_temp"], errors="raise")
    merged["error"] = merged["predicted_peak_temp"] - merged["actual_peak_temp"]

    merged = merged.merge(runtime, how="left", on="id_job", validate="many_to_one", indicator="runtime_merge")
    missing_runtime = int(merged["runtime_seconds"].isna().sum())
    if missing_runtime:
        raise ValueError(f"{window}s: {missing_runtime} prediction rows could not be matched to runtime_per_job.csv.")
    merged = merged.drop(columns=["runtime_merge"])

    merged["predicted_tier"] = merged["predicted_peak_temp"].apply(temp_to_tier)
    merged["actual_tier"] = merged["actual_peak_temp"].apply(temp_to_tier)
    merged["predictive_pwm_pct"] = merged["predicted_tier"].map(pwm_mapping)
    merged["stepped_reference_pwm_pct"] = merged["actual_tier"].map({
        "Low": 40.0,
        "Medium": 60.0,
        "High": 80.0,
        "Critical": 100.0,
    })
    merged["always_100_pwm_pct"] = 100.0

    runtime_seconds = merged["runtime_seconds"].to_numpy(dtype=float)
    for label, col in [
        ("predictive", "predictive_pwm_pct"),
        ("stepped_reference", "stepped_reference_pwm_pct"),
        ("always_100", "always_100_pwm_pct"),
    ]:
        pwm_fraction = merged[col].to_numpy(dtype=float) / 100.0
        merged[f"{label}_energy_proxy"] = pwm_fraction ** 3 * runtime_seconds

    merged["observation_window_seconds"] = window
    merged["energy_proxy_note"] = "normalized PWM^3 * job runtime; not measured electrical energy"
    return merged


def summarize(window: int, df: pd.DataFrame) -> dict:
    predictive = float(df["predictive_energy_proxy"].sum())
    stepped = float(df["stepped_reference_energy_proxy"].sum())
    always100 = float(df["always_100_energy_proxy"].sum())
    total_runtime_records = float(df["runtime_seconds"].sum())
    mae = float(np.mean(np.abs(df["error"].to_numpy(dtype=float))))
    rmse = float(np.sqrt(np.mean(np.square(df["error"].to_numpy(dtype=float)))))
    return {
        "window_seconds": window,
        "test_job_gpu_records": len(df),
        "unique_jobs": df["id_job"].nunique(),
        "sum_job_gpu_runtime_seconds": total_runtime_records,
        "predictive_energy_proxy": predictive,
        "stepped_40_60_80_100_energy_proxy": stepped,
        "always_100_energy_proxy": always100,
        "saving_vs_always_100_pct": (1 - predictive / always100) * 100 if always100 else np.nan,
        "saving_vs_stepped_pct": (1 - predictive / stepped) * 100 if stepped else np.nan,
        "mean_actual_peak_temp_C": float(df["actual_peak_temp"].mean()),
        "mean_predicted_peak_temp_C": float(df["predicted_peak_temp"].mean()),
        "MAE_C": mae,
        "RMSE_C": rmse,
        "predictive_energy_pct": (predictive / always100 * 100) if always100 else np.nan,
        "stepped_energy_pct": (stepped / always100 * 100) if always100 else np.nan,
        "always_100_energy_pct": 100.0,
        "energy_proxy_note": "normalized proxy only; not measured fan or facility electricity",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pwm_mapping = read_pwm_lookup()
    runtime = load_runtime()
    print("PWM mapping:", pwm_mapping)
    print("Runtime records:", len(runtime))

    summaries = []
    detail_frames = []
    for window in WINDOWS:
        print(f"Processing grouped-test predictions for {window}s...")
        detail = read_predictions_and_validate(window, pwm_mapping, runtime)
        summary = summarize(window, detail)
        summaries.append(summary)
        detail_frames.append(detail)
        detail_path = OUTPUT_DIR / f"energy_detail_{window}s.csv"
        detail.to_csv(detail_path, index=False)
        print(
            f"  Records={len(detail)}, jobs={detail['id_job'].nunique()}, "
            f"MAE={summary['MAE_C']:.3f} C, "
            f"saving vs always-100 proxy={summary['saving_vs_always_100_pct']:.2f}%"
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "energy_summary_all_windows.csv"
    summary_df.to_csv(summary_path, index=False)
    try:
        summary_df.to_excel(OUTPUT_DIR / "energy_summary_all_windows.xlsx", index=False)
    except Exception as exc:
        warnings.warn(f"Could not save XLSX summary (CSV was saved): {exc}")

    # Figures are supporting outputs; all values come from the grouped test predictions.
    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["window_seconds"], summary_df["predictive_energy_pct"], marker="o", label="Predictive")
    plt.plot(summary_df["window_seconds"], summary_df["stepped_energy_pct"], marker="o", label="Stepped reference")
    plt.plot(summary_df["window_seconds"], summary_df["always_100_energy_pct"], marker="o", label="Always 100% PWM")
    plt.xlabel("Early-observation window (s)")
    plt.ylabel("Normalized energy proxy (% of always-100% reference)")
    plt.title("Cooling-energy proxy across observation windows")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cooling_energy_utilization_across_windows.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["window_seconds"], summary_df["MAE_C"], marker="o")
    plt.xlabel("Early-observation window (s)")
    plt.ylabel("MAE (°C)")
    plt.title("Peak-temperature prediction MAE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "prediction_MAE_across_windows.png", dpi=160)
    plt.close()

    print("\nEnergy proxy analysis completed.")
    print("Summary:", summary_path)
    print("Important: savings are based on a normalized proxy, not measured electrical energy.")


if __name__ == "__main__":
    main()
