"""Rebuild the PWM lookup table using training jobs only.

Run from a Python environment with pandas installed:
    python calibrate_pwm_training_only.py

Expected files under C:\\datacenter_project:
    job_summary.csv
    group_test_job_ids.csv

The script keeps a backup of the existing full-dataset PWM table and then
writes a training-only calibration to pwm_lookup_table.csv. It does not edit
or retrain any ML models.
"""
from pathlib import Path
import shutil
import pandas as pd

PROJECT_DIR = Path(r"C:\datacenter_project")
SUMMARY_FILE = PROJECT_DIR / "job_summary.csv"
TEST_IDS_FILE = PROJECT_DIR / "group_test_job_ids.csv"
PWM_FILE = PROJECT_DIR / "pwm_lookup_table.csv"
BACKUP_FILE = PROJECT_DIR / "pwm_lookup_table_full_dataset_backup.csv"
CALIBRATION_INFO_FILE = PROJECT_DIR / "pwm_calibration_train_only_info.csv"

REQUIRED_SUMMARY = {"id_job", "peak_temp_c", "avg_power_w"}


def main() -> None:
    for path in (SUMMARY_FILE, TEST_IDS_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    summary = pd.read_csv(SUMMARY_FILE)
    missing = REQUIRED_SUMMARY - set(summary.columns)
    if missing:
        raise ValueError(f"{SUMMARY_FILE} is missing columns: {sorted(missing)}")

    test_manifest = pd.read_csv(TEST_IDS_FILE)
    if "id_job" not in test_manifest.columns:
        raise ValueError(f"{TEST_IDS_FILE} must contain an id_job column")

    summary["id_job"] = pd.to_numeric(summary["id_job"], errors="raise").astype("int64")
    summary["peak_temp_c"] = pd.to_numeric(summary["peak_temp_c"], errors="raise")
    summary["avg_power_w"] = pd.to_numeric(summary["avg_power_w"], errors="raise")
    test_ids = set(pd.to_numeric(test_manifest["id_job"], errors="raise").astype("int64"))

    all_ids = set(summary["id_job"].unique())
    if not test_ids or not test_ids.issubset(all_ids):
        raise ValueError("Test-job manifest does not match job_summary.csv")

    train = summary.loc[~summary["id_job"].isin(test_ids)].copy()
    if train.empty:
        raise ValueError("No training records remain after removing test jobs")
    if set(train["id_job"].unique()) & test_ids:
        raise AssertionError("Test-job leakage found in PWM calibration")

    q1, q2, q3 = train["peak_temp_c"].quantile([0.25, 0.50, 0.75]).tolist()

    def region(temp: float) -> str:
        if temp < q1:
            return "Low"
        if temp < q2:
            return "Medium"
        if temp < q3:
            return "High"
        return "Critical"

    train["thermal_region"] = train["peak_temp_c"].map(region)
    order = ["Low", "Medium", "High", "Critical"]
    region_power = (
        train.groupby("thermal_region")["avg_power_w"]
        .mean()
        .reindex(order)
    )
    if region_power.isna().any():
        raise ValueError(f"At least one region has no training records: {region_power.to_dict()}")
    critical_power = float(region_power.loc["Critical"])
    if critical_power <= 0:
        raise ValueError("Critical-region mean power must be positive")

    table = pd.DataFrame({
        "thermal_region": order,
        "region_avg_power_w": region_power.to_numpy(dtype=float),
    })
    table["relative_load"] = table["region_avg_power_w"] / critical_power
    table["pwm_pct"] = (table["relative_load"] * 100).round(1)
    table["pwm_pct_clipped"] = table["pwm_pct"].clip(lower=30.0, upper=100.0)

    # Preserve the old mapping once before replacing it.
    if PWM_FILE.exists() and not BACKUP_FILE.exists():
        shutil.copy2(PWM_FILE, BACKUP_FILE)
        print(f"Backed up previous PWM table to: {BACKUP_FILE}")

    table.to_csv(PWM_FILE, index=False)
    pd.DataFrame([{
        "calibration_source": "job_summary.csv excluding group_test_job_ids.csv",
        "training_job_count": train["id_job"].nunique(),
        "training_job_gpu_records": len(train),
        "test_job_count_excluded": len(test_ids),
        "q1_peak_temperature_C": q1,
        "q2_peak_temperature_C": q2,
        "q3_peak_temperature_C": q3,
        "minimum_pwm_percent": 30.0,
        "note": "Calibration uses training jobs only; proxy energy is not measured electrical energy.",
    }]).to_csv(CALIBRATION_INFO_FILE, index=False)

    print("\nTraining-only thermal thresholds (degrees C):")
    print(f"Q1={q1:.1f}, Q2={q2:.1f}, Q3={q3:.1f}")
    print("\nNew training-only PWM lookup table:")
    print(table.to_string(index=False))
    print(f"\nUpdated: {PWM_FILE}")
    print(f"Calibration information: {CALIBRATION_INFO_FILE}")
    print("Next: rerun control_layer2.py, then energy_analysis_grouped.py")


if __name__ == "__main__":
    main()
