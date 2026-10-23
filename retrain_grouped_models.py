"""Train all four model families with a shared job-level split.

Project data expected under C:\\datacenter_project\\{window}s\\model_a_early_{window}sec_vs_full_v4.csv.
The original raw telemetry stays at C:\\mit_supercloud_raw\\nvidia_smi.csv; this script does not rewrite it.

Outputs are written to the existing per-window model folders and project-root summary files.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# =====================================================================
# CONFIGURATION
# =====================================================================
PROJECT_DIR = Path(r"C:\datacenter_project")
RAW_NVIDIA_SMI = Path(r"C:\mit_supercloud_raw\nvidia_smi.csv")  # retained at original location; not read by this script
WINDOWS = [10, 30, 60, 120, 180, 240, 300, 360, 600]
FEATURES = [
    "early_power_draw_W",
    "early_utilization_gpu_pct",
    "early_utilization_memory_pct",
    "early_memory_used_MiB",
    "early_memory_free_MiB",
    "early_temperature_gpu",
    "early_temperature_memory",
]
TARGET = "full_temperature_gpu_peak"
KEYS = ["id_job", "node_id", "gpu_index"]
SPLIT_FILE = PROJECT_DIR / "group_test_job_ids.csv"
TEST_SIZE = 0.15
RANDOM_STATE = 42
CV_FOLDS = 5
# A 20-iteration randomized search is a practical starting budget for this four-model,
# nine-window re-evaluation. If changed, report the actual value in the paper.
N_ITER_SEARCH = 20
SEARCH_N_JOBS = max(1, min(4, (os.cpu_count() or 2) // 2))
BENCH_REPEATS = 50


XGB_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.3, 0.5],
}
RF_PARAMS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 8, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 1.0, None],
}
DT_PARAMS = {
    "max_depth": [3, 5, 8, 12, 16, None],
    "min_samples_split": [2, 3, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
}
LINEAR_ALPHAS = np.logspace(-5, 5, 21)


# =====================================================================
# GENERAL HELPERS
# =====================================================================
def window_dataset_path(window: int) -> Path:
    return PROJECT_DIR / f"{window}s" / f"model_a_early_{window}sec_vs_full_v4.csv"


def read_window(window: int) -> pd.DataFrame:
    path = window_dataset_path(window)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required early-window dataset is missing:\n{path}\n"
            "This script uses the prepared early-window CSV files. It does not regenerate them from nvidia_smi.csv."
        )
    df = pd.read_csv(path)
    required = KEYS + FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["id_job"] = pd.to_numeric(df["id_job"], errors="raise").astype("int64")
    df["gpu_index"] = pd.to_numeric(df["gpu_index"], errors="raise").astype("int64")
    df["node_id"] = df["node_id"].astype(str).str.strip()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid = df[FEATURES + [TARGET]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{path}: {int(invalid.sum())} rows contain missing/non-numeric model fields.")
    if not np.isfinite(df[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: model fields contain infinite values.")
    if df.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate job-node-GPU keys were found.")
    return df


def key_set(df: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, df[KEYS].itertuples(index=False, name=None)))


def get_or_create_test_jobs(base_df: pd.DataFrame) -> set[int]:
    """Create/reuse one test-job manifest shared by every model and window."""
    available = set(base_df["id_job"].unique().tolist())
    if SPLIT_FILE.exists():
        manifest = pd.read_csv(SPLIT_FILE)
        if "id_job" not in manifest.columns:
            raise ValueError(f"Split manifest {SPLIT_FILE} must contain an id_job column.")
        test_jobs = set(pd.to_numeric(manifest["id_job"], errors="raise").astype("int64").tolist())
        if not test_jobs or not test_jobs.issubset(available):
            raise ValueError(
                f"Existing split manifest {SPLIT_FILE} does not match the current 10-second dataset. "
                "Back it up or remove it deliberately, then rerun."
            )
        print(f"Reusing fixed job-level test manifest: {SPLIT_FILE}")
    else:
        unique_jobs = np.array(sorted(available), dtype=np.int64)
        train_jobs, test_jobs_array = train_test_split(
            unique_jobs, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        test_jobs = set(map(int, test_jobs_array))
        pd.DataFrame({"id_job": sorted(test_jobs)}).to_csv(SPLIT_FILE, index=False)
        print(f"Created fixed job-level test manifest: {SPLIT_FILE}")
    print(f"Unique jobs: {len(available):,}; test jobs: {len(test_jobs):,}; train jobs: {len(available - test_jobs):,}")
    return test_jobs


def save_importance_plot(importance_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if importance_df.empty:
        return
    plot_df = importance_df.sort_values("importance_abs" if "importance_abs" in importance_df.columns else "importance")
    xcol = "importance_abs" if "importance_abs" in plot_df.columns else "importance"
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df[xcol])
    plt.xlabel("Absolute coefficient" if xcol == "importance_abs" else "Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def feature_table(model, algorithm: str) -> pd.DataFrame:
    if algorithm == "LinearRegression":
        estimator = model.named_steps["regressor"]
        coef = np.asarray(estimator.coef_).reshape(-1)
        return pd.DataFrame({"feature": FEATURES, "coefficient": coef, "importance_abs": np.abs(coef)})
    values = np.asarray(model.feature_importances_, dtype=float)
    return pd.DataFrame({"feature": FEATURES, "importance": values}).sort_values("importance", ascending=False)


def artifact_path(window: int, algorithm: str, linear_name: str | None = None) -> Path:
    folder_names = {
        "XGBOOST": "XGBOOST",
        "RandomForest": "RandomForest",
        "DecisionTree": "DecisionTree",
        "LinearRegression": "LinearRegression",
    }
    folder = PROJECT_DIR / f"{window}s" / folder_names[algorithm]
    folder.mkdir(parents=True, exist_ok=True)
    if algorithm == "XGBOOST":
        return folder / f"model_a_{window}s_xgboost.json"
    if algorithm == "RandomForest":
        return folder / f"model_rf_{window}s_random_forest.joblib"
    if algorithm == "DecisionTree":
        return folder / f"model_dt_{window}s_decision_tree.joblib"
    if algorithm == "LinearRegression":
        if linear_name not in {"Ridge", "Lasso"}:
            raise ValueError("linear_name must be Ridge or Lasso for LinearRegression artifacts.")
        return folder / f"model_lin_{window}s_{linear_name.lower()}.joblib"
    raise ValueError(algorithm)


def prediction_path(window: int, algorithm: str) -> Path:
    folder = PROJECT_DIR / f"{window}s" / algorithm
    folder.mkdir(parents=True, exist_ok=True)
    prefixes = {
        "XGBOOST": "model_a",
        "RandomForest": "model_rf",
        "DecisionTree": "model_dt",
        "LinearRegression": "model_lin",
    }
    return folder / f"{prefixes[algorithm]}_{window}s_test_predictions.csv"


def load_saved_model(algorithm: str, path: Path):
    if algorithm == "XGBOOST":
        model = XGBRegressor(n_jobs=1)
        model.load_model(str(path))
        return model
    return joblib.load(path)


def benchmark_model(model, algorithm: str, path: Path, X_test: pd.DataFrame, window: int) -> dict:
    x_one = X_test.iloc[[0]].copy()
    for _ in range(5):
        model.predict(x_one)
    durations_ms = []
    for _ in range(BENCH_REPEATS):
        start = time.perf_counter()
        model.predict(x_one)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    batch_durations = []
    for _ in range(3):
        start = time.perf_counter()
        model.predict(X_test)
        batch_durations.append(time.perf_counter() - start)
    batch_sec = float(np.median(batch_durations))
    load_start = time.perf_counter()
    _loaded_model = load_saved_model(algorithm, path)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    del _loaded_model
    return {
        "window": f"{window}s",
        "algorithm": algorithm,
        "single_median_ms": float(np.median(durations_ms)),
        "single_p95_ms": float(np.percentile(durations_ms, 95)),
        "batch_rows_per_s": float(len(X_test) / batch_sec) if batch_sec > 0 else np.nan,
        "size_kb": float(path.stat().st_size / 1024.0),
        "load_time_ms": float(load_ms),
        "data": "grouped_test",
        "model_file": str(path),
    }


def metrics_row(window, algorithm, search, y_true, y_pred, n_train_rows, n_test_rows, train_jobs, test_jobs):
    return {
        "window_sec": window,
        "algorithm": algorithm,
        "cv_r2": float(search.best_score_),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
        "train_rows": int(n_train_rows),
        "test_rows": int(n_test_rows),
        "train_jobs": int(train_jobs),
        "test_jobs": int(test_jobs),
    }


def save_summaries(result_map: dict[str, list[dict]], benchmark_rows: list[dict], split_checks: list[dict]) -> None:
    summary_filenames = {
        "XGBOOST": "model_a_all_windows_summary.csv",
        "RandomForest": "model_rf_all_windows_summary.csv",
        "DecisionTree": "model_dt_all_windows_summary.csv",
        "LinearRegression": "model_lin_all_windows_summary.csv",
    }
    for algo, rows in result_map.items():
        df = pd.DataFrame(rows)
        if algo == "LinearRegression":
            df = df.rename(columns={"algorithm_name": "selected_model"})
        df.to_csv(PROJECT_DIR / summary_filenames[algo], index=False)
    all_metrics = pd.concat([pd.DataFrame(rows) for rows in result_map.values()], ignore_index=True)
    all_metrics.to_csv(PROJECT_DIR / "model_comparison_grouped.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(PROJECT_DIR / "benchmark_results_v2.csv", index=False)
    split_df = pd.DataFrame(split_checks)
    split_df.to_csv(PROJECT_DIR / "group_split_validation.csv", index=False)
    if not pd.DataFrame(benchmark_rows).empty:
        bench = pd.DataFrame(benchmark_rows)
        agg = bench.groupby("algorithm").agg(
            single_ms=("single_median_ms", "median"),
            rows_per_s=("batch_rows_per_s", "median"),
            size_kb=("size_kb", "median"),
            load_time_ms=("load_time_ms", "median"),
            windows=("window", "nunique"),
        ).reset_index()
        fastest = agg["single_ms"].min()
        smallest = agg["size_kb"].min()
        agg["slower_than_fastest_x"] = agg["single_ms"] / fastest
        agg["larger_than_smallest_x"] = agg["size_kb"] / smallest
        agg.to_csv(PROJECT_DIR / "benchmark_summary_v2.csv", index=False)


def main():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    if not PROJECT_DIR.is_dir():
        raise NotADirectoryError(PROJECT_DIR)
    if RAW_NVIDIA_SMI.exists():
        print(f"Raw NVIDIA-SMI file preserved at: {RAW_NVIDIA_SMI}")
    else:
        print(f"NOTE: Raw NVIDIA-SMI file not found at {RAW_NVIDIA_SMI}; not required for this training run because prepared window CSVs are used.")

    frames = {window: read_window(window) for window in WINDOWS}
    ref_keys = key_set(frames[WINDOWS[0]])
    ref_jobs = set(frames[WINDOWS[0]]["id_job"].unique())
    for window, df in frames.items():
        if key_set(df) != ref_keys:
            raise ValueError(f"The {window}s dataset does not contain the same job-node-GPU records as the 10s dataset.")
        if set(df["id_job"].unique()) != ref_jobs:
            raise ValueError(f"The unique job IDs in the {window}s dataset do not match the 10s dataset.")
        print(f"Verified {window}s: {len(df):,} job-GPU rows, {df['id_job'].nunique():,} unique jobs.")

    test_job_ids = get_or_create_test_jobs(frames[WINDOWS[0]])

    result_map = {a: [] for a in ["XGBOOST", "RandomForest", "DecisionTree", "LinearRegression"]}
    benchmark_rows = []
    split_checks = []
    cv = GroupKFold(n_splits=CV_FOLDS)

    for window in WINDOWS:
        df = frames[window].copy()
        test_mask = df["id_job"].isin(test_job_ids)
        train_df = df.loc[~test_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise ValueError(f"Empty train or test partition at {window}s.")
        train_jobs = set(train_df["id_job"].unique())
        test_jobs = set(test_df["id_job"].unique())
        overlap = train_jobs & test_jobs
        if overlap:
            raise AssertionError(f"Job leakage detected for {window}s: {len(overlap)} job IDs overlap.")
        X_train, y_train = train_df[FEATURES], train_df[TARGET]
        X_test, y_test = test_df[FEATURES], test_df[TARGET]
        groups_train = train_df["id_job"].to_numpy()
        split_checks.append({
            "window_sec": window,
            "all_records": len(df),
            "all_unique_jobs": df["id_job"].nunique(),
            "train_records": len(train_df),
            "test_records": len(test_df),
            "train_unique_jobs": len(train_jobs),
            "test_unique_jobs": len(test_jobs),
            "overlap_unique_jobs": len(overlap),
            "key_rows_match_all_windows": True,
        })
        print("\n" + "=" * 78)
        print(f"WINDOW {window}s | train rows={len(train_df)} ({len(train_jobs)} jobs) | test rows={len(test_df)} ({len(test_jobs)} jobs)")
        print("=" * 78)

        # -------------------------------------------------------------
        # XGBoost
        # -------------------------------------------------------------
        xgb_search = RandomizedSearchCV(
            estimator=XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            param_distributions=XGB_PARAMS,
            n_iter=N_ITER_SEARCH,
            cv=cv,
            scoring="r2",
            n_jobs=SEARCH_N_JOBS,
            random_state=RANDOM_STATE + window,
            verbose=0,
            refit=True,
            error_score="raise",
            pre_dispatch=SEARCH_N_JOBS,
        )
        xgb_search.fit(X_train, y_train, groups=groups_train)
        xgb_model = xgb_search.best_estimator_
        xgb_pred = xgb_model.predict(X_test)
        xgb_path = artifact_path(window, "XGBOOST")
        xgb_model.save_model(str(xgb_path))
        xgb_predictions = pd.DataFrame({
            "id_job": test_df["id_job"].to_numpy(),
            "node_id": test_df["node_id"].to_numpy(),
            "gpu_index": test_df["gpu_index"].to_numpy(),
            "actual_peak_temp": y_test.to_numpy(),
            "predicted_peak_temp": xgb_pred,
        })
        xgb_predictions["error"] = xgb_predictions["predicted_peak_temp"] - xgb_predictions["actual_peak_temp"]
        xgb_predictions.to_csv(prediction_path(window, "XGBOOST"), index=False)
        xgb_imp = feature_table(xgb_model, "XGBOOST")
        xgb_dir = xgb_path.parent
        xgb_imp.to_csv(xgb_dir / f"model_a_{window}s_feature_importance.csv", index=False)
        save_importance_plot(xgb_imp, xgb_dir / f"model_a_{window}s_feature_importance.png", f"XGBoost feature importance ({window}s)")
        xgb_row = metrics_row(window, "XGBOOST", xgb_search, y_test, xgb_pred, len(train_df), len(test_df), len(train_jobs), len(test_jobs))
        result_map["XGBOOST"].append(xgb_row)
        benchmark_rows.append(benchmark_model(xgb_model, "XGBOOST", xgb_path, X_test, window))
        print(f"XGBoost: test R2={xgb_row['r2']:.4f}, MAE={xgb_row['mae']:.3f} C, RMSE={xgb_row['rmse']:.3f} C")

        # -------------------------------------------------------------
        # Random Forest
        # -------------------------------------------------------------
        rf_search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            param_distributions=RF_PARAMS,
            n_iter=N_ITER_SEARCH,
            cv=cv,
            scoring="r2",
            n_jobs=SEARCH_N_JOBS,
            random_state=RANDOM_STATE + window,
            verbose=0,
            refit=True,
            error_score="raise",
            pre_dispatch=SEARCH_N_JOBS,
        )
        rf_search.fit(X_train, y_train, groups=groups_train)
        rf_model = rf_search.best_estimator_
        rf_pred = rf_model.predict(X_test)
        rf_path = artifact_path(window, "RandomForest")
        joblib.dump(rf_model, rf_path, compress=0)
        rf_predictions = pd.DataFrame({
            "id_job": test_df["id_job"].to_numpy(),
            "node_id": test_df["node_id"].to_numpy(),
            "gpu_index": test_df["gpu_index"].to_numpy(),
            "actual_peak_temp": y_test.to_numpy(),
            "predicted_peak_temp": rf_pred,
        })
        rf_predictions["error"] = rf_predictions["predicted_peak_temp"] - rf_predictions["actual_peak_temp"]
        rf_predictions.to_csv(prediction_path(window, "RandomForest"), index=False)
        rf_imp = feature_table(rf_model, "RandomForest")
        rf_dir = rf_path.parent
        rf_imp.to_csv(rf_dir / f"model_rf_{window}s_feature_importance.csv", index=False)
        save_importance_plot(rf_imp, rf_dir / f"model_rf_{window}s_feature_importance.png", f"Random Forest feature importance ({window}s)")
        rf_row = metrics_row(window, "RandomForest", rf_search, y_test, rf_pred, len(train_df), len(test_df), len(train_jobs), len(test_jobs))
        result_map["RandomForest"].append(rf_row)
        benchmark_rows.append(benchmark_model(rf_model, "RandomForest", rf_path, X_test, window))
        print(f"Random Forest: test R2={rf_row['r2']:.4f}, MAE={rf_row['mae']:.3f} C, RMSE={rf_row['rmse']:.3f} C")

        # -------------------------------------------------------------
        # Decision Tree
        # -------------------------------------------------------------
        dt_search = RandomizedSearchCV(
            estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
            param_distributions=DT_PARAMS,
            n_iter=N_ITER_SEARCH,
            cv=cv,
            scoring="r2",
            n_jobs=SEARCH_N_JOBS,
            random_state=RANDOM_STATE + window,
            verbose=0,
            refit=True,
            error_score="raise",
            pre_dispatch=SEARCH_N_JOBS,
        )
        dt_search.fit(X_train, y_train, groups=groups_train)
        dt_model = dt_search.best_estimator_
        dt_pred = dt_model.predict(X_test)
        dt_path = artifact_path(window, "DecisionTree")
        joblib.dump(dt_model, dt_path, compress=0)
        dt_predictions = pd.DataFrame({
            "id_job": test_df["id_job"].to_numpy(),
            "node_id": test_df["node_id"].to_numpy(),
            "gpu_index": test_df["gpu_index"].to_numpy(),
            "actual_peak_temp": y_test.to_numpy(),
            "predicted_peak_temp": dt_pred,
        })
        dt_predictions["error"] = dt_predictions["predicted_peak_temp"] - dt_predictions["actual_peak_temp"]
        dt_predictions.to_csv(prediction_path(window, "DecisionTree"), index=False)
        dt_imp = feature_table(dt_model, "DecisionTree")
        dt_dir = dt_path.parent
        dt_imp.to_csv(dt_dir / f"model_dt_{window}s_feature_importance.csv", index=False)
        save_importance_plot(dt_imp, dt_dir / f"model_dt_{window}s_feature_importance.png", f"Decision Tree feature importance ({window}s)")
        dt_row = metrics_row(window, "DecisionTree", dt_search, y_test, dt_pred, len(train_df), len(test_df), len(train_jobs), len(test_jobs))
        result_map["DecisionTree"].append(dt_row)
        benchmark_rows.append(benchmark_model(dt_model, "DecisionTree", dt_path, X_test, window))
        print(f"Decision Tree: test R2={dt_row['r2']:.4f}, MAE={dt_row['mae']:.3f} C, RMSE={dt_row['rmse']:.3f} C")

        # -------------------------------------------------------------
        # Regularized Linear Regression: compare Ridge and Lasso within grouped CV
        # -------------------------------------------------------------
        ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(random_state=RANDOM_STATE)),
        ])
        ridge_search = GridSearchCV(
            ridge_pipe,
            param_grid={"regressor__alpha": LINEAR_ALPHAS},
            cv=cv,
            scoring="r2",
            n_jobs=SEARCH_N_JOBS,
            refit=True,
            error_score="raise",
        )
        ridge_search.fit(X_train, y_train, groups=groups_train)

        lasso_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Lasso(random_state=RANDOM_STATE, max_iter=20000)),
        ])
        lasso_search = GridSearchCV(
            lasso_pipe,
            param_grid={"regressor__alpha": LINEAR_ALPHAS},
            cv=cv,
            scoring="r2",
            n_jobs=SEARCH_N_JOBS,
            refit=True,
            error_score="raise",
        )
        lasso_search.fit(X_train, y_train, groups=groups_train)
        if ridge_search.best_score_ >= lasso_search.best_score_:
            lin_search, lin_name = ridge_search, "Ridge"
        else:
            lin_search, lin_name = lasso_search, "Lasso"
        lin_model = lin_search.best_estimator_
        lin_pred = lin_model.predict(X_test)
        lin_path = artifact_path(window, "LinearRegression", lin_name)
        joblib.dump(lin_model, lin_path, compress=0)
        lin_predictions = pd.DataFrame({
            "id_job": test_df["id_job"].to_numpy(),
            "node_id": test_df["node_id"].to_numpy(),
            "gpu_index": test_df["gpu_index"].to_numpy(),
            "actual_peak_temp": y_test.to_numpy(),
            "predicted_peak_temp": lin_pred,
        })
        lin_predictions["error"] = lin_predictions["predicted_peak_temp"] - lin_predictions["actual_peak_temp"]
        lin_predictions.to_csv(prediction_path(window, "LinearRegression"), index=False)
        lin_imp = feature_table(lin_model, "LinearRegression")
        lin_dir = lin_path.parent
        lin_imp.to_csv(lin_dir / f"model_lin_{window}s_coefficients.csv", index=False)
        save_importance_plot(lin_imp, lin_dir / f"model_lin_{window}s_coefficients.png", f"{lin_name} coefficients ({window}s)")
        # Keep a combined score for the selected regularized linear model.
        lin_row = metrics_row(window, "LinearRegression", lin_search, y_test, lin_pred, len(train_df), len(test_df), len(train_jobs), len(test_jobs))
        lin_row["algorithm_name"] = lin_name
        lin_row["ridge_cv_r2"] = float(ridge_search.best_score_)
        lin_row["lasso_cv_r2"] = float(lasso_search.best_score_)
        result_map["LinearRegression"].append(lin_row)
        benchmark_rows.append(benchmark_model(lin_model, "LinearRegression", lin_path, X_test, window))
        print(f"Linear Regression ({lin_name}): test R2={lin_row['r2']:.4f}, MAE={lin_row['mae']:.3f} C, RMSE={lin_row['rmse']:.3f} C")

        # Save all result files after each completed window, so progress is not lost if a later window fails.
        save_summaries(result_map, benchmark_rows, split_checks)

    print("\nTraining and evaluation completed.")
    print(f"Grouped split validation: {PROJECT_DIR / 'group_split_validation.csv'}")
    print(f"Combined model comparison: {PROJECT_DIR / 'model_comparison_grouped.csv'}")
    print(f"Benchmark results: {PROJECT_DIR / 'benchmark_results_v2.csv'}")
    print("No job ID should occur in both the training and test partitions.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
