"""Generate IEEE-ready R², MAE, and RMSE model-comparison figures.

Expected input:
    C:\\datacenter_project\\model_comparison_grouped.csv

Expected columns (case-insensitive):
    window_sec, algorithm, r2, mae, rmse

Outputs:
    C:\\datacenter_project\\figures\\model_comparison\\model_comparison_r2.png/.svg
    C:\\datacenter_project\\figures\\model_comparison\\model_comparison_mae.png/.svg
    C:\\datacenter_project\\figures\\model_comparison\\model_comparison_rmse.png/.svg
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(r"C:\datacenter_project")
INPUT_FILE = PROJECT_DIR / "model_comparison_grouped.csv"
OUTPUT_DIR = PROJECT_DIR / "figures" / "model_comparison"

REQUIRED_COLUMNS = ["window_sec", "algorithm", "r2", "mae", "rmse"]
METRICS = {
    "r2": {
        "title": "R² Across Early-Observation Windows",
        "ylabel": "R² score",
        "filename": "model_comparison_r2",
    },
    "mae": {
        "title": "MAE Across Early-Observation Windows",
        "ylabel": "MAE (°C)",
        "filename": "model_comparison_mae",
    },
    "rmse": {
        "title": "RMSE Across Early-Observation Windows",
        "ylabel": "RMSE (°C)",
        "filename": "model_comparison_rmse",
    },
}

# Stable, readable labels for plots and legends.
ALGORITHM_LABELS = {
    "XGBOOST": "XGBoost",
    "XGBOOSTREGRESSOR": "XGBoost",
    "RANDOMFOREST": "Random Forest",
    "RANDOMFORESTREGRESSOR": "Random Forest",
    "DECISIONTREE": "Decision Tree",
    "DECISIONTREEREGRESSOR": "Decision Tree",
    "LINEARREGRESSION": "Regularized Linear Regression",
}


def normalize_algorithm(value: object) -> str:
    raw = str(value).strip()
    key = "".join(ch for ch in raw.upper() if ch.isalnum())
    return ALGORITHM_LABELS.get(key, raw)


def find_column(df: pd.DataFrame, expected: str) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    return lookup.get(expected.lower())


def main() -> int:
    if not INPUT_FILE.exists():
        print(f"ERROR: Results file not found:\n{INPUT_FILE}")
        print("Run retrain_grouped_models.py first, then run this script again.")
        return 1

    try:
        raw = pd.read_csv(INPUT_FILE)
    except Exception as exc:
        print(f"ERROR: Could not read {INPUT_FILE}: {exc}")
        return 1

    rename = {}
    for expected in REQUIRED_COLUMNS:
        actual = find_column(raw, expected)
        if actual is None:
            print(f"ERROR: Required column '{expected}' is missing.")
            print("Columns found:", list(raw.columns))
            return 1
        rename[actual] = expected

    df = raw.rename(columns=rename)[REQUIRED_COLUMNS].copy()
    for col in ["window_sec", "r2", "mae", "rmse"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["algorithm"] = df["algorithm"].map(normalize_algorithm)

    if df[["window_sec", "r2", "mae", "rmse", "algorithm"]].isna().any().any():
        bad_rows = df[df[["window_sec", "r2", "mae", "rmse", "algorithm"]].isna().any(axis=1)]
        print("ERROR: Some rows contain missing or non-numeric metric values:")
        print(bad_rows.to_string(index=False))
        return 1

    df["window_sec"] = df["window_sec"].astype(int)
    df = df.sort_values(["window_sec", "algorithm"])

    duplicate_check = df.duplicated(["window_sec", "algorithm"], keep=False)
    if duplicate_check.any():
        print("ERROR: More than one result exists for an algorithm/window pair.")
        print(df.loc[duplicate_check].to_string(index=False))
        print("Please check that model_comparison_grouped.csv contains the latest single run.")
        return 1

    windows = sorted(df["window_sec"].unique().tolist())
    algorithms = [
        name for name in ["XGBoost", "Random Forest", "Decision Tree", "Regularized Linear Regression"]
        if name in set(df["algorithm"])
    ]

    if len(windows) != 9:
        print(f"WARNING: Found {len(windows)} observation windows: {windows}")
        print("Expected nine windows: 10, 30, 60, 120, 180, 240, 300, 360, 600 seconds.")
    if len(algorithms) < 4:
        print(f"WARNING: Found only these model names: {algorithms}")
        print("Check that results from all four algorithms are in the input CSV.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Windows: {windows}")
    print(f"Models: {algorithms}")
    print(f"Rows: {len(df)}")

    for metric, details in METRICS.items():
        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        for algorithm in algorithms:
            series = (
                df[df["algorithm"] == algorithm]
                .sort_values("window_sec")
            )
            ax.plot(
                series["window_sec"],
                series[metric],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=algorithm,
            )

        ax.set_title(details["title"])
        ax.set_xlabel("Early-observation window (s)")
        ax.set_ylabel(details["ylabel"])
        ax.set_xticks(windows)
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(loc="best", frameon=True, fontsize=8)
        fig.tight_layout()

        base = OUTPUT_DIR / details["filename"]
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {base.with_suffix('.png')}")
        print(f"Saved: {base.with_suffix('.svg')}")
        print(f"Saved: {base.with_suffix('.pdf')}")

    # Also save a tidy copy of the exact data used for plotting.
    clean_csv = OUTPUT_DIR / "model_comparison_plot_data.csv"
    df.to_csv(clean_csv, index=False)
    print(f"Saved plot data: {clean_csv}")
    print("Done. These figures use the grouped-test metrics in model_comparison_grouped.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
