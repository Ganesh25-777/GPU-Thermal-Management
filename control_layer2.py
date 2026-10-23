"""Three-layer GPU thermal-management controller simulation.

Layer 1: selects a trained XGBoost model based on elapsed job time and predicts
         eventual peak GPU temperature from the supplied early-window features.
Layer 2: combines predicted temperature with current sensor temperature, uses a
         three-consecutive-sample persistence filter, and applies an immediate
         safety override at 82 C.
Layer 3: maps the final tier to a PWM percentage using pwm_lookup_table.csv.

This file outputs a PWM command for evaluation/demo. It does not send commands to
physical fan, IPMI, DCGM, Modbus, or other hardware interfaces.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


PROJECT_DIR = Path(r"C:\datacenter_project")
MODEL_WINDOWS = [10, 30, 60, 120, 180, 240, 300, 360, 600]
PWM_LOOKUP_FILE = PROJECT_DIR / "pwm_lookup_table.csv"
PWM_COLUMN = "pwm_pct_clipped"

FEATURES = [
    "early_power_draw_W",
    "early_utilization_gpu_pct",
    "early_utilization_memory_pct",
    "early_memory_used_MiB",
    "early_memory_free_MiB",
    "early_temperature_gpu",
    "early_temperature_memory",
]
TIER_ORDER = ["Low", "Medium", "High", "Critical"]
# Fixed boundaries used by both controller and evaluation.
TEMP_THRESHOLDS = {"Low": 35.0, "Medium": 45.0, "High": 71.0}
SAFETY_CEILING_C = 82.0
DEBOUNCE_PERSISTENCE = 3


def load_pwm_lookup_table(csv_path: Path) -> dict[str, float]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"PWM lookup table not found: {csv_path}")
    table = pd.read_csv(csv_path)
    required = {"thermal_region", PWM_COLUMN}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"PWM lookup table is missing columns: {sorted(missing)}")
    table["thermal_region"] = table["thermal_region"].astype(str).str.strip()
    table[PWM_COLUMN] = pd.to_numeric(table[PWM_COLUMN], errors="raise")
    lookup = dict(zip(table["thermal_region"], table[PWM_COLUMN].astype(float)))
    missing_tiers = [tier for tier in TIER_ORDER if tier not in lookup]
    if missing_tiers:
        raise ValueError(f"PWM lookup table is missing tiers: {missing_tiers}")
    for tier in TIER_ORDER:
        if not 0 <= lookup[tier] <= 100:
            raise ValueError(f"PWM for {tier} must be within 0..100; got {lookup[tier]}")
    return {tier: lookup[tier] for tier in TIER_ORDER}


TIER_TO_PWM_PCT = load_pwm_lookup_table(PWM_LOOKUP_FILE)


def temp_to_tier(temp_c: float) -> str:
    if temp_c < TEMP_THRESHOLDS["Low"]:
        return "Low"
    if temp_c < TEMP_THRESHOLDS["Medium"]:
        return "Medium"
    if temp_c < TEMP_THRESHOLDS["High"]:
        return "High"
    return "Critical"


def tier_rank(tier: str) -> int:
    return TIER_ORDER.index(tier)


def higher_tier(tier_a: str, tier_b: str) -> str:
    return tier_a if tier_rank(tier_a) >= tier_rank(tier_b) else tier_b


class ModelRegistry:
    """Loads exact model artifact filenames and selects by elapsed job time."""

    def __init__(self, model_dir: Path = PROJECT_DIR, windows: list[int] | None = None):
        self.model_dir = Path(model_dir)
        self.windows = windows or MODEL_WINDOWS
        self.models: dict[int, XGBRegressor] = {}
        missing = []
        for window in self.windows:
            model_path = self.model_dir / f"{window}s" / "XGBOOST" / f"model_a_{window}s_xgboost.json"
            if not model_path.is_file():
                missing.append(str(model_path))
                continue
            model = XGBRegressor(n_jobs=1)
            model.load_model(str(model_path))
            self.models[window] = model
            print(f"Loaded {window}s model: {model_path}")
        if missing:
            raise FileNotFoundError(
                "One or more trained XGBoost models are missing. Run retrain_grouped_models.py first.\n"
                + "\n".join(missing)
            )
        self.available_windows = sorted(self.models)
        if self.available_windows != sorted(self.windows):
            raise RuntimeError("Not all requested observation-window models were loaded.")

    def select_window(self, elapsed_seconds: float) -> int:
        usable = [window for window in self.available_windows if window <= elapsed_seconds]
        return max(usable) if usable else self.available_windows[0]

    def predict(self, elapsed_seconds: float, feature_row: dict) -> tuple[float, int]:
        window = self.select_window(elapsed_seconds)
        missing = [feature for feature in FEATURES if feature not in feature_row]
        if missing:
            raise ValueError(f"Feature row is missing required fields: {missing}")
        X = pd.DataFrame([{name: feature_row[name] for name in FEATURES}], columns=FEATURES)
        X = X.apply(pd.to_numeric, errors="raise")
        predicted_temp = float(self.models[window].predict(X)[0])
        return predicted_temp, window


class TierPersistenceFilter:
    """Require the same new tier for three consecutive updates outside emergencies."""

    def __init__(self, persistence: int = DEBOUNCE_PERSISTENCE):
        self.persistence = persistence
        self.history: deque[str] = deque(maxlen=persistence)
        self.active_tier = "Low"

    def update(self, candidate_tier: str, bypass: bool = False) -> str:
        if bypass:
            self.active_tier = candidate_tier
            self.history.clear()
            self.history.append(candidate_tier)
            return self.active_tier
        self.history.append(candidate_tier)
        if len(self.history) == self.persistence and all(
            tier == candidate_tier for tier in self.history
        ):
            self.active_tier = candidate_tier
        return self.active_tier


class JobCoolingController:
    """Combines predictive and measured temperatures and returns a PWM command."""

    def __init__(self, job_id: str, registry: ModelRegistry):
        self.job_id = job_id
        self.registry = registry
        self.filter = TierPersistenceFilter()

    def step(self, elapsed_seconds: float, feature_row: dict, current_sensor_temp: float) -> dict:
        current_sensor_temp = float(current_sensor_temp)
        predicted_temp, window_used = self.registry.predict(elapsed_seconds, feature_row)
        predicted_tier = temp_to_tier(predicted_temp)
        sensor_tier = temp_to_tier(current_sensor_temp)
        combined_tier = higher_tier(predicted_tier, sensor_tier)
        safety_triggered = current_sensor_temp >= SAFETY_CEILING_C
        if safety_triggered:
            combined_tier = "Critical"
        final_tier = self.filter.update(combined_tier, bypass=safety_triggered)
        pwm_pct = float(TIER_TO_PWM_PCT[final_tier])
        return {
            "job_id": self.job_id,
            "elapsed_seconds": float(elapsed_seconds),
            "model_window_used_seconds": int(window_used),
            "predicted_peak_temperature_C": round(predicted_temp, 2),
            "predicted_tier": predicted_tier,
            "current_sensor_temperature_C": current_sensor_temp,
            "sensor_tier": sensor_tier,
            "combined_tier_before_persistence": combined_tier,
            "safety_override_triggered": bool(safety_triggered),
            "final_tier": final_tier,
            "pwm_command_percent": pwm_pct,
        }


def demo() -> None:
    print("Loaded PWM mapping:", TIER_TO_PWM_PCT)
    registry = ModelRegistry()
    controller = JobCoolingController("demo_job", registry)
    sample_features = {
        "early_power_draw_W": 150.0,
        "early_utilization_gpu_pct": 70.0,
        "early_utilization_memory_pct": 55.0,
        "early_memory_used_MiB": 12000.0,
        "early_memory_free_MiB": 20000.0,
        "early_temperature_gpu": 45.0,
        "early_temperature_memory": 42.0,
    }
    print("\nNormal-control demonstration (same candidate repeated three times):")
    for idx in range(3):
        result = controller.step(elapsed_seconds=120, feature_row=sample_features, current_sensor_temp=46.0)
        print(f"Update {idx + 1}: {result}")
    print("\nSafety-override demonstration (must immediately issue Critical/100% with current lookup):")
    emergency_controller = JobCoolingController("emergency_demo", registry)
    result = emergency_controller.step(elapsed_seconds=120, feature_row=sample_features, current_sensor_temp=84.0)
    print(result)


if __name__ == "__main__":
    demo()
