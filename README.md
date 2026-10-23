# Early-Window GPU Thermal Prediction and Adaptive Cooling Control in Data Centers

A machine-learning-based research project that predicts the eventual peak temperature of a GPU job from its early telemetry and uses that prediction to guide safety-aware, adaptive cooling decisions.

> **Project status:** Research and prototype  
> **Research focus:** Job-level peak-temperature prediction, early-observation-window comparison, predictive cooling control, and normalized cooling-energy proxy analysis.

---

---

## Overview

Conventional reactive cooling responds to the temperature a GPU is experiencing now. This project explores a predictive alternative: use the initial telemetry of a running GPU job to estimate the **eventual peak GPU temperature**, then use that estimate—together with the current measured temperature—to select a cooling tier.

The workflow combines:

- Production GPU telemetry from the MIT SuperCloud environment
- Job–Node–GPU-level data organization
- Early-window feature construction
- Comparative machine-learning experiments
- Eventual peak-temperature prediction
- Thermal-region classification
- Safety-ceiling override
- Three-sample majority-voting debounce
- PWM-based cooling commands
- Runtime-weighted normalized cooling-energy proxy analysis

The intended outcome is a job-aware thermal-control framework that can investigate the trade-off between making a cooling decision early and making an accurate prediction.

## Objectives

- Predict a GPU job's eventual peak temperature from early-stage telemetry.
- Compare four machine-learning model families across nine observation windows.
- Examine the trade-off between prediction timing and accuracy.
- Use both predicted peak temperature and current sensor temperature in the control decision.
- Provide a safety-ceiling override for critical measured temperatures.
- Reduce unnecessary high-speed cooling in the modeled control strategy.
- Compare cooling strategies using a clearly qualified, normalized cooling-energy proxy.
- Prepare the framework for future physical closed-loop and measured-energy validation.

## Dataset

The project uses production GPU telemetry from the **MIT SuperCloud** high-performance computing environment.

| Property | Description |
|---|---|
| GPU architecture in the dataset | NVIDIA Tesla V100 |
| Raw telemetry size | Approximately 44 GB |
| Unique jobs | Approximately 2,508 |
| Processed Job–Node–GPU records | 2,739 |
| Telemetry period described in project notes | 5 February–3 March 2021 |
| Data organization | Job + compute node + GPU index |

Each processed record represents a unique combination of:

```text
id_job + Node + gpu_index
```

The raw telemetry is not included in this repository because of its size. See [Data and Model Availability](#data-and-model-availability).

## Data Processing

The data-processing pipeline prepares raw telemetry for job-level model training.

Key steps include:

1. Cleaning job identifiers and timestamps.
2. Cleaning node and GPU identifiers.
3. Removing records with missing required grouping or timestamp information.
4. Handling large job identifiers safely.
5. Preventing timestamp leakage across chunk boundaries.
6. Grouping records by `id_job`, `Node`, and `gpu_index`.
7. Identifying each group's earliest timestamp.
8. Generating early-window datasets and full-job peak-temperature targets.
9. Running verification checks on processed datasets.

The job start lookup is used during dataset generation to calculate elapsed time from the beginning of each Job–Node–GPU record.

## Early-Observation Windows

The model experiments evaluate nine initial observation windows:

```text
10, 30, 60, 120, 180, 240, 300, 360, and 600 seconds
```

For a given window, the model uses only telemetry derived from that initial period to predict the peak GPU temperature observed over the complete job.

Short windows allow earlier predictions but provide less information. Longer windows provide more telemetry but delay the point at which a prediction can be made. Comparing all nine windows helps characterize this trade-off.

## Machine-Learning Models

Four model families are evaluated:

1. **XGBoost**
2. **Random Forest**
3. **Decision Tree**
4. **Regularized Linear Regression**
   - Ridge
   - Lasso

The experiment design contains four model families across nine windows, corresponding to **36 model-family/window combinations**.

### Prediction features

The final XGBoost controller uses these seven features:

```python
FEATURES = [
    "early_power_draw_W",
    "early_utilization_gpu_pct",
    "early_utilization_memory_pct",
    "early_memory_used_MiB",
    "early_memory_free_MiB",
    "early_temperature_gpu",
    "early_temperature_memory",
]
```

### Prediction target

```text
full_temperature_gpu_peak
```

The target is the maximum GPU temperature observed over the complete job, not merely the temperature at the end of the early-observation window.

## Model Evaluation

The documented training methodology uses an 85% development/training portion and a separate 15% test portion, with `random_state = 42`. Hyperparameter selection and cross-validation should be confined to the training portion; the final test set should remain untouched until final evaluation.

The regression metrics are:

- **R²:** how well predictions explain variation in actual peak temperatures; higher is better.
- **MAE (°C):** mean absolute prediction error; lower is better.
- **RMSE (°C):** root mean squared prediction error, which penalizes large errors more strongly; lower is better.

The tables below reproduce the values in the project's metrics workbook. They are shown at the precision recorded in that workbook. Verify that the workbook corresponds to the final experiment run before citing these values in a paper.

### R² by observation window

| Window | Random Forest | XGBoost | Decision Tree | Linear Regression |
|---:|---:|---:|---:|---:|
| 10 s | 0.603 | 0.599 | 0.566 | 0.550 |
| 30 s | 0.676 | 0.666 | 0.670 | 0.590 |
| 60 s | 0.700 | 0.691 | 0.661 | 0.616 |
| 120 s | 0.722 | 0.729 | 0.664 | 0.639 |
| 180 s | 0.771 | 0.765 | 0.729 | 0.666 |
| 240 s | 0.822 | 0.819 | 0.749 | 0.698 |
| 300 s | 0.841 | 0.835 | 0.777 | 0.725 |
| 360 s | 0.868 | 0.868 | 0.804 | 0.750 |
| 600 s | 0.917 | 0.915 | 0.859 | 0.827 |

### MAE (°C) by observation window

| Window | Random Forest | XGBoost | Decision Tree | Linear Regression |
|---:|---:|---:|---:|---:|
| 10 s | 6.91 | 7.12 | 7.28 | 8.38 |
| 30 s | 6.01 | 6.23 | 6.16 | 7.98 |
| 60 s | 5.76 | 5.98 | 6.15 | 7.56 |
| 120 s | 5.23 | 5.22 | 6.08 | 7.07 |
| 180 s | 4.73 | 4.85 | 5.15 | 6.63 |
| 240 s | 3.95 | 4.03 | 5.01 | 6.21 |
| 300 s | 3.51 | 3.67 | 4.41 | 5.84 |
| 360 s | 3.07 | 3.09 | 3.84 | 5.49 |
| 600 s | 2.38 | 2.44 | 3.16 | 4.38 |

### RMSE (°C) by observation window

| Window | Random Forest | XGBoost | Decision Tree | Linear Regression |
|---:|---:|---:|---:|---:|
| 10 s | 10.46 | 10.51 | 10.91 | 11.12 |
| 30 s | 9.44 | 9.59 | 9.53 | 10.61 |
| 60 s | 9.09 | 9.22 | 9.65 | 10.27 |
| 120 s | 8.74 | 8.64 | 9.60 | 9.95 |
| 180 s | 7.95 | 8.04 | 8.62 | 9.57 |
| 240 s | 7.00 | 7.06 | 8.30 | 9.10 |
| 300 s | 6.62 | 6.75 | 7.83 | 8.68 |
| 360 s | 6.03 | 6.02 | 7.34 | 8.29 |
| 600 s | 4.79 | 4.83 | 6.21 | 6.88 |

### Summary

Performance generally improves as the observation window grows: R² increases while MAE and RMSE decrease. Random Forest has the highest recorded R² at several windows, while XGBoost is close in predictive performance and is used by the controller. Model choice should consider the complete results, inference latency, model size, and control requirements—not R² alone.

## Three-Layer Predictive Controller

The controller links early-window prediction to a cooling command.

```text
             Early GPU Telemetry
                     |
                     v
      +-----------------------------+
      | Layer 1: Prediction         |
      | Estimate eventual peak GPU  |
      | temperature using XGBoost   |
      +-----------------------------+
                     |
                     v
      +-----------------------------+
      | Layer 2: Thermal Control    |
      | Compare predicted and       |
      | measured temperature        |
      | Apply tier and safety logic |
      | Apply majority debounce     |
      +-----------------------------+
                     |
                     v
      +-----------------------------+
      | Layer 3: Cooling Actuation  |
      | Map selected tier to PWM    |
      | command using lookup table  |
      +-----------------------------+
```

At each control step, the framework:

1. Selects an available model window based on elapsed job time.
2. Predicts the eventual peak GPU temperature.
3. Converts the predicted temperature into a thermal tier.
4. Converts the current measured temperature into a thermal tier.
5. Chooses the more severe of the two tiers.
6. Applies the safety-ceiling override when required.
7. Applies the three-sample majority-voting debounce unless the safety override bypasses it.
8. Maps the active tier to a PWM command.

The controller code uses a model registry to load the available saved XGBoost models and a separate lookup table for PWM calibration values.

## Thermal Regions and Safety Logic

The controller uses the following thermal-region thresholds:

| Thermal tier | Temperature |
|---|---:|
| Low | `< 35°C` |
| Medium | `35°C to < 45°C` |
| High | `45°C to < 71°C` |
| Critical | `≥ 71°C` |

Both the predicted eventual peak temperature and the current sensor temperature contribute to the decision. The controller selects the more severe tier so that a lower predicted peak does not override a more concerning current measurement.

### Safety-ceiling override

The configured safety ceiling is:

```text
82°C
```

When the measured temperature reaches or exceeds this ceiling, the controller immediately selects `Critical` and bypasses normal debounce behavior. This is a software control rule in the prototype; physical deployment still requires hardware-specific safety validation.

### Majority-voting debounce

A three-sample majority mechanism is used to reduce rapid tier changes caused by noisy readings or predictions. It uses the most recent three candidate tiers and updates the active tier when at least two samples agree.

Illustrative sequences include:

```text
High -> Low -> High
Critical -> High -> Critical
Medium -> Low -> Medium
Low -> Medium -> Low
```

The safety-ceiling override takes priority over normal debounce behavior.

## PWM Cooling Mapping

The current data-derived mapping used by the predictive energy analysis is:

| Thermal tier | PWM command |
|---|---:|
| Low | 30% |
| Medium | 30% |
| High | 47.5% |
| Critical | 100% |

The controller loads PWM values from a lookup table rather than embedding the mapping directly in the control logic. This keeps calibration values separate from the controller and allows the table to be updated independently.

The PWM values are prototype settings and should not be interpreted as validated settings for every physical fan or data-center cooling system.

## Cooling-Energy Proxy Analysis

### Why a proxy is used

The MIT SuperCloud GPU telemetry used in this project does **not provide direct electrical power measurements for cooling fans, blowers, chillers, pumps, or the complete HVAC system**. Therefore, the analysis estimates relative cooling demand using a modeled proxy; it does not calculate measured data-center cooling energy in joules, watt-hours, or kilowatt-hours.

### Proxy formula

```text
E_proxy = (PWM / 100)^3 * runtime_seconds
```

where:

- `PWM` is the commanded fan speed as a percentage from 0 to 100.
- `runtime_seconds` is the job runtime in seconds.
- The cubic relationship is an assumed relative fan-power scaling model.
- `E_proxy` is a normalized, dimensionless proxy value—not a physical energy measurement.

For example, at 50% PWM for 100 seconds:

```text
E_proxy = (50 / 100)^3 * 100
        = 0.5^3 * 100
        = 12.5
```

The result `12.5` is a proxy value. It is not 12.5 J, Wh, or kWh.

### Normalization

The cooling-energy proxy for each strategy is normalized against the always-100% PWM baseline:

```text
E_normalized (%) =
    E_strategy / E_always_100_percent * 100
```

Interpretation:

| Normalized proxy | Interpretation |
|---:|---|
| 100% | Same modeled demand as continuous 100% PWM |
| 50% | Half the modeled demand of the 100% baseline |
| 20% | One-fifth the modeled demand of the 100% baseline |

Lower normalized values indicate lower modeled cooling demand under this proxy.

### Strategies compared

1. **Predictive controller:** Uses the proposed thermal-control logic and the predictive PWM mapping.
2. **40/60/80/100% stepped reference:** Uses 40%, 60%, 80%, and 100% PWM for Low, Medium, High, and Critical regions, respectively.
3. **Always-100% baseline:** Assumes continuous 100% PWM.

The stepped reference uses the actual eventual peak temperature to select its tier. It is therefore an **oracle/reference benchmark**, not a causal real-time controller, because a live controller would not know the eventual peak temperature in advance.

### Savings calculations

Savings relative to always-100% cooling:

```text
Savings_100 (%) =
    (1 - E_predictive / E_always_100_percent) * 100
```

Savings relative to the stepped reference:

```text
Savings_stepped (%) =
    (1 - E_predictive / E_stepped) * 100
```

These values must be described as **proxy-based cooling-energy savings** or **estimated reductions in normalized cooling-energy demand**. They are not measured electrical-energy savings for a complete data center.

## Workflow

```text
MIT SuperCloud GPU Telemetry
            ↓
       Data Cleaning
            ↓
   Job–Node–GPU Grouping
            ↓
  Early-Window Dataset Generation
            ↓
    Model Training and Evaluation
            ↓
       Model Comparison
            ↓
   XGBoost Predictive Controller
            ↓
  Thermal Decision + Safety Logic
            ↓
        PWM Command
            ↓
   Cooling-Energy Proxy Analysis
```

## Outputs and Visualizations

The analysis workflow can generate per-window prediction results, energy-summary CSV/XLSX files, and figures comparing normalized cooling-energy demand across prediction windows. Exact output files depend on which scripts have been run.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- Joblib
- OpenPyXL
- Jupyter Notebook
- ESP32
- PWM control

## Dependencies

The project uses Python with NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, XGBoost, Joblib, OpenPyXL, and Jupyter Notebook. Install the packages required by the script you intend to run. If a `requirements.txt` file is provided in the repository, use `pip install -r requirements.txt`.

## Data and Model Availability

The approximately 44 GB raw telemetry dataset is stored separately and is not included in this repository. Reproducing the full pipeline requires access to the MIT SuperCloud dataset, correctly configured local paths, generated early-window datasets, trained models, and the PWM lookup table. Avoid committing raw data, virtual environments, temporary files, or unnecessary large model artifacts.

## Limitations

1. **Cooling energy is modeled, not measured.** Direct electrical measurements are needed to estimate actual cooling energy in Wh or kWh.
2. The cubic PWM relationship is a simplified assumption; actual fan and system power characteristics depend on the hardware and operating conditions.
3. The stepped 40/60/80/100% strategy uses the eventual peak temperature and is an oracle/reference benchmark.
4. Results are based on MIT SuperCloud telemetry from NVIDIA Tesla V100 GPUs; performance may differ on other hardware and workloads.
5. Offline evaluation does not by itself prove real-time production effectiveness.
6. Physical deployment requires validation of sensor quality, actuator behavior, safe operating limits, and fail-safe behavior.
7. The current prototype's PWM commands should not be treated as universally valid fan calibration values.

## Future Work

- Validate the framework on newer GPU architectures.
- Evaluate generalization to unseen jobs and different workload types.
- Add prediction uncertainty and confidence-aware control.
- Investigate online model updates.
- Measure actual fan/blower electrical power.
- Convert proxy estimates to Wh/kWh using measured power.
- Validate closed-loop operation using an ESP32 or suitable industrial controller.
- Evaluate coordinated cooling across multiple GPUs.
- Investigate model-predictive control and reinforcement-learning approaches.
- Study the impact on total data-center energy consumption and Power Usage Effectiveness (PUE).

## Research Contributions

The project brings together:

- Early-window GPU telemetry
- Job-level eventual peak-temperature prediction
- Comparative evaluation across prediction windows
- Thermal-aware cooling-tier selection
- Safety-ceiling override
- Majority-voting debounce
- PWM-based cooling commands
- Runtime-weighted normalized cooling-energy proxy analysis

The central research idea is to use early workload behavior as an indicator of future GPU thermal demand and investigate whether that prediction can support proactive cooling decisions.

## Project Status

**Status: Research / Prototype**

### Developed
- [x] Telemetry preprocessing and Job–Node–GPU grouping
- [x] Early-window dataset generation
- [x] Multi-model comparison and XGBoost temperature prediction
- [x] Thermal regions, PWM lookup, and predictive controller
- [x] Safety-ceiling override and majority-voting debounce
- [x] Normalized cooling-energy proxy analysis

### Remaining validation
- [ ] Physical closed-loop cooling validation
- [ ] Measured fan/cooling-power validation and absolute Wh/kWh analysis
- [ ] Production-oriented deployment validation

## Team Members

- **Ganesh H K**
- **Smita S M**
- **B G Srusti**

## Acknowledgements

- Dr. Mala Sinnoor, Assistant Professor, Dr. Ambedkar Institute of Technology, for guidance during the project.
- MIT SuperCloud for the GPU telemetry dataset.
- The open-source Python and machine-learning ecosystem.
- Academic mentors and project contributors supporting the research.

## Keywords

`GPU Thermal Management` · `Early-Window Prediction` · `GPU Temperature Prediction` · `Predictive Cooling` · `Adaptive Cooling Control` · `Data Centers` · `HPC` · `XGBoost` · `Energy Efficiency` · `PWM Control` · `ESP32` · `Sustainable Computing`
"# GPU-Thermal-Management" 
