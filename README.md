# Job-Level Predictive GPU Thermal Management for Energy-Efficient Data Centre Cooling

## Overview

This project presents a machine-learning-based framework for **predictive GPU thermal management** in GPU-accelerated data centres and HPC environments.

Instead of waiting for a GPU to become hot and then increasing cooling, the system uses the **early telemetry of a running job** to predict its eventual peak GPU temperature. The prediction is then converted into a thermal-control decision and PWM cooling command.

The project combines:

- Job-level GPU telemetry
- Early-window feature extraction
- Machine-learning temperature prediction
- Thermal-region classification
- Safety-aware control
- Majority-voting debounce
- PWM-based cooling
- Runtime-weighted normalized cooling-energy analysis

---

## 🎯 Objectives

- Predict the eventual peak GPU temperature from early-job telemetry.
- Compare multiple ML algorithms across nine observation windows.
- Select a suitable model for predictive thermal control.
- Use predicted temperature together with current sensor temperature.
- Prevent unsafe operation using a thermal safety ceiling.
- Reduce unnecessary cooling through workload-aware PWM control.
- Quantify modeled cooling-energy reduction against reference strategies.

---

## 📊 Dataset

The project uses production GPU telemetry from the **MIT SuperCloud** dataset.

### Dataset characteristics

- GPU: NVIDIA Tesla V100
- Raw telemetry: approximately 44 GB
- Approximately **2,508 unique jobs**
- **2,739 Job–Node–GPU records** after processing
- Approximately 26 days of telemetry
- Collection period: **5 February – 3 March 2021**

Records are analyzed at the:

```text
Job + Node + GPU
```

level using:

```text
id_job + Node + gpu_index
```

---

## 🧹 Data Processing

The raw telemetry is cleaned before model development.

Main processing steps include:

- Large job-ID handling
- Timestamp cleaning
- Node-ID cleaning
- Removal of records with missing job/GPU/timestamp information
- Prevention of timestamp leakage across processing chunks
- Job–Node–GPU grouping
- Generation of early-observation datasets
- Final data verification

---

## ⏱️ Early Prediction Windows

The project evaluates nine early-observation windows:

```text
10s
30s
60s
120s
180s
240s
300s
360s
600s
```

For each window, early-job telemetry is used to predict the eventual full-job peak GPU temperature.

---

# 🤖 Machine Learning

Four model families are compared:

1. **XGBoost**
2. **Random Forest**
3. **Decision Tree**
4. **Regularized Linear Regression**
   - Ridge
   - Lasso

This gives:

```text
4 model families × 9 windows = 36 experiments
```

### Final XGBoost features

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

### Target

```text
full_temperature_gpu_peak
```

### Evaluation

The final methodology uses:

```text
85% → training/development
15% → untouched test set
```

with:

```python
random_state = 42
```

Hyperparameter tuning and cross-validation are performed within the training portion.

---

# 🏆 Model Selection

XGBoost is selected for the controller because it provides strong predictive performance while offering a practical accuracy/complexity trade-off.

Representative XGBoost performance:

| Window | Approx. R² |
|---:|---:|
| 10s | ~0.60 |
| 240s | ~0.82 |
| 600s | ~0.92 |

The project's metrics workbook should be treated as the authoritative source for exact experiment results.

---

# 🎮 Predictive Thermal Controller

The controller follows a three-layer architecture:

```text
          Early GPU Telemetry
                  │
                  ▼
      ┌───────────────────────┐
      │ Layer 1               │
      │ XGBoost Prediction    │
      │ Eventual Peak Temp    │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │ Layer 2               │
      │ Thermal Control       │
      │                       │
      │ Prediction + Sensor   │
      │ Safety Ceiling        │
      │ Majority Debounce     │
      └───────────┬───────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │ Layer 3               │
      │ PWM / Fan Actuation   │
      └───────────────────────┘
```

---

# 🌡️ Thermal Regions

The controller uses four temperature regions:

| Tier | Temperature |
|---|---:|
| Low | `< 35°C` |
| Medium | `35°C – <45°C` |
| High | `45°C – <71°C` |
| Critical | `≥ 71°C` |

The controller considers both:

- Predicted eventual peak temperature
- Current measured temperature

The more severe thermal tier is selected.

---

# 🚨 Safety Control

A safety ceiling of:

```text
82°C
```

is used.

If the measured temperature reaches or exceeds this value, the controller immediately forces:

```text
Critical
```

and bypasses normal debounce behavior.

---

# 🔄 Majority-Voting Debounce

A three-sample majority mechanism prevents unnecessary rapid switching.

Example:

```text
High → Low → High
```

remains:

```text
High
```

and:

```text
Critical → High → Critical
```

remains:

```text
Critical
```

This reduces actuator oscillation caused by noisy sensor/model decisions.

---

# 🌀 PWM Cooling

The current data-derived PWM mapping used in the energy analysis is:

| Thermal Tier | PWM |
|---|---:|
| Low | 30% |
| Medium | 30% |
| High | 47.5% |
| Critical | 100% |

PWM values are maintained through a lookup table so calibration data remains separate from controller logic.

---
#**Energy Analysis**
## Important: Cooling-Energy Proxy vs. Measured Energy

The MIT SuperCloud GPU telemetry dataset used in this project does **not provide direct electrical power measurements for the cooling fans or blowers**.

Therefore, this project does **not claim measured cooling energy in Joules (J), Watt-hours (Wh), or kilowatt-hours (kWh)**. Instead, a **normalized cooling-energy proxy** is used to compare the relative energy demand of different cooling strategies.

The proxy is defined as:

$$
E_{\text{proxy}} =
\left(\frac{PWM}{100}\right)^3
\times
runtime_{\text{seconds}}
$$

where:

- `PWM` is the commanded fan speed as a percentage (0–100%).
- `runtime_seconds` is the GPU job runtime in seconds.
- The cubic relationship represents an **assumed relative fan-power scaling model**, where fan power is approximately proportional to the cube of fan speed.
- The resulting value is a **dimensionless normalized proxy**, not a physical energy measurement.

### Example

For a fan operating at 50% PWM for 100 seconds:

$$
E_{\text{proxy}} =
(0.5)^3 \times 100
= 12.5
$$

The value `12.5` represents a **relative cooling-energy proxy value**. It should **not** be interpreted as 12.5 Joules, Wh, or kWh.

## 📐 Cooling-Energy Proxy and Normalized Energy

### Important: Proxy vs. Measured Energy

The MIT SuperCloud GPU telemetry dataset used in this project does **not provide direct electrical power measurements for the cooling fans or blowers**.

Therefore, this project does **not claim measured cooling energy in Joules (J), Watt-hours (Wh), or kilowatt-hours (kWh)**.

Instead, a **normalized cooling-energy proxy** is used to compare the relative cooling demand of different control strategies.

The proxy is defined as:

$$
E_{\text{proxy}} =
\left(\frac{PWM}{100}\right)^3
\times
runtime_{\text{seconds}}
$$

where:

- $PWM$ is the commanded fan speed in percent (0–100%).
- $runtime_{\text{seconds}}$ is the GPU job runtime in seconds.
- The cubic relationship represents an **assumed relative fan-power scaling model**, where fan power is approximated as proportional to the cube of fan speed.
- $E_{\text{proxy}}$ is a **normalized, dimensionless proxy value** and is not a physical energy measurement.

### Example

Suppose the cooling fan operates at **50% PWM** for **100 seconds**.

The cooling-energy proxy is calculated as:

**Step 1 — Convert PWM to a normalized value**

`PWM / 100 = 50 / 100 = 0.5`

**Step 2 — Apply the cubic fan-power relationship**

`(0.5)³ = 0.125`

**Step 3 — Multiply by the runtime**

`0.125 × 100 seconds = 12.5`

Therefore:

> **Cooling-Energy Proxy = 12.5**

The value **12.5 is a normalized proxy value**. It does **not** represent 12.5 Joules (J), Watt-hours (Wh), or kilowatt-hours (kWh).

### Formula

The general proxy formula is:

`E_proxy = (PWM / 100)³ × runtime_seconds`

> **Important:** `12.5` is a normalized proxy value. It does **not** represent 12.5 Joules (J), Watt-hours (Wh), or kilowatt-hours (kWh).

### 📊 Normalized Cooling Energy

To make the results easier to interpret, the cooling-energy proxy of each strategy is normalized against the **continuous 100% PWM baseline**.

The normalized cooling energy is:

$$
E_{\text{normalized}} =
\frac{E_{\text{strategy}}}
{E_{\text{always-100\%}}}
\times 100
$$

where:

- $E_{\text{strategy}}$ = cooling-energy proxy of the evaluated strategy.
- $E_{\text{always-100\%}}$ = cooling-energy proxy when the fan operates continuously at 100% PWM.
- $E_{\text{normalized}}$ = relative cooling-energy demand expressed as a percentage of the always-100% baseline.

Therefore:

| Normalized Energy | Interpretation |
|---:|---|
| **100%** | Same modeled cooling demand as continuous 100% PWM |
| **50%** | Half the modeled cooling demand of the 100% baseline |
| **20%** | One-fifth of the modeled cooling demand of the 100% baseline |
| **0%** | No modeled cooling demand |

Lower normalized energy indicates lower **modeled cooling demand**.

---

### 🔋 Cooling Strategies Compared

The main energy analysis compares three strategies:

1. **Predictive Controller**  
   Fan speed is selected using the proposed job-level predictive thermal controller.

2. **40/60/80/100% Reference**  
   Fan speed is selected according to predefined thermal tiers.

3. **Always-100% Baseline**  
   Fan operates continuously at 100% PWM and represents the reference maximum-cooling condition.

The main energy graph plots the normalized cooling-energy proxy of all three strategies on the same scale.

---

### 📋 Stepped Reference Strategy

The stepped reference strategy uses the following PWM levels:

| Thermal Tier | Temperature Region | Reference PWM |
|---|---|---:|
| Low | < 35°C | 40% |
| Medium | 35–<45°C | 60% |
| High | 45–<71°C | 80% |
| Critical | ≥71°C | 100% |

For this comparison, the **actual eventual peak temperature** is used to determine the thermal tier.

Therefore, the stepped strategy represents an **oracle/reference benchmark rather than a causal real-time controller**, because a real controller would not know the eventual peak temperature in advance.

---

## 📈 Proxy-Based Energy Savings

### Savings Against Always-100% Cooling

The percentage reduction relative to continuous 100% cooling is:

$$
\text{Savings}_{100}(\%) =
\left(
1 -
\frac{E_{\text{predictive}}}
{E_{\text{always-100\%}}}
\right)
\times 100
$$

Because normalized energy is defined relative to the always-100% baseline, this can also be written as:

$$
\text{Savings}_{100}(\%) =
100 - E_{\text{normalized,predictive}}
$$

For example, if:

$$
E_{\text{normalized,predictive}} = 40\%
$$

then:

$$
\text{Savings}_{100} = 100 - 40 = 60\%
$$

This means the predictive controller uses **60% less modeled cooling-energy proxy** than continuous 100% cooling.

---

### Savings Against the Stepped Reference

The percentage reduction relative to the stepped reference is:

$$
\text{Savings}_{\text{stepped}}(\%) =
\left(
1 -
\frac{E_{\text{predictive}}}
{E_{\text{stepped}}}
\right)
\times 100
$$

A positive value indicates that the predictive controller has a lower modeled cooling-energy proxy than the stepped reference.

A negative value indicates that the predictive controller has a higher modeled cooling-energy proxy than the stepped reference.

---

### ⚠️ Interpretation of the Energy Results

The reported energy percentages represent **relative savings estimated using the cooling-energy proxy**.

They should **not be interpreted as measured electrical-energy savings for the complete data centre**.

In particular, the analysis does not directly measure:

- Fan/blower electrical power
- Cooling-system electrical power
- Chiller power
- Pump power
- HVAC power
- Total data-centre power
- Actual Joules, Wh, or kWh consumed by cooling

Therefore, the results should be described as:

> **"proxy-based cooling-energy savings"**

or

> **"estimated reduction in normalized cooling-energy demand"**

rather than as measured data-centre energy savings.

# 🛠️ Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- OpenPyXL
- Jupyter Notebook
- ESP32
- PWM control

---

# 📦 Installation

```bash
git clone https://github.com/Ganesh25-777/GPU-Thermal-Management.git
cd <your-repository>

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
openpyxl
joblib
```

---

# ▶️ Workflow

```text
MIT SuperCloud Telemetry
          │
          ▼
     Data Cleaning
          │
          ▼
 Job–Node–GPU Grouping
          │
          ▼
 Early Window Generation
          │
          ▼
 ┌─────────────────────┐
 │ 10–600 s datasets   │
 └──────────┬──────────┘
            │
            ▼
       Model Training
            │
     ┌──────┼──────┐
     │      │      │
   XGB     RF     DT + LR
     │
     ▼
 Model Comparison
     │
     ▼
 XGBoost Controller
     │
     ▼
 Peak Temperature Prediction
     │
     ▼
 Thermal Decision
     │
     ├── Safety Ceiling
     ├── Majority Debounce
     └── PWM Lookup
            │
            ▼
      Cooling Actuation
            │
            ▼
 Normalized Energy Analysis
```

---

# 📊 Outputs

The energy-analysis workflow generates:

```text
energy_analysis/
├── energy_summary_all_windows.csv
├── energy_summary_all_windows.xlsx
├── energy_detail_10s.csv
├── energy_detail_30s.csv
├── energy_detail_60s.csv
├── energy_detail_120s.csv
├── energy_detail_180s.csv
├── energy_detail_240s.csv
├── energy_detail_300s.csv
├── energy_detail_360s.csv
└── energy_detail_600s.csv
```

Figures:

```text
energy_analysis/figures/
├── normalized_cooling_energy_comparison.png
└── prediction_MAE_across_windows.png
```

---

# ⚠️ Limitations

1. **Cooling energy is modeled, not directly measured.** Absolute Joules/Wh/kWh require actual cooling-system power measurements.
2. The cubic PWM relationship is a simplified fan-power model.
3. The stepped 40/60/80/100% strategy is an oracle/reference benchmark because it uses actual eventual temperature.
4. The models are trained and evaluated on MIT SuperCloud Tesla V100 telemetry and may behave differently on other hardware/workloads.
5. Physical deployment requires additional safety, actuator, and thermal validation.

---

# 🚀 Future Work

- Validate on newer NVIDIA GPU architectures.
- Add workload and thermal features.
- Add prediction uncertainty.
- Perform online model updating.
- Measure actual fan/blower electrical power.
- Convert normalized energy into Wh/kWh using measured power.
- Implement closed-loop ESP32/industrial-controller validation.
- Evaluate multi-GPU coordinated cooling.
- Investigate model-predictive and reinforcement-learning control.
- Evaluate impact on total data-centre energy and PUE.

---

# 📚 Research Contribution

The project combines:

```text
Early GPU Telemetry
        +
Job-Level Temperature Prediction
        +
Thermal-Aware Control
        +
Safety-Ceiling Logic
        +
Majority Debouncing
        +
PWM-Based Cooling
        +
Runtime-Weighted Energy Analysis
```

The central idea is to use **early workload behavior as an indicator of future GPU thermal demand**, allowing cooling resources to be allocated proactively instead of reacting only to current temperature.

---

# 👨‍💻 Project Status

**Status: Research / Prototype**

### Completed

- [x] MIT SuperCloud data preprocessing
- [x] Job–Node–GPU grouping
- [x] Early-window dataset generation
- [x] Multi-model comparison
- [x] XGBoost temperature prediction
- [x] Thermal-region definition
- [x] PWM lookup table
- [x] Predictive controller
- [x] Safety ceiling
- [x] Majority-voting debounce
- [x] Runtime-weighted cooling-energy proxy
- [x] Normalized energy comparison

### Future

- [ ] Physical closed-loop cooling validation
- [ ] Measured fan/cooling power validation
- [ ] Absolute Wh/kWh energy analysis
- [ ] Production-oriented deployment validation

---

### Team Members

- Ganesh H K
- Smita S M
- B G Srusti

# 🙏 Acknowledgements
- Dr. Mala Sinnoor, Assistant Professor, Dr. Ambedkar Institute of Technology for guidance during the project work.
- MIT SuperCloud for the GPU telemetry dataset.
- The open-source Python and machine-learning ecosystem.
- Academic mentors and project contributors supporting the research.

---

## 🔗 Keywords

`GPU Thermal Management` · `GPU Cooling` · `Data Center` · `HPC` · `Machine Learning` · `XGBoost` · `Predictive Cooling` · `Energy Efficiency` ·  `GPU Temperature Prediction` · `Embedded Systems` · `ESP32` · `PWM Control` · `Green Computing` · `Sustainable Computing` 
