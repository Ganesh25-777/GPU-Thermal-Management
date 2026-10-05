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

# ⚡ Energy Analysis

## Important: Proxy vs. Real Energy

The current dataset does not provide direct electrical measurements of the cooling fans/blowers.

Therefore, the project uses a **normalized cooling-energy proxy** rather than claiming measured Joules, Wh, or kWh.

The proxy is:

\[
E_{proxy} =
\left(\frac{PWM}{100}\right)^3
\times runtime_{seconds}
\]

For example, 50% PWM for 100 seconds gives:

\[
(0.5)^3 \times 100 = 12.5
\]

The value `12.5` is a **normalized proxy value**, not 12.5 Joules.

---

## 📐 Normalized Cooling Energy

All strategies are normalized against continuous 100% cooling:

\[
E_{normalized} =
\frac{E_{strategy}}
{E_{always100}}
\times 100
\]

Therefore:

```text
100% → continuous 100% cooling baseline
50%  → half of the baseline modeled cooling energy
20%  → one-fifth of the baseline modeled cooling energy
```

The main energy graph compares:

1. **Predictive Controller**
2. **40/60/80/100% Reference**
3. **Always-100% Baseline**

Lower normalized energy means lower modeled cooling demand.

---

## Reference Strategy

The stepped reference uses:

| Thermal Tier | Reference PWM |
|---|---:|
| Low | 40% |
| Medium | 60% |
| High | 80% |
| Critical | 100% |

Because it uses the actual eventual peak temperature, this is an **oracle/reference benchmark**, not a causal real-time controller.

---

# 📈 Energy-Saving Metrics

Against continuous 100% cooling:

\[
Saving =
\left(
1-\frac{E_{predictive}}
{E_{always100}}
\right)\times100
\]

Against the stepped reference:

\[
Saving =
\left(
1-\frac{E_{predictive}}
{E_{stepped}}
\right)\times100
\]

These represent **modeled cooling-energy savings**, not total data-centre electrical-energy savings.

---

# 📁 Project Structure

```text
GPU-Thermal-Management/
│
├── 01_Raw_Data/
│
├── 02_Data_Cleaning/
│   └── verification/
│
├── 03_Early_Window_Dataset_Generation/
│
├── 04_Early_Window_Datasets/
│   ├── 10s/
│   ├── 30s/
│   ├── 60s/
│   ├── 120s/
│   ├── 180s/
│   ├── 240s/
│   ├── 300s/
│   ├── 360s/
│   └── 600s/
│
├── 05_Model_Training/
│   ├── XGBoost/
│   ├── Random_Forest/
│   ├── Decision_Tree/
│   └── Ridge_Lasso/
│
├── 06_Model_Comparison/
├── 07_Thermal_Regions_PWM/
├── 08_Controller/
├── 09_Energy_Analysis/
├── 10_Hardware_Prototype/
├── 11_Paper/
│
├── README.md
└── requirements.txt
```

---

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
git clone https://github.com/<your-username>/<your-repository>.git
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

# 📄 Citation

```bibtex
@misc{gpu_thermal_management,
  title  = {Job-Level Predictive GPU Thermal Management for Energy-Efficient Data Centre Cooling},
  author = {Ganesh H K},
  year   = {2026},
  note   = {Research Prototype}
}
```

---

# 📜 License

This project is intended for academic and research purposes.

Add an appropriate open-source license after confirming the licensing conditions of the dataset, source code, and third-party materials included in the repository.

---

# 🙏 Acknowledgements

- MIT SuperCloud for the GPU telemetry dataset.
- The open-source Python and machine-learning ecosystem.
- Academic mentors and project contributors supporting the research.

---

## 🔗 Keywords

`GPU Thermal Management` · `GPU Cooling` · `Data Center` · `HPC` · `Machine Learning` · `XGBoost` · `Predictive Cooling` · `Energy Efficiency` · `Tesla V100` · `GPU Temperature Prediction` · `Embedded Systems` · `ESP32` · `PWM Control` · `Green Computing` · `Sustainable Computing`
