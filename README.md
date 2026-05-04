![Python](https://img.shields.io/badge/Python-3.14-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)


# AI-Based Wind Turbine Condition Monitoring Using Machine Learning

## Overview
This project presents a simulation-based approach to wind turbine condition monitoring using machine learning techniques. The system models turbine operational data and applies anomaly detection to identify potential faults in real time.

The goal is to demonstrate how data-driven methods can support predictive maintenance in wind energy systems.

---

## Key Features
- Simulation of wind turbine operational data (vibration, temperature, speed)
- Feature engineering (moving averages, signal trends)
- Anomaly detection using Isolation Forest (unsupervised learning)
- Visualization of fault conditions using time-series plots
- Export of processed dataset for further analysis

---

## Methodology

### 1. Data Simulation
Synthetic SCADA-like data was generated to represent:
- Rotational speed (RPM)
- Temperature (°C)
- Vibration levels

Fault conditions were introduced to simulate:
- Mechanical imbalance (vibration spikes)
- Overheating (temperature rise)

---

### 2. Feature Engineering
Additional features were extracted to improve detection:
- Moving average of vibration signals
- Rate of change of temperature

---

### 3. Machine Learning Model
An **Isolation Forest algorithm** was used to detect anomalies in the dataset.

The model:
- Identifies abnormal patterns without labeled data
- Separates normal vs faulty system behavior

---

## Results

### Vibration Anomaly Detection
![Vibration](results/vibration_plot.png)

The model detects abnormal vibration spikes indicating mechanical faults.

### Temperature Anomaly Detection
![Temperature](results/temperature_plot.png)

Temperature anomalies indicate overheating conditions in turbine operation.

---

## Tools & Technologies
- Python/Pycharm
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Project Structure

## How to Run

```bash
pip install -r requirements.txt
python src/wind_monitoring.py


## Research Report
Full technical report available here:
[Download PDF](report/Wind_Turbine_Monitoring_Report.pdf)


## Future Work
- Apply model to real SCADA datasets
- Improve anomaly detection using deep learning (LSTM)
- Deploy real-time monitoring system


