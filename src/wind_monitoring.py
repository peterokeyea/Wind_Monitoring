import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -----------------------------
# STEP 1: Generate data
# -----------------------------
np.random.seed(42)
time = np.arange(0, 1000)

speed = 1500 + np.random.normal(0, 5, len(time))
temperature = 40 + np.random.normal(0, 1, len(time))
vibration = 0.02 + np.random.normal(0, 0.005, len(time))

# Inject faults
vibration[500:550] += 0.05
temperature[700:750] += 20

# -----------------------------
# STEP 2: Create dataset
# -----------------------------
data = pd.DataFrame({
    'Speed': speed,
    'Temperature': temperature,
    'Vibration': vibration
})

# -----------------------------
# STEP 3: Feature Engineering
# -----------------------------
data['Vibration_MA'] = data['Vibration'].rolling(window=10).mean()
data['Temp_Change'] = data['Temperature'].diff()

data = data.fillna(0)

# -----------------------------
# STEP 4: Machine Learning Model
# -----------------------------
model = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = model.fit_predict(data)

# Convert (-1 = anomaly, 1 = normal)
data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})

# -----------------------------
# STEP 5: Plot results
# -----------------------------
import matplotlib.pyplot as plt

# -------- FIGURE 1 (Vibration) --------
lt.figure()
plt.plot(data['Vibration'], label='Vibration')
plt.scatter(data.index[data['Anomaly'] == 1],
            data['Vibration'][data['Anomaly'] == 1])
plt.title("AI-Based Anomaly Detection (Vibration)")
plt.xlabel("Time")
plt.ylabel("Vibration")
plt.legend()

plt.savefig("vibration_plot.png", dpi=300)   # ✅ SAVE FIRST
plt.show()                          # ✅ THEN SHOW

# -------- FIGURE 2 (Temperature) --------
plt.figure()
plt.plot(data['Temperature'], label='Temperature')
plt.scatter(data.index[data['Anomaly'] == 1],
            data['Temperature'][data['Anomaly'] == 1])
plt.title("AI-Based Anomaly Detection (Temperature)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()

plt.savefig("temperature_plot.png", dpi=300)  # ✅ SAVE FIRST
plt.show()

# -----------------------------
# STEP 6: Save results
# -----------------------------
data.to_csv("wind_monitoring.csv", index=False)

print("dataset saved as wind_monitoring.csv")

import os
print(os.getcwd())
