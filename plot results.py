import pandas as pd
import matplotlib.pyplot as plt

# Load your saved dataset (or pass `grouped` directly)
grouped = pd.read_csv("processed_results.csv")  # Save your grouped DataFrame earlier

plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'], grouped['hysteresis_voltage'], alpha=0.5, color='blue')
plt.xlabel('Cycle Number')
plt.ylabel('Hysteresis Voltage (V)')
plt.title('Hysteresis Voltage vs. Cycle Number')
plt.grid(True)
plt.show()
