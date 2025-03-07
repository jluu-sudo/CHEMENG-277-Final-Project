import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Load & Preprocess Data (Discharge and Charge)
# ========================
discharge_file = "Copy of Oxford_battery_data_discharge.csv"
charge_file = "Copy of Oxford_battery_data_charge.csv"

# Load datasets
discharge_data = pd.read_csv(discharge_file, low_memory=False)
charge_data = pd.read_csv(charge_file, low_memory=False)

# Sort datasets by cycle order
discharge_data = discharge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])
charge_data = charge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Ensure current values are negative for discharge
if discharge_data['current'].max() > 0:
    discharge_data['current'] = -discharge_data['current']

# Assume 1 second has passed between each row
discharge_data['time_diff'] = 1
charge_data['time_diff'] = 1

# Compute discharge capacity incrementally
discharge_data['incremental_discharge'] = (discharge_data['current'] * discharge_data['time_diff']).abs() / 3600
discharge_data['discharge_capacity'] = discharge_data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum() * 1000

# ========================
# 2. Combine Charge & Discharge for Continuous Timeline
# ========================
# Add a column to indicate whether it's a charge or discharge cycle
discharge_data['type'] = 'discharge'
charge_data['type'] = 'charge'

# Combine both datasets into one continuous time series
full_data = pd.concat([discharge_data, charge_data]).sort_values(by=['cell_number', 'cycle_number', 'time'])

# Re-index time to reflect a continuous sequence within each cycle
full_data['time'] = full_data.groupby(['cell_number', 'cycle_number']).cumcount()

# ========================
# 3. Remove Unrealistic Temperature Values
# ========================
full_data_filtered = full_data[full_data['temperature'] >= 35].copy()

# ========================
# 4. Compute Voltage Hysteresis
# ========================
# Compute hysteresis as the voltage difference between charge and discharge within the same cycle
hysteresis_df = full_data_filtered.pivot_table(index=['cell_number', 'cycle_number'],
                                               columns='type',
                                               values='voltage',
                                               aggfunc='mean')

# Compute hysteresis feature (charge voltage - discharge voltage)
hysteresis_df['hysteresis_voltage'] = hysteresis_df['charge'] - hysteresis_df['discharge']
hysteresis_df.reset_index(inplace=True)

# ========================
# 5. Aggregate Features for SOH Calculation
# ========================
agg_funcs = {
    'voltage': 'mean',
    'temperature': 'mean',
    'discharge_capacity': 'max'
}

grouped = full_data_filtered.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()

# Rename columns for clarity
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'temperature': 'avg_temperature',
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)

# Merge with hysteresis data
grouped = pd.merge(grouped, hysteresis_df[['cell_number', 'cycle_number', 'hysteresis_voltage']],
                   on=['cell_number', 'cycle_number'], how='left')

# ========================
# 6. Compute State of Health (SOH)
# ========================
def compute_soh(df):
    cycle_1_capacity = df.loc[df['cycle_number'] == 1, 'max_discharge_capacity']
    if cycle_1_capacity.empty or cycle_1_capacity.values[0] <= 0:
        df['SOH'] = np.nan
    else:
        baseline = cycle_1_capacity.values[0]
        df['SOH'] = (df['max_discharge_capacity'] / baseline) * 100
    return df

grouped = grouped.groupby('cell_number', group_keys=False).apply(compute_soh).dropna(subset=['SOH'])

# ========================
# 7. Define Features & Labels
# ========================
features = ['avg_voltage', 'avg_temperature', 'cycle_number', 'hysteresis_voltage']
X = grouped[features].values
y = grouped['SOH'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 8. Time-Based Train/Test Split
# ========================
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)

split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# ========================
# 9. Train Models & Evaluate Performance
# ========================
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression (alpha=1)": linear_model.Ridge(alpha=1),
    "Lasso Regression (alpha=1)": linear_model.Lasso(alpha=1, max_iter=10000)
}

print("\n--- Model Performance on Time-Based Train/Test Split (80/20) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}:")
    print(f"  R2 on Training Set = {model.score(X_train, y_train):.4f}")
    print(f"  R2 on Test Set = {model.score(X_test, y_test):.4f}")

# ========================
# 10. Print Feature Coefficients
# ========================
print("\nFeature Coefficients for Linear Regression:")
print(dict(zip(features, models["Linear Regression"].coef_)))

# ========================
# 11. Print Key Dataset Metrics (AFTER Model Training)
# ========================
print("\n🔹 Key Dataset Metrics (For Reference, Not Used in Modeling) 🔹")
print(f"Total rows in dataset: {len(full_data_filtered)}")
print(f"🔹 Unique battery cells in dataset: {full_data_filtered['cell_number'].unique()}")
print(f"🔹 Unique discharge cycles: {full_data_filtered['cycle_number'].nunique()}")
print(f"🔹 Average Hysteresis Voltage: {hysteresis_df['hysteresis_voltage'].mean():.4f} V")
