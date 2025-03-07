import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Load & Preprocess Data
# ========================
file_path = "Copy of Oxford_battery_data_discharge.csv"

data = pd.read_csv(file_path, low_memory=False)
data = data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Ensure current values are negative for discharge
if data['current'].max() > 0:
    data['current'] = -data['current']

# Assume 1 second has passed between each row
data['time_diff'] = 1  # Replacing calculated time difference with a fixed 1-second interval

# Compute incremental discharge (A·s converted to mAh)
data['incremental_discharge'] = (data['current'] * data['time_diff']).abs() / 3600

# Compute cumulative discharge capacity per cycle
data['discharge_capacity'] = data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum() * 1000

# ========================
# 2. Remove Unrealistic Temperature Values
# ========================
data_filtered = data[data['temperature'] >= 35].copy()  # Remove temperature values below 35°C

# ========================
# 3. Aggregate Features for SOH Calculation
# ========================
agg_funcs = {
    'voltage': 'mean',
    'temperature': 'mean',  # Using only filtered temperature values
    'discharge_capacity': 'max'
}

grouped = data_filtered.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()

# Rename columns for clarity
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'temperature': 'avg_temperature',
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)

# ========================
# 4. Compute State of Health (SOH)
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
# 5. Define Features & Labels
# ========================
features = ['avg_voltage', 'avg_temperature', 'cycle_number']
X = grouped[features].values
y = grouped['SOH'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 6. Time-Based Train/Test Split
# ========================
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)

split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# ========================
# 7. Train Models & Evaluate Performance
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
# 8. Print Feature Coefficients
# ========================
print("\nFeature Coefficients for Linear Regression:")
print(dict(zip(features, models["Linear Regression"].coef_)))

# ========================
# 9. Print Key Dataset Metrics (AFTER Model Training)
# ========================
print("\n🔹 Key Dataset Metrics (For Reference, Not Used in Modeling) 🔹")
print(f"Total rows in dataset: {len(data_filtered)}")
print(f"🔹 Unique battery cells in dataset: {data_filtered['cell_number'].unique()}")
print(f"🔹 Total unique battery cells: {len(data_filtered['cell_number'].unique())}")
print(data_filtered['cell_number'].value_counts())
print(f"🔹 Unique discharge cycles: {data_filtered['cycle_number'].nunique()}")

print("\n🔹 Cycle-Based Feature Summary:")
print(f"🔹 Average voltage range: {data_filtered['voltage'].min()}V - {data_filtered['voltage'].max()}V")
print(f"🔹 Average temperature range: {data_filtered['temperature'].min()}°C - {data_filtered['temperature'].max()}°C")
print(f"🔹 Average discharge time per cycle: {data_filtered['time_diff'].mean():.4f} seconds")

