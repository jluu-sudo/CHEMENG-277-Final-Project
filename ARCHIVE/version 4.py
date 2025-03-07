import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Load & Preprocess Data
# ========================
file_path = "Copy of Oxford_battery_data_discharge.csv"

data = pd.read_csv(file_path, low_memory=False) #
#Loads the battery discharge dataset from a CSV file using pandas (pd.read_csv).
#low_memory=False ensures efficient reading of large datasets by disabling automatic datatype inference.

data = data.sort_values(by=['cell_number', 'cycle_number', 'time'])
#Sorts the dataset by cell_number, cycle_number, and time to maintain chronological order for proper discharge capacity calculations.

if data['current'].max() > 0:
    data['current'] = -data['current']
#Ensures that all current values are negative since the dataset records discharge data. If any positive values exist, it inverts them to be negative.

data['time_diff'] = data.groupby(['cell_number', 'cycle_number'])['time'].diff().fillna(0)
#Computes the time difference (time_diff) between consecutive rows within each cycle for each battery cell.
#This is crucial for computing incremental discharge capacity
#.fillna(0) ensures no missing values.

data['incremental_discharge'] = (data['current'] * data['time_diff']).abs() / 3600
#Uses current and time difference to compute incremental discharge energy.
#Converts from Amp-seconds (A·s) to milliamp-hours (mAh) by dividing by 3600
data['discharge_capacity'] = data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum() * 1000
#Computes cumulative discharge capacity per cycle by summing up incremental discharge values.
#Scales the final value by 1000 to match the expected mAh units.



# ========================
# 2. Aggregate Features for SOH Calculation
# ========================
agg_funcs = {
    'voltage': 'mean',
    'temperature': 'mean',
    'discharge_capacity': 'max'
}
#computes average voltage per cycle, average temp per cycle, and maximum discharge capacity per cycle (used for SOH calculations)
grouped = data.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()
#Groups the data by battery cell and cycle number.
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'temperature': 'avg_temperature',
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)
#Renames the aggregated columns for clarity.

# ========================
# 3. Compute State of Health (SOH)
# ========================
def compute_soh(df):
    cycle_1_capacity = df.loc[df['cycle_number'] == 1, 'max_discharge_capacity']
    if cycle_1_capacity.empty or cycle_1_capacity.values[0] <= 0:
        df['SOH'] = np.nan
    else:
        baseline = cycle_1_capacity.values[0]
        df['SOH'] = (df['max_discharge_capacity'] / baseline) * 100
    return df
#Defines a function to compute State of Health (SOH) for each battery.
#If Cycle 1 data is missing, assigns NaN.

grouped = grouped.groupby('cell_number', group_keys=False).apply(compute_soh).dropna(subset=['SOH'])
#Applies compute_soh to each battery cell.
#Drops rows with NaN SOH values.

# ========================
# 4. Define Features & Labels
# ========================
features = ['avg_voltage', 'avg_temperature', 'cycle_number']
X = grouped[features].values
y = grouped['SOH'].values
#Extracts the predictor variables (X) and the target variable (y = SOH)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#•	Standardizes the feature set to ensure all variables have equal weight in the model.



# ========================
# 5. Time-Based Train/Test Split
# ========================
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)
#•	Sorts the dataset chronologically before splitting.Ensures models are trained only on past data, preventing future cycle leakage.

split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]
#splits dataset

# ========================
# 6. Train Models & Evaluate Performance
# ========================
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression (alpha=1)": linear_model.Ridge(alpha=1),
    "Lasso Regression (alpha=1)": linear_model.Lasso(alpha=1, max_iter=10000)
}
#defines the 3 regression models


print("\n--- Model Performance on Time-Based Train/Test Split (80/20) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}:")
    print(f"  R2 on Training Set = {model.score(X_train, y_train):.4f}")
    print(f"  R2 on Test Set = {model.score(X_test, y_test):.4f}")
#•	Trains and evaluates each model.R² Score is printed for both training and test sets.


# ========================
# 7. Print Feature Coefficients
# ========================
print("\nFeature Coefficients for Linear Regression:")
print(dict(zip(features, models["Linear Regression"].coef_)))

# ========================
# 8. Print Key Dataset Metrics (AFTER Model Training)
# ========================
print("\n🔹 Key Dataset Metrics (For Reference, Not Used in Modeling) 🔹")
print(f"Total rows in dataset: {len(data)}")
print(f"🔹 Unique battery cells in dataset: {data['cell_number'].unique()}")
print(f"🔹 Total unique battery cells: {len(data['cell_number'].unique())}")
print(data['cell_number'].value_counts())
print(f"🔹 Unique discharge cycles: {data['cycle_number'].nunique()}")

print("\n🔹 Cycle-Based Feature Summary:")
print(f"🔹 Average voltage range: {data['voltage'].min()}V - {data['voltage'].max()}V")
print(f"🔹 Average temperature range: {data['temperature'].min()}°C - {data['temperature'].max()}°C")
print(f"🔹 Average discharge time per cycle: {data['time_diff'].mean():.4f} seconds")