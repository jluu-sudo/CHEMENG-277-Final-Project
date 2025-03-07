import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Data Loading & Preprocessing (Discharge Data)
# ========================
file_path = "Copy of Oxford_battery_data_discharge.csv"
try:
    data = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the path and filename.")
    exit()

# Ensure time is sorted correctly for proper computation
data = data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Ensure current is negative for discharge
if data['current'].max() > 0:
    print("⚠️ Warning: Current values appear to be positive. Inverting current direction.")
    data['current'] = -data['current']

# Compute time difference (in seconds)
data['time_diff'] = data.groupby(['cell_number', 'cycle_number'])['time'].diff().fillna(0)

# Compute incremental discharge (convert A·s to mAh)
data['incremental_discharge'] = (data['current'] * data['time_diff']) / 3600  # Convert to mAh

# Take absolute values to ensure all discharge capacities are positive
data['incremental_discharge'] = data['incremental_discharge'].abs()

# Compute cumulative discharge per cycle
data['discharge_capacity'] = data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum()

# 🔹 Scale discharge capacity by 1000 to match expected mAh units
data['discharge_capacity'] *= 1000  # Convert from microamp-hours (µAh) to milliamp-hours (mAh)

# Debug output to check if discharge capacity is computed correctly
print(f"🔍 Scaled Discharge Capacity Summary:\n{data['discharge_capacity'].describe()}")

# Ensure the correct columns exist for prediction
required_columns = {'cell_number', 'cycle_number', 'voltage', 'temperature', 'discharge_capacity'}
if not required_columns.issubset(data.columns):
    print("Error: Missing required columns in CSV. Expected columns:", required_columns)
    exit()

# Check for cycle 1 presence in the dataset
valid_cells = data[data['cycle_number'] == 1]['cell_number'].unique()
print(f"Number of cells with valid cycle 1 data: {len(valid_cells)}")

if len(valid_cells) == 0:
    print("Error: No battery cells contain cycle 1 data. SOH computation cannot proceed.")
    exit()

# ========================
# 2. Aggregate Features
# ========================
agg_funcs = {
    'voltage': 'mean',
    'temperature': 'mean',
    'discharge_capacity': 'max'  # Use max discharge capacity in each cycle
}

# Define `grouped` BEFORE using `groupby.apply()`
grouped = data.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()

# Rename columns for clarity
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'temperature': 'avg_temperature',
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)

# Ensure `grouped` is properly created before calling compute_soh()
print(f"🔍 Grouped Data Summary:\n{grouped.describe()}")


# ========================
# 3. Define SOH Computation Function
# ========================
def compute_soh(df):
    """Compute SOH based on the maximum discharge capacity in cycle 1."""
    cycle_1_capacity = df.loc[df['cycle_number'] == 1, 'max_discharge_capacity']

    if cycle_1_capacity.empty or cycle_1_capacity.isnull().all() or cycle_1_capacity.values[0] <= 0:
        print(f"⚠️ Warning: Cell {df['cell_number'].iloc[0]} has no valid cycle 1 discharge data. Assigning NaN SOH.")
        df['SOH'] = np.nan  # Assign NaN if cycle 1 is missing or corrupted
    else:
        baseline = cycle_1_capacity.values[0]  # Extract the first valid value
        print(f"🔍 Cell {df['cell_number'].iloc[0]} - Cycle 1 Capacity: {baseline} mAh")  # Debugging print
        df['SOH'] = (df['max_discharge_capacity'] / baseline) * 100  # Standard SOH formula

    return df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Load & Preprocess Data
# ========================
file_path = "Copy of Oxford_battery_data_discharge.csv"

# Load data
data = pd.read_csv(file_path, low_memory=False)

# Ensure time is sorted for proper discharge computation
data = data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Ensure current values are negative for discharge
if data['current'].max() > 0:
    data['current'] = -data['current']

# Compute time differences (s) and incremental discharge (A·s to mAh)
data['time_diff'] = data.groupby(['cell_number', 'cycle_number'])['time'].diff().fillna(0)
data['incremental_discharge'] = (data['current'] * data['time_diff']).abs() / 3600  # Convert to mAh

# Compute cumulative discharge per cycle and scale to mAh
data['discharge_capacity'] = data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum() * 1000

# ========================
# 2. Aggregate Features for SOH Calculation
# ========================
agg_funcs = {
    'voltage': 'mean',
    'temperature': 'mean',
    'discharge_capacity': 'max'  # Max discharge capacity per cycle
}
grouped = data.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()

# Rename columns for clarity
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'temperature': 'avg_temperature',
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)


# ========================
# 3. Compute State of Health (SOH)
# ========================
def compute_soh(df):
    """Compute SOH as max capacity compared to cycle 1 capacity."""
    cycle_1_capacity = df.loc[df['cycle_number'] == 1, 'max_discharge_capacity']

    if cycle_1_capacity.empty or cycle_1_capacity.values[0] <= 0:
        df['SOH'] = np.nan  # Assign NaN if invalid cycle 1 data
    else:
        baseline = cycle_1_capacity.values[0]
        df['SOH'] = (df['max_discharge_capacity'] / baseline) * 100

    return df


# Apply SOH computation to each cell
grouped = grouped.groupby('cell_number', group_keys=False).apply(compute_soh).dropna(subset=['SOH'])

# ========================
# 4. Define Features & Labels
# ========================
features = ['avg_voltage', 'avg_temperature', 'cycle_number']
X = grouped[features].values
y = grouped['SOH'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 5. Time-Based Train/Test Split (80/20)
# ========================
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)

# Train-test split (80% train, 20% test)
split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# ========================
# 6. Train Models & Evaluate Performance
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
# 7. Lasso Regression: Effect of Regularization Parameter (alpha)
# ========================
alpha_values = [0.1, 0.25, 0.5, 0.75, 0.9]
lasso_r2_test = []

for a in alpha_values:
    lasso_model = linear_model.Lasso(alpha=a, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    lasso_r2_test.append(lasso_model.score(X_test, y_test))

# Plot Lasso R2 vs Alpha
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, lasso_r2_test, marker='o', linestyle='-')
plt.xlabel("Alpha (Regularization Parameter)")
plt.ylabel("R2 on Test Set")
plt.title("Effect of Lasso Regularization Parameter on Model Generalizability (Time-Based Split)")
plt.grid(True)
plt.show()

# ========================
# 8. Print Feature Coefficients
# ========================
print("Feature Coefficients for Linear Regression:")
print(dict(zip(features, models["Linear Regression"].coef_)))

# Ensure the function is defined BEFORE calling it
grouped = grouped.groupby('cell_number', group_keys=False, observed=True).apply(compute_soh)

# Drop rows where SOH is NaN
grouped.dropna(subset=['SOH'], inplace=True)

# Debugging step: Print SOH statistics
print(f"SOH Computation Summary:\n{grouped['SOH'].describe()}")

# Check if dataset is empty after SOH computation
if grouped.empty:
    print("Error: No valid data remains after SOH computation. Ensure the dataset contains valid discharge data.")
    exit()

# ========================
# 4. Define Features and Labels (Removing Current)
# ========================
features = ['avg_voltage', 'avg_temperature', 'cycle_number']  # Only use relevant predictors
X = grouped[features].values
y = grouped['SOH'].values

# Check if dataset is empty before scaling
if X.shape[0] == 0:
    print("Error: No valid samples found after filtering. SOH computation may have removed all data.")
    exit()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 5. Time-Based Train/Test Split
# ========================
# Sort by cycle number to ensure chronological order
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values

# Standardize using the same scaler
X_seq_scaled = scaler.transform(X_seq)

# Time-based split: first 80% of cycles for training, last 20% for testing
split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

print("\n--- Model Performance on Time-Based Train/Test Split (80/20) ---")
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression (alpha=1)": linear_model.Ridge(alpha=1),
    "Lasso Regression (alpha=1)": linear_model.Lasso(alpha=1, max_iter=10000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}:")
    print(f"  R2 on Training Set = {model.score(X_train, y_train):.4f}")
    print(f"  R2 on Test Set = {model.score(X_test, y_test):.4f}")

# ========================
# 6. Lasso Regression: Effect of Regularization Parameter (alpha)
# ========================
alpha_values = [0.1, 0.25, 0.5, 0.75, 0.9]
lasso_r2_test = []

print("\n--- Lasso Regression: Effect of Alpha on Time-Based Split ---")
for a in alpha_values:
    lasso_model = linear_model.Lasso(alpha=a, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    r2_test = lasso_model.score(X_test, y_test)
    lasso_r2_test.append(r2_test)
    print(f"Alpha = {a}: R2 Test = {r2_test:.4f}")

# Print feature importance from Linear Regression
print("Feature Coefficients for Linear Regression:")
print(dict(zip(features, models["Linear Regression"].coef_)))
