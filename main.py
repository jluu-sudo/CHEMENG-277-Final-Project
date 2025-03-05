import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

file_path = "Copy of Oxford_battery_data_charge.csv"
data = pd.read_csv(file_path, low_memory=False)

# Print number of rows
print(f"Total rows in dataset: {len(data)}")
# Get unique battery cell numbers
unique_cells = data['cell_number'].unique()
print(f"Unique battery cells in dataset: {unique_cells}")
print(f"Total unique battery cells: {len(unique_cells)}")
print(data['cell_number'].value_counts())  # Number of rows per battery cell
print("Unique charge cycles:", data['cycle_number'].nunique())

# ========================
# 1. Data Loading & Preprocessing
# ========================
file_path = "Copy of Oxford_battery_data_charge.csv"
try:
    data = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the path and filename.")
    exit()

# Compute charge as the cumulative sum of current * time difference per cycle
if {'time', 'current', 'cell_number', 'cycle_number'}.issubset(data.columns):
    data['time_diff'] = data.groupby(['cell_number', 'cycle_number'])['time'].diff().fillna(0)
    data['charge'] = (data['current'] * data['time_diff']) / 3600  # Convert As to mAh
    data['charge'] = data.groupby(['cell_number', 'cycle_number'])['charge'].cumsum()
else:
    print("Error: Required columns for charge calculation are missing. Check column names in CSV.")
    exit()

# Now check for required columns **after** computing charge
required_columns = {'cell_number', 'cycle_number', 'voltage', 'current', 'temperature', 'charge'}
if not required_columns.issubset(data.columns):
    print("Error: Missing required columns in CSV. Expected columns:", required_columns)
    exit()

# ========================
# 2. Aggregate Features
# ========================
agg_funcs = {
    'voltage': 'mean',
    'current': 'mean',
    'temperature': 'mean',
    'charge': 'max'
}
grouped = data.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()

# Rename columns for clarity
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'current': 'avg_current',
    'temperature': 'avg_temperature',
    'charge': 'max_charge'
}, inplace=True)


# ========================
# 2. Derive State of Health (SOH)
# ========================
def compute_soh(df):
    """Compute SOH as a percentage of the max charge from cycle 1."""
    if 1 not in df['cycle_number'].values:
        print(f"Warning: Cell {df['cell_number'].iloc[0]} missing cycle 1 data. SOH calculation may be incorrect.")
        return df
    baseline = df.loc[df['cycle_number'] == 1, 'max_charge'].values[0]
    df['SOH'] = (df['max_charge'] / baseline) * 100
    return df

# Compute SOH for each battery cell
grouped = grouped.groupby('cell_number', group_keys=False).apply(compute_soh)

# ========================
# 3. Define Features and Labels
# ========================
features = ['avg_voltage', 'avg_current', 'avg_temperature', 'cycle_number']
X = grouped[features].values
y = grouped['SOH'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 4. Modeling on Entire Dataset (for comparison)
# ========================
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression (alpha=1)": linear_model.Ridge(alpha=1),
    "Lasso Regression (alpha=1)": linear_model.Lasso(alpha=1, max_iter=10000)
}

print("\n--- Model Performance on Entire Dataset ---")
for name, model in models.items():
    model.fit(X_scaled, y)
    print(f"{name}: R2 = {model.score(X_scaled, y):.4f}")

# ========================
# 5. Sequential Train/Test Split
# ========================
grouped_sorted = grouped.sort_values(by='cycle_number')
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values

# Standardize using the same scaler
X_seq_scaled = scaler.transform(X_seq)

# Split data: first 50% for training, rest for testing
split_index = int(0.5 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

print("\n--- Model Performance on Train/Test Split ---")
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

print("\n--- Lasso Regression: Effect of Alpha ---")
for a in alpha_values:
    lasso_model = linear_model.Lasso(alpha=a, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    r2_test = lasso_model.score(X_test, y_test)
    lasso_r2_test.append(r2_test)
    print(f"Alpha = {a}: R2 Test = {r2_test:.4f}")

# Plot Lasso R2 vs Alpha
plt.figure(figsize=(8, 5))
plt.plot(alpha_values, lasso_r2_test, marker='o', linestyle='-')
plt.xlabel("Alpha (Regularization Parameter)")
plt.ylabel("R2 on Test Set")
plt.title("Effect of Lasso Regularization Parameter on Model Generalizability")
plt.grid(True)
plt.show()
