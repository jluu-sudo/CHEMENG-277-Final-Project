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
    'voltage': ['mean', 'min', 'max'],
    'temperature': ['mean', 'min', 'max', 'std'],
    'discharge_capacity': 'max'  # Max discharge capacity per cycle
}
grouped = data.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()

# Flatten multi-index column names
grouped.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in grouped.columns]

# Rename for clarity
grouped.rename(columns={
    'voltage_mean': 'avg_voltage',
    'voltage_min': 'min_voltage',
    'voltage_max': 'max_voltage',
    'temperature_mean': 'avg_temperature',
    'temperature_min': 'min_temperature',
    'temperature_max': 'max_temperature',
    'temperature_std': 'temp_std_dev',
    'discharge_capacity_max': 'max_discharge_capacity'
}, inplace=True)

# Compute total discharge time per cycle
grouped['total_discharge_time'] = data.groupby(['cell_number', 'cycle_number'])['time'].max().values - \
                                  data.groupby(['cell_number', 'cycle_number'])['time'].min().values

# Convert time to minutes if needed
grouped['total_discharge_time'] = grouped['total_discharge_time'] / 60

# ========================
# 3. Compute State of Health (SOH)
# ========================
def compute_soh(df):
    """Compute SOH as max capacity compared to cycle 1 capacity."""
    cycle_1_capacity = df.loc[df['cycle_number_'] == 1, 'max_discharge_capacity']

    if cycle_1_capacity.empty or cycle_1_capacity.values[0] <= 0:
        df['SOH'] = np.nan  # Assign NaN if invalid cycle 1 data
    else:
        baseline = cycle_1_capacity.values[0]
        df['SOH'] = (df['max_discharge_capacity'] / baseline) * 100

    return df

# Apply SOH computation to each cell
grouped = grouped.groupby('cell_number_', group_keys=False).apply(compute_soh).dropna(subset=['SOH'])

# ========================
# 4. Data Validation & Checks
# ========================
# Print correlation between total_discharge_time and SOH
correlation_matrix = grouped[['total_discharge_time', 'SOH']].corr()
print("\n🔍 Correlation between total_discharge_time and SOH:")
print(correlation_matrix)

# Check if total_discharge_time is too dominant
if abs(correlation_matrix.loc['total_discharge_time', 'SOH']) > 0.9:
    print("\n⚠️ Warning: total_discharge_time is highly correlated with SOH. Removing it from features.")
    drop_time_feature = True
else:
    drop_time_feature = False

# Print summary of total_discharge_time
print("\n🔍 Distribution of total_discharge_time:")
print(grouped['total_discharge_time'].describe())

# Print sample values per battery cell
print("\n🔍 Sample total_discharge_time values per battery:")
print(grouped.groupby('cell_number_')['total_discharge_time'].head(5))

# ========================
# 5. Define Features & Labels
# ========================
features = ['avg_voltage', 'min_voltage', 'max_voltage', 'avg_temperature', 'temp_std_dev', 'cycle_number_']

# Remove total_discharge_time if flagged
if not drop_time_feature:
    features.append('total_discharge_time')

X = grouped[features].values
y = grouped['SOH'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 6. Time-Based Train/Test Split (80/20)
# ========================
grouped_sorted = grouped.sort_values(by=['cell_number_', 'cycle_number_'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)

# Train-test split (80% train, 20% test)
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
# 8. Lasso Regression: Effect of Regularization Parameter (alpha)
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
# 9. Print Feature Coefficients
# ========================
print("Feature Coefficients for Linear Regression:")
print(dict(zip(features, models["Linear Regression"].coef_)))

