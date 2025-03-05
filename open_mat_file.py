import scipy.io
import pandas as pd
import numpy as np

# Load the .mat file
mat_file = "ExampleDC_C1.mat"
data = scipy.io.loadmat(mat_file)

# Loop through all variables and display them as DataFrames
for key, value in data.items():
    if not key.startswith("__"):  # Ignore metadata
        print(f"\nVariable Name: {key}")

        # Convert NumPy array to DataFrame if it's 2D
        if isinstance(value, np.ndarray) and value.ndim == 2:
            df = pd.DataFrame(value)
            print(df.head())  # Print first 5 rows
        else:
            print(value)  # Print raw data if not tabular

