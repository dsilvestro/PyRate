import os
print(os.environ.get('VIRTUAL_ENV', 'No active virtual environment'))


import pandas as pd
import numpy as np

# Load the data
# Assuming the data is tab-separated. Adjust the separator if it's different.
df = pd.read_csv('Carnivora_1_G_BDS_BDNN_16_8TVc_mcmc.log', sep='\t')

# Check for NaN values
nan_count = df.isna().sum()
print("Columns with NaN values:")
print(nan_count[nan_count > 0])

# Check for infinite values
inf_count = np.isinf(df).sum()
print("\nColumns with infinite values:")
print(inf_count[inf_count > 0])

# Define a function to check for extreme values
def check_extreme_values(series, lower_percentile=0.01, upper_percentile=99.99):
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    extreme_low = series[series < lower]
    extreme_high = series[series > upper]
    return extreme_low, extreme_high

# Check each numeric column for extreme values
print("\nExtreme values:")
for column in df.select_dtypes(include=[np.number]).columns:
    extreme_low, extreme_high = check_extreme_values(df[column])
    if not extreme_low.empty or not extreme_high.empty:
        print(f"\nColumn: {column}")
        if not extreme_low.empty:
            print("  Extreme low values:")
            print(extreme_low)
        if not extreme_high.empty:
            print("  Extreme high values:")
            print(extreme_high)

# Calculate basic statistics for each numeric column
stats = df.describe()
print("\nBasic statistics:")
print(stats)