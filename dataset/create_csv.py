import pandas as pd
import numpy as np
from io import StringIO

# Raw data as a multi-line string (easier to work with in this example)
with open(r"./new_data.lst", "r+") as F:
    data = F.read()

# Use StringIO to treat the string as a file
data_io = StringIO(data)

# Read the data, specifying space as the delimiter and providing column names
df = pd.read_csv(data_io, delim_whitespace=True, header=None, names=[
    "YEAR", "DOY", "Hour", "Scalar_B", "BX_GSE", "BY_GSE", "BY_GSM", "BZ_GSM", "Sigma_BX", "Sigma_BY", "Sigma_BZ",
    "SW_Temp", "SW_Density", "SW_Speed", "SW_Flow_long", "SW_Flow_lat", "AP_Density", "Sigma_T", "Sigma_n", "Sigma_V", "Flow_Pressure", "E_Field", "Dst"
])

# Create the datetime column
df['datetime'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str) + df['Hour'].astype(str), format='%Y%j%H')
df['timestamp'] = pd.to_datetime(df['datetime'], unit='s')
df['timestamp'] = df['timestamp'].astype('int64') // 10**9

# Drop the original YEAR, DOY, and Hour columns
df = df.drop(columns=['YEAR', 'DOY', 'Hour', 'datetime'])

# Reorder columns to put 'datetime' at the beginning
df = df[['timestamp'] + [col for col in df.columns if col != 'timestamp']]

# --- Placeholder for Missing Value Handling (Forward Fill) ---
df = df.ffill()  # Forward fill.  Replace with your preferred method.

import warnings

def replace_placeholder_nans(df, placeholder_patterns):
    df_cleaned = df.copy()  # Create a copy to avoid modifying the original DataFrame

    for col in df_cleaned.columns:
        
        if col == "Dst" or col == "timestamp":
            continue
        
        for pattern in placeholder_patterns:
            try:
                # Convert the column to numeric if possible. Handle errors.
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

                # Use regex for flexibility and to ensure ONLY 9s are matched.
                with warnings.catch_warnings():  # Suppress the UserWarning locally
                    warnings.simplefilter("ignore", UserWarning)
                    df_cleaned[col] = df_cleaned[col].mask(df_cleaned[col].astype(str).str.contains(pattern, regex=True))
            except:
                # If not convertible to numeric, skip the column
                continue

    return df_cleaned


def find_placeholder_patterns(df, num_nines=1, max_num_nines=6):
    patterns = []
    for n in range(num_nines, max_num_nines + 1):
        # Match integers consisting only of 9s (at least n nines)
        patterns.append(r"^\s*9{" + str(n) + r",}\s*$")

        # Match decimals with only 9s before the decimal, and only 0s after.
        patterns.append(r"^\s*9{" + str(n) + r",}\.0+\s*$")
        patterns.append(r"^\s*\.9{" + str(n) + r",}\s*$") # Start with a decimal
        patterns.append(r"^\s*9{" + str(n) + r",}\.9+0*\s*$")  # 9s followed by 9s and then 0s

    patterns = list(dict.fromkeys(patterns))
    return patterns

# Find the placeholder patterns
placeholder_patterns = find_placeholder_patterns(df)

# Clean the DataFrame
df = replace_placeholder_nans(df, placeholder_patterns)

# Save to CSV
csv_output = df.to_csv("./dataset.csv", index=False)
print(csv_output)