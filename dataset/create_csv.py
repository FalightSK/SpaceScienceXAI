import pandas as pd
import numpy as np
from io import StringIO

# Raw data as a multi-line string (easier to work with in this example)
with open("./dataset.lst", "r+") as F:
    data = F.read()

# Use StringIO to treat the string as a file
data_io = StringIO(data)

# Read the data, specifying space as the delimiter and providing column names
df = pd.read_csv(data_io, delim_whitespace=True, header=None, names=[
    "YEAR", "DOY", "Hour", "Scalar_B", "BX_GSE", "BY_GSM", "BZ_GSM",
    "SW_Temp", "SW_Density", "SW_Speed", "Flow_Pressure", "E_Field", "Dst"
])

# Create the datetime column
df['datetime'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str) + df['Hour'].astype(str), format='%Y%j%H')

# Drop the original YEAR, DOY, and Hour columns
df = df.drop(columns=['YEAR', 'DOY', 'Hour'])

# Reorder columns to put 'datetime' at the beginning
df = df[['datetime'] + [col for col in df.columns if col != 'datetime']]

# --- Placeholder for Missing Value Handling (Forward Fill) ---
df = df.ffill()  # Forward fill.  Replace with your preferred method.

# Save to CSV
csv_output = df.to_csv("./dataset.csv", index=False)
print(csv_output)