import pandas as pd
import glob
import os

files = sorted(glob.glob("data/raw/*.csv"))
print(f"Found {len(files)} CSV files.")

if len(files) > 0:
    first_file = files[-2]  # use a recent full month, e.g. 202603
    print(f"Reading {first_file}...")
    df = pd.read_csv(first_file)
    print("Header:", df.columns.tolist())
    print("Dtypes:\n", df.dtypes)
    print("First 5 rows:\n", df.head())
    print("Last 5 rows:\n", df.tail())
    print("Unique symbols:", df.get("symbol", df.get("sym", "NO SYMBOL COLUMN")).unique() if "symbol" in df.columns or "sym" in df.columns else "NO SYMBOL")
