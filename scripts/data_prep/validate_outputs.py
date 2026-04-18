import pandas as pd
import pyarrow.parquet as pq
import random
import os
import glob

def validate_outputs():
    roots = ['ES', 'NQ', 'RTY']
    
    # Pre-calculate counts of valid rows from source to verify
    # (Using the script we already ran to get total counts per symbol,
    # but for simplicity we'll just check no duplicate rows, monotonic ts, etc.)
    
    for root in roots:
        fname = f"ohlcv_1m_{root}.parquet"
        if not os.path.exists(fname):
            print(f"File {fname} not found!")
            continue
            
        print(f"\n--- Validating {fname} ---")
        df = pd.read_parquet(fname)
        
        # 1. Check exact column order and dtypes
        expected_cols = ['ts_event', 'rtype', 'publisher_id', 'instrument_id', 
                         'open', 'high', 'low', 'close', 'volume', 'symbol', 'contract_root']
        actual_cols = df.columns.tolist()
        if actual_cols != expected_cols:
            print(f"ERROR: Column order mismatch.\nExpected: {expected_cols}\nActual:   {actual_cols}")
        else:
            print("Column order: OK")
            
        print("Dtypes:\n", df.dtypes)
        
        if df['ts_event'].dt.tz.zone != 'UTC':
            print(f"ERROR: ts_event timezone is {df['ts_event'].dt.tz}, expected UTC")
        else:
            print("Timezone: OK (UTC)")
            
        # 2. Check for duplicate (ts_event, symbol)
        dups = df.duplicated(subset=['ts_event', 'symbol']).sum()
        if dups > 0:
            print(f"ERROR: Found {dups} duplicate (ts_event, symbol) pairs!")
        else:
            print("Duplicate check: OK (No duplicates)")
            
        # 3. Check monotonic non-decreasing ts_event within symbol
        monotonic_ok = True
        symbols = df['symbol'].unique()
        print(f"Checking monotonicity for {len(symbols)} symbols...")
        for sym in symbols:
            sym_df = df[df['symbol'] == sym]
            if not sym_df['ts_event'].is_monotonic_increasing:
                print(f"ERROR: ts_event is not monotonically increasing for symbol {sym}")
                monotonic_ok = False
        if monotonic_ok:
            print("Monotonic check: OK (ts_event is monotonic within all symbols)")
            
        # 4. Sample 10 random (symbol, ts_event) pairs to check against original
        print("Sampling 10 random rows to verify round-trip...")
        samples = df.sample(10, random_state=42)
        
        files = sorted(glob.glob("data/raw/*.csv"))
        
        success = True
        for _, sample_row in samples.iterrows():
            sym = sample_row['symbol']
            ts = sample_row['ts_event']
            # Find the file that contains this timestamp
            # CSV timestamps look like "2010-06-06T22:00:00.000000000Z"
            # To be simple and not scan all 191 files, we just format the timestamp and grep for it.
            # actually reading through all files just for 10 samples might be slow, let's use bash grep
            ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            # If the CSV has trailing zeros or just Z, we might need a looser match.
            ts_str_short = ts.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Use os.popen to grep
            # Find the month file
            month_str = ts.strftime('%Y%m')
            candidate_files = [f for f in files if month_str in f]
            
            found = False
            for cf in candidate_files:
                df_csv = pd.read_csv(cf)
                # Parse ts_event in csv for reliable comparison
                match = df_csv[(df_csv['symbol'] == sym) & (df_csv['ts_event'].str.startswith(ts_str_short))]
                if not match.empty:
                    found = True
                    csv_row = match.iloc[0]
                    # compare
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if csv_row[col] != sample_row[col]:
                            print(f"ERROR: Mismatch for {sym} at {ts} on column {col}! CSV={csv_row[col]}, Parquet={sample_row[col]}")
                            success = False
            if not found:
                print(f"WARNING: Could not find {sym} at {ts} in expected CSV files for sampling.")
                success = False
                
        if success:
            print("Round-trip sampling: OK (All 10 samples match exactly)")

if __name__ == '__main__':
    validate_outputs()
