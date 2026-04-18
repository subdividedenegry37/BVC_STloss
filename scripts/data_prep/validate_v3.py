import pandas as pd
import pyarrow.parquet as pq
import random
import os
import glob

def validate_outputs():
    roots = ['ES', 'NQ', 'RTY']
    
    for root in roots:
        fname = f"ohlcv_1m_{root}.parquet"
        if not os.path.exists(fname):
            print(f"File {fname} not found!")
            continue
            
        print(f"\n--- Validating {fname} ---")
        df = pd.read_parquet(fname)
        
        # 1. Check exact column order and dtypes
        expected_cols = ['ts_event', 'rtype', 'publisher_id', 'instrument_id', 
                         'open', 'high', 'low', 'close', 'volume', 'symbol', 'contract_root', 'expiry_date']
        actual_cols = df.columns.tolist()
        if actual_cols != expected_cols:
            print(f"ERROR: Column order mismatch.\nExpected: {expected_cols}\nActual:   {actual_cols}")
            return
        else:
            print("Column order: OK")
            
        print("Dtypes:")
        print(df.dtypes)
        
        if df['ts_event'].dt.tz.zone != 'UTC':
            print(f"ERROR: ts_event timezone is {df['ts_event'].dt.tz}, expected UTC")
            return
        else:
            print("Timezone: OK (UTC)")
            
        # 2. Check for duplicate (instrument_id, ts_event)
        dups = df.duplicated(subset=['instrument_id', 'ts_event']).sum()
        if dups > 0:
            print(f"ERROR: Found {dups} duplicate (instrument_id, ts_event) pairs!")
            return
        else:
            print("Duplicate check: OK (No duplicates)")
            
        # 3. Check monotonic non-decreasing ts_event within instrument_id
        monotonic_ok = True
        iids = df['instrument_id'].unique()
        print(f"Checking monotonicity for {len(iids)} instrument_ids...")
        for iid in iids:
            iid_df = df[df['instrument_id'] == iid]
            if not iid_df['ts_event'].is_monotonic_increasing:
                print(f"ERROR: ts_event is not monotonically increasing for instrument_id {iid}")
                monotonic_ok = False
        if monotonic_ok:
            print("Monotonic check: OK (ts_event is monotonic within all instrument_ids)")
        else:
            return
            
        # 4. Check one-to-one instrument_id <-> symbol
        one_to_one_ok = True
        for iid in iids:
            symbols_for_iid = df[df['instrument_id'] == iid]['symbol'].unique()
            if len(symbols_for_iid) != 1:
                print(f"ERROR: Multiple symbols for instrument_id {iid}: {symbols_for_iid}")
                one_to_one_ok = False
        if one_to_one_ok:
            print("One-to-one instrument_id <-> symbol check: OK")
        else:
            return
            
        # Total row count and distinct instrument_ids
        print(f"Total rows: {len(df):,}")
        print(f"Distinct instrument_ids: {len(iids)}")
            
        # 5. Sample 10 random rows to verify round-trip
        print("Sampling 10 random rows to verify round-trip...")
        samples = df.sample(10, random_state=42)
        
        files = sorted(glob.glob("data/raw/*.csv"))
        
        success = True
        for _, sample_row in samples.iterrows():
            sym_full = sample_row['symbol'] # e.g. ESM2010
            # get original symbol (e.g. ESM0) for csv search
            sym_orig = sym_full[:3] + sym_full[-1:] 
            
            ts = sample_row['ts_event']
            iid = sample_row['instrument_id']
            
            ts_str_short = ts.strftime('%Y-%m-%dT%H:%M:%S')
            month_str = ts.strftime('%Y%m')
            candidate_files = [f for f in files if month_str in f]
            
            found = False
            for cf in candidate_files:
                df_csv = pd.read_csv(cf)
                match = df_csv[(df_csv['instrument_id'] == iid) & (df_csv['ts_event'].str.startswith(ts_str_short))]
                if not match.empty:
                    found = True
                    csv_row = match.iloc[0]
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if abs(csv_row[col] - sample_row[col]) > 1e-8: # floating point check
                            print(f"ERROR: Mismatch for {sym_full} at {ts} on column {col}! CSV={csv_row[col]}, Parquet={sample_row[col]}")
                            success = False
            if not found:
                print(f"WARNING: Could not find {sym_full} (orig {sym_orig}, iid {iid}) at {ts} in expected CSV files for sampling.")
                success = False
                
        if success:
            print("Round-trip sampling: OK (All 10 samples match exactly)")
        else:
            return
            
    print("\nAll validation checks PASSED.")

if __name__ == '__main__':
    validate_outputs()
