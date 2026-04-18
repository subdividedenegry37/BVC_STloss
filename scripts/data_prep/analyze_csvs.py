import os
import glob
import pandas as pd
import json

def analyze():
    files = sorted(glob.glob("data/raw/*.csv"))
    total_size = sum(os.path.getsize(f) for f in files)
    print(f"Located {len(files)} CSV files in data/raw.")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    
    # Read first file for schema
    df_first = pd.read_csv(files[0])
    print("\n--- CSV Inspection (First File) ---")
    print("Header:", df_first.columns.tolist())
    print("Dtypes:\n", df_first.dtypes)
    print("First 5 rows:\n", df_first.head().to_string())
    print("Last 5 rows:\n", df_first.tail().to_string())
    print("Columns count:", len(df_first.columns))
    
    # Read all files to get symbols and row counts
    print("\nProcessing all files to get row counts and symbol stats...")
    total_rows = 0
    symbol_stats = {}
    
    for f in files:
        # Use usecols to only read needed columns to save memory and time
        df = pd.read_csv(f, usecols=['ts_event', 'symbol'])
        total_rows += len(df)
        
        # Filter for ES, NQ, RTY standard quarterly contracts
        # Databento symbols typically look like ESZ5, NQH6, RTYM6
        # Exclude spreads like ESH6-ESM6
        df_outright = df[~df['symbol'].str.contains('-')]
        
        agg = df_outright.groupby('symbol').agg(
            first_ts=('ts_event', 'min'),
            last_ts=('ts_event', 'max'),
            count=('ts_event', 'count')
        ).reset_index()
        
        for _, row in agg.iterrows():
            sym = row['symbol']
            if sym not in symbol_stats:
                symbol_stats[sym] = {'first_ts': row['first_ts'], 'last_ts': row['last_ts'], 'count': row['count']}
            else:
                symbol_stats[sym]['first_ts'] = min(symbol_stats[sym]['first_ts'], row['first_ts'])
                symbol_stats[sym]['last_ts'] = max(symbol_stats[sym]['last_ts'], row['last_ts'])
                symbol_stats[sym]['count'] += row['count']
                
    print(f"\nTotal rows across all CSVs: {total_rows:,}")
    
    # Group by root
    roots = {'ES': {}, 'NQ': {}, 'RTY': {}}
    for sym, stats in symbol_stats.items():
        if sym.startswith('ES'): roots['ES'][sym] = stats
        elif sym.startswith('NQ'): roots['NQ'][sym] = stats
        elif sym.startswith('RTY'): roots['RTY'][sym] = stats
        
    for root, syms in roots.items():
        print(f"\n--- {root} Contracts ({len(syms)} outrights) ---")
        # sort by first_ts
        sorted_syms = sorted(syms.items(), key=lambda x: x[1]['first_ts'])
        for sym, stats in sorted_syms:
            print(f"{sym:6s} : {stats['count']:>8,} rows | {stats['first_ts']} -> {stats['last_ts']}")

if __name__ == '__main__':
    analyze()
