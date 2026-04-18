import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import re

def convert_csv_to_parquet():
    files = sorted(glob.glob("data/raw/*.csv"))
    print(f"Found {len(files)} files.")

    roots = ['ES', 'NQ', 'RTY']
    accumulated_data = {r: [] for r in roots}

    valid_pattern = re.compile(r'^(ES|NQ|RTY)[HMUZ][0-9]$')

    print("Reading and parsing CSV files...")
    for i, f in enumerate(files):
        if i % 20 == 0:
            print(f"Processed {i}/{len(files)} files...")

        df = pd.read_csv(f)

        # Ensure instrument_id is int64, no nulls
        if not pd.api.types.is_integer_dtype(df['instrument_id']):
            df['instrument_id'] = pd.to_numeric(df['instrument_id'], errors='raise', downcast='integer')
        if df['instrument_id'].isnull().any():
            raise ValueError(f"Found null instrument_id in file {f}")
        df['instrument_id'] = df['instrument_id'].astype('int64')

        # 1. Tighter symbol filter
        valid_mask = df['symbol'].str.match(valid_pattern)
        df = df[valid_mask].copy()

        if df.empty:
            continue

        # 2. Synthesize contract_root
        df['contract_root'] = df['symbol'].str.extract(r'^(ES|NQ|RTY)')[0]

        # 3. Cast ts_event to datetime64[ns, UTC]
        df['ts_event'] = pd.to_datetime(df['ts_event'], format='mixed', utc=True)

        cols = ['ts_event', 'rtype', 'publisher_id', 'instrument_id',
                'open', 'high', 'low', 'close', 'volume', 'symbol', 'contract_root']
        df = df[cols]

        for root in roots:
            root_df = df[df['contract_root'] == root]
            if not root_df.empty:
                accumulated_data[root].append(root_df)

    print("\nWriting Parquet files...")
    for root in roots:
        print(f"Processing root {root}...")
        if not accumulated_data[root]:
            print(f"No data for {root}")
            continue

        df_root = pd.concat(accumulated_data[root], ignore_index=True)

        # Determine 4-digit symbol and expiry date grouped by instrument_id
        # Month mapping
        month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}

        def get_third_friday(year, month):
            import datetime
            first_day = datetime.date(year, month, 1)
            first_friday_offset = (4 - first_day.weekday()) % 7
            first_friday = first_day + datetime.timedelta(days=first_friday_offset)
            return first_friday + datetime.timedelta(days=14)

        # Group by instrument_id and resolve symbols
        iid_map = {}
        for iid, group in df_root.groupby('instrument_id'):
            first_ts = group['ts_event'].min()
            last_ts = group['ts_event'].max()
            sym = group['symbol'].iloc[0]

            match = re.match(r'^(ES|NQ|RTY)([HMUZ])([0-9])$', sym)
            _, month_code, year_digit = match.groups()
            year_digit = int(year_digit)

            first_year = first_ts.year
            last_year = last_ts.year
            median_year = (first_year + last_year) / 2

            candidate_years = [2010 + year_digit, 2020 + year_digit]
            best_year = min(candidate_years, key=lambda y: abs(y - median_year))

            new_sym = f"{root}{month_code}{best_year}"
            expiry_date = get_third_friday(best_year, month_map[month_code])

            iid_map[iid] = {'new_symbol': new_sym, 'expiry_date': expiry_date}

        # Apply mapping
        df_root['symbol'] = df_root['instrument_id'].map(lambda x: iid_map[x]['new_symbol'])
        df_root['expiry_date'] = df_root['instrument_id'].map(lambda x: iid_map[x]['expiry_date'])

        # Convert expiry_date to pyarrow date32 equivalent (pandas datetime without time)
        df_root['expiry_date'] = pd.to_datetime(df_root['expiry_date']).dt.date

        # Reorder columns
        cols_final = ['ts_event', 'rtype', 'publisher_id', 'instrument_id',
                'open', 'high', 'low', 'close', 'volume', 'symbol', 'contract_root', 'expiry_date']
        df_root = df_root[cols_final]

        # 4. Sort rows by (symbol, ts_event) -> now grouped implicitly by instrument_id since symbols are unique
        df_root.sort_values(by=['symbol', 'ts_event'], inplace=True)

        # 5. RangeIndex 0..N-1, no DatetimeIndex
        df_root.reset_index(drop=True, inplace=True)

        schema = pa.schema([
            ('ts_event', pa.timestamp('ns', tz='UTC')),
            ('rtype', pa.int64()),
            ('publisher_id', pa.int64()),
            ('instrument_id', pa.int64()),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.int64()),
            ('symbol', pa.string()),
            ('contract_root', pa.string()),
            ('expiry_date', pa.date32())
        ])

        table = pa.Table.from_pandas(df_root, schema=schema, preserve_index=False)

        import os
        os.makedirs("data/parquet", exist_ok=True)
        out_file = f"data/parquet/ohlcv_1m_{root}.parquet"
        pq.write_table(table, out_file)
        print(f"Wrote {len(df_root):,} rows to {out_file}")

if __name__ == '__main__':
    import time
    import tracemalloc
    tracemalloc.start()
    t0 = time.time()
    convert_csv_to_parquet()
    t1 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nWall-clock runtime: {t1 - t0:.2f} seconds")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")
